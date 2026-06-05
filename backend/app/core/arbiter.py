"""The VRAM arbiter — the architectural heart of HFabric.

On a 16 GB card you cannot hold an LLM (~12 GB) and a diffusion model at the
same time. The arbiter enforces the invariant **at most one GPU resident at a
time**: requesting a different model unloads the current one first. Swaps are
serialized by a lock and announced on the event bus so the UI can show what is
happening.
"""

from __future__ import annotations

import asyncio

from ..backends.base import GpuBackend
from .enums import EventType
from .events import Event, EventBus


class GpuArbiter:
    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._lock = asyncio.Lock()
        self._current: GpuBackend | None = None
        self._warm_backends: list[GpuBackend] = []

    @property
    def current(self) -> GpuBackend | None:
        return self._current

    async def ensure(self, backend: GpuBackend) -> None:
        """Guarantee ``backend`` is the sole GPU resident and loaded."""
        async with self._lock:
            if self._current is backend and backend.loaded:
                return
            if self._current is not None and self._current is not backend:
                await self._bus.publish(Event(
                    EventType.ARBITER_NOTE,
                    reason="swap",
                    message=(
                        f"Swapping models: unloading {self._current.descriptor.name} "
                        f"to free VRAM for {backend.descriptor.name}."
                    ),
                    model=backend.descriptor.name,
                    family=backend.descriptor.family.value,
                ))
                await self._unload_current(allow_keep_warm=True)
            if not backend.loaded:
                await self._guard_budget(backend)
                warm_resume = backend.warm
                await self._bus.publish(Event(
                    EventType.MODEL_LOADING,
                    resident=backend.resident_key,
                    model=backend.descriptor.name,
                    family=backend.descriptor.family.value,
                    warm_resume=warm_resume,
                ))
                await backend.load()
                await self._record_profile(backend)
                if backend in self._warm_backends:
                    self._warm_backends.remove(backend)
                await self._bus.publish(Event(
                    EventType.MODEL_LOADED,
                    resident=backend.resident_key,
                    model=backend.descriptor.name,
                    family=backend.descriptor.family.value,
                    load_report=backend.load_report,
                    warm_resume=warm_resume,
                ))
            self._current = backend
            await self._publish_status()

    async def free_all(self) -> None:
        async with self._lock:
            if self._current is not None:
                await self._unload_current(allow_keep_warm=False)
            for backend in list(self._warm_backends):
                await self._unload_warm(backend)
            await self._publish_status()

    async def _unload_current(self, *, allow_keep_warm: bool = False) -> None:
        cur = self._current
        assert cur is not None
        keep_warm, reason = self._keep_warm_decision(cur, allow_keep_warm=allow_keep_warm)
        await self._bus.publish(Event(
            EventType.MODEL_UNLOADING,
            resident=cur.resident_key,
            model=cur.descriptor.name,
            keep_warm=keep_warm,
            reason=reason,
        ))
        if keep_warm and await cur.park():
            if cur not in self._warm_backends:
                self._warm_backends.append(cur)
            await self._bus.publish(Event(
                EventType.MODEL_UNLOADED,
                resident=cur.resident_key,
                model=cur.descriptor.name,
                kept_warm=True,
                reason=reason,
            ))
            self._current = None
            return

        await cur.unload()
        if cur in self._warm_backends:
            self._warm_backends.remove(cur)
        await self._bus.publish(Event(
            EventType.MODEL_UNLOADED,
            resident=cur.resident_key,
            model=cur.descriptor.name,
            kept_warm=False,
            reason=reason,
        ))
        self._current = None

    async def _unload_warm(self, backend: GpuBackend) -> None:
        await self._bus.publish(Event(
            EventType.MODEL_UNLOADING,
            resident=backend.resident_key,
            model=backend.descriptor.name,
            keep_warm=False,
            warm=True,
        ))
        await backend.unload()
        if backend in self._warm_backends:
            self._warm_backends.remove(backend)
        await self._bus.publish(Event(
            EventType.MODEL_UNLOADED,
            resident=backend.resident_key,
            model=backend.descriptor.name,
            kept_warm=False,
            warm=True,
        ))

    def _keep_warm_decision(
        self, backend: GpuBackend, *, allow_keep_warm: bool
    ) -> tuple[bool, str]:
        from ..config import settings  # noqa: PLC0415
        from ..util import sysmon  # noqa: PLC0415

        if not allow_keep_warm:
            return False, "forced unload"
        if not settings.keep_warm_models:
            return False, "keep-warm disabled"
        if settings.keep_warm_max_models <= 0:
            return False, "keep-warm max is zero"
        if len(self._warm_backends) >= settings.keep_warm_max_models and backend not in self._warm_backends:
            return False, "keep-warm pool is full"
        if not backend.can_keep_warm:
            return False, "backend does not support keep-warm"
        if settings.stub_mode:
            return True, "stub keep-warm"

        d = backend.descriptor
        return sysmon.can_keep_warm(d.family, d.size_bytes, d.quant, d.id)

    async def _record_profile(self, backend: GpuBackend) -> None:
        """Learn this model's measured RAM/VRAM from its load report (P7.2).

        Best-effort: a telemetry/DB hiccup must never break a successful load.
        The load starts from a clean baseline (the previous model is already
        unloaded), so end-minus-start RSS is the model's own footprint."""
        from ..config import settings  # noqa: PLC0415

        if settings.stub_mode or not settings.learn_memory_profiles:
            return
        ram_gb, vram_gb = _measured_from_report(backend.load_report)
        if ram_gb is None and vram_gb is None:
            return
        d = backend.descriptor
        try:
            from ..db.session import session_scope  # noqa: PLC0415
            from ..services import model_profile_service as mps  # noqa: PLC0415
            from ..util import sysmon  # noqa: PLC0415

            async with session_scope() as s:
                await mps.record(
                    s, model_id=d.id, family=d.family.value, quant=d.quant,
                    ram_gb=ram_gb, vram_gb=vram_gb,
                )
            sysmon.set_learned_profile(d.id, ram_gb=ram_gb, vram_gb=vram_gb)
        except Exception:  # noqa: BLE001 - telemetry must not break a load
            pass

    async def _guard_budget(self, backend: GpuBackend) -> None:
        """Refuse a load that would risk the pagefile (raises MemoryError, which
        the worker turns into a clear job error instead of an OOM hang). Before
        raising, publish a structured note so the UI can show *why* it refused."""
        from ..config import settings  # noqa: PLC0415
        from ..util import sysmon  # noqa: PLC0415

        if settings.stub_mode:
            return
        d = backend.descriptor
        decision = sysmon.ram_budget(d.family, d.size_bytes, d.quant, d.id)
        if decision["ok"]:
            return
        await self._bus.publish(Event(
            EventType.ARBITER_NOTE,
            reason="ram_budget",
            message=(
                f"Refused {d.name}: needs ~{decision['need_gb']:.1f} GB + "
                f"{decision['headroom_gb']:.0f} GB headroom, only "
                f"{decision['available_gb']:.1f} GB RAM free."
            ),
            model=d.name,
            family=d.family.value,
            predicted_gb=decision["need_gb"],
            available_gb=decision["available_gb"],
        ))
        raise MemoryError(sysmon.ram_budget_message(decision))

    def status(self) -> dict:
        cur = self._current
        return {
            "resident": cur.resident_key if cur else None,
            "model_id": cur.descriptor.id if cur else None,
            "model": cur.descriptor.name if cur else None,
            "family": cur.descriptor.family.value if cur else None,
            "warm": [
                {
                    "resident": b.resident_key,
                    "model_id": b.descriptor.id,
                    "model": b.descriptor.name,
                    "family": b.descriptor.family.value,
                }
                for b in self._warm_backends
            ],
        }

    async def _publish_status(self) -> None:
        await self._bus.publish(Event(EventType.GPU_STATUS, **self.status()))


def _measured_from_report(report: dict | None) -> tuple[float | None, float | None]:
    """Extract (ram_gb, vram_gb) measurements from an image backend load report.

    RAM = the process RSS the load added (end - start); VRAM = the process
    reserved VRAM after the load (falls back to the device-used delta). Small or
    negative deltas (noise / gc) are dropped. LLM reports are ``None`` (the model
    is a separate process), so they contribute nothing here."""
    memory = (report or {}).get("memory") or {}
    start = memory.get("start") or {}
    end = memory.get("end") or {}

    def rss(snap: dict) -> float | None:
        return (snap.get("ram") or {}).get("process_rss_gb")

    ram_gb: float | None = None
    if rss(start) is not None and rss(end) is not None:
        delta = rss(end) - rss(start)
        if delta >= 0.3:
            ram_gb = round(delta, 2)

    vram_gb: float | None = None
    reserved = (end.get("cuda_process") or {}).get("reserved_gb")
    if reserved:
        vram_gb = round(reserved, 2)
    else:
        used_start = (start.get("vram") or {}).get("used_gb")
        used_end = (end.get("vram") or {}).get("used_gb")
        if used_start is not None and used_end is not None and used_end - used_start > 0.3:
            vram_gb = round(used_end - used_start, 2)

    return ram_gb, vram_gb
