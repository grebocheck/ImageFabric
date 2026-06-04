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
                await self._unload_current(allow_keep_warm=True)
            if not backend.loaded:
                self._check_budget(backend)
                warm_resume = backend.warm
                await self._bus.publish(Event(
                    EventType.MODEL_LOADING,
                    resident=backend.resident_key,
                    model=backend.descriptor.name,
                    family=backend.descriptor.family.value,
                    warm_resume=warm_resume,
                ))
                await backend.load()
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
        return sysmon.can_keep_warm(d.family, d.size_bytes, d.quant)

    def _check_budget(self, backend: GpuBackend) -> None:
        """Refuse a load that would risk the pagefile (raises MemoryError, which
        the worker turns into a clear job error instead of an OOM hang)."""
        from ..config import settings  # noqa: PLC0415
        from ..util import sysmon  # noqa: PLC0415

        if settings.stub_mode:
            return
        d = backend.descriptor
        sysmon.check_ram_budget(d.family, d.size_bytes, d.quant)

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
