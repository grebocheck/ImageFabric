"""System memory telemetry + a pre-load budget guard.

Goal (ROADMAP objective #1): every single model load must fit comfortably, so
the app never OOMs, hangs, or spills to the Windows pagefile (which wears the
SSD). We surface RAM/VRAM continuously and refuse a load *with a clear error*
rather than letting the OS page out.
"""

from __future__ import annotations

import psutil

from ..config import settings
from ..core.enums import ModelFamily

_GB = 1e9
_proc = psutil.Process()

# Learned per-model memory measurements (model_id -> {"ram_gb", "vram_gb"}),
# primed from the DB at startup and updated after each real load (P7.2). These
# override the static size*factor estimates below when present.
_learned: dict[str, dict[str, float]] = {}


def set_learned_profile(model_id: str, *, ram_gb: float | None = None, vram_gb: float | None = None) -> None:
    """Merge a measurement into the cache, keeping the conservative max."""
    cur = _learned.setdefault(model_id, {})
    if ram_gb is not None:
        cur["ram_gb"] = max(cur.get("ram_gb", 0.0), ram_gb)
    if vram_gb is not None:
        cur["vram_gb"] = max(cur.get("vram_gb", 0.0), vram_gb)


def get_learned_profile(model_id: str | None) -> dict[str, float] | None:
    return _learned.get(model_id) if model_id else None


def prime_learned_profiles(rows: list[dict]) -> None:
    """Load DB-persisted profiles into the cache at startup."""
    _learned.clear()
    for row in rows:
        set_learned_profile(row["model_id"], ram_gb=row.get("ram_gb"), vram_gb=row.get("vram_gb"))


def learned_count() -> int:
    return len(_learned)


def ram_stats() -> dict:
    vm = psutil.virtual_memory()
    return {
        "total_gb": round(vm.total / _GB, 2),
        "available_gb": round(vm.available / _GB, 2),
        "used_gb": round((vm.total - vm.available) / _GB, 2),
        "percent": vm.percent,
        "process_rss_gb": round(_proc.memory_info().rss / _GB, 2),
    }


_nvml_ready: bool | None = None


def _nvml_handle():
    """Lazy NVML init. NVML reads device-wide VRAM (incl. other processes like
    llama-server) WITHOUT creating a CUDA context in our process — important so
    telemetry doesn't itself touch CUDA."""
    global _nvml_ready
    import pynvml  # noqa: PLC0415

    if _nvml_ready is None:
        pynvml.nvmlInit()
        _nvml_ready = True
    return pynvml.nvmlDeviceGetHandleByIndex(0)


def vram_stats() -> dict | None:
    try:
        import pynvml  # noqa: PLC0415

        mem = pynvml.nvmlDeviceGetMemoryInfo(_nvml_handle())
        return {
            "total_gb": round(mem.total / _GB, 2),
            "free_gb": round(mem.free / _GB, 2),
            "used_gb": round(mem.used / _GB, 2),
        }
    except Exception:
        return None


def cuda_compute_capability() -> tuple[int, int] | None:
    try:
        import pynvml  # noqa: PLC0415

        return tuple(pynvml.nvmlDeviceGetCudaComputeCapability(_nvml_handle()))
    except Exception:
        return None


def is_blackwell_gpu() -> bool:
    capability = cuda_compute_capability()
    return bool(capability and capability[0] >= 12)


def snapshot() -> dict:
    return {"ram": ram_stats(), "vram": vram_stats()}


def _is_nunchaku_quant(quant: str | None) -> bool:
    return bool(quant and quant.startswith("nunchaku"))


def estimate_ram_need_gb(
    family: ModelFamily, size_bytes: int, quant: str | None, model_id: str | None = None
) -> float:
    """Rough CPU-RAM a load will need, by model kind. A learned measurement (the
    real peak RSS a load added) overrides the heuristic once we have one."""
    prof = get_learned_profile(model_id)
    if prof and prof.get("ram_gb"):
        return prof["ram_gb"] + settings.learned_ram_margin_gb
    gb = size_bytes / _GB
    if family is ModelFamily.GGUF:
        return 2.0  # llama-server mmaps the gguf (disk-backed) -> low RSS
    if family is ModelFamily.FLUX2 and _is_nunchaku_quant(quant):
        # fp4 transformer + bnb Qwen3 encoder; measured peak RSS ~12.8 GB
        # (ROADMAP P3.3). size_bytes here sums every .safetensors in the folder
        # (fp4 *and* int4 variants ~10.5 GB), so gb + 3 tracks the real peak.
        return gb + 3.0
    if family is ModelFamily.FLUX2:
        # klein loaded in bnb 4-bit with low_cpu_mem_usage: diffusers streams the
        # bf16 shards and quantizes each to ~4-bit on the fly, so the full bf16
        # repo (size_bytes ~32 GB) never lands in RAM at once. Measured peak RSS
        # was ~9.8 GB at 768² (ROADMAP P3), so ~0.3x the repo + a working-buffer
        # margin tracks reality. The old 0.4x + 3 over-predicted ~16 GB and
        # caused false pre-load refusals on a 32 GB box with a warm allocator.
        return gb * 0.3 + 1.5
    if family is ModelFamily.QWEN_IMAGE:
        if quant and quant.startswith("bnb-"):
            return gb * 0.25 + 3.0
        return gb * 0.65 + 2.0
    if family is ModelFamily.Z_IMAGE:
        return gb * 0.8 + 2.0
    if _is_nunchaku_quant(quant):
        return gb + 4.0  # + the int4 T5 (~3 GB) and headroom
    return gb * 1.3  # diffusers single-file materialization overhead


def estimate_vram_need_gb(
    family: ModelFamily, size_bytes: int, quant: str | None, model_id: str | None = None
) -> float | None:
    """Rough resident VRAM estimate shown in the UI before the user queues work.
    Prefers a learned measurement when available."""
    prof = get_learned_profile(model_id)
    if prof and prof.get("vram_gb"):
        return round(prof["vram_gb"], 1)
    gb = size_bytes / _GB
    if family is ModelFamily.FLUX and _is_nunchaku_quant(quant):
        return 9.8  # M0 measured SVDQuant fp4 on RTX 5070 Ti
    if family is ModelFamily.FLUX:
        return max(16.0, round(gb, 1))  # raw fp8 path can overflow 16 GB cards
    if family is ModelFamily.FLUX2 and _is_nunchaku_quant(quant):
        return 8.0  # SVDQuant transformer + bnb Qwen3 text encoder (target)
    if family is ModelFamily.FLUX2:
        return 13.0  # klein 9B bnb 4-bit + model offload (estimate)
    if family is ModelFamily.QWEN_IMAGE:
        return 15.0 if quant and quant.startswith("bnb-") else 16.0
    if family is ModelFamily.Z_IMAGE:
        return 15.0
    if family is ModelFamily.SDXL:
        return round(min(12.5, max(8.0, gb * 1.65)), 1)
    if family is ModelFamily.GGUF:
        return round(max(2.0, gb + 0.75), 1)  # full offload: weights + context/KV
    return None


def ram_budget(
    family: ModelFamily, size_bytes: int, quant: str | None, model_id: str | None = None
) -> dict:
    """Predicted-vs-available RAM decision for a load. Used both to refuse a load
    and to *explain* the refusal in the UI (arbiter transparency, ROADMAP P7.1)."""
    need = estimate_ram_need_gb(family, size_bytes, quant, model_id)
    available = ram_stats()["available_gb"]
    headroom = settings.min_free_ram_gb
    return {
        "ok": available >= need + headroom,
        "need_gb": round(need, 1),
        "available_gb": available,
        "headroom_gb": round(headroom, 1),
        "learned": bool(get_learned_profile(model_id) and get_learned_profile(model_id).get("ram_gb")),
    }


def ram_budget_message(decision: dict) -> str:
    return (
        f"Not enough RAM to load this model safely: needs ~{decision['need_gb']:.1f} GB + "
        f"{decision['headroom_gb']:.0f} GB headroom, but only {decision['available_gb']:.1f} GB "
        f"is available. Free some memory or pick a lighter model — refusing to "
        f"load rather than risk pagefile thrashing."
    )


def check_ram_budget(
    family: ModelFamily, size_bytes: int, quant: str | None, model_id: str | None = None
) -> None:
    """Raise a clear MemoryError if loading would risk the pagefile."""
    decision = ram_budget(family, size_bytes, quant, model_id)
    if not decision["ok"]:
        raise MemoryError(ram_budget_message(decision))


def can_keep_warm(
    family: ModelFamily, size_bytes: int, quant: str | None, model_id: str | None = None
) -> tuple[bool, str]:
    """Return whether parking a model in CPU RAM keeps enough free headroom."""
    need = estimate_ram_need_gb(family, size_bytes, quant, model_id)
    available = ram_stats()["available_gb"]
    required = need + settings.keep_warm_min_available_ram_gb
    if available < required:
        return (
            False,
            f"parking would need ~{need:.1f} GB RAM plus "
            f"{settings.keep_warm_min_available_ram_gb:.1f} GB keep-warm headroom, "
            f"but only {available:.1f} GB is available",
        )
    return True, (
        f"parking allowed: needs ~{need:.1f} GB RAM, "
        f"{available:.1f} GB available"
    )
