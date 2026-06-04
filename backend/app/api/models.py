"""Model discovery + live GPU/arbiter status."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..backends.registry import ModelRegistry
from ..config import settings
from ..core.arbiter import GpuArbiter
from ..core.enums import ModelFamily
from ..schemas import GpuStatusOut, LoraOut, ModelOut
from ..util import sysmon
from .deps import get_arbiter, get_registry

router = APIRouter(prefix="/api", tags=["models"])


@router.get("/models", response_model=list[ModelOut])
async def list_models(
    registry: ModelRegistry = Depends(get_registry),
    arbiter: GpuArbiter = Depends(get_arbiter),
) -> list[ModelOut]:
    current = arbiter.current
    out: list[ModelOut] = []
    for d in registry.descriptors():
        loaded = current is not None and current.descriptor.id == d.id
        existing = registry.peek_backend(d.id)
        warm = bool(existing and existing.warm)
        # raw fp8 FLUX (no quant backend) is the slow / high-mem path on 16 GB
        slow = d.family is ModelFamily.FLUX and d.quant is None
        out.append(ModelOut(
            id=d.id, name=d.name, family=d.family, job_type=d.job_type,
            size_bytes=d.size_bytes, loaded=loaded, warm=warm, quant=d.quant,
            estimated_vram_gb=sysmon.estimate_vram_need_gb(d.family, d.size_bytes, d.quant),
            slow=slow,
        ))
    return out


@router.get("/loras", response_model=list[LoraOut])
async def list_loras(
    family: ModelFamily | None = None,
    registry: ModelRegistry = Depends(get_registry),
) -> list[LoraOut]:
    return [
        LoraOut(id=l.id, name=l.name, family=l.family, size_bytes=l.size_bytes)
        for l in registry.loras(family)
    ]


@router.get("/settings")
async def runtime_settings(
    registry: ModelRegistry = Depends(get_registry),
    arbiter: GpuArbiter = Depends(get_arbiter),
) -> dict:
    descriptors = registry.descriptors()
    return {
        "stub_mode": settings.stub_mode,
        "paths": {
            "image_models_dir": str(settings.image_models_dir),
            "lora_models_dir": str(settings.lora_models_dir),
            "llm_models_dir": str(settings.llm_models_dir),
            "tts_models_dir": str(settings.tts_models_dir),
            "outputs_dir": str(settings.outputs_dir),
            "db_path": str(settings.db_path),
            "llama_server_bin": str(settings.llama_server_bin),
            "llama_tts_bin": str(settings.llama_tts_bin),
        },
        "memory": {
            "min_free_ram_gb": settings.min_free_ram_gb,
            "keep_warm_models": settings.keep_warm_models,
            "keep_warm_max_models": settings.keep_warm_max_models,
            "keep_warm_min_available_ram_gb": settings.keep_warm_min_available_ram_gb,
            "mem_poll_seconds": settings.mem_poll_seconds,
        },
        "acceleration": {
            "attention_backend": settings.attention_backend,
            "attention_allow_tf32": settings.attention_allow_tf32,
            "attention_matmul_precision": settings.attention_matmul_precision,
            "torch_compile": settings.torch_compile,
            "torch_compile_mode": settings.torch_compile_mode,
            "flux_step_cache": settings.flux_step_cache,
            "sdxl_turbo_lora": settings.sdxl_turbo_lora,
            "tts_gpu_layers": settings.tts_gpu_layers,
            "tts_timeout_seconds": settings.tts_timeout_seconds,
        },
        "counts": {
            "models": len(descriptors),
            "image_models": sum(1 for d in descriptors if d.job_type.value == "image"),
            "llm_models": sum(1 for d in descriptors if d.job_type.value == "llm"),
            "loras": len(registry.loras()),
            "tts_models": len(list(settings.tts_models_dir.glob("*.gguf")))
            if settings.tts_models_dir.exists()
            else 0,
        },
        "gpu": arbiter.status(),
        "mem": sysmon.snapshot(),
    }


@router.get("/gpu", response_model=GpuStatusOut)
async def gpu_status(arbiter: GpuArbiter = Depends(get_arbiter)) -> GpuStatusOut:
    return GpuStatusOut(**arbiter.status())


@router.post("/gpu/free", response_model=GpuStatusOut)
async def gpu_free(arbiter: GpuArbiter = Depends(get_arbiter)) -> GpuStatusOut:
    await arbiter.free_all()
    return GpuStatusOut(**arbiter.status())
