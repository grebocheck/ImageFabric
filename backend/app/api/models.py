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
        prof = sysmon.get_learned_profile(d.id)
        out.append(ModelOut(
            id=d.id, name=d.name, family=d.family, job_type=d.job_type,
            size_bytes=d.size_bytes, loaded=loaded, warm=warm, quant=d.quant,
            estimated_vram_gb=sysmon.estimate_vram_need_gb(d.family, d.size_bytes, d.quant, d.id),
            vram_measured=bool(prof and prof.get("vram_gb")),
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
            "transcription_models_dir": str(settings.transcription_models_dir),
            "embed_models_dir": str(settings.embed_models_dir),
            "vision_models_dir": str(settings.vision_models_dir),
            "outputs_dir": str(settings.outputs_dir),
            "db_path": str(settings.db_path),
            "llama_server_bin": str(settings.llama_server_bin),
            "llama_tts_bin": str(settings.llama_tts_bin),
            "llama_mtmd_bin": str(settings.llama_mtmd_bin),
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
            "qwen_image_quant": settings.qwen_image_quant,
            "qwen_image_offload": settings.qwen_image_offload,
            "qwen_image_default_steps": settings.qwen_image_default_steps,
            "qwen_image_default_guidance": settings.qwen_image_default_guidance,
            "qwen_image_default_width": settings.qwen_image_default_width,
            "qwen_image_default_height": settings.qwen_image_default_height,
            "z_image_offload": settings.z_image_offload,
            "z_image_default_steps": settings.z_image_default_steps,
            "z_image_default_guidance": settings.z_image_default_guidance,
            "z_image_default_width": settings.z_image_default_width,
            "z_image_default_height": settings.z_image_default_height,
            "sdxl_turbo_lora": settings.sdxl_turbo_lora,
            "image_cleanup_after_each_job": settings.image_cleanup_after_each_job,
            "image_lora_cache_max": settings.image_lora_cache_max,
            "image_recycle_cuda_growth_gb": settings.image_recycle_cuda_growth_gb,
            "image_recycle_min_jobs": settings.image_recycle_min_jobs,
            "tts_gpu_layers": settings.tts_gpu_layers,
            "tts_timeout_seconds": settings.tts_timeout_seconds,
            "transcription_device": settings.transcription_device,
            "transcription_compute_type": settings.transcription_compute_type,
            "transcription_timeout_seconds": settings.transcription_timeout_seconds,
            "embed_gpu_layers": settings.embed_gpu_layers,
            "embed_timeout_seconds": settings.embed_timeout_seconds,
            "rag_chunk_chars": settings.rag_chunk_chars,
            "rag_chunk_overlap": settings.rag_chunk_overlap,
            "vision_gpu_layers": settings.vision_gpu_layers,
            "vision_timeout_seconds": settings.vision_timeout_seconds,
        },
        "counts": {
            "models": len(descriptors),
            "image_models": sum(1 for d in descriptors if d.job_type.value == "image"),
            "llm_models": sum(1 for d in descriptors if d.job_type.value == "llm"),
            "loras": len(registry.loras()),
            "tts_models": len(list(settings.tts_models_dir.glob("*.gguf")))
            if settings.tts_models_dir.exists()
            else 0,
            "transcription_models": len([
                p for p in settings.transcription_models_dir.iterdir()
                if not p.name.startswith(".")
            ])
            if settings.transcription_models_dir.exists()
            else 0,
            "embed_models": len(list(settings.embed_models_dir.glob("*.gguf")))
            if settings.embed_models_dir.exists()
            else 0,
            "vision_models": len(list(settings.vision_models_dir.glob("*.gguf")))
            if settings.vision_models_dir.exists()
            else 0,
            "learned_profiles": sysmon.learned_count(),
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
