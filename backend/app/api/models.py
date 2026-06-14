"""Model discovery + live GPU/arbiter status."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from ..backends.registry import ModelRegistry
from ..config import settings
from ..core.arbiter import GpuArbiter
from ..core.enums import ModelFamily
from ..schemas import GpuStatusOut, LoraOut, ModelOut, ModelProfileOut
from ..services import capability_profile, model_compatibility, settings_overrides
from ..services import model_profile_service as mps
from ..util import sysmon
from .deps import get_arbiter, get_registry, get_session

router = APIRouter(prefix="/api", tags=["models"])


@router.get("/models", response_model=list[ModelOut])
async def list_models(
    registry: ModelRegistry = Depends(get_registry),
    arbiter: GpuArbiter = Depends(get_arbiter),
) -> list[ModelOut]:
    current = arbiter.current
    out: list[ModelOut] = []
    profile = capability_profile.get_capability_profile()
    for d in registry.descriptors():
        loaded = current is not None and current.descriptor.id == d.id
        existing = registry.peek_backend(d.id)
        warm = bool(existing and existing.warm)
        # raw fp8 FLUX (no quant backend) is the slow / high-mem path on 16 GB
        slow = d.family is ModelFamily.FLUX and d.quant is None
        prof = sysmon.get_learned_profile(d.id)
        estimated_vram = sysmon.estimate_vram_need_gb(d.family, d.size_bytes, d.quant, d.id)
        compat = model_compatibility.compatibility_for_model(
            d,
            profile=profile,
            estimated_vram_gb=estimated_vram,
        )
        out.append(ModelOut(
            id=d.id, name=d.name, family=d.family, job_type=d.job_type,
            size_bytes=d.size_bytes, loaded=loaded, warm=warm, quant=d.quant,
            estimated_vram_gb=estimated_vram,
            vram_measured=bool(prof and prof.get("vram_gb")),
            slow=slow,
            available=compat["available"],
            runtime_mode=compat["runtime_mode"],
            unavailable_reason=compat["unavailable_reason"],
            compatibility_warnings=compat["compatibility_warnings"],
            recommendation=compat.get("recommendation", "neutral"),
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


@router.get("/models/profiles", response_model=list[ModelProfileOut])
async def list_model_profiles(
    session: AsyncSession = Depends(get_session),
    registry: ModelRegistry = Depends(get_registry),
) -> list[ModelProfileOut]:
    by_id = {d.id: d for d in registry.descriptors()}
    rows = await mps.load_all(session)
    return [
        ModelProfileOut(
            model_id=row.model_id,
            model=by_id.get(row.model_id).name if row.model_id in by_id else row.model_id,
            family=row.family,
            quant=row.quant,
            ram_gb=row.ram_gb,
            vram_gb=row.vram_gb,
            samples=row.samples,
            updated_at=row.updated_at,
        )
        for row in rows
    ]


@router.delete("/models/profiles")
async def reset_all_model_profiles(session: AsyncSession = Depends(get_session)) -> dict[str, int]:
    deleted = await mps.delete_all(session)
    sysmon.clear_learned_profiles()
    return {"deleted": deleted}


@router.delete("/models/profiles/{model_id}")
async def reset_model_profile(
    model_id: str,
    all_profiles: bool = Query(False, alias="all"),
    session: AsyncSession = Depends(get_session),
) -> dict[str, int]:
    if all_profiles:
        deleted = await mps.delete_all(session)
        sysmon.clear_learned_profiles()
        return {"deleted": deleted}
    deleted = await mps.delete(session, model_id)
    sysmon.delete_learned_profile(model_id)
    return {"deleted": deleted}


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
        "generation_defaults": {
            "default_steps": settings.default_steps,
            "default_guidance": settings.default_guidance,
            "default_width": settings.default_width,
            "default_height": settings.default_height,
            "keep_warm_models": settings.keep_warm_models,
            "keep_warm_max_models": settings.keep_warm_max_models,
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
        "capability": capability_profile.get_capability_profile(),
    }


@router.get("/capabilities")
async def runtime_capabilities(refresh: bool = Query(False)) -> dict[str, Any]:
    return capability_profile.get_capability_profile(refresh=refresh)


@router.get("/settings/overrides")
async def get_settings_overrides() -> dict[str, Any]:
    return settings_overrides.payload()


@router.put("/settings/overrides")
async def put_settings_overrides(body: dict[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(body) - settings_overrides.WRITABLE_KEYS)
    if unknown:
        raise HTTPException(422, f"settings are env-only or unknown: {', '.join(unknown)}")
    try:
        return settings_overrides.save(body)
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@router.get("/gpu", response_model=GpuStatusOut)
async def gpu_status(arbiter: GpuArbiter = Depends(get_arbiter)) -> GpuStatusOut:
    return GpuStatusOut(**arbiter.status())


@router.post("/gpu/free", response_model=GpuStatusOut)
async def gpu_free(arbiter: GpuArbiter = Depends(get_arbiter)) -> GpuStatusOut:
    await arbiter.free_all()
    return GpuStatusOut(**arbiter.status())
