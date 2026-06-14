from __future__ import annotations

from pathlib import Path

from app.backends.base import ModelDescriptor
from app.core.enums import ModelFamily
from app.services import model_compatibility


def desc(
    family: ModelFamily = ModelFamily.SDXL,
    *,
    quant: str | None = None,
    size_bytes: int = 0,
) -> ModelDescriptor:
    return ModelDescriptor(
        id="m",
        name="M",
        family=family,
        path=Path("m.safetensors"),
        size_bytes=size_bytes,
        quant=quant,
    )


def profile(
    *,
    backend: str = "cuda",
    effective_stub: bool = False,
    disabled: list[str] | None = None,
    vram_mb: int | None = 16384,
    model_policy: dict | None = None,
) -> dict:
    return {
        "backend": backend,
        "effective_stub_mode": effective_stub,
        "disabled_features": disabled or [],
        "primary_gpu": {"vram_mb": vram_mb} if vram_mb else None,
        "model_policy": model_policy or {},
    }


def test_stub_profile_keeps_models_queueable():
    compat = model_compatibility.compatibility_for_model(
        desc(ModelFamily.FLUX, quant="nunchaku-fp4"),
        profile=profile(effective_stub=True, disabled=["nunchaku_cuda"]),
        estimated_vram_gb=99,
    )

    assert compat["available"] is True
    assert compat["runtime_mode"] == "stub"
    assert compat["unavailable_reason"] is None


def test_nunchaku_requires_cuda_feature_in_real_mode():
    compat = model_compatibility.compatibility_for_model(
        desc(ModelFamily.FLUX, quant="nunchaku-fp4"),
        profile=profile(backend="rocm", disabled=["nunchaku_cuda"]),
        estimated_vram_gb=8,
    )

    assert compat["available"] is False
    assert "Nunchaku" in compat["unavailable_reason"]


def test_rocm_blocks_bitsandbytes_quantized_image_models():
    compat = model_compatibility.compatibility_for_model(
        desc(ModelFamily.QWEN_IMAGE, quant="bnb-nf4"),
        profile=profile(backend="rocm"),
        estimated_vram_gb=12,
    )

    assert compat["available"] is False
    assert "bitsandbytes" in compat["unavailable_reason"]


def test_real_mode_blocks_models_that_exceed_gpu_vram():
    compat = model_compatibility.compatibility_for_model(
        desc(ModelFamily.QWEN_IMAGE),
        profile=profile(backend="cuda", vram_mb=8192),
        estimated_vram_gb=15,
    )

    assert compat["available"] is False
    assert "Estimated VRAM" in compat["unavailable_reason"]


def test_recommendation_reflects_model_policy_buckets():
    policy = {"image": {"recommended": ["sdxl"], "advanced": ["flux"], "hidden": []}}
    rec = profile(backend="cuda", model_policy=policy)

    sdxl = model_compatibility.compatibility_for_model(
        desc(ModelFamily.SDXL), profile=rec, estimated_vram_gb=8,
    )
    flux = model_compatibility.compatibility_for_model(
        desc(ModelFamily.FLUX, quant="nunchaku-fp4"), profile=rec, estimated_vram_gb=10,
    )
    assert sdxl["recommendation"] == "recommended"
    assert flux["recommendation"] == "advanced"


def test_recommendation_is_neutral_for_llm_and_stub():
    policy = {"image": {"recommended": ["sdxl"], "advanced": [], "hidden": []}}
    llm = model_compatibility.compatibility_for_model(
        desc(ModelFamily.GGUF), profile=profile(backend="cuda", model_policy=policy),
    )
    stub = model_compatibility.compatibility_for_model(
        desc(ModelFamily.SDXL), profile=profile(effective_stub=True, model_policy=policy),
    )
    assert llm["recommendation"] == "neutral"
    assert stub["recommendation"] == "neutral"


async def test_create_jobs_rejects_unavailable_model(monkeypatch, app_client):
    def reject(_desc):
        raise ValueError("blocked by capability profile")

    monkeypatch.setattr(model_compatibility, "require_model_available", reject)
    models = (await app_client.get("/api/models")).json()
    img = next(model for model in models if model["job_type"] == "image")

    response = await app_client.post(
        "/api/jobs",
        json=[{
            "type": "image",
            "model_id": img["id"],
            "params": {"prompt": "blocked", "steps": 1},
        }],
    )

    assert response.status_code == 409
    assert "blocked by capability profile" in response.text
