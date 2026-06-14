"""Per-model compatibility checks derived from the active capability profile."""

from __future__ import annotations

from typing import Any

from ..backends.base import ModelDescriptor
from ..core.enums import JobType, ModelFamily
from ..util import sysmon
from . import capability_profile


def compatibility_for_model(
    desc: ModelDescriptor,
    *,
    profile: dict[str, Any] | None = None,
    estimated_vram_gb: float | None = None,
) -> dict[str, Any]:
    """Return queue/UI compatibility metadata for one discovered model."""
    if estimated_vram_gb is None:
        estimated_vram_gb = sysmon.estimate_vram_need_gb(
            desc.family, desc.size_bytes, desc.quant, desc.id
        )

    profile = profile or capability_profile.get_capability_profile()
    if profile.get("effective_stub_mode"):
        return {
            "available": True,
            "runtime_mode": "stub",
            "unavailable_reason": None,
            "compatibility_warnings": ["Runs through the STUB pipeline in this process."],
            "recommendation": "neutral",
        }

    backend = str(profile.get("backend") or "cpu")
    disabled = set(profile.get("disabled_features") or [])
    warnings: list[str] = []

    if desc.job_type is JobType.IMAGE:
        unavailable = _image_unavailable_reason(desc, profile, backend, disabled, estimated_vram_gb)
        if unavailable:
            return {
                "available": False,
                "runtime_mode": "disabled",
                "unavailable_reason": unavailable,
                "compatibility_warnings": warnings,
                "recommendation": "hidden",
            }
        if backend == "rocm" and desc.family in {
            ModelFamily.FLUX,
            ModelFamily.FLUX2,
            ModelFamily.QWEN_IMAGE,
            ModelFamily.Z_IMAGE,
        }:
            warnings.append("This image family is not fully validated on ROCm yet.")

    if desc.job_type is JobType.LLM and backend == "rocm" and "cuda_llama_binaries" in disabled:
        warnings.append("ROCm disables CUDA llama binaries; use a CPU/ROCm-safe llama build or lower GPU layers.")

    return {
        "available": True,
        "runtime_mode": "real",
        "unavailable_reason": None,
        "compatibility_warnings": warnings,
        "recommendation": _recommendation(desc, profile),
    }


def _recommendation(desc: ModelDescriptor, profile: dict[str, Any]) -> str:
    """Hardware-fit hint for an available model: recommended / advanced / neutral.

    Derived from the capability profile's per-family `model_policy` (P20.3). LLMs
    have no per-family policy yet, so they stay neutral.
    """
    if desc.job_type is not JobType.IMAGE:
        return "neutral"
    image_policy = ((profile.get("model_policy") or {}).get("image")) or {}
    family = desc.family.value
    if family in (image_policy.get("recommended") or []):
        return "recommended"
    if family in (image_policy.get("advanced") or []):
        return "advanced"
    return "neutral"


def require_model_available(desc: ModelDescriptor) -> None:
    """Raise ``ValueError`` when a model should not be queued in this runtime."""
    compat = compatibility_for_model(desc)
    if not compat["available"]:
        raise ValueError(str(compat["unavailable_reason"] or "model is unavailable in this runtime"))


def _image_unavailable_reason(
    desc: ModelDescriptor,
    profile: dict[str, Any],
    backend: str,
    disabled: set[str],
    estimated_vram_gb: float | None,
) -> str | None:
    quant = desc.quant or ""
    if quant.startswith("nunchaku") and "nunchaku_cuda" in disabled:
        return "Nunchaku image models require the NVIDIA CUDA/Nunchaku profile."

    if backend == "rocm" and quant.startswith("bnb-"):
        return "bitsandbytes-quantized image models are not enabled for the ROCm profile yet."

    if backend == "cpu":
        return "Real image model loading requires an accelerator profile; use STUB/CPU-safe mode instead."

    gpu_vram_gb = _primary_gpu_vram_gb(profile)
    if estimated_vram_gb and gpu_vram_gb and estimated_vram_gb > gpu_vram_gb + 0.5:
        return f"Estimated VRAM need is ~{estimated_vram_gb:.1f} GB, but the selected GPU reports {gpu_vram_gb:.1f} GB."

    return None


def _primary_gpu_vram_gb(profile: dict[str, Any]) -> float | None:
    gpu = profile.get("primary_gpu") if isinstance(profile.get("primary_gpu"), dict) else None
    if not gpu:
        return None
    try:
        value = gpu.get("vram_mb")
        return float(value) / 1024 if value else None
    except (TypeError, ValueError):
        return None
