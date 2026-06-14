"""Bridge hardware detection to runtime acceleration settings (P20.5).

The :mod:`capability_profile` knows which acceleration knobs are safe on the
detected GPU. This module applies those to the live ``settings`` for any knob the
user did *not* set explicitly (env var or persisted override), so a fresh machine
gets a working configuration without env archaeology.

Design rule: autotune only ever moves a knob toward the *safer / more correct*
value for the hardware (e.g. ``math`` attention and TF32 off on a pre-Ampere,
ROCm, or CPU box). It never auto-enables an aggressive path such as
``torch.compile`` — that stays an explicit opt-in.
"""

from __future__ import annotations

from typing import Any

# Knobs autotune is allowed to adjust. Deliberately excludes torch_compile.
AUTOTUNE_KEYS = ("attention_backend", "flux_step_cache", "attention_allow_tf32")


def capability_acceleration(profile: dict[str, Any]) -> dict[str, Any]:
    """The acceleration knob values implied by the detected hardware profile."""
    runtime_defaults = profile.get("runtime_defaults") or {}
    backend = str(profile.get("backend") or "cpu")
    gpu = profile.get("primary_gpu") if isinstance(profile.get("primary_gpu"), dict) else {}
    cap = (gpu or {}).get("compute_capability_tuple") or []
    ampere_plus = (
        backend == "cuda"
        and len(cap) >= 2
        and (int(cap[0]), int(cap[1])) >= (8, 0)
    )

    desired: dict[str, Any] = {}
    if runtime_defaults.get("attention_backend"):
        desired["attention_backend"] = runtime_defaults["attention_backend"]
    if runtime_defaults.get("flux_step_cache"):
        desired["flux_step_cache"] = runtime_defaults["flux_step_cache"]
    # TF32 tensor-core math is an NVIDIA Ampere+ feature; it is meaningless or
    # unsupported on pre-Ampere NVIDIA, ROCm, and CPU, so disable it there.
    desired["attention_allow_tf32"] = ampere_plus
    return desired


def apply_autotune(
    settings: Any,
    profile: dict[str, Any],
    *,
    user_set: set[str],
    enabled: bool = True,
) -> dict[str, dict[str, Any]]:
    """Mutate ``settings`` toward the hardware-appropriate acceleration defaults.

    Skips any knob in ``user_set`` (explicitly chosen via env or a saved
    override). Returns ``{key: {"from", "to"}}`` for every value actually
    changed, so the caller can log the auto-tuning transparently.
    """
    if not enabled:
        return {}

    desired = capability_acceleration(profile)
    applied: dict[str, dict[str, Any]] = {}
    for key in AUTOTUNE_KEYS:
        if key not in desired or key in user_set:
            continue
        current = getattr(settings, key, None)
        if current != desired[key]:
            applied[key] = {"from": current, "to": desired[key]}
            setattr(settings, key, desired[key])
    return applied
