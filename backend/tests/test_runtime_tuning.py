from __future__ import annotations

from types import SimpleNamespace

from app.services.runtime_tuning import apply_autotune, capability_acceleration


def _settings() -> SimpleNamespace:
    # Mirrors the static config defaults the autotune may override.
    return SimpleNamespace(
        attention_backend="auto",
        flux_step_cache="fb",
        attention_allow_tf32=True,
    )


def profile(backend: str, *, attention: str, step_cache: str, cap: list[int] | None = None) -> dict:
    return {
        "backend": backend,
        "runtime_defaults": {"attention_backend": attention, "flux_step_cache": step_cache},
        "primary_gpu": {"compute_capability_tuple": cap or []},
    }


def test_blackwell_leaves_safe_defaults_untouched():
    s = _settings()
    p = profile("cuda", attention="auto", step_cache="fb", cap=[12, 0])
    applied = apply_autotune(s, p, user_set=set())
    # Defaults already match a capable CUDA card -> nothing changes.
    assert applied == {}
    assert s.attention_allow_tf32 is True


def test_pre_ampere_nvidia_forces_math_and_disables_tf32():
    s = _settings()
    p = profile("cuda", attention="math", step_cache="off", cap=[6, 1])
    applied = apply_autotune(s, p, user_set=set())
    assert s.attention_backend == "math"
    assert s.flux_step_cache == "off"
    assert s.attention_allow_tf32 is False
    assert applied["attention_allow_tf32"] == {"from": True, "to": False}


def test_rocm_disables_tf32_and_step_cache():
    s = _settings()
    p = profile("rocm", attention="auto", step_cache="off")
    apply_autotune(s, p, user_set=set())
    assert s.attention_allow_tf32 is False
    assert s.flux_step_cache == "off"
    assert s.attention_backend == "auto"


def test_cpu_safe_forces_math_attention():
    s = _settings()
    p = profile("cpu", attention="math", step_cache="off")
    apply_autotune(s, p, user_set=set())
    assert s.attention_backend == "math"
    assert s.attention_allow_tf32 is False


def test_user_pinned_knobs_are_respected():
    s = _settings()
    p = profile("cpu", attention="math", step_cache="off")
    # The user explicitly chose these via env/override -> autotune must not touch.
    applied = apply_autotune(s, p, user_set={"attention_backend", "attention_allow_tf32", "flux_step_cache"})
    assert applied == {}
    assert s.attention_backend == "auto"
    assert s.attention_allow_tf32 is True
    assert s.flux_step_cache == "fb"


def test_disabled_flag_is_a_no_op():
    s = _settings()
    p = profile("cpu", attention="math", step_cache="off")
    assert apply_autotune(s, p, user_set=set(), enabled=False) == {}
    assert s.attention_backend == "auto"


def test_never_auto_enables_torch_compile():
    s = _settings()
    s.torch_compile = False
    p = profile("cuda", attention="auto", step_cache="fb", cap=[12, 0])
    p["runtime_defaults"]["torch_compile"] = True
    apply_autotune(s, p, user_set=set())
    assert s.torch_compile is False  # autotune never touches compile


def test_capability_acceleration_ampere_gate():
    assert capability_acceleration(profile("cuda", attention="auto", step_cache="fb", cap=[8, 6]))[
        "attention_allow_tf32"
    ] is True
    assert capability_acceleration(profile("cuda", attention="auto", step_cache="fb", cap=[7, 5]))[
        "attention_allow_tf32"
    ] is False
