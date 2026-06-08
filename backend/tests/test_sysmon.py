"""Memory telemetry + the pre-load RAM budget guard (ROADMAP objective #1).

The guard is what keeps the app off the pagefile, so its math — predicted-vs-
available, learned-overrides-static — must not regress. RAM readings are
monkeypatched so the assertions don't depend on the host's free memory.
"""

from __future__ import annotations

import pytest

from app.config import settings
from app.core.enums import ModelFamily
from app.util import sysmon


@pytest.fixture(autouse=True)
def _clean_learned_cache():
    sysmon._learned.clear()
    yield
    sysmon._learned.clear()


def _fix_available_ram(monkeypatch, available_gb: float):
    monkeypatch.setattr(sysmon, "ram_stats", lambda: {"available_gb": available_gb})


# --------------------------------------------------------- static estimates


def test_gguf_ram_estimate_is_low_because_mmap():
    # llama-server mmaps the GGUF (disk-backed), so RSS stays low regardless of size.
    assert sysmon.estimate_ram_need_gb(ModelFamily.GGUF, 8 * 1_000_000_000, None) == 2.0


def test_flux_nunchaku_vram_is_measured_constant():
    assert sysmon.estimate_vram_need_gb(ModelFamily.FLUX, 0, "nunchaku-fp4") == 9.8


def test_raw_flux_vram_can_overflow_16gb_card():
    # Raw fp8 path: estimate is at least 16 GB so the UI flags the overflow risk.
    assert sysmon.estimate_vram_need_gb(ModelFamily.FLUX, 20 * 1_000_000_000, None) >= 16.0


def test_sdxl_vram_is_clamped():
    huge = sysmon.estimate_vram_need_gb(ModelFamily.SDXL, 100 * 1_000_000_000, None)
    tiny = sysmon.estimate_vram_need_gb(ModelFamily.SDXL, 1_000_000, None)
    assert huge == 12.5  # upper clamp
    assert tiny == 8.0   # lower clamp


def test_qwen_and_z_image_vram_estimates_are_family_specific():
    assert sysmon.estimate_vram_need_gb(ModelFamily.QWEN_IMAGE, 54 * 1_000_000_000, "bnb-nf4") == 15.0
    assert sysmon.estimate_vram_need_gb(ModelFamily.Z_IMAGE, 31 * 1_000_000_000, None) == 15.0


# ------------------------------------------------------ learned overrides


def test_learned_profile_overrides_static_ram_estimate():
    sysmon.set_learned_profile("flux-x", ram_gb=11.0)
    got = sysmon.estimate_ram_need_gb(ModelFamily.FLUX, 999, "nunchaku-fp4", "flux-x")
    assert got == pytest.approx(11.0 + settings.learned_ram_margin_gb)


def test_learned_profile_keeps_conservative_max():
    sysmon.set_learned_profile("m", ram_gb=8.0, vram_gb=9.0)
    sysmon.set_learned_profile("m", ram_gb=6.0, vram_gb=12.0)  # lower RAM ignored
    prof = sysmon.get_learned_profile("m")
    assert prof == {"ram_gb": 8.0, "vram_gb": 12.0}


def test_prime_learned_profiles_replaces_cache():
    sysmon.set_learned_profile("stale", ram_gb=5.0)
    sysmon.prime_learned_profiles([{"model_id": "fresh", "ram_gb": 7.0, "vram_gb": 8.0}])
    assert sysmon.get_learned_profile("stale") is None
    assert sysmon.learned_count() == 1
    assert sysmon.get_learned_profile("fresh")["vram_gb"] == 8.0


def test_learned_vram_flag_in_budget():
    sysmon.set_learned_profile("m", ram_gb=4.0)
    d = sysmon.ram_budget(ModelFamily.SDXL, 1, None, "m")
    assert d["learned"] is True


# -------------------------------------------------------------- the guard


def test_ram_budget_ok_when_room(monkeypatch):
    _fix_available_ram(monkeypatch, 20.0)
    d = sysmon.ram_budget(ModelFamily.GGUF, 8 * 1_000_000_000, None)
    assert d["ok"] is True
    assert d["need_gb"] == 2.0


def test_ram_budget_refuses_when_tight(monkeypatch):
    _fix_available_ram(monkeypatch, 3.0)
    # SDXL single-file ~10 GB * 1.3 = 13 GB need, only 3 GB free -> refuse.
    d = sysmon.ram_budget(ModelFamily.SDXL, 10 * 1_000_000_000, None)
    assert d["ok"] is False


def test_check_ram_budget_raises_clear_memory_error(monkeypatch):
    _fix_available_ram(monkeypatch, 1.0)
    with pytest.raises(MemoryError, match="Not enough RAM"):
        sysmon.check_ram_budget(ModelFamily.SDXL, 10 * 1_000_000_000, None)


def test_check_ram_budget_passes_when_room(monkeypatch):
    _fix_available_ram(monkeypatch, 30.0)
    sysmon.check_ram_budget(ModelFamily.GGUF, 8 * 1_000_000_000, None)  # no raise


def test_headroom_is_respected(monkeypatch):
    # need(GGUF)=2.0; with headroom 2.5 the load needs 4.5 GB free. 4.0 must fail.
    _fix_available_ram(monkeypatch, 4.0)
    assert sysmon.ram_budget(ModelFamily.GGUF, 0, None)["ok"] is False
    _fix_available_ram(monkeypatch, 5.0)
    assert sysmon.ram_budget(ModelFamily.GGUF, 0, None)["ok"] is True


def test_can_keep_warm_decision(monkeypatch):
    _fix_available_ram(monkeypatch, 30.0)
    ok, _ = sysmon.can_keep_warm(ModelFamily.SDXL, 5 * 1_000_000_000, None)
    assert ok is True
    _fix_available_ram(monkeypatch, 5.0)
    ok, msg = sysmon.can_keep_warm(ModelFamily.SDXL, 10 * 1_000_000_000, None)
    assert ok is False
    assert "parking would need" in msg
