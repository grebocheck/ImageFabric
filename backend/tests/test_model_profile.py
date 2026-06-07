"""Learned model-profile persistence (ROADMAP P7.2).

The budget guard trusts these rows over its static heuristic, so the running
*max* must be conservative: a single low sample can never lower a recorded peak.
Runs against the throwaway SQLite DB pinned in conftest.
"""

from __future__ import annotations

import pytest

from app.db.session import init_db, session_scope
from app.services import model_profile_service as mps


@pytest.fixture(autouse=True)
async def _db():
    await init_db()
    yield
    # Clean the table so each test starts fresh (the temp DB file is reused).
    from sqlalchemy import delete

    from app.db.models import ModelProfile
    async with session_scope() as s:
        await s.execute(delete(ModelProfile))


async def test_record_creates_then_keeps_running_max():
    async with session_scope() as s:
        await mps.record(s, model_id="m1", family="sdxl", quant=None, ram_gb=8.0, vram_gb=10.0)
    # A higher VRAM sample but lower RAM sample: RAM must hold its peak.
    async with session_scope() as s:
        await mps.record(s, model_id="m1", family="sdxl", quant=None, ram_gb=6.0, vram_gb=12.0)

    async with session_scope() as s:
        rows = await mps.load_all(s)
    by_id = {r.model_id: r for r in rows}
    assert by_id["m1"].ram_gb == 8.0   # conservative max, not the later 6.0
    assert by_id["m1"].vram_gb == 12.0
    assert by_id["m1"].samples == 2


async def test_record_ignores_none_samples():
    async with session_scope() as s:
        await mps.record(s, model_id="m2", family="flux", quant="nunchaku-fp4", ram_gb=11.0, vram_gb=9.8)
    async with session_scope() as s:
        # An LLM-style report contributes nothing (both None) but still counts a sample.
        await mps.record(s, model_id="m2", family="flux", quant="nunchaku-fp4", ram_gb=None, vram_gb=None)

    async with session_scope() as s:
        rows = await mps.load_all(s)
    prof = {r.model_id: r for r in rows}["m2"]
    assert prof.ram_gb == 11.0
    assert prof.vram_gb == 9.8
    assert prof.samples == 2


async def test_load_all_round_trips_multiple_models():
    async with session_scope() as s:
        await mps.record(s, model_id="a", family="sdxl", quant=None, ram_gb=8.0, vram_gb=10.0)
        await mps.record(s, model_id="b", family="gguf", quant=None, ram_gb=2.0, vram_gb=13.0)
    async with session_scope() as s:
        rows = await mps.load_all(s)
    assert {r.model_id for r in rows} == {"a", "b"}
