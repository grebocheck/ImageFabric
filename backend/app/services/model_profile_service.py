"""Persistence for learned per-model memory profiles (ROADMAP P7.2).

After a real load the arbiter records the model's measured RAM/VRAM here; the
budget guard then prefers these measurements over the static heuristic. We keep
a conservative running *max* so a single low sample never under-predicts.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.models import ModelProfile


async def record(
    session: AsyncSession,
    *,
    model_id: str,
    family: str,
    quant: str | None,
    ram_gb: float | None,
    vram_gb: float | None,
) -> ModelProfile:
    prof = await session.get(ModelProfile, model_id)
    if prof is None:
        prof = ModelProfile(
            model_id=model_id, family=family, quant=quant,
            ram_gb=ram_gb, vram_gb=vram_gb, samples=1,
        )
        session.add(prof)
        return prof
    if ram_gb is not None:
        prof.ram_gb = max(prof.ram_gb or 0.0, ram_gb)
    if vram_gb is not None:
        prof.vram_gb = max(prof.vram_gb or 0.0, vram_gb)
    prof.samples += 1
    prof.updated_at = datetime.now(timezone.utc)
    return prof


async def load_all(session: AsyncSession) -> list[ModelProfile]:
    return list((await session.execute(select(ModelProfile))).scalars().all())
