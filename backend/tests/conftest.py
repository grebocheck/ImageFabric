"""Shared test setup.

Critically, this runs *before* any `app.*` import: `app.config` builds a cached
`Settings` and `app.db.session` builds the async engine at import time, both from
the environment. So we pin STUB mode, a throwaway SQLite file, and temp model
dirs (seeded with dummy model files) here, so tests never touch the GPU stack or
the real `data/hfabric.db` and the registry still has something to discover.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import struct
import tempfile

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

_TMP = Path(tempfile.gettempdir()) / "hfabric_test"
_IMAGE_DIR = _TMP / "image"
_LLM_DIR = _TMP / "llm"
_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
_LLM_DIR.mkdir(parents=True, exist_ok=True)


def _write_safetensors(path: Path, keys: list[str]) -> None:
    """Write a minimal valid safetensors file (8-byte header length + JSON header
    + 2 data bytes) so the classifier reads real keys without multi-GB weights."""
    header = {k: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]} for k in keys}
    blob = json.dumps(header).encode("utf-8")
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(b"\x00\x00")


# An SDXL-classified image model (UNet `input_blocks.*` key) and a GGUF LLM
# (recognized purely by extension). Enough for the stub pipeline + swap test.
_SDXL = _IMAGE_DIR / "stub-sdxl.safetensors"
if not _SDXL.exists():
    _write_safetensors(_SDXL, ["model.diffusion_model.input_blocks.0.0.weight"])
_GGUF = _LLM_DIR / "stub-llm.gguf"
if not _GGUF.exists():
    _GGUF.write_bytes(b"GGUF\x00")

os.environ.setdefault("HFAB_STUB_MODE", "true")
# Popping is not enough: pydantic-settings also reads the repo .env FILE, so a
# developer's real HFAB_API_TOKEN/HFAB_HOST would leak into the suite (every
# request suddenly 401s). Env vars take precedence over env_file - pin them.
os.environ["HFAB_API_TOKEN"] = ""
os.environ["HFAB_HOST"] = "127.0.0.1"
os.environ["HFAB_PORT"] = "8260"
os.environ["HFAB_DB_PATH"] = str(_TMP / "hfabric_test.db")
os.environ["HFAB_DATA_DIR"] = str(_TMP / "data")
os.environ["HFAB_OUTPUTS_DIR"] = str(_TMP / "outputs")
os.environ["HFAB_IMAGE_MODELS_DIR"] = str(_IMAGE_DIR)
os.environ["HFAB_LLM_MODELS_DIR"] = str(_LLM_DIR)
# Keep the budget guard deterministic regardless of the host's free RAM.
os.environ.setdefault("HFAB_LEARN_MEMORY_PROFILES", "false")


@pytest.fixture
async def isolated_runtime(monkeypatch, tmp_path):
    """Point global settings + DB session helpers at a per-test runtime tree."""
    from app.config import settings
    from app.db import session as db_session

    data_dir = tmp_path / "data"
    monkeypatch.setattr(settings, "data_dir", data_dir)
    monkeypatch.setattr(settings, "outputs_dir", data_dir / "outputs")
    monkeypatch.setattr(settings, "logs_dir", data_dir / "logs")
    monkeypatch.setattr(settings, "runtime_dir", data_dir / "runtime")
    monkeypatch.setattr(settings, "backups_dir", data_dir / "backups")
    monkeypatch.setattr(settings, "db_path", data_dir / "hfabric.db")

    old_engine = db_session.engine
    old_session_local = db_session.SessionLocal
    new_engine = create_async_engine(settings.db_url, future=True)
    db_session.engine = new_engine
    db_session.SessionLocal = async_sessionmaker(
        new_engine,
        expire_on_commit=False,
        class_=AsyncSession,
    )
    try:
        yield {
            "data_dir": data_dir,
            "db_path": settings.db_path,
            "outputs_dir": settings.outputs_dir,
            "logs_dir": settings.logs_dir,
            "runtime_dir": settings.runtime_dir,
            "backups_dir": settings.backups_dir,
        }
    finally:
        await new_engine.dispose()
        db_session.engine = old_engine
        db_session.SessionLocal = old_session_local
