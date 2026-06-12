"""Stub-mode tests for the native realtime voice session (P6R.2)."""

from __future__ import annotations

import asyncio

from httpx import ASGITransport, AsyncClient
import pytest

from app.config import settings
from app.main import app
from app.services.voice_engine import engine as engine_mod
from app.services.voice_engine import realtime


@pytest.fixture
async def client(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "voice_models_dir", tmp_path / "voice")
    monkeypatch.setattr(settings, "voice_pretrain_dir", tmp_path / "pretrain")
    monkeypatch.setattr(engine_mod, "_ENGINE", None)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c
    realtime.stop_session()
    monkeypatch.setattr(engine_mod, "_ENGINE", None)


async def test_session_lifecycle_and_metrics(client):
    before = (await client.get("/api/voice/engine/status")).json()
    assert before["live"] is False
    assert before["metrics"]["input_vu"] == 0.0

    started = await client.post("/api/voice/engine/session/start", json={"model_id": "stub-voice"})
    assert started.status_code == 200
    body = started.json()
    assert body["live"] is True
    metrics = body["metrics"]
    assert 0.0 < metrics["input_vu"] <= 1.0
    assert 0.0 < metrics["output_vu"] <= 1.0
    assert metrics["total_ms"] == 5.0
    assert metrics["chunk_ms"] > 0

    # A second start while live is refused.
    again = await client.post("/api/voice/engine/session/start", json={"model_id": "stub-voice"})
    assert again.status_code == 409

    stopped = await client.post("/api/voice/engine/session/stop")
    assert stopped.status_code == 200
    assert stopped.json()["live"] is False


async def test_session_start_unknown_model_404(client):
    response = await client.post("/api/voice/engine/session/start", json={"model_id": "nope"})
    assert response.status_code == 404
    assert not realtime.session_active()


async def test_voice_lane_parks_queued_jobs(client):
    """A queued GPU job must stay QUEUED while a native session is live and
    run after the session stops (the worker's voice lane)."""
    start = await client.post("/api/voice/engine/session/start", json={"model_id": "stub-voice"})
    assert start.status_code == 200

    models = (await client.get("/api/models")).json()
    image_model = next(m for m in models if m["job_type"] == "image")
    job = (await client.post("/api/jobs", json=[{
        "type": "image",
        "model_id": image_model["id"],
        "params": {"prompt": "voice lane parking test", "steps": 1},
    }])).json()[0]

    # Give the worker a few scheduler ticks: the job must NOT start.
    for _ in range(6):
        await asyncio.sleep(0.05)
        current = (await client.get(f"/api/jobs/{job['id']}")).json()
        assert current["status"] == "queued"

    stop = await client.post("/api/voice/engine/session/stop")
    assert stop.status_code == 200

    async def wait_done() -> str:
        while True:
            state = (await client.get(f"/api/jobs/{job['id']}")).json()
            if state["status"] in {"done", "error"}:
                return state["status"]
            await asyncio.sleep(0.05)

    status = await asyncio.wait_for(wait_done(), timeout=10.0)
    assert status == "done"
