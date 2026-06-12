from __future__ import annotations

import asyncio

from httpx import ASGITransport, AsyncClient

from app.main import app


async def test_stub_job_writes_rotating_file_log(isolated_runtime):
    log_path = isolated_runtime["logs_dir"] / "hfabric.log"
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            models = (await client.get("/api/models")).json()
            image_model = next(model["id"] for model in models if model["job_type"] == "image")
            created = (await client.post(
                "/api/jobs",
                json=[{
                    "type": "image",
                    "model_id": image_model,
                    "params": {"prompt": "log test", "steps": 1, "width": 256, "height": 256},
                }],
            )).json()
            await _wait_done(client, created[0]["id"])
            text = await _wait_log(log_path, "event=job.done")

    assert "event=startup.config" in text
    assert "event=job.started" in text
    assert "event=job.done" in text


async def _wait_done(client: AsyncClient, job_id: str) -> None:
    deadline = asyncio.get_event_loop().time() + 20.0
    while asyncio.get_event_loop().time() < deadline:
        status = (await client.get(f"/api/jobs/{job_id}")).json()["status"]
        if status in {"done", "error", "cancelled"}:
            assert status == "done"
            return
        await asyncio.sleep(0.1)
    raise AssertionError(f"job {job_id} did not finish")


async def _wait_log(path, needle: str) -> str:
    deadline = asyncio.get_event_loop().time() + 5.0
    while asyncio.get_event_loop().time() < deadline:
        if path.exists():
            text = path.read_text(encoding="utf-8")
            if needle in text:
                return text
        await asyncio.sleep(0.05)
    raise AssertionError(f"{needle!r} was not written to {path}")
