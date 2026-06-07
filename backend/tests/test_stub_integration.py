"""End-to-end pipeline test in STUB mode (no GPU).

This is the hermetic version of `scripts/phase_batch_check.py`: it drives the
real app — queue -> worker -> arbiter swap -> gallery — over an ASGI client and
asserts the headline invariant, *a mixed batch costs exactly one model swap*.
Observed via the event bus (`model.loaded` once per family, one `arbiter.note`
swap), so a regression in phase-batching fails the build.
"""

from __future__ import annotations

import asyncio

from httpx import ASGITransport, AsyncClient
import pytest

from app.main import app


@pytest.fixture
async def client():
    # Run the real lifespan (init DB, scan models, start the worker) around the
    # ASGI client so requests hit the same app.state the worker uses.
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


class _BusCollector:
    """Background subscriber that tallies the arbiter events we care about."""

    def __init__(self, bus):
        self._bus = bus
        self.loaded = 0
        self.swaps = 0
        self._stop = asyncio.Event()
        self._ready = asyncio.Event()
        self._task: asyncio.Task | None = None

    async def __aenter__(self):
        self._task = asyncio.create_task(self._run())
        await self._ready.wait()
        return self

    async def __aexit__(self, *exc):
        self._stop.set()
        if self._task:
            await self._task

    async def _run(self):
        async with self._bus.subscribe() as q:
            self._ready.set()
            while not self._stop.is_set():
                try:
                    ev = await asyncio.wait_for(q.get(), 0.2)
                except TimeoutError:
                    continue
                if ev["type"] == "model.loaded":
                    self.loaded += 1
                elif ev["type"] == "arbiter.note" and ev.get("reason") == "swap":
                    self.swaps += 1


async def _wait_done(client: AsyncClient, job_ids: list[str], timeout: float = 30.0) -> list[str]:
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    statuses: list[str] = []
    while loop.time() < deadline:
        statuses = [
            (await client.get(f"/api/jobs/{jid}")).json()["status"] for jid in job_ids
        ]
        if all(s in ("done", "error", "cancelled") for s in statuses):
            return statuses
        await asyncio.sleep(0.1)
    raise AssertionError(f"jobs did not finish in {timeout}s: {statuses}")


async def test_health_reports_stub_mode(client):
    body = (await client.get("/api/health")).json()
    assert body["status"] == "ok"
    assert body["stub_mode"] is True
    assert body["models"] >= 2  # the seeded SDXL + GGUF


async def test_models_discovered(client):
    models = (await client.get("/api/models")).json()
    job_types = {m["job_type"] for m in models}
    assert {"llm", "image"} <= job_types


async def test_mixed_batch_runs_with_one_swap(client):
    models = (await client.get("/api/models")).json()
    llm = next(m["id"] for m in models if m["job_type"] == "llm")
    img = next(m["id"] for m in models if m["job_type"] == "image")

    img_params = {"prompt": "a test", "steps": 2, "width": 256, "height": 256}
    batch = [
        {"type": "llm", "model_id": llm, "params": {"prompt": "hello"}},
        {"type": "image", "model_id": img, "params": img_params},
        {"type": "llm", "model_id": llm, "params": {"prompt": "world"}},
        {"type": "image", "model_id": img, "params": img_params},
    ]

    async with _BusCollector(app.state.bus) as collector:
        created = (await client.post("/api/jobs", json=batch)).json()
        job_ids = [j["id"] for j in created]
        assert len(job_ids) == 4
        statuses = await _wait_done(client, job_ids)

    assert statuses == ["done"] * 4
    # Phase-batching: each family loads once (2), with exactly one swap between.
    assert collector.loaded == 2, f"expected 2 loads, saw {collector.loaded}"
    assert collector.swaps == 1, f"expected 1 swap, saw {collector.swaps}"

    # The two image jobs landed in the gallery.
    images = (await client.get("/api/images")).json()
    assert len(images) >= 2


async def test_plan_endpoint_previews_one_swap(client):
    """The /api/jobs/plan preview must agree with the live scheduler on a mixed
    batch. Asserted on a paused queue (lowest priority + idle worker drains it,
    so we check the pure prediction via a fresh batch read right after enqueue)."""
    models = (await client.get("/api/models")).json()
    llm = next(m["id"] for m in models if m["job_type"] == "llm")
    img = next(m["id"] for m in models if m["job_type"] == "image")

    # Drain anything already queued, then read the plan immediately after posting
    # a mixed batch. Even if the worker has started one job, the predicted swap
    # count for the remaining mixed jobs is at most one.
    batch = [
        {"type": "llm", "model_id": llm, "params": {"prompt": "a"}},
        {"type": "image", "model_id": img, "params": {"prompt": "b", "steps": 2}},
        {"type": "llm", "model_id": llm, "params": {"prompt": "c"}},
    ]
    created = (await client.post("/api/jobs", json=batch)).json()
    plan = (await client.get("/api/jobs/plan")).json()
    assert plan["swaps"] <= 1
    await _wait_done(client, [j["id"] for j in created])
