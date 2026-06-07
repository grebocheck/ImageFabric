"""FastAPI application entrypoint — wires the foundation together.

Lifespan: init DB -> scan models -> build bus/arbiter/worker -> start worker.
Shutdown: stop worker -> free the GPU. Everything GPU-related flows through the
single Worker + GpuArbiter, so the VRAM invariant holds no matter how requests
arrive.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import (
    chat,
    code,
    gallery,
    jobs,
    llm,
    models,
    notes,
    presets,
    rag,
    transcription,
    tts,
    vision,
    voice,
    ws,
)
from .backends.registry import ModelRegistry
from .config import settings
from .core.arbiter import GpuArbiter
from .core.enums import EventType
from .core.events import Event, EventBus
from .core.scheduler import Worker
from .db.session import init_db
from .services.embedding_service import embedding_service
from .util import sysmon


async def _mem_monitor(bus: EventBus) -> None:
    """Broadcast RAM/VRAM so the UI can see pressure (never guess at it)."""
    while True:
        await bus.publish(Event(EventType.MEM_STATUS, **sysmon.snapshot()))
        await asyncio.sleep(settings.mem_poll_seconds)


async def _prime_learned_profiles() -> None:
    """Load persisted per-model memory measurements into the sysmon cache (P7.2)."""
    from .db.session import session_scope
    from .services import model_profile_service as mps

    try:
        async with session_scope() as s:
            rows = await mps.load_all(s)
        sysmon.prime_learned_profiles([
            {"model_id": r.model_id, "ram_gb": r.ram_gb, "vram_gb": r.vram_gb} for r in rows
        ])
    except Exception:  # noqa: BLE001 - missing profiles must not block startup
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await _prime_learned_profiles()

    registry = ModelRegistry()
    registry.scan()
    bus = EventBus()
    arbiter = GpuArbiter(bus)
    worker = Worker(bus, arbiter, registry)

    app.state.registry = registry
    app.state.bus = bus
    app.state.arbiter = arbiter
    app.state.worker = worker

    worker.start()
    mem_task = asyncio.create_task(_mem_monitor(bus), name="hfabric-mem-monitor")
    try:
        yield
    finally:
        mem_task.cancel()
        await embedding_service.stop()
        voice.stop_server()
        await worker.stop()


app = FastAPI(title="HFabric", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(models.router)
app.include_router(jobs.router)
app.include_router(llm.router)
app.include_router(chat.router)
app.include_router(code.router)
app.include_router(gallery.router)
app.include_router(notes.router)
app.include_router(presets.router)
app.include_router(rag.router)
app.include_router(transcription.router)
app.include_router(tts.router)
app.include_router(vision.router)
app.include_router(voice.router)
app.include_router(ws.router)


@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "stub_mode": settings.stub_mode,
        "models": len(app.state.registry.descriptors()),
        "gpu": app.state.arbiter.status(),
        "mem": sysmon.snapshot(),
    }
