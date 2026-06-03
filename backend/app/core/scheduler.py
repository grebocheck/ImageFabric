"""The single GPU worker + phase-batching scheduler.

One worker owns the GPU, so GPU work is naturally serialized. The scheduler
picks the next job to *minimize model swaps*: within the highest-priority tier it
prefers a job that needs the model already resident, then one of the same type,
then the oldest. The effect on a mixed batch is the intended flow — drain all
LLM jobs, swap once, drain all image jobs.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from ..backends.base import ImageBackend, LLMBackend
from ..backends.registry import ModelRegistry
from ..db.models import Image, Job
from ..db.session import session_scope
from .arbiter import GpuArbiter
from .enums import EventType, JobStatus, JobType
from .events import Event, EventBus


@dataclass
class JobSnapshot:
    id: str
    type: JobType
    model_id: str
    params: dict[str, Any]


class Worker:
    def __init__(self, bus: EventBus, arbiter: GpuArbiter, registry: ModelRegistry) -> None:
        self._bus = bus
        self._arbiter = arbiter
        self._registry = registry
        self._wakeup = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._running = False

    # ----------------------------------------------------------- lifecycle
    def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="imgfab-worker")

    async def stop(self) -> None:
        self._running = False
        self._wakeup.set()
        if self._task:
            await self._task
        await self._arbiter.free_all()

    def notify(self) -> None:
        """Wake the worker (call after enqueue/cancel)."""
        self._wakeup.set()

    # ---------------------------------------------------------------- loop
    async def _loop(self) -> None:
        # On restart, recover jobs that were RUNNING when we died.
        await self._requeue_orphans()
        while self._running:
            snap = await self._pick_next()
            if snap is None:
                self._wakeup.clear()
                try:
                    await asyncio.wait_for(self._wakeup.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    pass
                continue
            await self._run(snap)

    async def _requeue_orphans(self) -> None:
        async with session_scope() as s:
            rows = (await s.execute(
                select(Job).where(Job.status == JobStatus.RUNNING)
            )).scalars().all()
            for job in rows:
                job.status = JobStatus.QUEUED
                job.progress = 0.0

    # --------------------------------------------------- phase-batch select
    async def _pick_next(self) -> JobSnapshot | None:
        async with session_scope() as s:
            rows = (await s.execute(
                select(Job)
                .where(Job.status == JobStatus.QUEUED)
                .order_by(Job.priority.desc(), Job.created_at.asc())
            )).scalars().all()
            if not rows:
                return None

            top_priority = rows[0].priority
            tier = [j for j in rows if j.priority == top_priority]

            cur = self._arbiter.current
            chosen = None
            if cur is not None:
                chosen = next((j for j in tier if j.model_id == cur.descriptor.id), None)
                if chosen is None:
                    chosen = next(
                        (j for j in tier if j.type == cur.descriptor.job_type), None
                    )
            if chosen is None:
                chosen = tier[0]

            chosen.status = JobStatus.RUNNING
            chosen.started_at = datetime.now(timezone.utc)
            # enum columns come back as plain strings from SQLite -> normalize so
            # identity checks (`snap.type is JobType.IMAGE`) work downstream.
            return JobSnapshot(chosen.id, JobType(chosen.type), chosen.model_id, dict(chosen.params))

    # ----------------------------------------------------------- run a job
    async def _run(self, snap: JobSnapshot) -> None:
        await self._bus.publish(Event(EventType.JOB_STARTED, job_id=snap.id, job_type=snap.type.value))
        try:
            backend = self._registry.get_backend(snap.model_id)
            await self._arbiter.ensure(backend)

            last_emit = 0.0

            async def progress(frac: float, note: str | None) -> None:
                nonlocal last_emit
                now = asyncio.get_running_loop().time()
                if now - last_emit >= 0.1 or frac >= 1.0:
                    last_emit = now
                    await self._bus.publish(Event(
                        EventType.JOB_PROGRESS, job_id=snap.id, progress=frac, note=note
                    ))

            if snap.type is JobType.IMAGE:
                assert isinstance(backend, ImageBackend)
                records = await backend.generate(self._with_lora_paths(snap.params), progress)
                await self._finish_image(snap, records)
            else:
                assert isinstance(backend, LLMBackend)

                async def on_token(tok: str) -> None:
                    await self._bus.publish(Event(EventType.LLM_TOKEN, job_id=snap.id, token=tok))

                text = await backend.complete(snap.params, on_token)
                await self._finish_llm(snap, text)

        except Exception as exc:  # noqa: BLE001
            await self._fail(snap, repr(exc))

    def _with_lora_paths(self, params: dict[str, Any]) -> dict[str, Any]:
        raw_loras = params.get("loras") or []
        if not isinstance(raw_loras, list) or not raw_loras:
            return params
        lora_paths: dict[str, str] = {}
        for item in raw_loras:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                continue
            lora = self._registry.get_lora(item["id"])
            lora_paths[lora.id] = str(lora.path)
        if not lora_paths:
            return params
        return {**params, "_lora_paths": lora_paths}

    async def _finish_image(self, snap: JobSnapshot, records: list[dict[str, Any]]) -> None:
        image_ids: list[str] = []
        async with session_scope() as s:
            for rec in records:
                img = Image(
                    job_id=snap.id,
                    path=rec["path"],
                    thumb_path=rec.get("thumb_path"),
                    seed=rec.get("seed"),
                    width=rec.get("width"),
                    height=rec.get("height"),
                    params=rec.get("params", {}),
                )
                s.add(img)
                await s.flush()
                image_ids.append(img.id)
            job = await s.get(Job, snap.id)
            if job:
                job.status = JobStatus.DONE
                job.progress = 1.0
                job.result = {"image_ids": image_ids}
                job.finished_at = datetime.now(timezone.utc)
        for iid, rec in zip(image_ids, records):
            await self._bus.publish(Event(
                EventType.IMAGE_READY, job_id=snap.id, image_id=iid,
                thumb=rec.get("thumb_path"), path=rec["path"],
            ))
        await self._bus.publish(Event(EventType.JOB_DONE, job_id=snap.id, job_type=snap.type.value))

    async def _finish_llm(self, snap: JobSnapshot, text: str) -> None:
        async with session_scope() as s:
            job = await s.get(Job, snap.id)
            if job:
                job.status = JobStatus.DONE
                job.progress = 1.0
                job.result = {"text": text}
                job.finished_at = datetime.now(timezone.utc)
        await self._bus.publish(Event(
            EventType.JOB_DONE, job_id=snap.id, job_type=snap.type.value, text=text
        ))

    async def _fail(self, snap: JobSnapshot, error: str) -> None:
        async with session_scope() as s:
            job = await s.get(Job, snap.id)
            if job:
                job.status = JobStatus.ERROR
                job.error = error
                job.finished_at = datetime.now(timezone.utc)
        await self._bus.publish(Event(EventType.JOB_ERROR, job_id=snap.id, error=error))
