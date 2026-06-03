"""Queue endpoints: create / list / inspect / cancel / reprioritize jobs."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from ..backends.base import ModelDescriptor
from ..backends.registry import ModelRegistry
from ..core.enums import EventType, JobStatus, JobType
from ..core.events import EventBus
from ..core.scheduler import Worker
from pydantic import BaseModel

from ..schemas import JobCreate, JobOut, PriorityUpdate
from ..services import prompt_service, queue_service
from .deps import get_bus, get_registry, get_session, get_worker

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


def _validate_model(registry: ModelRegistry, payload: JobCreate) -> ModelDescriptor:
    try:
        desc = registry.get_descriptor(payload.model_id)
    except KeyError:
        raise HTTPException(404, f"unknown model_id: {payload.model_id}")
    if desc.job_type != payload.type:
        raise HTTPException(
            400, f"model '{desc.id}' is {desc.job_type.value}, not {payload.type.value}"
        )
    return desc


def _normalize_loras(registry: ModelRegistry, desc: ModelDescriptor, payload: JobCreate) -> None:
    raw = payload.params.get("loras")
    if payload.type is not JobType.IMAGE or not raw:
        return
    if not isinstance(raw, list):
        raise HTTPException(400, "params.loras must be a list")

    public_loras: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            lora_id = item
            weight = 1.0
        elif isinstance(item, dict):
            lora_id = item.get("id")
            weight = item.get("weight", 1.0)
        else:
            raise HTTPException(400, "each LoRA must be an id string or object")
        if not isinstance(lora_id, str) or not lora_id:
            raise HTTPException(400, "each LoRA needs an id")
        try:
            weight = float(weight)
        except (TypeError, ValueError):
            raise HTTPException(400, f"invalid LoRA weight for {lora_id!r}")
        if weight < -2.0 or weight > 2.0:
            raise HTTPException(400, f"LoRA weight for {lora_id!r} must be between -2 and 2")
        try:
            lora = registry.get_lora(lora_id)
        except KeyError:
            raise HTTPException(404, f"unknown LoRA id: {lora_id}")
        if lora.family is not None and desc.family is not lora.family:
            raise HTTPException(
                400,
                f"LoRA '{lora.name}' targets {lora.family.value}, "
                f"but model '{desc.name}' is {desc.family.value}",
            )
        public_loras.append({
            "id": lora.id,
            "name": lora.name,
            "family": lora.family.value if lora.family else None,
            "weight": weight,
        })

    params = dict(payload.params)
    params["loras"] = public_loras
    payload.params = params


@router.post("", response_model=list[JobOut])
async def create_jobs(
    payloads: list[JobCreate],
    session: AsyncSession = Depends(get_session),
    registry: ModelRegistry = Depends(get_registry),
    bus: EventBus = Depends(get_bus),
    worker: Worker = Depends(get_worker),
) -> list[JobOut]:
    """Accept a *batch* of jobs in one call (the core 'throw a batch in' flow)."""
    if not payloads:
        raise HTTPException(400, "no jobs provided")
    created = []
    for payload in payloads:
        desc = _validate_model(registry, payload)
        _normalize_loras(registry, desc, payload)
        job = await queue_service.create_job(session, payload)
        created.append(job)
    await session.commit()
    for job in created:
        bus.emit(EventType.JOB_CREATED, job_id=job.id, job_type=job.type.value)
    worker.notify()
    return [JobOut.model_validate(j) for j in created]


class ExpandRequest(BaseModel):
    idea: str
    model_id: str
    style: str | None = None
    priority: int = 0


@router.post("/expand", response_model=JobOut)
async def expand_idea(
    body: ExpandRequest,
    session: AsyncSession = Depends(get_session),
    registry: ModelRegistry = Depends(get_registry),
    bus: EventBus = Depends(get_bus),
    worker: Worker = Depends(get_worker),
) -> JobOut:
    """Queue an LLM job that expands a short idea into a rich image prompt.
    Tokens stream over the WebSocket as ``llm.token`` events for this job id."""
    payload = JobCreate(
        type=JobType.LLM,
        model_id=body.model_id,
        params=prompt_service.build_expansion_params(body.idea, style=body.style),
        priority=body.priority,
    )
    _validate_model(registry, payload)
    job = await queue_service.create_job(session, payload)
    await session.commit()
    bus.emit(EventType.JOB_CREATED, job_id=job.id, job_type=job.type.value)
    worker.notify()
    return JobOut.model_validate(job)


@router.get("", response_model=list[JobOut])
async def list_jobs(
    status: JobStatus | None = None,
    type: JobType | None = None,
    limit: int = Query(200, le=1000),
    session: AsyncSession = Depends(get_session),
) -> list[JobOut]:
    jobs = await queue_service.list_jobs(session, status=status, type=type, limit=limit)
    return [JobOut.model_validate(j) for j in jobs]


@router.get("/{job_id}", response_model=JobOut)
async def get_job(job_id: str, session: AsyncSession = Depends(get_session)) -> JobOut:
    job = await queue_service.get_job(session, job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return JobOut.model_validate(job)


@router.delete("/{job_id}", response_model=JobOut)
async def cancel_job(
    job_id: str,
    session: AsyncSession = Depends(get_session),
    bus: EventBus = Depends(get_bus),
) -> JobOut:
    job = await queue_service.cancel_job(session, job_id)
    if not job:
        raise HTTPException(404, "job not found")
    await session.commit()
    bus.emit(EventType.JOB_CANCELLED, job_id=job.id)
    return JobOut.model_validate(job)


@router.post("/{job_id}/priority", response_model=JobOut)
async def set_priority(
    job_id: str,
    body: PriorityUpdate,
    session: AsyncSession = Depends(get_session),
    worker: Worker = Depends(get_worker),
) -> JobOut:
    job = await queue_service.set_priority(session, job_id, body.priority)
    if not job:
        raise HTTPException(404, "job not found")
    await session.commit()
    worker.notify()
    return JobOut.model_validate(job)


@router.post("/clear")
async def clear_finished(session: AsyncSession = Depends(get_session)) -> dict[str, int]:
    n = await queue_service.clear_finished(session)
    await session.commit()
    return {"removed": n}
