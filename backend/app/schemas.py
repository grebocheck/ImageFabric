"""Pydantic request/response models for the REST API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from .core.enums import JobStatus, JobType, ModelFamily


# --------------------------------------------------------------------- models
class ModelOut(BaseModel):
    id: str
    name: str
    family: ModelFamily
    job_type: JobType
    size_bytes: int
    loaded: bool
    warm: bool = False
    quant: str | None = None
    estimated_vram_gb: float | None = None
    # True for models that are slow / memory-heavy on 16 GB (raw fp8 FLUX) so the
    # UI can warn before a click triggers a long, VRAM-overflowing run.
    slow: bool = False


class GpuStatusOut(BaseModel):
    resident: str | None = None
    model_id: str | None = None
    model: str | None = None
    family: str | None = None
    warm: list[dict[str, str]] = Field(default_factory=list)


class LoraOut(BaseModel):
    id: str
    name: str
    family: ModelFamily | None = None
    size_bytes: int


# ----------------------------------------------------------------------- jobs
class JobCreate(BaseModel):
    type: JobType
    model_id: str
    params: dict[str, Any] = Field(default_factory=dict)
    priority: int = 0


class JobOut(BaseModel):
    id: str
    type: JobType
    status: JobStatus
    priority: int
    model_id: str
    params: dict[str, Any]
    progress: float
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None

    model_config = {"from_attributes": True}


class PriorityUpdate(BaseModel):
    priority: int


# --------------------------------------------------------------------- images
class ImageOut(BaseModel):
    id: str
    job_id: str
    seed: int | None = None
    width: int | None = None
    height: int | None = None
    params: dict[str, Any]
    created_at: datetime
    url: str
    thumb_url: str | None = None

    model_config = {"from_attributes": True}


# -------------------------------------------------------------------- presets
class PresetCreate(BaseModel):
    name: str
    type: JobType
    params: dict[str, Any] = Field(default_factory=dict)


class PresetOut(BaseModel):
    id: str
    name: str
    type: JobType
    params: dict[str, Any]
    created_at: datetime

    model_config = {"from_attributes": True}
