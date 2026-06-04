"""Pydantic request/response models for the REST API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

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


# ----------------------------------------------------------------------- chat
class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    error: bool = False
    job_id: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ConversationOut(BaseModel):
    id: str
    title: str
    model_id: str | None = None
    system: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ConversationDetailOut(ConversationOut):
    messages: list[MessageOut] = Field(default_factory=list)


class ConversationCreate(BaseModel):
    title: str | None = None
    model_id: str | None = None
    system: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class ConversationUpdate(BaseModel):
    title: str | None = None
    model_id: str | None = None
    system: str | None = None
    params: dict[str, Any] | None = None


class MessageImport(BaseModel):
    role: str = Field(pattern="^(user|assistant|system)$")
    content: str = ""
    error: bool = False
    created_at: datetime | None = None


class ConversationImport(BaseModel):
    title: str | None = None
    model_id: str | None = None
    system: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    messages: list[MessageImport] = Field(default_factory=list)


class ChatImportIn(BaseModel):
    conversations: list[ConversationImport] = Field(default_factory=list)


class ChatImportOut(BaseModel):
    imported: int
    conversations: list[ConversationDetailOut] = Field(default_factory=list)


class ChatSend(BaseModel):
    content: str
    model_id: str
    system: str | None = None
    temperature: float = 0.8
    max_tokens: int = 512
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repeat_penalty: float | None = None
    seed: int | None = None
    stop: list[str] | None = None
    image_tool: bool = False
    image_model_id: str | None = None
    document_tool: bool = False
    rag_top_k: int = 5


class ChatSendOut(BaseModel):
    job_id: str
    conversation: ConversationOut
    user_message: MessageOut
    assistant_message: MessageOut


class ImageChatSend(BaseModel):
    prompt: str
    model_id: str
    negative: str | None = None
    steps: int | None = None
    width: int | None = None
    height: int | None = None
    seed: int | None = None


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


class PresetImportItem(BaseModel):
    name: str
    type: JobType
    params: dict[str, Any] = Field(default_factory=dict)


class PresetImportIn(BaseModel):
    presets: list[PresetImportItem] = Field(default_factory=list)
    on_conflict: Literal["rename", "skip"] = "rename"


class PresetImportOut(BaseModel):
    imported: int
    skipped: int = 0
    presets: list[PresetOut] = Field(default_factory=list)


# ---------------------------------------------------------------------- notes
class NoteCreate(BaseModel):
    title: str | None = None
    content: str = ""


class NoteUpdate(BaseModel):
    title: str | None = None
    content: str | None = None


class NoteOut(BaseModel):
    id: str
    title: str
    content: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
