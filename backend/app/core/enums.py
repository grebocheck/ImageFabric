"""Domain enums shared across the backend."""

from __future__ import annotations

from enum import Enum


class JobType(str, Enum):
    LLM = "llm"
    IMAGE = "image"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


class ModelFamily(str, Enum):
    FLUX = "flux"
    FLUX2 = "flux2"
    QWEN_IMAGE = "qwen-image"
    Z_IMAGE = "z-image"
    SDXL = "sdxl"
    GGUF = "gguf"
    UNKNOWN = "unknown"

    @property
    def job_type(self) -> JobType:
        return JobType.LLM if self is ModelFamily.GGUF else JobType.IMAGE


class EventType(str, Enum):
    """Event names published on the bus and streamed over the WebSocket."""

    # model lifecycle (the VRAM arbiter)
    MODEL_LOADING = "model.loading"
    MODEL_LOADED = "model.loaded"
    MODEL_UNLOADING = "model.unloading"
    MODEL_UNLOADED = "model.unloaded"
    GPU_STATUS = "gpu.status"
    MEM_STATUS = "mem.status"
    # human-readable reason the arbiter held/swapped/refused (transparency)
    ARBITER_NOTE = "arbiter.note"

    # job lifecycle
    JOB_CREATED = "job.created"
    JOB_STARTED = "job.started"
    JOB_PROGRESS = "job.progress"
    JOB_DONE = "job.done"
    JOB_ERROR = "job.error"
    JOB_CANCELLED = "job.cancelled"

    # streamed LLM tokens
    LLM_TOKEN = "llm.token"

    # produced artifact
    IMAGE_READY = "image.ready"

    # realtime voice lifecycle
    VOICE_SESSION_STARTED = "voice.session.started"
    VOICE_SESSION_STOPPED = "voice.session.stopped"
