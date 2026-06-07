"""Backend interfaces.

A *backend* wraps exactly one model file and is a GPU *resident*: the arbiter
guarantees that at most one resident is loaded into VRAM at a time. Heavy ML
imports (torch/diffusers) live inside the concrete implementations and are
imported lazily, so this module — and the whole foundation — imports cleanly
without a GPU stack.
"""

from __future__ import annotations

import abc
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.enums import JobType, ModelFamily

# progress callback: (fraction_0_to_1, optional human note)
ProgressCb = Callable[[float, str | None], Awaitable[None]]
# token callback for streamed LLM output
TokenCb = Callable[[str], Awaitable[None]]


class GenerationCancelled(Exception):
    """Raised inside a backend to abort in-flight work after ``request_stop()``.

    The worker catches it and marks the job *cancelled* (not *errored*)."""


@dataclass(frozen=True)
class ModelDescriptor:
    id: str
    name: str
    family: ModelFamily
    path: Path
    size_bytes: int
    # quantization backend, e.g. "nunchaku" for SVDQuant int4/fp4 transformers
    quant: str | None = None

    @property
    def job_type(self) -> JobType:
        return self.family.job_type


@dataclass(frozen=True)
class LoraDescriptor:
    id: str
    name: str
    path: Path
    size_bytes: int
    family: ModelFamily | None = None


class GpuBackend(abc.ABC):
    """Common lifecycle for anything that occupies VRAM."""

    def __init__(self, descriptor: ModelDescriptor) -> None:
        self.descriptor = descriptor
        self._loaded = False
        self._warm = False
        self._load_report: dict[str, Any] | None = None

    @property
    def resident_key(self) -> str:
        return f"{self.descriptor.job_type.value}:{self.descriptor.id}"

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def warm(self) -> bool:
        return self._warm

    @property
    def can_keep_warm(self) -> bool:
        return False

    @property
    def load_report(self) -> dict[str, Any] | None:
        return self._load_report

    def request_stop(self) -> None:
        """Best-effort interrupt of in-flight work (to cancel a running job).

        No-op by default; backends that can interrupt (LLM streaming, image
        denoise step loop) override this."""

    async def after_job(self, job_id: str, params: dict[str, Any], *, failed: bool = False) -> dict[str, Any] | None:
        """Optional post-job stabilization hook.

        The arbiter keeps a model resident across same-model jobs for speed, so
        concrete backends can use this to release temporary allocations without
        unloading the resident model. Returning a dict publishes lightweight
        diagnostics on the event bus.
        """
        return None

    @abc.abstractmethod
    async def load(self) -> None: ...

    @abc.abstractmethod
    async def unload(self) -> None: ...

    async def park(self) -> bool:
        """Move out of VRAM but keep CPU state warm. Default: unsupported."""
        return False


class ImageBackend(GpuBackend):
    @abc.abstractmethod
    async def generate(
        self, params: dict[str, Any], progress: ProgressCb
    ) -> list[dict[str, Any]]:
        """Return a list of produced image records: ``{path, seed, width, height}``."""


class LLMBackend(GpuBackend):
    @abc.abstractmethod
    async def complete(
        self, params: dict[str, Any], on_token: TokenCb | None = None
    ) -> str:
        """Return the full generated text (also streamed via ``on_token``)."""
