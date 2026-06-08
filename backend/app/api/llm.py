"""LLM runtime knobs that need a server (re)launch to take effect.

``llama-server`` is started with a fixed context size (``-c``) and GPU-offload
layer count (``-ngl``); changing those means relaunching the process. These
endpoints mutate the shared settings and, if an LLM is currently resident, free
it so the next chat reloads with the new values. Per-message knobs (temperature,
max_tokens) are NOT here — they travel with each ``/api/jobs/chat`` request.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..config import (
    CONTEXT_TYPES,
    DEFAULT_CONTEXT_TYPE,
    LLAMA_BACKENDS,
    settings,
)
from ..core.arbiter import GpuArbiter
from ..core.enums import ModelFamily
from .deps import get_arbiter

router = APIRouter(prefix="/api/llm", tags=["llm"])

CTX_MIN, CTX_MAX = 512, 131072
NGL_MIN, NGL_MAX = 0, 999


class LlmConfigUpdate(BaseModel):
    ctx: int | None = None
    ngl: int | None = None
    backend: str | None = None
    context_type: str | None = None


def _llm_resident(arbiter: GpuArbiter) -> bool:
    cur = arbiter.current
    return bool(cur and cur.descriptor.family is ModelFamily.GGUF)


def _backends_status() -> list[dict]:
    out = []
    for bid, spec in LLAMA_BACKENDS.items():
        bin_path = getattr(settings, spec["bin_attr"])
        out.append({
            "id": bid,
            "label": spec["label"],
            "available": bin_path.exists(),
            "path": str(bin_path),
            "context_types": list(spec["context_types"]),
        })
    return out


def _status(arbiter: GpuArbiter, **extra) -> dict:
    cur = arbiter.current
    loaded = _llm_resident(arbiter)
    return {
        "ctx": settings.llama_ctx,
        "ngl": settings.llama_ngl,
        "backend": settings.llama_backend,
        "backends": _backends_status(),
        "context_type": settings.llama_context_type,
        "context_types": [
            {"id": key, "label": spec["label"], "experimental": spec["experimental"]}
            for key, spec in CONTEXT_TYPES.items()
        ],
        "stub": settings.stub_mode,
        "loaded": loaded,
        "model_id": cur.descriptor.id if loaded else None,
        "defaults": {"temperature": 0.8, "max_tokens": 512},
        **extra,
    }


@router.post("/stop")
async def stop_generation(arbiter: GpuArbiter = Depends(get_arbiter)) -> dict:
    """Interrupt the LLM that is currently streaming (best-effort)."""
    cur = arbiter.current
    if cur and cur.descriptor.family is ModelFamily.GGUF and hasattr(cur, "request_stop"):
        cur.request_stop()
        return {"stopped": True}
    return {"stopped": False}


@router.get("/config")
async def get_config(arbiter: GpuArbiter = Depends(get_arbiter)) -> dict:
    return _status(arbiter)


@router.post("/config")
async def set_config(
    body: LlmConfigUpdate, arbiter: GpuArbiter = Depends(get_arbiter)
) -> dict:
    # Compute the target backend / context-type first and validate the *pair*
    # before committing anything, so we never leave settings in a state the
    # selected llama build can't actually launch.
    target_backend = body.backend if body.backend is not None else settings.llama_backend
    target_ct = body.context_type if body.context_type is not None else settings.llama_context_type

    if body.backend is not None and body.backend not in LLAMA_BACKENDS:
        raise HTTPException(
            status_code=422,
            detail=f"unknown backend {body.backend!r}; choose one of {sorted(LLAMA_BACKENDS)}",
        )
    if body.context_type is not None and body.context_type not in CONTEXT_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"unknown context_type {body.context_type!r}; choose one of {sorted(CONTEXT_TYPES)}",
        )

    note = None
    supported = LLAMA_BACKENDS[target_backend]["context_types"]
    if target_ct not in supported:
        if body.context_type is not None:
            # An explicit type the (resulting) backend can't run -> reject and
            # tell the caller how to make it valid.
            raise HTTPException(
                status_code=422,
                detail=f"context type {target_ct!r} is not supported by the "
                f"{target_backend!r} backend; supported: {list(supported)}. "
                f"Switch to a backend that lists it (e.g. 'turbo' for turbo3/turbo4).",
            )
        # A backend switch left the existing type unsupported -> gracefully fall
        # back to the always-valid default instead of erroring.
        note = (
            f"context type reset to '{DEFAULT_CONTEXT_TYPE}' — "
            f"not supported by the '{target_backend}' backend"
        )
        target_ct = DEFAULT_CONTEXT_TYPE

    # Commit.
    changed = False
    if body.ctx is not None:
        ctx = max(CTX_MIN, min(CTX_MAX, body.ctx))
        if ctx != settings.llama_ctx:
            settings.llama_ctx = ctx
            changed = True
    if body.ngl is not None:
        ngl = max(NGL_MIN, min(NGL_MAX, body.ngl))
        if ngl != settings.llama_ngl:
            settings.llama_ngl = ngl
            changed = True
    if target_backend != settings.llama_backend:
        settings.llama_backend = target_backend
        changed = True
    if target_ct != settings.llama_context_type:
        settings.llama_context_type = target_ct
        changed = True

    # New launch knobs only bite on the next server start, so drop the running
    # LLM (the next chat reloads it). Image models are untouched unless the LLM
    # is the current resident.
    reloaded = False
    if changed and _llm_resident(arbiter):
        await arbiter.free_all()
        reloaded = True

    return _status(arbiter, changed=changed, reloaded=reloaded, note=note)
