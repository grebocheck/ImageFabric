"""Local voice-changer workspace (P6, w-okada Voice Changer / MMVCServerSIO).

w-okada is a realtime voice-conversion **server**, not a pip import: HFabric
detects it and builds UI on its API. P6.1 is the shell — it detects a local
install (``MMVCServerSIO.exe``), reads its ``model_dir`` slots (each a folder
with ``params.json`` + a ``.safetensors``/``.pth`` weight and an ``.index``),
and probes whether the server is reachable. Driving the realtime conversion API
is P6.2; until then ``/convert`` returns a clear error instead of faking a result.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..config import settings

router = APIRouter(prefix="/api/voice", tags=["voice"])


def _exe() -> Path:
    return settings.voice_wokada_dir / "MMVCServerSIO.exe"


def _wokada_installed() -> bool:
    return _exe().exists()


def _model_dir() -> Path:
    return settings.voice_wokada_dir / "model_dir"


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.glob("*") if f.is_file())


def _models() -> list[dict[str, Any]]:
    """Read w-okada model slots (folders that carry a params.json)."""
    root = _model_dir()
    if not root.exists():
        return []
    out: list[dict[str, Any]] = []
    for slot in sorted(root.iterdir()):
        params = slot / "params.json"
        if not slot.is_dir() or not params.exists():
            continue
        try:
            meta = json.loads(params.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        index_file = str(meta.get("indexFile") or "")
        out.append({
            "id": slot.name,
            "slot": slot.name,
            "name": str(meta.get("name") or slot.name),
            "type": str(meta.get("voiceChangerType") or "RVC"),
            "version": str(meta.get("version") or ""),
            "sampling_rate": meta.get("samplingRate"),
            "f0": bool(meta.get("f0", False)),
            "has_index": bool(index_file) and (slot / index_file).exists(),
            "size_bytes": _dir_size(slot),
        })
    return out


async def _server_reachable() -> bool:
    """True if the w-okada server answers at all (any HTTP response)."""
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            await client.get(settings.voice_wokada_url)
        return True
    except httpx.HTTPError:
        return False


@router.get("/status")
async def voice_status() -> dict:
    installed = _wokada_installed()
    reachable = await _server_reachable()
    return {
        "engine": "w-okada",
        "wokada_dir": str(settings.voice_wokada_dir),
        "wokada_installed": installed,
        "executable": str(_exe()) if installed else None,
        "model_dir": str(_model_dir()),
        "server_url": settings.voice_wokada_url,
        "server_reachable": reachable,
        "models": _models(),
        "device": settings.voice_device,
        # w-okada is realtime: it's "ready" once its server is up.
        "ready": reachable,
        "realtime": reachable,
    }


@router.post("/convert")
async def voice_convert(
    file: UploadFile = File(...),  # noqa: ARG001 - accepted now, used once the API is driven
    model_id: str = Form(...),
    pitch: int = Form(0),  # noqa: ARG001
) -> dict:
    """Placeholder. w-okada is a realtime engine; driving its conversion API is
    wired in P6.2. Refuse clearly rather than fake a result."""
    if not next((m for m in _models() if m["id"] == model_id), None):
        raise HTTPException(404, "voice model not found")
    if not await _server_reachable():
        raise HTTPException(
            503,
            f"w-okada server is not reachable at {settings.voice_wokada_url}. "
            "Start MMVCServerSIO, then retry.",
        )
    raise HTTPException(501, "voice conversion via the w-okada API is not wired yet (P6.2)")
