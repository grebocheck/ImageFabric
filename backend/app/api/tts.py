"""TTS workspace endpoints.

Only local GGUF files under models/tts are accepted. The API intentionally does
not use llama.cpp's download flags, so a run cannot hide network/model placement
side effects.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import time
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..config import settings

router = APIRouter(prefix="/api/tts", tags=["tts"])


class TtsGenerateIn(BaseModel):
    model_id: str
    text: str = Field(min_length=1, max_length=4000)
    vocoder_id: str | None = None
    use_guide_tokens: bool = False


def _models() -> list[dict]:
    root = settings.tts_models_dir
    if not root.exists():
        return []
    out = []
    for path in sorted(root.glob("*.gguf")):
        out.append({
            "id": path.stem.lower().replace(" ", "-"),
            "name": path.stem,
            "path": str(path),
            "size_bytes": path.stat().st_size,
        })
    return out


def _model_map() -> dict[str, dict]:
    return {m["id"]: m for m in _models()}


def _day_dir() -> Path:
    d = settings.outputs_dir / datetime.now(UTC).strftime("%Y-%m-%d")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _audio_path(audio_id: str) -> Path | None:
    if not re.fullmatch(r"[a-f0-9]{32}", audio_id):
        return None
    for path in settings.outputs_dir.glob(f"*/tts-{audio_id}.wav"):
        return path
    return None


@router.get("/status")
async def tts_status() -> dict:
    models = _models()
    return {
        "binary": str(settings.llama_tts_bin),
        "binary_exists": settings.llama_tts_bin.exists(),
        "models_dir": str(settings.tts_models_dir),
        "models": models,
        "ready": settings.llama_tts_bin.exists() and bool(models),
    }


@router.post("/generate")
async def generate_tts(body: TtsGenerateIn) -> dict:
    if not settings.llama_tts_bin.exists():
        raise HTTPException(503, "llama-tts binary not found")

    text = body.text.strip()
    if not text:
        raise HTTPException(422, "text is empty")

    models = _model_map()
    model = models.get(body.model_id)
    if not model:
        raise HTTPException(404, "TTS model not found")

    vocoder = None
    if body.vocoder_id:
        vocoder = models.get(body.vocoder_id)
        if not vocoder:
            raise HTTPException(404, "TTS vocoder model not found")

    audio_id = uuid.uuid4().hex
    out_path = _day_dir() / f"tts-{audio_id}.wav"
    command = [
        str(settings.llama_tts_bin),
        "-m",
        str(model["path"]),
        "-p",
        text,
        "-o",
        str(out_path),
        "-ngl",
        str(settings.tts_gpu_layers),
    ]
    if vocoder:
        command.extend(["-mv", str(vocoder["path"])])
    if body.use_guide_tokens:
        command.append("--tts-use-guide-tokens")

    started = time.monotonic()
    try:
        proc = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=settings.tts_timeout_seconds
        )
    except TimeoutError as exc:
        if "proc" in locals():
            proc.kill()
            await proc.communicate()
        raise HTTPException(504, "TTS generation timed out") from exc
    except OSError as exc:
        raise HTTPException(500, f"could not start llama-tts: {exc}") from exc

    duration = time.monotonic() - started
    stdout_text = stdout.decode("utf-8", errors="replace")
    stderr_text = stderr.decode("utf-8", errors="replace")
    if proc.returncode != 0:
        out_path.unlink(missing_ok=True)
        detail = stderr_text.strip() or stdout_text.strip() or f"exit code {proc.returncode}"
        raise HTTPException(500, detail[-2000:])
    if not out_path.exists():
        raise HTTPException(500, "llama-tts finished without producing audio")

    metadata = {
        "id": audio_id,
        "type": "tts",
        "text": text,
        "model_id": body.model_id,
        "model_path": model["path"],
        "vocoder_id": body.vocoder_id,
        "vocoder_path": vocoder["path"] if vocoder else None,
        "use_guide_tokens": body.use_guide_tokens,
        "tts_gpu_layers": settings.tts_gpu_layers,
        "duration_seconds": duration,
        "output_path": str(out_path),
        "created_at": datetime.now(UTC).isoformat(),
    }
    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "id": audio_id,
        "url": f"/api/tts/audio/{audio_id}/file",
        "path": str(out_path),
        "metadata_path": str(meta_path),
        "model_id": body.model_id,
        "vocoder_id": body.vocoder_id,
        "duration_seconds": duration,
    }


@router.get("/audio/{audio_id}/file")
async def tts_audio(audio_id: str) -> FileResponse:
    path = _audio_path(audio_id)
    if not path or not path.exists():
        raise HTTPException(404, "audio not found")
    return FileResponse(path, media_type="audio/wav")
