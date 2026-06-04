"""Local transcription workspace endpoints.

This API is intentionally model-gated: it only accepts local Whisper models
under models/transcribe and never asks Whisper libraries to download weights.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..config import settings

router = APIRouter(prefix="/api/transcription", tags=["transcription"])

ALLOWED_EXTS = {
    ".aac",
    ".flac",
    ".m4a",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".oga",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
}


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def _id(path: Path) -> str:
    value = path.stem if path.is_file() else path.name
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or uuid.uuid4().hex


def _models() -> list[dict[str, Any]]:
    root = settings.transcription_models_dir
    if not root.exists():
        return []

    out: list[dict[str, Any]] = []
    for path in sorted(root.iterdir()):
        if path.name.startswith("."):
            continue
        if path.is_dir():
            markers = {"config.json", "model.bin", "tokenizer.json", "vocabulary.json"}
            if not any((path / marker).exists() for marker in markers):
                continue
            out.append({
                "id": _id(path),
                "name": path.name,
                "path": str(path),
                "size_bytes": _size(path),
                "engine": "faster-whisper",
            })
        elif path.suffix.lower() in {".pt", ".pth"}:
            out.append({
                "id": _id(path),
                "name": path.stem,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "engine": "openai-whisper",
            })
    return out


def _model_map() -> dict[str, dict[str, Any]]:
    return {m["id"]: m for m in _models()}


def _day_dir() -> Path:
    d = settings.outputs_dir / datetime.now(timezone.utc).strftime("%Y-%m-%d")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_ext(filename: str | None) -> str:
    ext = Path(filename or "").suffix.lower()
    return ext if ext in ALLOWED_EXTS else ".wav"


def _metadata_path(transcription_id: str) -> Path | None:
    if not re.fullmatch(r"[a-f0-9]{32}", transcription_id):
        return None
    for path in settings.outputs_dir.glob(f"*/transcription-{transcription_id}.json"):
        return path
    return None


@router.get("/status")
async def transcription_status() -> dict:
    engines = {
        "faster-whisper": _has("faster_whisper"),
        "openai-whisper": _has("whisper"),
    }
    models = _models()
    return {
        "models_dir": str(settings.transcription_models_dir),
        "models": models,
        "engines": engines,
        "device": settings.transcription_device,
        "compute_type": settings.transcription_compute_type,
        "max_upload_mb": settings.transcription_max_upload_mb,
        "ready": any(engines.values()) and bool(models),
    }


@router.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    model_id: str = Form(...),
    language: str | None = Form(None),
    task: str = Form("transcribe"),
    initial_prompt: str | None = Form(None),
) -> dict:
    if task not in {"transcribe", "translate"}:
        raise HTTPException(422, "task must be transcribe or translate")

    model = _model_map().get(model_id)
    if not model:
        raise HTTPException(404, "transcription model not found")

    engine = model["engine"]
    module_name = "faster_whisper" if engine == "faster-whisper" else "whisper"
    if not _has(module_name):
        raise HTTPException(503, f"{engine} is not installed")

    max_bytes = settings.transcription_max_upload_mb * 1024 * 1024
    payload = await file.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise HTTPException(413, f"audio upload exceeds {settings.transcription_max_upload_mb} MB")
    if not payload:
        raise HTTPException(422, "audio file is empty")

    transcription_id = uuid.uuid4().hex
    audio_path = _day_dir() / f"transcription-{transcription_id}{_safe_ext(file.filename)}"
    audio_path.write_bytes(payload)

    started = time.monotonic()
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                _run_transcription,
                engine,
                model,
                audio_path,
                (language or "").strip() or None,
                task,
                (initial_prompt or "").strip() or None,
            ),
            timeout=settings.transcription_timeout_seconds,
        )
    except TimeoutError as exc:
        raise HTTPException(504, "transcription timed out") from exc
    except Exception as exc:  # noqa: BLE001 - surface local model/runtime failures.
        raise HTTPException(500, f"transcription failed: {exc}") from exc

    duration = time.monotonic() - started
    metadata = {
        "id": transcription_id,
        "type": "transcription",
        "engine": engine,
        "model_id": model_id,
        "model_path": model["path"],
        "language": language,
        "task": task,
        "duration_seconds": duration,
        "audio_path": str(audio_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        **result,
    }
    meta_path = audio_path.with_name(f"transcription-{transcription_id}.json")
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "id": transcription_id,
        "metadata_url": f"/api/transcription/result/{transcription_id}/metadata",
        "metadata_path": str(meta_path),
        "duration_seconds": duration,
        **result,
    }


@router.get("/result/{transcription_id}/metadata")
async def transcription_metadata(transcription_id: str):
    from fastapi.responses import FileResponse

    path = _metadata_path(transcription_id)
    if not path or not path.exists():
        raise HTTPException(404, "transcription metadata not found")
    return FileResponse(path, media_type="application/json")


def _run_transcription(
    engine: str,
    model: dict[str, Any],
    audio_path: Path,
    language: str | None,
    task: str,
    initial_prompt: str | None,
) -> dict[str, Any]:
    if engine == "faster-whisper":
        faster_whisper = importlib.import_module("faster_whisper")
        whisper_model = faster_whisper.WhisperModel(
            model["path"],
            device=settings.transcription_device,
            compute_type=settings.transcription_compute_type,
            local_files_only=True,
        )
        segments_iter, info = whisper_model.transcribe(
            str(audio_path),
            language=language,
            task=task,
            initial_prompt=initial_prompt,
        )
        segments = [
            {"start": float(s.start), "end": float(s.end), "text": s.text.strip()}
            for s in segments_iter
        ]
        return {
            "text": " ".join(s["text"] for s in segments).strip(),
            "segments": segments,
            "detected_language": getattr(info, "language", None),
            "language_probability": getattr(info, "language_probability", None),
        }

    whisper = importlib.import_module("whisper")
    whisper_model = whisper.load_model(
        model["path"],
        device=settings.transcription_device,
        download_root=str(settings.transcription_models_dir),
    )
    result = whisper_model.transcribe(
        str(audio_path),
        language=language,
        task=task,
        initial_prompt=initial_prompt,
        fp16=settings.transcription_device != "cpu",
    )
    segments = [
        {
            "start": float(s.get("start", 0.0)),
            "end": float(s.get("end", 0.0)),
            "text": str(s.get("text", "")).strip(),
        }
        for s in result.get("segments", [])
    ]
    return {
        "text": str(result.get("text", "")).strip(),
        "segments": segments,
        "detected_language": result.get("language"),
        "language_probability": None,
    }
