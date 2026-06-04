"""Local vision workspace endpoints.

Uses llama-mtmd-cli with local GGUF + mmproj files only. No -hf/default flags are
used, so analysis cannot hide model downloads.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from ..config import settings

router = APIRouter(prefix="/api/vision", tags=["vision"])

ALLOWED_EXTS = {".jpeg", ".jpg", ".png"}


def _model_id(path: Path) -> str:
    return re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")


def _vision_files() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = settings.vision_models_dir
    if not root.exists():
        return [], []
    models: list[dict[str, Any]] = []
    projectors: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.gguf")):
        item = {
            "id": _model_id(path),
            "name": path.stem,
            "path": str(path),
            "size_bytes": path.stat().st_size,
        }
        if path.name.lower().startswith("mmproj"):
            projectors.append(item)
        else:
            models.append(item)
    return models, projectors


def _maps() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    models, projectors = _vision_files()
    return {m["id"]: m for m in models}, {p["id"]: p for p in projectors}


def _day_dir() -> Path:
    d = settings.outputs_dir / datetime.now(timezone.utc).strftime("%Y-%m-%d")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_ext(filename: str | None) -> str:
    ext = Path(filename or "").suffix.lower()
    return ext if ext in ALLOWED_EXTS else ".png"


def _metadata_path(result_id: str) -> Path | None:
    if not re.fullmatch(r"[a-f0-9]{32}", result_id):
        return None
    for path in settings.outputs_dir.glob(f"*/vision-{result_id}.json"):
        return path
    return None


@router.get("/status")
async def vision_status() -> dict:
    models, projectors = _vision_files()
    return {
        "binary": str(settings.llama_mtmd_bin),
        "binary_exists": settings.llama_mtmd_bin.exists(),
        "models_dir": str(settings.vision_models_dir),
        "models": models,
        "projectors": projectors,
        "ready": settings.llama_mtmd_bin.exists() and bool(models) and bool(projectors),
        "gpu_layers": settings.vision_gpu_layers,
        "max_upload_mb": settings.vision_max_upload_mb,
    }


@router.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    prompt: str = Form("Describe the image."),
    model_id: str = Form(...),
    projector_id: str = Form(...),
) -> dict:
    if not settings.llama_mtmd_bin.exists():
        raise HTTPException(503, "llama-mtmd-cli binary not found")
    clean_prompt = prompt.strip()
    if not clean_prompt:
        raise HTTPException(422, "prompt is empty")

    models, projectors = _maps()
    model = models.get(model_id)
    projector = projectors.get(projector_id)
    if not model:
        raise HTTPException(404, "vision model not found")
    if not projector:
        raise HTTPException(404, "vision projector not found")

    max_bytes = settings.vision_max_upload_mb * 1024 * 1024
    payload = await file.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise HTTPException(413, f"image upload exceeds {settings.vision_max_upload_mb} MB")
    if not payload:
        raise HTTPException(422, "image file is empty")

    result_id = uuid.uuid4().hex
    image_path = _day_dir() / f"vision-{result_id}{_safe_ext(file.filename)}"
    image_path.write_bytes(payload)

    command = [
        str(settings.llama_mtmd_bin),
        "-m",
        str(model["path"]),
        "--mmproj",
        str(projector["path"]),
        "--image",
        str(image_path),
        "-p",
        clean_prompt,
        "-ngl",
        str(settings.vision_gpu_layers),
        "--no-warmup",
    ]
    if settings.vision_gpu_layers <= 0:
        command.append("--no-mmproj-offload")

    started = time.monotonic()
    try:
        proc = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(settings.llama_mtmd_bin.parent),
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=settings.vision_timeout_seconds
        )
    except TimeoutError as exc:
        if "proc" in locals():
            proc.kill()
            await proc.communicate()
        raise HTTPException(504, "vision analysis timed out") from exc
    except OSError as exc:
        raise HTTPException(500, f"could not start llama-mtmd-cli: {exc}") from exc

    duration = time.monotonic() - started
    stdout_text = stdout.decode("utf-8", errors="replace").strip()
    stderr_text = stderr.decode("utf-8", errors="replace").strip()
    if proc.returncode != 0:
        detail = stderr_text or stdout_text or f"exit code {proc.returncode}"
        raise HTTPException(500, detail[-2000:])

    text = _clean_stdout(stdout_text)
    metadata = {
        "id": result_id,
        "type": "vision",
        "prompt": clean_prompt,
        "text": text,
        "stdout": stdout_text[-8000:],
        "stderr": stderr_text[-8000:],
        "model_id": model_id,
        "model_path": model["path"],
        "projector_id": projector_id,
        "projector_path": projector["path"],
        "vision_gpu_layers": settings.vision_gpu_layers,
        "duration_seconds": duration,
        "image_path": str(image_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = image_path.with_name(f"vision-{result_id}.json")
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "id": result_id,
        "text": text,
        "metadata_url": f"/api/vision/result/{result_id}/metadata",
        "metadata_path": str(meta_path),
        "duration_seconds": duration,
    }


@router.get("/result/{result_id}/metadata")
async def vision_metadata(result_id: str) -> FileResponse:
    path = _metadata_path(result_id)
    if not path or not path.exists():
        raise HTTPException(404, "vision metadata not found")
    return FileResponse(path, media_type="application/json")


def _clean_stdout(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    noisy_prefixes = (
        "system_info:",
        "sampler ",
        "llama_",
        "ggml_",
        "common_",
        "build:",
        "main:",
    )
    clean = [line for line in lines if not line.lower().startswith(noisy_prefixes)]
    return "\n".join(clean or lines).strip()
