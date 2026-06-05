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
import subprocess
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from ..config import settings
from ..core.arbiter import GpuArbiter
from ..core.scheduler import Worker
from .deps import get_arbiter, get_worker

router = APIRouter(prefix="/api/voice", tags=["voice"])

# Handle to the w-okada server we launched (None if we didn't start it).
_proc: subprocess.Popen | None = None
_session_active = False

F0_DETECTORS = {
    "rmvpe_onnx",
    "rmvpe",
    "crepe_onnx_tiny",
    "crepe_onnx_full",
    "crepe_tiny",
    "crepe_full",
    "fcpe",
    "fcpe_onnx",
}


class VoiceSettingsUpdate(BaseModel):
    model_id: str | None = None
    pitch: int | None = None
    index_ratio: float | None = None
    protect: float | None = None
    f0_detector: str | None = None


def _server_running() -> bool:
    return _proc is not None and _proc.poll() is None


def stop_server() -> bool:
    """Terminate the w-okada server (process tree) if we started it."""
    global _proc, _session_active
    if _proc is None:
        return False
    try:
        subprocess.run(
            ["taskkill", "/PID", str(_proc.pid), "/T", "/F"],
            capture_output=True,
            check=False,
        )
    finally:
        _proc = None
        _session_active = False
    return True


def _exe() -> Path:
    return settings.voice_wokada_dir / "MMVCServerSIO.exe"


def _stored_setting_file() -> Path:
    return settings.voice_wokada_dir / "stored_setting.json"


def _wokada_installed() -> bool:
    return _exe().exists()


def _model_dir() -> Path:
    return settings.voice_wokada_dir / "model_dir"


def _stored_settings() -> dict[str, Any]:
    path = _stored_setting_file()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


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


def _api_url(path: str) -> str:
    return f"{settings.voice_wokada_url.rstrip('/')}/{path.lstrip('/')}"


async def _server_reachable() -> bool:
    """True if the w-okada server answers at all (any HTTP response)."""
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            await client.get(settings.voice_wokada_url)
        return True
    except httpx.HTTPError:
        return False


async def _wokada_get(path: str, *, timeout: float = 1.5) -> dict[str, Any] | None:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            res = await client.get(_api_url(path))
            res.raise_for_status()
            data = res.json()
    except (httpx.HTTPError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


async def _wokada_update(key: str, value: Any) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            res = await client.post(_api_url("/update_settings"), data={"key": key, "val": str(value)})
            res.raise_for_status()
            data = res.json()
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        raise HTTPException(502, f"w-okada update failed for {key}: {exc}") from exc
    return data if isinstance(data, dict) else {}


def _settings_subset(raw: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "modelSlotIndex",
        "enableServerAudio",
        "serverAudioStated",
        "tran",
        "formantShift",
        "indexRatio",
        "protect",
        "f0Detector",
        "serverReadChunkSize",
        "crossFadeOverlapSize",
        "extraConvertSize",
        "serverInputDeviceId",
        "serverOutputDeviceId",
        "serverMonitorDeviceId",
        "serverInputAudioGain",
        "serverOutputAudioGain",
        "serverMonitorAudioGain",
        "inputSampleRate",
        "outputSampleRate",
    )
    return {k: raw.get(k) for k in keys if k in raw}


def _bool_setting(value: Any) -> bool:
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        return bool(value)


def _model_slot(model_id: str | None) -> int | None:
    if not model_id:
        return None
    model = next((m for m in _models() if m["id"] == model_id), None)
    if model is None:
        raise HTTPException(404, "voice model not found")
    try:
        return int(model["slot"])
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, f"invalid w-okada model slot: {model['slot']}") from exc


def _clamp_pitch(value: int) -> int:
    return max(-24, min(24, int(value)))


def _clamp_ratio(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _validate_f0_detector(value: str) -> str:
    if value not in F0_DETECTORS:
        raise HTTPException(400, f"unsupported f0 detector: {value}")
    return value


async def _apply_voice_settings(body: VoiceSettingsUpdate) -> None:
    slot = _model_slot(body.model_id)
    if slot is not None:
        await _wokada_update("modelSlotIndex", slot)
    if body.pitch is not None:
        await _wokada_update("tran", _clamp_pitch(body.pitch))
    if body.index_ratio is not None:
        await _wokada_update("indexRatio", _clamp_ratio(body.index_ratio))
    if body.protect is not None:
        await _wokada_update("protect", _clamp_ratio(body.protect))
    if body.f0_detector is not None:
        await _wokada_update("f0Detector", _validate_f0_detector(body.f0_detector))


async def _status_payload() -> dict:
    installed = _wokada_installed()
    info = await _wokada_get("/info") if installed else None
    reachable = info is not None or await _server_reachable()
    performance = await _wokada_get("/performance", timeout=0.8) if reachable else None
    settings_raw = info or _stored_settings()
    server_audio_enabled = _bool_setting(settings_raw.get("enableServerAudio"))
    server_audio_started = _bool_setting(settings_raw.get("serverAudioStated"))
    selected_slot = settings_raw.get("modelSlotIndex")
    return {
        "engine": "w-okada",
        "wokada_dir": str(settings.voice_wokada_dir),
        "wokada_installed": installed,
        "executable": str(_exe()) if installed else None,
        "stored_setting": str(_stored_setting_file()),
        "model_dir": str(_model_dir()),
        "server_url": settings.voice_wokada_url,
        "server_reachable": reachable,
        "server_running": _server_running(),  # launched by us this session
        "server_audio_enabled": server_audio_enabled,
        "server_audio_started": server_audio_started,
        "selected_model_slot": str(selected_slot) if selected_slot is not None else None,
        "models": _models(),
        "device": settings.voice_device,
        "settings": _settings_subset(settings_raw),
        "performance": performance,
        "voice_lane_active": _session_active or (reachable and server_audio_enabled),
        # w-okada is realtime: it's "ready" once its server is up.
        "ready": reachable,
        "realtime": reachable,
    }


@router.get("/status")
async def voice_status() -> dict:
    return await _status_payload()


async def voice_lane_active() -> bool:
    """Used by the worker to keep HFabric GPU jobs queued during live voice."""
    if _session_active:
        return True
    info = await _wokada_get("/info", timeout=0.25)
    if not info:
        return False
    return _bool_setting(info.get("enableServerAudio"))


@router.post("/start")
async def voice_start() -> dict:
    """Launch MMVCServerSIO as a managed subprocess (it serves its UI/API on
    voice_wokada_url). Boot takes a while; poll /status for server_reachable."""
    global _proc
    if not _wokada_installed():
        raise HTTPException(404, f"w-okada not found at {settings.voice_wokada_dir}")
    if await _server_reachable() or _server_running():
        return {"running": True, "already": True}
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    try:
        _proc = subprocess.Popen(
            [str(_exe())],
            cwd=str(settings.voice_wokada_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
    except OSError as exc:
        raise HTTPException(500, f"could not start w-okada: {exc}") from exc
    return {"running": True, "pid": _proc.pid}


@router.post("/stop")
async def voice_stop_server() -> dict:
    return {"stopped": stop_server()}


@router.post("/settings")
async def voice_settings(body: VoiceSettingsUpdate) -> dict:
    if not await _server_reachable():
        raise HTTPException(503, f"w-okada server is not reachable at {settings.voice_wokada_url}")
    await _apply_voice_settings(body)
    return await _status_payload()


@router.post("/session/start")
async def voice_session_start(
    body: VoiceSettingsUpdate,
    arbiter: GpuArbiter = Depends(get_arbiter),
    worker: Worker = Depends(get_worker),
) -> dict:
    """Enable w-okada server-audio mode after releasing HFabric's GPU resident."""
    global _session_active
    if not await _server_reachable():
        raise HTTPException(503, f"w-okada server is not reachable at {settings.voice_wokada_url}")
    if worker.running_job_id:
        raise HTTPException(409, f"GPU job is still running ({worker.running_job_id}); wait or stop it first")
    await arbiter.free_all()
    await _apply_voice_settings(body)
    await _wokada_update("enableServerAudio", 1)
    _session_active = True
    return await _status_payload()


@router.post("/session/stop")
async def voice_session_stop() -> dict:
    global _session_active
    if await _server_reachable():
        await _wokada_update("enableServerAudio", 0)
    _session_active = False
    return await _status_payload()


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
