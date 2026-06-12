"""RVC model-slot discovery for the native voice engine."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from ...config import settings

MODEL_EXTS = {".pth", ".safetensors"}


@dataclass(frozen=True)
class VoiceSlot:
    id: str
    slot: str
    name: str
    type: str
    version: str
    sampling_rate: int | str | None
    f0: bool | None
    has_index: bool
    size_bytes: int
    source: str
    path: str
    model_path: str
    index_path: str | None

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "slot": self.slot,
            "name": self.name,
            "type": self.type,
            "version": self.version,
            "sampling_rate": self.sampling_rate,
            "f0": self.f0,
            "has_index": self.has_index,
            "size_bytes": self.size_bytes,
            "source": self.source,
        }


def _model_roots() -> tuple[tuple[str, Path], ...]:
    return (
        ("local", settings.voice_models_dir),
    )


def _dir_size(path: Path) -> int:
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def _read_params(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _choose_model_file(slot: Path, meta: dict[str, Any]) -> Path | None:
    hinted = meta.get("modelFile") or meta.get("model_file") or meta.get("model")
    if isinstance(hinted, str) and (slot / hinted).suffix.lower() in MODEL_EXTS and (slot / hinted).is_file():
        return slot / hinted
    files = sorted(path for path in slot.iterdir() if path.is_file() and path.suffix.lower() in MODEL_EXTS)
    return files[0] if files else None


def _choose_index_file(slot: Path, meta: dict[str, Any]) -> Path | None:
    hinted = meta.get("indexFile") or meta.get("index_file") or meta.get("index")
    if isinstance(hinted, str) and (slot / hinted).suffix.lower() == ".index" and (slot / hinted).is_file():
        return slot / hinted
    files = sorted(path for path in slot.iterdir() if path.is_file() and path.suffix.lower() == ".index")
    return files[0] if files else None


def _sampling_rate(meta: dict[str, Any]) -> int | str | None:
    value = meta.get("samplingRate", meta.get("sampling_rate"))
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _f0(meta: dict[str, Any]) -> bool | None:
    if "f0" not in meta:
        return None
    return bool(meta.get("f0"))


def _slot_from_dir(slot: Path, source: str, seen: set[str]) -> VoiceSlot | None:
    if slot.name.lower().endswith(".zip"):
        return None
    if not slot.is_dir():
        return None
    meta = _read_params(slot / "params.json")
    model_file = _choose_model_file(slot, meta)
    if model_file is None:
        return None
    index_file = _choose_index_file(slot, meta)
    base_id = slot.name
    model_id = base_id if base_id not in seen else f"{source}:{base_id}"
    seen.add(model_id)
    return VoiceSlot(
        id=model_id,
        slot=slot.name,
        name=str(meta.get("name") or model_file.stem or slot.name),
        type=str(meta.get("voiceChangerType") or meta.get("type") or "RVC"),
        version=str(meta.get("version") or ""),
        sampling_rate=_sampling_rate(meta),
        f0=_f0(meta),
        has_index=index_file is not None,
        size_bytes=_dir_size(slot),
        source=source,
        path=str(slot),
        model_path=str(model_file),
        index_path=str(index_file) if index_file else None,
    )


def discover_slots(*, include_private: bool = False) -> list[dict[str, Any]]:
    """Scan local model folders for RVC slots.

    Discovery looks only at filenames and optional ``params.json``. Checkpoint
    metadata requires torch/safetensors and is read by the loader, not here.
    """
    seen: set[str] = set()
    slots: list[VoiceSlot] = []
    for source, root in _model_roots():
        if not root.exists():
            continue
        for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
            slot = _slot_from_dir(child, source, seen)
            if slot is not None:
                slots.append(slot)
    if include_private:
        return [slot.__dict__.copy() for slot in slots]
    return [slot.public() for slot in slots]


def get_slot(model_id: str) -> dict[str, Any] | None:
    for slot in discover_slots(include_private=True):
        if slot["id"] == model_id:
            return slot
    return None
