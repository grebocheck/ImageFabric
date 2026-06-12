"""Pretrained asset discovery for the native RVC engine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...config import settings


@dataclass(frozen=True)
class RequiredAsset:
    name: str
    filenames: tuple[str, ...]


REQUIRED_ASSETS = (
    RequiredAsset("content_vec", ("content_vec_500.onnx", "content_vec_500.fp16.onnx")),
    RequiredAsset("rmvpe", ("rmvpe.pt",)),
)


def _candidate_dirs() -> tuple[tuple[str, Path], ...]:
    return (
        ("local", settings.voice_pretrain_dir),
    )


def _find_asset(asset: RequiredAsset) -> dict[str, Any]:
    for source, root in _candidate_dirs():
        for filename in asset.filenames:
            path = root / filename
            if path.is_file():
                return {
                    "name": asset.name,
                    "path": str(path),
                    "found": True,
                    "source": source,
                }
    return {"name": asset.name, "path": None, "found": False, "source": None}


def discover_assets() -> dict[str, Any]:
    """Return required RVC pretrain assets and an overall readiness flag.

    Discovery is pure pathlib work and intentionally does not probe file
    contents.
    """
    assets = [_find_asset(asset) for asset in REQUIRED_ASSETS]
    return {"ready": all(item["found"] for item in assets), "assets": assets}


def searched_dirs() -> list[str]:
    """Human-readable search roots for precise missing-asset errors."""
    return [str(path) for _, path in _candidate_dirs()]
