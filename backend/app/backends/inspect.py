"""Lightweight model classification.

We only read the safetensors *header* (a small JSON blob at the start of the
file), never the multi-GB tensor data, so scanning is instant. This is the same
trick used to confirm the local files: FLUX has ``double_blocks``/``single_blocks``
while SDXL has UNet ``input_blocks`` under ``model.diffusion_model``.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

from ..core.enums import ModelFamily


def _read_safetensors_keys(path: Path, *, probe_limit: int = 4096) -> list[str]:
    with path.open("rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    header.pop("__metadata__", None)
    keys = list(header.keys())
    return keys[:probe_limit]


def classify_image_model(path: Path) -> ModelFamily:
    try:
        keys = _read_safetensors_keys(path)
    except Exception:
        return ModelFamily.UNKNOWN

    joined = "\n".join(keys)
    # FLUX.2 uses a new modulation scheme (double_stream_modulation_*) on top of
    # the double/single block layout — check it first to tell klein from FLUX.1.
    if "double_stream_modulation" in joined or "single_stream_modulation" in joined:
        return ModelFamily.FLUX2
    if "double_blocks" in joined or "single_blocks" in joined:
        return ModelFamily.FLUX
    if "input_blocks" in joined or "conditioner.embedders" in joined:
        return ModelFamily.SDXL
    # Heuristic fallback by size: FLUX checkpoints are much larger than SDXL.
    try:
        return ModelFamily.FLUX if path.stat().st_size > 10 * 1024**3 else ModelFamily.SDXL
    except OSError:
        return ModelFamily.UNKNOWN


def is_flux2_dir(path: Path) -> bool:
    """FLUX.2 [klein] ships as a multi-file diffusers repo (not a single
    .safetensors), so we detect it by a ``model_index.json`` whose pipeline
    class is a Flux2 variant."""
    index = path / "model_index.json"
    if not index.is_file():
        return False
    try:
        data = json.loads(index.read_text(encoding="utf-8"))
    except Exception:
        return False
    return "Flux2" in str(data.get("_class_name", ""))


def classify_lora_model(path: Path) -> ModelFamily | None:
    text = " ".join(part.lower() for part in path.parts[-5:])
    if "flux" in text:
        return ModelFamily.FLUX
    if "sdxl" in text or "sd_xl" in text or "noobai" in text:
        return ModelFamily.SDXL

    try:
        keys = _read_safetensors_keys(path, probe_limit=1024)
    except Exception:
        return None
    joined = "\n".join(keys).lower()
    if "single_transformer_blocks" in joined or "double_blocks" in joined or ".transformer." in joined:
        return ModelFamily.FLUX
    if "lora_unet" in joined or ".unet." in joined or "conditioner.embedders" in joined:
        return ModelFamily.SDXL
    return None
