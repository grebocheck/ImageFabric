"""Model classification from safetensors headers (pure, no weights read).

The registry trusts these to tell FLUX / FLUX.2 / SDXL apart by header keys, with
a size fallback. Getting this wrong routes a model to the wrong backend, so the
key-pattern detection is worth pinning down.
"""

from __future__ import annotations

import json
from pathlib import Path
import struct

from app.backends.inspect import (
    classify_diffusers_dir,
    classify_image_model,
    classify_lora_model,
    is_flux2_dir,
)
from app.core.enums import ModelFamily


def _safetensors(path: Path, keys: list[str]) -> Path:
    header = {k: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]} for k in keys}
    blob = json.dumps(header).encode("utf-8")
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(b"\x00\x00")
    return path


def test_classify_flux2_by_modulation_keys(tmp_path):
    p = _safetensors(tmp_path / "m.safetensors", ["double_stream_modulation_x.weight"])
    assert classify_image_model(p) is ModelFamily.FLUX2


def test_classify_flux_by_block_keys(tmp_path):
    p = _safetensors(tmp_path / "m.safetensors", ["double_blocks.0.img_attn.qkv.weight"])
    assert classify_image_model(p) is ModelFamily.FLUX


def test_classify_sdxl_by_unet_keys(tmp_path):
    p = _safetensors(tmp_path / "m.safetensors", ["model.diffusion_model.input_blocks.0.0.weight"])
    assert classify_image_model(p) is ModelFamily.SDXL


def test_classify_small_unknown_file_falls_back_to_sdxl(tmp_path):
    # Valid header, no recognized keys, small file -> size heuristic picks SDXL.
    p = _safetensors(tmp_path / "m.safetensors", ["some.random.tensor"])
    assert classify_image_model(p) is ModelFamily.SDXL


def test_classify_unreadable_file_is_unknown(tmp_path):
    bad = tmp_path / "broken.safetensors"
    bad.write_bytes(b"not a safetensors header")
    assert classify_image_model(bad) is ModelFamily.UNKNOWN


def test_classify_lora_by_path_name(tmp_path):
    flux = _safetensors(tmp_path / "my-flux-lora.safetensors", ["x"])
    sdxl = _safetensors(tmp_path / "noobai-style.safetensors", ["x"])
    assert classify_lora_model(flux) is ModelFamily.FLUX
    assert classify_lora_model(sdxl) is ModelFamily.SDXL


def test_classify_lora_by_header_when_name_is_neutral(tmp_path):
    p = _safetensors(tmp_path / "adapter.safetensors", ["lora_unet_down_blocks_0.weight"])
    assert classify_lora_model(p) is ModelFamily.SDXL


def test_is_flux2_dir_detects_pipeline_class(tmp_path):
    d = tmp_path / "klein"
    d.mkdir()
    (d / "model_index.json").write_text(json.dumps({"_class_name": "Flux2Pipeline"}), encoding="utf-8")
    assert is_flux2_dir(d) is True


def test_is_flux2_dir_false_for_other_pipelines(tmp_path):
    d = tmp_path / "sdxl"
    d.mkdir()
    (d / "model_index.json").write_text(json.dumps({"_class_name": "StableDiffusionXLPipeline"}), encoding="utf-8")
    assert is_flux2_dir(d) is False


def test_is_flux2_dir_false_without_index(tmp_path):
    d = tmp_path / "bare"
    d.mkdir()
    assert is_flux2_dir(d) is False


def test_classify_diffusers_dir_detects_qwen_image(tmp_path):
    d = tmp_path / "qwen-image-2512"
    d.mkdir()
    (d / "model_index.json").write_text(json.dumps({"_class_name": "QwenImagePipeline"}), encoding="utf-8")
    assert classify_diffusers_dir(d) is ModelFamily.QWEN_IMAGE


def test_classify_diffusers_dir_detects_z_image(tmp_path):
    d = tmp_path / "z-image-turbo"
    d.mkdir()
    (d / "model_index.json").write_text(json.dumps({"_class_name": "ZImagePipeline"}), encoding="utf-8")
    assert classify_diffusers_dir(d) is ModelFamily.Z_IMAGE


def test_classify_diffusers_dir_ignores_unknown_pipeline(tmp_path):
    d = tmp_path / "other"
    d.mkdir()
    (d / "model_index.json").write_text(json.dumps({"_class_name": "StableDiffusionXLPipeline"}), encoding="utf-8")
    assert classify_diffusers_dir(d) is None
