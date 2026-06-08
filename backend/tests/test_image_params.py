"""Image-backend param resolution (no GPU — the backend instantiates without
torch; only `load()`/`generate()` touch it).

These lock the family-aware defaults that enforce a load-bearing constraint: a
FLUX.2 [klein] job left at app defaults must fall back to klein's distilled
6-step / 768² settings, never the 28-step / 1024² SDXL defaults.
"""

from __future__ import annotations

from pathlib import Path

from app.backends.base import ModelDescriptor
from app.backends.image_diffusers import DiffusersImageBackend
from app.config import settings
from app.core.enums import ModelFamily


def _backend(family: ModelFamily) -> DiffusersImageBackend:
    desc = ModelDescriptor(
        id="m", name="M", family=family, path=Path("x"), size_bytes=0, quant=None
    )
    return DiffusersImageBackend(desc)


def test_public_params_strips_private_keys():
    out = DiffusersImageBackend._public_params(
        {"prompt": "x", "_lora_paths": {"a": "/p"}, "steps": 4}
    )
    assert out == {"prompt": "x", "steps": 4}


def test_dimension_uses_param_then_default_for_sdxl():
    b = _backend(ModelFamily.SDXL)
    assert b._dimension({"width": 512}, "width", settings.default_width, settings.flux2_default_width) == 512
    assert b._dimension({}, "width", settings.default_width, settings.flux2_default_width) == settings.default_width


def test_dimension_falls_back_to_flux2_default_when_unset():
    b = _backend(ModelFamily.FLUX2)
    # absent -> klein default (768), present -> honored
    assert b._dimension({}, "width", settings.default_width, settings.flux2_default_width) == settings.flux2_default_width
    assert b._dimension({"width": 1024}, "width", settings.default_width, settings.flux2_default_width) == 1024


def test_dimension_falls_back_to_qwen_and_z_image_defaults_when_unset():
    qwen = _backend(ModelFamily.QWEN_IMAGE)
    z_image = _backend(ModelFamily.Z_IMAGE)
    assert qwen._dimension({}, "width", settings.default_width, settings.flux2_default_width) == settings.qwen_image_default_width
    assert qwen._dimension({}, "height", settings.default_height, settings.flux2_default_height) == settings.qwen_image_default_height
    assert z_image._dimension({}, "width", settings.default_width, settings.flux2_default_width) == settings.z_image_default_width
    assert z_image._dimension({}, "height", settings.default_height, settings.flux2_default_height) == settings.z_image_default_height
    assert z_image._dimension({"width": 768}, "width", settings.default_width, settings.flux2_default_width) == 768


def test_steps_flux2_default_when_untouched():
    b = _backend(ModelFamily.FLUX2)
    assert b._steps({}) == settings.flux2_default_steps
    # An explicit, non-default value is respected.
    assert b._steps({"steps": 20}) == 20


def test_steps_sdxl_uses_global_default():
    b = _backend(ModelFamily.SDXL)
    assert b._steps({}) == settings.default_steps
    assert b._steps({"steps": 15}) == 15


def test_steps_qwen_and_z_image_defaults_when_untouched():
    assert _backend(ModelFamily.QWEN_IMAGE)._steps({}) == settings.qwen_image_default_steps
    assert _backend(ModelFamily.Z_IMAGE)._steps({}) == settings.z_image_default_steps
    assert _backend(ModelFamily.Z_IMAGE)._steps({"steps": 12}) == 12


def test_guidance_flux2_default_when_untouched():
    b = _backend(ModelFamily.FLUX2)
    assert b._guidance({}) == settings.flux2_default_guidance
    assert b._guidance({"guidance": 7.5}) == 7.5


def test_guidance_qwen_and_z_image_defaults_when_untouched():
    assert _backend(ModelFamily.QWEN_IMAGE)._guidance({}) == settings.qwen_image_default_guidance
    assert _backend(ModelFamily.Z_IMAGE)._guidance({}) == settings.z_image_default_guidance
    assert _backend(ModelFamily.Z_IMAGE)._guidance({"guidance": 1.5}) == 1.5
