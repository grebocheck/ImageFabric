"""Discovers local model files and hands out (cached) backends for them.

Scanning only reads safetensors headers, so it is instant even for the 16 GB
FLUX file. Backends are created lazily on first use and cached; the arbiter is
what decides which one is actually resident in VRAM.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..config import settings
from ..core.enums import ModelFamily
from .base import GpuBackend, LoraDescriptor, ModelDescriptor
from .image_diffusers import DiffusersImageBackend
from .inspect import classify_image_model, classify_lora_model, is_flux2_dir
from .llm_llamacpp import LlamaCppBackend

LORA_EXTENSIONS = {".safetensors", ".pt", ".bin"}


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _nunchaku_quant(name: str) -> str:
    if "fp4" in name:
        return "nunchaku-fp4"
    if "int4" in name or "awq" in name:
        return "nunchaku-int4"
    return "nunchaku"


class ModelRegistry:
    def __init__(self) -> None:
        self._descriptors: dict[str, ModelDescriptor] = {}
        self._loras: dict[str, LoraDescriptor] = {}
        self._backends: dict[str, GpuBackend] = {}

    def scan(self) -> None:
        self._descriptors.clear()
        self._loras.clear()
        for path in sorted(settings.image_models_dir.glob("*.safetensors")):
            name = path.stem.lower()
            if "svdq" in name or "nunchaku" in name:
                # SVDQuant transformer-only checkpoint (Blackwell fp4/int4 turbo)
                self._add(path, ModelFamily.FLUX, quant=_nunchaku_quant(name))
            else:
                self._add(path, classify_image_model(path))
        # FLUX.2 [klein] is a multi-file diffusers repo dropped in as a folder.
        for sub in sorted(p for p in settings.image_models_dir.iterdir() if p.is_dir()):
            if is_flux2_dir(sub):
                self._add(sub, ModelFamily.FLUX2, quant=settings.flux2_quant)
        for path in sorted(settings.llm_models_dir.glob("*.gguf")):
            self._add(path, ModelFamily.GGUF)
        for root in self._lora_scan_roots():
            if not root.exists():
                continue
            for path in sorted(p for p in root.rglob("*") if p.suffix.lower() in LORA_EXTENSIONS):
                self._add_lora(path)

    def _lora_scan_roots(self) -> list[Path]:
        roots = [
            settings.lora_models_dir,
            settings.image_models_dir / "lora",
            settings.image_models_dir / "loras",
        ]
        deduped: list[Path] = []
        seen: set[Path] = set()
        for root in roots:
            resolved = root.resolve()
            if resolved not in seen:
                deduped.append(root)
                seen.add(resolved)
        return deduped

    @staticmethod
    def _path_size(path: Path) -> int:
        if path.is_dir():
            return sum(
                f.stat().st_size for f in path.rglob("*.safetensors") if f.is_file()
            )
        try:
            return path.stat().st_size
        except OSError:
            return 0

    def _add(self, path, family: ModelFamily, quant: str | None = None) -> None:
        mid = _slug(path.stem)
        size = self._path_size(path)
        self._descriptors[mid] = ModelDescriptor(
            id=mid, name=path.stem, family=family, path=path, size_bytes=size, quant=quant
        )

    def _add_lora(self, path: Path) -> None:
        try:
            rel = path.relative_to(settings.root)
        except ValueError:
            rel = Path(path.name)
        lid = _slug(rel.with_suffix("").as_posix())
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        self._loras[lid] = LoraDescriptor(
            id=lid,
            name=path.stem,
            path=path,
            size_bytes=size,
            family=classify_lora_model(path),
        )

    def descriptors(self) -> list[ModelDescriptor]:
        return list(self._descriptors.values())

    def get_descriptor(self, model_id: str) -> ModelDescriptor:
        if model_id not in self._descriptors:
            raise KeyError(f"unknown model id: {model_id}")
        return self._descriptors[model_id]

    def loras(self, family: ModelFamily | None = None) -> list[LoraDescriptor]:
        loras = list(self._loras.values())
        if family is None:
            return loras
        return [l for l in loras if l.family is None or l.family is family]

    def get_lora(self, lora_id: str) -> LoraDescriptor:
        if lora_id not in self._loras:
            raise KeyError(f"unknown lora id: {lora_id}")
        return self._loras[lora_id]

    def get_backend(self, model_id: str) -> GpuBackend:
        if model_id in self._backends:
            return self._backends[model_id]
        desc = self.get_descriptor(model_id)
        backend: GpuBackend
        if desc.family is ModelFamily.GGUF:
            backend = LlamaCppBackend(desc)
        else:
            backend = DiffusersImageBackend(desc)
        self._backends[model_id] = backend
        return backend

    def peek_backend(self, model_id: str) -> GpuBackend | None:
        return self._backends.get(model_id)

    def loaded_backends(self) -> list[GpuBackend]:
        return [b for b in self._backends.values() if b.loaded]
