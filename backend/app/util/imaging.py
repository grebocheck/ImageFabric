"""Image saving helpers shared by the stub and the real diffusers backend.

Reproducibility is a first-class goal: every saved PNG embeds its full
generation parameters in a ``parameters`` text chunk *and* gets a JSON sidecar,
so any image can be traced back to its exact prompt/seed/settings.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, PngImagePlugin


def day_dir(outputs_dir: Path) -> Path:
    d = outputs_dir / datetime.now(UTC).strftime("%Y-%m-%d")
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_png(img: Image.Image, path: Path, metadata: dict[str, Any]) -> None:
    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", json.dumps(metadata, ensure_ascii=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG", pnginfo=info)
    path.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def make_thumbnail(img: Image.Image, path: Path, size: int = 384) -> None:
    thumb = img.copy()
    thumb.thumbnail((size, size))
    path.parent.mkdir(parents=True, exist_ok=True)
    thumb.save(path, format="WEBP", quality=80)


def make_placeholder(width: int, height: int, lines: list[str]) -> Image.Image:
    """A labelled gradient stand-in used in STUB mode so the gallery/queue
    pipeline can be exercised end-to-end without a real diffusion model."""
    img = Image.new("RGB", (width, height))
    px = img.load()
    for y in range(height):
        for x in range(0, width, 4):  # step 4 px: fast enough for a stub
            r = int(40 + 60 * x / max(width, 1))
            g = int(30 + 90 * y / max(height, 1))
            b = int(80 + 50 * (x + y) / max(width + height, 1))
            for dx in range(4):
                if x + dx < width:
                    px[x + dx, y] = (r, g, b)
    draw = ImageDraw.Draw(img)
    ty = 16
    for line in lines:
        draw.text((16, ty), line[:80], fill=(235, 235, 245))
        ty += 18
    return img
