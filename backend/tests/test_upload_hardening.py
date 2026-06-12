from __future__ import annotations

import io

from httpx import ASGITransport, AsyncClient
from PIL import Image as PILImage
import pytest

from app.api import transcription
from app.config import settings
from app.main import app


@pytest.fixture
async def client(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "api_token", None)
    monkeypatch.setattr(settings, "outputs_dir", tmp_path / "outputs")
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    PILImage.new("RGB", (8, 8), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


async def test_image_upload_caps_cover_source_and_mask(client, monkeypatch):
    monkeypatch.setattr(settings, "image_upload_max_mb", 0)
    payload = _png_bytes()

    source = await client.post("/api/images/upload", files={"file": ("source.png", payload, "image/png")})
    mask = await client.post("/api/images/upload-mask", files={"file": ("mask.png", payload, "image/png")})

    assert source.status_code == 413
    assert mask.status_code == 413


async def test_transcription_upload_cap_runs_before_model_execution(client, monkeypatch):
    monkeypatch.setattr(settings, "transcription_max_upload_mb", 0)
    monkeypatch.setattr(
        transcription,
        "_model_map",
        lambda: {"stub": {"id": "stub", "engine": "faster-whisper", "path": "stub"}},
    )
    monkeypatch.setattr(transcription, "_has", lambda _module: True)

    response = await client.post(
        "/api/transcription/transcribe",
        data={"model_id": "stub"},
        files={"file": ("tone.wav", b"x", "audio/wav")},
    )

    assert response.status_code == 413


async def test_vision_upload_cap_runs_before_subprocess(client, monkeypatch, tmp_path):
    bin_path = tmp_path / "llama-mtmd-cli.exe"
    bin_path.write_bytes(b"stub")
    models_dir = tmp_path / "vision"
    models_dir.mkdir()
    (models_dir / "vision-model.gguf").write_bytes(b"stub")
    (models_dir / "mmproj-test.gguf").write_bytes(b"stub")
    monkeypatch.setattr(settings, "llama_mtmd_bin", bin_path)
    monkeypatch.setattr(settings, "vision_models_dir", models_dir)
    monkeypatch.setattr(settings, "vision_max_upload_mb", 0)

    response = await client.post(
        "/api/vision/analyze",
        data={"model_id": "vision-model", "projector_id": "mmproj-test", "prompt": "describe"},
        files={"file": ("image.png", b"x", "image/png")},
    )

    assert response.status_code == 413

