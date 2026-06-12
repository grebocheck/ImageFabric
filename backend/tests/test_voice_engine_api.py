from __future__ import annotations

import io
import math
import struct
import wave

from httpx import ASGITransport, AsyncClient
import pytest

from app.config import settings
from app.main import app
from app.services.voice_engine import engine as engine_mod


@pytest.fixture
async def client(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "voice_models_dir", tmp_path / "voice")
    monkeypatch.setattr(settings, "voice_pretrain_dir", tmp_path / "pretrain")
    monkeypatch.setattr(settings, "voice_max_upload_mb", 64)
    monkeypatch.setattr(engine_mod, "_ENGINE", None)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c
    monkeypatch.setattr(engine_mod, "_ENGINE", None)


def _wav_bytes(freq: float = 440.0, duration: float = 0.2, rate: int = 16000) -> bytes:
    frames = bytearray()
    for i in range(int(duration * rate)):
        sample = int(math.sin(2.0 * math.pi * freq * i / rate) * 16000)
        frames.extend(struct.pack("<h", sample))
    out = io.BytesIO()
    with wave.open(out, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(bytes(frames))
    return out.getvalue()


def _add_fake_slot() -> str:
    slot = settings.voice_models_dir / "fake-slot"
    slot.mkdir(parents=True)
    (slot / "fake.pth").write_bytes(b"not a real checkpoint")
    return "fake-slot"


async def test_status_reports_stub_ready_and_fake_devices(client):
    body = (await client.get("/api/voice/engine/status")).json()

    assert body["engine"] == "native-rvc"
    assert body["stub"] is True
    assert body["ready"] is True
    assert len(body["audio_devices"]["inputs"]) == 2
    assert len(body["audio_devices"]["outputs"]) == 2


async def test_settings_clamps_and_rejects_bad_f0(client):
    response = await client.post(
        "/api/voice/engine/settings",
        json={
            "pitch": 99,
            "index_ratio": 3,
            "protect": -1,
            "server_input_gain": 9,
            "server_output_gain": -2,
        },
    )
    assert response.status_code == 200
    current = response.json()["settings"]
    assert current["pitch"] == 24
    assert current["index_ratio"] == 1.0
    assert current["protect"] == 0.0
    assert current["server_input_gain"] == 4.0
    assert current["server_output_gain"] == 0.0

    bad = await client.post("/api/voice/engine/settings", json={"f0_detector": "nope"})
    assert bad.status_code == 400


async def test_convert_round_trip_is_deterministic(client):
    model_id = _add_fake_slot()
    payload = _wav_bytes()
    data = {"model_id": model_id, "pitch": "5", "index_ratio": "0.25", "protect": "0.3"}

    first = await client.post(
        "/api/voice/engine/convert",
        data=data,
        files={"file": ("tone.wav", payload, "audio/wav")},
    )
    second = await client.post(
        "/api/voice/engine/convert",
        data=data,
        files={"file": ("tone.wav", payload, "audio/wav")},
    )

    assert first.status_code == 200
    assert second.status_code == 200
    first_body = first.json()
    second_body = second.json()
    assert first_body["token"] != second_body["token"]
    assert first_body["url"].endswith(first_body["token"])
    assert first_body["duration_s"] > 0
    assert first_body["sample_rate"] == 16000
    assert first_body["model_id"] == model_id
    assert first_body["params"]["pitch"] == 5
    assert "stub_convert" in first_body["timings_ms"]

    first_wav = (await client.get(first_body["url"])).content
    second_wav = (await client.get(second_body["url"])).content
    assert first_wav == second_wav

    with wave.open(io.BytesIO(first_wav), "rb") as reader:
        assert reader.getnchannels() == 1
        assert reader.getframerate() == 16000
        assert reader.getnframes() > 0


async def test_convert_enforces_upload_size_before_full_buffer(client, monkeypatch):
    model_id = _add_fake_slot()
    monkeypatch.setattr(settings, "voice_max_upload_mb", 0)

    response = await client.post(
        "/api/voice/engine/convert",
        data={"model_id": model_id},
        files={"file": ("tone.wav", _wav_bytes(), "audio/wav")},
    )

    assert response.status_code == 413


async def test_file_token_guard_returns_404(client):
    assert (await client.get("/api/voice/engine/file/not-a-token")).status_code == 404
    assert (await client.get("/api/voice/engine/file/..%2Fsecret")).status_code == 404
