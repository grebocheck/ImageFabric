from __future__ import annotations

import io
import json
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
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
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


def _add_fake_pretrain() -> None:
    settings.voice_pretrain_dir.mkdir(parents=True, exist_ok=True)
    (settings.voice_pretrain_dir / "content_vec_500.onnx").write_bytes(b"onnx")
    (settings.voice_pretrain_dir / "rmvpe.pt").write_bytes(b"pt")


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
            "speaker_id": 999,
            "index_ratio": 3,
            "protect": -1,
            "noise_scale": 9,
            "f0_smoothing": -5,
            "input_highpass_hz": 999,
            "input_gate_db": -999,
            "input_formant": 9,
            "input_denoise": "dtln",
            "silence_threshold_db": -999,
            "silence_hold_ms": 9999,
            "server_input_gain": 9,
            "server_output_gain": -2,
        },
    )
    assert response.status_code == 200
    current = response.json()["settings"]
    assert current["pitch"] == 24
    assert current["speaker_id"] == 255
    assert current["index_ratio"] == 1.0
    assert current["protect"] == 0.0
    assert current["noise_scale"] == 1.0
    assert current["f0_smoothing"] == 0.0
    assert current["input_highpass_hz"] == 300
    assert current["input_gate_db"] == -90.0
    assert current["input_formant"] == 2.0
    assert current["input_denoise"] == "dtln"
    assert current["silence_threshold_db"] == -90.0
    assert current["silence_hold_ms"] == 2000.0
    assert current["server_input_gain"] == 4.0
    assert current["server_output_gain"] == 0.0

    off = await client.post("/api/voice/engine/settings", json={"input_gate_db": 0, "input_highpass_hz": "off"})
    assert off.status_code == 200
    assert off.json()["settings"]["input_gate_db"] == -90.0
    assert off.json()["settings"]["input_highpass_hz"] == 0

    bad = await client.post("/api/voice/engine/settings", json={"f0_detector": "nope"})
    assert bad.status_code == 400

    bad_denoise = await client.post("/api/voice/engine/settings", json={"input_denoise": "spectral-magic"})
    assert bad_denoise.status_code == 400

    accepted = await client.post("/api/voice/engine/settings", json={"input_denoise": "dtln"})
    assert accepted.status_code == 200
    assert accepted.json()["settings"]["input_denoise"] == "dtln"


async def test_settings_persist_and_fresh_engine_loads_file(client):
    response = await client.post(
        "/api/voice/engine/settings",
        json={
            "pitch": 7,
            "speaker_id": 2,
            "index_ratio": 0.33,
            "silence_threshold_db": -52,
            "silence_hold_ms": 250,
            "server_input_device_id": 1,
        },
    )
    assert response.status_code == 200

    path = settings.data_dir / "voice-settings.json"
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["pitch"] == 7
    assert persisted["speaker_id"] == 2
    assert persisted["index_ratio"] == 0.33
    assert persisted["silence_threshold_db"] == -52.0
    assert persisted["silence_hold_ms"] == 250.0

    engine_mod._ENGINE = None
    fresh = (await client.get("/api/voice/engine/status")).json()["settings"]
    assert fresh["pitch"] == 7
    assert fresh["speaker_id"] == 2
    assert fresh["index_ratio"] == 0.33
    assert fresh["silence_threshold_db"] == -52.0
    assert fresh["silence_hold_ms"] == 250.0
    assert fresh["server_input_device_id"] == 1


async def test_named_voice_presets_persist_clean_settings(client):
    empty = await client.get("/api/voice/engine/presets")
    assert empty.status_code == 200
    assert empty.json() == []

    response = await client.post(
        "/api/voice/engine/presets",
        json={
            "name": "  Clear test  ",
            "model_id": "voice-a",
            "settings": {
                "pitch": 12,
                "index_ratio": 0.25,
                "noise_scale": 0.45,
                "f0_smoothing": 0.0,
                "server_input_device_id": 99,
                "unknown_future_key": "ignored",
            },
        },
    )
    assert response.status_code == 200
    saved = response.json()
    assert saved["name"] == "Clear test"
    assert saved["model_id"] == "voice-a"
    assert saved["settings"] == {
        "pitch": 12,
        "index_ratio": 0.25,
        "noise_scale": 0.45,
        "f0_smoothing": 0.0,
    }
    assert saved["id"]

    listed = (await client.get("/api/voice/engine/presets")).json()
    assert [item["id"] for item in listed] == [saved["id"]]

    updated_response = await client.patch(
        f"/api/voice/engine/presets/{saved['id']}",
        json={
            "name": "Updated clear",
            "model_id": "voice-b",
            "settings": {"pitch": -7, "index_ratio": 0.4, "unknown_future_key": "ignored"},
        },
    )
    assert updated_response.status_code == 200
    updated = updated_response.json()
    assert updated["name"] == "Updated clear"
    assert updated["model_id"] == "voice-b"
    assert updated["settings"] == {"pitch": -7, "index_ratio": 0.4}

    deleted = await client.delete(f"/api/voice/engine/presets/{saved['id']}")
    assert deleted.status_code == 200
    assert (await client.get("/api/voice/engine/presets")).json() == []


async def test_corrupt_settings_file_falls_back_to_defaults(client):
    path = settings.data_dir / "voice-settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not-json", encoding="utf-8")
    engine_mod._ENGINE = None

    current = (await client.get("/api/voice/engine/status")).json()["settings"]

    assert current["pitch"] == settings.voice_pitch
    assert current["index_ratio"] == settings.voice_index_ratio
    assert current["noise_scale"] == settings.voice_noise_scale
    assert current["f0_smoothing"] == settings.voice_f0_smoothing
    assert current["silence_threshold_db"] == -72.0
    assert current["silence_hold_ms"] == 250.0


async def test_settings_load_clamps_and_flags_missing_devices(client):
    path = settings.data_dir / "voice-settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "pitch": 99,
            "index_ratio": 9,
            "silence_threshold_db": -999,
            "silence_hold_ms": 9999,
            "server_input_device_id": 99,
            "server_output_device_id": 88,
            "server_monitor_device_id": 77,
            "unknown_future_key": "ignored",
        }),
        encoding="utf-8",
    )
    engine_mod._ENGINE = None

    current = (await client.get("/api/voice/engine/status")).json()["settings"]

    assert current["pitch"] == 24
    assert current["index_ratio"] == 1.0
    assert current["silence_threshold_db"] == -90.0
    assert current["silence_hold_ms"] == 2000.0
    assert current["device_missing"] == {"input": True, "output": True, "monitor": True}


async def test_convert_round_trip_is_deterministic(client):
    model_id = _add_fake_slot()
    payload = _wav_bytes()
    data = {
        "model_id": model_id,
        "pitch": "5",
        "speaker_id": "3",
        "index_ratio": "0.25",
        "protect": "0.3",
        "noise_scale": "0.05",
        "f0_smoothing": "0.4",
        "input_highpass_hz": "120",
        "input_gate_db": "-50",
        "input_formant": "1.5",
    }

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
    assert first_body["mp3_url"].endswith(f"{first_body['token']}/mp3")
    assert first_body["duration_s"] > 0
    assert first_body["sample_rate"] == 16000
    assert first_body["model_id"] == model_id
    assert first_body["params"]["pitch"] == 5
    assert first_body["params"]["speaker_id"] == 3
    assert first_body["params"]["noise_scale"] == 0.05
    assert first_body["params"]["f0_smoothing"] == 0.4
    assert first_body["params"]["input_highpass_hz"] == 120
    assert first_body["params"]["input_gate_db"] == -50.0
    assert first_body["params"]["input_formant"] == 1.5
    assert first_body["params"]["input_denoise"] == "off"
    assert "stub_convert" in first_body["timings_ms"]

    first_wav = (await client.get(first_body["url"])).content
    second_wav = (await client.get(second_body["url"])).content
    assert first_wav == second_wav

    mp3 = await client.get(first_body["mp3_url"])
    assert mp3.status_code == 200
    assert mp3.headers["content-type"].startswith("audio/mpeg")
    assert len(mp3.content) > 100

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


async def test_convert_dtln_missing_assets_returns_503_without_loading_torch(client, monkeypatch):
    monkeypatch.setattr(settings, "stub_mode", False)
    model_id = _add_fake_slot()
    _add_fake_pretrain()

    response = await client.post(
        "/api/voice/engine/convert",
        data={"model_id": model_id, "input_denoise": "dtln"},
        files={"file": ("tone.wav", _wav_bytes(), "audio/wav")},
    )

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "denoise_dtln" in detail
    assert str(settings.voice_pretrain_dir / "denoise") in detail


async def test_file_token_guard_returns_404(client):
    assert (await client.get("/api/voice/engine/file/not-a-token")).status_code == 404
    assert (await client.get("/api/voice/engine/file/..%2Fsecret")).status_code == 404
