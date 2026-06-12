from __future__ import annotations

import json

from app.config import settings
from app.services.voice_engine import assets, slots


def test_asset_discovery_prefers_local(monkeypatch, tmp_path):
    local = tmp_path / "local-pretrain"
    local.mkdir()
    (local / "content_vec_500.onnx").write_bytes(b"onnx")
    (local / "rmvpe.pt").write_bytes(b"pt")
    monkeypatch.setattr(settings, "voice_pretrain_dir", local)

    found = assets.discover_assets()

    assert found["ready"] is True
    assert {item["name"]: item["source"] for item in found["assets"]} == {
        "content_vec": "local",
        "rmvpe": "local",
    }


def test_asset_discovery_missing_not_ready(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "voice_pretrain_dir", tmp_path / "missing-local")

    found = assets.discover_assets()

    assert found["ready"] is False
    assert [item["found"] for item in found["assets"]] == [False, False]


def test_slot_discovery_params_bare_and_zip_ignored(monkeypatch, tmp_path):
    local = tmp_path / "voice"
    local.mkdir()

    params_slot = local / "slot-a"
    params_slot.mkdir()
    (params_slot / "voice.pth").write_bytes(b"pth")
    (params_slot / "voice.index").write_bytes(b"index")
    (params_slot / "params.json").write_text(
        json.dumps({
            "name": "Display Voice",
            "voiceChangerType": "RVC",
            "version": "v2",
            "samplingRate": 48000,
            "f0": True,
            "indexFile": "voice.index",
        }),
        encoding="utf-8",
    )

    bare = local / "bare"
    bare.mkdir()
    (bare / "bare.pth").write_bytes(b"pth")

    (local / "not-a-model.zip").write_bytes(b"zip")

    monkeypatch.setattr(settings, "voice_models_dir", local)

    found = slots.discover_slots()
    by_id = {item["id"]: item for item in found}

    assert set(by_id) == {"bare", "slot-a"}
    assert by_id["slot-a"]["name"] == "Display Voice"
    assert by_id["slot-a"]["has_index"] is True
    assert by_id["slot-a"]["sampling_rate"] == 48000
    assert by_id["slot-a"]["f0"] is True
    assert by_id["bare"]["version"] == ""
