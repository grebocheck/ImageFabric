"""Coordinator for native RVC model loading and conversion."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
import wave

from ...config import settings
from . import assets as asset_discovery
from . import dsp, slots
from .f0 import F0_DETECTORS, SUPPORTED_DETECTORS


def _clamp_pitch(value: int) -> int:
    return max(-24, min(24, int(value)))


def _clamp_ratio(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _clamp_gain(value: float) -> float:
    return max(0.0, min(4.0, float(value)))


def _clamp_device_id(value: int, *, allow_none: bool = False) -> int:
    n = int(value)
    if allow_none and n < 0:
        return -1
    return max(0, n)


def _validate_f0_detector(value: str) -> str:
    if value not in F0_DETECTORS:
        raise ValueError(f"unsupported f0 detector: {value}")
    if value not in SUPPORTED_DETECTORS:
        raise ValueError(f"native voice engine supports only rmvpe for now (got {value})")
    return value


def _clamp_sample_rate(value: int) -> int:
    return max(8_000, min(192_000, int(value)))


def _clamp_chunk_size(value: int) -> int:
    return max(1, min(1_024, int(value)))


def _clamp_seconds(value: float, *, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, float(value)))


@dataclass
class ConvertParams:
    pitch: int
    index_ratio: float
    protect: float
    f0_detector: str
    input_highpass_hz: int
    input_gate_db: float
    input_formant: float

    def public(self) -> dict[str, float | int | str]:
        return {
            "pitch": self.pitch,
            "index_ratio": self.index_ratio,
            "protect": self.protect,
            "f0_detector": self.f0_detector,
            "input_highpass_hz": self.input_highpass_hz,
            "input_gate_db": self.input_gate_db,
            "input_formant": self.input_formant,
        }


class VoiceEngine:
    def __init__(self) -> None:
        self.pitch = _clamp_pitch(settings.voice_pitch)
        self.index_ratio = _clamp_ratio(settings.voice_index_ratio)
        self.protect = _clamp_ratio(settings.voice_protect)
        self.f0_detector = _validate_f0_detector(settings.voice_f0_detector)
        self.input_highpass_hz = dsp.clamp_input_highpass_hz(settings.voice_input_highpass_hz)
        self.input_gate_db = dsp.clamp_input_gate_db(settings.voice_input_gate_db)
        self.input_formant = dsp.clamp_input_formant(settings.voice_input_formant)
        self.device = settings.voice_device
        self.server_input_device_id: int | None = None
        self.server_output_device_id: int | None = None
        self.server_monitor_device_id: int | None = None
        self.server_input_gain = 1.0
        self.server_output_gain = 1.0
        self.server_monitor_gain = 1.0
        # Realtime session knobs (P6R.2); w-okada conventions kept for UI parity.
        self.server_audio_sample_rate = 48_000
        self.server_read_chunk_size = 133
        self.cross_fade_overlap_size = 0.05
        self.extra_convert_size = 2.0
        self.pass_through = False
        self._lock = asyncio.Lock()
        self._loaded_model_id: str | None = None
        self._loaded_model = None

    @property
    def loaded_model_id(self) -> str | None:
        return self._loaded_model_id

    def settings_payload(self) -> dict:
        return {
            "pitch": self.pitch,
            "index_ratio": self.index_ratio,
            "protect": self.protect,
            "f0_detector": self.f0_detector,
            "input_highpass_hz": self.input_highpass_hz,
            "input_gate_db": self.input_gate_db,
            "input_formant": self.input_formant,
            "server_input_device_id": self.server_input_device_id,
            "server_output_device_id": self.server_output_device_id,
            "server_monitor_device_id": self.server_monitor_device_id,
            "server_input_gain": self.server_input_gain,
            "server_output_gain": self.server_output_gain,
            "server_monitor_gain": self.server_monitor_gain,
            "server_audio_sample_rate": self.server_audio_sample_rate,
            "server_read_chunk_size": self.server_read_chunk_size,
            "cross_fade_overlap_size": self.cross_fade_overlap_size,
            "extra_convert_size": self.extra_convert_size,
            "pass_through": self.pass_through,
        }

    def update_settings(self, data: dict) -> None:
        if data.get("pitch") is not None:
            self.pitch = _clamp_pitch(data["pitch"])
        if data.get("index_ratio") is not None:
            self.index_ratio = _clamp_ratio(data["index_ratio"])
        if data.get("protect") is not None:
            self.protect = _clamp_ratio(data["protect"])
        if data.get("f0_detector") is not None:
            self.f0_detector = _validate_f0_detector(str(data["f0_detector"]))
        if data.get("input_highpass_hz") is not None:
            self.input_highpass_hz = dsp.clamp_input_highpass_hz(data["input_highpass_hz"])
        if data.get("input_gate_db") is not None:
            self.input_gate_db = dsp.clamp_input_gate_db(data["input_gate_db"])
        if data.get("input_formant") is not None:
            self.input_formant = dsp.clamp_input_formant(data["input_formant"])
        if data.get("server_input_device_id") is not None:
            self.server_input_device_id = _clamp_device_id(data["server_input_device_id"])
        if data.get("server_output_device_id") is not None:
            self.server_output_device_id = _clamp_device_id(data["server_output_device_id"])
        if data.get("server_monitor_device_id") is not None:
            self.server_monitor_device_id = _clamp_device_id(
                data["server_monitor_device_id"],
                allow_none=True,
            )
        if data.get("server_input_gain") is not None:
            self.server_input_gain = _clamp_gain(data["server_input_gain"])
        if data.get("server_output_gain") is not None:
            self.server_output_gain = _clamp_gain(data["server_output_gain"])
        if data.get("server_monitor_gain") is not None:
            self.server_monitor_gain = _clamp_gain(data["server_monitor_gain"])
        if data.get("server_audio_sample_rate") is not None:
            self.server_audio_sample_rate = _clamp_sample_rate(data["server_audio_sample_rate"])
        if data.get("server_read_chunk_size") is not None:
            self.server_read_chunk_size = _clamp_chunk_size(data["server_read_chunk_size"])
        if data.get("cross_fade_overlap_size") is not None:
            self.cross_fade_overlap_size = _clamp_seconds(
                data["cross_fade_overlap_size"], min_value=0.0, max_value=1.0
            )
        if data.get("extra_convert_size") is not None:
            self.extra_convert_size = _clamp_seconds(
                data["extra_convert_size"], min_value=0.0, max_value=20.0
            )
        if data.get("pass_through") is not None:
            self.pass_through = bool(data["pass_through"])

    def models(self) -> list[dict]:
        models = slots.discover_slots()
        if settings.stub_mode and not models:
            return [{
                "id": "stub-voice",
                "slot": "stub-voice",
                "name": "Stub RVC Voice",
                "type": "RVC",
                "version": "stub",
                "sampling_rate": 48000,
                "f0": True,
                "has_index": False,
                "size_bytes": 0,
                "source": "stub",
            }]
        return models

    def get_model(self, model_id: str) -> dict | None:
        slot = slots.get_slot(model_id)
        if slot is not None:
            return slot
        if settings.stub_mode and model_id == "stub-voice" and not slots.discover_slots():
            return {
                "id": "stub-voice",
                "slot": "stub-voice",
                "name": "Stub RVC Voice",
                "type": "RVC",
                "version": "stub",
                "sampling_rate": 48000,
                "f0": True,
                "has_index": False,
                "size_bytes": 0,
                "source": "stub",
                "path": "",
                "model_path": "",
                "index_path": None,
            }
        return None

    def assets(self) -> dict:
        found = asset_discovery.discover_assets()
        if not settings.stub_mode:
            return found
        return {
            "ready": True,
            "assets": [
                {
                    "name": item["name"],
                    "path": item["path"],
                    "found": True,
                    "source": item["source"] or "stub",
                }
                for item in found["assets"]
            ],
        }

    def _params(
        self,
        pitch: int | None,
        index_ratio: float | None,
        protect: float | None,
        input_highpass_hz: int | str | None = None,
        input_gate_db: float | str | None = None,
        input_formant: float | None = None,
    ) -> ConvertParams:
        return ConvertParams(
            pitch=self.pitch if pitch is None else _clamp_pitch(pitch),
            index_ratio=self.index_ratio if index_ratio is None else _clamp_ratio(index_ratio),
            protect=self.protect if protect is None else _clamp_ratio(protect),
            f0_detector=self.f0_detector,
            input_highpass_hz=(
                self.input_highpass_hz
                if input_highpass_hz is None
                else dsp.clamp_input_highpass_hz(input_highpass_hz)
            ),
            input_gate_db=self.input_gate_db if input_gate_db is None else dsp.clamp_input_gate_db(input_gate_db),
            input_formant=self.input_formant if input_formant is None else dsp.clamp_input_formant(input_formant),
        )

    async def unload(self) -> None:
        async with self._lock:
            self._loaded_model = None
            self._loaded_model_id = None
            if not settings.stub_mode:
                try:
                    import torch  # noqa: PLC0415

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    return

    async def convert_file(
        self,
        input_path: Path,
        output_path: Path,
        model_id: str,
        *,
        pitch: int | None = None,
        index_ratio: float | None = None,
        protect: float | None = None,
        input_highpass_hz: int | str | None = None,
        input_gate_db: float | str | None = None,
        input_formant: float | None = None,
    ) -> dict:
        params = self._params(pitch, index_ratio, protect, input_highpass_hz, input_gate_db, input_formant)
        async with self._lock:
            if settings.stub_mode:
                return await asyncio.to_thread(
                    self._convert_stub_sync,
                    input_path,
                    output_path,
                    params,
                    model_id,
                )
            return await asyncio.to_thread(
                self._convert_real_sync,
                input_path,
                output_path,
                params,
                model_id,
            )

    def _convert_stub_sync(
        self,
        input_path: Path,
        output_path: Path,
        params: ConvertParams,
        model_id: str,
    ) -> dict:
        with wave.open(str(input_path), "rb") as reader:
            nchannels = reader.getnchannels()
            sampwidth = reader.getsampwidth()
            framerate = reader.getframerate()
            nframes = reader.getnframes()
            frames = reader.readframes(nframes)

        frame_size = max(1, nchannels * sampwidth)
        chunks = [frames[i : i + frame_size] for i in range(0, len(frames), frame_size)]
        if chunks:
            factor = 2.0 ** (params.pitch / 12.0)
            transformed = bytearray()
            for i in range(len(chunks)):
                transformed.extend(chunks[int(i * factor) % len(chunks)])
            out_frames = bytes(transformed)
        else:
            out_frames = b""

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output_path), "wb") as writer:
            writer.setnchannels(nchannels)
            writer.setsampwidth(sampwidth)
            writer.setframerate(framerate)
            writer.writeframes(out_frames)

        duration = (len(out_frames) // frame_size) / float(framerate) if framerate else 0.0
        return {
            "duration_s": duration,
            "sample_rate": framerate,
            "timings_ms": {"stub_convert": 0.0},
            "model_id": model_id,
            "params": params.public(),
        }

    def load_model_sync(self, model_id: str):
        """Blocking model load for the realtime session (worker-thread side).
        In stub mode there is nothing to load — return the slot dict."""
        if settings.stub_mode:
            slot = self.get_model(model_id)
            if slot is None:
                raise FileNotFoundError("voice model not found")
            return slot
        return self._load_real_model(model_id)

    def _load_real_model(self, model_id: str):
        if self._loaded_model_id == model_id and self._loaded_model is not None:
            return self._loaded_model
        slot = slots.get_slot(model_id)
        if slot is None:
            raise FileNotFoundError("voice model not found")
        assets = asset_discovery.discover_assets()
        missing = [item["name"] for item in assets["assets"] if not item["found"]]
        if missing:
            dirs = ", ".join(asset_discovery.searched_dirs())
            raise RuntimeError(f"missing required voice pretrain asset(s): {', '.join(missing)}; searched {dirs}")
        asset_paths = {item["name"]: item["path"] for item in assets["assets"]}
        from . import pipeline  # noqa: PLC0415

        self._loaded_model = pipeline.load_model(slot, asset_paths, self.device)
        self._loaded_model_id = model_id
        return self._loaded_model

    def _convert_real_sync(
        self,
        input_path: Path,
        output_path: Path,
        params: ConvertParams,
        model_id: str,
    ) -> dict:
        import soundfile as sf  # noqa: PLC0415

        from . import pipeline  # noqa: PLC0415

        loaded = self._load_real_model(model_id)
        audio, sr, timings = pipeline.convert(
            input_path,
            loaded,
            pitch=params.pitch,
            index_ratio=params.index_ratio,
            protect=params.protect,
            f0_detector=params.f0_detector,
            input_highpass_hz=params.input_highpass_hz,
            input_gate_db=params.input_gate_db,
            input_formant=params.input_formant,
            device=self.device,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio, sr, subtype="PCM_16")
        duration = len(audio) / float(sr) if sr else 0.0
        return {
            "duration_s": duration,
            "sample_rate": sr,
            "timings_ms": timings,
            "model_id": model_id,
            "params": params.public(),
        }


_ENGINE: VoiceEngine | None = None


def get_engine() -> VoiceEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = VoiceEngine()
    return _ENGINE
