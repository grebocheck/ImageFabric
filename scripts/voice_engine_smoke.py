#!/usr/bin/env python3
"""Smoke-test the native RVC voice engine with real RVC v2 + RMVPE assets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.voice_engine import assets as asset_discovery  # noqa: E402
from app.services.voice_engine import pipeline  # noqa: E402
from app.services.voice_engine.rvc.models import SynthesizerTrnMs768NSFsid  # noqa: E402
from app.services.voice_engine.rvc.rmvpe import RMVPE  # noqa: E402

DEFAULT_MODEL = Path(r"D:\MMVCServerSIO\model_dir\chocola_yagiyukiv2\chocola_yagiyukiv2.pth")
DEFAULT_OUT = Path("data/runtime/voice_smoke.wav")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--formant", type=float, default=1.0)
    return parser.parse_args()


def asset_paths() -> dict[str, str]:
    found = asset_discovery.discover_assets()
    print("asset discovery:")
    for item in found["assets"]:
        print(f"  {item['name']}: found={item['found']} source={item['source']} path={item['path']}")
    if not found["ready"]:
        missing = ", ".join(item["name"] for item in found["assets"] if not item["found"])
        searched = ", ".join(asset_discovery.searched_dirs())
        raise AssertionError(f"missing required asset(s): {missing}; searched {searched}")
    return {item["name"]: item["path"] for item in found["assets"]}


def rmvpe_sanity(rmvpe_path: str, device: str) -> float:
    rmvpe = RMVPE(Path(rmvpe_path), device=device)
    sr = 16000
    t = np.arange(sr, dtype=np.float32) / sr
    audio = (0.2 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
    f0 = rmvpe.infer_from_audio(audio, sr=sr)
    voiced = f0[f0 > 0]
    if voiced.size == 0:
        raise AssertionError("RMVPE returned no voiced frames for a 220 Hz sine")
    median = float(np.median(voiced))
    print(f"rmvpe_sanity: frames={len(f0)} voiced={len(voiced)} median_f0_hz={median:.3f}")
    if not (198.0 <= median <= 242.0):
        raise AssertionError(f"RMVPE median f0 {median:.3f} Hz is outside +/-10% of 220 Hz")
    return median


def strict_load(model_path: Path, device: str) -> None:
    checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise AssertionError(f"checkpoint is not a dict: {model_path}")
    model = SynthesizerTrnMs768NSFsid(*checkpoint["config"], is_half=False)
    incompatible = model.load_state_dict(checkpoint["weight"], strict=True)
    missing = len(incompatible.missing_keys)
    unexpected = len(incompatible.unexpected_keys)
    if missing or unexpected:
        raise AssertionError(
            f"strict load failed: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.remove_weight_norm()
    model.eval().to(device)
    print(f"strict_load: tensors={len(checkpoint['weight'])} missing={missing} unexpected={unexpected}")


def synth_voice_like(sr: int = 16000, duration: float = 2.0) -> np.ndarray:
    n = int(sr * duration)
    t = np.arange(n, dtype=np.float32) / sr
    f0 = 150.0 * (1.0 + 0.02 * np.sin(2.0 * np.pi * 3.1 * t))
    phase = np.cumsum(f0, dtype=np.float64) / sr
    saw = (2.0 * (phase - np.floor(phase)) - 1.0).astype(np.float32)

    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    spectrum = np.fft.rfft(saw)
    envelope = (
        0.08
        + 1.00 * np.exp(-0.5 * ((freqs - 700.0) / 120.0) ** 2)
        + 0.75 * np.exp(-0.5 * ((freqs - 1220.0) / 160.0) ** 2)
        + 0.45 * np.exp(-0.5 * ((freqs - 2600.0) / 260.0) ** 2)
    )
    voice = np.fft.irfft(spectrum * envelope, n=n).astype(np.float32)
    amp = (0.72 + 0.28 * np.sin(2.0 * np.pi * 4.5 * t + 0.3)).astype(np.float32)
    voice *= amp
    peak = float(np.max(np.abs(voice)))
    if peak > 0:
        voice = voice / peak
    return (0.35 * voice).astype(np.float32)


def spectral_flatness(audio: np.ndarray) -> float:
    y = np.asarray(audio, dtype=np.float64).reshape(-1)
    if y.size == 0:
        return 0.0
    mag = np.abs(np.fft.rfft(y * np.hanning(y.size))) + 1e-12
    return float(np.exp(np.mean(np.log(mag))) / np.mean(mag))


def spectral_centroid(audio: np.ndarray, sr: int) -> float:
    y = np.asarray(audio, dtype=np.float64).reshape(-1)
    if y.size == 0:
        return 0.0
    mag = np.abs(np.fft.rfft(y * np.hanning(y.size))) + 1e-12
    freqs = np.fft.rfftfreq(y.size, 1.0 / sr)
    return float(np.sum(freqs * mag) / np.sum(mag))


def validate_output(
    label: str,
    audio: np.ndarray,
    sr: int,
    *,
    input_duration: float,
) -> float:
    if sr != 32000:
        raise AssertionError(f"{label}: expected output sr 32000, got {sr}")
    if not np.all(np.isfinite(audio)):
        raise AssertionError(f"{label}: output contains NaN or Inf")
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.000001:
        raise AssertionError(f"{label}: peak {peak:.6f} exceeds 1.0")
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    if rms <= 1e-5:
        raise AssertionError(f"{label}: output RMS is too close to silence ({rms:.8f})")
    duration = len(audio) / float(sr)
    if not (input_duration * 0.8 <= duration <= input_duration * 1.2):
        raise AssertionError(f"{label}: duration {duration:.3f}s is not within 20% of {input_duration:.3f}s")

    t = np.arange(len(audio), dtype=np.float32) / sr
    sine = np.sin(2.0 * np.pi * 150.0 * t).astype(np.float32)
    flat = spectral_flatness(audio)
    sine_flat = spectral_flatness(sine)
    print(
        f"{label}: sr={sr} duration_s={duration:.3f} peak={peak:.6f} "
        f"rms={rms:.6f} flatness={flat:.8f} sine_flatness={sine_flat:.8f}"
    )
    if flat <= max(sine_flat * 5.0, sine_flat + 1e-4):
        raise AssertionError(
            f"{label}: spectral flatness {flat:.8f} is too close to pure-sine baseline {sine_flat:.8f}"
        )
    return flat


def print_timings(label: str, timings: dict[str, float]) -> None:
    ordered = ", ".join(f"{key}={value:.3f}" for key, value in timings.items())
    print(f"{label} timings_ms: {ordered}")


def slot_for_model(model_path: Path) -> dict:
    index_files = sorted(model_path.parent.glob("*.index"))
    return {
        "id": model_path.parent.name or model_path.stem,
        "slot": model_path.parent.name or model_path.stem,
        "name": model_path.stem,
        "type": "RVC",
        "version": "v2",
        "sampling_rate": 32000,
        "f0": True,
        "has_index": bool(index_files),
        "size_bytes": model_path.stat().st_size,
        "source": "smoke",
        "path": str(model_path.parent),
        "model_path": str(model_path),
        "index_path": str(index_files[0]) if index_files else None,
    }


def main() -> int:
    args = parse_args()
    warnings.filterwarnings(
        "ignore",
        message="`torch.nn.utils.weight_norm` is deprecated.*",
        category=FutureWarning,
    )
    np.random.seed(1234)
    torch.manual_seed(1234)
    paths = asset_paths()
    rmvpe_sanity(paths["rmvpe"], args.device)
    strict_load(args.model, args.device)

    runtime_dir = ROOT / "data" / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    input_path = runtime_dir / "voice_smoke_input.wav"
    input_audio = synth_voice_like()
    input_sr = 16000
    input_duration = len(input_audio) / float(input_sr)
    sf.write(str(input_path), input_audio, input_sr, subtype="PCM_16")
    print(f"input: path={input_path} sr={input_sr} duration_s={input_duration:.3f}")

    loaded = pipeline.load_model(slot_for_model(args.model), paths, args.device)

    out0, sr0, timings0 = pipeline.convert(
        input_path,
        loaded,
        pitch=0,
        index_ratio=0.0,
        protect=0.5,
        f0_detector="rmvpe",
        device=args.device,
    )
    validate_output("convert_index_0", out0, sr0, input_duration=input_duration)
    print_timings("convert_index_0", timings0)

    out1, sr1, timings1 = pipeline.convert(
        input_path,
        loaded,
        pitch=0,
        index_ratio=0.5,
        protect=0.5,
        f0_detector="rmvpe",
        device=args.device,
    )
    validate_output("convert_index_0_5", out1, sr1, input_duration=input_duration)
    print_timings("convert_index_0_5", timings1)

    out_formant, sr_formant, timings_formant = pipeline.convert(
        input_path,
        loaded,
        pitch=0,
        index_ratio=0.0,
        protect=0.5,
        f0_detector="rmvpe",
        input_formant=args.formant,
        device=args.device,
    )
    validate_output("convert_formant", out_formant, sr_formant, input_duration=input_duration)
    print_timings("convert_formant", timings_formant)
    formant_duration = len(out_formant) / float(sr_formant)
    if not (input_duration * 0.95 <= formant_duration <= input_duration * 1.05):
        raise AssertionError(
            f"convert_formant: duration {formant_duration:.3f}s is not within 5% of {input_duration:.3f}s"
        )
    base_centroid = spectral_centroid(out0, sr0)
    formant_centroid = spectral_centroid(out_formant, sr_formant)
    print(
        f"formant_centroid: base_hz={base_centroid:.3f} "
        f"formant_{args.formant:+.2f}_hz={formant_centroid:.3f}"
    )
    if args.formant > 0 and formant_centroid <= base_centroid:
        raise AssertionError("positive formant did not raise converted spectral centroid")
    if args.formant < 0 and formant_centroid >= base_centroid:
        raise AssertionError("negative formant did not lower converted spectral centroid")

    out_path = args.out if args.out.is_absolute() else ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), out1, sr1, subtype="PCM_16")
    print(f"wrote: {out_path}")
    print("voice engine smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
