"""Offline native RVC conversion pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

from . import dsp
from .f0 import create_f0_extractor
from .features import ContentVec


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 3)


def _sample_rate(value: str | int | None) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        if value.endswith("k") and value[:-1].isdigit():
            return int(value[:-1]) * 1000
        try:
            return int(value)
        except ValueError:
            pass
    return 40000


def f0_to_coarse(f0):
    import numpy as np  # noqa: PLC0415

    f0 = np.asarray(f0, dtype=np.float32)
    f0_mel = 1127.0 * np.log1p(np.maximum(f0, 0.0) / 700.0)
    f0_mel_min = 1127.0 * np.log1p(50.0 / 700.0)
    f0_mel_max = 1127.0 * np.log1p(1100.0 / 700.0)
    voiced = f0_mel > 0
    f0_mel[voiced] = (f0_mel[voiced] - f0_mel_min) * 254.0 / (f0_mel_max - f0_mel_min) + 1.0
    f0_mel[f0_mel <= 1.0] = 1.0
    f0_mel[f0_mel > 255.0] = 255.0
    return np.rint(f0_mel).astype("int64")


def _resize_1d(values, target: int):
    import numpy as np  # noqa: PLC0415

    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == target:
        return arr
    if arr.size == 0:
        return np.zeros(target, dtype=np.float32)
    old = np.linspace(0.0, 1.0, num=arr.size, endpoint=True)
    new = np.linspace(0.0, 1.0, num=target, endpoint=True)
    return np.interp(new, old, arr).astype(np.float32)


def _upsample_features(features):
    import numpy as np  # noqa: PLC0415

    feats = np.asarray(features, dtype=np.float32)
    if feats.ndim != 2:
        raise ValueError(f"ContentVec output must be [T, 768], got {feats.shape}")
    target = max(1, feats.shape[0] * 2)
    old = np.linspace(0.0, 1.0, num=feats.shape[0], endpoint=True)
    new = np.linspace(0.0, 1.0, num=target, endpoint=True)
    out = np.empty((target, feats.shape[1]), dtype=np.float32)
    for dim in range(feats.shape[1]):
        out[:, dim] = np.interp(new, old, feats[:, dim])
    return out


def _index_mix(features, index_state: dict[str, Any] | None, index_ratio: float):
    if not index_state or index_ratio <= 0.0:
        return features
    import numpy as np  # noqa: PLC0415

    index = index_state["index"]
    big_npy = index_state["big_npy"]
    if big_npy is None:
        return features
    k = min(8, int(getattr(index, "ntotal", 0) or 0))
    if k <= 0:
        return features
    distances, indices = index.search(features.astype(np.float32), k)
    valid = indices >= 0
    safe_indices = np.where(valid, indices, 0)
    neighbors = big_npy[safe_indices]
    # Upstream RVC weighting: faiss returns squared-L2 distances and neighbors
    # are weighted by square(1/d), normalized. The previous exp(-d) collapsed
    # for typical ContentVec distances (d ~ 60: 74% of frames had every
    # neighbor clamped, averaging 8 spread-out vectors into ~zero), so at high
    # index_ratio the synthesizer got empty phones - audible as a pitch-only
    # "ahh" with no articulation. Measured on the real index: retrieved norm
    # 0.11 / cos 0.06 (old) vs 6.67 / cos 0.67 (this formula) for query
    # features of norm 7.4.
    safe_distances = np.asarray(distances, dtype=np.float32)
    safe_distances = np.where(np.isfinite(safe_distances), np.maximum(safe_distances, 1e-8), np.inf)
    weights = np.square(1.0 / safe_distances)
    weights = np.where(valid, weights, 0.0)
    denom = np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
    weights = weights / denom
    retrieved = (neighbors * weights[..., None]).sum(axis=1).astype(np.float32)
    return features * (1.0 - index_ratio) + retrieved * index_ratio


def _load_index(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    import faiss  # noqa: PLC0415

    index = faiss.read_index(str(path))
    count = int(getattr(index, "ntotal", 0) or 0)
    if count <= 0:
        return {"index": index, "big_npy": None}
    big_npy = index.reconstruct_n(0, count)
    return {"index": index, "big_npy": big_npy}


@dataclass
class LoadedVoiceModel:
    slot: dict[str, Any]
    synthesizer: Any
    version: str
    f0: bool
    sample_rate: int
    content_vec: ContentVec
    f0_model_path: Path
    index_state: dict[str, Any] | None


def load_model(slot: dict[str, Any], assets: dict[str, str], device: str) -> LoadedVoiceModel:
    import torch  # noqa: PLC0415

    model_path = Path(slot["model_path"])
    if model_path.suffix.lower() == ".pth":
        # Standard RVC checkpoints store a Python dict with `weight`, `config`,
        # `f0`, `version`, and `sr`. `weights_only=False` is required because the
        # outer dict is not a pure tensor state_dict in current torch releases.
        checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError("RVC checkpoint did not contain a dict")
        state = checkpoint.get("weight") or {}
        config = list(checkpoint.get("config") or [])
        f0 = bool(checkpoint.get("f0", 1))
        version = str(checkpoint.get("version") or "v1")
        sample_rate = _sample_rate(checkpoint.get("sr") or (config[-1] if config else None))
    elif model_path.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file  # noqa: PLC0415

        state = load_file(str(model_path), device="cpu")
        config = []
        f0 = bool(slot.get("f0", True))
        version = str(slot.get("version") or "v2")
        sample_rate = _sample_rate(slot.get("sampling_rate"))
    else:
        raise ValueError(f"unsupported RVC model file: {model_path.suffix}")

    if version != "v2":
        raise NotImplementedError("native voice engine P6R.1 supports RVC v2 checkpoints only")
    if not config:
        raise NotImplementedError("native voice engine real mode requires RVC checkpoint config metadata")

    from .rvc.models import SynthesizerTrnMs768NSFsid, SynthesizerTrnMs768NSFsid_nono  # noqa: PLC0415

    cls = SynthesizerTrnMs768NSFsid if f0 else SynthesizerTrnMs768NSFsid_nono
    synthesizer = cls(*config, is_half=False)
    incompatible = synthesizer.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "RVC checkpoint strict load failed: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    synthesizer.remove_weight_norm()
    synthesizer.eval().to(device)

    return LoadedVoiceModel(
        slot=slot,
        synthesizer=synthesizer,
        version=version,
        f0=f0,
        sample_rate=sample_rate,
        content_vec=ContentVec(Path(assets["content_vec"])),
        f0_model_path=Path(assets["rmvpe"]),
        index_state=_load_index(slot.get("index_path")),
    )


def convert(
    input_path: Path,
    loaded: LoadedVoiceModel,
    *,
    pitch: int,
    index_ratio: float,
    protect: float,
    f0_detector: str,
    device: str,
    input_highpass_hz: int = dsp.DEFAULT_INPUT_HIGHPASS_HZ,
    input_gate_db: float = dsp.DEFAULT_INPUT_GATE_DB,
    input_formant: float = dsp.DEFAULT_INPUT_FORMANT,
):
    import numpy as np  # noqa: PLC0415
    import soundfile as sf  # noqa: PLC0415
    import soxr  # noqa: PLC0415

    timings: dict[str, float] = {}

    stage = time.perf_counter()
    audio, sr = sf.read(str(input_path), dtype="float32", always_2d=True)
    mono = np.mean(audio, axis=1).astype(np.float32)
    timings["load_audio"] = _ms(stage)

    stage = time.perf_counter()
    audio_16k = mono if int(sr) == 16000 else soxr.resample(mono, int(sr), 16000).astype(np.float32)
    timings["resample_16k"] = _ms(stage)

    out, out_sr, core_timings = convert_audio(
        audio_16k,
        loaded,
        pitch=pitch,
        index_ratio=index_ratio,
        protect=protect,
        f0_detector=f0_detector,
        input_highpass_hz=input_highpass_hz,
        input_gate_db=input_gate_db,
        input_formant=input_formant,
        device=device,
    )
    timings.update(core_timings)
    return out, out_sr, timings


def convert_audio(
    audio_16k,
    loaded: LoadedVoiceModel,
    *,
    pitch: int,
    index_ratio: float,
    protect: float,
    f0_detector: str,
    device: str,
    input_highpass_hz: int = dsp.DEFAULT_INPUT_HIGHPASS_HZ,
    input_gate_db: float = dsp.DEFAULT_INPUT_GATE_DB,
    input_formant: float = dsp.DEFAULT_INPUT_FORMANT,
):
    """Shared conversion core: 16 kHz mono float32 array -> (audio @ model sr,
    sr, per-stage timings). The offline file path and the realtime chunk
    processor both run through here so their outputs can never diverge."""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    timings: dict[str, float] = {}

    stage = time.perf_counter()
    analysis_audio, formant_factor = dsp.process_input(
        audio_16k,
        input_highpass_hz=input_highpass_hz,
        input_gate_db=input_gate_db,
        input_formant=input_formant,
    )
    timings["input_dsp"] = _ms(stage)

    stage = time.perf_counter()
    base_features = loaded.content_vec.extract(analysis_audio)
    timings["content_vec"] = _ms(stage)

    stage = time.perf_counter()
    mixed_features = _index_mix(base_features, loaded.index_state, float(index_ratio))
    timings["index_mix"] = _ms(stage)

    stage = time.perf_counter()
    features = _upsample_features(mixed_features)
    protected_features = _upsample_features(base_features)
    timings["feature_upsample"] = _ms(stage)

    f0 = None
    pitch_coarse = None
    stage = time.perf_counter()
    if loaded.f0:
        extractor = create_f0_extractor(f0_detector, loaded.f0_model_path, device)
        f0 = extractor.compute(analysis_audio, sr=16000)
        f0 = _resize_1d(f0, features.shape[0])
        f0 = dsp.compensate_f0_for_input_formant(f0, formant_factor)
        if pitch:
            f0 = f0 * (2.0 ** (float(pitch) / 12.0))
        pitch_coarse = f0_to_coarse(f0)
    timings["f0"] = _ms(stage)

    stage = time.perf_counter()
    if loaded.f0 and protect < 0.5 and f0 is not None:
        pitchff = np.where(f0 > 0.0, 1.0, float(protect)).astype(np.float32)[:, None]
        features = features * pitchff + protected_features * (1.0 - pitchff)
    timings["protect"] = _ms(stage)

    stage = time.perf_counter()
    phone = torch.from_numpy(features[None, :, :]).to(device=device, dtype=torch.float32)
    phone_lengths = torch.tensor([features.shape[0]], device=device, dtype=torch.long)
    sid = torch.tensor([0], device=device, dtype=torch.long)
    with torch.no_grad():
        if loaded.f0:
            assert pitch_coarse is not None and f0 is not None
            pitch_tensor = torch.from_numpy(pitch_coarse[None, :]).to(device=device, dtype=torch.long)
            f0_tensor = torch.from_numpy(f0[None, :]).to(device=device, dtype=torch.float32)
            output = loaded.synthesizer.infer(
                phone,
                phone_lengths,
                pitch_tensor,
                f0_tensor,
                sid,
            )[0]
        else:
            output = loaded.synthesizer.infer(phone, phone_lengths, sid)[0]
    timings["synth"] = _ms(stage)

    stage = time.perf_counter()
    out = output.detach().cpu().numpy().reshape(-1).astype(np.float32)
    out = dsp.compensate_output_duration_for_input_formant(
        out,
        formant_factor,
        sample_rate=loaded.sample_rate,
    )
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1.0:
        out = out / peak
    timings["postprocess"] = _ms(stage)
    return out, loaded.sample_rate, timings
