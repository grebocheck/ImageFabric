"""Input-side DSP helpers for native voice conversion.

All functions operate on mono float32-ish arrays. Heavy dependencies are loaded
inside functions so importing the voice engine stays cheap in stub/API tests.
"""

from __future__ import annotations

INPUT_SAMPLE_RATE = 16_000
DEFAULT_INPUT_HIGHPASS_HZ = 80
DEFAULT_INPUT_GATE_DB = -60.0
DEFAULT_INPUT_FORMANT = 0.0
GATE_OFF_DB = -90.0


def clamp_input_highpass_hz(value: object) -> int:
    """Clamp high-pass cutoff to 0..300 Hz. 0 or "off" disables the filter."""
    if isinstance(value, str) and value.strip().lower() == "off":
        return 0
    return max(0, min(300, int(float(value))))


def clamp_input_gate_db(value: object) -> float:
    """Clamp gate threshold to -90..-20 dBFS.

    -90 dB is treated as the off position. A legacy/UI value of 0 or the string
    "off" also maps to -90 so callers can disable the gate without remembering
    the exact sentinel.
    """
    if isinstance(value, str) and value.strip().lower() == "off":
        return GATE_OFF_DB
    n = float(value)
    if n == 0.0:
        return GATE_OFF_DB
    return max(GATE_OFF_DB, min(-20.0, n))


def clamp_input_formant(value: object) -> float:
    """Clamp input-side formant/brightness shift to +/-2 semitone-like units."""
    return max(-2.0, min(2.0, float(value)))


def input_formant_factor(input_formant: float) -> float:
    return float(2.0 ** (clamp_input_formant(input_formant) / 12.0))


def compensate_f0_for_input_formant(f0, factor: float):
    import numpy as np  # noqa: PLC0415

    arr = np.asarray(f0, dtype=np.float32)
    if factor <= 0.0 or abs(factor - 1.0) < 1e-6:
        return arr
    return (arr / float(factor)).astype(np.float32)


def _as_float32(audio):
    import numpy as np  # noqa: PLC0415

    return np.asarray(audio, dtype=np.float32).reshape(-1)


def apply_highpass(audio_16k, cutoff_hz: object = DEFAULT_INPUT_HIGHPASS_HZ, *, sample_rate: int = INPUT_SAMPLE_RATE):
    """FFT-domain high-pass with a raised-cosine transition below cutoff."""
    import numpy as np  # noqa: PLC0415

    audio = _as_float32(audio_16k)
    cutoff = clamp_input_highpass_hz(cutoff_hz)
    if cutoff <= 0 or audio.size == 0:
        return audio.astype(np.float32, copy=True)

    n = int(audio.size)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sample_rate))
    spectrum = np.fft.rfft(audio)

    gain = np.ones_like(freqs, dtype=np.float32)
    stop_hz = float(cutoff) * 0.75
    gain[freqs <= stop_hz] = 0.0
    transition = (freqs > stop_hz) & (freqs < float(cutoff))
    if np.any(transition):
        x = (freqs[transition] - stop_hz) / max(float(cutoff) - stop_hz, 1e-6)
        gain[transition] = (0.5 - 0.5 * np.cos(np.pi * x)).astype(np.float32)

    return np.fft.irfft(spectrum * gain, n=n).astype(np.float32)


def apply_noise_gate(
    audio_16k,
    gate_db: object = DEFAULT_INPUT_GATE_DB,
    *,
    sample_rate: int = INPUT_SAMPLE_RATE,
):
    """Frame-RMS noise gate with attack/release smoothing.

    The off position is -90 dBFS (or 0/"off" through the clamp helper). Closed
    gain is -80 dB rather than zero to avoid hard pumping artifacts.
    """
    import numpy as np  # noqa: PLC0415

    audio = _as_float32(audio_16k)
    threshold = clamp_input_gate_db(gate_db)
    if threshold <= GATE_OFF_DB or audio.size == 0:
        return audio.astype(np.float32, copy=True)

    frame = max(1, int(round(float(sample_rate) * 0.010)))
    frame_count = int(np.ceil(audio.size / frame))
    padded = np.pad(audio, (0, frame_count * frame - audio.size))
    frames = padded.reshape(frame_count, frame)
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float32), axis=1))
    db = 20.0 * np.log10(np.maximum(rms, 1e-12))

    floor = np.float32(10.0 ** (-80.0 / 20.0))
    target = np.where(db >= threshold, 1.0, floor).astype(np.float32)

    env = np.empty_like(target)
    env[0] = target[0]
    frame_s = frame / float(sample_rate)
    attack = 1.0 - np.exp(-frame_s / 0.005)
    release = 1.0 - np.exp(-frame_s / 0.120)
    for i in range(1, frame_count):
        coeff = attack if target[i] > env[i - 1] else release
        env[i] = env[i - 1] + np.float32(coeff) * (target[i] - env[i - 1])

    frame_pos = np.arange(frame_count, dtype=np.float32) * frame + (frame - 1) / 2.0
    sample_pos = np.arange(audio.size, dtype=np.float32)
    gain = np.interp(sample_pos, frame_pos, env, left=float(env[0]), right=float(env[-1])).astype(np.float32)
    return (audio * gain).astype(np.float32)


def apply_input_formant(
    audio_16k,
    input_formant: object = DEFAULT_INPUT_FORMANT,
    *,
    sample_rate: int = INPUT_SAMPLE_RATE,
) -> tuple[object, float]:
    """Shift the analysis signal's formant/brightness by resampling.

    Positive values are brighter: the analysis signal is resampled to
    ``sample_rate / factor`` and then interpreted by ContentVec/RMVPE as 16 kHz,
    raising the measured spectrum by ``factor``. The downstream f0 path divides
    by the same factor so pitch transpose remains the dedicated pitch control.
    """
    import numpy as np  # noqa: PLC0415

    audio = _as_float32(audio_16k)
    factor = input_formant_factor(clamp_input_formant(input_formant))
    if abs(factor - 1.0) < 1e-6 or audio.size == 0:
        return audio.astype(np.float32, copy=True), 1.0

    import soxr  # noqa: PLC0415

    shifted = soxr.resample(audio, float(sample_rate), float(sample_rate) / factor).astype(np.float32)
    return shifted, factor


def compensate_output_duration_for_input_formant(
    audio,
    factor: float,
    *,
    sample_rate: int,
):
    """Undo the analysis-path duration change before offline return/realtime tailing.

    Keeping this as the single output-length correction point means the realtime
    ``ChunkProcessor`` can continue to take the tail matching the current chunk,
    while offline conversion returns roughly the same duration as the source.
    """
    import numpy as np  # noqa: PLC0415

    out = np.asarray(audio, dtype=np.float32).reshape(-1)
    if abs(float(factor) - 1.0) < 1e-6 or out.size == 0:
        return out.astype(np.float32, copy=True)

    import soxr  # noqa: PLC0415

    return soxr.resample(out, float(sample_rate), float(sample_rate) * float(factor)).astype(np.float32)


def process_input(
    audio_16k,
    *,
    input_highpass_hz: object = DEFAULT_INPUT_HIGHPASS_HZ,
    input_gate_db: object = DEFAULT_INPUT_GATE_DB,
    input_formant: object = DEFAULT_INPUT_FORMANT,
) -> tuple[object, float]:
    """Apply input clean-up and analysis-side character shift."""
    cleaned = apply_highpass(audio_16k, input_highpass_hz)
    cleaned = apply_noise_gate(cleaned, input_gate_db)
    return apply_input_formant(cleaned, input_formant)
