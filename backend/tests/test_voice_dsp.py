from __future__ import annotations

import math

from app.services.voice_engine import dsp


def _rms(x) -> float:
    import numpy as np

    arr = np.asarray(x, dtype=np.float32)
    return float(np.sqrt(np.mean(np.square(arr)))) if arr.size else 0.0


def _db_ratio(after, before) -> float:
    return 20.0 * math.log10(max(_rms(after), 1e-12) / max(_rms(before), 1e-12))


def _sine(freq: float, *, duration: float = 1.0, sr: int = 16_000, rms_db: float = -12.0):
    import numpy as np

    t = np.arange(int(duration * sr), dtype=np.float32) / sr
    amp = (10.0 ** (rms_db / 20.0)) * math.sqrt(2.0)
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def _centroid(x, *, sr: int = 16_000) -> float:
    import numpy as np

    arr = np.asarray(x, dtype=np.float32)
    window = np.hanning(arr.size).astype(np.float32)
    mag = np.abs(np.fft.rfft(arr * window)) + 1e-12
    freqs = np.fft.rfftfreq(arr.size, 1.0 / sr)
    return float(np.sum(freqs * mag) / np.sum(mag))


def test_highpass_attenuates_low_sine_and_preserves_high_sine():
    import numpy as np

    low = _sine(50.0)
    high = _sine(1000.0)

    low_out = dsp.apply_highpass(low, 80)
    high_out = dsp.apply_highpass(high, 80)

    assert len(low_out) == len(low)
    assert _db_ratio(low_out, low) <= -20.0
    assert abs(_db_ratio(high_out, high)) <= 1.0
    assert np.allclose(dsp.apply_highpass(high, 0), high)


def test_noise_gate_drops_quiet_noise_and_keeps_tone():
    import numpy as np

    rng = np.random.default_rng(1234)
    noise = rng.normal(0.0, 1.0, 8000).astype(np.float32)
    noise *= (10.0 ** (-70.0 / 20.0)) / max(_rms(noise), 1e-12)
    tone = _sine(440.0, duration=0.5, rms_db=-20.0)
    signal = np.concatenate([noise, tone]).astype(np.float32)

    gated = dsp.apply_noise_gate(signal, -60)

    assert _db_ratio(gated[:8000], signal[:8000]) <= -30.0
    assert abs(_db_ratio(gated[8000:], signal[8000:])) <= 1.0
    assert np.allclose(dsp.apply_noise_gate(signal, -90), signal)


def test_formant_shift_moves_centroid_and_compensates_f0():
    import numpy as np

    sr = 16_000
    t = np.arange(sr, dtype=np.float32) / sr
    harmonic = np.zeros_like(t)
    for i in range(1, 20):
        harmonic += (1.0 / i) * np.sin(2.0 * np.pi * 120.0 * i * t)
    harmonic = (0.2 * harmonic / np.max(np.abs(harmonic))).astype(np.float32)

    neutral, r0 = dsp.apply_input_formant(harmonic, 0.0)
    bright, r_up = dsp.apply_input_formant(harmonic, 1.0)
    dark, r_down = dsp.apply_input_formant(harmonic, -1.0)

    assert r0 == 1.0
    assert _centroid(bright) > _centroid(neutral)
    assert _centroid(dark) < _centroid(neutral)

    f0 = np.array([0.0, 110.0, 220.0], dtype=np.float32)
    assert np.allclose(dsp.compensate_f0_for_input_formant(f0, r_up), f0 / r_up)
    assert r_down < 1.0 < r_up
