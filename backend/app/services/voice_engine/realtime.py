"""Native realtime voice session (P6R.2).

A live session owns a sounddevice duplex stream: the PortAudio callback only
moves samples in/out of ring buffers; a dedicated worker thread pulls fixed
chunks, re-converts a sliding context window through the shared
``pipeline.convert_audio`` core, and stitches chunk seams with a small
SOLA-style alignment + equal-power crossfade so they don't click.

Threading model: the audio callback and the worker communicate through
``_InputRing`` / ``_OutputRing`` (lock-guarded deques of float32 samples);
settings are read from the ``VoiceEngine`` once per chunk (plain attribute
reads — atomic enough in CPython for floats/ints), so pitch/index/protect/
gain/pass-through changes apply on the next chunk without a restart.

STUB mode never imports sounddevice/torch: ``StubRealtimeSession`` just flips
``live`` and synthesizes deterministic VU/timing values so the API, the UI,
and the voice-lane parking are all testable in CI.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any

from ...config import settings

if TYPE_CHECKING:
    from .engine import VoiceEngine

# w-okada convention kept for UI parity: read chunk = N * 128 samples.
CHUNK_UNIT_SAMPLES = 128
# The conversion context must stay bounded no matter what extra_convert_size
# says, or a long setting would make every chunk arbitrarily expensive.
MAX_CONTEXT_SECONDS = 8.0
# SOLA: how far (in samples @ stream sr) we search for the best seam alignment.
SOLA_SEARCH_MS = 10.0


class _Ring:
    """A minimal lock-guarded float32 sample FIFO (numpy-backed)."""

    def __init__(self) -> None:
        import numpy as np  # noqa: PLC0415

        self._np = np
        self._buf = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()

    def push(self, samples) -> None:
        with self._lock:
            self._buf = self._np.concatenate([self._buf, samples.astype(self._np.float32, copy=False)])

    def pull(self, count: int):
        """Take exactly ``count`` samples; missing samples are zero-padded.
        Returns (samples, missing_count)."""
        with self._lock:
            have = len(self._buf)
            if have >= count:
                out, self._buf = self._buf[:count], self._buf[count:]
                return out, 0
            out = self._np.concatenate([self._buf, self._np.zeros(count - have, dtype=self._np.float32)])
            self._buf = self._buf[:0]
            return out, count - have

    def available(self) -> int:
        with self._lock:
            return len(self._buf)

    def drop_to(self, max_samples: int) -> int:
        """Bound the queue (drop oldest); returns how many were dropped."""
        with self._lock:
            extra = len(self._buf) - max_samples
            if extra > 0:
                self._buf = self._buf[extra:]
                return extra
            return 0


class ChunkProcessor:
    """The realtime conversion core, audio-device-free so the bench script and
    the live session run the *same* code: keeps a rolling input context,
    converts context + chunk, and SOLA-crossfades the seam with the previous
    output tail."""

    def __init__(self, engine: VoiceEngine, loaded: Any, stream_sr: int) -> None:
        import numpy as np  # noqa: PLC0415

        self._np = np
        self._engine = engine
        self._loaded = loaded
        self.stream_sr = int(stream_sr)
        self._context_16k = np.zeros(0, dtype=np.float32)
        self._prev_tail = np.zeros(0, dtype=np.float32)  # @ stream sr
        self.last_timings: dict[str, float] = {}

    def _crossfade_samples(self) -> int:
        return max(0, int(float(self._engine.cross_fade_overlap_size) * self.stream_sr))

    def _context_limit_16k(self) -> int:
        extra = min(float(self._engine.extra_convert_size), MAX_CONTEXT_SECONDS)
        return max(CHUNK_UNIT_SAMPLES, int(extra * 16000))

    def process(self, chunk):
        """Convert one chunk (float32 @ stream sr) -> same-length output."""
        import soxr  # noqa: PLC0415

        np = self._np
        started = time.perf_counter()
        timings: dict[str, float] = {}

        stage = time.perf_counter()
        chunk_16k = soxr.resample(chunk, self.stream_sr, 16000).astype(np.float32)
        self._context_16k = np.concatenate([self._context_16k, chunk_16k])
        limit = self._context_limit_16k()
        if len(self._context_16k) > limit:
            self._context_16k = self._context_16k[-limit:]
        timings["resample_16k"] = round((time.perf_counter() - stage) * 1000, 3)

        from . import pipeline  # noqa: PLC0415

        # Input cleanup/formant DSP runs inside convert_audio on the full
        # rolling context, not on only the new raw chunk, so the converted tail
        # used for stitching is derived from the same processed history.
        converted, model_sr, core_timings = pipeline.convert_audio(
            self._context_16k,
            self._loaded,
            pitch=self._engine.pitch,
            index_ratio=self._engine.index_ratio,
            protect=self._engine.protect,
            f0_detector=self._engine.f0_detector,
            input_highpass_hz=self._engine.input_highpass_hz,
            input_gate_db=self._engine.input_gate_db,
            input_formant=self._engine.input_formant,
            device=self._engine.device,
        )
        timings.update(core_timings)

        stage = time.perf_counter()
        out_stream = soxr.resample(converted, model_sr, self.stream_sr).astype(np.float32)
        fade = self._crossfade_samples()
        need = len(chunk) + fade
        tail = out_stream[-need:] if len(out_stream) >= need else np.concatenate(
            [np.zeros(need - len(out_stream), dtype=np.float32), out_stream]
        )

        if fade > 0 and len(self._prev_tail) >= fade:
            head = self._sola_align(tail, self._prev_tail[-fade:], fade)
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            blended = self._prev_tail[-fade:] * np.sqrt(1.0 - ramp) + head[:fade] * np.sqrt(ramp)
            out = np.concatenate([blended, head[fade : fade + len(chunk) - fade]]) if len(chunk) > fade else blended[: len(chunk)]
            if len(out) < len(chunk):
                out = np.concatenate([out, np.zeros(len(chunk) - len(out), dtype=np.float32)])
        else:
            out = tail[-len(chunk):]
        self._prev_tail = tail
        timings["stitch"] = round((time.perf_counter() - stage) * 1000, 3)
        timings["total"] = round((time.perf_counter() - started) * 1000, 3)
        self.last_timings = timings
        return out.astype(np.float32, copy=False)

    def _sola_align(self, tail, prev_fade, fade: int):
        """Shift the new tail by up to SOLA_SEARCH_MS to maximize correlation
        with the previous fade region, so the crossfade adds in phase."""
        np = self._np
        search = min(int(self.stream_sr * SOLA_SEARCH_MS / 1000.0), max(0, len(tail) - fade - 1))
        if search <= 0 or fade <= 0:
            return tail
        best_offset = 0
        best_score = -np.inf
        target = prev_fade
        norm_t = float(np.sqrt(np.dot(target, target)) + 1e-8)
        for offset in range(0, search + 1):
            cand = tail[offset : offset + fade]
            score = float(np.dot(cand, target)) / (norm_t * float(np.sqrt(np.dot(cand, cand)) + 1e-8))
            if score > best_score:
                best_score = score
                best_offset = offset
        return tail[best_offset:]


def _rms(samples) -> float:
    import numpy as np  # noqa: PLC0415

    if len(samples) == 0:
        return 0.0
    return float(min(1.0, np.sqrt(np.mean(np.square(samples))) * 4.0))


class RealtimeSession:
    """Owns the duplex stream + worker thread for one live voice session."""

    def __init__(self, engine: VoiceEngine) -> None:
        self._engine = engine
        self._stream = None
        self._monitor_stream = None
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()
        self._metrics_lock = threading.Lock()
        self._metrics: dict[str, Any] = {
            "input_vu": 0.0,
            "output_vu": 0.0,
            "timings_ms": {},
            "total_ms": None,
            "chunk_ms": None,
            "overruns": 0,
            "underruns": 0,
        }
        self._input_ring: _Ring | None = None
        self._output_ring: _Ring | None = None
        self._monitor_ring: _Ring | None = None
        self._processor: ChunkProcessor | None = None
        self.stream_sr = 48000
        self.chunk_samples = 0
        self.error: str | None = None

    # ------------------------------------------------------------ lifecycle
    def start(self, model_id: str) -> None:
        import sounddevice as sd  # noqa: PLC0415

        engine = self._engine
        loaded = engine.load_model_sync(model_id)
        self.stream_sr = int(engine.server_audio_sample_rate)
        self.chunk_samples = max(1, int(engine.server_read_chunk_size)) * CHUNK_UNIT_SAMPLES
        self._input_ring = _Ring()
        self._output_ring = _Ring()
        self._processor = ChunkProcessor(engine, loaded, self.stream_sr)
        with self._metrics_lock:
            self._metrics["chunk_ms"] = round(self.chunk_samples / self.stream_sr * 1000, 1)

        in_dev = engine.server_input_device_id
        out_dev = engine.server_output_device_id
        mon_dev = engine.server_monitor_device_id

        def callback(indata, outdata, frames, time_info, status) -> None:  # noqa: ARG001
            import numpy as np  # noqa: PLC0415

            assert self._input_ring is not None and self._output_ring is not None
            self._input_ring.push(indata[:, 0])
            # Bound the input queue to ~2s so a stalled worker degrades
            # (drops old audio) instead of growing without limit.
            dropped = self._input_ring.drop_to(self.stream_sr * 2)
            samples, missing = self._output_ring.pull(frames)
            if missing or dropped:
                with self._metrics_lock:
                    self._metrics["underruns"] += 1 if missing else 0
                    self._metrics["overruns"] += 1 if dropped else 0
            outdata[:] = (samples * float(self._engine.server_output_gain)).reshape(-1, 1).astype(np.float32)

        self._stop.clear()
        self._stream = sd.Stream(
            samplerate=self.stream_sr,
            blocksize=0,
            channels=1,
            dtype="float32",
            device=(
                in_dev if in_dev is not None and in_dev >= 0 else None,
                out_dev if out_dev is not None and out_dev >= 0 else None,
            ),
            callback=callback,
        )
        if mon_dev is not None and mon_dev >= 0 and mon_dev != out_dev:
            self._monitor_ring = _Ring()
            self._monitor_stream = sd.OutputStream(
                samplerate=self.stream_sr,
                blocksize=0,
                channels=1,
                dtype="float32",
                device=mon_dev,
                callback=self._monitor_callback,
            )
        self._worker = threading.Thread(target=self._run, name="hfabric-voice-rt", daemon=True)
        self._stream.start()
        if self._monitor_stream is not None:
            self._monitor_stream.start()
        self._worker.start()

    def _monitor_callback(self, outdata, frames, time_info, status) -> None:  # noqa: ARG002
        import numpy as np  # noqa: PLC0415

        assert self._monitor_ring is not None
        samples, _missing = self._monitor_ring.pull(frames)
        outdata[:] = (samples * float(self._engine.server_monitor_gain)).reshape(-1, 1).astype(np.float32)

    def stop(self) -> None:
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=5.0)
            self._worker = None
        for stream in (self._stream, self._monitor_stream):
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:  # noqa: BLE001 - device teardown must not raise
                    pass
        self._stream = None
        self._monitor_stream = None
        self._processor = None
        if not settings.stub_mode:
            try:
                import torch  # noqa: PLC0415

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass

    # --------------------------------------------------------------- worker
    def _run(self) -> None:
        import numpy as np  # noqa: PLC0415

        assert self._input_ring is not None and self._output_ring is not None
        assert self._processor is not None
        wait_s = max(0.001, self.chunk_samples / self.stream_sr / 8)
        while not self._stop.is_set():
            if self._input_ring.available() < self.chunk_samples:
                time.sleep(wait_s)
                continue
            chunk, _ = self._input_ring.pull(self.chunk_samples)
            chunk = chunk * float(self._engine.server_input_gain)
            in_vu = _rms(chunk)
            try:
                if self._engine.pass_through:
                    out = chunk.astype(np.float32, copy=False)
                    timings: dict[str, float] = {"pass_through": 0.0, "total": 0.0}
                else:
                    out = self._processor.process(chunk)
                    timings = dict(self._processor.last_timings)
            except Exception as exc:  # noqa: BLE001 - keep the stream alive, surface the error
                self.error = repr(exc)
                out = np.zeros_like(chunk)
                timings = {"error": 0.0}
            self._output_ring.push(out)
            if self._monitor_ring is not None:
                self._monitor_ring.push(out)
            total = timings.get("total")
            with self._metrics_lock:
                self._metrics["input_vu"] = in_vu
                self._metrics["output_vu"] = _rms(out)
                self._metrics["timings_ms"] = timings
                self._metrics["total_ms"] = total

    def metrics(self) -> dict[str, Any]:
        with self._metrics_lock:
            return dict(self._metrics)


class StubRealtimeSession:
    """CI stand-in: no audio devices, deterministic moving metrics."""

    def __init__(self, engine: VoiceEngine) -> None:
        self._engine = engine
        self._started = time.monotonic()
        self.error: str | None = None

    def start(self, model_id: str) -> None:  # noqa: ARG002
        self._started = time.monotonic()

    def stop(self) -> None:
        return None

    def metrics(self) -> dict[str, Any]:
        tick = int((time.monotonic() - self._started) * 4)
        chunk_ms = round(int(self._engine.server_read_chunk_size) * CHUNK_UNIT_SAMPLES
                         / int(self._engine.server_audio_sample_rate) * 1000, 1)
        return {
            "input_vu": ((tick % 10) + 1) / 10.0,
            "output_vu": ((tick % 7) + 1) / 10.0,
            "timings_ms": {"stub": 1.0, "total": 5.0},
            "total_ms": 5.0,
            "chunk_ms": chunk_ms,
            "overruns": 0,
            "underruns": 0,
        }


_SESSION: RealtimeSession | StubRealtimeSession | None = None
_SESSION_LOCK = threading.Lock()


def session_active() -> bool:
    return _SESSION is not None


def current_session() -> RealtimeSession | StubRealtimeSession | None:
    return _SESSION


def start_session(engine: VoiceEngine, model_id: str) -> None:
    """Create + start the singleton session. Raises on failure (no half-open
    session is left behind)."""
    global _SESSION
    with _SESSION_LOCK:
        if _SESSION is not None:
            raise RuntimeError("a voice session is already live")
        session: RealtimeSession | StubRealtimeSession
        session = StubRealtimeSession(engine) if settings.stub_mode else RealtimeSession(engine)
        try:
            session.start(model_id)
        except Exception:
            session.stop()
            raise
        _SESSION = session


def stop_session() -> bool:
    global _SESSION
    with _SESSION_LOCK:
        if _SESSION is None:
            return False
        try:
            _SESSION.stop()
        finally:
            _SESSION = None
    return True
