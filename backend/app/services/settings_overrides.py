"""Writable runtime settings overrides for local user-facing knobs.

Server binding and authentication stay environment-only. The rest of the
day-to-day tuning surface is described here so the Settings tab can render
typed controls and persist values to ``data/settings-overrides.json``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal

from ..config import CONTEXT_TYPES, LLAMA_BACKENDS, settings

SettingKind = Literal["boolean", "integer", "number", "text", "choice", "path"]
SettingValue = int | float | bool | str | None


@dataclass(frozen=True)
class SettingSpec:
    key: str
    label: str
    group: str
    kind: SettingKind
    description: str = ""
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    multiple_of: int | None = None
    choices: tuple[tuple[str, str], ...] = ()
    nullable: bool = False
    restart_required: bool = False

    def payload(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "key": self.key,
            "label": self.label,
            "group": self.group,
            "kind": self.kind,
            "description": self.description,
            "restart_required": self.restart_required,
        }
        if self.minimum is not None:
            out["min"] = self.minimum
        if self.maximum is not None:
            out["max"] = self.maximum
        if self.step is not None:
            out["step"] = self.step
        if self.multiple_of is not None:
            out["multiple_of"] = self.multiple_of
        if self.choices:
            out["choices"] = [{"value": value, "label": label} for value, label in self.choices]
        if self.nullable:
            out["nullable"] = True
        return out


GROUPS: tuple[dict[str, str], ...] = (
    {
        "id": "runtime",
        "label": "Runtime mode",
        "description": "Global execution mode and startup-level behavior.",
    },
    {
        "id": "paths",
        "label": "Model and binary paths",
        "description": "Local model folders and llama.cpp executable locations.",
    },
    {
        "id": "llm",
        "label": "LLM runtime",
        "description": "llama.cpp launch defaults for chat, code, embeddings, and multimodal tools.",
    },
    {
        "id": "image_defaults",
        "label": "Image defaults",
        "description": "Composer defaults used when creating new image jobs.",
    },
    {
        "id": "family_defaults",
        "label": "Image model families",
        "description": "Family-specific defaults for FLUX.2, Qwen-Image, and Z-Image.",
    },
    {
        "id": "acceleration",
        "label": "Acceleration",
        "description": "Attention, compile, cache, cleanup, and LoRA runtime tuning.",
    },
    {
        "id": "memory",
        "label": "Memory policy",
        "description": "RAM guards, keep-warm behavior, and learned profile margins.",
    },
    {
        "id": "tools",
        "label": "Speech, RAG, vision",
        "description": "CPU/GPU placement and upload limits for local tools.",
    },
    {
        "id": "voice",
        "label": "Voice defaults",
        "description": "Defaults used by the native voice engine before per-session changes.",
    },
    {
        "id": "sources",
        "label": "Advanced model sources",
        "description": "Repo IDs and base folders used by specialized image pipelines.",
    },
)


def _choices(values: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    return tuple((value, value) for value in values)


def _llama_backend_choices() -> tuple[tuple[str, str], ...]:
    return tuple((key, str(value["label"])) for key, value in LLAMA_BACKENDS.items())


def _context_type_choices() -> tuple[tuple[str, str], ...]:
    return tuple((key, str(value["label"])) for key, value in CONTEXT_TYPES.items())


SPECS: tuple[SettingSpec, ...] = (
    # Runtime
    SettingSpec("stub_mode", "Stub mode", "runtime", "boolean", "Run without heavy GPU/ML backends.", restart_required=True),

    # Paths
    SettingSpec("image_models_dir", "Image models", "paths", "path", restart_required=True),
    SettingSpec("lora_models_dir", "LoRA models", "paths", "path", restart_required=True),
    SettingSpec("llm_models_dir", "LLM models", "paths", "path", restart_required=True),
    SettingSpec("tts_models_dir", "TTS models", "paths", "path", restart_required=True),
    SettingSpec("transcription_models_dir", "Transcription models", "paths", "path", restart_required=True),
    SettingSpec("embed_models_dir", "Embedding models", "paths", "path", restart_required=True),
    SettingSpec("vision_models_dir", "Vision models", "paths", "path", restart_required=True),
    SettingSpec("voice_models_dir", "Voice models", "paths", "path", restart_required=True),
    SettingSpec("voice_pretrain_dir", "Voice pretrain assets", "paths", "path", restart_required=True),
    SettingSpec("llama_server_bin", "llama-server", "paths", "path", restart_required=True),
    SettingSpec("llama_server_bin_turbo", "Turbo llama-server", "paths", "path", restart_required=True),
    SettingSpec("llama_tts_bin", "llama-tts", "paths", "path", restart_required=True),
    SettingSpec("llama_mtmd_bin", "llama-mtmd-cli", "paths", "path", restart_required=True),

    # LLM runtime
    SettingSpec("llama_host", "llama.cpp host", "llm", "text", restart_required=True),
    SettingSpec("llama_port", "Chat port", "llm", "integer", minimum=1, maximum=65535, step=1, restart_required=True),
    SettingSpec("llama_embed_port", "Embedding port", "llm", "integer", minimum=1, maximum=65535, step=1, restart_required=True),
    SettingSpec("llama_ngl", "GPU layers", "llm", "integer", "999 means full offload.", minimum=0, maximum=999, step=1),
    SettingSpec("llama_ctx", "Context tokens", "llm", "integer", minimum=512, maximum=262144, step=512),
    SettingSpec("llama_backend", "Backend build", "llm", "choice", choices=_llama_backend_choices()),
    SettingSpec("llama_context_type", "KV cache type", "llm", "choice", choices=_context_type_choices()),

    # Image defaults
    SettingSpec("default_steps", "Steps", "image_defaults", "integer", minimum=1, maximum=150, step=1),
    SettingSpec("default_guidance", "Guidance", "image_defaults", "number", minimum=0, maximum=30, step=0.1),
    SettingSpec("default_width", "Width", "image_defaults", "integer", minimum=256, maximum=2048, step=64, multiple_of=64),
    SettingSpec("default_height", "Height", "image_defaults", "integer", minimum=256, maximum=2048, step=64, multiple_of=64),
    SettingSpec("img2img_default_strength", "img2img strength", "image_defaults", "number", minimum=0, maximum=1, step=0.01),
    SettingSpec("image_upload_max_mb", "Image upload MB", "image_defaults", "integer", minimum=1, maximum=2048, step=1),

    # Family defaults
    SettingSpec("flux2_quant", "FLUX.2 quant", "family_defaults", "choice", choices=_choices(("bnb-nf4", "bnb-fp4", "none"))),
    SettingSpec("flux2_offload", "FLUX.2 offload", "family_defaults", "choice", choices=_choices(("model", "sequential", "none"))),
    SettingSpec("flux2_default_steps", "FLUX.2 steps", "family_defaults", "integer", minimum=1, maximum=150, step=1),
    SettingSpec("flux2_default_guidance", "FLUX.2 guidance", "family_defaults", "number", minimum=0, maximum=30, step=0.1),
    SettingSpec("flux2_default_width", "FLUX.2 width", "family_defaults", "integer", minimum=256, maximum=2048, step=64, multiple_of=64),
    SettingSpec("flux2_default_height", "FLUX.2 height", "family_defaults", "integer", minimum=256, maximum=2048, step=64, multiple_of=64),
    SettingSpec("qwen_image_quant", "Qwen quant", "family_defaults", "choice", choices=_choices(("bnb-nf4", "bnb-fp4", "none"))),
    SettingSpec("qwen_image_offload", "Qwen offload", "family_defaults", "choice", choices=_choices(("model", "sequential", "none"))),
    SettingSpec("qwen_image_default_steps", "Qwen steps", "family_defaults", "integer", minimum=1, maximum=150, step=1),
    SettingSpec("qwen_image_default_guidance", "Qwen guidance", "family_defaults", "number", minimum=0, maximum=30, step=0.1),
    SettingSpec("qwen_image_default_width", "Qwen width", "family_defaults", "integer", minimum=256, maximum=2048, step=64, multiple_of=64),
    SettingSpec("qwen_image_default_height", "Qwen height", "family_defaults", "integer", minimum=256, maximum=2048, step=64, multiple_of=64),
    SettingSpec("z_image_offload", "Z-Image offload", "family_defaults", "choice", choices=_choices(("model", "sequential", "none"))),
    SettingSpec("z_image_default_steps", "Z-Image steps", "family_defaults", "integer", minimum=1, maximum=150, step=1),
    SettingSpec("z_image_default_guidance", "Z-Image guidance", "family_defaults", "number", minimum=0, maximum=30, step=0.1),
    SettingSpec("z_image_default_width", "Z-Image width", "family_defaults", "integer", minimum=256, maximum=2048, step=64, multiple_of=64),
    SettingSpec("z_image_default_height", "Z-Image height", "family_defaults", "integer", minimum=256, maximum=2048, step=64, multiple_of=64),

    # Acceleration
    SettingSpec("torch_compile", "torch.compile", "acceleration", "boolean"),
    SettingSpec("torch_compile_mode", "Compile mode", "acceleration", "choice", choices=_choices(("default", "reduce-overhead", "max-autotune"))),
    SettingSpec("torch_compile_warmup", "Compile warmup", "acceleration", "boolean"),
    SettingSpec("torch_compile_warmup_size", "Warmup size", "acceleration", "integer", minimum=256, maximum=2048, step=64, multiple_of=64),
    SettingSpec("flux_step_cache", "FLUX step cache", "acceleration", "choice", choices=_choices(("off", "fb", "teacache"))),
    SettingSpec("flux_fb_cache_threshold", "FB cache threshold", "acceleration", "number", minimum=0, maximum=1, step=0.01),
    SettingSpec("flux_fb_cache_double", "Double FB cache", "acceleration", "boolean"),
    SettingSpec("flux_teacache_threshold", "TeaCache threshold", "acceleration", "number", minimum=0, maximum=1, step=0.01),
    SettingSpec("flux_teacache_skip_steps", "TeaCache skip steps", "acceleration", "integer", minimum=0, maximum=100, step=1),
    SettingSpec("attention_backend", "Attention backend", "acceleration", "choice", choices=_choices(("auto", "flash", "efficient", "math", "cudnn"))),
    SettingSpec("attention_allow_tf32", "Allow TF32", "acceleration", "boolean"),
    SettingSpec("attention_matmul_precision", "Matmul precision", "acceleration", "choice", choices=_choices(("highest", "high", "medium"))),
    SettingSpec("sdxl_turbo_lora", "SDXL turbo LoRA", "acceleration", "text", "Local path or Hugging Face repo/file id.", nullable=True, restart_required=True),
    SettingSpec("sdxl_turbo_lora_weight", "SDXL turbo weight", "acceleration", "number", minimum=0, maximum=2, step=0.05),
    SettingSpec("sdxl_turbo_steps", "SDXL turbo steps", "acceleration", "integer", minimum=1, maximum=50, step=1),
    SettingSpec("sdxl_turbo_guidance", "SDXL turbo guidance", "acceleration", "number", minimum=0, maximum=30, step=0.1),
    SettingSpec("image_cleanup_after_each_job", "Cleanup after each job", "acceleration", "boolean"),
    SettingSpec("image_lora_cache_max", "LoRA cache max", "acceleration", "integer", minimum=0, maximum=32, step=1),
    SettingSpec("image_recycle_cuda_growth_gb", "Recycle CUDA growth GB", "acceleration", "number", minimum=0, maximum=64, step=0.1),
    SettingSpec("image_recycle_min_jobs", "Recycle min jobs", "acceleration", "integer", minimum=1, maximum=200, step=1),

    # Memory
    SettingSpec("keep_warm_models", "Keep models warm", "memory", "boolean"),
    SettingSpec("keep_warm_max_models", "Warm model limit", "memory", "integer", minimum=0, maximum=8, step=1),
    SettingSpec("keep_warm_min_available_ram_gb", "Warm RAM headroom GB", "memory", "number", minimum=0, maximum=128, step=0.5),
    SettingSpec("min_free_ram_gb", "Minimum free RAM GB", "memory", "number", minimum=0.5, maximum=128, step=0.5),
    SettingSpec("mem_poll_seconds", "Memory poll seconds", "memory", "number", minimum=0.5, maximum=60, step=0.5),
    SettingSpec("learn_memory_profiles", "Learn memory profiles", "memory", "boolean"),
    SettingSpec("learned_ram_margin_gb", "Learned RAM margin GB", "memory", "number", minimum=0, maximum=64, step=0.1),

    # Speech, RAG, vision
    SettingSpec("tts_gpu_layers", "TTS GPU layers", "tools", "integer", minimum=0, maximum=999, step=1),
    SettingSpec("tts_timeout_seconds", "TTS timeout seconds", "tools", "integer", minimum=30, maximum=7200, step=30),
    SettingSpec("transcription_device", "Transcription device", "tools", "choice", choices=_choices(("cpu", "cuda", "auto"))),
    SettingSpec("transcription_compute_type", "Transcription compute", "tools", "choice", choices=_choices(("int8", "int8_float16", "float16", "float32"))),
    SettingSpec("transcription_timeout_seconds", "Transcription timeout seconds", "tools", "integer", minimum=60, maximum=7200, step=60),
    SettingSpec("transcription_max_upload_mb", "Transcription upload MB", "tools", "integer", minimum=1, maximum=4096, step=1),
    SettingSpec("embed_gpu_layers", "Embedding GPU layers", "tools", "integer", minimum=0, maximum=999, step=1),
    SettingSpec("embed_timeout_seconds", "Embedding timeout seconds", "tools", "integer", minimum=10, maximum=1800, step=10),
    SettingSpec("rag_chunk_chars", "RAG chunk chars", "tools", "integer", minimum=200, maximum=8000, step=50),
    SettingSpec("rag_chunk_overlap", "RAG chunk overlap", "tools", "integer", minimum=0, maximum=2000, step=10),
    SettingSpec("vision_gpu_layers", "Vision GPU layers", "tools", "integer", minimum=0, maximum=999, step=1),
    SettingSpec("vision_timeout_seconds", "Vision timeout seconds", "tools", "integer", minimum=60, maximum=7200, step=60),
    SettingSpec("vision_max_upload_mb", "Vision upload MB", "tools", "integer", minimum=1, maximum=1024, step=1),

    # Voice defaults
    SettingSpec("voice_device", "Voice device", "voice", "choice", choices=_choices(("cuda", "cpu"))),
    SettingSpec("voice_timeout_seconds", "Voice timeout seconds", "voice", "integer", minimum=60, maximum=7200, step=60),
    SettingSpec("voice_max_upload_mb", "Voice upload MB", "voice", "integer", minimum=1, maximum=1024, step=1),
    SettingSpec("voice_pitch", "Pitch", "voice", "integer", minimum=-24, maximum=24, step=1),
    SettingSpec("voice_speaker_id", "Speaker ID", "voice", "integer", minimum=0, maximum=255, step=1),
    SettingSpec("voice_index_ratio", "Index ratio", "voice", "number", minimum=0, maximum=1, step=0.01),
    SettingSpec("voice_protect", "Protect", "voice", "number", minimum=0, maximum=1, step=0.01),
    SettingSpec("voice_noise_scale", "Noise scale", "voice", "number", minimum=0, maximum=1, step=0.01),
    SettingSpec("voice_f0_smoothing", "F0 smoothing", "voice", "number", minimum=0, maximum=1, step=0.01),
    SettingSpec("voice_f0_detector", "F0 detector", "voice", "choice", choices=_choices(("fcpe", "rmvpe", "crepe_tiny", "crepe_full"))),
    SettingSpec("voice_input_highpass_hz", "Input high-pass Hz", "voice", "integer", minimum=0, maximum=300, step=5),
    SettingSpec("voice_input_gate_db", "Input gate dB", "voice", "number", minimum=-90, maximum=-20, step=1),
    SettingSpec("voice_input_formant", "Input formant", "voice", "number", minimum=-2, maximum=2, step=0.1),
    SettingSpec("voice_input_denoise", "Input denoise", "voice", "choice", choices=_choices(("off", "dtln"))),

    # Advanced sources
    SettingSpec("flux_config_repo", "FLUX config repo", "sources", "text", restart_required=True),
    SettingSpec("flux_t5_nunchaku", "FLUX T5 nunchaku", "sources", "text", restart_required=True),
    SettingSpec("flux2_nunchaku_dir", "FLUX.2 nunchaku dir", "sources", "path", restart_required=True),
    SettingSpec("flux2_nunchaku_base_dir", "FLUX.2 base dir", "sources", "path", restart_required=True),
    SettingSpec("flux2_klein_repo", "FLUX.2 klein repo", "sources", "text", restart_required=True),
    SettingSpec("qwen_image_base_repo", "Qwen base repo", "sources", "path", restart_required=True),
    SettingSpec("qwen_image_nunchaku_blocks_on_gpu", "Qwen blocks on GPU", "sources", "integer", minimum=1, maximum=60, step=1),
    SettingSpec("qwen_image_nunchaku_text_encoder_quant", "Qwen text encoder quant", "sources", "choice", choices=_choices(("bnb-nf4", "bnb-fp4", "none"))),
    SettingSpec("z_image_base_repo", "Z-Image base repo", "sources", "path", restart_required=True),
    SettingSpec("z_image_nunchaku_offload", "Z-Image nunchaku offload", "sources", "choice", choices=_choices(("model", "none"))),
)

SPEC_BY_KEY = {spec.key: spec for spec in SPECS}
WRITABLE_KEYS = frozenset(SPEC_BY_KEY)


def overrides_path() -> Path:
    return settings.data_dir / "settings-overrides.json"


def current_values() -> dict[str, SettingValue]:
    values: dict[str, SettingValue] = {}
    for spec in SPECS:
        value = getattr(settings, spec.key)
        values[spec.key] = str(value) if isinstance(value, Path) else value
    return values


def payload() -> dict[str, Any]:
    return {
        "values": current_values(),
        "writable_keys": sorted(WRITABLE_KEYS),
        "groups": list(GROUPS),
        "schema": [spec.payload() for spec in SPECS],
        "path": str(overrides_path()),
    }


def load() -> set[str]:
    """Apply the persisted override file and return the keys it set (if any)."""
    path = overrides_path()
    if not path.exists():
        return set()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("settings overrides must be a JSON object")
    unknown = sorted(set(raw) - WRITABLE_KEYS)
    if unknown:
        raise ValueError(f"settings overrides contain unsupported keys: {', '.join(unknown)}")
    sanitized = _sanitize(raw)
    _apply(sanitized)
    return set(sanitized)


def save(patch: dict[str, Any]) -> dict[str, Any]:
    values = {**current_values(), **_sanitize(patch)}
    _apply(values)
    path = overrides_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload()


def _apply(values: dict[str, SettingValue]) -> None:
    for key, value in values.items():
        spec = SPEC_BY_KEY[key]
        setattr(settings, key, _runtime_value(spec, value))


def _runtime_value(spec: SettingSpec, value: SettingValue) -> Any:
    if spec.kind == "path":
        return Path(str(value))
    return value


def _sanitize(raw: dict[str, Any]) -> dict[str, SettingValue]:
    out: dict[str, SettingValue] = {}
    for key, value in raw.items():
        spec = SPEC_BY_KEY.get(key)
        if spec is None:
            continue
        out[key] = _sanitize_value(spec, value)
    return out


def _sanitize_value(spec: SettingSpec, value: Any) -> SettingValue:
    if spec.kind == "boolean":
        return _as_bool(value)
    if spec.kind == "integer":
        n = _clamp_int(value, spec.minimum, spec.maximum)
        if spec.multiple_of:
            n = _round_to(n, spec.multiple_of)
            n = _clamp_int(n, spec.minimum, spec.maximum)
        return n
    if spec.kind == "number":
        return _clamp_float(value, spec.minimum, spec.maximum)
    if spec.kind == "choice":
        clean = str(value).strip()
        allowed = {choice for choice, _label in spec.choices}
        if clean not in allowed:
            raise ValueError(f"{spec.key} must be one of: {', '.join(sorted(allowed))}")
        return clean
    if spec.kind == "text":
        clean = str(value).strip()
        if spec.nullable and clean == "":
            return None
        if not spec.nullable and clean == "":
            raise ValueError(f"{spec.key} cannot be empty")
        return clean
    if spec.kind == "path":
        clean = str(value).strip()
        if not clean:
            raise ValueError(f"{spec.key} cannot be empty")
        return clean
    raise ValueError(f"unsupported setting kind for {spec.key}: {spec.kind}")


def _clamp_int(value: Any, low: float | None, high: float | None) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"expected integer setting, got {value!r}") from exc
    if low is not None:
        n = max(int(low), n)
    if high is not None:
        n = min(int(high), n)
    return n


def _clamp_float(value: Any, low: float | None, high: float | None) -> float:
    try:
        n = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"expected numeric setting, got {value!r}") from exc
    if low is not None:
        n = max(low, n)
    if high is not None:
        n = min(high, n)
    return n


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        clean = value.strip().lower()
        if clean in {"1", "true", "yes", "on"}:
            return True
        if clean in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"expected boolean setting, got {value!r}")


def _round_to(value: int, multiple: int) -> int:
    return max(multiple, int(round(value / multiple) * multiple))
