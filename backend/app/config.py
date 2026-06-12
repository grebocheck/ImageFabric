"""Central configuration for HFabric.

Paths are resolved relative to the repository root (the folder that contains
``ImageModels/`` and ``LLM/``) so the app finds the existing model files without
any copying. Everything is overridable via environment variables or a ``.env``
file (prefix ``HFAB_``).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# repo root = .../ImageFabric  (this file is .../ImageFabric/backend/app/config.py)
ROOT = Path(__file__).resolve().parents[2]

# llama.cpp binaries are named `llama-server.exe` on Windows, `llama-server` on
# Linux/macOS. Default to the right one for the host (overridable via HFAB_*).
_EXE = ".exe" if sys.platform == "win32" else ""

# KV-cache ("context") quantization presets. Each maps to llama.cpp's
# --cache-type-k / --cache-type-v plus whether flash-attention must be forced on
# (llama.cpp requires FA for a *quantized* V cache, otherwise the server aborts).
# `turbo3` / `turbo4` are Google DeepMind's TurboQuant types: they only exist in
# a TurboQuant-patched llama.cpp build, NOT in upstream, so they are flagged
# experimental and the UI warns about the required binary.
CONTEXT_TYPES: dict[str, dict] = {
    "f16":    {"label": "F16 (default, full precision)", "k": "f16",    "v": "f16",    "flash_attn": False, "experimental": False},
    "q8_0":   {"label": "Q8_0 (~2x smaller cache)",      "k": "q8_0",   "v": "q8_0",   "flash_attn": True,  "experimental": False},
    "q4_0":   {"label": "Q4_0 (~4x smaller cache)",      "k": "q4_0",   "v": "q4_0",   "flash_attn": True,  "experimental": False},
    "turbo2": {"label": "TurboQuant turbo2 (experimental)", "k": "turbo2", "v": "turbo2", "flash_attn": True,  "experimental": True},
    "turbo3": {"label": "TurboQuant turbo3 (experimental)", "k": "turbo3", "v": "turbo3", "flash_attn": True,  "experimental": True},
    "turbo4": {"label": "TurboQuant turbo4 (experimental)", "k": "turbo4", "v": "turbo4", "flash_attn": True,  "experimental": True},
}
DEFAULT_CONTEXT_TYPE = "f16"


def resolve_context_type(name: str | None) -> dict:
    """Look up a context-type preset, falling back to the f16 default."""
    return CONTEXT_TYPES.get(name or "", CONTEXT_TYPES[DEFAULT_CONTEXT_TYPE])


# Selectable llama.cpp builds ("backends"). Upstream llama.cpp and a
# TurboQuant-patched build are *different binaries* with different capabilities:
# only the patched one understands the turbo3 / turbo4 cache types. Modelling
# them explicitly lets the UI offer the matching context types and lets us reject
# an impossible (backend, context_type) pair *before* launching the server,
# instead of the server aborting with an opaque error. `bin_attr` names the
# Settings field that holds that build's `llama-server` path.
LLAMA_BACKENDS: dict[str, dict] = {
    "default": {
        "label": "Standard llama.cpp",
        "bin_attr": "llama_server_bin",
        "context_types": ("f16", "q8_0", "q4_0"),
    },
    "turbo": {
        "label": "TurboQuant build",
        "bin_attr": "llama_server_bin_turbo",
        "context_types": ("f16", "q8_0", "q4_0", "turbo2", "turbo3", "turbo4"),
    },
}
DEFAULT_LLAMA_BACKEND = "default"


def resolve_llama_backend(name: str | None) -> dict:
    """Look up a llama-backend spec, falling back to the standard build."""
    return LLAMA_BACKENDS.get(name or "", LLAMA_BACKENDS[DEFAULT_LLAMA_BACKEND])


def backend_supports_context_type(backend: str | None, context_type: str | None) -> bool:
    """Whether the given llama backend can run the given context (cache) type."""
    return (context_type or DEFAULT_CONTEXT_TYPE) in resolve_llama_backend(backend)["context_types"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HFAB_",
        env_file=ROOT / ".env",
        extra="ignore",
    )

    # --- server ---
    host: str = "127.0.0.1"
    port: int = 8260
    # Optional API bearer token. Keep unset for loopback-only local use; set it
    # before binding to a LAN interface.
    api_token: str | None = None
    # Vite dev server origin(s) allowed via CORS.
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])

    # --- mode ---
    # STUB mode runs the whole pipeline (queue -> arbiter swap -> progress -> gallery)
    # WITHOUT torch/diffusers/llama. Flip to False in M0 once the GPU stack is in.
    stub_mode: bool = True

    # --- paths ---
    # Project default: all local model weights and multi-file model repos live
    # under ROOT/models so scans never depend on an unclear external location.
    # Env overrides remain available for development, but the documented default
    # layout is models/image, models/lora, models/llm, and models/tts.
    root: Path = ROOT
    image_models_dir: Path = ROOT / "models" / "image"
    llm_models_dir: Path = ROOT / "models" / "llm"
    lora_models_dir: Path = ROOT / "models" / "lora"
    tts_models_dir: Path = ROOT / "models" / "tts"
    transcription_models_dir: Path = ROOT / "models" / "transcribe"
    embed_models_dir: Path = ROOT / "models" / "embed"
    vision_models_dir: Path = ROOT / "models" / "vision"
    voice_models_dir: Path = ROOT / "models" / "voice"
    voice_pretrain_dir: Path = ROOT / "models" / "voice" / "pretrain"
    data_dir: Path = ROOT / "data"
    outputs_dir: Path = ROOT / "data" / "outputs"
    logs_dir: Path = ROOT / "data" / "logs"
    runtime_dir: Path = ROOT / "data" / "runtime"
    backups_dir: Path = ROOT / "data" / "backups"
    db_path: Path = ROOT / "data" / "hfabric.db"

    # --- llama.cpp ---
    # Path to a CUDA(sm_120) `llama-server` binary. Used in real (non-stub) mode.
    llama_server_bin: Path = ROOT / "bin" / "llama" / f"llama-server{_EXE}"
    # Optional separate TurboQuant-patched `llama-server` build (supports the
    # turbo3 / turbo4 cache types). Kept in its own folder so it can coexist with
    # the standard build; selected via `llama_backend` = "turbo".
    llama_server_bin_turbo: Path = ROOT / "bin" / "llama-turbo" / f"llama-server{_EXE}"
    llama_tts_bin: Path = ROOT / "bin" / "llama" / f"llama-tts{_EXE}"
    llama_mtmd_bin: Path = ROOT / "bin" / "llama" / f"llama-mtmd-cli{_EXE}"
    llama_host: str = "127.0.0.1"
    llama_port: int = 8261
    llama_embed_port: int = 8262
    # Default launch knobs (tunable per-model later). 999 = offload all layers.
    llama_ngl: int = 999
    llama_ctx: int = 8192
    # Active llama.cpp build (see LLAMA_BACKENDS). "default" = standard upstream;
    # "turbo" = TurboQuant-patched build that also offers turbo3 / turbo4.
    llama_backend: str = DEFAULT_LLAMA_BACKEND
    # KV-cache quantization preset (see CONTEXT_TYPES). "f16" keeps the cache at
    # full precision; quantized / TurboQuant types shrink it so a longer context
    # fits the same VRAM. Changing it relaunches llama-server (new launch knob).
    llama_context_type: str = DEFAULT_CONTEXT_TYPE
    # TTS runs through a separate llama.cpp CLI for now. Keep it CPU-only by
    # default so it cannot bypass the shared GPU arbiter.
    tts_gpu_layers: int = 0
    tts_timeout_seconds: int = 600
    # Transcription is CPU-only by default so it cannot bypass the shared GPU
    # arbiter. Local model folders/files are required; no hidden downloads.
    transcription_device: str = "cpu"
    transcription_compute_type: str = "int8"
    transcription_timeout_seconds: int = 1800
    transcription_max_upload_mb: int = 512
    # RAG embeddings run through a dedicated llama-server in embeddings mode.
    # Keep it CPU-only by default so it can coexist with the GPU arbiter.
    embed_gpu_layers: int = 0
    embed_timeout_seconds: int = 120
    rag_chunk_chars: int = 1200
    rag_chunk_overlap: int = 160
    # Vision uses llama-mtmd-cli for local multimodal GGUF + mmproj pairs. Keep
    # it CPU-only by default; raising this should be treated like GPU work.
    vision_gpu_layers: int = 0
    vision_timeout_seconds: int = 900
    vision_max_upload_mb: int = 64
    # Voice changer (P6R, native RVC). CUDA by default: the realtime session is
    # arbiter-coordinated (frees the resident + parks GPU jobs via the voice
    # lane), and the P6R.2 bench shows CPU only sustains realtime at chunk 192
    # while CUDA has ~2.7x headroom at chunk 133. The synthesizer itself is
    # tiny (~60 MB VRAM); ContentVec stays on onnxruntime-CPU either way.
    voice_device: str = "cuda"
    voice_timeout_seconds: int = 600
    voice_max_upload_mb: int = 64
    voice_pitch: int = 0
    voice_index_ratio: float = 1.0
    voice_protect: float = 0.5
    voice_f0_detector: str = "rmvpe"
    # w-okada Voice Changer (MMVCServerSIO) runs as its own realtime server; we
    # detect it and build UI on its API rather than importing it. Override the
    # install dir with HFAB_VOICE_WOKADA_DIR. Models live in <dir>/model_dir as
    # numbered slots (each with params.json + a .safetensors/.pth + .index).
    voice_wokada_dir: Path = Path("D:/MMVCServerSIO")
    voice_wokada_url: str = "http://127.0.0.1:18888"

    # --- FLUX loading (M0 finding) ---
    # The local flux_dev is an fp8 all-in-one checkpoint; diffusers needs a
    # (non-gated) repo to assemble the pipeline config + tokenizers. Weights
    # still come from the local file.
    flux_config_repo: str = "ChuckMcSneed/FLUX.1-dev"
    # int4 T5 for the nunchaku FLUX path (keeps RAM ~3 GB instead of ~10 GB bf16,
    # and avoids reading the local 16 GB fp8 file just to borrow its encoders).
    flux_t5_nunchaku: str = "nunchaku-tech/nunchaku-t5/awq-int4-flux.1-t5xxl.safetensors"

    # --- FLUX.2 [klein] (diffusers-native + optional nunchaku transformer) ---
    # klein uses a small Qwen3 text encoder (not FLUX.2 [dev]'s 24 GB Mistral),
    # so 9B in 4-bit + offload fits 16 GB. Drop the klein repo into a FOLDER
    # under models/image/ (it is multi-file, not a single .safetensors); it is
    # auto-detected by its model_index.json. FLUX.2 [dev] (32B + Mistral-24B) is
    # intentionally NOT supported — it blows the RAM/VRAM budget. The nunchaku
    # FLUX.2 transformer path is experimental until the upstream PR lands; the
    # sidecar code + SVDQuant weights live under flux2_nunchaku_dir.
    flux2_quant: str = "bnb-nf4"          # bnb-nf4 | bnb-fp4 | none (bf16)
    flux2_offload: str = "model"          # model | sequential | none
    flux2_default_steps: int = 6          # klein is distilled -> few steps
    flux2_default_guidance: float = 4.0
    flux2_default_width: int = 768
    flux2_default_height: int = 768
    flux2_nunchaku_dir: Path = ROOT / "models" / "image" / "flux2-klein-9b-nunchaku"
    flux2_nunchaku_base_dir: Path = ROOT / "models" / "image" / "flux2-klein-9b"
    # When a single-file klein transformer is used, the text encoder (Qwen3), VAE,
    # tokenizer and config come from this (license-gated) repo.
    flux2_klein_repo: str = "black-forest-labs/FLUX.2-klein-9B"

    # --- Qwen-Image-2512 / Z-Image-Turbo (multi-file Diffusers repos) ---
    # Qwen-Image-2512 is large (~54 GB of bf16 shards), so default to
    # bitsandbytes 4-bit for the transformer + text encoder and model offload.
    # Z-Image-Turbo is an 8-step distilled model; the official recipe uses
    # guidance 0.0 and 1024^2 output.
    qwen_image_quant: str = "bnb-nf4"     # bnb-nf4 | bnb-fp4 | none (bf16)
    qwen_image_offload: str = "model"     # model | sequential | none
    qwen_image_default_steps: int = 50
    qwen_image_default_guidance: float = 4.0  # sent as true_cfg_scale
    qwen_image_default_width: int = 1328
    qwen_image_default_height: int = 1328
    z_image_offload: str = "model"        # model | sequential | none
    z_image_default_steps: int = 9
    z_image_default_guidance: float = 0.0  # Turbo model wants CFG off
    z_image_default_width: int = 1024
    z_image_default_height: int = 1024

    # --- Nunchaku SVDQuant fp4 fast paths for Qwen-Image / Z-Image ---
    # The bf16 repos above are huge; a Nunchaku SVDQuant fp4 transformer (a single
    # ~12 GB file for Qwen, ~4 GB for Z-Image) is transformer-only, so it borrows
    # the text encoder / VAE / tokenizer / scheduler from the matching local bf16
    # repo folder below. Drop the fp4 .safetensors into models/image/ and the
    # registry auto-detects it (name contains "nunchaku"/"svdq" + the family).
    qwen_image_base_repo: Path = ROOT / "models" / "image" / "qwen-image-2512"
    z_image_base_repo: Path = ROOT / "models" / "image" / "z-image-turbo"
    # On <18 GB cards Nunchaku Qwen-Image needs per-layer transformer offload
    # (transformer.set_offload) + sequential CPU offload for the bf16 Qwen2.5-VL
    # text encoder. num_blocks_on_gpu trades VRAM for speed (raise it if VRAM
    # allows). Z-Image's fp4 transformer is small enough to keep resident.
    qwen_image_nunchaku_blocks_on_gpu: int = 20
    # The bf16 Qwen2.5-VL text encoder (~16 GB) won't fit 32 GB RAM under sequential
    # offload alongside the fp4 transformer; quantize it to 4-bit by default.
    qwen_image_nunchaku_text_encoder_quant: str = "bnb-nf4"  # bnb-nf4 | bnb-fp4 | none
    z_image_nunchaku_offload: str = "model"  # model | none (none = keep resident)

    # --- image acceleration ---
    # P1.1: Opt-in compile because Blackwell compile can spike RAM/VRAM during
    # graph capture. When enabled, the backend records before/after memory in
    # the model.loaded event and does a tiny warmup pass.
    torch_compile: bool = False
    torch_compile_mode: str = "max-autotune"
    torch_compile_warmup: bool = True
    torch_compile_warmup_size: int = 512
    # P1.2: Nunchaku first-block cache for FLUX. "fb" is the native adapter in
    # nunchaku; "teacache" wraps each generation with TeaCache; "off" disables.
    flux_step_cache: str = "fb"
    flux_fb_cache_threshold: float = 0.12
    flux_fb_cache_double: bool = False
    flux_teacache_threshold: float = 0.6
    flux_teacache_skip_steps: int = 0
    # P2.2: PyTorch scaled-dot-product attention backend selector. "auto" lets
    # PyTorch choose; "flash", "efficient", "math", and "cudnn" force a native
    # SDPA backend when the installed torch build exposes it.
    attention_backend: str = "auto"
    attention_allow_tf32: bool = True
    attention_matmul_precision: str = "high"
    # P1.4: Optional SDXL turbo LoRA. Set to a local .safetensors file, local
    # folder, or Hugging Face repo id to make SDXL default to low-step turbo mode.
    sdxl_turbo_lora: str | None = None
    sdxl_turbo_lora_weight: float = 1.0
    sdxl_turbo_steps: int = 4
    sdxl_turbo_guidance: float = 1.0
    # Runtime stabilization for long image sessions. This keeps the resident
    # model loaded, but releases per-job temporaries and bounds adapter growth.
    image_cleanup_after_each_job: bool = True
    image_lora_cache_max: int = 2
    image_recycle_cuda_growth_gb: float = 2.0
    image_recycle_min_jobs: int = 6

    # --- optional keep-warm policy (P2.1) ---
    # OFF by default. When enabled, the arbiter may park one image pipeline in
    # CPU RAM between swaps instead of fully deleting it. It is still not a VRAM
    # resident, and the RAM guard below decides whether parking is allowed.
    keep_warm_models: bool = False
    keep_warm_max_models: int = 1
    keep_warm_min_available_ram_gb: float = 10.0

    # --- memory budget (protect the SSD from pagefile writes) ---
    # Refuse a load if it would leave less than this much free RAM.
    min_free_ram_gb: float = 2.5
    # How often to broadcast a mem.status event (seconds).
    mem_poll_seconds: float = 3.0

    # --- learned memory profiles (P7.2) ---
    # Record each model's measured peak RAM/VRAM after a load and feed it back
    # into the budget guard, replacing the static size*factor heuristic once a
    # real measurement exists. Safety margin added on top of the measured RAM.
    learn_memory_profiles: bool = True
    learned_ram_margin_gb: float = 1.5

    # --- img2img (P13.4) ---
    # Max accepted source-image upload size, and the default denoise strength
    # (low = keep more of the source, high = follow the prompt more freely).
    image_upload_max_mb: int = 24
    img2img_default_strength: float = 0.6

    # --- image generation defaults ---
    default_steps: int = 28
    default_guidance: float = 3.5  # FLUX-ish; SDXL overrides to ~6.0 at runtime
    default_width: int = 1024
    default_height: int = 1024

    @property
    def db_url(self) -> str:
        return f"sqlite+aiosqlite:///{self.db_path.as_posix()}"

    @property
    def active_llama_bin(self) -> Path:
        """The `llama-server` path for the currently selected llama backend."""
        return getattr(self, resolve_llama_backend(self.llama_backend)["bin_attr"])

    def ensure_dirs(self) -> None:
        for d in (
            self.data_dir,
            self.outputs_dir,
            self.logs_dir,
            self.runtime_dir,
            self.backups_dir,
            self.image_models_dir,
            self.lora_models_dir,
            self.llm_models_dir,
            self.tts_models_dir,
            self.transcription_models_dir,
            self.embed_models_dir,
            self.vision_models_dir,
            self.voice_models_dir,
            self.voice_pretrain_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
