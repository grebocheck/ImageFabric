"""Central configuration for ImageFabric.

Paths are resolved relative to the repository root (the folder that contains
``ImageModels/`` and ``LLM/``) so the app finds the existing model files without
any copying. Everything is overridable via environment variables or a ``.env``
file (prefix ``IMGFAB_``).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# repo root = .../ImageFabric  (this file is .../ImageFabric/backend/app/config.py)
ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="IMGFAB_",
        env_file=".env",
        extra="ignore",
    )

    # --- server ---
    host: str = "127.0.0.1"
    port: int = 8260
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
    data_dir: Path = ROOT / "data"
    outputs_dir: Path = ROOT / "data" / "outputs"
    db_path: Path = ROOT / "data" / "imagefabric.db"

    # --- llama.cpp ---
    # Path to a CUDA(sm_120) `llama-server` binary. Used in real (non-stub) mode.
    llama_server_bin: Path = ROOT / "bin" / "llama" / "llama-server.exe"
    llama_tts_bin: Path = ROOT / "bin" / "llama" / "llama-tts.exe"
    llama_mtmd_bin: Path = ROOT / "bin" / "llama" / "llama-mtmd-cli.exe"
    llama_host: str = "127.0.0.1"
    llama_port: int = 8261
    llama_embed_port: int = 8262
    # Default launch knobs (tunable per-model later). 999 = offload all layers.
    llama_ngl: int = 999
    llama_ctx: int = 8192
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

    # --- image generation defaults ---
    default_steps: int = 28
    default_guidance: float = 3.5  # FLUX-ish; SDXL overrides to ~6.0 at runtime
    default_width: int = 1024
    default_height: int = 1024

    @property
    def db_url(self) -> str:
        return f"sqlite+aiosqlite:///{self.db_path.as_posix()}"

    def ensure_dirs(self) -> None:
        for d in (
            self.data_dir,
            self.outputs_dir,
            self.image_models_dir,
            self.lora_models_dir,
            self.llm_models_dir,
            self.tts_models_dir,
            self.transcription_models_dir,
            self.embed_models_dir,
            self.vision_models_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
