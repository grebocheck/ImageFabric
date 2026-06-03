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

    # --- paths (all under the repo root by default) ---
    root: Path = ROOT
    image_models_dir: Path = ROOT / "models" / "image"
    llm_models_dir: Path = ROOT / "models" / "llm"
    data_dir: Path = ROOT / "data"
    outputs_dir: Path = ROOT / "data" / "outputs"
    db_path: Path = ROOT / "data" / "imagefabric.db"

    # --- llama.cpp ---
    # Path to a CUDA(sm_120) `llama-server` binary. Used in real (non-stub) mode.
    llama_server_bin: Path = ROOT / "bin" / "llama" / "llama-server.exe"
    llama_host: str = "127.0.0.1"
    llama_port: int = 8261
    # Default launch knobs (tunable per-model later). 999 = offload all layers.
    llama_ngl: int = 999
    llama_ctx: int = 8192

    # --- FLUX loading (M0 finding) ---
    # The local flux_dev is an fp8 all-in-one checkpoint; diffusers needs a
    # (non-gated) repo to assemble the pipeline config + tokenizers. Weights
    # still come from the local file.
    flux_config_repo: str = "ChuckMcSneed/FLUX.1-dev"

    # --- image generation defaults ---
    default_steps: int = 28
    default_guidance: float = 3.5  # FLUX-ish; SDXL overrides to ~6.0 at runtime
    default_width: int = 1024
    default_height: int = 1024

    @property
    def db_url(self) -> str:
        return f"sqlite+aiosqlite:///{self.db_path.as_posix()}"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.outputs_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
