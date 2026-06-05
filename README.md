# HFabric

Local app that pairs an **LLM (prompt generation)** with **diffusion image
generation**, built to be frugal with memory on a single 16 GB GPU. Its core is a
**VRAM arbiter**: only one heavy model lives in VRAM at a time, and a
**phase-batching scheduler** swaps LLM ↔ image models as few times as possible
(ideally once per batch).

Target hardware: **RTX 5070 Ti 16 GB (Blackwell), 32 GB RAM, Windows 11**.

## Status

**Architectural foundation — complete and verified in STUB mode.**
The whole pipeline (model discovery → queue → arbiter swap → live progress over
WebSocket → gallery with reproducible metadata) runs today **without** torch or
llama.cpp. Real model loading is wired but lazy, and is turned on in milestone M0
by flipping `HFAB_STUB_MODE=false` after the GPU stack is installed.

## Architecture

```
                       ┌───────────────── FastAPI (app.main) ─────────────────┐
  React + Tailwind ──► │  REST /api/*          WebSocket /ws (event stream)    │
  (Vite, :5173)        │     │                        ▲                        │
                       │     ▼                        │ events                 │
                       │  Queue (SQLite) ─► Worker ─► EventBus                  │
                       │                      │                                │
                       │                      ▼  phase-batching                │
                       │                 GpuArbiter  ── at most ONE resident   │
                       │                   /     \                             │
                       │     DiffusersImageBackend   LlamaCppBackend           │
                       │     (FLUX fp8 / SDXL)       (llama-server subprocess) │
                       └───────────────────────────────────────────────────────┘
```

Key modules (backend):
- `app/core/arbiter.py` — the VRAM arbiter (load/unload, one resident max).
- `app/core/scheduler.py` — single GPU worker + phase-batching select.
- `app/core/events.py` — in-process pub/sub, streamed over `/ws`.
- `app/backends/` — `registry` (scan model files), `image_diffusers`, `llm_llamacpp`.
- `app/db/` — SQLAlchemy models; the queue is persisted and resumes on restart.

## Run

Easiest — double-click **`run.bat`** (or from a terminal):

```bat
run.bat          REM REAL mode: real models on the GPU (default)
run.bat stub     REM STUB mode: full pipeline, no GPU/ML stack
```

It bootstraps the venv + npm deps on first run, frees any stale ports left by an
earlier run, then runs the backend (`:8260`) and the Vite dev server (`:5173`)
**together in one window** and opens <http://localhost:5173> for you. Ctrl+C
stops both.

PowerShell equivalent:

```powershell
.\scripts\run.ps1          # REAL mode
.\scripts\run.ps1 -Stub    # STUB mode
```

> **Note:** the launcher kills whatever is holding ports 8260/8261/5173 before
> starting. A leftover backend from a previous run holding port 8260 is what
> caused the `WinError 10013` "socket forbidden" failure; freeing it first fixes
> that.

Models are read in place from `models/`: image checkpoints/repos under
`models/image/`, LoRAs under `models/lora/`, and GGUF LLMs under `models/llm/`.
Nothing is copied. See [models/README.md](models/README.md).

## Configuration

Env vars (prefix `HFAB_`, or a `.env` file in repo root). Highlights:

| Var | Default | Meaning |
|-----|---------|---------|
| `HFAB_STUB_MODE` | `true` | Run without GPU/ML stack (foundation mode). |
| `HFAB_PORT` | `8260` | Backend port. |
| `HFAB_LLAMA_SERVER_BIN` | `bin/llama/llama-server.exe` | CUDA(sm_120) llama.cpp build. |
| `HFAB_LLAMA_NGL` | `999` | GPU layers to offload (999 = full offload). |
| `HFAB_LLAMA_CTX` | `8192` | llama.cpp context size. |
| `HFAB_LORA_MODELS_DIR` | `models/lora` | Local SDXL/FLUX LoRA files. |
| `HFAB_TTS_MODELS_DIR` | `models/tts` | Local llama-tts GGUF voice/acoustic models. |
| `HFAB_TRANSCRIPTION_MODELS_DIR` | `models/transcribe` | Local Whisper model folders or `.pt` files. |
| `HFAB_TRANSCRIPTION_DEVICE` | `cpu` | Transcription device; CPU-first so it does not bypass the GPU arbiter by default. |
| `HFAB_EMBED_MODELS_DIR` | `models/embed` | Local GGUF embedding models for RAG. |
| `HFAB_EMBED_GPU_LAYERS` | `0` | GPU layers for the RAG embedding server; CPU-only by default. |
| `HFAB_VISION_MODELS_DIR` | `models/vision` | Local multimodal GGUF + mmproj files. |
| `HFAB_LLAMA_MTMD_BIN` | `bin/llama/llama-mtmd-cli.exe` | llama.cpp multimodal CLI. |
| `HFAB_VISION_GPU_LAYERS` | `0` | GPU layers for vision analysis; CPU-only by default. |
| `HFAB_FLUX_STEP_CACHE` | `fb` | FLUX acceleration: `fb`, `teacache`, or `off`. |
| `HFAB_ATTENTION_BACKEND` | `auto` | PyTorch SDPA selector: `auto`, `flash`, `efficient`, `math`, or `cudnn`. |
| `HFAB_TORCH_COMPILE` | `false` | Compile FLUX transformer and run a warmup pass. |
| `HFAB_SDXL_TURBO_LORA` | unset | Optional SDXL DMD2/Lightning-style LoRA source. |
| `HFAB_IMAGE_CLEANUP_AFTER_EACH_JOB` | `true` | Release per-image temporary allocations while keeping the resident model loaded. |
| `HFAB_IMAGE_LORA_CACHE_MAX` | `2` | Max runtime LoRA adapters to keep in a resident image pipeline. |
| `HFAB_KEEP_WARM_MODELS` | `false` | Park one image pipeline in CPU RAM between swaps. |

### llama.cpp memory knobs

The LLM backend starts `llama-server` as a subprocess with `-ngl
HFAB_LLAMA_NGL` and `--fit off`. The default `999` keeps the GGUF fully
offloaded to VRAM, while `--fit off` prevents llama.cpp from silently reducing
offload when another process has touched CUDA.

HFabric leaves llama.cpp's mmap default enabled (it does not pass
`--no-mmap`), so the GGUF file stays disk-backed and process RSS remains low.
When the arbiter switches to an image model, it terminates `llama-server`; that
is the expected way to release llama.cpp VRAM completely.

### Image acceleration knobs

`HFAB_FLUX_STEP_CACHE=fb` enables nunchaku's native first-block cache for FLUX
pipelines. Use `teacache` for the TeaCache context manager, or `off` to compare
baseline quality/speed.

`HFAB_ATTENTION_BACKEND=auto` leaves scaled-dot-product attention backend
selection to PyTorch. Set it to `flash`, `efficient`, `math`, or `cudnn` to force
a native `torch.nn.attention.sdpa_kernel` backend when the installed torch build
and CUDA device expose it. The load report records available native SDPA
backends, float8 dtype support, and whether external `flash_attn`/`xformers`
packages are installed; the local environment currently uses PyTorch native SDPA
rather than those external packages.

`HFAB_ATTENTION_ALLOW_TF32=true` and
`HFAB_ATTENTION_MATMUL_PRECISION=high` set the CUDA matmul/precision policy
before image generation.

`HFAB_TORCH_COMPILE=true` wraps the FLUX transformer with `torch.compile` using
`HFAB_TORCH_COMPILE_MODE` (default `max-autotune`) and runs a 1-step warmup.
The `model.loaded` WebSocket event includes a `load_report` with RAM/VRAM before
and after compile/warmup. Compile is best-effort: if the installed torch/nunchaku
combination fails during compile or warmup, the backend rolls back to the
original transformer, records the failure in `load_report`, and continues
generation without compile.

Set `HFAB_SDXL_TURBO_LORA` to a local `.safetensors`, folder, or Hugging Face
repo id to load an SDXL turbo LoRA. When active, untouched default steps/guidance
are replaced by `HFAB_SDXL_TURBO_STEPS` and `HFAB_SDXL_TURBO_GUIDANCE`.

Long image sessions run a lightweight post-job stabilization pass by default:
`gc.collect()`, `torch.cuda.empty_cache()`, `torch.cuda.ipc_collect()`, bounded
runtime LoRA adapter cleanup, and an adaptive soft-recycle if CUDA allocated
memory drifts above the loaded baseline. Tune with
`HFAB_IMAGE_CLEANUP_AFTER_EACH_JOB`, `HFAB_IMAGE_LORA_CACHE_MAX`,
`HFAB_IMAGE_RECYCLE_CUDA_GROWTH_GB`, and `HFAB_IMAGE_RECYCLE_MIN_JOBS`.
Use `python scripts\sdxl_resident_drift_test.py --jobs 8` against a running
REAL backend to validate repeated same-model SDXL generations without unloading.

### LoRA management

Drop SDXL/FLUX LoRA files under `models/lora` (or set
`HFAB_LORA_MODELS_DIR`). The backend scans `.safetensors`, `.pt`, and `.bin`
files on startup, exposes them at `/api/loras`, and validates queued
`params.loras` against the selected image model. The composer filters compatible
LoRAs and stores only public `{id,name,family,weight}` metadata in jobs/presets;
local file paths are resolved by the worker right before generation.

### Speech workspaces

The TTS tab scans `models/tts` for local `.gguf` files and calls
`bin/llama/llama-tts.exe`. It defaults to `HFAB_TTS_GPU_LAYERS=0`, so speech
generation stays CPU-only unless explicitly changed.

The Transcribe tab is similarly gated. `/api/transcription/status` reports local
Whisper engines (`faster-whisper` or `openai-whisper`) and scans
`models/transcribe` for model folders/files. `/api/transcription/transcribe`
accepts an audio upload only when both an engine and a local model are present;
it writes transcript metadata under `data/outputs/<date>/`.

The Vision tab scans `models/vision` for a local multimodal GGUF and `mmproj`
pair, then calls `bin/llama/llama-mtmd-cli.exe` for PNG/JPEG analysis. It
defaults to `HFAB_VISION_GPU_LAYERS=0`, and stores JSON result sidecars under
`data/outputs/<date>/`.

### RAG workspace

The RAG tab scans `models/embed` for local GGUF embedding models and starts a
dedicated `llama-server` on `HFAB_LLAMA_EMBED_PORT` (default 8262) in
`--embeddings` mode on first use. `HFAB_EMBED_GPU_LAYERS=0` keeps it CPU-only
by default, so document indexing/search does not take VRAM from the shared
arbiter.

Indexed documents are chunked into SQLite `rag_documents` / `rag_chunks` rows
with normalized embedding vectors. Search returns top chunks by cosine score,
and the RAG tab can create an LLM conversation with the retrieved context
inserted into the user turn.

The LLM chat tab also has a **Document tool** toggle. When enabled, the model may
emit a structured `search_documents` call; HFabric runs local RAG search,
then queues a child LLM turn with the retrieved context so the final response
streams into the same assistant message.

### History, export, settings

The gallery history supports `/api/images?q=...` search across image ids, job ids,
seeds, prompts, models, and JSON metadata. Each image has a PNG download endpoint
and `/api/images/{id}/metadata` for reproducibility export.

`/api/settings` exposes a read-only runtime snapshot for the settings drawer:
model paths, memory guard values, acceleration knobs, model/LoRA counts, GPU
status, and current memory telemetry.

### Keep-warm policy

`HFAB_KEEP_WARM_MODELS=true` lets the arbiter park up to
`HFAB_KEEP_WARM_MAX_MODELS` image pipeline(s) in CPU RAM when switching to a
different model. Parked models are not VRAM residents; `/api/gpu` and the header
show them as `CPU warm`, and `/api/gpu/free` unloads them.

Parking is skipped unless available RAM can satisfy the model estimate plus
`HFAB_KEEP_WARM_MIN_AVAILABLE_RAM_GB` headroom, so this feature should not push
Windows toward the pagefile. It is off by default.

### Runtime checks

With the backend running in real GPU mode:

```powershell
python .\scripts\swap_leak_test.py --cycles 3
```

The test forces `LLM -> FLUX(nunchaku) -> SDXL -> LLM`, frees the GPU after each
cycle, then checks that process RSS and VRAM return close to a warm baseline.
Use `--strict-cold-baseline` only when diagnosing one-time import/cache growth.

To validate live phase-batching against the running app:

```powershell
python .\scripts\phase_batch_check.py
```

It queues `LLM -> image -> LLM -> image` in one batch and asserts that the worker
starts jobs as `LLM -> LLM -> image -> image`, producing exactly one model-family
swap.

To queue a same-seed quality A/B across image models:

```powershell
python .\scripts\quality_ab.py --family flux --limit 2 --free-gpu-first --json-out data\runtime\quality-ab.json
```

The runner prints the job ids, image ids, and `/api/images/{id}/file` URLs. It
uses the model metadata exposed by `/api/models`, including `nunchaku-fp4` and
`nunchaku-int4` quant labels when those filenames are present. Pass
`--continue-on-error` when the comparison should record incompatible candidates
and continue.

For SDXL turbo validation, put a Lightning/DMD2 LoRA in `models/lora` and start
the backend with `HFAB_SDXL_TURBO_LORA=<path>`. The local M1 run used
`models/lora/sdxl_lightning_4step_lora.safetensors`, `HFAB_SDXL_TURBO_STEPS=4`,
and `HFAB_SDXL_TURBO_GUIDANCE=1.0`.

## Next: milestone M0 (GPU bring-up)

1. Install the Blackwell GPU stack:
   ```powershell
   .\.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   .\.venv\Scripts\pip install -r backend\requirements-gpu.txt
   ```
   Verify: `python -c "import torch; print(torch.cuda.get_device_capability())"` → `(12, 0)`.
2. Drop a CUDA(sm_120) `llama-server.exe` into `bin/llama/`.
3. Set `HFAB_STUB_MODE=false` and generate one image with FLUX, one with SDXL,
   and one LLM completion — the same UI, now backed by real models.
```
