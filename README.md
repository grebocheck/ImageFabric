# HFabric

Local app that pairs an **LLM (prompt generation)** with **diffusion image
generation**, built to be frugal with memory on a single 16 GB GPU. Its core is a
**VRAM arbiter**: only one heavy model lives in VRAM at a time, and a
**phase-batching scheduler** swaps LLM ↔ image models as few times as possible
(ideally once per batch).

Target hardware: **RTX 5070 Ti 16 GB (Blackwell), 32 GB RAM, Windows 11**.

## Status

**Working app — real-GPU validated (M0/M1).**
The full pipeline (model discovery → queue → arbiter swap → live progress over
WebSocket → gallery with reproducible metadata) runs on the GPU today: SDXL,
FLUX (Nunchaku fp4), FLUX.2 [klein], and GGUF LLMs via llama-server, plus the
chat / history / RAG / voice workspaces. See the [ROADMAP](ROADMAP.md) for the
shipped milestones and the active backlog.

The same pipeline also runs **without** torch or llama.cpp in **STUB mode**
(`HFAB_STUB_MODE=true`) — used for UI work and as the basis for CI (see
[Testing](#testing)). Real model loading is the default; `run.bat` / `run.ps1`
run REAL mode, while `run.bat stub` forces STUB.

## License And Models

HFabric application code is free and open-source software under the [MIT License](LICENSE).

AI model weights, LoRA adapters, GGUF files, checkpoints, tokenizers, datasets,
and voices are **not included** in this repository and are not licensed by the
HFabric MIT License. They are user-supplied runtime inputs with their own
provider licenses and terms. See [MODEL_NOTICE.md](MODEL_NOTICE.md) for the
full model notice.

## Installation

### System Requirements

| Requirement | Specification |
|---|---|
| **GPU** | NVIDIA RTX 5070 Ti (Blackwell, sm_120) with 16 GB VRAM; or any CUDA 12.x-compatible GPU (with memory adjustments). |
| **RAM** | 32 GB minimum (16 GB for models, 16 GB for OS + processes). |
| **OS** | Windows 11 recommended; Windows 10 may work with CUDA driver support. |
| **Disk** | 150 GB free: ~80 GB for model files (FLUX/SDXL/LLMs), ~50 GB working space. |

### Prerequisites

1. **Python 3.12+** ([download](https://www.python.org/downloads/))
   - Ensure `python` and `pip` are in your PATH
   - Verify: `python --version`

2. **Node.js 18+** ([download](https://nodejs.org/))
   - Includes npm
   - Verify: `node --version` and `npm --version`

3. **NVIDIA GPU drivers** (for REAL mode)
   - Update to latest from [nvidia.com](https://www.nvidia.com/Download/driverDetails.aspx)
   - Verify: Run `nvidia-smi` in a terminal; you should see GPU info and CUDA version ≥ 12.1

4. **Git** (optional, for model downloads)
   - Used by `huggingface-cli` to pull models
   - Verify: `git --version`

### Automated Setup (Recommended)

**Use the setup script for your platform** to run an interactive guided install:

```bat
setup.bat          # Windows interactive: choose STUB, REAL, or REAL+models
setup.bat stub     # STUB mode (no GPU, ~1 min)
setup.bat real     # REAL mode + GPU stack (10–15 min)
setup.bat all      # REAL mode + GPU stack + download ALL models (30–60 min + 80 GB)
```

```bash
./setup.sh          # Linux/macOS interactive: choose STUB, REAL, or REAL+models
./setup.sh stub     # STUB mode (no GPU)
./setup.sh real     # REAL mode + GPU stack
./setup.sh all      # REAL mode + download curated models
```

Or from PowerShell on Windows:
```powershell
.\setup.ps1        # Guided setup
.\setup.ps1 -Stub  # STUB mode
.\setup.ps1 -Real  # REAL mode only
.\setup.ps1 -DownloadAll  # REAL mode + models
```

**What the setup script does:**
1. ✓ Checks Python 3.12+, Node.js 18+, NVIDIA drivers (if needed)
2. ✓ Creates and activates Python venv
3. ✓ Installs pip dependencies (`requirements.txt` + optionally `requirements-gpu.txt`)
4. ✓ Installs npm packages (`frontend/package.json`)
5. ✓ Optionally installs PyTorch + CUDA 12.8 (for RTX 5070 Ti)
6. ✓ Optionally installs Nunchaku (FLUX acceleration)
7. ✓ Optionally downloads curated models (FLUX, SDXL, LLMs, etc.)

After setup finishes, run `run.bat` (or `run.ps1`) to start the app.

---

### Quick Start (STUB Mode — No GPU)

**STUB mode runs the entire pipeline without GPU/ML libraries.** Perfect for UI testing, debugging, and verifying the foundation.

#### Step 1: Clone & enter the repo
```bash
cd d:\VSCode\ImageFabric
```

#### Step 2: Run (one command)
```bat
run.bat stub
```

Or on PowerShell:
```powershell
.\scripts\run.ps1 -Stub
```

**What happens:**
- First run: bootstraps Python virtual environment and npm dependencies (~1–2 min).
- Backend starts at `http://localhost:8260`
- Frontend dev server starts at `http://localhost:5173`
- Browser opens automatically
- **Ctrl+C** stops both servers

#### Step 3: Verify
1. Open <http://localhost:5173> in your browser
2. Try the chat/image form — you'll see mock responses (no real GPU calls)
3. Check the backend console for any errors

---

### GPU Setup (REAL Mode)

**REAL mode loads actual LLMs and diffusion models onto your GPU.** This requires PyTorch, CUDA wheels, and model files.

#### Step 1: Install CUDA 12.8 PyTorch

For **RTX 5070 Ti (Blackwell, sm_120)**, you need CUDA 12.8 wheels:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

For other GPUs, adjust the index URL (e.g., `cu121` for CUDA 12.1):
- [PyTorch Start Locally](https://pytorch.org/get-started/locally/) — select your CUDA version
- Verify install: `python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability())"`

#### Step 2: Install GPU-accelerated backends

```bash
pip install -r backend/requirements-gpu.txt
```

This adds diffusers, transformers, accelerate, bitsandbytes, and related libraries.

#### Step 3: (Optional) Install Nunchaku for FLUX

For faster FLUX generation (~18 s/1024px on RTX 5070 Ti with SVDQuant fp4), install the Nunchaku wheel matching your torch/CUDA/Python:

```bash
pip install https://github.com/nunchaku-ai/nunchaku/releases/download/v1.3.0dev20260213/nunchaku-1.3.0.dev20260213+cu12.8torch2.11-cp312-cp312-win_amd64.whl
```

Then download the fp4 FLUX model:

```bash
huggingface-cli download nunchaku-tech/nunchaku-flux.1-dev svdq-fp4_r32-flux.1-dev.safetensors --local-dir models/image
```

#### Step 4: Download models

Models live in `models/` and are **not copied** into the venv or elsewhere. HFabric reads them in place.

**Image models** (FLUX/SDXL — goes in `models/image/`):

```bash
# FLUX ComfyUI checkpoint (fp8, baseline reference)
huggingface-cli download black-forest-labs/FLUX.1-dev flux_dev.safetensors --local-dir models/image

# SDXL Lightning (faster SDXL)
huggingface-cli download ByteDance/SDXL-Lightning sdxl_lightning_4step_lora.safetensors --local-dir models/lora
```

**LLM models** (GGUF format — goes in `models/llm/`):

```bash
# Example: Gemma 3 12B quantized
huggingface-cli download Gron1-ai/Gemma-3-12B-it-Heretic-v2-GGUF gemma-3-12b-it-heretic-v2-Q4_K_M.gguf --local-dir models/llm
```

See [models/README.md](models/README.md) for the full curated list and setup hints.

#### Step 5: Run in REAL mode

```bat
run.bat
```

Or PowerShell:
```powershell
.\scripts\run.ps1
```

**Environment file (optional):**

Create `.env` in the repo root to override defaults (e.g., CUDA device, step cache mode):

```env
HFAB_STUB_MODE=false
HFAB_FLUX_STEP_CACHE=fb
HFAB_TORCH_COMPILE=true
HFAB_PORT=8260
HFAB_LLAMA_NGL=999
```

See [Configuration](#configuration) section for all knobs.

#### Step 6: Verify GPU usage

1. Open another terminal and run: `nvidia-smi -l 1` (updates every 1 second)
2. In the app, submit a generation job
3. Watch your GPU memory fill up, then empty after completion
4. Backend console shows timing, memory snapshots, and model load/unload events

---

### Troubleshooting

#### **"WinError 10013: socket forbidden"**
Ports 8260 or 5173 are already in use by a previous run:
- Run `run.bat` or `run.ps1` again — they auto-kill stale processes
- Or manually: `netstat -ano | findstr :8260` and `taskkill /PID <pid> /F`

#### **"ModuleNotFoundError: No module named 'torch'"**
PyTorch not installed or virtual environment not activated:
- Delete `backend\.venv` and re-run `run.bat` to bootstrap from scratch
- Or manually: `pip install -r backend/requirements-gpu.txt`

#### **"CUDA out of memory"**
Model is too large for your GPU, or swap settings need tuning:
- Reduce `HFAB_LLAMA_NGL` (e.g., `HFAB_LLAMA_NGL=32` to offload only 32 layers to GPU)
- Disable compile: `HFAB_TORCH_COMPILE=false`
- Try smaller models or lower resolution requests
- Check [Configuration](#configuration) for memory tuning knobs

#### **Models not found / "No image models discovered"**
Models are in the wrong folder or not yet downloaded:
- Ensure files exist in `models/image/`, `models/llm/`, `models/lora/`, etc.
- Verify paths in the Configuration table (`HFAB_*_MODELS_DIR`)
- Re-run `huggingface-cli download` commands above

#### **Vite dev server won't start**
Port 5173 conflict or npm dependencies not installed:
- Kill any process on 5173: `netstat -ano | findstr :5173`
- Or let `run.bat` do it automatically (it frees ports before starting)
- Check `npm install` in `frontend/` ran successfully

#### **Backend crashes after first request**
Usually STUB mode reaching a code path that requires ML libraries:
- Ensure you ran `pip install -r backend/requirements-gpu.txt` for REAL mode
- Or use STUB mode (`run.bat stub`) if you're not ready for GPU

#### **Get help**
- Check backend logs (printed to terminal where `run.bat` runs)
- Check browser console (F12 in Firefox/Chrome)
- Search existing issues or documentation in the repo

---

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

## Testing

The whole pipeline runs in STUB mode (no GPU/ML stack), so the memory-budget
logic, the phase-batching scheduler, and the queue → arbiter swap → gallery flow
are all testable on a plain machine. CI runs both suites on every push/PR
(`.github/workflows/ci.yml`).

**Backend** (pytest, stub mode — hermetic temp DB + dummy model files):

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt pytest pytest-asyncio ruff
.\.venv\Scripts\ruff check app tests
.\.venv\Scripts\python -m pytest
```

Key coverage: `tests/test_scheduler.py` (the *one-swap-per-mixed-batch* invariant),
`tests/test_sysmon.py` (the RAM budget guard), `tests/test_model_profile.py`
(learned-profile running max), and `tests/test_stub_integration.py` (the full
queue → swap → gallery flow over an ASGI client).

**Frontend** (vitest + Testing Library):

```powershell
cd frontend
npm install
npx tsc -b      # typecheck
npm test        # vitest run
```

The runtime GPU checks above (`scripts/swap_leak_test.py`,
`scripts/phase_batch_check.py`, `scripts/quality_ab.py`) complement these — they
validate the *real* GPU path against a running backend.
