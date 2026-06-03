# ImageFabric — Roadmap & Prioritized Backlog

> Rebalanced after M0: **speed is not the only goal — RAM frugality is now a
> first-class objective**, because exhausting the 32 GB of RAM makes Windows
> spill to the pagefile, and those constant pagefile *writes* wear the SSD.

## Objectives (in priority order)

1. **RAM frugality — every single model load must fit comfortably, so the app
   never OOMs, hangs, or spills to the pagefile.** Keep the working set well under
   physical RAM (hard budget: peak ≈ **≤ 26 GB of 32 GB**). Optimization (small
   quantized models, no wasteful loads) is what keeps us away from the limit —
   not aggressive process-killing. Stopping the *previous* model when the user
   **switches** models is fine and expected; the goal is that loading any one
   model on its own is always safely within budget.
2. **VRAM frugality — one resident heavy model** (the arbiter), ≤ 16 GB, with a
   safety margin so we never overflow into shared/system VRAM (that path is the
   23-min FLUX disaster from M0).
3. **Speed on Blackwell** — fp4/fp8 compute, `torch.compile`, step-caching.

## Memory invariants

- VRAM: exactly one resident heavy model at a time (LLM **or** an image model).
- RAM: a guard checks predicted peak vs. available RAM **before** a load; if a
  load wouldn't fit it reports clearly and waits/queues — it must never push the
  OS into the pagefile or leave the app hung "doing nothing because it's out of
  memory".
- Switching models frees the previous one cleanly (this is expected, and made
  rare by phase-batching): llama-server is shut down (the only way to release its
  VRAM); diffusers pipelines are `del` + `gc.collect()` + `empty_cache()` +
  `ipc_collect()`. We do **not** kill as a routine memory tactic — optimization
  keeps each load within budget so we don't have to.
- Telemetry: process RSS + system available RAM + VRAM are surfaced in
  `/api/health` and over the WebSocket so we can *see* pressure, not guess.

---

## Backlog

### P0 — Memory hygiene & correctness (do first)

- [x] **P0.1 — Nunchaku FLUX encoders without the 16 GB read.** Today the nunchaku
  path calls `FluxPipeline.from_single_file(flux_dev_fp8)` just to borrow
  T5/CLIP/VAE — that reads 16 GB from SSD and briefly materializes the ~12 GB fp8
  transformer only to throw it away. Replace with:
  - T5 → `NunchakuT5EncoderModel` (int4, ~3 GB) from `nunchaku-tech/nunchaku-t5`,
  - CLIP-L → `openai/clip-vit-large-patch14` (~250 MB, non-gated),
  - VAE → FLUX VAE from the non-gated config repo (small).
  **Win:** ~10 GB → ~4 GB RAM, removes a 16 GB SSD read per FLUX load, lower VRAM.
- [x] **P0.2 — RAM guard + telemetry.** Add `psutil`; report RSS / available RAM /
  VRAM in `/api/health` and as a `mem.status` WS event. Before any model load the
  arbiter checks predicted peak vs. a configurable budget and defers if it would
  breach it (prevents pagefile thrash by construction).
- [x] **P0.3 — Swap-loop leak test.** Automated LLM→FLUX→SDXL→LLM ×N loop asserting
  RAM and VRAM return to baseline each cycle (catch leaks / fragmentation).
- [x] **P0.4 — Default FLUX = nunchaku.** Flag the raw fp8 `flux_dev` entry as
  "slow / high-mem" (or hide it) so a click can't accidentally trigger a 23-min,
  VRAM-overflowing run. Surface quant/est-VRAM per model in the UI.
- [x] **P0.5 — Confirm llama-server is mmap + full-offload** (disk-backed, no
  pagefile; VRAM via `-ngl 999`) and document the knobs.

P0 implementation notes:
- RAM/VRAM telemetry + pre-load guard live in `backend/app/util/sysmon.py` and
  are exposed through `/api/health`, `/api/models`, and `mem.status` WS events.
- Nunchaku FLUX uses `NunchakuT5EncoderModel` and the non-gated FLUX config repo;
  it no longer reads the local 16 GB fp8 checkpoint to borrow encoders.
- The swap-loop leak runner is `scripts/swap_leak_test.py`.

### P1 — Speed & live UX

- [x] **P1.1 — `torch.compile`** on the transformer (mode=max-autotune) + a warmup
  pass; measure RAM/VRAM *during* compile (it can spike — keep within budget).
- [x] **P1.2 — Step-caching (TeaCache / First-Block-Cache)** for FLUX → ~1.5–2×
  fewer compute steps at near-equal quality; low memory cost.
- [x] **P1.3 — Live phase-batching validation** in the running app: a mixed batch
  must do exactly **one** LLM↔image swap; add denoise-progress preview to the UI.
- [x] **P1.4 — SDXL turbo** via DMD2/Lightning LoRA (4–8 steps) → ~1–2 s/image.
- [x] **P1.5 — Frontend polish:** presets, queue drag-reorder, gallery metadata panel.

P1 implementation notes:
- `IMGFAB_TORCH_COMPILE=true` enables guarded compile + warmup; `model.loaded`
  includes a `load_report` with RAM/VRAM snapshots.
- FLUX step caching is configurable with `IMGFAB_FLUX_STEP_CACHE=fb|teacache|off`
  and defaults to nunchaku first-block cache.
- SDXL turbo LoRA support is wired through `IMGFAB_SDXL_TURBO_LORA`; real speed
  numbers still need a GPU-mode run with the chosen LoRA.
- Live phase-batching validation runner: `scripts/phase_batch_check.py`.
- Denoise progress preview uses `job.progress` WebSocket notes in the queue UI.
- Presets, queued-job drag reorder, and gallery metadata are wired in the React UI
  using the existing REST APIs.

### P2 — Optional / later

- [x] **P2.1 — Keep-warm policy** (park the hot model in CPU RAM between swaps to skip
  an SSD reload) — **OFF by default**, gated behind the RAM budget; only engages
  if there's headroom, never causes paging.
- [x] **P2.2 — fp8 / FlashAttention** for attention blocks.
- [x] **P2.3 — LoRA management** for SDXL + FLUX.
- [x] **P2.4 — History/search, export, settings UI.**
- [x] **P2.5 — Quality A/B:** nunchaku fp4 vs int4 vs a GGUF fallback.

P2 implementation notes:
- Keep-warm is controlled by `IMGFAB_KEEP_WARM_MODELS` and only parks image
  backends. The arbiter enforces `IMGFAB_KEEP_WARM_MAX_MODELS` and calls the RAM
  guard before parking; `/api/gpu/free` unloads resident and warm models.
- Attention is controlled by `IMGFAB_ATTENTION_BACKEND=auto|flash|efficient|math|cudnn`.
  The diffusers backend wraps generation in PyTorch's native SDPA selector,
  reports available SDPA kernels, float8 dtype support, and optional
  `flash_attn`/`xformers` package presence in `model.loaded` metadata.
- LoRA management scans `models/lora` (or `IMGFAB_LORA_MODELS_DIR`), exposes
  `/api/loras`, validates queued LoRA ids/weights against the selected image
  model, and lazy-loads selected adapters into SDXL/FLUX diffusers pipelines.
- History/search/export/settings are wired through `/api/images?q=...`,
  `/api/images/{id}/metadata`, PNG downloads, and a read-only settings drawer
  backed by `/api/settings`.
- Quality A/B uses `scripts/quality_ab.py` to queue same-seed comparisons across
  selected image model ids. Nunchaku checkpoints are labeled as `nunchaku-fp4`
  or `nunchaku-int4` when the filename identifies the variant. A true image GGUF
  fallback can be included once such a backend/model is registered; this is
  separate from the existing llama.cpp LLM GGUF path.

---

## Audit — post P0–P2 (2026-06-03)

Every backlog item below (P0.1–P2.5) is **implemented in code** and the project
passes static integrity checks:

- Backend byte-compiles and imports cleanly in stub mode (25 routes registered).
- Frontend `tsc --noEmit` passes; `SettingsPanel`, LoRA picker, presets,
  queue drag-reorder, and gallery search/metadata are all wired into `App.tsx`.
- Environment is real-mode ready: `.venv` with the GPU stack, `frontend/node_modules`,
  `bin/llama/llama-server.exe` (CUDA), and the fp4/fp8/SDXL/GGUF model files all present.
- New helper scripts shipped: `scripts/swap_leak_test.py`, `phase_batch_check.py`,
  `quality_ab.py`.
- Convenient launcher added: **`run.bat`** (REAL mode by default, `run.bat stub`
  for no-GPU mode).

### M1 — Real-GPU validation (the remaining gap)

The code is done; what is **not yet recorded** is a live GPU run confirming the
numbers and invariants. To close M1, run with `IMGFAB_STUB_MODE=false` and capture:

- [ ] **M1.1** — Swap-loop leak test green over ≥3 cycles
  (`python scripts\swap_leak_test.py --cycles 3`): RSS + VRAM return to baseline.
- [ ] **M1.2** — Phase-batching proven live (`python scripts\phase_batch_check.py`):
  a mixed batch does exactly one LLM↔image swap.
- [ ] **M1.3** — SDXL-turbo LoRA real speed numbers (target ~1–2 s/image) with a
  chosen DMD2/Lightning LoRA via `IMGFAB_SDXL_TURBO_LORA`.
- [ ] **M1.4** — `torch.compile` + step-cache speed/VRAM measured against the
  baseline, staying within the ≤26 GB RAM / ≤16 GB VRAM budget.
- [ ] **M1.5** — Quality A/B captured (`python scripts\quality_ab.py`): nunchaku
  fp4 vs int4 vs GGUF fallback.

### P3 — FLUX.2 [klein] (new model family)

- [x] **P3.1 — FLUX.2 [klein] via diffusers + bitsandbytes 4-bit.** Added a new
  `ModelFamily.FLUX2`. klein uses a small **Qwen3** text encoder (not FLUX.2
  [dev]'s 24 GB Mistral), so 9B in bnb-nf4 + model-offload fits 16 GB. Loaded
  with diffusers' `Flux2KleinPipeline` (already in our diffusers 0.38) — we do
  **not** use nunchaku here because its FLUX.2 transformer is still unreleased
  (PR #926). FLUX.2 [dev] (32B + Mistral-24B) is intentionally out of scope.
- [ ] **P3.2 — Live GPU validation.** Needs the model downloaded + a real run to
  confirm VRAM/RAM/speed and tune `flux2_default_steps`/`guidance`.
- [ ] **P3.3 — nunchaku FLUX.2 fast path.** Once nunchaku ships
  `NunchakuFlux2Transformer2DModel`, add an SVDQuant fp4 path (Blackwell) for a
  ~3× speedup, mirroring the existing FLUX.1 nunchaku loader.

P3 implementation notes:
- **To enable:** drop the klein repo into a *folder* under `models/image/` (it
  is multi-file, not a single `.safetensors`); it is auto-detected by its
  `model_index.json`. Example:
  `huggingface-cli download black-forest-labs/FLUX.2-klein-9B --local-dir models/image/flux2-klein-9b`.
- Knobs: `IMGFAB_FLUX2_QUANT` (`bnb-nf4`|`bnb-fp4`|`none`),
  `IMGFAB_FLUX2_OFFLOAD` (`model`|`sequential`|`none`),
  `IMGFAB_FLUX2_DEFAULT_STEPS` (6), `IMGFAB_FLUX2_DEFAULT_GUIDANCE` (4.0).
- Detection, sizing, RAM/VRAM estimates and a stub generate were verified
  end-to-end with a fake klein folder; the real-model run is P3.2.

### P4 — Chat workspace & superapp shell

- [ ] **P4.1 — Full chat tab.** Grow the LLM tab into a ChatGPT-class local tool
  (persistent conversations, markdown/code blocks, stop/regenerate/edit, sampling
  controls, context meter), then a tabbed **superapp** shell. Detailed phased
  plan: [docs/chat-plan.md](docs/chat-plan.md).

P3/UX notes:
- Gallery reworked: the just-generated image is shown large in **full
  resolution** (no upscaled thumbnail), with Copy-to-clipboard, a click-to-zoom
  lightbox, **Show in folder** (`/api/images/{id}/reveal` opens the OS file
  manager), PNG/JSON export, and a history strip of previous results.

---

## Done — M0 (GPU bring-up)

Stack: torch 2.11+cu128 (cap 12,0) · diffusers 0.38 · transformers <5 ·
bitsandbytes · llama.cpp CUDA-13.3 · nunchaku 1.3 (fp4).

| Model | Speed | VRAM |
|-------|-------|------|
| SDXL (NoobAI) | ~5.6 s / 1024² | 11 GB |
| FLUX (Nunchaku fp4) | ~18.7 s / 1024² | 9.8 GB |
| gpt-oss-20B (llama-server) | streaming | 12.5 GB |

All validated end-to-end through the worker (arbiter → backend → gallery) with
`IMGFAB_STUB_MODE=false`. Known issue addressed by this roadmap: the nunchaku
path's encoder loading is RAM-wasteful (P0.1).
