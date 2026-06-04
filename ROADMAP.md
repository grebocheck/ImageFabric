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
- [x] **M1.2** — Phase-batching proven live (`python scripts\phase_batch_check.py`):
  a mixed batch does exactly one LLM↔image swap.
- [ ] **M1.3** — SDXL-turbo LoRA real speed numbers (target ~1–2 s/image) with a
  chosen DMD2/Lightning LoRA via `IMGFAB_SDXL_TURBO_LORA`.
- [x] **M1.4** — `torch.compile` + step-cache speed/VRAM measured against the
  baseline, staying within the ≤26 GB RAM / ≤16 GB VRAM budget.
- [ ] **M1.5** — Quality A/B captured (`python scripts\quality_ab.py`): nunchaku
  fp4 vs int4 vs GGUF fallback.

M1 live validation notes (2026-06-04, RTX 5070 Ti / REAL mode):
- M1.1 strict cold-baseline run failed after cycle 1: VRAM returned
  1.63→1.91 GB, but backend RSS stayed at 5.58 GB from first torch/diffusers
  imports, over the script's 1 GB RAM slack. A warm-process 3-cycle run passed
  with `--ram-slack-gb 6`: baseline RSS 5.58 GB; after cycles 1/2/3 =
  6.09/6.05/6.01 GB; VRAM 1.91/1.89/1.91 GB. Conclusion: steady-state is
  stable, but the strict cold-start criterion is not green yet.
- M1.2 passed: queued LLM/image/LLM/image started as LLM/LLM/image/image;
  loaded families were `gguf`, `flux`; family swaps = 1.
- M1.3 blocked: `models/lora` contains no SDXL turbo LoRA and
  `/api/settings` reports `sdxl_turbo_lora=null`, `loras=0`.
- M1.4 measured with FLUX nunchaku fp4, 12 steps, 768²: step-cache off =
  19.26 s, peak RSS 12.72 GB, peak VRAM 11.78 GB; first-block cache (`fb`) =
  15.98 s, peak RSS 12.78 GB, peak VRAM 11.75 GB. `torch.compile=true`
  currently fails on the nunchaku transformer in Inductor (`aten.addmm`), so
  the backend now rolls back to the original transformer and continues; fallback
  run completed in 18.53 s, peak RSS 12.81 GB, peak VRAM 12.01 GB.
- M1.5 partial: safe A/B captured for available image models only:
  FLUX nunchaku fp4 image `7de8fd274dae4c8abc8d088810050f37` and SDXL image
  `ec44fa6ed35d439180f2a98b594a5e2c` (`data/runtime/m1-quality-ab-safe.json`).
  Target fp4-vs-int4-vs-image-GGUF remains blocked because no int4/GGUF image
  model is registered.

### P3 — FLUX.2 [klein] (new model family)

- [x] **P3.1 — FLUX.2 [klein] via diffusers + bitsandbytes 4-bit.** Added a new
  `ModelFamily.FLUX2`. klein uses a small **Qwen3** text encoder (not FLUX.2
  [dev]'s 24 GB Mistral), so 9B in bnb-nf4 + model-offload fits 16 GB. Loaded
  with diffusers' `Flux2KleinPipeline` (already in our diffusers 0.38) — we do
  **not** use nunchaku here because its FLUX.2 transformer is still unreleased
  (PR #926). FLUX.2 [dev] (32B + Mistral-24B) is intentionally out of scope.
- [x] **P3.2 — Live GPU validation.** Validated on the local 16 GB GPU through
  the normal REST queue + WebSocket progress + arbiter + diffusers path. The
  checked profile is 768x768, 6 steps, guidance 4.0, seed 20260604; runtime
  report: `data/runtime/p3-flux2-live-check.json`.
- [x] **P3.3 — nunchaku FLUX.2 transformer fast path.** Added an experimental
  SVDQuant fp4 runtime for FLUX.2 klein via the local sidecar
  `NunchakuFlux2Transformer2DModel` code shipped with the quantized model. The
  official nunchaku package on this machine still does not expose the class
  top-level, so ImageFabric imports the sidecar from
  `models/image/flux2-klein-9b-nunchaku/` without patching `.venv`. The Qwen3
  text encoder remains diffusers bitsandbytes 4-bit until the separate nunchaku
  Qwen3 text-encoder support lands.

P3 implementation notes:
- **To enable:** drop the klein repo into a *folder* under `models/image/` (it
  is multi-file, not a single `.safetensors`); it is auto-detected by its
  `model_index.json`. Example:
  `huggingface-cli download black-forest-labs/FLUX.2-klein-9B --local-dir models/image/flux2-klein-9b`.
- Knobs: `IMGFAB_FLUX2_QUANT` (`bnb-nf4`|`bnb-fp4`|`none`),
  `IMGFAB_FLUX2_OFFLOAD` (`model`|`sequential`|`none`),
  `IMGFAB_FLUX2_DEFAULT_STEPS` (6), `IMGFAB_FLUX2_DEFAULT_GUIDANCE` (4.0),
  `IMGFAB_FLUX2_DEFAULT_WIDTH`/`HEIGHT` (768).
- Detection, sizing, RAM/VRAM estimates and a stub generate were verified
  end-to-end with a fake klein folder before the real-model run.
- 2026-06-04: the local single-file transformer
  `models/image/flux-2-klein-9b.safetensors` was converted into the validated
  diffusers repo folder `models/image/flux2-klein-9b/` using cached
  tokenizer/text-encoder/VAE files. The Hugging Face token on this machine can
  list the gated repo but cannot download transformer weights directly (401), so
  this folder is the local runtime source.
- Live result: job `562d3d5fe1a5458ea9db8d964151e2dd`, image
  `e6f4d2ce41d240ba931a61779dc3066b`, output
  `data/outputs/2026-06-04/20260604_7159.png`. Runtime was ~41.94 s at 768x768;
  peak sampled process RSS 9.79 GB, peak sampled VRAM 14.8 GB, minimum sampled
  VRAM free 2.3 GB. 1024x1024 is not the default for FLUX.2 on this 16 GB GPU.
- When the validated repo folder exists, the registry hides the original-format
  FLUX.2 `.safetensors` from the UI so it remains a conversion source rather
  than a duplicate runtime target.
- 2026-06-04 P3.3: downloaded
  `models/image/flux2-klein-9b-nunchaku/svdq-fp4_r32-FLUX.2-klein-9B-Nunchaku.safetensors`
  plus sidecar `transformer_flux2.py` / `torch_transfer_utils.py`. Registry now
  detects it as `flux2` + `nunchaku-fp4`, and the UI prefers it over the bnb
  FLUX.2 entry.
- Live nunchaku result: cold run job `6693e6d4ee514eda94876833afd0dc97`, image
  `518f80ceebde4a00a99f0f3d31c5080e`, report
  `data/runtime/p3-flux2-nunchaku-live-check.json`; 768x768 / 6 steps took
  ~26.41 s end-to-end, peak sampled VRAM 14.81 GB. Warm resident run job
  `2bc7eef584184e0698fe124c899379ea`, image
  `e7bf5935f81c46c69c425d6d35924051`, report
  `data/runtime/p3-flux2-nunchaku-warm-check.json`; 6 steps took ~1.55 s but
  sampled VRAM free dipped to ~0.22 GB, so FLUX.2 remains pinned to 768x768 by
  default on this 16 GB GPU.

### P4 — Chat workspace & superapp shell

- [x] **P4.1 — Chat C1 (real chat tab).** Persistent conversations, sidebar,
  markdown + code blocks with copy, stop/regenerate/edit, per-conversation model
  settings, context meter. Shipped 2026-06-04.
- [x] **P4.2 — Chat C2 (sampling + personas + stats).** Full sampling controls
  (top_p/top_k/min_p/repeat_penalty/seed/stop), persona presets, and client-side
  tokens/sec + TTFT. Shipped 2026-06-04.
- [x] **P4.3 — Chat→image bridge (C3.3).** `/image <prompt>` in chat queues an
  image job on the shared arbiter and renders the result inline (persisted).
  Shipped 2026-06-04.
- [x] **P4.3b — Model-driven `generate_image` tool (C3.4).** Chat can enable an
  Image tool; structured LLM `generate_image` replies queue child image jobs on
  the shared arbiter and stream the result back into the conversation. Shipped
  2026-06-04.
- [x] **P4.4 — Superapp shell (C4.1/C4.2/C4.3).** Command palette (Ctrl+K) with
  navigation + actions, conversation search, conversation export to Markdown, a
  live **System** monitor tab (VRAM/RAM/runtime from `mem.status`), and a
  declarative **workspace registry** (tabs are one `workspaces` array — adding a
  tab is one entry). Shipped 2026-06-04.
- [x] **P4.5a — Import conversations/presets/personas.** Importable JSON bundles
  restore conversations with messages plus image/LLM presets; persona presets
  are covered because they are stored as `llm` presets. Shipped 2026-06-04.
- [x] **P4.5b — Notes/Scratch workspace.** Persistent SQLite-backed notes with
  search, autosave, create/delete, and a workspace-registry tab. Shipped
  2026-06-04.
- [x] **P4.5c — TTS workspace + gated generation.** A dedicated TTS tab reports
  `llama-tts.exe`, scans `models/tts` for local `.gguf` voice/acoustic models,
  and can generate WAV files through the local binary once a local model is
  present. It defaults to CPU-only (`IMGFAB_TTS_GPU_LAYERS=0`) so it does not
  bypass the shared GPU arbiter. Shipped 2026-06-04.
- [x] **P4.5d — Code assistant workspace.** A Code tab searches/reads local
  repository text files (with `models`, `data`, `.venv`, `node_modules`, and
  `bin` ignored), packages selected files as context, creates a focused LLM
  conversation, and jumps to the LLM tab for streaming/history. Shipped
  2026-06-04.
- [ ] **P4.5e — Remaining superapp + model-gated chat.** More workspace tabs
  (transcription/whisper), broader model-driven function-calling, **vision**
  (needs a multimodal GGUF), **RAG** (needs an embedding model), and live TTS
  validation once a `models/tts/*.gguf` model is installed. Phased plan:
  [docs/chat-plan.md](docs/chat-plan.md).

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
