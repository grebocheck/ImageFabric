# HFabric — Roadmap & Prioritized Backlog

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
- `HFAB_TORCH_COMPILE=true` enables guarded compile + warmup; `model.loaded`
  includes a `load_report` with RAM/VRAM snapshots.
- FLUX step caching is configurable with `HFAB_FLUX_STEP_CACHE=fb|teacache|off`
  and defaults to nunchaku first-block cache.
- SDXL turbo LoRA support is wired through `HFAB_SDXL_TURBO_LORA`; real speed
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
- Keep-warm is controlled by `HFAB_KEEP_WARM_MODELS` and only parks image
  backends. The arbiter enforces `HFAB_KEEP_WARM_MAX_MODELS` and calls the RAM
  guard before parking; `/api/gpu/free` unloads resident and warm models.
- Attention is controlled by `HFAB_ATTENTION_BACKEND=auto|flash|efficient|math|cudnn`.
  The diffusers backend wraps generation in PyTorch's native SDPA selector,
  reports available SDPA kernels, float8 dtype support, and optional
  `flash_attn`/`xformers` package presence in `model.loaded` metadata.
- LoRA management scans `models/lora` (or `HFAB_LORA_MODELS_DIR`), exposes
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
numbers and invariants. To close M1, run with `HFAB_STUB_MODE=false` and capture:

- [x] **M1.1** — Swap-loop leak test green over ≥3 cycles
  (`python scripts\swap_leak_test.py --cycles 3`): RSS + VRAM return to baseline.
- [x] **M1.2** — Phase-batching proven live (`python scripts\phase_batch_check.py`):
  a mixed batch does exactly one LLM↔image swap.
- [x] **M1.3** — SDXL-turbo LoRA real speed numbers (target ~1–2 s/image) with a
  chosen DMD2/Lightning LoRA via `HFAB_SDXL_TURBO_LORA`.
- [x] **M1.4** — `torch.compile` + step-cache speed/VRAM measured against the
  baseline, staying within the ≤26 GB RAM / ≤16 GB VRAM budget.
- [x] **M1.5** — Quality A/B captured (`python scripts\quality_ab.py`): nunchaku
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
- M1.1 final: `scripts/swap_leak_test.py` now takes a warm baseline after two
  unmeasured cycles, so one-time Python/torch/diffusers/nunchaku imports are not
  treated as leaks. The measured 3-cycle run passed with default slack:
  baseline RSS 8.77 GB / VRAM 2.09 GB; after cycles 1/2/3 = RSS
  8.89/5.55/8.76 GB and VRAM 2.09/2.09/2.09 GB.
- M1.2 re-run passed in the final environment: queued `LLM/image/LLM/image`,
  started as `llm,llm,image,image`, loaded families `gguf,flux`, swaps = 1.
- M1.3 final: downloaded ByteDance `sdxl_lightning_4step_lora.safetensors` to
  `models/lora/` and ran SDXL with `HFAB_SDXL_TURBO_LORA`, 1024², default-like
  request params. Cold end-to-end run was 19.39 s (includes pipeline + LoRA
  load); warm resident job `c2d54f14ffaf408ab118bf395bf07b7b`, image
  `bb5d91e175cb441981d99b092f11d717`, wrote
  `data/outputs/2026-06-04/20260606_9896.png` with actual metadata
  `steps=4`, `guidance=1.0`; job duration was 1.67 s
  (`data/runtime/m1-sdxl-turbo-warm.json`). Visual note: output is non-blank but
  this base-model/LoRA/prompt combo is not a quality recommendation.
- M1.5 final: downloaded FLUX.2 nunchaku int4 for the attempted comparison, then
  captured `data/runtime/m1-quality-ab-flux2.json` at 768² / 6 steps /
  guidance 4.0. nunchaku-fp4 job `7409b46ebfde47a48915071cd2001aeb`, image
  `a45d0c73e4da4a12af2b2f39e95319c5`, output
  `data/outputs/2026-06-04/20260604_4776.png`, job duration 12.05 s. bnb-nf4
  fallback job `c1e3344f6d354d5db2ad65dcd0cb7a68`, image
  `9bad5b3f0aec4421bb9a4a3672c3742a`, output
  `data/outputs/2026-06-04/20260604_5055.png`, job duration 26.71 s.
  nunchaku-int4 failed on this Blackwell GPU with
  `Please use "fp4" quantization for Blackwell GPUs`; the registry now hides
  FLUX.2 nunchaku-int4 on sm_120 so the UI does not expose a known-broken
  choice. Image-GGUF remains unsupported by this backend, so bnb-nf4 is the
  practical fallback for M1 quality comparison.

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
  top-level, so HFabric imports the sidecar from
  `models/image/flux2-klein-9b-nunchaku/` without patching `.venv`. The Qwen3
  text encoder remains diffusers bitsandbytes 4-bit until the separate nunchaku
  Qwen3 text-encoder support lands.

P3 implementation notes:
- **To enable:** drop the klein repo into a *folder* under `models/image/` (it
  is multi-file, not a single `.safetensors`); it is auto-detected by its
  `model_index.json`. Example:
  `huggingface-cli download black-forest-labs/FLUX.2-klein-9B --local-dir models/image/flux2-klein-9b`.
- Knobs: `HFAB_FLUX2_QUANT` (`bnb-nf4`|`bnb-fp4`|`none`),
  `HFAB_FLUX2_OFFLOAD` (`model`|`sequential`|`none`),
  `HFAB_FLUX2_DEFAULT_STEPS` (6), `HFAB_FLUX2_DEFAULT_GUIDANCE` (4.0),
  `HFAB_FLUX2_DEFAULT_WIDTH`/`HEIGHT` (768).
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
  present. It defaults to CPU-only (`HFAB_TTS_GPU_LAYERS=0`) so it does not
  bypass the shared GPU arbiter. Shipped 2026-06-04.
- [x] **P4.5d — Code assistant workspace.** A Code tab searches/reads local
  repository text files (with `models`, `data`, `.venv`, `node_modules`, and
  `bin` ignored), packages selected files as context, creates a focused LLM
  conversation, and jumps to the LLM tab for streaming/history. Shipped
  2026-06-04.
- [x] **P4.5e1 — Transcription workspace shell.** Added a dedicated Transcribe
  tab plus `/api/transcription/*` endpoints. It scans `models/transcribe` for
  local Whisper models, reports installed engines (`faster-whisper` /
  `openai-whisper`), accepts audio uploads, and writes transcript JSON sidecars
  under `data/outputs/<date>/`. It is model-gated and CPU-first by default, so
  it does not bypass the shared GPU arbiter. Shipped 2026-06-04.
- [x] **P4.5e2 — TTS live validation.** With local `OuteTTS-0.2-500M-Q8_0.gguf`
  and `WavTokenizer-Large-75-F16.gguf`, `llama-tts.exe` produced a non-empty
  CPU-only WAV at `data/outputs/2026-06-04/tts-live-validation.wav` (122,924
  bytes; llama.cpp reported ~3.1 s total runtime). This validates the local
  binary/model path, not voice quality. Shipped 2026-06-04.
- [x] **P4.5e3a — RAG workspace + local embeddings.** Added a dedicated RAG tab
  and `/api/rag/*`: document text/file indexing, persistent SQLite
  `rag_documents`/`rag_chunks`, local CPU-only llama.cpp embeddings via
  `models/embed/nomic-embed-text-v1.5.f16.gguf` on port 8262, vector search, and
  "Send to LLM" with retrieved context/citation slots. Live smoke passed in
  STUB app mode: indexed a one-chunk document and retrieved it as top match for
  a VRAM-arbiter question (score ~0.709); the smoke document was deleted after
  verification. Shipped 2026-06-04.
- [x] **P4.5e3b — Vision workspace.** Added a dedicated Vision tab and
  `/api/vision/*` endpoints using local `llama-mtmd-cli.exe`,
  `models/vision/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf`, and
  `mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf`. It accepts PNG/JPEG uploads,
  saves result JSON sidecars, and defaults to CPU-only (`HFAB_VISION_GPU_LAYERS=0`)
  so it does not bypass the shared GPU arbiter. Live smoke passed in STUB app
  mode: a local PNG was described in ~7.36 s. WEBP is intentionally not accepted
  because llama-mtmd failed to decode it locally. Shipped 2026-06-04.
- [x] **P4.5e3c — Broader model-driven function-calling.** Chat tool mode now
  supports both `generate_image` and `search_documents`. The new Document tool
  lets the LLM reply with `{"tool":"search_documents","query":"...","top_k":5}`;
  the worker runs local RAG search, creates a child LLM job with retrieved
  context, and streams the final answer back into the same assistant message.
  A direct parser/child-job smoke passed for the `search_documents` branch.
  Shipped 2026-06-04.

P3/UX notes:
- Gallery reworked: the just-generated image is shown large in **full
  resolution** (no upscaled thumbnail), with Copy-to-clipboard, a click-to-zoom
  lightbox, **Show in folder** (`/api/images/{id}/reveal` opens the OS file
  manager), PNG/JSON export, and a history strip of previous results.

### P5 — UX & visual polish (comfort + identity)

> Functionally the superapp is complete; what's missing is *feel*. The shell is
> a flat dark grid of native `<select>`s with almost no motion, no brand mark,
> and no window into what the model is doing while it works. P5 makes the LLM and
> image tabs **comfortable to live in**: a real visual identity, "something is
> happening" feedback, styled controls, and a streaming view of the model's
> reasoning. This is presentation only — no change to the memory/VRAM invariants.

**P5.A — Visual identity & design system**
- [x] **P5.A1 — Brand mark + app shell.** SVG HFabric mark (violet tile + woven
  "H") shipped 2026-06-04 as [Logo.tsx](frontend/src/components/Logo.tsx) in the
  header lockup, plus a matching [favicon.svg](frontend/public/favicon.svg).
  *Remaining:* a packaged desktop/window icon for the VS Code-extension shell.
- [x] **P5.A2 — Design tokens (foundation).** Shipped 2026-06-04: Tailwind 4
  `@theme` tokens in [index.css](frontend/src/index.css) — `base` / `surface` /
  `surface-2` / `fg` / `accent` (+ hover) — replacing every ad-hoc hex literal
  (`#0b0d12`, `#e6e8ef`, `bg-[#101219]`, `bg-[#12151c]`). Added a keyboard-only
  `:focus-visible` accent ring and accent `::selection`. *Remaining:* migrate the
  scattered `violet-*` utilities to the `accent` token (one-knob theming for
  P5.A3) and add radii/elevation tokens + success/warn/error colors.
- [ ] **P5.A3 — Theme toggle (optional).** Light/dim/dark variants driven by the
  same tokens, persisted to `localStorage` next to `hfabric.view`.

**P5.B — "Work is happening" feedback (execution animation)**
- [x] **P5.B1 — Global activity indicator.** Shipped 2026-06-04/05: the header dot
  in [ModelStatus.tsx](frontend/src/components/ModelStatus.tsx) swaps to an
  `animate-spin` accent ring whenever any job is running (chat streams included),
  the active model is shown, and a subtle inline **VRAM bar** (used/total from
  `mem.status`) sits in the header.
- [x] **P5.B2 — Animated denoise preview.** Shipped 2026-06-04: the queue keeps
  its per-step progress bar + denoise grid, and `ResultPreview` now shows a
  shimmering skeleton tile while an image job runs (plus a thin shimmer bar over
  the last result) that resolves into the final image. `generating` is derived
  from running image jobs.
- [x] **P5.B3 — Skeleton loaders + transitions.** Shipped 2026-06-04: a `.skeleton`
  shimmer utility + `animate-fade-in` enter transition (Tailwind `@theme`
  keyframes), applied to toasts, queue cards, and gallery tiles. *Remaining:*
  dedicated skeleton placeholders while the conversation/model lists are still
  fetching (needs per-list loading flags).
- [x] **P5.B4 — Toasts.** Shipped 2026-06-04: a module-level toast store +
  `ToastHost` ([Toast.tsx](frontend/src/components/Toast.tsx)). Image `job.done`
  fires a success toast that jumps to History on click; `job.error` fires an
  error toast. Reusable for the P5.D3 control kit. Also added a global
  `prefers-reduced-motion` reset for all P5.B animations.

**P5.C — LLM tab comfort**
- [x] **P5.C1 — Thinking / reasoning panel.** Shipped 2026-06-04:
  [Thinking.tsx](frontend/src/components/Thinking.tsx) parses `<think>…</think>`
  / `<thinking>` out of the assistant stream and renders a collapsible disclosure
  that auto-expands with a spinner while reasoning streams, then auto-collapses
  once the answer begins. *Remaining:* gpt-oss Harmony `analysis`-channel parsing
  (needs confirming how llama.cpp surfaces it) for models that don't use `<think>`.
- [x] **P5.C2 — Composer ergonomics.** Live token/context meter, tok/s + TTFT
  stats, Stop affordance and a busy/"generating" state were already in place; the
  auto-growing composer textarea (caps at 200 px, then scrolls) was added
  2026-06-04. *Remaining:* inline model/persona quick-switch chips.
- [x] **P5.C3 — Styled selectors.** Shipped 2026-06-05: **every** native
  `<select>` in the app is now the shared keyboard-navigable `Select` with badge
  hints — Image (model/LoRA/preset), LLM (model/persona), and the RAG / Vision /
  TTS / Transcribe / Code tabs. *Remaining:* in-dropdown text search/filter.

**P5.D — Image tab comfort**
- [ ] **P5.D1 — Model & LoRA picker as cards.** Replace the native `<select>`s in
  [ImageComposer.tsx](frontend/src/components/ImageComposer.tsx) with visual
  cards carrying the existing quant / est-VRAM / "slow" badges; LoRA picker with
  weight sliders and on/off toggles.
- [x] **P5.D2 — Prompt & size ergonomics.** Aspect-ratio quick buttons
  (respecting the 768² FLUX.2 pin from P3) and a styled sampler/steps/guidance
  control group shipped 2026-06-04. *Remaining:* prompt-history recall.
- [x] **P5.D3 — Reusable control kit.** Shipped: `Select`, `Toast`, `Toggle`,
  `Badge`, and `Slider` are factored into shared components and adopted (all
  native `<select>`s gone; chat tool checkboxes → `Toggle`; queue type chip →
  `Badge`; LoRA weight → `Slider`). Other tabs can now pull from the same kit;
  migrating the remaining ad-hoc number inputs onto `Slider` is incremental.

P5 design constraints:
- Pure presentation: must not add a resident model, change swap behavior, or
  touch the RAM/VRAM guard. Animations stay GPU-cheap (CSS transforms/opacity).
- Keep the **workspace registry** contract from P4.4 — a new tab is still one
  entry in the `workspaces` array; the control kit (P5.D3) plugs into that.
- Accessibility: every restyled control keeps keyboard operability and a focus
  ring; respect `prefers-reduced-motion` for all P5.B animations.

P5 progress (2026-06-04):
- Shared dark **`Select`** ([frontend/src/components/Select.tsx](frontend/src/components/Select.tsx))
  shipped — keyboard-navigable, badge hints (quant/est-VRAM). It is the start of
  the P5.D3 control kit and replaces the native `<select>`s in the Image tab
  (model / LoRA / preset). *Remaining for C3/D3:* apply it to the LLM personas
  and the RAG/Vision/TTS/Transcribe tabs, and add `Slider`/`Toggle`/`Toast`/`Badge`.
- P5.D1 is still a dropdown (with badges), not yet visual cards.
- P5.A/P5.B (brand mark, design tokens, activity indicator, denoise animation,
  skeletons, toasts) and P5.C (thinking panel, composer ergonomics) are still open.

### Shipped 2026-06-04 — Images page rebuild + queue/RAM reliability

Out of the P5 effort plus user feedback, the Images workspace was rebuilt and a
few reliability bugs were fixed (committed to `main`):

- [x] **Images page relayout.** Two-column `composer | (result + queue)`: a wider
  composer, a compact **`ResultPreview`** card, and a dedicated **History** tab
  ([Gallery](frontend/src/components/Gallery.tsx)) for full browsing + metadata.
- [x] **Scroll/visibility fix.** Added `min-h-0` through the flex/grid chain so the
  composer scrolls and the **Queue** button stays pinned and visible (it had been
  pushed off-screen).
- [x] **Robust lightbox.** `vh/vw` absolute-centered image (was blank under the
  old `max-h-full` flex layout). Verified the `/file` endpoint serves real PNGs.
- [x] **Composer persistence.** Model/params/LoRA/preset persist to
  `localStorage` (`hfabric.image.composer`) so switching tabs no longer resets them.
- [x] **Cancel running jobs.** `GenerationCancelled` + `request_stop` threaded
  through backend/arbiter/worker; a **Stop** button in the queue aborts the
  in-flight LLM stream or diffusion denoise (best-effort, marks the job cancelled).
- [x] **FLUX.2 RAM guard fix.** The bnb estimate was ~16 GB vs a measured ~9.8 GB
  peak, causing false pre-load refusals; retuned to ~0.3× the repo + margin.
- [x] **Startup hygiene.** Quieted transient vite proxy errors during backend boot;
  `run.ps1` now frees the embeddings port and sweeps stray `llama-*` children.

### P6 — Real-time voice changer (new)

> The user wants **real-time voice conversion** (mic → target voice → output)
> and notes the gap is the *interface* — existing tools (RVC GUIs, w-okada) work
> but the UX is rough. Our differentiator is a clean, integrated control surface
> in the same superapp shell. This is the most research-y feature on the roadmap:
> latency, dropout-free buffering, and voice quality must be **measured**, so it
> is staged offline-first, like the TTS/Vision/Transcribe tabs were.

**Architecture reality (decided: w-okada / MMVCServerSIO):**
- w-okada **already owns the realtime duplex audio loop, device I/O, and
  virtual-cable output**. We do **not** rebuild that in Python. HFabric detects
  the local install, (later) launches `MMVCServerSIO.exe` as a managed
  subprocess, and **drives its API** — building a cleaner control surface than
  w-okada's own UI (the original gap the user hit).
- A live session **pins the GPU** (w-okada uses it for inference). It needs a
  **voice lane** coordinated with the arbiter — refuse/pause heavy image/LLM
  jobs while a session is live; surface this in the UI.
- Local install on this machine: `D:\MMVCServerSIO` (override with
  `HFAB_VOICE_WOKADA_DIR`); models live in `<dir>\model_dir` as numbered slots
  (each a `params.json` + `.safetensors`/`.pth` + `.index`).

- [~] **P6.1 — Voice detection shell.** Shipped 2026-06-05: a **Voice** tab
  ([VoicePanel.tsx](frontend/src/components/VoicePanel.tsx)) + `/api/voice/status`
  that detects the w-okada install (`MMVCServerSIO.exe`), reads the `model_dir`
  slots (name / type / version / f0 / index / size from each `params.json`), and
  probes whether the server is reachable. `/api/voice/convert` is wired but
  **gated** (503 until the server is up / API is driven). Smoke-tested live
  against the real install: `installed=true`, 1 voice (`chocola_yagiyukiv2`,
  RVC v2). *Remaining:* launch/manage the server + drive its conversion API.
- [~] **P6.2 — Drive the w-okada server.** In progress 2026-06-05: HFabric can
  launch/stop `MMVCServerSIO.exe` as a managed subprocess, proxy w-okada's real
  `GET /info`, `GET /performance`, and `POST /update_settings` API, select a
  model slot, set live params (pitch / index-ratio / protect / f0 method), and
  start/stop the real server-audio stream via `serverAudioStated` /
  `enableServerAudio`. The Worker now treats live voice as a GPU **voice lane**
  and leaves queued image/LLM jobs parked while it is active; starting a live
  session refuses if a GPU job is currently running and frees any idle HFabric
  resident first. *Remaining:* latency measurement and richer w-okada
  performance display.
- [~] **P6.3 — Output routing into other apps.** In progress 2026-06-05:
  HFabric now normalizes w-okada's `serverAudioInputDevices` /
  `serverAudioOutputDevices` from `/info` and surfaces input/output/monitor
  device pickers, sample-rate, chunk-size, and gain controls in the Voice tab.
  Added [voice-routing.md](docs/voice-routing.md) for the one-time
  VB-CABLE/VoiceMeeter setup. *Remaining:* validate the selectors against a
  live server/audio device session and add friendlier handling for unsupported
  sample-rate combinations.
- [~] **P6.4 — The UI (the differentiator).** In progress 2026-06-05: Voice tab
  now includes normalized live metrics from `/performance`, input/output VU
  bars, pitch + formant + index/protect controls, latency↔quality presets
  (chunk/crossfade/extra-buffer), and bypass / PTT controls driven through
  w-okada's `passThrough` setting. *Remaining:* validate live meter values
  against an actual audio stream, add a waveform monitor, and refine the
  performance breakdown once the server is running.

P6 constraints:
- Wrap w-okada (don't reinvent inference or the audio loop); our scope is
  launch/management, the API integration, and a better UI.
- Respect ROADMAP objective #1: the voice lane is checked against the same
  `sysmon` VRAM/RAM budget before a session starts.

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
`HFAB_STUB_MODE=false`. Known issue addressed by this roadmap: the nunchaku
path's encoder loading is RAM-wasteful (P0.1).
