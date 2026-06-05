# HFabric — Roadmap & Prioritized Backlog

> Status: **working prototype.** Core arbiter, image + chat workspaces, and the
> superapp shell are shipped and in real use. The next push is on our two
> differentiators — the **memory arbiter** and the **comfort of the generation
> pages** — plus a real **history/browse** experience.

## Objectives (in priority order)

1. **RAM frugality** — every model load must fit comfortably so the app never
   OOMs, hangs, or spills to the pagefile (pagefile *writes* wear the SSD). Hard
   budget: peak ≈ **≤ 26 GB of 32 GB**. Optimization (small quantized models, no
   wasteful loads) keeps us away from the limit — not aggressive process-killing.
2. **VRAM frugality** — exactly **one resident heavy model** at a time (≤ 16 GB)
   with a safety margin, so we never overflow into shared/system VRAM (that path
   is the 23-min FLUX disaster from M0).
3. **Speed on Blackwell** — fp4/fp8 compute, `torch.compile`, step-caching.

## Memory invariants (do not break these)

- VRAM: exactly one resident heavy model (LLM **or** an image model).
- RAM: a guard checks predicted peak vs. available RAM **before** a load; if it
  wouldn't fit it reports clearly and waits/queues — never pushes the OS into the
  pagefile or leaves the app hung "out of memory".
- Switching models frees the previous one cleanly (expected, made rare by
  phase-batching): llama-server is shut down; diffusers pipelines are `del` +
  `gc.collect()` + `empty_cache()` + `ipc_collect()`. Killing is **not** a routine
  memory tactic.
- Telemetry: process RSS + system available RAM + VRAM are surfaced in
  `/api/health` and over the WebSocket (`mem.status`) so we can *see* pressure.

Code anchors: `backend/app/core/arbiter.py`, `backend/app/util/sysmon.py`.

---

## Active backlog

### P7 — Memory arbiter, depth (differentiator #1)

> The arbiter works, but it is mostly *invisible* and uses *static* estimates.
> This phase makes it observable and self-correcting — our edge over ComfyUI et al.

- [x] **P7.1 — Arbiter decision transparency.** The arbiter/worker now emit a
  structured `arbiter.note` event (`backend/app/core/enums.py` → `ARBITER_NOTE`)
  with the reason: `swap` (unloading X for Y), `ram_budget` (predicted-vs-available
  GB, still raises a clear `MemoryError`), and `voice_lane`/`idle` (queue parked /
  resumed). The Queue header shows blocking reasons (`ram_budget`/`voice_lane`);
  the System tab shows an Arbiter status panel. *Remaining:* per-job attribution on
  the exact queued card, and a keep-warm-eviction reason.
- [x] **P7.2 — Learned memory estimates.** After each real load the arbiter
  records the model's measured RAM (incremental process RSS) and VRAM (process
  reserved) from `load_report` into a `model_profiles` SQLite table (conservative
  running max), primes a `sysmon` cache at startup, and the RAM-budget guard +
  VRAM estimate now prefer the measurement (model-id keyed) over the static
  `size_bytes × factor` heuristic — falling back only for never-loaded models.
  Knobs: `HFAB_LEARN_MEMORY_PROFILES`, `HFAB_LEARNED_RAM_MARGIN_GB`. The Image
  picker labels a learned VRAM figure "measured". *Remaining:* a UI list of
  learned profiles and a reset control; LLM-subprocess VRAM (its report is `None`).
- [x] **P7.3 — Memory-pressure timeline.** The System tab now keeps a rolling
  buffer of `mem.status` samples (last ~90) and draws a sparkline of VRAM-used and
  RAM-% with dashed **swap markers** where the resident model changed. *Remaining:*
  optional process-RSS series and hover tooltips.
- [ ] **P7.4 — Swap-plan preview.** Expose the worker's phase-batching plan: show
  the predicted LLM↔image swap count for the current queue and let the user see
  (and trust) that a mixed batch will swap once, not N times.

### P8 — Generation pages: functionality & comfort (differentiator #2)

> Make the image and text pages genuinely pleasant to live in. Several of these
> were carried over from P5.C/P5.D.

- [x] **P8.1 — Persist the "Jobs" count.** `count` now saves into
  `hfabric.image.composer` with the rest of the composer state (it was the one
  control resetting to 1 on reload).
- [ ] **P8.2 — Image prompt-history recall.** A recall control (↑/dropdown) over
  recent image prompts, mirroring the LLM composer's history (shipped in P5.C2).
- [x] **P8.3 — Reproduce / vary from a result.** The History detail modal has
  *Edit in composer* (restore full params + LoRA + seed; model re-resolved by name)
  and *Variation* (same params, new seed). Wired through `ComposerApply` →
  `ImageComposer.apply`. *Remaining:* a quick "reproduce" action on the Images-tab
  `ResultPreview` card too.
- [ ] **P8.4 — Model & LoRA picker as cards** (was P5.D1). Replace the `<select>`s
  in `ImageComposer.tsx` with visual cards carrying quant / est-VRAM / "slow"
  badges; LoRA picker with inline weight sliders and on/off toggles.
- [ ] **P8.5 — Text page comfort.** gpt-oss Harmony `analysis`-channel parsing for
  the Thinking panel (models that don't emit `<think>`); confirm how llama.cpp
  surfaces it first.
- [ ] **P8.6 — In-dropdown search** for the shared `Select` (was the P5.C3
  remainder) — matters once the model/LoRA/voice lists grow.

### P9 — History / browse rework (the dedicated viewer)

> The current History tab (`Gallery.tsx`) is a single featured image + a flat
> strip + metadata. The user wants a real **place to browse and search** the back
> catalogue. This is a rebuild, not a tweak.

- [x] **P9.1 — Grid gallery + paging.** `Gallery.tsx` is now a responsive
  thumbnail grid (lazy `thumb_url`) that self-fetches `/api/images` with
  `limit`/`offset` and a **Load more** button; tiles open a detail modal.
- [~] **P9.2 — Filters.** Model filter (from `/api/images/stats` counts) + date
  range (today/7d/30d) + free-text prompt/seed search, shown as removable chips
  and combinable. Backend: `model`/`date_from`/`date_to` query params on
  `/api/images`. *Remaining:* size and LoRA filters; a true `family` column
  (today the snapshot only stores the model *name*).
- [~] **P9.3 — Favorites, tags, delete.** Single delete (row + files) shipped via
  `DELETE /api/images/{id}`. *Remaining:* favorites + free-text tags (needs new
  SQLite columns) and filtering on them.
- [~] **P9.4 — Bulk + export.** Multi-select **Select** mode with bulk delete
  shipped; per-item PNG/JSON export + "Show in folder" kept in the detail modal.
  *Remaining:* bulk export bundle.
- [x] **P9.5 — Generation counters.** `/api/images/stats` returns total / today /
  per-model counts; History header shows "N total · M today" and per-model counts
  feed the model filter. *Remaining:* surface the same feed in the System tab.

### P6 — Real-time voice changer (in progress)

> Real-time voice conversion (mic → target voice → output). We **wrap w-okada /
> MMVCServerSIO** (it owns the realtime duplex audio loop, device I/O, and
> virtual-cable output) and build a cleaner control surface — the original gap the
> user hit. Local install: `D:\MMVCServerSIO` (override `HFAB_VOICE_WOKADA_DIR`);
> models in `<dir>\model_dir` as numbered slots. A live session **pins the GPU**,
> so it gets a **voice lane** coordinated with the arbiter (refuse/park heavy jobs
> while live), checked against the same `sysmon` budget.

- [~] **P6.1 — Voice detection shell.** Voice tab + `/api/voice/status` detect the
  install, read `model_dir` slots, and probe the server. `/api/voice/convert` is
  wired but gated (503 until the server is driven). *Remaining:* launch/manage the
  server + drive its conversion API.
- [~] **P6.2 — Drive the w-okada server.** Launch/stop `MMVCServerSIO.exe` as a
  managed subprocess; proxy `GET /info`, `GET /performance`, `POST /update_settings`;
  select a slot, set live params, start/stop the server-audio stream. Worker parks
  queued jobs while voice is live and refuses a live start if a GPU job is running.
  *Remaining:* latency measurement and richer performance display.
- [~] **P6.3 — Output routing.** Input/output/monitor device pickers, sample-rate,
  chunk-size, gain in the Voice tab from `/info`. See
  [voice-routing.md](docs/voice-routing.md) for VB-CABLE/VoiceMeeter setup.
  *Remaining:* validate selectors against a live audio session + friendlier
  handling of unsupported sample-rate combos.
- [~] **P6.4 — The UI (the differentiator).** Live metrics from `/performance`,
  VU bars, rolling waveform, timing-stage breakdown, pitch/formant/index/protect
  controls, latency/quality presets, bypass/PTT via w-okada `passThrough`.
  *Remaining:* validate meters/timings against a real stream; tune stage labels.

### P5 — UX polish, remaining tails

Most of P5 shipped (see Shipped). Still open:

- [ ] **P5.A3 — Theme toggle (optional).** Light/dim/dark variants from the shared
  tokens, persisted to `localStorage` next to `hfabric.view`.
- [ ] **P5.A2 tail — token migration.** Migrate scattered `violet-*` utilities to
  the `accent` token (one-knob theming) and add radii/elevation + success/warn/
  error color tokens.
- [ ] **P5.A1 tail — packaged window icon** for the VS Code-extension shell.
- [ ] **P5.B3 tail — list skeletons** while conversation/model lists fetch (needs
  per-list loading flags).

P5 design constraints (still binding): pure presentation — no resident model, no
swap-behavior change, no touching the RAM/VRAM guard; animations stay GPU-cheap
(CSS transforms/opacity); a new tab is still **one entry** in the `workspaces`
array (P4.4 registry); every restyled control keeps keyboard operability + focus
ring and respects `prefers-reduced-motion`.

---

## Shipped (condensed)

Done and in use. Kept terse on purpose — detailed run logs live in
`data/runtime/*.json`, not here.

- **M0 — GPU bring-up.** Stack: torch 2.11+cu128 (cap 12,0) · diffusers 0.38 ·
  transformers <5 · bitsandbytes · llama.cpp CUDA-13.3 · nunchaku 1.3 (fp4).
  Validated end-to-end (arbiter → backend → gallery), `HFAB_STUB_MODE=false`.

  | Model | Speed | VRAM |
  |-------|-------|------|
  | SDXL (NoobAI) | ~5.6 s / 1024² | 11 GB |
  | FLUX (Nunchaku fp4) | ~18.7 s / 1024² | 9.8 GB |
  | gpt-oss-20B (llama-server) | streaming | 12.5 GB |

- **P0 — Memory hygiene.** Nunchaku FLUX borrows encoders without the 16 GB read
  (`NunchakuT5EncoderModel` int4 + CLIP-L + non-gated VAE); RAM/VRAM telemetry +
  pre-load guard (`sysmon.py`) in `/api/health`, `/api/models`, `mem.status`;
  swap-loop leak runner (`scripts/swap_leak_test.py`); raw fp8 FLUX flagged
  slow/high-mem; llama-server confirmed mmap + full-offload (`-ngl 999`).
- **P1 — Speed & live UX.** `HFAB_TORCH_COMPILE` guarded compile + warmup;
  `HFAB_FLUX_STEP_CACHE=fb|teacache|off` (default first-block); SDXL turbo LoRA
  (`HFAB_SDXL_TURBO_LORA`); live phase-batching (`scripts/phase_batch_check.py`);
  denoise progress preview; presets, queue drag-reorder, gallery metadata.
- **P2 — Optional.** Keep-warm (`HFAB_KEEP_WARM_MODELS` / `_MAX_MODELS`, RAM-guarded,
  off by default); attention backend (`HFAB_ATTENTION_BACKEND`); LoRA management
  (`/api/loras`, validated + cache-bounded by `HFAB_IMAGE_LORA_CACHE_MAX`);
  history/search/export + read-only settings drawer; quality A/B
  (`scripts/quality_ab.py`).
- **M1 — Real-GPU validation** (RTX 5070 Ti). Swap-loop steady-state stable;
  phase-batching does one swap for a mixed batch; SDXL-turbo warm ~1.67 s/image;
  FLUX nunchaku fp4 12-step 768² ~16 s with first-block cache.
- **P3 — FLUX.2 [klein].** New `ModelFamily.FLUX2` via diffusers (Qwen3 encoder,
  bnb-nf4 + model-offload) and an experimental nunchaku SVDQuant fp4 sidecar.
  Knobs: `HFAB_FLUX2_QUANT/_OFFLOAD/_DEFAULT_STEPS/_GUIDANCE/_WIDTH/_HEIGHT`. Enable
  by dropping the multi-file klein repo under `models/image/` (auto-detected by
  `model_index.json`). FLUX.2 [dev] (32B + Mistral-24B) is out of scope.
- **P4 — Chat workspace & superapp shell.** Real chat (persistent conversations,
  markdown/code, stop/regenerate/edit, sampling + personas + tok/s + TTFT);
  chat→image bridge (`/image …`) + model-driven `generate_image`/`search_documents`
  tools; command palette (Ctrl+K), search, export, System monitor, declarative
  **workspace registry**; import bundles; Notes, TTS, Code, Transcribe, RAG (local
  embeddings), and Vision workspaces (all model-gated, CPU-first, GPU-arbiter-safe).
- **P5 (most) — UX polish.** Brand mark + favicon; Tailwind 4 `@theme` tokens;
  global activity indicator + header VRAM bar; animated denoise preview; skeletons,
  toasts, fade-ins (with `prefers-reduced-motion` reset); Thinking/reasoning panel;
  composer ergonomics (auto-grow, token/context meter, quick-switch chips, LLM
  prompt-history); shared keyboard-navigable `Select`/`Toggle`/`Badge`/`Slider`
  control kit replacing every native `<select>`; shared workspace chrome.
- **Images page rebuild + reliability.** Two-column composer | (result + queue);
  scroll/visibility fix; robust lightbox; composer persistence
  (`hfabric.image.composer`); cancel running jobs (`request_stop` →
  `GenerationCancelled`); FLUX.2 RAM-guard retune; startup hygiene.
- **Long-session image stabilization.** Worker calls `GpuBackend.after_job(...)`
  after every job; diffusers backend runs gc/`empty_cache`/`ipc_collect`, bounds
  runtime LoRA adapters, and soft-recycles the resident pipeline on CUDA-memory
  drift. Tunables: `HFAB_IMAGE_CLEANUP_AFTER_EACH_JOB`, `HFAB_IMAGE_LORA_CACHE_MAX`,
  `HFAB_IMAGE_RECYCLE_CUDA_GROWTH_GB`, `HFAB_IMAGE_RECYCLE_MIN_JOBS`. Runner:
  `scripts/sdxl_resident_drift_test.py`.

### Hard-won facts (load-bearing constraints — don't relearn the hard way)

- **FLUX.2 klein is pinned to 768²** on the 16 GB GPU: a warm 6-step run is ~1.5 s
  but sampled VRAM-free dipped to ~0.22 GB — 1024² is not safe by default.
- **nunchaku-int4 FLUX.2 is broken on Blackwell (sm_120)** ("use fp4 quantization
  for Blackwell"); the registry hides it. Use **fp4**; bnb-nf4 is the practical
  fallback. **Image-GGUF is unsupported** by this backend (separate from the
  llama.cpp LLM GGUF path).
- **`torch.compile` fails on the nunchaku transformer** in Inductor (`aten.addmm`);
  the backend auto-rolls-back to the original transformer and continues.
- **Cold-start RSS ~5.5–8.8 GB is not a leak** — it's one-time torch/diffusers/
  nunchaku imports. `swap_leak_test.py` takes a warm baseline after two unmeasured
  cycles so these aren't flagged.
- When the validated FLUX.2 repo *folder* exists, the registry hides the original
  single-file `.safetensors` so it's a conversion source, not a duplicate target.

---

## Where to add the next thing

- A new workspace tab = one entry in the `workspaces` array (P4.4 registry) + a
  component using the shared control kit + chrome.
- Anything touching model loading goes through the arbiter (`ensure`/`free_all`)
  and the `sysmon` budget — never load a model directly.
- New env knobs follow the `HFAB_*` convention and are surfaced in `/api/settings`.
