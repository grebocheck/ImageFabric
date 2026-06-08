# HFabric — Roadmap & Prioritized Backlog

> Status: **working app, real-GPU validated (M0/M1).** The arbiter, image + chat
> workspaces, history/browse, and the superapp shell are shipped and in real use.
> What's left is genuinely *in-flight or unbuilt*: the **real-time voice changer**
> (the one phase still in progress), an **engineering safety net** (there are no
> automated tests yet), and the **loose ends** trailing the shipped phases.

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

### P6 — Real-time voice changer (in progress — the only live build)

> Real-time voice conversion (mic → target voice → output). We **wrap w-okada /
> MMVCServerSIO** (it owns the realtime duplex audio loop, device I/O, and
> virtual-cable output) and build a cleaner control surface. Local install:
> `D:\MMVCServerSIO` (override `HFAB_VOICE_WOKADA_DIR`); models in `<dir>\model_dir`
> as numbered slots. A live session **pins the GPU**, so it gets a **voice lane**
> coordinated with the arbiter (refuse/park heavy jobs while live), checked against
> the same `sysmon` budget.
>
> All four sub-items are wired but **none is validated against a real audio
> stream** — that end-to-end validation is the gating work for this phase.

- [~] **P6.1 — Voice detection shell.** Voice tab + `/api/voice/status` detect the
  install, read `model_dir` slots, probe the server. `/api/voice/convert` is wired
  but gated (503). *Remaining:* launch/manage the server + drive its conversion API.
- [~] **P6.2 — Drive the w-okada server.** Launch/stop `MMVCServerSIO.exe` as a
  managed subprocess; proxy `GET /info`, `GET /performance`, `POST /update_settings`;
  select a slot, set live params, start/stop the server-audio stream; park queued
  GPU jobs while live. *Remaining:* latency measurement + richer performance display.
- [~] **P6.3 — Output routing.** Input/output/monitor device pickers, sample-rate,
  chunk-size, gain from `/info`. See [voice-routing.md](docs/voice-routing.md).
  *Remaining:* validate selectors against a live session + friendlier handling of
  unsupported sample-rate combos.
- [~] **P6.4 — The UI (the differentiator).** Live `/performance` metrics, VU bars,
  rolling waveform, timing-stage breakdown, pitch/formant/index/protect controls,
  latency/quality presets, bypass/PTT via `passThrough`. *Remaining:* validate
  meters/timings against a real stream; tune stage labels.

### P10 — Test & CI safety net (new — engineering foundation)

> The memory invariants above *are* the product, and there is currently **no
> regression net**: zero automated tests, no CI, no committed lint/format config.
> The `scripts/*` runners are manual checks against a live GPU backend, not unit
> tests. Crucially, the whole pipeline already runs in **STUB mode with no GPU**,
> so most of this is cheap to build and CI-friendly.

- [x] **P10.1 — Unit tests for the pure logic.** No torch, no GPU
  (`backend/tests/`): `scheduler.select_in_tier`/`plan_queue` (phase-batching
  order + swap count) and `Worker._strip_reasoning` in `test_scheduler.py`;
  `sysmon` budget math (predicted-vs-available; learned-vs-static, headroom,
  keep-warm) in `test_sysmon.py`; `model_profile_service` conservative running-max
  in `test_model_profile.py`. 31 cases.
- [x] **P10.2 — STUB-mode integration test.** `test_stub_integration.py` drives
  the real app over an httpx ASGI client with the lifespan running: posts a mixed
  batch and asserts via the event bus that each family loads once and there is
  exactly **one** swap, then that both images land in the gallery — the hermetic
  `phase_batch_check.py`. Hermetic temp DB + dummy model files (conftest).
- [x] **P10.3 — Frontend unit tests.** Vitest + Testing Library (`npm test`):
  `Thinking.test.ts` (reasoning split states) and `Select.test.tsx` (open / filter
  / choose / no-options). 11 cases. *Remaining:* composer-state (de)serialization.
- [x] **P10.4 — CI workflow.** `.github/workflows/ci.yml` runs on push/PR:
  backend `ruff check` + `pytest` (stub), frontend `tsc -b` + `vitest`.

### P11 — Code health & docs (new)

> Tidy debt that's accumulating quietly while features land.

- [~] **P11.1 — Decompose the oversized screens.** *Started:* the pure,
  view-agnostic logic is extracted into tested helper modules with **no behavior
  change** — `chatHelpers.ts` (import-bundle parsing, sampling coercion, model
  labelling; 152 lines) out of `ChatPanel.tsx` (1125 → 1003) and
  `imageComposerHelpers.ts` (persisted composer state, model ranking, LoRA
  compatibility, formatters; 99 lines) out of `ImageComposer.tsx` (698 → 631),
  each with a vitest suite. *Remaining:* the harder sub-component / hook splits of
  the same two files, `VoicePanel.tsx` (~749), and `backends/image_diffusers.py`
  (~1004, GPU — needs real-hardware verification, not just typecheck).
- [x] **P11.2 — Commit lint/format config.** `backend/pyproject.toml` now holds a
  ruff config (E/F/I/B/C4/UP, with the manual-judgment rules deferred and
  documented) and the pytest config; 75 mechanical issues auto-fixed across the
  backend so the tree is green and CI-enforceable. Frontend gets `vitest` wired in
  `vite.config.ts` + `package.json`. *Remaining:* frontend eslint/prettier.
- [x] **P11.3 — Sync the docs with reality.** `README.md` "Status" now reflects the
  real-GPU-validated M0/M1 state and the actual STUB/REAL default story; the stale
  "Next: milestone M0" section is replaced by a **Testing** section pointing at the
  new suites + CI. *Remaining:* generating the giant knob table from `/api/settings`
  instead of hand-maintaining it.

### P12 — Generation-page & arbiter loose ends (new — gathers shipped-phase tails)

> The shipped P7/P8/P9 phases each left a small, named remainder. Collected here so
> they don't get lost in the "Shipped" log.

- [ ] **P12.1 — Learned-profile management.** A UI list of learned `model_profiles`
  with a reset control (P7.2 tail), plus capture LLM-subprocess VRAM (its
  `load_report` is currently `None`, so the LLM is the one model with no measured
  figure).
- [ ] **P12.2 — Per-job arbiter attribution.** Surface the blocking/swap reason on
  the *exact* queued card (not just the Queue header) and add a keep-warm-eviction
  reason (P7.1 tail).
- [ ] **P12.3 — Inline previews on the Images tab.** Show the swap-plan preview
  inline on the Images-tab queue (P7.4 tail) and add a quick reproduce/vary action
  on the `ResultPreview` card (P8.3 tail).
- [ ] **P12.4 — Memory timeline depth.** Optional process-RSS series + hover
  tooltips on the System-tab sparkline (P7.3 tail).

### P13 — Generation pages, round 2 (Images + LLM comfort)

> Direct user feedback after living in the pages: the model **cards** (P8.4) eat
> too much vertical space, the result strip is too small, the detail view can't
> zoom, and there's no img2img. The card *styling* was liked — keep it, just move
> it into a dropdown.

- [x] **P13.1 — Model picker back to a dropdown.** `Select` gained an optional
  `renderOption` (keyboard-nav / search / click-outside intact); the new
  `ModelPicker.tsx` wraps it so each option shows name + measured-VRAM + all badges
  (family / quant / fast-path / slow / loaded) on **one row**. `ImageComposer`'s
  `ModelCard` grid and the LLM page's plain model `Select` both use it now —
  consistent and far more compact.
- [x] **P13.2 — Bigger, scrollable result strip.** `ResultPreview`'s recent strip
  now shows up to **50** larger (68px) tiles, wrapping into a bounded
  (`max-h-44`) vertically-scrollable area.
- [x] **P13.3 — Zoom in the detail view.** `ZoomableImage.tsx` (wheel zoom +
  drag-pan + double-click reset + on-screen ±/Reset, clamped 1–8×) is used by both
  the `ResultPreview` lightbox (now Esc-closable) and the History detail modal.
- [x] **P13.4 — img2img (image + prompt → image).** *Shipped + real-GPU smoke
  validated:* a
  source-image upload (`POST /api/images/upload` → opaque token under
  `outputs/uploads`, served back for preview), a composer drop/upload slot +
  strength slider (SDXL-gated), the `init_image`/`strength` params flowing through
  the queue, the STUB generation path, and tests (upload round-trip, SDXL-only
  guard, strength clamp, low-step effective-strength guard, end-to-end stub job).
  *Real path:* a SDXL
  `StableDiffusionXLImg2ImgPipeline` sharing the resident pipeline's weights (no
  extra VRAM), validated on RTX 5070 Ti with a 512² / 2-step smoke. FLUX/FLUX.2
  img2img fail fast with a clear message until wired + validated on hardware.
- [x] **P13.5 — Inpainting / region edit (mask).** Built on P13.4: the composer
  now has a source-image mask canvas with brush, lasso/freehand fill, erase, undo,
  clear, and feather controls. The mask is uploaded as a normalized PNG
  (`POST /api/images/upload-mask`) and queued as `mask_image` alongside
  `init_image`; STUB generation and upload round-trips are covered by tests. The
  real SDXL path uses a `StableDiffusionXLInpaintPipeline` view sharing the
  resident pipeline's weights (no extra resident model), validated on RTX 5070 Ti
  with a 512² / 2-step smoke. FLUX/FLUX.2 still fail fast with a clear SDXL-only
  message until their inpaint paths are wired and validated on hardware.
- [x] **P13.6 — Selectable llama backend + context type (KV-cache quantization).**
  The LLM panel gained a *Llama backend* dropdown and a *Context type* dropdown
  next to *Context window*. Two builds are modelled (`LLAMA_BACKENDS` in
  `config.py`): `default` (standard upstream llama.cpp) and `turbo` (a separate
  TurboQuant-patched `llama-server`, path `HFAB_LLAMA_SERVER_BIN_TURBO`). Context
  types (`CONTEXT_TYPES`): `f16` (default), `q8_0` (~2× smaller cache), `q4_0`
  (~4×), and Google DeepMind's **TurboQuant** `turbo3` / `turbo4` — the last two
  only offered when the `turbo` backend is selected. Each preset maps to
  `--cache-type-k/-v` and forces `--flash-attn on` for quantized caches; the
  active backend decides which `llama-server` binary launches. Changing either
  relaunches `llama-server` via the existing ctx/ngl reload path
  (`POST /api/llm/config { backend, context_type }`).
  **No-surprise guarantees:** the API validates the `(backend, context_type)`
  pair *before* committing (422 on an impossible explicit pairing; a backend
  switch that orphans the current type gracefully resets it to `f16` with a note),
  the backend preflights the pair + binary existence before spawning, and
  `llama-server` stderr is drained into a 40-line ring buffer surfaced in startup
  errors (so a bad launch reports *unknown cache type* instead of an opaque exit).
  `q8_0`/`q4_0` work on any recent CUDA build; `turbo3`/`turbo4` need the patched
  build at `HFAB_LLAMA_SERVER_BIN_TURBO`. Covered by 18 tests (arg-builder per
  preset + per backend, stderr tail/bound, `/api/llm/config` get/set, invalid
  pairings, graceful reset, 422s).

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
- **P5 — UX polish.** Brand mark + favicon; Tailwind 4 `@theme` tokens (one-knob
  `accent`, radii/elevation + status colors); light/dim/dark theme toggle; global
  activity indicator + header VRAM bar; animated denoise preview; skeletons,
  toasts, fade-ins (with `prefers-reduced-motion` reset); Thinking/reasoning panel;
  composer ergonomics (auto-grow, token/context meter, quick-switch chips, LLM
  prompt-history); shared keyboard-navigable `Select`/`Toggle`/`Badge`/`Slider`
  control kit replacing every native `<select>`; shared workspace chrome; packaged
  window icon for the VS Code-extension shell.
- **P7 — Memory arbiter depth.** Structured `arbiter.note` events (swap / ram_budget
  / voice_lane / idle) surfaced in the Queue header + System Arbiter panel; learned
  per-model RAM/VRAM profiles in a `model_profiles` SQLite table (conservative
  running max) that the RAM-budget guard + VRAM estimate prefer over the static
  heuristic (`HFAB_LEARN_MEMORY_PROFILES`, `HFAB_LEARNED_RAM_MARGIN_GB`);
  memory-pressure sparkline with swap markers; swap-plan preview via the shared
  pure `scheduler.select_in_tier` + `GET /api/jobs/plan`. *Tails → P12.*
- **P8 — Generation pages: functionality & comfort.** Persisted "Jobs" count;
  image prompt-history recall (↑ dropdown); reproduce/vary from a result
  (Edit-in-composer + Variation); model & LoRA pickers as cards with measured-VRAM
  badges; Harmony (gpt-oss) `reasoning_content` re-wrapped as `<think>` for the
  Thinking panel (and stripped via `_strip_reasoning` everywhere it would pollute a
  prompt/tool-call JSON — `/expand`, generic jobs, tool-call parsing — with the tag
  always closed even on a cut-short stream); in-dropdown search for the shared
  `Select`; chat copy/selection polish. *Tails → P12.*
- **P9 — History / browse rework.** Responsive thumbnail grid (lazy `thumb_url`,
  `limit`/`offset` + Load-more) with a detail modal; combinable filter chips
  (model/family/date/size/LoRA/favorites/tags/free-text) backed by `/api/images`
  query params + `/api/images/stats`; favorites + free-text tags + single delete
  (`PATCH`/`DELETE /api/images/{id}`); multi-select bulk delete + ZIP export
  (`POST /api/images/export`); generation counters (total/today/per-model) in the
  History header + System tab.
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
</content>
</invoke>
