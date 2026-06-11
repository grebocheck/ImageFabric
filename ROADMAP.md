# HFabric — Roadmap & Prioritized Backlog

> Status: **working app, real-GPU validated (M0/M1), test/CI safety net in place
> (143 tests green).** Rebuilt 2026-06-11 from the findings of the
> [project audit](docs/audit-2026-06.md) (overall grade **B / 7.5**). The backlog
> below is ordered by risk: lock the front door (P14), validate the one unproven
> phase (P6), then harden data/tests, then pay down code health, then grow.
>
> Marking: `[ ]` not started · `[~]` in progress / partially done · `[x]` done.

## Objectives (in priority order)

1. **RAM frugality** — every model load must fit comfortably so the app never
   OOMs, hangs, or spills to the pagefile (pagefile *writes* wear the SSD). Hard
   budget: peak ≈ **≤ 26 GB of 32 GB**. Optimization (small quantized models, no
   wasteful loads) keeps us away from the limit — not aggressive process-killing.
2. **VRAM frugality** — exactly **one resident heavy model** at a time (≤ 16 GB)
   with a safety margin, so we never overflow into shared/system VRAM (that path
   is the 23-min FLUX disaster from M0).
3. **Speed on Blackwell** — fp4/fp8 compute, `torch.compile`, step-caching.
4. **Trustworthy by default** *(new, from the audit)* — the API must be safe to
   leave running: not reachable by strangers, debuggable after a crash, and
   restorable after a disk failure.

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

### P14 — Security & network exposure (NEW — do first; audit W1)

> The API has **no authentication** and the local `.env` binds it to
> `0.0.0.0:80`. Any LAN device can delete images, relaunch llama-server, start
> the voice subprocess, and pop the desktop file manager via
> `POST /api/images/{id}/reveal`. CORS only restricts *browsers* — it is not an
> auth layer. Until this phase lands, treat LAN exposure as unsafe.

- [ ] **P14.1 — Decide and enforce the bind posture.** Revert the local `.env` to
  the code default (`127.0.0.1:8260`) *or* make LAN exposure deliberate: keep
  `HFAB_HOST=0.0.0.0` only together with P14.2 auth. Add a loud startup warning
  (log + `/api/health` field + UI toast) whenever the server is bound to a
  non-loopback address without a token configured. Document the threat model in
  README ("local single-user app; LAN exposure requires HFAB_API_TOKEN").
- [ ] **P14.2 — Optional bearer-token auth.** New `HFAB_API_TOKEN` setting; when
  set, a middleware rejects any request without `Authorization: Bearer <token>`
  (and the WebSocket without a `?token=` query param). The frontend reads the
  token from a small login prompt persisted in `localStorage` and attaches it in
  `api/client.ts` + `useEvents.ts`. Empty/unset token = current open behavior on
  loopback. Tests: 401 paths, WS rejection, health stays open (or not — decide).
- [ ] **P14.3 — Gate desktop-reaching endpoints to loopback.** `reveal_image`
  (spawns `explorer`/`open`/`xdg-open`) must refuse requests whose client address
  is not `127.0.0.1/::1`, regardless of token — a remote caller has no business
  driving the local desktop. Audit for other desktop/process endpoints as they
  appear (voice launch falls under the token, not loopback, since the UI may be
  remote-legit later).
- [ ] **P14.4 — Upload/content hardening sweep.** Verify every upload path
  enforces its size cap *before* reading the body where possible
  (`image_upload_max_mb`, `transcription_max_upload_mb`, `vision_max_upload_mb`,
  `voice_max_upload_mb`); confirm Pillow re-encodes (not just opens) uploaded
  images so crafted files never land raw in `outputs/uploads`; add a test per
  cap. One pass, mostly verification — the token guard in `util/uploads.py` is
  already right.

### P6 — Real-time voice changer (carried — the one unproven phase; audit W2)

> Real-time voice conversion (mic → target voice → output) wrapping w-okada /
> MMVCServerSIO (local install `D:\MMVCServerSIO`, override
> `HFAB_VOICE_WOKADA_DIR`). A live session pins the GPU via a **voice lane**
> coordinated with the arbiter. All four sub-items are wired but **none is
> validated against a real audio stream** — that end-to-end validation is the
> gating work, and ~530 lines of `api/voice.py` are effectively unverified until
> it happens.

- [~] **P6.1 — Voice detection shell.** Detection, `model_dir` slots, server
  probe shipped. *Remaining:* none beyond the live validation in P6.5.
- [~] **P6.2 — Drive the w-okada server.** Launch/stop as managed subprocess,
  settings proxy, slot select, queue parking shipped. *Remaining:* latency
  measurement + richer performance display.
- [~] **P6.3 — Output routing.** Device pickers, sample-rate, chunk, gain
  shipped (see [voice-routing.md](docs/voice-routing.md)). *Remaining:* validate
  selectors live + friendlier handling of unsupported sample-rate combos.
- [~] **P6.4 — The UI (the differentiator).** Live metrics, VU bars, waveform,
  pitch/formant controls, presets, PTT shipped. *2026-06 UX rework (user
  feedback):* the page is now a guided setup flow (Engine → Voice → Audio
  devices → Go live) with the input/output/monitor pickers promoted to a
  first-class step, an explicit **Monitor "hear myself" toggle** (auto-picks the
  output device, gain beside it), debounced auto-apply for all routing changes
  with an applying/applied hint, and pure logic extracted to `voiceHelpers.ts`
  (+10 tests; `VoiceMeters.tsx` / `VoicePanelParts.tsx` split keeps the panel at
  ~650 lines). *Remaining:* validate meters/timings against a real stream; tune
  stage labels.
- [ ] **P6.5 — End-to-end live validation (gates the phase).** With a real mic +
  virtual cable: start a session, confirm conversion + monitor output, measure
  round-trip latency at 2–3 chunk sizes, confirm the voice lane actually parks a
  queued image job and resumes it after stop, confirm llama/image jobs cannot
  steal the GPU mid-session, and write the measured numbers into this file.
  Fix what breaks; only then mark P6.1–P6.4 done.

### P15 — Reliability & data layer (NEW — audit W3, W7)

> The schema is migrated by a hand-rolled `ALTER TABLE` helper, nothing is
> logged to disk, and there is no backup story. Each is cheap now and expensive
> after the tenth table / first lost database.

- [ ] **P15.1 — Real migrations.** Introduce Alembic (async SQLite) with the
  current schema as revision 0; fold the `_ensure_image_columns` logic into a
  proper revision and delete it. `init_db()` runs `upgrade head` at startup.
  Document "how to add a column" in the README dev section. Test: fresh DB and
  a pre-migration DB both come up.
- [ ] **P15.2 — Structured file logging.** Rotating file handler under
  `data/logs/` (size-capped, ~5×10 MB): startup config summary, every arbiter
  note (swap / refusal / warm-evict with the numbers), job start/done/error with
  duration, llama-server stderr tail on failure, unhandled exceptions. Plain
  `logging` is fine; the point is post-mortem debuggability (the "it hung
  overnight" case), not an ELK stack.
- [ ] **P15.3 — Backup & restore story.** A `scripts/backup.py` (or `.ps1`) that
  snapshots `data/hfabric.db` (using the SQLite backup API, not file-copy of a
  live WAL DB) plus a manifest of `outputs/`; README section on what to back up
  and how to restore. Optional: a retention-N rotation.
- [ ] **P15.4 — Orphan-process audit.** If the backend dies hard, can
  `llama-server` / `MMVCServerSIO` outlive it and hold VRAM? Track child PIDs in
  a pidfile and reap stale ones on startup (with a log line), mirroring the
  existing orphan-job requeue. Test with a simulated kill in stub mode where
  possible.

### P16 — Test depth & quality gates (NEW — audit W4; absorbs P11.2 tail)

> The core (scheduler/sysmon/arbiter/gallery) is well covered; the edges —
> chat, rag, tts, transcription, vision, voice, notes, presets, code routers and
> chat/embedding services — have **zero** tests, and CI measures nothing, so
> holes stay invisible.

- [ ] **P16.1 — Coverage measurement in CI.** Add `pytest-cov` to the backend CI
  job, publish the report in the run summary, and set a floor that today's suite
  already passes (start ~55–60% on `app/`, excluding `backends/image_diffusers.py`
  and other GPU-only modules via `.coveragerc`). Ratchet the floor up as P16.2
  lands — never down.
- [ ] **P16.2 — Router tests for the untested half.** Stub-mode httpx tests per
  router, reusing the `test_stub_integration.py` conftest: chat (conversation
  CRUD, message flow, `/image` bridge, regenerate/edit), rag (ingest → search →
  cited answer path with a fake embed server), notes/presets/code (CRUD +
  validation), tts/transcription/vision (request validation + the
  no-model/no-binary error paths — the subprocess itself can be monkeypatched),
  voice (status/config endpoints with a fake install dir). Target: every router
  file imported by `main.py` has a test file.
- [ ] **P16.3 — Frontend lint in CI (P11.2 tail).** eslint (typescript-eslint +
  react-hooks) + prettier, config committed, `npm run lint` wired into
  `.github/workflows/ci.yml`. Auto-fix the initial sweep in its own commit so
  review stays readable.
- [ ] **P16.4 — Frontend flow tests.** Testing-Library tests for the three
  highest-value flows: ChatPanel (send → streamed reply → thinking panel split),
  Gallery (filter chips combine + bulk select), QueuePanel (job states + cancel).
  Mock `api/client.ts` + the WS hook; no real backend.
- [ ] **P16.5 — Real-GPU smoke checklist.** One doc (`docs/gpu-smoke.md`)
  consolidating the manual runners (`scripts/swap_leak_test.py`,
  `phase_batch_check.py`, `sdxl_resident_drift_test.py`, `image_live_bench.py`,
  `kv_cache_*.py`) into a 15-minute ordered checklist with expected numbers from
  M1, to run after any torch/diffusers/driver bump. The scripts exist — the
  missing artifact is the checklist with pass criteria.

### P17 — Code health round 2 (carries P11.1; audit W5, W6, W9)

> Helper extraction (P11.1 first half) proved the pattern: pure logic out, tests
> on, no behavior change. Now the hard splits — plus the reproducibility and
> hygiene items that protect the verified GPU environment.

- [ ] **P17.1 — Split `ChatPanel.tsx` (1070).** Extract `useConversation` (load /
  send / stream / regenerate state machine) and `useChatStream` (WS token
  assembly) hooks plus `MessageList` / `MessageComposer` subcomponents. Behavior
  frozen by P16.4's flow test written *first*.
- [ ] **P17.2 — Split `backends/image_diffusers.py` (1435, GPU).** One module per
  family loader (`sdxl.py`, `flux.py`, `flux2.py`, `qwen_z.py`) + shared
  infrastructure (`memory.py`: cleanup/recycle/LoRA cache; `pipelines.py`:
  img2img/inpaint views). Pure import-shuffle commits, each followed by a
  real-hardware smoke (P16.5 checklist) — typecheck alone does not validate this
  file.
- [ ] **P17.3 — Split the remaining big screens.** `VoicePanel.tsx` (749 — after
  P6.5 so validation isn't chasing a moving target), `ImageComposer.tsx` (697 —
  extract the mask/source-image block and the param form), `Gallery.tsx` (632 —
  extract filter bar + detail modal).
- [ ] **P17.4 — Generate API types from OpenAPI.** Replace the hand-maintained
  halves of `types.ts` (515 lines) with `openapi-typescript` output generated
  from the FastAPI schema (`npm run gen:api`, checked-in output, CI check that it
  is current). Kills the backend↔frontend drift class; also makes the README knob
  table generatable later (P11.3 tail).
- [ ] **P17.5 — Environment reproducibility.** Turn the comment-instructions in
  `requirements-gpu.txt` into an executable `scripts/install_gpu_stack.ps1`
  (torch cu128 index, nunchaku wheel URL, verification step) and freeze the
  *verified* working set with `pip freeze > requirements-gpu.lock` so the M0/M1
  stack can be rebuilt after a disk failure without archaeology.
- [ ] **P17.6 — Repo hygiene.** Untrack `frontend/tsconfig.tsbuildinfo`
  (gitignore it); add a friendly-message layer over job errors so the UI shows a
  readable line while `repr(exc)` goes to the P15.2 log.

### P12 — Generation-page & arbiter loose ends (carried as-is)

> The shipped P7/P8/P9 phases each left a small, named remainder. Lower priority
> than the audit-driven phases above, but kept so they don't get lost.

- [ ] **P12.1 — Learned-profile management.** A UI list of learned
  `model_profiles` with a reset control (P7.2 tail), plus capture LLM-subprocess
  VRAM (its `load_report` is currently `None`, so the LLM is the one model with
  no measured figure).
- [ ] **P12.2 — Per-job arbiter attribution.** Surface the blocking/swap reason
  on the *exact* queued card (not just the Queue header) and add a
  keep-warm-eviction reason (P7.1 tail).
- [ ] **P12.3 — Inline previews on the Images tab.** Show the swap-plan preview
  inline on the Images-tab queue (P7.4 tail) and add a quick reproduce/vary
  action on the `ResultPreview` card (P8.3 tail).
- [ ] **P12.4 — Memory timeline depth.** Optional process-RSS series + hover
  tooltips on the System-tab sparkline (P7.3 tail).

### P18 — Distribution & run story (NEW — audit W8)

> Today the app runs as uvicorn + Vite *dev* server on two ports. A production
> mode simplifies daily use and halves the attack surface P14 protects.

- [ ] **P18.1 — Production serving mode.** `npm run build` output served by
  FastAPI (StaticFiles + SPA fallback) behind `HFAB_SERVE_FRONTEND=true`; one
  port, no Node at runtime. CI builds the frontend to prove the build stays
  green.
- [ ] **P18.2 — One-command launcher.** `run.bat`/`run.sh` gain a `--prod` path:
  build-if-stale, start backend, wait on `/api/health`, open the browser. Stub
  vs. real mode stays an env concern (`HFAB_STUB_MODE`).
- [ ] **P18.3 — Writable settings (safe subset).** Promote the read-only settings
  drawer: persist a whitelisted subset (defaults for steps/size, keep-warm
  toggle, theme already local) to a `data/settings-overrides.json` loaded at
  startup. Anything memory-safety-related stays env-only by design.
- [ ] **P18.4 — Model download manager.** Surface `scripts/fetch_models.py` in
  the UI: a curated list of the verified models (from
  [imagefabric-models](docs) + MODEL_NOTICE.md) with size, license note, target
  dir, and a progress bar; refuses to start a download the RAM/disk budget can't
  hold.

### P19 — Generation features (growth — after the foundation phases)

- [ ] **P19.1 — FLUX / FLUX.2 img2img + inpaint.** Wire the two families through
  the existing `init_image`/`mask_image` plumbing (currently SDXL-only,
  fail-fast elsewhere); validate on hardware with the P16.5 checklist; respect
  the klein 768² pin.
- [ ] **P19.2 — Upscaler as an arbiter job.** A small Real-ESRGAN/SwinIR model
  behind a new job type, loaded through the arbiter like everything else; "Upscale
  2×/4×" action on `ResultPreview` and the History detail modal.
- [ ] **P19.3 — ControlNet for SDXL.** One vetted ControlNet (canny or depth) as
  an optional conditioning input in the composer, with the VRAM cost measured
  and recorded in the model profile before it ships.
- [ ] **P19.4 — Prompt library.** Named, taggable prompt/style snippets
  (DB-backed, exportable) insertable from the composer and the chat `/image`
  bridge; absorbs the existing prompt-history recall.

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
- **P3.4 — Qwen/Z-Image image families.** Added `ModelFamily.QWEN_IMAGE` and
  `ModelFamily.Z_IMAGE` for multi-file Diffusers repos detected by
  `model_index.json`. Qwen-Image-2512 defaults to bnb-nf4 + 1328² / 50 steps /
  true CFG 4.0; Z-Image-Turbo defaults to 1024² / 9 steps / guidance 0.0.
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
- **P10 — Test & CI safety net.** Unit tests for the pure logic (scheduler /
  sysmon / model profiles), the hermetic STUB-mode integration test (one swap for
  a mixed batch, images land in the gallery), Vitest + Testing Library on the
  frontend, and `.github/workflows/ci.yml` (ruff + pytest + tsc + vitest on
  push/PR). 143 tests green as of the 2026-06 audit.
- **P11 — Code health & docs (first half).** Pure-logic helper extraction with
  vitest suites (`chatHelpers.ts`, `imageComposerHelpers.ts`); committed ruff +
  pytest config (75 mechanical fixes); README synced to the real M0/M1 state.
  *Remaining halves → P16.3 (frontend lint), P17.1–.3 (the hard splits), P17.4
  (generated knob table prerequisite).*
- **P13 — Generation pages, round 2.** Model picker back to a one-row dropdown
  (`ModelPicker.tsx`); bigger scrollable result strip; `ZoomableImage` lightbox;
  img2img (upload → token → strength, SDXL real path validated on hardware);
  inpainting with a full mask editor (brush/lasso/feather, SDXL inpaint pipeline
  view validated); selectable llama backend + KV-cache context type incl.
  TurboQuant (`turbo3`/`turbo4`) with pair validation + stderr surfacing, 18
  tests.
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
- Qwen-Image-2512 is a large bf16 repo (~54 GB of model files); keep
  `HFAB_QWEN_IMAGE_QUANT=bnb-nf4` unless deliberately testing full bf16.
  Z-Image-Turbo is distilled; use guidance 0.0 unless comparing variants.

---

## Where to add the next thing

- A new workspace tab = one entry in the `workspaces` array (P4.4 registry) + a
  component using the shared control kit + chrome.
- Anything touching model loading goes through the arbiter (`ensure`/`free_all`)
  and the `sysmon` budget — never load a model directly.
- New env knobs follow the `HFAB_*` convention and are surfaced in `/api/settings`.
