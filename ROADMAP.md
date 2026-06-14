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

- [x] **P14.1 — Decide and enforce the bind posture.** Revert the local `.env` to
  the code default (`127.0.0.1:8260`) *or* make LAN exposure deliberate: keep
  `HFAB_HOST=0.0.0.0` only together with P14.2 auth. Add a loud startup warning
  (log + `/api/health` field + UI toast) whenever the server is bound to a
  non-loopback address without a token configured. Document the threat model in
  README ("local single-user app; LAN exposure requires HFAB_API_TOKEN").
- [x] **P14.2 — Optional bearer-token auth.** New `HFAB_API_TOKEN` setting; when
  set, a middleware rejects any request without `Authorization: Bearer <token>`
  (and the WebSocket without a `?token=` query param). The frontend reads the
  token from a small login prompt persisted in `localStorage` and attaches it in
  `api/client.ts` + `useEvents.ts`. Empty/unset token = current open behavior on
  loopback. Tests: 401 paths, WS rejection, health stays open (or not — decide).
- [x] **P14.3 — Gate desktop-reaching endpoints to loopback.** `reveal_image`
  (spawns `explorer`/`open`/`xdg-open`) must refuse requests whose client address
  is not `127.0.0.1/::1`, regardless of token — a remote caller has no business
  driving the local desktop. Audit for other desktop/process endpoints as they
  appear (voice launch falls under the token, not loopback, since the UI may be
  remote-legit later).
- [x] **P14.4 — Upload/content hardening sweep.** Verify every upload path
  enforces its size cap *before* reading the body where possible
  (`image_upload_max_mb`, `transcription_max_upload_mb`, `vision_max_upload_mb`,
  `voice_max_upload_mb`); confirm Pillow re-encodes (not just opens) uploaded
  images so crafted files never land raw in `outputs/uploads`; add a test per
  cap. One pass, mostly verification — the token guard in `util/uploads.py` is
  already right.

### P6R — Native voice engine (REPLANNED 2026-06-11 — replaces the w-okada wrap)

> **Direction change (user decision):** the voice changer is a **native
> in-process RVC engine** with our own functions and settings, built for
> correctness and reliability. Pretrained assets live under
> `models/voice/pretrain` (`content_vec_500.onnx`, `rmvpe.pt`) and the user's
> voice model is standard RVC v2 under `models/voice/chocola_yagiyukiv2/`
> (`.pth`/`.safetensors` + faiss `.index`) — no downloads required. The 2026-06
> Voice-tab UX rework (guided setup flow, monitor mode, auto-apply,
> `voiceHelpers.ts`) now targets only the native engine.
>
> Inference stack: audio → 16 kHz mono (soxr) → ContentVec features
> (onnxruntime) → optional faiss index mix (`index_ratio`) → RMVPE f0 + pitch
> shift → vendored RVC v2 synthesizer (torch, MIT-attributed) → protect blend →
> output. Deps added to the GPU venv: `sounddevice`, `soundfile`, `faiss-cpu`,
> `onnxruntime`, `soxr`.

- [x] **P6R.1 — Engine core + offline conversion.** Shipped: the
  `backend/app/services/voice_engine/` package — pretrain-asset discovery
  (`models/voice/pretrain`), RVC model discovery (`models/voice` slots), the **real**
  vendored RVC v2 inference network (enc_p / flow / NSF-HiFi-GAN dec / emb_g,
  ~2 400 lines, MIT-attributed) and the **real** RMVPE (E2E0 DeepUnet + BiGRU,
  numpy slaney mel — no librosa dep), ContentVec via onnxruntime, the offline
  `convert()` pipeline (index mix, f0 shift + coarse, protect blend, per-stage
  timings), sounddevice enumeration, deterministic STUB path, parallel API
  (`/api/voice/engine/*`: status / settings / convert / file with size caps +
  token guard), 9 stub tests (116 total green). **Gate passed** on the real
  `chocola_yagiyukiv2.pth` via `scripts/voice_engine_smoke.py`: strict
  state-dict load 457/0/0; RMVPE median 220.25 Hz on a 220 Hz tone; converted
  output flatness 0.17–0.23 vs 0.0008 sine baseline; CUDA warm synth **20.7 ms**
  for a 2 s clip (CPU ~320 ms) — realtime-viable. *Note:* first fake-synth
  attempt was rejected at review; the smoke script is now the permanent
  anti-fake gate for this code.
- [x] **P6R.2 — Realtime session.** *Backend shipped:* `realtime.py` —
  `ChunkProcessor` (rolling 16 k context capped at 8 s, shared
  `pipeline.convert_audio` core, SOLA seam alignment + equal-power crossfade)
  and `RealtimeSession` (sounddevice duplex stream, lock-guarded sample rings,
  worker thread, separate monitor OutputStream with its own gain, pass-through,
  per-chunk VU/timings/over-underrun metrics, bounded input queue); engine
  gained the realtime knobs (`server_audio_sample_rate`, `server_read_chunk_
  size`, `cross_fade_overlap_size`, `extra_convert_size`, `pass_through` —
  settings apply per-chunk without restart); API `session/start` (409 on busy
  GPU, frees the arbiter resident) / `session/stop` / `live`+`metrics` in
  `/status`; the worker's voice lane now parks GPU jobs for the native session.
  Stub session + 3 tests incl. an
  integration test proving a queued image job parks while live and runs after
  stop (129 backend tests green). Also fixed: the test suite now pins
  `HFAB_API_TOKEN`/`HFAB_HOST` so a developer's real `.env` can't leak 401s
  into it. *Validation 2026-06-12:* `scripts/voice_realtime_bench.py` loads the
  real `chocola_yagiyukiv2` model and feeds 20 synthetic 48 kHz chunks through
  `ChunkProcessor`; stitched output was finite, exact-length, and within 30% RMS
  of one-shot offline `pipeline.convert_audio` for every run. The first bench
  exposed a per-chunk hot-path bug — `create_f0_extractor` reloaded the ~180 MB
  RMVPE checkpoint from disk **every chunk** (~0.5 s/chunk); it is now memoized
  per (detector, path, device). After the fix, **CUDA is realtime at every
  chunk size**: 96/133/192 → mean/p95 **224.6/374.7**, **129.3/140.8**,
  **125.5/141.4** ms vs chunk **256.0/354.7/512.0** ms (chunk 133 has ~2.7×
  headroom). CPU is realtime only at chunk 192 (**478.9/514.5** ms vs 512 ms);
  use CUDA (`HFAB_VOICE_DEVICE=cuda`) for live sessions. *Remaining:*
  real-device validation with a mic lives in P6R.4.
- [x] **P6R.3 — UI rewire to the native engine.** Shipped 2026-06-12: Voice tab
  now calls `/api/voice/engine/*`; Engine step shows native readiness, assets,
  loaded model and cpu/cuda device; device pickers use native sounddevice
  enumeration; live toggle starts/stops the native session and surfaces
  `session_error` + metrics; tuning/audio controls map to native snake_case
  settings; offline convert has file picker, voice/pitch controls, inline player,
  download link and timings. Verification: `frontend/ npx.cmd tsc -b`,
  `frontend/ npm.cmd test`, `backend/ .venv\Scripts\python.exe -m pytest -p
  no:cacheprovider` (workspace temp root), and `backend/ ruff check app tests`
  are green.
- [x] **P6R polish - Input-side cleanup + character.** Added dependency-light
  numpy DSP before ContentVec/RMVPE: smooth FFT high-pass, RMS noise gate, and
  input-side formant/brightness with f0 compensation so the existing pitch knob
  remains the pitch control. The realtime path applies it to the rolling context
  before chunk-tail stitching; offline `/convert` accepts compact overrides.
- [x] **P6R polish - Neural input denoise.** Added optional breizhn/DTLN
  streaming ONNX denoise before HPF/gate/formant. Offline conversion resets and
  runs one DTLN pass over the 16 kHz file; realtime denoises each new chunk once
  before it enters the rolling context so overlapping context is not processed
  twice. The optional `denoise_dtln` asset pair is listed separately and does
  not block base engine readiness; selecting `dtln` without the weights returns
  a 503 naming `models/voice/pretrain/denoise`.
- [x] **P6R polish - Idle squelch + persisted live tuning.** Realtime now
  measures each denoised chunk before RVC conversion and uses a hysteresis
  squelch (`silence_threshold_db`, `silence_hold_ms`) to output pure zeros and
  skip ContentVec/RMVPE/synth while idle; offline conversion is unchanged. Voice
  settings persist to `data/voice-settings.json` through atomic writes, `/status`
  flags missing persisted audio-device ids, and the UI exposes live restart
  hints plus a Recommended chip for the DTLN setup (pitch intentionally
  untouched). *Validation 2026-06-12:* the root `.venv` CUDA bench
  (`scripts\voice_realtime_bench.py --device cuda`) with real
  `chocola_yagiyukiv2` produced 96/133/192 mean/p95 **231.2/392.0**,
  **132.7/159.3**,
  **122.7/131.3** ms vs chunk **256.0/354.7/512.0** ms. The added squelch
  segment fed five silent chunks between voiced chunks: **4/5** silence chunks
  squelched after hold, max squelched time **0.190 ms**, and the first voiced
  chunk after silence converted in **129.0 ms** with RMS **0.101110**.
- [x] **P6R polish - Realtime pipeline rebuild for seam/sibilance quality
  (2026-06-13).** The live path was re-architected after persistent reports of
  stutter ("дьоргання") and lisping ("шипелявість"): (1) every per-chunk
  one-shot `soxr.resample` was replaced with stateful `soxr.ResampleStream`s
  (in: stream->16k, out: model->stream) so filter edges are never baked into
  the stream; (2) the conversion context now advances in blocks that are a
  multiple of 640 (LCM of the ContentVec hop 320 and DTLN hop 128) over a
  fixed-length window, keeping the feature/f0 frame grid aligned between
  conversions; (3) the SOLA stitch moved to the model sample rate and now
  stores the *aligned* continuation of the emitted audio as the next seam
  reference (the old code stored the unaligned tail, repeating/skipping up to
  10 ms at every seam); (4) the synthesizer's latent noise is pinned per
  absolute frame via a noise ring passed through `convert_audio` ->
  `infer(latent_noise=...)`, and SineGen's per-call random harmonic phase is
  now a deterministic per-instance constant, so overlapping context
  re-synthesizes identically; (5) HPF became a streaming biquad
  (`dsp.StreamingHighpass`), the noise gate was dropped from the realtime
  chain (squelch + DTLN cover it; gate remains offline), the input formant is
  a streaming resampler stage, and DTLN gained a gapless `process_stream` (the
  old same-length contract inserted ~2.7 ms zero gaps at chunk boundaries);
  (6) feature upsampling switched from linear interp to frame-repeat to match
  RVC training (crisper consonants), and `voice_protect` default fixed
  0.5 -> 0.33 (0.5 disables consonant protection entirely); (7) UI quality
  presets were retuned (extra 5 s -> 2 s — ContentVec is CPU-bound and 5 s
  context overran the chunk budget; Feminine preset is now a full profile:
  +12 st, +0.5 input formant, index 0.5, protect 0.33, DTLN). *Validation
  2026-06-13:* root `.venv` CUDA bench with real `chocola_yagiyukiv2`
  (pitch +12, index 0.5, protect 0.33, DTLN on, extra 2.0): 96/133/192
  mean/p95 **254.0/364.7**, **145.4/165.5**, **145.0/159.9** ms vs chunk
  **256.0/354.7/512.0** ms; stitched vs offline RMS within 5%; squelch emits
  pure zeros in ≤12 ms (input chain only) and voice resumes in 148 ms.
- [~] **P6R.4 — Live validation + legacy voice removal (gates the phase).** Code
  part done 2026-06-12: deleted the old wrapper router, launch path, settings,
  pidfile reap hook, discovery fallbacks, frontend legacy client/types/helpers,
  and collapsed the voice lane to native `realtime.session_active()`. Remaining
  with a real mic: confirm conversion + monitor output, measure round-trip
  latency at 2–3 chunk sizes, confirm a queued image job parks during the
  session and resumes after, then write the numbers into this file.

### P15 — Reliability & data layer (NEW — audit W3, W7)

> The schema is migrated by a hand-rolled `ALTER TABLE` helper, nothing is
> logged to disk, and there is no backup story. Each is cheap now and expensive
> after the tenth table / first lost database.

- [x] **P15.1 — Real migrations.** Introduce Alembic (async SQLite) with the
  current schema as revision 0; fold the `_ensure_image_columns` logic into a
  proper revision and delete it. `init_db()` runs `upgrade head` at startup.
  Document "how to add a column" in the README dev section. Test: fresh DB and
  a pre-migration DB both come up.
  - Done: `backend/alembic.ini` + `backend/migrations/`; startup runs Alembic
    through the async engine, and legacy image rows missing `family` /
    `favorite` / `tags` are covered by a raw-SQL upgrade test.
- [x] **P15.2 — Structured file logging.** Rotating file handler under
  `data/logs/` (size-capped, ~5×10 MB): startup config summary, every arbiter
  note (swap / refusal / warm-evict with the numbers), job start/done/error with
  duration, llama-server stderr tail on failure, unhandled exceptions. Plain
  `logging` is fine; the point is post-mortem debuggability (the "it hung
  overnight" case), not an ELK stack.
  - Done: one event-bus subscriber in lifespan writes `data/logs/hfabric.log`
    through a 10 MB × 5 rotating handler; stub integration asserts startup and
    job lifecycle lines.
- [x] **P15.3 — Backup & restore story.** A `scripts/backup.py` (or `.ps1`) that
  snapshots `data/hfabric.db` (using the SQLite backup API, not file-copy of a
  live WAL DB) plus a manifest of `outputs/`; README section on what to back up
  and how to restore. Optional: a retention-N rotation.
  - Done: `scripts/backup.py` snapshots the DB, writes an output manifest, and
    applies `--keep N` retention; README has restore order.
- [x] **P15.4 — Orphan-process audit.** If the backend dies hard, can
  `llama-server` outlive it and hold VRAM? Track child PIDs in
  a pidfile and reap stale ones on startup (with a log line), mirroring the
  existing orphan-job requeue. Test with a simulated kill in stub mode where
  possible.
  - Done: `data/runtime/llama-server.pid` is written/removed around the managed
    subprocess; startup reaps matching stale processes and tests cover dead and
    wrong-name pidfiles.

### P16 — Test depth & quality gates (NEW — audit W4; absorbs P11.2 tail)

> The core (scheduler/sysmon/arbiter/gallery) is well covered; the edges —
> chat, rag, tts, transcription, vision, voice, notes, presets, code routers and
> chat/embedding services — have **zero** tests, and CI measures nothing, so
> holes stay invisible.

- [~] **P16.1 — Coverage measurement in CI.** Add `pytest-cov` to the backend CI
  job, publish the report in the run summary, and set a floor that today's suite
  already passes (start ~55–60% on `app/`, excluding `backends/image_diffusers.py`
  and other GPU-only modules via `.coveragerc`). Ratchet the floor up as P16.2
  lands — never down.
  - Phase E: `.coveragerc` added for `app/` with GPU-only omissions, CI installs
    `pytest-cov`, publishes the terminal report to the run summary, and uses the
    conservative sandbox floor `--cov-fail-under=60`. Reviewer must install
    `pytest-cov` and recalibrate to measured coverage minus 3 points.
- [x] **P16.2 — Router tests for the untested half.** Stub-mode httpx tests per
  router, reusing the `test_stub_integration.py` conftest: chat (conversation
  CRUD, message flow, `/image` bridge, regenerate/edit), rag (ingest → search →
  cited answer path with a fake embed server), notes/presets/code (CRUD +
  validation), tts/transcription/vision (request validation + the
  no-model/no-binary error paths — the subprocess itself can be monkeypatched),
  voice (native status/config/session validation paths). Target: every router
  file imported by `main.py` has a test file.
  - Phase E: added `test_chat_api.py`, `test_rag_api.py`,
    `test_notes_presets_code.py`, and `test_tts_transcription_vision.py`; full
    backend suite is 147 tests green in stub mode.
- [~] **P16.3 — Frontend lint in CI (P11.2 tail).** eslint (typescript-eslint +
  react-hooks) + prettier, config committed, `npm run lint` wired into
  `.github/workflows/ci.yml`. Auto-fix the initial sweep in its own commit so
  review stays readable.
  - Phase E: flat ESLint config, Prettier config, `npm run lint`, devDependency
    entries, and CI lint step are added. Sandbox registry access timed out while
    refreshing `package-lock.json`; `npm run lint` cannot execute until the
    reviewer installs the new dev dependencies and runs the initial autofix.
- [x] **P16.4 — Frontend flow tests.** Testing-Library tests for the three
  highest-value flows: ChatPanel (send → streamed reply → thinking panel split),
  Gallery (filter chips combine + bulk select), QueuePanel (job states + cancel).
  Mock `api/client.ts` + the WS hook; no real backend.
  - Phase E: ChatPanel, Gallery, and QueuePanel flow tests added; frontend suite
    is 49 tests green.
- [x] **P16.5 — Real-GPU smoke checklist.** One doc (`docs/gpu-smoke.md`)
  consolidating the manual runners (`scripts/swap_leak_test.py`,
  `phase_batch_check.py`, `sdxl_resident_drift_test.py`, `image_live_bench.py`,
  `kv_cache_*.py`) into a 15-minute ordered checklist with expected numbers from
  M1, to run after any torch/diffusers/driver bump. The scripts exist — the
  missing artifact is the checklist with pass criteria.
  - Phase E: `docs/gpu-smoke.md` added with ordered pass criteria and the M1/P6R
    image + native voice benchmark numbers.

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

- [x] **P12.1 — Learned-profile management.** A UI list of learned
  `model_profiles` with a reset control (P7.2 tail), plus capture LLM-subprocess
  VRAM (its `load_report` is currently `None`, so the LLM is the one model with
  no measured figure).
  - Phase F: System tab lists learned profiles with per-row/all reset; backend
    has GET/DELETE profile endpoints and clears the sysmon cache. LLM subprocess
    VRAM capture remains out of scope and is called out in the UI.
- [x] **P12.2 — Per-job arbiter attribution.** Surface the blocking/swap reason
  on the *exact* queued card (not just the Queue header) and add a
  keep-warm-eviction reason (P7.1 tail).
  - Phase F: Queue cards attach matching arbiter notes by model id/target model
    id; warm-evict notes include target attribution.
- [x] **P12.3 — Inline previews on the Images tab.** Show the swap-plan preview
  inline on the Images-tab queue (P7.4 tail) and add a quick reproduce/vary
  action on the `ResultPreview` card (P8.3 tail).
  - Phase F: Images queue shows the predicted plan inline; ResultPreview reuses
    the History composer-apply path for Reproduce/Vary.
- [x] **P12.4 — Memory timeline depth.** Optional process-RSS series + hover
  tooltips on the System-tab sparkline (P7.3 tail).
  - Phase F: System sparkline has an App RSS toggle and hover tooltip with time,
    RAM, VRAM, and process RSS values.

### P18 — Distribution & run story (NEW — audit W8)

> Today the app runs as uvicorn + Vite *dev* server on two ports. A production
> mode simplifies daily use and halves the attack surface P14 protects.

- [x] **P18.1 — Production serving mode.** `npm run build` output served by
  FastAPI (StaticFiles + SPA fallback) behind `HFAB_SERVE_FRONTEND=true`; one
  port, no Node at runtime. CI builds the frontend to prove the build stays
  green.
  - Phase G: FastAPI serves frontend/dist behind HFAB_SERVE_FRONTEND=true with
    SPA fallback, missing-dist 503 guidance, no-cache index, long-cache assets;
    CI now runs `npm run build`.
- [x] **P18.2 — One-command launcher.** `run.bat`/`run.sh` gain a `--prod` path:
  build-if-stale, start backend, wait on `/api/health`, open the browser. Stub
  vs. real mode stays an env concern (`HFAB_STUB_MODE`).
  - Phase G: `run.bat`, `scripts/run.ps1`, and `run.sh` support prod mode with
    stale dist detection, health wait, and browser open on the backend port.
- [x] **P18.3 — Settings tab + minimal env.** Promote the read-only settings
  drawer into a first-class workspace tab and move day-to-day runtime knobs out
  of `.env`. Keep `.env` for system startup posture only: host, port, optional
  API token, frontend serving, and Vite dev-server host/port. Everything else
  is typed, grouped, searchable, saved to `data/settings-overrides.json`, and
  loaded at startup.
  - Phase G: GET/PUT `/api/settings/overrides` now returns a schema for the
    Settings tab (groups, types, min/max, choices, restart-required markers),
    validates/clamps values, persists the file, applies live settings, and
    re-runs directory creation after path overrides load. The topbar Settings
    button is gone because Settings is a workspace tab. `.env.example` documents
    the minimal system-only env surface. Empty/unset `HFAB_API_TOKEN` is the
    supported local-no-auth mode.
- [ ] **P18.4 — Model download manager.** Surface `scripts/fetch_models.py` in
  the UI: a curated list of the verified models (from
  [imagefabric-models](docs) + MODEL_NOTICE.md) with size, license note, target
  dir, and a progress bar; refuses to start a download the RAM/disk budget can't
  hold.

### P20 — Universal GPU & installer story (NEW — make it usable beyond this machine)

> Goal: the app should make the hard installation/runtime decisions itself.
> A normal user should see "Recommended" and "Use safe defaults", not CUDA/ROCm
> wheel archaeology. The installer detects the machine, chooses the best backend
> profile, installs compatible packages, writes only system env when needed, and
> stores runtime choices in Settings.
>
> Current upstream facts to anchor the plan: PyTorch publishes separate CPU,
> CUDA, and ROCm pip install paths and verifies GPU availability with
> `torch.cuda.is_available()`; NVIDIA exposes per-GPU compute capability tables;
> AMD's official ROCm support matrix is Linux-focused and says unsupported GPUs
> are not officially supported. Keep source links in the installer docs:
> PyTorch local install, NVIDIA CUDA GPU Compute Capability, AMD ROCm system
> requirements / compatibility matrix, and AMD ROCm PyTorch install.

- [~] **P20.1 — Hardware probe + support report.** Add one cross-platform probe
  (`scripts/hardware_probe.py` + PowerShell wrapper) that emits JSON:
  OS/build, Python version, RAM, disk free, GPU vendor/model/VRAM, NVIDIA driver
  + compute capability when present, AMD ROCm-visible device + LLVM target when
  present, and whether `torch` can see the accelerator. The app and installer
  both consume this report; no duplicate detection logic in batch scripts.
  - First slice: `scripts/hardware_probe.py` runs with stdlib only and emits the
    shared JSON report. It uses `nvidia-smi`, Windows CIM, `lspci`, `rocminfo`,
    and optional `torch` visibility when available; missing tools are skipped.
- [~] **P20.2 — Installer profile resolver.** Replace "pick these packages by
  hand" with a resolver that chooses one profile:
  `nvidia-cuda`, `amd-rocm-linux`, `cpu-safe`, and later optional
  `amd-directml-windows` if it proves useful. Each profile has a package index,
  lockfile, verification command, and post-install health check. The installer
  asks only when there are two valid choices; otherwise it chooses the safest
  working profile automatically.
  - First slice: `scripts/install_profiles.py` consumes the hardware report (or
    probes live), selects `nvidia-cuda`, `amd-rocm-linux`, or `cpu-safe`, emits
    package index/packages, verification snippet, runtime defaults, disabled
    features, warnings, and source links. Fake-probe tests cover Blackwell,
    lower-VRAM NVIDIA, Linux AMD ROCm, Windows AMD fallback, CPU fallback, and
    invalid forced profile rejection. `setup.ps1` and `setup.sh` now call the
    resolver in their default path and install profile-specific PyTorch wheels
    and requirements instead of hardcoding CUDA for every REAL setup.
  - Launcher slice: `run.bat`/`scripts/run.ps1`/`run.sh` now use the same
    resolver when `HFAB_STUB_MODE` is not explicitly set, so unsupported
    machines start CPU-safe automatically while NVIDIA/ROCm profiles start
    REAL mode. Explicit `stub` and `HFAB_STUB_MODE` remain overrides.
- [~] **P20.3 — NVIDIA beyond RTX 50 / Blackwell.** Support NVIDIA 50xx, 40xx,
  30xx, and practical lower tiers by capability and VRAM, not by one validated
  RTX 5070 Ti path. Runtime policy must auto-disable Blackwell-only fast paths
  where they do not apply, choose attention/compile/cache defaults per compute
  capability, and clamp model recommendations by VRAM. Minimum viable tiers:
  8 GB = SDXL/LLM-small safe mode; 12 GB = SDXL + selected quantized LLMs;
  16 GB+ = current richer image/LLM paths. Tests should fake probe reports for
  multiple compute capabilities and prove the resolver never recommends an
  impossible package/model path.
  - First slice: the resolver now makes NVIDIA runtime defaults capability-aware
    instead of Blackwell-only. Compute capability maps to an architecture name
    (`pascal`/`turing`/`ampere`/`ada`/`hopper`/`blackwell`); attention defaults
    to `math` below Ampere and `auto` at 8.0+; `flux_step_cache` and
    `allow_nunchaku` require the fp4-capable Ampere+ path; `blackwell_fast_paths`
    stays 12.0-only. Pre-Ampere cards drop `nunchaku_cuda` from
    `optional_features` so `setup.ps1`/`setup.sh` never fetch an unusable CUDA
    nunchaku wheel. A new `model_policy` block buckets image families into
    recommended/advanced/hidden by tier + nunchaku capability (and an LLM
    param-size hint), surfaced through `/api/capabilities`; P20.7 wires it to the
    download manager. Fake-probe tests cover caps 6.1/7.5/8.0/8.6/8.9/9.0/12.0
    across VRAM tiers and prove no impossible package/model path is recommended.
- [ ] **P20.4 — AMD GPU path.** Implement a first-class AMD profile instead of
  treating non-NVIDIA as "CPU only". Linux ROCm is the primary target because
  PyTorch ROCm wheels and AMD's support matrix are Linux-centered. The probe
  should mark AMD GPUs as: official ROCm-supported, community/experimental, or
  unsupported. Official path installs ROCm-compatible PyTorch wheels, verifies
  `torch.cuda.is_available()` under the ROCm build, and automatically disables
  CUDA-only libraries/features (`nunchaku`, CUDA llama binaries, CUDA-specific
  attention assumptions) in favor of ROCm-safe or CPU fallbacks.
  - First slice: added `backend/requirements-rocm.txt` so the AMD profile avoids
    CUDA-only packages (`nunchaku`, CUDA ONNX Runtime EP assumptions) while still
    installing the common diffusers/transformers/audio stack.
- [ ] **P20.5 — Runtime capability gates.** Move backend feature decisions from
  env assumptions to a `CapabilityProfile`: vendor, backend (`cuda`/`rocm`/`cpu`),
  VRAM, supported dtypes/attention, available binaries, and known unsafe
  features. The model registry, Settings tab, and composer should hide or label
  incompatible options instead of letting users choose combinations that will
  fail after a long load.
  - First slice: added `backend/app/services/capability_profile.py`, which
    imports the shared hardware/profile resolver and exposes an active runtime
    capability object (`selected_profile`, `active_profile`, `backend`,
    `hardware_tier`, feature flags, disabled features, warnings). `/api/settings`
    now includes this object and `/api/capabilities` exposes it directly; the
    Settings and System tabs show the selected profile/backend/tier. Tests cover
    CUDA, STUB override, ROCm, CPU-safe, and API/settings payload parity.
  - Model gate slice: added `backend/app/services/model_compatibility.py`.
    `/api/models` now marks each model with `available`, `runtime_mode`,
    `unavailable_reason`, and compatibility warnings. The image composer avoids
    disabled models, the picker labels them, and `/api/jobs` rejects unavailable
    models server-side. Current guards cover STUB passthrough, nunchaku CUDA
    requirements, ROCm bitsandbytes exclusions, and estimated VRAM over budget.
  - Autotune slice: `backend/app/services/runtime_tuning.py` applies the detected
    profile's *safe* acceleration defaults to live settings at startup
    (`_autotune_acceleration` in `main.py`) for any knob the user did not pin via
    env or a saved override — `attention_backend` (math below Ampere / non-CUDA),
    `flux_step_cache` (off without the fp4 fast path), and `attention_allow_tf32`
    (NVIDIA Ampere+ only). It only ever tunes toward safety, never auto-enables
    `torch.compile`, is gated off in STUB mode, and never blocks startup. Knob
    logic is unit-tested (`test_runtime_tuning.py`).
  - Note on the device string: real image execution currently runs on CUDA and
    ROCm (PyTorch's ROCm build aliases the `cuda` device, so the existing
    `image_diffusers.py` `.to("cuda")` path works on AMD/Linux). CPU/Apple and
    other backends route to STUB. A device abstraction for true MPS/CPU image
    inference is tracked separately (see P20.9).
- [~] **P20.6 — User-facing installer UX.** Build a simple "Setup doctor" page:
  detected hardware, selected profile, missing driver/runtime, package status,
  and one action button. Text should be plain: "NVIDIA GPU detected, installing
  CUDA build" / "AMD GPU detected, ROCm works on Linux for this card" /
  "GPU path unavailable, using CPU-safe mode." Advanced details stay expandable.
  - First slice: `SetupDoctor` (in the System tab) reads `/api/capabilities` and
    shows a plain-language headline (NVIDIA CUDA / AMD ROCm / CPU-safe / forced
    STUB), detected hardware (GPU, VRAM, architecture, compute cap., tier),
    selected profile/backend, a package-status chip row from the feature flags,
    the per-hardware model recommendation buckets (from P20.3 `model_policy`),
    warnings, and one "Re-run detection" button (`?refresh=true` re-probe).
    Considered profiles, disabled features, and source docs sit behind an
    "Advanced details" toggle. Pure status/label helpers are unit-tested
    (`setupDoctorHelpers.test.ts`).
- [~] **P20.7 — Model recommendation by hardware.** Tie the model download
  manager (P18.4) to the capability profile. Users should see curated models
  that fit their VRAM/RAM/disk budget, with "Recommended" preselected and
  impossible models hidden behind an Advanced filter.
  - Picker slice (ahead of the download manager): `/api/models` now carries a
    per-model `recommendation` (`recommended`/`advanced`/`hidden`/`neutral`)
    derived from the capability `model_policy`, and `ModelPicker` badges the
    models that fit the detected hardware. Full curated download UX still waits
    on P18.4; this makes the existing chooser hardware-aware in the meantime.
- [~] **P20.8 — CI and smoke matrix without owning every GPU.** Add fake-probe
  unit tests for NVIDIA/AMD/CPU resolver decisions, plus optional self-hosted or
  manual smoke scripts for real CUDA and ROCm machines. Store every real-machine
  validation in `docs/gpu-smoke.md` with date, GPU, driver, package profile, and
  pass/fail notes.
  - First slice: fake-probe coverage now runs in CI via `test_install_profiles`,
    `test_capability_profile`, `test_model_compatibility`, and a new
    `test_install_smoke` (all stdlib/STUB, no GPU). `scripts/install_smoke.py`
    is the real-machine counterpart: it probes, resolves the profile, and grades
    it against `torch` visibility (CUDA build vs HIP build vs no accelerator),
    plus a feature-sanity guard (no `nunchaku_cuda` on pre-Ampere/non-CUDA) and
    an in-process verify-snippet run; exit code + a paste-ready markdown summary.
    `docs/gpu-smoke.md` gains an installer-profile smoke section and a
    real-machine validation log table (date/GPU/driver/profile/result).
- [ ] **P20.9 — Device abstraction + Apple Silicon (MPS).** Replace the hard-coded
  `.to("cuda")` / `torch.cuda.*` calls in `image_diffusers.py` with a single
  `runtime.device()` / accelerator helper sourced from the CapabilityProfile, so
  image inference can target `cuda`, `rocm` (already cuda-aliased), `mps`, or a
  CPU fallback without scattering backend assumptions. Then add an `apple-mps`
  profile: the probe detects Apple Silicon (Darwin/arm64 + `torch.backends.mps`),
  the resolver installs the standard PyPI torch wheels, disables all CUDA-only
  libraries (nunchaku, CUDA llama binaries, TF32), recommends SDXL + llama.cpp
  Metal, and hides fp4 families. Gate everything conservatively and validate on a
  real Mac before promoting any model from "advanced" to "recommended". This is
  the prerequisite for treating non-CUDA accelerators as first-class rather than
  routing them to STUB.

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
