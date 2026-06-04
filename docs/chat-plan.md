# Chat workspace — plan ("local LLM tool", superapp foundation)

> Goal: grow the **LLM tab** from a single-shot prompt helper into a real,
> ChatGPT-class local tool — and make the app a clean base for a **personal
> superapp**: several tabbed workspaces over one shared GPU arbiter.
>
> Status today: full-width chat, multi-turn, streaming, per-message
> temperature/max_tokens/system, context-window control. State is in-memory only
> (lost on refresh), plain text rendering, no stop button, one conversation.

## Architecture principle

Everything keeps flowing through the existing **GpuArbiter + queue + event bus**,
so chat and image generation never fight over the 16 GB VRAM (phase-batching
already does the LLM↔image swap). New workspaces plug in the same way. The DB
(`data/imagefabric.db`) gains chat tables; nothing else changes shape.

---

## Phase C1 — Make it feel like a real chat (highest value) ✅ SHIPPED 2026-06-04

- [x] **C1.1 Persistent conversations.** SQLite `conversation`/`message` tables +
  CRUD (`/api/chat/conversations`, `.../messages`). The worker writes the reply
  back into the assistant message; conversations survive restart.
- [x] **C1.2 Conversation sidebar.** New / select / delete; auto-title from the
  first user turn. *(search across chats: later.)*
- [x] **C1.3 Markdown + code blocks.** `react-markdown` + `remark-gfm` +
  `rehype-highlight`, per-code-block and per-message **Copy**.
- [x] **C1.4 Stop / regenerate / edit.** Stop interrupts the running stream
  (`/api/llm/stop` flag); regenerate last answer; edit a user message and re-run
  (truncate-from + resend).
- [x] **C1.5 Context meter.** Live `~tokens / n_ctx` readout in the composer.

## Phase C2 — Model & sampling control ✅ SHIPPED 2026-06-04

- [x] **C2.1 Per-conversation sampling**, persisted: model, system, temperature,
  max_tokens, top_p, top_k, min_p, repeat_penalty, stop (seed sent per-message,
  not persisted). Passed straight through to llama-server.
- [x] **C2.2 Persona presets** — save/apply/delete the current system + sampling
  as a reusable persona (stored as `llm`-typed presets).
- [x] **C2.3 Streaming stats** — tokens/sec + time-to-first-token shown in the
  composer (measured client-side from the token stream).

## Phase C3 — Power features

- [x] **C3.3 — chat→image bridge.** `/image <prompt>` in chat queues an image
  job on the shared arbiter; the worker writes the result back into the
  assistant message as markdown so it renders inline and persists. Shipped
  2026-06-04.
- [x] **C3.4 — model-driven `generate_image` tool.** Chat can enable an Image
  tool; when the LLM replies with a structured `generate_image` call, the worker
  queues the child image job on the same arbiter and streams the result back into
  the conversation. Shipped 2026-06-04.
- [ ] **C3.1 Vision (multimodal).** We already ship `llama-mtmd`/`llava` binaries
  — wire image attachments (paste/drop) to a multimodal GGUF so you can chat
  about images. *(Needs a multimodal GGUF downloaded.)*
- [ ] **C3.2 Chat-with-documents (RAG).** Drop in PDFs/notes → local embeddings +
  a lightweight vector store → retrieved context injected per turn. *(Needs an
  embedding model + vector store.)*

## Phase C4 — Superapp shell

- [x] **C4.3 Command palette (Ctrl+K)** — navigate tabs + run actions (settings,
  free GPU); ⌘K button in the header. Plus conversation **search** in the chat
  sidebar and **export** a conversation to Markdown. Shipped 2026-06-04.
- [x] **C4.2 (first new tab) — System monitor.** A third workspace showing live
  VRAM/RAM gauges + resident/warm models + runtime info, fed by the `mem.status`
  WebSocket stream. No model needed. Shipped 2026-06-04. Tabs are now a
  declarative list (groundwork for C4.1).
- [x] **C4.1 Workspace registry.** Tabs are a single `workspaces` array in App
  (`{id, label, render}`) that drives the header tabs, the command-palette nav
  commands, and the main render. Adding a tab is one entry. Shipped 2026-06-04.
- [x] **C4.2 Notes/Scratch tab.** Persistent notes with search, autosave,
  create/delete, and a workspace-registry tab. Shipped 2026-06-04.
- [x] **C4.2 TTS workspace + gated generation.** The app now has a TTS workspace
  that detects `llama-tts.exe`, scans `models/tts` for local `.gguf` models,
  and generates WAV files once a local model exists. It defaults to CPU-only so
  it does not bypass the shared GPU arbiter. Shipped 2026-06-04.
- [ ] **C4.2 more tabs** (binaries already present): transcription (whisper),
  code assistant, plus live TTS validation once a local `models/tts/*.gguf`
  model is installed.
- [x] **C4.4 Import** of conversations/presets/personas. JSON bundles can
  restore conversations with messages plus image/LLM presets; persona presets
  are covered because they are stored as `llm` presets. Shipped 2026-06-04.

---

## Suggested order

C1 first (it's what makes the tab feel real and is mostly frontend + a small DB
addition), then C2 (cheap — llama-server already supports the knobs), then pick
between C3.1 (vision), C3.3 (image tool bridge) and C4 depending on what you want
the superapp to be. Each phase is shippable on its own.

## Near-term dependencies / notes

- Frontend: `react-markdown`, `remark-gfm`, a syntax highlighter (e.g.
  `highlight.js`/`shiki`). Small additions.
- Backend: new `conversation`/`message` tables + CRUD; pass full sampling params
  through `LlamaCppBackend.complete()`; expose llama-server `timings`.
- Stop button needs no new backend — `DELETE /api/jobs/{id}` already cancels.
