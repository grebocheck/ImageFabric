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

## Phase C1 — Make it feel like a real chat (highest value)

- **C1.1 Persistent conversations.** New SQLite tables `conversation` and
  `message`; CRUD endpoints (`/api/chat/conversations`, `.../messages`). The chat
  job writes the assistant reply back to the conversation. Survives restart.
- **C1.2 Conversation sidebar.** List / new / rename / delete / search past
  chats; auto-title from the first user turn (or a tiny LLM summarization call).
- **C1.3 Markdown + code blocks.** Render assistant output as GitHub-flavored
  markdown (`react-markdown` + `remark-gfm`) with syntax highlighting and a
  per-code-block **Copy** button. Copy-message and copy-conversation too.
- **C1.4 Stop / regenerate / edit.** Stop button cancels the running LLM job
  mid-stream (reuse `DELETE /api/jobs/{id}`); regenerate last answer; edit an
  earlier user message and re-run from that point (truncate + resend).
- **C1.5 Context meter.** Live token/char count and a "X / n_ctx" usage bar so
  you can see when you're about to overflow the window.

## Phase C2 — Model & sampling control

- **C2.1 Per-conversation settings**, persisted: model, system prompt,
  temperature, top_p, top_k, min_p, repeat_penalty, max_tokens, seed, stop
  sequences. (llama-server's `/v1/chat/completions` accepts all of these — just
  pass them through `complete()`.)
- **C2.2 System-prompt / persona presets** library (reuse the presets table with
  a `chat` type), one-click apply.
- **C2.3 Streaming stats.** tokens/sec + time-to-first-token from llama-server
  `timings`, shown under each reply.

## Phase C3 — Power features

- **C3.1 Vision (multimodal).** We already ship `llama-mtmd`/`llava` binaries —
  wire image attachments (paste/drop) to a multimodal GGUF so you can chat about
  images. Bridges naturally with the Images tab.
- **C3.2 Chat-with-documents (RAG).** Drop in PDFs/notes → local embeddings +
  a lightweight vector store (sqlite-vss or an in-process index) → retrieved
  context injected per turn. Fully local.
- **C3.3 Tools / function-calling.** Let the model call local tools — most
  interesting: a **`generate_image` tool that queues an image job** (chat ↔
  image bridge), plus file read/write and optional web fetch. Gated/allowlisted.
- **C3.4 Slash-commands & snippet library** (`/summarize`, `/image …`, saved
  prompts).

## Phase C4 — Superapp shell

- **C4.1 Workspace/plugin registry.** Promote tabs to a declared list of
  workspaces (Images, Chat, + future). Each is a self-contained module over the
  shared arbiter/queue.
- **C4.2 Candidate future tabs** (we already have the binaries for some):
  Transcription (whisper), **TTS** (`llama-tts`), Code assistant, Notes/scratch,
  Batch/automation runner.
- **C4.3 Command palette (Ctrl+K)**, global shortcuts, settings hub, theming.
- **C4.4 Export/import** of conversations, presets, personas; everything under
  `data/` for easy backup.

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
