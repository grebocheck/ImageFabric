import type { ChatConversation, ChatConversationDetail, ChatConversationImport, ChatImportResult, ChatSendBody, ChatSendResult, CodeFile, CodeFileContent, ImageItem, ImageStats, Job, JobCreate, JobType, LlmConfig, Lora, Model, Note, Preset, PresetImportItem, PresetImportResult, QueuePlan, RagDocument, RagSearchResponse, RagStatus, RuntimeSettings, TranscriptionResult, TranscriptionStatus, TtsGenerateBody, TtsGenerateResult, TtsStatus, VisionResult, VisionStatus, VoiceSettingsUpdate, VoiceStatus } from "../types";

const JSON_HEADERS = { "Content-Type": "application/json" };

async function j<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json() as Promise<T>;
}

export const api = {
  listModels: () => fetch("/api/models").then(j<Model[]>),
  listLoras: () => fetch("/api/loras").then(j<Lora[]>),
  runtimeSettings: () => fetch("/api/settings").then(j<RuntimeSettings>),
  gpuStatus: () => fetch("/api/gpu").then(j),
  freeGpu: () => fetch("/api/gpu/free", { method: "POST" }).then(j),

  listJobs: () => fetch("/api/jobs").then(j<Job[]>),
  createJobs: (jobs: JobCreate[]) =>
    fetch("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(jobs),
    }).then(j<Job[]>),
  queuePlan: () => fetch("/api/jobs/plan").then(j<QueuePlan>),
  getJob: (id: string) => fetch(`/api/jobs/${id}`).then(j<Job>),
  cancelJob: (id: string) => fetch(`/api/jobs/${id}`, { method: "DELETE" }).then(j<Job>),
  getLlmConfig: () => fetch("/api/llm/config").then(j<LlmConfig>),
  setLlmConfig: (body: { ctx?: number; ngl?: number }) =>
    fetch("/api/llm/config", { method: "POST", headers: JSON_HEADERS, body: JSON.stringify(body) })
      .then(j<LlmConfig & { changed: boolean; reloaded: boolean }>),
  stopLlm: () => fetch("/api/llm/stop", { method: "POST" }).then(j<{ stopped: boolean }>),

  // --- chat conversations ---
  listConversations: () => fetch("/api/chat/conversations").then(j<ChatConversation[]>),
  createConversation: (body: { title?: string; model_id?: string; system?: string; params?: Record<string, unknown> } = {}) =>
    fetch("/api/chat/conversations", { method: "POST", headers: JSON_HEADERS, body: JSON.stringify(body) })
      .then(j<ChatConversation>),
  getConversation: (id: string) => fetch(`/api/chat/conversations/${id}`).then(j<ChatConversationDetail>),
  updateConversation: (id: string, body: Partial<Pick<ChatConversation, "title" | "model_id" | "system">>) =>
    fetch(`/api/chat/conversations/${id}`, { method: "PATCH", headers: JSON_HEADERS, body: JSON.stringify(body) })
      .then(j<ChatConversation>),
  deleteConversation: (id: string) =>
    fetch(`/api/chat/conversations/${id}`, { method: "DELETE" }).then(j<{ deleted: boolean }>),
  importConversations: (conversations: ChatConversationImport[]) =>
    fetch("/api/chat/import", {
      method: "POST",
      headers: JSON_HEADERS,
      body: JSON.stringify({ conversations }),
    }).then(j<ChatImportResult>),
  sendChatMessage: (id: string, body: ChatSendBody) =>
    fetch(`/api/chat/conversations/${id}/messages`, { method: "POST", headers: JSON_HEADERS, body: JSON.stringify(body) })
      .then(j<ChatSendResult>),
  sendChatImage: (id: string, body: { prompt: string; model_id: string; negative?: string; steps?: number; width?: number; height?: number; seed?: number }) =>
    fetch(`/api/chat/conversations/${id}/image`, { method: "POST", headers: JSON_HEADERS, body: JSON.stringify(body) })
      .then(j<ChatSendResult>),
  truncateFrom: (id: string, messageId: string) =>
    fetch(`/api/chat/conversations/${id}/messages/${messageId}`, { method: "DELETE" }).then(j<{ removed: number }>),
  setPriority: (id: string, priority: number) =>
    fetch(`/api/jobs/${id}/priority`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ priority }),
    }).then(j<Job>),
  clearFinished: () => fetch("/api/jobs/clear", { method: "POST" }).then(j),

  listImages: (q?: string) => {
    const params = q?.trim() ? `?q=${encodeURIComponent(q.trim())}` : "";
    return fetch(`/api/images${params}`).then(j<ImageItem[]>);
  },
  queryImages: (opts: { q?: string; model?: string; size?: string; lora?: string; date_from?: string; date_to?: string; limit?: number; offset?: number } = {}) => {
    const p = new URLSearchParams();
    if (opts.q?.trim()) p.set("q", opts.q.trim());
    if (opts.model) p.set("model", opts.model);
    if (opts.size) p.set("size", opts.size);
    if (opts.lora) p.set("lora", opts.lora);
    if (opts.date_from) p.set("date_from", opts.date_from);
    if (opts.date_to) p.set("date_to", opts.date_to);
    if (opts.limit != null) p.set("limit", String(opts.limit));
    if (opts.offset != null) p.set("offset", String(opts.offset));
    const qs = p.toString();
    return fetch(`/api/images${qs ? `?${qs}` : ""}`).then(j<ImageItem[]>);
  },
  imageStats: () => fetch("/api/images/stats").then(j<ImageStats>),
  exportImages: async (imageIds: string[]) => {
    const res = await fetch("/api/images/export", {
      method: "POST",
      headers: JSON_HEADERS,
      body: JSON.stringify({ image_ids: imageIds }),
    });
    if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
    return res.blob();
  },
  deleteImage: (id: string) => fetch(`/api/images/${id}`, { method: "DELETE" }).then(j<{ deleted: string }>),
  revealImage: (id: string) => fetch(`/api/images/${id}/reveal`, { method: "POST" }).then(j),
  listPresets: () => fetch("/api/presets").then(j<Preset[]>),
  createPreset: (name: string, type: JobType, params: Record<string, unknown>) =>
    fetch("/api/presets", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, type, params }),
    }).then(j<Preset>),
  importPresets: (presets: PresetImportItem[], on_conflict: "rename" | "skip" = "rename") =>
    fetch("/api/presets/import", {
      method: "POST",
      headers: JSON_HEADERS,
      body: JSON.stringify({ presets, on_conflict }),
    }).then(j<PresetImportResult>),
  deletePreset: (id: string) => fetch(`/api/presets/${id}`, { method: "DELETE" }).then(j),

  // --- notes ---
  listNotes: (q?: string) => {
    const params = q?.trim() ? `?q=${encodeURIComponent(q.trim())}` : "";
    return fetch(`/api/notes${params}`).then(j<Note[]>);
  },
  createNote: (body: { title?: string; content?: string } = {}) =>
    fetch("/api/notes", { method: "POST", headers: JSON_HEADERS, body: JSON.stringify(body) })
      .then(j<Note>),
  updateNote: (id: string, body: Partial<Pick<Note, "title" | "content">>) =>
    fetch(`/api/notes/${id}`, { method: "PATCH", headers: JSON_HEADERS, body: JSON.stringify(body) })
      .then(j<Note>),
  deleteNote: (id: string) =>
    fetch(`/api/notes/${id}`, { method: "DELETE" }).then(j<{ deleted: string }>),

  ttsStatus: () => fetch("/api/tts/status").then(j<TtsStatus>),
  generateTts: (body: TtsGenerateBody) =>
    fetch("/api/tts/generate", { method: "POST", headers: JSON_HEADERS, body: JSON.stringify(body) })
      .then(j<TtsGenerateResult>),

  transcriptionStatus: () => fetch("/api/transcription/status").then(j<TranscriptionStatus>),
  transcribeAudio: (body: { file: File; model_id: string; language?: string; task?: string; initial_prompt?: string }) => {
    const form = new FormData();
    form.append("file", body.file);
    form.append("model_id", body.model_id);
    if (body.language?.trim()) form.append("language", body.language.trim());
    if (body.task) form.append("task", body.task);
    if (body.initial_prompt?.trim()) form.append("initial_prompt", body.initial_prompt.trim());
    return fetch("/api/transcription/transcribe", { method: "POST", body: form }).then(j<TranscriptionResult>);
  },

  ragStatus: () => fetch("/api/rag/status").then(j<RagStatus>),
  listRagDocuments: (q?: string) => {
    const params = q?.trim() ? `?q=${encodeURIComponent(q.trim())}` : "";
    return fetch(`/api/rag/documents${params}`).then(j<RagDocument[]>);
  },
  createRagDocument: (body: { title?: string; content: string; source?: string; model_id?: string }) =>
    fetch("/api/rag/documents", { method: "POST", headers: JSON_HEADERS, body: JSON.stringify(body) })
      .then(j<RagDocument>),
  uploadRagDocument: (body: { file: File; title?: string; model_id?: string }) => {
    const form = new FormData();
    form.append("file", body.file);
    if (body.title?.trim()) form.append("title", body.title.trim());
    if (body.model_id) form.append("model_id", body.model_id);
    return fetch("/api/rag/documents/upload", { method: "POST", body: form }).then(j<RagDocument>);
  },
  deleteRagDocument: (id: string) => fetch(`/api/rag/documents/${id}`, { method: "DELETE" }).then(j<{ deleted: string }>),
  searchRag: (body: { query: string; top_k?: number; model_id?: string }) =>
    fetch("/api/rag/search", { method: "POST", headers: JSON_HEADERS, body: JSON.stringify(body) })
      .then(j<RagSearchResponse>),

  visionStatus: () => fetch("/api/vision/status").then(j<VisionStatus>),
  voiceStatus: () => fetch("/api/voice/status").then(j<VoiceStatus>),
  voiceStartServer: () => fetch("/api/voice/start", { method: "POST" }).then(j<{ running: boolean; already?: boolean; pid?: number }>),
  voiceStopServer: () => fetch("/api/voice/stop", { method: "POST" }).then(j<{ stopped: boolean }>),
  voiceApplySettings: (body: VoiceSettingsUpdate) =>
    fetch("/api/voice/settings", { method: "POST", headers: JSON_HEADERS, body: JSON.stringify(body) }).then(j<VoiceStatus>),
  voiceStartSession: (body: VoiceSettingsUpdate) =>
    fetch("/api/voice/session/start", { method: "POST", headers: JSON_HEADERS, body: JSON.stringify(body) }).then(j<VoiceStatus>),
  voiceStopSession: () => fetch("/api/voice/session/stop", { method: "POST" }).then(j<VoiceStatus>),
  analyzeVision: (body: { file: File; prompt: string; model_id: string; projector_id: string }) => {
    const form = new FormData();
    form.append("file", body.file);
    form.append("prompt", body.prompt);
    form.append("model_id", body.model_id);
    form.append("projector_id", body.projector_id);
    return fetch("/api/vision/analyze", { method: "POST", body: form }).then(j<VisionResult>);
  },

  listCodeFiles: (q?: string) => {
    const params = q?.trim() ? `?q=${encodeURIComponent(q.trim())}` : "";
    return fetch(`/api/code/files${params}`).then(j<CodeFile[]>);
  },
  getCodeFile: (path: string) => fetch(`/api/code/file?path=${encodeURIComponent(path)}`).then(j<CodeFileContent>),
};
