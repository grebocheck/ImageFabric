import type { ChatConversation, ChatConversationDetail, ChatConversationImport, ChatImportResult, ChatSendBody, ChatSendResult, ImageItem, Job, JobCreate, JobType, LlmConfig, Lora, Model, Note, Preset, PresetImportItem, PresetImportResult, RuntimeSettings, TtsGenerateBody, TtsGenerateResult, TtsStatus } from "../types";

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
  getJob: (id: string) => fetch(`/api/jobs/${id}`).then(j<Job>),
  cancelJob: (id: string) => fetch(`/api/jobs/${id}`, { method: "DELETE" }).then(j<Job>),
  getLlmConfig: () => fetch("/api/llm/config").then(j<LlmConfig>),
  setLlmConfig: (body: { ctx?: number; ngl?: number }) =>
    fetch("/api/llm/config", { method: "POST", headers: JSON_HEADERS, body: JSON.stringify(body) })
      .then(j<LlmConfig & { changed: boolean; reloaded: boolean }>),
  stopLlm: () => fetch("/api/llm/stop", { method: "POST" }).then(j<{ stopped: boolean }>),

  // --- chat conversations ---
  listConversations: () => fetch("/api/chat/conversations").then(j<ChatConversation[]>),
  createConversation: (body: { title?: string; model_id?: string; system?: string } = {}) =>
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
};
