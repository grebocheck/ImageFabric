import type { ChatConversation, ChatConversationDetail, ChatConversationImport, ChatImportResult, ChatSendBody, ChatSendResult, CodeFile, CodeFileContent, HealthStatus, ImageItem, ImageStats, Job, JobCreate, JobType, LlmConfig, Lora, Model, ModelProfile, Note, Preset, PresetImportItem, PresetImportResult, QueuePlan, RagDocument, RagSearchResponse, RagStatus, RuntimeSettings, SettingsOverrides, TranscriptionResult, TranscriptionStatus, TtsGenerateBody, TtsGenerateResult, TtsStatus, VisionResult, VisionStatus, VoiceEngineConvertResult, VoiceEnginePreset, VoiceEngineSettingsUpdate, VoiceEngineStatus } from "../types";

const JSON_HEADERS = { "Content-Type": "application/json" };
const TOKEN_KEY = "hfabric.apiToken";

type AuthEvent = { token: string; unauthorized: boolean };
type AuthListener = (event: AuthEvent) => void;
const authListeners = new Set<AuthListener>();

function readToken(): string {
  try {
    return localStorage.getItem(TOKEN_KEY)?.trim() ?? "";
  } catch {
    return "";
  }
}

function emitAuth(unauthorized = false) {
  const event = { token: readToken(), unauthorized };
  for (const listener of authListeners) listener(event);
}

function authHeaders(headers?: HeadersInit): Headers {
  const next = new Headers(headers);
  const token = readToken();
  if (token) next.set("Authorization", `Bearer ${token}`);
  return next;
}

export async function apiFetch(input: RequestInfo | URL, init: RequestInit = {}): Promise<Response> {
  const res = await globalThis.fetch(input, { ...init, headers: authHeaders(init.headers) });
  if (res.status === 401) emitAuth(true);
  return res;
}

const fetch = apiFetch;

export function apiAssetUrl(url: string | null | undefined): string {
  if (!url) return "";
  const token = readToken();
  if (!token || !url.startsWith("/api/")) return url;
  const separator = url.includes("?") ? "&" : "?";
  return `${url}${separator}token=${encodeURIComponent(token)}`;
}

function withImageAuth(image: ImageItem): ImageItem {
  return {
    ...image,
    url: apiAssetUrl(image.url),
    thumb_url: image.thumb_url ? apiAssetUrl(image.thumb_url) : null,
  };
}

function withTtsAuth(result: TtsGenerateResult): TtsGenerateResult {
  return { ...result, url: apiAssetUrl(result.url) };
}

export const apiAuth = {
  getToken: readToken,
  setToken(token: string) {
    const clean = token.trim();
    try {
      if (clean) localStorage.setItem(TOKEN_KEY, clean);
      else localStorage.removeItem(TOKEN_KEY);
    } catch {
      // Auth remains in-memory-unavailable; the next call will still fail loudly.
    }
    emitAuth(false);
  },
  clearToken() {
    this.setToken("");
  },
  subscribe(listener: AuthListener): () => void {
    authListeners.add(listener);
    return () => authListeners.delete(listener);
  },
};

async function j<T>(res: Response): Promise<T> {
  if (res.status === 401) emitAuth(true);
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json() as Promise<T>;
}

export const api = {
  health: () => globalThis.fetch("/api/health").then(j<HealthStatus>),
  listModels: () => fetch("/api/models").then(j<Model[]>),
  listLoras: () => fetch("/api/loras").then(j<Lora[]>),
  listModelProfiles: () => fetch("/api/models/profiles").then(j<ModelProfile[]>),
  resetModelProfile: (id: string) => fetch(`/api/models/profiles/${encodeURIComponent(id)}`, { method: "DELETE" }).then(j<{ deleted: number }>),
  resetAllModelProfiles: () => fetch("/api/models/profiles", { method: "DELETE" }).then(j<{ deleted: number }>),
  runtimeSettings: () => fetch("/api/settings").then(j<RuntimeSettings>),
  settingsOverrides: () => fetch("/api/settings/overrides").then(j<SettingsOverrides>),
  saveSettingsOverrides: (body: Partial<SettingsOverrides["values"]>) =>
    fetch("/api/settings/overrides", { method: "PUT", headers: JSON_HEADERS, body: JSON.stringify(body) })
      .then(j<SettingsOverrides>),
  gpuStatus: () => fetch("/api/gpu").then(j),
  freeGpu: () => fetch("/api/gpu/free", { method: "POST" }).then(j),

  listJobs: () => fetch("/api/jobs").then(j<Job[]>),
  createJobs: (jobs: JobCreate[]) =>
    fetch("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(jobs),
    }).then(j<Job[]>),
  uploadInitImage: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return fetch("/api/images/upload", { method: "POST", body: fd })
      .then(j<{ init_image: string; url: string; width: number; height: number }>)
      .then((res) => ({ ...res, url: apiAssetUrl(res.url) }));
  },
  uploadMaskImage: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return fetch("/api/images/upload-mask", { method: "POST", body: fd })
      .then(j<{ mask_image: string; url: string; width: number; height: number }>)
      .then((res) => ({ ...res, url: apiAssetUrl(res.url) }));
  },
  queuePlan: () => fetch("/api/jobs/plan").then(j<QueuePlan>),
  getJob: (id: string) => fetch(`/api/jobs/${id}`).then(j<Job>),
  cancelJob: (id: string) => fetch(`/api/jobs/${id}`, { method: "DELETE" }).then(j<Job>),
  getLlmConfig: () => fetch("/api/llm/config").then(j<LlmConfig>),
  setLlmConfig: (body: { ctx?: number; ngl?: number; backend?: string; context_type?: string }) =>
    fetch("/api/llm/config", { method: "POST", headers: JSON_HEADERS, body: JSON.stringify(body) })
      .then(j<LlmConfig & { changed: boolean; reloaded: boolean; note: string | null }>),
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
    return fetch(`/api/images${params}`).then(j<ImageItem[]>).then((rows) => rows.map(withImageAuth));
  },
  queryImages: (opts: { q?: string; model?: string; family?: string; size?: string; lora?: string; favorite?: boolean; tag?: string; date_from?: string; date_to?: string; limit?: number; offset?: number } = {}) => {
    const p = new URLSearchParams();
    if (opts.q?.trim()) p.set("q", opts.q.trim());
    if (opts.model) p.set("model", opts.model);
    if (opts.family) p.set("family", opts.family);
    if (opts.size) p.set("size", opts.size);
    if (opts.lora) p.set("lora", opts.lora);
    if (opts.favorite != null) p.set("favorite", String(opts.favorite));
    if (opts.tag) p.set("tag", opts.tag);
    if (opts.date_from) p.set("date_from", opts.date_from);
    if (opts.date_to) p.set("date_to", opts.date_to);
    if (opts.limit != null) p.set("limit", String(opts.limit));
    if (opts.offset != null) p.set("offset", String(opts.offset));
    const qs = p.toString();
    return fetch(`/api/images${qs ? `?${qs}` : ""}`).then(j<ImageItem[]>).then((rows) => rows.map(withImageAuth));
  },
  imageStats: () => fetch("/api/images/stats").then(j<ImageStats>),
  updateImage: (id: string, body: { favorite?: boolean; tags?: string[] }) =>
    fetch(`/api/images/${id}`, { method: "PATCH", headers: JSON_HEADERS, body: JSON.stringify(body) })
      .then(j<ImageItem>)
      .then(withImageAuth),
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
      .then(j<TtsGenerateResult>)
      .then(withTtsAuth),

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
  voiceEngineStatus: () => fetch("/api/voice/engine/status").then(j<VoiceEngineStatus>),
  voiceEngineSettings: (body: VoiceEngineSettingsUpdate) =>
    fetch("/api/voice/engine/settings", { method: "POST", headers: JSON_HEADERS, body: JSON.stringify(body) })
      .then(j<VoiceEngineStatus>),
  voiceEnginePresets: () => fetch("/api/voice/engine/presets").then(j<VoiceEnginePreset[]>),
  voiceEnginePresetCreate: (body: { name: string; settings: VoiceEngineSettingsUpdate; model_id?: string | null }) =>
    fetch("/api/voice/engine/presets", { method: "POST", headers: JSON_HEADERS, body: JSON.stringify(body) })
      .then(j<VoiceEnginePreset>),
  voiceEnginePresetUpdate: (id: string, body: { name?: string | null; settings?: VoiceEngineSettingsUpdate | null; model_id?: string | null }) =>
    fetch(`/api/voice/engine/presets/${id}`, { method: "PATCH", headers: JSON_HEADERS, body: JSON.stringify(body) })
      .then(j<VoiceEnginePreset>),
  voiceEnginePresetDelete: (id: string) =>
    fetch(`/api/voice/engine/presets/${id}`, { method: "DELETE" }).then(j<{ deleted: string }>),
  voiceEngineSessionStart: (modelId: string) =>
    fetch("/api/voice/engine/session/start", {
      method: "POST",
      headers: JSON_HEADERS,
      body: JSON.stringify({ model_id: modelId }),
    }).then(j<VoiceEngineStatus>),
  voiceEngineSessionStop: () => fetch("/api/voice/engine/session/stop", { method: "POST" }).then(j<VoiceEngineStatus>),
  voiceEngineRecordingStart: () => fetch("/api/voice/engine/recording/start", { method: "POST" }).then(j<VoiceEngineStatus>),
  voiceEngineRecordingStop: () =>
    fetch("/api/voice/engine/recording/stop", { method: "POST" })
      .then(j<VoiceEngineStatus>)
      .then((res) => ({
        ...res,
        recording_result: res.recording_result
          ? {
              ...res.recording_result,
              url: apiAssetUrl(res.recording_result.url),
              mp3_url: apiAssetUrl(res.recording_result.mp3_url),
            }
          : undefined,
      })),
  voiceEngineConvert: (form: FormData) =>
    fetch("/api/voice/engine/convert", { method: "POST", body: form })
      .then(j<VoiceEngineConvertResult>)
      .then((res) => ({ ...res, url: apiAssetUrl(res.url), mp3_url: apiAssetUrl(res.mp3_url) })),
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
  assetUrl: apiAssetUrl,
  downloadUrlBlob: async (url: string) => {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
    return res.blob();
  },
};
