import type { ChatMessage, ImageItem, Job, JobCreate, JobType, LlmConfig, Lora, Model, Preset, RuntimeSettings } from "../types";

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
  chat: (body: {
    model_id: string;
    messages: ChatMessage[];
    system?: string;
    temperature?: number;
    max_tokens?: number;
  }) =>
    fetch("/api/jobs/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(j<Job>),
  getJob: (id: string) => fetch(`/api/jobs/${id}`).then(j<Job>),
  getLlmConfig: () => fetch("/api/llm/config").then(j<LlmConfig>),
  setLlmConfig: (body: { ctx?: number; ngl?: number }) =>
    fetch("/api/llm/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(j<LlmConfig & { changed: boolean; reloaded: boolean }>),
  cancelJob: (id: string) => fetch(`/api/jobs/${id}`, { method: "DELETE" }).then(j<Job>),
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
  deletePreset: (id: string) => fetch(`/api/presets/${id}`, { method: "DELETE" }).then(j),
};
