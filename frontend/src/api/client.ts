import type { ImageItem, Job, JobCreate, JobType, Lora, Model, Preset, RuntimeSettings } from "../types";

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
  expand: (idea: string, model_id: string, style?: string) =>
    fetch("/api/jobs/expand", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ idea, model_id, style }),
    }).then(j<Job>),
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
  listPresets: () => fetch("/api/presets").then(j<Preset[]>),
  createPreset: (name: string, type: JobType, params: Record<string, unknown>) =>
    fetch("/api/presets", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, type, params }),
    }).then(j<Preset>),
  deletePreset: (id: string) => fetch(`/api/presets/${id}`, { method: "DELETE" }).then(j),
};
