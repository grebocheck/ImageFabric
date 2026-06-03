import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api/client";
import { useEvents } from "./api/useEvents";
import { Composer } from "./components/Composer";
import { Gallery } from "./components/Gallery";
import { ModelStatus } from "./components/ModelStatus";
import { QueuePanel } from "./components/QueuePanel";
import { SettingsPanel } from "./components/SettingsPanel";
import type { BusEvent, GpuStatus, ImageItem, Job, Lora, Model, Preset } from "./types";

export default function App() {
  const [models, setModels] = useState<Model[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [images, setImages] = useState<ImageItem[]>([]);
  const [presets, setPresets] = useState<Preset[]>([]);
  const [loras, setLoras] = useState<Lora[]>([]);
  const [gpu, setGpu] = useState<GpuStatus>({ resident: null, model_id: null, model: null, family: null, warm: [] });
  const [settingsOpen, setSettingsOpen] = useState(false);

  const [promptDraft, setPromptDraft] = useState("");
  const [expanding, setExpanding] = useState(false);
  const expandJobId = useRef<string | null>(null);

  const refreshJobs = useCallback(() => api.listJobs().then(setJobs).catch(() => {}), []);
  const refreshImages = useCallback((q?: string) => api.listImages(q).then(setImages).catch(() => {}), []);
  const refreshPresets = useCallback(() => api.listPresets().then(setPresets).catch(() => {}), []);

  useEffect(() => {
    api.listModels().then(setModels).catch(() => {});
    api.listLoras().then(setLoras).catch(() => {});
    refreshJobs();
    refreshImages();
    refreshPresets();
  }, [refreshJobs, refreshImages, refreshPresets]);

  const onEvent = useCallback(
    (e: BusEvent) => {
      switch (e.type) {
        case "gpu.status":
          setGpu({
            resident: (e.resident as string) ?? null,
            model_id: (e.model_id as string) ?? null,
            model: (e.model as string) ?? null,
            family: (e.family as string) ?? null,
            warm: Array.isArray(e.warm) ? (e.warm as GpuStatus["warm"]) : [],
          });
          break;
        case "job.progress":
          setJobs((prev) =>
            prev.map((j) => (
              j.id === e.job_id
                ? {
                    ...j,
                    progress: e.progress as number,
                    progress_note: typeof e.note === "string" ? e.note : j.progress_note,
                  }
                : j
            )),
          );
          break;
        case "llm.token":
          if (e.job_id === expandJobId.current) setPromptDraft((p) => p + (e.token as string));
          break;
        case "job.created":
        case "job.started":
        case "job.cancelled":
        case "job.error":
          refreshJobs();
          break;
        case "job.done":
          if (e.job_id === expandJobId.current) {
            setExpanding(false);
            expandJobId.current = null;
            if (typeof e.text === "string") setPromptDraft(e.text);
          }
          refreshJobs();
          if (e.job_type === "image") refreshImages();
          break;
        case "image.ready":
          refreshImages();
          break;
      }
    },
    [refreshJobs, refreshImages],
  );

  const { connected } = useEvents(onEvent);

  const onExpand = useCallback(async (idea: string, llmModelId: string, style?: string) => {
    setPromptDraft("");
    setExpanding(true);
    try {
      const job = await api.expand(idea, llmModelId, style);
      expandJobId.current = job.id;
    } catch {
      setExpanding(false);
    }
  }, []);

  const onFree = useCallback(() => api.freeGpu().catch(() => {}), []);

  return (
    <div className="flex h-screen flex-col">
      <ModelStatus gpu={gpu} connected={connected} onFree={onFree} onSettings={() => setSettingsOpen(true)} />
      <main className="grid flex-1 grid-cols-[380px_320px_1fr] gap-4 overflow-hidden p-4">
        <div className="overflow-y-auto">
          <Composer
            models={models}
            loras={loras}
            presets={presets}
            onPresetsChanged={refreshPresets}
            promptDraft={promptDraft}
            setPromptDraft={setPromptDraft}
            expanding={expanding}
            onExpand={onExpand}
          />
        </div>
        <QueuePanel jobs={jobs} onChanged={refreshJobs} />
        <Gallery images={images} onSearch={refreshImages} />
      </main>
      <SettingsPanel open={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </div>
  );
}
