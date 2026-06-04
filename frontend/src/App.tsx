import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "./api/client";
import { useEvents } from "./api/useEvents";
import { ChatPanel } from "./components/ChatPanel";
import { CommandPalette, type Command } from "./components/CommandPalette";
import { ImageComposer } from "./components/ImageComposer";
import { Gallery } from "./components/Gallery";
import { ModelStatus, type View } from "./components/ModelStatus";
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
  const [view, setView] = useState<View>("images");
  const [paletteOpen, setPaletteOpen] = useState(false);

  const [promptDraft, setPromptDraft] = useState("");

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
        case "job.created":
        case "job.started":
        case "job.cancelled":
        case "job.error":
          refreshJobs();
          break;
        case "job.done":
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

  const onFree = useCallback(() => api.freeGpu().catch(() => {}), []);

  // global Ctrl/Cmd+K opens the command palette
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setPaletteOpen((v) => !v);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const commands = useMemo<Command[]>(() => [
    { id: "images", label: "Go to Images", hint: "tab", run: () => setView("images") },
    { id: "llm", label: "Go to LLM / Chat", hint: "tab", run: () => setView("llm") },
    { id: "settings", label: "Open Settings", run: () => setSettingsOpen(true) },
    { id: "free", label: "Free GPU", hint: "unload models", run: onFree },
  ], [onFree]);

  const imageJobs = jobs.filter((j) => j.type === "image");

  return (
    <div className="flex h-screen flex-col">
      <ModelStatus
        gpu={gpu}
        connected={connected}
        view={view}
        onView={setView}
        onFree={onFree}
        onSettings={() => setSettingsOpen(true)}
        onPalette={() => setPaletteOpen(true)}
      />

      {view === "images" ? (
        <main className="grid flex-1 grid-cols-[380px_320px_1fr] gap-4 overflow-hidden p-4">
          <div className="overflow-y-auto">
            <ImageComposer
              models={models}
              loras={loras}
              presets={presets}
              onPresetsChanged={refreshPresets}
              promptDraft={promptDraft}
              setPromptDraft={setPromptDraft}
            />
          </div>
          <QueuePanel jobs={imageJobs} onChanged={refreshJobs} />
          <Gallery images={images} onSearch={refreshImages} />
        </main>
      ) : (
        <main className="flex-1 overflow-hidden p-4">
          <ChatPanel models={models} />
        </main>
      )}

      <SettingsPanel open={settingsOpen} onClose={() => setSettingsOpen(false)} />
      <CommandPalette open={paletteOpen} commands={commands} onClose={() => setPaletteOpen(false)} />
    </div>
  );
}
