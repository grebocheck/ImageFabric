import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import { api } from "./api/client";
import { useEvents } from "./api/useEvents";
import { ChatPanel } from "./components/ChatPanel";
import type { ChatJump } from "./components/ChatPanel";
import { CodePanel } from "./components/CodePanel";
import { CommandPalette, type Command } from "./components/CommandPalette";
import { ImageComposer } from "./components/ImageComposer";
import { Gallery } from "./components/Gallery";
import { ModelStatus, type View } from "./components/ModelStatus";
import { NotesPanel } from "./components/NotesPanel";
import { QueuePanel } from "./components/QueuePanel";
import { SettingsPanel } from "./components/SettingsPanel";
import { SystemPanel } from "./components/SystemPanel";
import { TtsPanel } from "./components/TtsPanel";
import type { BusEvent, GpuStatus, ImageItem, Job, Lora, MemSnapshot, Model, Preset } from "./types";

// A workspace is one top-level tab. Adding a tab = one entry here (label drives
// the header tab + command palette; render() owns the whole main area).
type Workspace = { id: View; label: string; render: () => ReactNode };

export default function App() {
  const [models, setModels] = useState<Model[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [images, setImages] = useState<ImageItem[]>([]);
  const [presets, setPresets] = useState<Preset[]>([]);
  const [loras, setLoras] = useState<Lora[]>([]);
  const [gpu, setGpu] = useState<GpuStatus>({ resident: null, model_id: null, model: null, family: null, warm: [] });
  const [mem, setMem] = useState<MemSnapshot | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [view, setView] = useState<View>("images");
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [chatJump, setChatJump] = useState<ChatJump | null>(null);

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
        case "mem.status":
          setMem({
            ram: (e.ram as MemSnapshot["ram"]) ?? null,
            vram: (e.vram as MemSnapshot["vram"]) ?? null,
          });
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

  const imageJobs = jobs.filter((j) => j.type === "image");

  // --- workspace registry: the single source for tabs + main rendering ---
  const workspaces: Workspace[] = [
    {
      id: "images",
      label: "Images",
      render: () => (
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
      ),
    },
    {
      id: "llm",
      label: "LLM",
      render: () => (
        <main className="flex-1 overflow-hidden p-4">
          <ChatPanel models={models} jump={chatJump} />
        </main>
      ),
    },
    {
      id: "notes",
      label: "Notes",
      render: () => (
        <main className="flex-1 overflow-hidden p-4">
          <NotesPanel />
        </main>
      ),
    },
    {
      id: "tts",
      label: "TTS",
      render: () => (
        <main className="flex-1 overflow-hidden p-4">
          <TtsPanel />
        </main>
      ),
    },
    {
      id: "code",
      label: "Code",
      render: () => (
        <main className="flex-1 overflow-hidden p-4">
          <CodePanel
            models={models}
            onOpenChat={(conversationId, jobId) => {
              setChatJump({ conversationId, jobId, nonce: Date.now() });
              setView("llm");
            }}
          />
        </main>
      ),
    },
    {
      id: "system",
      label: "System",
      render: () => (
        <main className="flex-1 overflow-hidden p-4">
          <SystemPanel gpu={gpu} mem={mem} />
        </main>
      ),
    },
  ];
  const active = workspaces.find((w) => w.id === view) ?? workspaces[0];

  const commands = useMemo<Command[]>(() => [
    ...workspaces.map((w) => ({ id: `go-${w.id}`, label: `Go to ${w.label}`, hint: "tab", run: () => setView(w.id) })),
    { id: "settings", label: "Open Settings", run: () => setSettingsOpen(true) },
    { id: "free", label: "Free GPU", hint: "unload models", run: onFree },
    // eslint-disable-next-line react-hooks/exhaustive-deps
  ], [onFree]);

  return (
    <div className="flex h-screen flex-col">
      <ModelStatus
        gpu={gpu}
        connected={connected}
        view={view}
        tabs={workspaces.map(({ id, label }) => ({ id, label }))}
        onView={setView}
        onFree={onFree}
        onSettings={() => setSettingsOpen(true)}
        onPalette={() => setPaletteOpen(true)}
      />

      {active.render()}

      <SettingsPanel open={settingsOpen} onClose={() => setSettingsOpen(false)} />
      <CommandPalette open={paletteOpen} commands={commands} onClose={() => setPaletteOpen(false)} />
    </div>
  );
}
