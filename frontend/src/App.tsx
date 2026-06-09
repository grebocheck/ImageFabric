import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
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
import { ResultPreview } from "./components/ResultPreview";
import { RagPanel } from "./components/RagPanel";
import { SettingsPanel } from "./components/SettingsPanel";
import { SystemPanel } from "./components/SystemPanel";
import { toast, ToastHost } from "./components/Toast";
import { TranscriptionPanel } from "./components/TranscriptionPanel";
import { TtsPanel } from "./components/TtsPanel";
import { VisionPanel } from "./components/VisionPanel";
import { VoicePanel } from "./components/VoicePanel";
import type { AppTheme, ArbiterNote, BusEvent, ComposerApply, GpuStatus, ImageItem, Job, Lora, MemPoint, MemSnapshot, Model, Preset } from "./types";

const MEM_HISTORY_MAX = 90; // rolling timeline points (~a few minutes at the poll rate)
const THEME_KEY = "hfabric.theme";
const THEMES: AppTheme[] = ["dark", "dim", "light"];
const THEME_META: Record<AppTheme, string> = {
  dark: "#0b0d12",
  dim: "#12151b",
  light: "#f5f7fb",
};

// A workspace is one top-level tab. Adding a tab = one entry here (label drives
// the header tab + command palette; render() owns the whole main area).
type Workspace = { id: View; label: string; render: () => ReactNode };

function readTheme(): AppTheme {
  const value = localStorage.getItem(THEME_KEY);
  return value === "dark" || value === "dim" || value === "light" ? value : "dark";
}

export default function App() {
  const [models, setModels] = useState<Model[]>([]);
  const [modelsLoading, setModelsLoading] = useState(true);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [images, setImages] = useState<ImageItem[]>([]);
  const [presets, setPresets] = useState<Preset[]>([]);
  const [presetsLoading, setPresetsLoading] = useState(true);
  const [loras, setLoras] = useState<Lora[]>([]);
  const [lorasLoading, setLorasLoading] = useState(true);
  const [gpu, setGpu] = useState<GpuStatus>({ resident: null, model_id: null, model: null, family: null, warm: [] });
  const [mem, setMem] = useState<MemSnapshot | null>(null);
  const [memHistory, setMemHistory] = useState<MemPoint[]>([]);
  const [arbiterNote, setArbiterNote] = useState<ArbiterNote | null>(null);
  // latest resident model name, read inside the mem.status handler without
  // making it depend on (and re-subscribe to) gpu state.
  const gpuRef = useRef<GpuStatus>(gpu);
  useEffect(() => { gpuRef.current = gpu; }, [gpu]);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [view, setView] = useState<View>(() => (localStorage.getItem("hfabric.view") as View) || "images");
  const [theme, setTheme] = useState<AppTheme>(() => readTheme());
  const [paletteOpen, setPaletteOpen] = useState(false);
  const tabIdsRef = useRef<View[]>([]);
  const [chatJump, setChatJump] = useState<ChatJump | null>(null);

  const [promptDraft, setPromptDraft] = useState("");
  // LLM composer draft, lifted so it survives tab switches (ChatPanel unmounts
  // when you leave the LLM tab).
  const [chatDraft, setChatDraft] = useState("");
  // History self-fetches; bump this to make it reload after a new image lands.
  const [imageEpoch, setImageEpoch] = useState(0);
  // A "reproduce from History" request handed to the image composer.
  const [composerApply, setComposerApply] = useState<ComposerApply | null>(null);

  const refreshJobs = useCallback(() => api.listJobs().then(setJobs).catch(() => {}), []);
  const refreshImages = useCallback((q?: string) => api.listImages(q).then(setImages).catch(() => {}), []);
  const refreshModels = useCallback(async () => {
    setModelsLoading(true);
    try {
      setModels(await api.listModels());
    } catch {
      // The UI keeps its last known model list if the backend is momentarily down.
    } finally {
      setModelsLoading(false);
    }
  }, []);
  const refreshLoras = useCallback(async () => {
    setLorasLoading(true);
    try {
      setLoras(await api.listLoras());
    } catch {
      // Same stale-list behavior as models.
    } finally {
      setLorasLoading(false);
    }
  }, []);
  const refreshPresets = useCallback(async () => {
    setPresetsLoading(true);
    try {
      setPresets(await api.listPresets());
    } catch {
      // Presets are optional polish; failed refresh should not block generation.
    } finally {
      setPresetsLoading(false);
    }
  }, []);

  useEffect(() => {
    void refreshModels();
    void refreshLoras();
    refreshJobs();
    refreshImages();
    refreshPresets();
  }, [refreshModels, refreshLoras, refreshJobs, refreshImages, refreshPresets]);

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
          refreshJobs();
          break;
        case "job.error":
          refreshJobs();
          toast.error(`Job failed${typeof e.error === "string" && e.error ? `: ${e.error}` : ""}`);
          break;
        case "job.done":
          refreshJobs();
          if (e.job_type === "image") {
            refreshImages();
            setImageEpoch((n) => n + 1);
            toast.success("Image ready", { onClick: () => setView("history") });
          }
          break;
        case "image.ready":
          refreshImages();
          setImageEpoch((n) => n + 1);
          break;
        case "mem.status": {
          const snap: MemSnapshot = {
            ram: (e.ram as MemSnapshot["ram"]) ?? null,
            vram: (e.vram as MemSnapshot["vram"]) ?? null,
          };
          setMem(snap);
          setMemHistory((prev) => [
            ...prev,
            { ts: e.ts, ram: snap.ram, vram: snap.vram, resident: gpuRef.current.model },
          ].slice(-MEM_HISTORY_MAX));
          break;
        }
        case "arbiter.note":
          setArbiterNote({
            reason: String(e.reason ?? ""),
            message: String(e.message ?? ""),
            model: typeof e.model === "string" ? e.model : undefined,
            family: typeof e.family === "string" ? e.family : undefined,
            predicted_gb: typeof e.predicted_gb === "number" ? e.predicted_gb : undefined,
            available_gb: typeof e.available_gb === "number" ? e.available_gb : undefined,
            ts: e.ts,
          });
          if (e.reason === "ram_budget") toast.error(String(e.message ?? "Load refused by RAM guard"));
          break;
      }
    },
    [refreshJobs, refreshImages],
  );

  const { connected } = useEvents(onEvent);

  const onFree = useCallback(() => api.freeGpu().catch(() => {}), []);
  const cycleTheme = useCallback(() => {
    setTheme((current) => THEMES[(THEMES.indexOf(current) + 1) % THEMES.length]);
  }, []);

  // Reproduce a History image in the composer. The stored snapshot keys the
  // model by *name*; resolve it back to a live model id when one matches.
  const onReproduce = useCallback(
    (image: ImageItem, opts: { keepSeed: boolean }) => {
      const modelName = typeof image.params?.model === "string" ? image.params.model : "";
      const model = models.find((m) => m.job_type === "image" && m.name === modelName);
      const params = { ...image.params, seed: opts.keepSeed ? image.seed ?? -1 : -1 };
      setComposerApply({ model_id: model?.id, params, nonce: Date.now() });
      setView("images");
      toast.success(opts.keepSeed ? "Loaded into composer" : "Loaded as variation (new seed)");
    },
    [models],
  );

  // remember the last active tab
  useEffect(() => { localStorage.setItem("hfabric.view", view); }, [view]);
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    document.documentElement.classList.toggle("dark", theme !== "light");
    document.querySelector('meta[name="theme-color"]')?.setAttribute("content", THEME_META[theme]);
    localStorage.setItem(THEME_KEY, theme);
  }, [theme]);

  // global shortcuts: Ctrl/Cmd+K opens the palette; Alt+1..N switches tabs
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setPaletteOpen((v) => !v);
      } else if (e.altKey && /^[1-9]$/.test(e.key)) {
        const target = tabIdsRef.current[Number(e.key) - 1];
        if (target) { e.preventDefault(); setView(target); }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const imageJobs = jobs.filter((j) => j.type === "image");
  const busy = jobs.some((j) => j.status === "running");
  // Changes whenever the pending queue changes, so the System tab can refetch
  // the swap-plan preview without polling.
  const queueKey = useMemo(
    () => jobs
      .filter((j) => j.status === "queued" || j.status === "running")
      .map((j) => `${j.id}:${j.status}:${j.priority}`)
      .join("|"),
    [jobs],
  );

  // --- workspace registry: the single source for tabs + main rendering ---
  const workspaces: Workspace[] = [
    {
      id: "images",
      label: "Images",
      render: () => (
        <main className="grid flex-1 grid-cols-[390px_minmax(0,1fr)_330px] grid-rows-[minmax(0,1fr)] gap-4 overflow-hidden p-4 max-[1240px]:grid-cols-[380px_minmax(0,1fr)] max-[1240px]:grid-rows-[minmax(0,1fr)_300px] max-[860px]:block max-[860px]:overflow-y-auto">
          <ImageComposer
            models={models}
            modelsLoading={modelsLoading}
            loras={loras}
            lorasLoading={lorasLoading}
            presets={presets}
            presetsLoading={presetsLoading}
            onPresetsChanged={refreshPresets}
            promptDraft={promptDraft}
            setPromptDraft={setPromptDraft}
            apply={composerApply}
          />
          <ResultPreview
            images={images}
            onOpenHistory={() => setView("history")}
            generating={imageJobs.some((j) => j.status === "running")}
          />
          <QueuePanel jobs={imageJobs} onChanged={refreshJobs} note={arbiterNote} />
        </main>
      ),
    },
    {
      id: "history",
      label: "History",
      render: () => (
        <main className="flex-1 overflow-hidden p-4">
          <Gallery models={models} reloadSignal={imageEpoch} onReproduce={onReproduce} />
        </main>
      ),
    },
    {
      id: "llm",
      label: "LLM",
      render: () => (
        <main className="flex-1 overflow-hidden p-4">
          <ChatPanel models={models} modelsLoading={modelsLoading} jump={chatJump} draft={chatDraft} setDraft={setChatDraft} />
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
      id: "transcription",
      label: "Transcribe",
      render: () => (
        <main className="flex-1 overflow-hidden p-4">
          <TranscriptionPanel />
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
            modelsLoading={modelsLoading}
            onOpenChat={(conversationId, jobId) => {
              setChatJump({ conversationId, jobId, nonce: Date.now() });
              setView("llm");
            }}
          />
        </main>
      ),
    },
    {
      id: "rag",
      label: "RAG",
      render: () => (
        <main className="flex-1 overflow-hidden p-4">
          <RagPanel
            models={models}
            modelsLoading={modelsLoading}
            onOpenChat={(conversationId, jobId) => {
              setChatJump({ conversationId, jobId, nonce: Date.now() });
              setView("llm");
            }}
          />
        </main>
      ),
    },
    {
      id: "vision",
      label: "Vision",
      render: () => (
        <main className="flex-1 overflow-hidden p-4">
          <VisionPanel />
        </main>
      ),
    },
    {
      id: "voice",
      label: "Voice",
      render: () => (
        <main className="flex-1 overflow-hidden p-4">
          <VoicePanel />
        </main>
      ),
    },
    {
      id: "system",
      label: "System",
      render: () => (
        <main className="flex-1 overflow-hidden p-4">
          <SystemPanel gpu={gpu} mem={mem} history={memHistory} note={arbiterNote} queueKey={queueKey} imageSignal={imageEpoch} />
        </main>
      ),
    },
  ];
  const active = workspaces.find((w) => w.id === view) ?? workspaces[0];
  tabIdsRef.current = workspaces.map((w) => w.id);

  const commands = useMemo<Command[]>(() => [
    ...workspaces.map((w) => ({ id: `go-${w.id}`, label: `Go to ${w.label}`, hint: "tab", run: () => setView(w.id) })),
    { id: "settings", label: "Open Settings", run: () => setSettingsOpen(true) },
    { id: "theme", label: "Cycle Theme", hint: theme, run: cycleTheme },
    { id: "free", label: "Free GPU", hint: "unload models", run: onFree },
    // eslint-disable-next-line react-hooks/exhaustive-deps
  ], [cycleTheme, onFree, theme]);

  return (
    <div className="flex h-screen flex-col">
      <ModelStatus
        gpu={gpu}
        connected={connected}
        busy={busy}
        mem={mem}
        view={view}
        theme={theme}
        tabs={workspaces.map(({ id, label }) => ({ id, label }))}
        onView={setView}
        onFree={onFree}
        onTheme={cycleTheme}
        onSettings={() => setSettingsOpen(true)}
        onPalette={() => setPaletteOpen(true)}
      />

      {active.render()}

      <SettingsPanel open={settingsOpen} onClose={() => setSettingsOpen(false)} />
      <CommandPalette open={paletteOpen} commands={commands} onClose={() => setPaletteOpen(false)} />
      <ToastHost />
    </div>
  );
}
