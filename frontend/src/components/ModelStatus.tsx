import { Logo } from "./Logo";
import type { AppTheme, GpuStatus, MemSnapshot } from "../types";

export type View = "images" | "history" | "llm" | "notes" | "tts" | "transcription" | "code" | "rag" | "vision" | "voice" | "system";

const familyColor: Record<string, string> = {
  flux: "bg-accent",
  flux2: "bg-sky-600",
  "qwen-image": "bg-violet-600",
  "z-image": "bg-cyan-600",
  sdxl: "bg-pink-600",
  gguf: "bg-emerald-600",
};

const themeLabel: Record<AppTheme, string> = {
  dark: "Dark",
  dim: "Dim",
  light: "Light",
};

export function ModelStatus({
  gpu,
  connected,
  busy,
  mem,
  view,
  theme,
  tabs,
  onView,
  onFree,
  onTheme,
  onSettings,
  onPalette,
}: {
  gpu: GpuStatus;
  connected: boolean;
  busy: boolean;
  mem: MemSnapshot | null;
  view: View;
  theme: AppTheme;
  tabs: { id: View; label: string }[];
  onView: (v: View) => void;
  onFree: () => void;
  onTheme: () => void;
  onSettings: () => void;
  onPalette: () => void;
}) {
  return (
    <header className="flex items-center justify-between gap-4 border-b border-line px-5 py-3">
      <div className="flex min-w-0 items-center gap-5">
        <div className="flex shrink-0 items-center gap-2">
          <Logo className="h-7 w-7" />
          <span className="text-lg font-semibold tracking-tight">HFabric</span>
          {busy ? (
            <svg className="h-3.5 w-3.5 animate-spin text-accent" viewBox="0 0 24 24" fill="none" aria-label="working">
              <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="3" className="opacity-25" />
              <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
            </svg>
          ) : (
            <span
              className={`h-2 w-2 rounded-full ${connected ? "bg-emerald-400" : "bg-red-500"}`}
              title={connected ? "connected" : "disconnected"}
            />
          )}
        </div>

        <nav className="flex min-w-0 items-center gap-1 overflow-x-auto rounded-lg border border-white/10 bg-black/20 p-1">
          {tabs.map((t) => (
            <button
              key={t.id}
              onClick={() => onView(t.id)}
              className={`rounded-md px-3 py-1 text-sm font-medium transition ${
                view === t.id
                  ? "bg-white/15 text-white"
                  : "text-white/50 hover:text-white/80"
              }`}
            >
              {t.label}
            </button>
          ))}
        </nav>
      </div>

      <div className="flex shrink-0 items-center gap-3 text-sm">
        {mem?.vram ? (
          <div
            className="flex items-center gap-1.5"
            title={`VRAM ${mem.vram.used_gb.toFixed(1)} / ${mem.vram.total_gb.toFixed(1)} GB`}
          >
            <span className="text-xs text-white/35">VRAM</span>
            <div className="h-1.5 w-16 overflow-hidden rounded-full bg-white/10">
              <div
                className="h-full bg-accent transition-all"
                style={{ width: `${Math.min(100, (mem.vram.used_gb / Math.max(1, mem.vram.total_gb)) * 100)}%` }}
              />
            </div>
          </div>
        ) : null}
        <span className="text-white/50">Active model:</span>
        {gpu.model ? (
          <span className="flex min-w-0 items-center gap-2">
            <span
              className={`rounded px-1.5 py-0.5 text-xs font-medium ${
                familyColor[gpu.family ?? ""] ?? "bg-slate-600"
              }`}
            >
              {gpu.family}
            </span>
            <span className="max-w-52 truncate font-mono" title={gpu.model}>{gpu.model}</span>
            <span className="rounded bg-emerald-500/15 px-1.5 py-0.5 text-[10px] font-medium text-emerald-300">
              on GPU
            </span>
          </span>
        ) : (
          <span className="text-white/40">idle</span>
        )}
        {gpu.warm?.length ? (
          <span className="flex items-center gap-1 text-xs text-white/45">
            <span>CPU warm:</span>
            <span
              className="max-w-44 truncate font-mono text-white/60"
              title={gpu.warm.map((m) => m.model).join(", ")}
            >
              {gpu.warm.map((m) => m.model).join(", ")}
            </span>
          </span>
        ) : null}
        <button
          onClick={onFree}
          disabled={!gpu.model && !gpu.warm?.length}
          className="rounded border border-white/15 px-2.5 py-1 text-xs hover:bg-white/10 disabled:opacity-30"
        >
          Free GPU
        </button>
        <button
          onClick={onTheme}
          title="Cycle theme"
          className="rounded border border-white/15 px-2.5 py-1 text-xs hover:bg-white/10"
        >
          {themeLabel[theme]}
        </button>
        <button
          onClick={onSettings}
          className="rounded border border-white/15 px-2.5 py-1 text-xs hover:bg-white/10"
        >
          Settings
        </button>
        <button
          onClick={onPalette}
          title="Command palette (Ctrl+K)"
          className="rounded border border-white/15 px-2 py-1 text-xs text-white/50 hover:bg-white/10"
        >
          Ctrl K
        </button>
      </div>
    </header>
  );
}
