import type { GpuStatus } from "../types";

export type View = "images" | "llm" | "notes" | "tts" | "transcription" | "code" | "rag" | "vision" | "system";

const familyColor: Record<string, string> = {
  flux: "bg-violet-600",
  flux2: "bg-sky-600",
  sdxl: "bg-pink-600",
  gguf: "bg-emerald-600",
};

export function ModelStatus({
  gpu,
  connected,
  view,
  tabs,
  onView,
  onFree,
  onSettings,
  onPalette,
}: {
  gpu: GpuStatus;
  connected: boolean;
  view: View;
  tabs: { id: View; label: string }[];
  onView: (v: View) => void;
  onFree: () => void;
  onSettings: () => void;
  onPalette: () => void;
}) {
  return (
    <header className="flex items-center justify-between border-b border-white/10 px-5 py-3">
      <div className="flex items-center gap-5">
        <div className="flex items-center gap-2">
          <span className="text-lg font-semibold tracking-tight">HFabric</span>
          <span
            className={`h-2 w-2 rounded-full ${connected ? "bg-emerald-400" : "bg-red-500"}`}
            title={connected ? "connected" : "disconnected"}
          />
        </div>

        {/* --- workspace tabs --- */}
        <nav className="flex items-center gap-1 rounded-lg border border-white/10 bg-black/20 p-1">
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

      <div className="flex items-center gap-3 text-sm">
        <span className="text-white/50">Active model:</span>
        {gpu.model ? (
          <span className="flex items-center gap-2">
            <span
              className={`rounded px-1.5 py-0.5 text-xs font-medium ${
                familyColor[gpu.family ?? ""] ?? "bg-slate-600"
              }`}
            >
              {gpu.family}
            </span>
            <span className="font-mono">{gpu.model}</span>
            <span className="rounded bg-emerald-500/15 px-1.5 py-0.5 text-[10px] font-medium text-emerald-300">
              on GPU
            </span>
          </span>
        ) : (
          <span className="text-white/40">— idle —</span>
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
          ⌘K
        </button>
      </div>
    </header>
  );
}
