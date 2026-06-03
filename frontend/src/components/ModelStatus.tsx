import type { GpuStatus } from "../types";

const familyColor: Record<string, string> = {
  flux: "bg-violet-600",
  sdxl: "bg-pink-600",
  gguf: "bg-emerald-600",
};

export function ModelStatus({
  gpu,
  connected,
  onFree,
  onSettings,
}: {
  gpu: GpuStatus;
  connected: boolean;
  onFree: () => void;
  onSettings: () => void;
}) {
  return (
    <header className="flex items-center justify-between border-b border-white/10 px-5 py-3">
      <div className="flex items-center gap-3">
        <span className="text-lg font-semibold tracking-tight">ImageFabric</span>
        <span
          className={`h-2 w-2 rounded-full ${connected ? "bg-emerald-400" : "bg-red-500"}`}
          title={connected ? "connected" : "disconnected"}
        />
      </div>

      <div className="flex items-center gap-3 text-sm">
        <span className="text-white/50">VRAM resident:</span>
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
      </div>
    </header>
  );
}
