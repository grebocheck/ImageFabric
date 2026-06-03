import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { RuntimeSettings } from "../types";

export function SettingsPanel({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [settings, setSettings] = useState<RuntimeSettings | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!open) return;
    setError("");
    api.runtimeSettings()
      .then(setSettings)
      .catch((err) => setError(err instanceof Error ? err.message : "Could not load settings"));
  }, [open]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-20 bg-black/40" onClick={onClose}>
      <aside
        className="absolute right-4 top-16 flex max-h-[calc(100vh-5rem)] w-[420px] max-w-[calc(100vw-2rem)] flex-col rounded-lg border border-white/10 bg-zinc-950 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-white/10 px-4 py-3">
          <h2 className="text-sm font-semibold text-white/75">Settings</h2>
          <button onClick={onClose} className="rounded border border-white/15 px-2 py-1 text-xs hover:bg-white/10">
            Close
          </button>
        </div>
        <div className="min-h-0 overflow-y-auto p-4 text-sm">
          {error ? <div className="rounded-md border border-red-400/25 bg-red-400/10 p-2 text-xs text-red-200">{error}</div> : null}
          {!settings && !error ? <div className="text-white/35">loading...</div> : null}
          {settings ? (
            <div className="flex flex-col gap-4">
              <Section title="Runtime" rows={{
                "Stub mode": settings.stub_mode,
                "Models": settings.counts.models,
                "Image models": settings.counts.image_models,
                "LLM models": settings.counts.llm_models,
                "LoRAs": settings.counts.loras,
              }} />
              <Section title="Acceleration" rows={settings.acceleration} />
              <Section title="Memory" rows={settings.memory} />
              <Section title="Paths" rows={settings.paths} mono />
            </div>
          ) : null}
        </div>
      </aside>
    </div>
  );
}

function Section({ title, rows, mono = false }: { title: string; rows: Record<string, unknown>; mono?: boolean }) {
  return (
    <section>
      <h3 className="mb-2 text-xs uppercase tracking-wide text-white/40">{title}</h3>
      <dl className="divide-y divide-white/5 rounded-md border border-white/10">
        {Object.entries(rows).map(([key, value]) => (
          <div key={key} className="grid grid-cols-[140px_1fr] gap-2 px-2.5 py-2">
            <dt className="text-white/40">{key}</dt>
            <dd className={`min-w-0 truncate text-white/75 ${mono ? "font-mono text-xs" : ""}`} title={format(value)}>
              {format(value)}
            </dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

function format(value: unknown): string {
  if (value == null || value === "") return "-";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(2);
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}
