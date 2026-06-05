import { useCallback, useEffect, useState } from "react";
import { api } from "../api/client";
import { Badge } from "./Badge";
import type { VoiceStatus } from "../types";

function size(bytes: number): string {
  if (!bytes) return "0 B";
  const gb = bytes / 1e9;
  return gb >= 1 ? `${gb.toFixed(2)} GB` : `${(bytes / 1e6).toFixed(0)} MB`;
}

export function VoicePanel() {
  const [status, setStatus] = useState<VoiceStatus | null>(null);
  const [error, setError] = useState("");

  const refresh = useCallback(() => {
    api
      .voiceStatus()
      .then((s) => { setStatus(s); setError(""); })
      .catch((e) => setError(e instanceof Error ? e.message : "failed to load status"));
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  const models = status?.models ?? [];

  return (
    <div className="mx-auto flex h-full max-w-2xl flex-col gap-4 overflow-y-auto p-1">
      <header className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-white/85">Voice changer</h2>
          <p className="mt-1 text-sm text-white/45">
            Real-time voice conversion via <span className="text-white/65">w-okada</span> (MMVCServerSIO).
            Driving its conversion API lands in P6.2 — this tab detects the install, server, and voices.
          </p>
        </div>
        <button
          onClick={refresh}
          className="shrink-0 rounded-md border border-white/15 px-2.5 py-1.5 text-xs text-white/70 transition hover:bg-white/10 hover:text-white"
        >
          Refresh
        </button>
      </header>

      {error ? (
        <div className="rounded-md border border-red-400/30 bg-red-400/10 px-3 py-2 text-sm text-red-200">{error}</div>
      ) : null}

      <section className="rounded-lg border border-white/10 bg-surface p-4">
        <div className="mb-3 text-xs font-medium uppercase tracking-wide text-white/40">Engine</div>
        <div className="flex flex-col gap-2 text-sm">
          <Row label="Install" value={status?.wokada_installed ? (status.executable ?? "found") : "not found"} ok={status?.wokada_installed} mono />
          <Row label="Server" value={status?.server_reachable ? "reachable" : "not running"} ok={status?.server_reachable} />
          <Row label="URL" value={status?.server_url ?? "…"} mono />
          <Row label="Device" value={status?.device ?? "…"} />
        </div>
        {status && status.wokada_installed && !status.server_reachable ? (
          <p className="mt-3 rounded-md border border-amber-500/30 bg-amber-500/10 px-2.5 py-2 text-xs text-amber-100">
            Install detected but server not running. Launch <code className="text-amber-50">MMVCServerSIO.exe</code> to enable conversion.
          </p>
        ) : null}
        {status?.server_reachable ? (
          <a
            href={status.server_url}
            target="_blank"
            rel="noreferrer"
            className="mt-3 inline-block rounded-md border border-white/15 px-3 py-1.5 text-xs text-white/75 transition hover:bg-white/10 hover:text-white"
          >
            Open w-okada UI ↗
          </a>
        ) : null}
      </section>

      <section className="rounded-lg border border-white/10 bg-surface p-4">
        <div className="mb-3 flex items-center justify-between">
          <div className="text-xs font-medium uppercase tracking-wide text-white/40">Voices</div>
          <Badge>{models.length}</Badge>
        </div>
        {models.length === 0 ? (
          <p className="text-sm leading-6 text-white/40">
            No voice slots found in <code className="text-white/60">{status?.model_dir ?? "model_dir"}</code>.
            Import an RVC model in the w-okada UI, then Refresh.
          </p>
        ) : (
          <ul className="flex flex-col gap-1.5">
            {models.map((m) => (
              <li
                key={m.id}
                className="flex items-center justify-between gap-2 rounded-md border border-white/10 bg-black/20 px-3 py-2"
              >
                <span className="flex min-w-0 items-center gap-2">
                  <span className="text-[10px] text-white/30">#{m.slot}</span>
                  <span className="min-w-0 truncate text-sm text-white/80" title={m.name}>{m.name}</span>
                </span>
                <span className="flex shrink-0 items-center gap-1.5">
                  <Badge color="bg-violet-700/50 text-violet-100">{m.type}{m.version ? ` ${m.version}` : ""}</Badge>
                  {m.f0 ? <Badge color="bg-sky-700/50 text-sky-100">f0</Badge> : null}
                  {m.has_index ? <Badge color="bg-emerald-700/55 text-emerald-100">index</Badge> : <Badge>no index</Badge>}
                  <span className="font-mono text-xs text-white/35">{size(m.size_bytes)}</span>
                </span>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}

function Row({ label, value, ok, mono = false }: { label: string; value: string; ok?: boolean; mono?: boolean }) {
  return (
    <div className="flex items-center justify-between gap-3">
      <span className="shrink-0 text-white/40">{label}</span>
      <span
        className={`min-w-0 truncate text-right ${mono ? "font-mono text-xs" : ""} ${
          ok === undefined ? "text-white/70" : ok ? "text-emerald-300/80" : "text-amber-300/70"
        }`}
        title={value}
      >
        {value}
      </span>
    </div>
  );
}
