import { useEffect, useState } from "react";
import { api } from "../api/client";
import { Select } from "./Select";
import type { TranscriptionResult, TranscriptionStatus } from "../types";

const field = "w-full rounded-md bg-black/30 border border-white/10 px-2.5 py-1.5 text-sm outline-none focus:border-emerald-500";

function size(bytes: number): string {
  if (!bytes) return "0 B";
  const gb = bytes / 1e9;
  return gb >= 1 ? `${gb.toFixed(2)} GB` : `${(bytes / 1e6).toFixed(1)} MB`;
}

function stamp(seconds: number): string {
  const m = Math.floor(seconds / 60).toString().padStart(2, "0");
  const s = Math.floor(seconds % 60).toString().padStart(2, "0");
  return `${m}:${s}`;
}

export function TranscriptionPanel() {
  const [status, setStatus] = useState<TranscriptionStatus | null>(null);
  const [modelId, setModelId] = useState("");
  const [language, setLanguage] = useState("");
  const [task, setTask] = useState("transcribe");
  const [initialPrompt, setInitialPrompt] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<TranscriptionResult | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    api.transcriptionStatus().then((s) => {
      setStatus(s);
      setModelId((prev) => prev || s.models[0]?.id || "");
    }).catch(() => {});
  }, []);

  const models = status?.models ?? [];
  const selected = models.find((m) => m.id === modelId);
  const engineReady = selected ? Boolean(status?.engines[selected.engine]) : false;
  const ready = Boolean(status?.ready && modelId && engineReady);
  const canTranscribe = ready && Boolean(file) && !loading;

  async function onTranscribe() {
    if (!canTranscribe || !file) return;
    setLoading(true);
    setError("");
    try {
      const next = await api.transcribeAudio({
        file,
        model_id: modelId,
        language,
        task,
        initial_prompt: initialPrompt,
      });
      setResult(next);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  const statusText = error
    || (ready ? "ready" : selected && !engineReady ? `${selected.engine} missing` : "waiting for local model");

  return (
    <div className="flex h-full gap-3">
      <aside className="flex w-80 shrink-0 flex-col gap-3 rounded-lg border border-white/10 p-4">
        <div>
          <h2 className="text-sm font-semibold text-white/75">Transcribe</h2>
          <div className="mt-1 text-xs text-white/35">
            {status?.ready ? "local Whisper ready" : "local Whisper waiting"}
          </div>
        </div>

        <div className="space-y-1.5 rounded-md border border-white/10 bg-black/20 p-3 text-xs">
          <Row label="Models" value={status?.models_dir ?? "..."} mono />
          <Row label="Device" value={status ? `${status.device} / ${status.compute_type}` : "..."} />
          <Row label="Engines" value={engineSummary(status)} />
          <Row label="Limit" value={status ? `${status.max_upload_mb} MB` : "..."} />
        </div>

        <label>
          <div className="text-xs uppercase tracking-wide text-white/40">Model</div>
          <Select
            value={modelId}
            onChange={setModelId}
            placeholder="no transcription models"
            className="mt-1"
            options={models.map((m) => ({ value: m.id, label: m.name, hint: `${m.engine}, ${size(m.size_bytes)}` }))}
          />
        </label>

        <label>
          <div className="text-xs uppercase tracking-wide text-white/40">Audio</div>
          <input
            type="file"
            accept="audio/*,video/mp4,video/webm"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            className={`${field} mt-1 file:mr-3 file:rounded file:border-0 file:bg-white/10 file:px-2 file:py-1 file:text-xs file:text-white/70`}
          />
        </label>

        <div className="grid grid-cols-2 gap-2">
          <label>
            <div className="text-xs uppercase tracking-wide text-white/40">Task</div>
            <Select
              value={task}
              onChange={setTask}
              className="mt-1"
              options={[
                { value: "transcribe", label: "transcribe" },
                { value: "translate", label: "translate" },
              ]}
            />
          </label>
          <label>
            <div className="text-xs uppercase tracking-wide text-white/40">Language</div>
            <input
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              placeholder="auto"
              className={`${field} mt-1`}
            />
          </label>
        </div>

        <label className="flex min-h-0 flex-1 flex-col">
          <div className="text-xs uppercase tracking-wide text-white/40">Prompt</div>
          <textarea
            value={initialPrompt}
            onChange={(e) => setInitialPrompt(e.target.value)}
            className={`${field} mt-1 min-h-0 flex-1 resize-none`}
          />
        </label>
      </aside>

      <section className="flex min-w-0 flex-1 flex-col rounded-lg border border-white/10">
        <div className="flex items-center justify-between gap-3 border-b border-white/10 px-4 py-3">
          <div className="min-w-0">
            <div className="text-sm font-semibold text-white/75">Transcript</div>
            <div className="truncate text-xs text-white/35" title={file?.name}>
              {file?.name || "No audio selected"}
            </div>
          </div>
          {result && (
            <a href={result.metadata_url} download className="shrink-0 text-xs text-emerald-300 hover:text-emerald-200">
              Download JSON
            </a>
          )}
        </div>

        <textarea
          readOnly
          value={result?.text ?? ""}
          className="min-h-0 flex-1 resize-none bg-transparent p-4 text-sm leading-6 text-white/80 outline-none"
        />

        {result && (
          <div className="max-h-52 overflow-y-auto border-t border-white/10 p-3">
            <div className="mb-2 flex items-center gap-3 text-xs text-white/40">
              <span>{result.duration_seconds.toFixed(1)}s</span>
              {result.detected_language && <span>{result.detected_language}</span>}
              {typeof result.language_probability === "number" && (
                <span>{Math.round(result.language_probability * 100)}%</span>
              )}
            </div>
            <div className="space-y-1">
              {result.segments.map((segment, idx) => (
                <div key={`${segment.start}-${idx}`} className="grid grid-cols-[90px_1fr] gap-3 rounded-md px-2 py-1.5 text-xs hover:bg-white/5">
                  <span className="font-mono text-white/35">{stamp(segment.start)}-{stamp(segment.end)}</span>
                  <span className="text-white/65">{segment.text}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="flex items-center justify-between border-t border-white/10 p-3">
          <span
            className={`min-w-0 truncate text-xs ${error ? "text-red-300" : "text-white/35"}`}
            title={statusText}
          >
            {statusText}
          </span>
          <button
            onClick={() => void onTranscribe()}
            disabled={!canTranscribe}
            className="rounded-md bg-emerald-600 px-4 py-1.5 text-sm font-medium hover:bg-emerald-500 disabled:opacity-30 disabled:hover:bg-emerald-600"
          >
            {loading ? "Transcribing..." : "Transcribe"}
          </button>
        </div>
      </section>
    </div>
  );
}

function engineSummary(status: TranscriptionStatus | null): string {
  if (!status) return "...";
  return Object.entries(status.engines)
    .map(([name, ok]) => `${name}:${ok ? "yes" : "no"}`)
    .join(", ");
}

function Row({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="grid grid-cols-[70px_1fr] gap-2">
      <span className="text-white/35">{label}</span>
      <span className={`truncate text-white/65 ${mono ? "font-mono" : ""}`} title={value}>{value}</span>
    </div>
  );
}
