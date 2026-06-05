import { useEffect, useState } from "react";
import { api } from "../api/client";
import { Select } from "./Select";
import type { TtsGenerateResult, TtsStatus } from "../types";

function size(bytes: number): string {
  if (!bytes) return "0 B";
  const gb = bytes / 1e9;
  return gb >= 1 ? `${gb.toFixed(2)} GB` : `${(bytes / 1e6).toFixed(1)} MB`;
}

export function TtsPanel() {
  const [status, setStatus] = useState<TtsStatus | null>(null);
  const [text, setText] = useState("Hello from HFabric.");
  const [modelId, setModelId] = useState("");
  const [vocoderId, setVocoderId] = useState("");
  const [useGuideTokens, setUseGuideTokens] = useState(false);
  const [result, setResult] = useState<TtsGenerateResult | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    api.ttsStatus().then((s) => {
      setStatus(s);
      setModelId((prev) => prev || s.models[0]?.id || "");
    }).catch(() => {});
  }, []);

  const models = status?.models ?? [];
  const ready = Boolean(status?.ready && modelId);
  const canGenerate = ready && Boolean(text.trim()) && !loading;

  async function onGenerate() {
    if (!canGenerate) return;
    setLoading(true);
    setError("");
    try {
      const next = await api.generateTts({
        model_id: modelId,
        text: text.trim(),
        vocoder_id: vocoderId || null,
        use_guide_tokens: useGuideTokens,
      });
      setResult(next);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex h-full gap-3">
      <aside className="flex w-80 shrink-0 flex-col gap-3 rounded-lg border border-white/10 p-4">
        <div>
          <h2 className="text-sm font-semibold text-white/75">TTS</h2>
          <div className="mt-1 text-xs text-white/35">
            {status?.binary_exists ? "llama-tts found" : "llama-tts missing"}
          </div>
        </div>

        <div className="space-y-1.5 rounded-md border border-white/10 bg-black/20 p-3 text-xs">
          <Row label="Binary" value={status?.binary ?? "..."} mono />
          <Row label="Models" value={status?.models_dir ?? "..."} mono />
          <Row label="Ready" value={ready ? "yes" : "waiting for model"} />
        </div>

        <label>
          <div className="text-xs uppercase tracking-wide text-white/40">Model</div>
          <Select
            value={modelId}
            onChange={setModelId}
            placeholder="no TTS models"
            className="mt-1"
            options={models.map((m) => ({ value: m.id, label: m.name, hint: size(m.size_bytes) }))}
          />
        </label>

        <label>
          <div className="text-xs uppercase tracking-wide text-white/40">Vocoder</div>
          <Select
            value={vocoderId}
            onChange={setVocoderId}
            placeholder="none"
            className="mt-1"
            options={[{ value: "", label: "none" }, ...models.map((m) => ({ value: m.id, label: m.name }))]}
          />
        </label>

        <label className="flex items-center gap-2 text-xs text-white/55">
          <input
            type="checkbox"
            checked={useGuideTokens}
            onChange={(e) => setUseGuideTokens(e.target.checked)}
            className="h-4 w-4 accent-emerald-500"
          />
          Guide tokens
        </label>
      </aside>

      <section className="flex min-w-0 flex-1 flex-col rounded-lg border border-white/10">
        <div className="border-b border-white/10 px-4 py-3">
          <div className="text-sm font-semibold text-white/75">Scratch text</div>
        </div>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          className="min-h-0 flex-1 resize-none bg-transparent p-4 text-sm leading-6 text-white/80 outline-none placeholder:text-white/25"
        />
        {result && (
          <div className="border-t border-white/10 p-3">
            <div className="mb-2 flex items-center justify-between text-xs text-white/45">
              <span>{result.duration_seconds.toFixed(1)}s</span>
              <a href={result.url} download className="text-emerald-300 hover:text-emerald-200">
                Download WAV
              </a>
            </div>
            <audio controls src={result.url} className="w-full" />
          </div>
        )}
        <div className="flex items-center justify-between border-t border-white/10 p-3">
          <span
            className={`min-w-0 truncate text-xs ${error ? "text-red-300" : "text-white/35"}`}
            title={error || undefined}
          >
            {error || (ready ? "ready" : "waiting for local model")}
          </span>
          <button
            onClick={onGenerate}
            disabled={!canGenerate}
            className="rounded-md bg-emerald-600 px-4 py-1.5 text-sm font-medium hover:bg-emerald-500 disabled:opacity-30 disabled:hover:bg-emerald-600"
            title={ready ? "Generate WAV" : "Waiting for local TTS model"}
          >
            {loading ? "Generating..." : "Generate"}
          </button>
        </div>
      </section>
    </div>
  );
}

function Row({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="grid grid-cols-[70px_1fr] gap-2">
      <span className="text-white/35">{label}</span>
      <span className={`truncate text-white/65 ${mono ? "font-mono" : ""}`} title={value}>{value}</span>
    </div>
  );
}
