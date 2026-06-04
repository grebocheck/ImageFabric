import { useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import type { VisionResult, VisionStatus } from "../types";

const field = "w-full rounded-md bg-black/30 border border-white/10 px-2.5 py-1.5 text-sm outline-none focus:border-emerald-500";

function size(bytes: number): string {
  if (!bytes) return "0 B";
  const gb = bytes / 1e9;
  return gb >= 1 ? `${gb.toFixed(2)} GB` : `${(bytes / 1e6).toFixed(1)} MB`;
}

export function VisionPanel() {
  const [status, setStatus] = useState<VisionStatus | null>(null);
  const [modelId, setModelId] = useState("");
  const [projectorId, setProjectorId] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [prompt, setPrompt] = useState("Describe the image.");
  const [result, setResult] = useState<VisionResult | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    api.visionStatus().then((s) => {
      setStatus(s);
      setModelId((prev) => prev || s.models[0]?.id || "");
      setProjectorId((prev) => prev || s.projectors[0]?.id || "");
    }).catch(() => {});
  }, []);

  const preview = useMemo(() => file ? URL.createObjectURL(file) : "", [file]);
  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);

  const ready = Boolean(status?.ready && modelId && projectorId);
  const canAnalyze = ready && Boolean(file) && Boolean(prompt.trim()) && !loading;

  async function analyze() {
    if (!canAnalyze || !file) return;
    setLoading(true);
    setError("");
    try {
      const next = await api.analyzeVision({
        file,
        prompt: prompt.trim(),
        model_id: modelId,
        projector_id: projectorId,
      });
      setResult(next);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="grid h-full grid-cols-[320px_1fr] gap-3">
      <aside className="flex min-h-0 flex-col gap-3 rounded-lg border border-white/10 p-4">
        <div>
          <h2 className="text-sm font-semibold text-white/75">Vision</h2>
          <div className="mt-1 text-xs text-white/35">
            {status?.ready ? "local multimodal ready" : "waiting for local model"}
          </div>
        </div>

        <div className="space-y-1.5 rounded-md border border-white/10 bg-black/20 p-3 text-xs">
          <Row label="Binary" value={status?.binary_exists ? "found" : "missing"} />
          <Row label="Models" value={status?.models_dir ?? "..."} mono />
          <Row label="GPU" value={status ? `${status.gpu_layers} layers` : "..."} />
          <Row label="Limit" value={status ? `${status.max_upload_mb} MB` : "..."} />
        </div>

        <label>
          <div className="text-xs uppercase tracking-wide text-white/40">Model</div>
          <select value={modelId} onChange={(e) => setModelId(e.target.value)} className={`${field} mt-1`}>
            {(status?.models ?? []).length === 0 && <option value="">no vision models</option>}
            {(status?.models ?? []).map((m) => <option key={m.id} value={m.id}>{m.name} ({size(m.size_bytes)})</option>)}
          </select>
        </label>

        <label>
          <div className="text-xs uppercase tracking-wide text-white/40">Projector</div>
          <select value={projectorId} onChange={(e) => setProjectorId(e.target.value)} className={`${field} mt-1`}>
            {(status?.projectors ?? []).length === 0 && <option value="">no mmproj models</option>}
            {(status?.projectors ?? []).map((m) => <option key={m.id} value={m.id}>{m.name} ({size(m.size_bytes)})</option>)}
          </select>
        </label>

        <label>
          <div className="text-xs uppercase tracking-wide text-white/40">Image</div>
          <input
            type="file"
            accept=".png,.jpg,.jpeg,image/png,image/jpeg"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            className={`${field} mt-1 file:mr-3 file:rounded file:border-0 file:bg-white/10 file:px-2 file:py-1 file:text-xs file:text-white/70`}
          />
        </label>

        <label className="flex min-h-0 flex-1 flex-col">
          <div className="text-xs uppercase tracking-wide text-white/40">Prompt</div>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            className={`${field} mt-1 min-h-0 flex-1 resize-none`}
          />
        </label>

        <div className="flex items-center justify-between gap-3">
          <span className={`min-w-0 truncate text-xs ${error ? "text-red-300" : "text-white/35"}`} title={error || undefined}>
            {error || (ready ? "ready" : "waiting")}
          </span>
          <button
            onClick={() => void analyze()}
            disabled={!canAnalyze}
            className="shrink-0 rounded-md bg-emerald-600 px-4 py-1.5 text-sm font-medium hover:bg-emerald-500 disabled:opacity-30"
          >
            {loading ? "Analyzing..." : "Analyze"}
          </button>
        </div>
      </aside>

      <section className="grid min-h-0 grid-cols-[minmax(280px,0.9fr)_1fr] gap-3">
        <div className="flex min-h-0 flex-col rounded-lg border border-white/10">
          <div className="border-b border-white/10 px-4 py-3 text-sm font-semibold text-white/75">Image</div>
          <div className="flex min-h-0 flex-1 items-center justify-center overflow-hidden bg-black/20 p-3">
            {preview ? (
              <img src={preview} alt="" className="max-h-full max-w-full object-contain" />
            ) : (
              <div className="text-sm text-white/25">No image selected</div>
            )}
          </div>
        </div>

        <div className="flex min-h-0 flex-col rounded-lg border border-white/10">
          <div className="flex items-center justify-between border-b border-white/10 px-4 py-3">
            <span className="text-sm font-semibold text-white/75">Result</span>
            {result && (
              <a href={result.metadata_url} download className="text-xs text-emerald-300 hover:text-emerald-200">
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
            <div className="border-t border-white/10 px-4 py-2 text-xs text-white/35">
              {result.duration_seconds.toFixed(1)}s
            </div>
          )}
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
