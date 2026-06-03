import { useState } from "react";
import { api } from "../api/client";
import type { Job } from "../types";

const statusColor: Record<string, string> = {
  queued: "text-white/50",
  running: "text-violet-300",
  done: "text-emerald-400",
  error: "text-red-400",
  cancelled: "text-white/30",
};

const order: Record<string, number> = { running: 0, queued: 1, error: 2, done: 3, cancelled: 4 };
const previewCells = Array.from({ length: 16 });
const cellColors = ["bg-white/45", "bg-violet-300/45", "bg-cyan-300/35", "bg-fuchsia-300/35"];

export function QueuePanel({ jobs, onChanged }: { jobs: Job[]; onChanged: () => void }) {
  const [draggedId, setDraggedId] = useState<string | null>(null);
  const sorted = [...jobs].sort(
    (a, b) =>
      (order[a.status] - order[b.status])
      || (b.priority - a.priority)
      || Date.parse(a.created_at) - Date.parse(b.created_at),
  );

  const reorderQueued = async (targetId: string) => {
    if (!draggedId || draggedId === targetId) return;
    const queued = sorted.filter((job) => job.status === "queued");
    const from = queued.findIndex((job) => job.id === draggedId);
    const to = queued.findIndex((job) => job.id === targetId);
    if (from < 0 || to < 0) return;

    const next = [...queued];
    const [moved] = next.splice(from, 1);
    next.splice(to, 0, moved);
    await Promise.all(next.map((job, i) => api.setPriority(job.id, next.length - i)));
    setDraggedId(null);
    onChanged();
  };

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between px-1 pb-2">
        <h2 className="text-sm font-semibold text-white/70">Queue</h2>
        <button
          onClick={() => api.clearFinished().then(onChanged)}
          className="text-xs text-white/40 hover:text-white/80"
        >
          clear finished
        </button>
      </div>
      <div className="flex-1 space-y-2 overflow-y-auto pr-1">
        {sorted.length === 0 && <div className="px-1 text-sm text-white/30">empty</div>}
        {sorted.map((job) => (
          <div
            key={job.id}
            draggable={job.status === "queued"}
            onDragStart={(e) => {
              if (job.status !== "queued") return;
              setDraggedId(job.id);
              e.dataTransfer.effectAllowed = "move";
            }}
            onDragEnd={() => setDraggedId(null)}
            onDragOver={(e) => {
              if (job.status === "queued" && draggedId) e.preventDefault();
            }}
            onDrop={(e) => {
              e.preventDefault();
              void reorderQueued(job.id);
            }}
            className={`rounded-md border bg-black/20 p-2 text-sm ${
              draggedId === job.id
                ? "border-violet-400/60 opacity-60"
                : "border-white/10"
            } ${job.status === "queued" ? "cursor-grab active:cursor-grabbing" : ""}`}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span
                  className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${
                    job.type === "llm" ? "bg-emerald-700" : "bg-violet-700"
                  }`}
                >
                  {job.type}
                </span>
                <span className="truncate font-mono text-xs text-white/50">{job.model_id}</span>
              </div>
              <span className={`text-xs ${statusColor[job.status]}`}>{job.status}</span>
            </div>

            {job.params?.prompt != null && (
              <div className="mt-1 truncate text-xs text-white/40">{String(job.params.prompt)}</div>
            )}

            {job.status === "running" && (
              <>
                <div className="mt-1.5 h-1 overflow-hidden rounded bg-white/10">
                  <div
                    className="h-full bg-violet-500 transition-all"
                    style={{ width: `${Math.round(job.progress * 100)}%` }}
                  />
                </div>
                {job.type === "image" && (
                  <DenoisePreview progress={job.progress} note={job.progress_note} />
                )}
              </>
            )}

            {job.status === "error" && (
              <div className="mt-1 truncate text-xs text-red-400/80" title={job.error ?? ""}>{job.error}</div>
            )}

            {job.status === "queued" && (
              <div className="mt-1.5 flex gap-2">
                <button
                  onClick={() => api.setPriority(job.id, job.priority + 1).then(onChanged)}
                  className="text-xs text-white/40 hover:text-white/90"
                >
                  ↑ priority ({job.priority})
                </button>
                <button
                  onClick={() => api.cancelJob(job.id).then(onChanged)}
                  className="text-xs text-red-400/70 hover:text-red-300"
                >
                  cancel
                </button>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function DenoisePreview({ progress, note }: { progress: number; note?: string | null }) {
  const clamped = Math.min(1, Math.max(0, progress || 0));
  const noiseCells = Math.ceil((1 - clamped) * previewCells.length);
  const pct = Math.round(clamped * 100);

  return (
    <div className="mt-2 flex min-h-16 gap-2 rounded-md border border-white/10 bg-black/20 p-2">
      <div className="relative grid h-14 w-14 shrink-0 grid-cols-4 gap-px overflow-hidden rounded border border-white/10 bg-black/40 p-1">
        {previewCells.map((_, i) => (
          <span
            key={i}
            className={`${cellColors[i % cellColors.length]} rounded-[1px] transition-opacity`}
            style={{ opacity: i < noiseCells ? 0.85 : 0.12 }}
          />
        ))}
        <span className="absolute inset-0 flex items-center justify-center text-[10px] font-semibold text-white drop-shadow">
          {pct}%
        </span>
      </div>
      <div className="min-w-0 flex-1 self-center">
        <div className="truncate text-xs text-white/70">{note ?? "denoising"}</div>
        <div className="mt-1 text-[10px] uppercase tracking-wide text-white/35">denoise</div>
      </div>
    </div>
  );
}
