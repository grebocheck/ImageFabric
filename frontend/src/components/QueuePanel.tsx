import { useMemo, useState } from "react";
import { api } from "../api/client";
import { Badge } from "./Badge";
import type { Job } from "../types";

const statusColor: Record<string, string> = {
  queued: "text-white/55",
  running: "text-violet-300",
  done: "text-emerald-400",
  error: "text-red-400",
  cancelled: "text-white/30",
};

const statusBorder: Record<string, string> = {
  queued: "border-l-white/25",
  running: "border-l-violet-400",
  done: "border-l-emerald-400",
  error: "border-l-red-400",
  cancelled: "border-l-white/15",
};

const order: Record<string, number> = { running: 0, queued: 1, error: 2, done: 3, cancelled: 4 };
const previewCells = Array.from({ length: 16 });
const cellColors = ["bg-white/45", "bg-violet-300/45", "bg-cyan-300/35", "bg-fuchsia-300/35"];

export function QueuePanel({ jobs, onChanged }: { jobs: Job[]; onChanged: () => void }) {
  const [draggedId, setDraggedId] = useState<string | null>(null);
  const sorted = useMemo(
    () => [...jobs].sort(
      (a, b) =>
        (order[a.status] - order[b.status])
        || (b.priority - a.priority)
        || Date.parse(a.created_at) - Date.parse(b.created_at),
    ),
    [jobs],
  );
  const running = jobs.filter((job) => job.status === "running").length;
  const queued = jobs.filter((job) => job.status === "queued").length;
  const finished = jobs.filter((job) => job.status === "done" || job.status === "cancelled").length;

  const reorderQueued = async (targetId: string) => {
    if (!draggedId || draggedId === targetId) return;
    const queuedJobs = sorted.filter((job) => job.status === "queued");
    const from = queuedJobs.findIndex((job) => job.id === draggedId);
    const to = queuedJobs.findIndex((job) => job.id === targetId);
    if (from < 0 || to < 0) return;

    const next = [...queuedJobs];
    const [moved] = next.splice(from, 1);
    next.splice(to, 0, moved);
    await Promise.all(next.map((job, i) => api.setPriority(job.id, next.length - i)));
    setDraggedId(null);
    onChanged();
  };

  return (
    <section className="flex h-full min-h-0 flex-col overflow-hidden rounded-lg border border-white/10 bg-surface max-[1240px]:col-span-2 max-[860px]:h-[520px]">
      <div className="border-b border-white/10 px-3 py-3">
        <div className="flex items-start justify-between gap-3">
          <div>
            <h2 className="text-sm font-semibold text-white/85">Queue</h2>
            <div className="mt-1 flex gap-2 text-[11px] text-white/40">
              <span>{running} running</span>
              <span>{queued} queued</span>
              <span>{finished} finished</span>
            </div>
          </div>
          <button
            onClick={() => api.clearFinished().then(onChanged)}
            disabled={!finished && !jobs.some((job) => job.status === "error")}
            className="rounded-md border border-white/15 px-2.5 py-1.5 text-xs text-white/55 transition hover:bg-white/10 hover:text-white disabled:opacity-30"
          >
            Clear
          </button>
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto p-2">
        {sorted.length === 0 ? (
          <div className="flex h-full items-center justify-center rounded-md border border-dashed border-white/10 text-sm text-white/30">
            Empty queue
          </div>
        ) : (
          <div className="space-y-2">
            {sorted.map((job) => (
              <JobCard
                key={job.id}
                job={job}
                draggedId={draggedId}
                setDraggedId={setDraggedId}
                reorderQueued={reorderQueued}
                onChanged={onChanged}
              />
            ))}
          </div>
        )}
      </div>
    </section>
  );
}

function JobCard({
  job,
  draggedId,
  setDraggedId,
  reorderQueued,
  onChanged,
}: {
  job: Job;
  draggedId: string | null;
  setDraggedId: (id: string | null) => void;
  reorderQueued: (targetId: string) => Promise<void>;
  onChanged: () => void;
}) {
  const progress = Math.round((job.progress || 0) * 100);
  const prompt = String(job.params?.prompt ?? "");

  return (
    <article
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
      className={`animate-fade-in rounded-md border border-l-2 bg-black/20 p-2.5 text-sm transition ${
        draggedId === job.id ? "border-violet-400/60 opacity-60" : `border-white/10 ${statusBorder[job.status]}`
      } ${job.status === "queued" ? "cursor-grab active:cursor-grabbing" : ""}`}
    >
      <div className="flex min-w-0 items-center justify-between gap-2">
        <div className="flex min-w-0 items-center gap-2">
          <Badge color="bg-violet-700/80 text-white">{job.type}</Badge>
          <span className="min-w-0 truncate font-mono text-xs text-white/55" title={job.model_id}>{job.model_id}</span>
        </div>
        <span className={`shrink-0 text-xs ${statusColor[job.status]}`}>{job.status}</span>
      </div>

      {prompt ? (
        <div className="mt-1.5 line-clamp-2 text-xs leading-4 text-white/45" title={prompt}>{prompt}</div>
      ) : null}

      {job.status === "running" ? (
        <>
          <div className="mt-2 flex items-center gap-2">
            <div className="h-1.5 flex-1 overflow-hidden rounded bg-white/10">
              <div
                className="h-full bg-violet-500 transition-all"
                style={{ width: `${progress}%` }}
              />
            </div>
            <span className="w-8 text-right text-[11px] text-white/45">{progress}%</span>
          </div>
          {job.type === "image" ? (
            <DenoisePreview progress={job.progress} note={job.progress_note} />
          ) : null}
          <div className="mt-2 flex justify-end">
            <button
              onClick={() => api.cancelJob(job.id).then(onChanged).catch(() => {})}
              className="rounded-md border border-red-400/30 px-2.5 py-1 text-xs text-red-300 hover:bg-red-400/10"
            >
              Stop
            </button>
          </div>
        </>
      ) : null}

      {job.status === "error" ? (
        <div className="mt-1.5 line-clamp-2 text-xs text-red-400/85" title={job.error ?? ""}>{job.error}</div>
      ) : null}

      {job.status === "queued" ? (
        <div className="mt-2 flex items-center justify-between gap-2">
          <button
            onClick={() => api.setPriority(job.id, job.priority + 1).then(onChanged)}
            className="text-xs text-white/45 hover:text-white/90"
          >
            Priority {job.priority}
          </button>
          <button
            onClick={() => api.cancelJob(job.id).then(onChanged)}
            className="text-xs text-red-400/75 hover:text-red-300"
          >
            Cancel
          </button>
        </div>
      ) : null}
    </article>
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
        <div className="mt-1 text-[10px] uppercase tracking-wide text-white/35">preview</div>
      </div>
    </div>
  );
}
