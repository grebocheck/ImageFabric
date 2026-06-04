import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api/client";
import type { ImageItem } from "../types";

const actionBtn = "rounded-md border border-white/15 px-2.5 py-1.5 text-xs text-white/70 transition hover:bg-white/10 hover:text-white";

export function ResultPreview({ images, onOpenHistory, generating = false }: { images: ImageItem[]; onOpenHistory: () => void; generating?: boolean }) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [lightbox, setLightbox] = useState(false);
  const [note, setNote] = useState("");
  const noteTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastLatest = useRef<string | null>(null);

  const selected = useMemo(
    () => images.find((img) => img.id === selectedId) ?? images[0] ?? null,
    [images, selectedId],
  );

  useEffect(() => {
    const latest = images[0]?.id ?? null;
    if (latest && latest !== lastLatest.current) {
      lastLatest.current = latest;
      setSelectedId(latest);
    }
    if (selectedId && !images.some((img) => img.id === selectedId)) {
      setSelectedId(images[0]?.id ?? null);
    }
  }, [images, selectedId]);

  const flash = useCallback((msg: string) => {
    setNote(msg);
    if (noteTimer.current) clearTimeout(noteTimer.current);
    noteTimer.current = setTimeout(() => setNote(""), 2200);
  }, []);

  const copyImage = async () => {
    if (!selected) return;
    try {
      const blob = await (await fetch(selected.url)).blob();
      await navigator.clipboard.write([new ClipboardItem({ [blob.type || "image/png"]: blob })]);
      flash("copied");
    } catch {
      flash("copy blocked");
    }
  };

  const reveal = async () => {
    if (!selected) return;
    try {
      await api.revealImage(selected.id);
      flash("opened folder");
    } catch {
      flash("could not open");
    }
  };

  const params = selected?.params ?? {};
  const facts = selected
    ? [
        selected.width && selected.height ? `${selected.width}x${selected.height}` : "",
        text(params.steps) ? `${text(params.steps)} steps` : "",
        text(params.guidance) ? `cfg ${text(params.guidance)}` : "",
        selected.seed == null || selected.seed === -1 ? "random seed" : `seed ${selected.seed}`,
      ].filter(Boolean)
    : [];

  return (
    <section className="flex h-full min-h-0 min-w-0 flex-col overflow-hidden rounded-lg border border-white/10 bg-surface max-[860px]:mb-4 max-[860px]:h-[720px]">
      <div className="flex items-center justify-between gap-3 border-b border-white/10 px-3 py-3">
        <div className="min-w-0">
          <h2 className="text-sm font-semibold text-white/85">Result</h2>
          <p className="mt-0.5 truncate text-xs text-white/40">
            {selected ? text(params.prompt) || "Generated image" : "No image yet"}
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          {note ? <span className="text-xs text-emerald-300/85">{note}</span> : null}
          <button onClick={onOpenHistory} className={actionBtn}>History</button>
        </div>
      </div>

      <div className="relative flex min-h-0 flex-1 items-center justify-center overflow-hidden bg-black/35">
        {generating ? <div className="skeleton absolute inset-x-0 top-0 z-10 h-0.5" /> : null}
        {selected ? (
          <button
            onClick={() => setLightbox(true)}
            className="group flex h-full w-full items-center justify-center p-4"
            title="Open detail view"
          >
            <img src={selected.url} alt="" className="max-h-full max-w-full object-contain shadow-2xl shadow-black/50" />
            <span className="absolute right-3 top-3 rounded-md border border-white/10 bg-black/60 px-2 py-1 text-[11px] text-white/65 opacity-0 transition group-hover:opacity-100">
              Detail
            </span>
          </button>
        ) : generating ? (
          <div className="flex h-full w-full flex-col items-center justify-center gap-3 p-8">
            <div className="skeleton h-40 w-40 rounded-lg" />
            <span className="text-xs text-white/40">generating…</span>
          </div>
        ) : (
          <div className="flex h-full w-full items-center justify-center p-8 text-sm text-white/30">
            Queue a generation to see the result here.
          </div>
        )}
      </div>

      <div className="border-t border-white/10 bg-black/20 p-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex flex-wrap items-center gap-1.5">
            <button onClick={copyImage} disabled={!selected} className={`${actionBtn} disabled:opacity-30`}>Copy</button>
            <button onClick={() => setLightbox(true)} disabled={!selected} className={`${actionBtn} disabled:opacity-30`}>Detail</button>
            <button onClick={reveal} disabled={!selected} className={`${actionBtn} disabled:opacity-30`}>Folder</button>
            {selected ? (
              <>
                <a href={selected.url} download={`${selected.id}.png`} className={actionBtn}>PNG</a>
                <a href={`/api/images/${selected.id}/metadata`} download className={actionBtn}>JSON</a>
              </>
            ) : null}
          </div>
          <div className="min-w-0 truncate text-xs text-white/40">
            {facts.length ? facts.join(" / ") : "Waiting for output"}
          </div>
        </div>

        {images.length > 1 ? (
          <div className="mt-3 flex h-16 gap-2 overflow-x-auto pb-1">
            {images.slice(0, 18).map((img) => (
              <button
                key={img.id}
                onClick={() => setSelectedId(img.id)}
                title={text(img.params?.prompt)}
                className={`relative h-14 w-14 shrink-0 overflow-hidden rounded-md border transition ${
                  selected?.id === img.id ? "border-violet-400/90" : "border-white/10 hover:border-white/35"
                }`}
              >
                <img src={img.thumb_url ?? img.url} alt="" loading="lazy" className="h-full w-full object-cover" />
              </button>
            ))}
          </div>
        ) : null}
      </div>

      {lightbox && selected && (
        <div
          className="fixed inset-0 z-30 bg-black/90"
          onClick={() => setLightbox(false)}
        >
          <img
            src={selected.url}
            alt=""
            className="absolute left-1/2 top-1/2 max-h-[92vh] max-w-[92vw] -translate-x-1/2 -translate-y-1/2 object-contain"
            onClick={(e) => e.stopPropagation()}
          />
          <button
            onClick={() => setLightbox(false)}
            className="absolute right-5 top-5 rounded-md border border-white/20 bg-black/60 px-3 py-1.5 text-sm hover:bg-white/10"
          >
            Close
          </button>
        </div>
      )}
    </section>
  );
}

function text(value: unknown): string {
  if (value == null) return "";
  return String(value);
}
