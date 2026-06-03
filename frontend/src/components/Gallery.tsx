import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api/client";
import type { ImageItem } from "../types";

export function Gallery({ images, onSearch }: { images: ImageItem[]; onSearch: (q?: string) => void | Promise<void> }) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [searching, setSearching] = useState(false);
  const [lightbox, setLightbox] = useState(false);
  const [note, setNote] = useState("");
  const noteTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastLatest = useRef<string | null>(null);

  const selected = useMemo(
    () => images.find((img) => img.id === selectedId) ?? images[0] ?? null,
    [images, selectedId],
  );

  // Feature the freshest image as soon as it lands (a new generation), while
  // still letting the user click back through history until the next one.
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

  const submitSearch = async () => {
    setSearching(true);
    try {
      await onSearch(query);
    } finally {
      setSearching(false);
    }
  };

  const clearSearch = async () => {
    setQuery("");
    setSearching(true);
    try {
      await onSearch("");
    } finally {
      setSearching(false);
    }
  };

  const copyImage = async () => {
    if (!selected) return;
    try {
      const blob = await (await fetch(selected.url)).blob();
      await navigator.clipboard.write([new ClipboardItem({ [blob.type || "image/png"]: blob })]);
      flash("image copied to clipboard");
    } catch {
      flash("copy failed — browser blocked clipboard");
    }
  };

  const reveal = async () => {
    if (!selected) return;
    try {
      await api.revealImage(selected.id);
      flash("opened in file explorer");
    } catch {
      flash("could not open explorer");
    }
  };

  return (
    <div className="flex h-full flex-col gap-2">
      <div className="flex items-center justify-between gap-2 px-1">
        <h2 className="text-sm font-semibold text-white/70">Result</h2>
        <div className="grid grid-cols-[minmax(110px,1fr)_auto_auto] gap-1.5">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") void submitSearch();
            }}
            placeholder="search"
            className="w-full rounded-md border border-white/10 bg-black/30 px-2 py-1 text-xs outline-none focus:border-violet-500"
          />
          <button onClick={() => void submitSearch()} disabled={searching}
            className="rounded-md border border-white/15 px-2 py-1 text-xs hover:bg-white/10 disabled:opacity-30">
            Search
          </button>
          <button onClick={() => void clearSearch()} disabled={searching || !query}
            className="rounded-md border border-white/15 px-2 py-1 text-xs hover:bg-white/10 disabled:opacity-30">
            Clear
          </button>
        </div>
      </div>

      <div className="grid min-h-0 flex-1 grid-cols-[1fr_300px] gap-3">
        {/* --- featured --- */}
        <div className="flex min-h-0 min-w-0 flex-col gap-2">
          {selected ? (
            <>
              <button
                onClick={() => setLightbox(true)}
                className="group relative min-h-0 flex-1 overflow-hidden rounded-lg border border-white/10 bg-black/40"
                title="click to view full size"
              >
                <img
                  src={selected.url}
                  alt=""
                  className="h-full w-full object-contain"
                />
                <span className="absolute right-2 top-2 rounded bg-black/60 px-1.5 py-0.5 text-[10px] text-white/70 opacity-0 transition group-hover:opacity-100">
                  click to enlarge
                </span>
              </button>
              <div className="flex flex-wrap items-center gap-1.5">
                <button onClick={copyImage} className={actionBtn}>Copy</button>
                <button onClick={() => setLightbox(true)} className={actionBtn}>Detail</button>
                <button onClick={reveal} className={actionBtn}>Show in folder</button>
                <a href={selected.url} download={`${selected.id}.png`} className={actionBtn}>PNG</a>
                <a href={`/api/images/${selected.id}/metadata`} download className={actionBtn}>JSON</a>
                {note && <span className="ml-1 text-xs text-emerald-300/80">{note}</span>}
              </div>
            </>
          ) : (
            <div className="flex flex-1 items-center justify-center rounded-lg border border-white/10 text-sm text-white/30">
              no images yet
            </div>
          )}

          {/* --- history strip --- */}
          {images.length > 1 && (
            <div className="flex gap-2 overflow-x-auto pb-1">
              {images.map((img) => (
                <button
                  key={img.id}
                  onClick={() => setSelectedId(img.id)}
                  title={String(img.params?.prompt ?? "")}
                  className={`relative h-20 w-20 shrink-0 overflow-hidden rounded-md border ${
                    selected?.id === img.id ? "border-violet-400/80" : "border-white/10 hover:border-white/30"
                  }`}
                >
                  <img src={img.url} alt="" loading="lazy" className="h-full w-full object-cover" />
                </button>
              ))}
            </div>
          )}
        </div>

        <MetadataPanel image={selected} />
      </div>

      {lightbox && selected && (
        <div
          className="fixed inset-0 z-30 flex items-center justify-center bg-black/85 p-6"
          onClick={() => setLightbox(false)}
        >
          <img
            src={selected.url}
            alt=""
            className="max-h-full max-w-full object-contain"
            onClick={(e) => e.stopPropagation()}
          />
          <button
            onClick={() => setLightbox(false)}
            className="absolute right-5 top-5 rounded-md border border-white/20 bg-black/50 px-3 py-1 text-sm hover:bg-white/10"
          >
            Close
          </button>
        </div>
      )}
    </div>
  );
}

const actionBtn = "rounded border border-white/15 px-2.5 py-1 text-xs hover:bg-white/10";

function MetadataPanel({ image }: { image: ImageItem | null }) {
  if (!image) {
    return <aside className="min-h-0 border-l border-white/10 pl-3 text-sm text-white/30">metadata</aside>;
  }

  const params = image.params ?? {};
  const copyPrompt = () => {
    const p = text(params.prompt);
    if (p) navigator.clipboard?.writeText(p).catch(() => {});
  };

  return (
    <aside className="min-h-0 overflow-y-auto border-l border-white/10 pl-3 text-sm">
      <h3 className="mb-2 font-semibold text-white/70">Metadata</h3>
      <dl className="space-y-2">
        <Meta label="Model" value={text(params.model)} />
        <Meta label="Seed" value={text(image.seed)} />
        <Meta label="Size" value={image.width && image.height ? `${image.width}x${image.height}` : ""} />
        <Meta label="Steps" value={text(params.steps)} />
        <Meta label="Guidance" value={text(params.guidance)} />
        <Meta label="LoRA" value={loraSummary(params.loras)} />
        <Meta label="Created" value={new Date(image.created_at).toLocaleString()} />
      </dl>
      <div className="mt-3">
        <div className="flex items-center justify-between">
          <div className="text-xs uppercase tracking-wide text-white/35">Prompt</div>
          <button onClick={copyPrompt} className="text-[11px] text-white/40 hover:text-white/80">copy</button>
        </div>
        <p className="mt-1 whitespace-pre-wrap break-words text-xs leading-5 text-white/70">
          {text(params.prompt) || "-"}
        </p>
      </div>
      {text(params.negative) ? (
        <div className="mt-3">
          <div className="text-xs uppercase tracking-wide text-white/35">Negative</div>
          <p className="mt-1 whitespace-pre-wrap break-words text-xs leading-5 text-white/55">
            {text(params.negative)}
          </p>
        </div>
      ) : null}
    </aside>
  );
}

function Meta({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-xs uppercase tracking-wide text-white/35">{label}</dt>
      <dd className="mt-0.5 truncate text-white/70" title={value}>{value || "-"}</dd>
    </div>
  );
}

function text(value: unknown): string {
  if (value == null) return "";
  return String(value);
}

function loraSummary(value: unknown): string {
  if (!Array.isArray(value) || !value.length) return "";
  return value
    .map((item) => {
      if (!item || typeof item !== "object") return "";
      const name = "name" in item ? text(item.name) : "";
      const weight = "weight" in item ? text(item.weight) : "";
      return weight ? `${name || "LoRA"} @ ${weight}` : name;
    })
    .filter(Boolean)
    .join(", ");
}
