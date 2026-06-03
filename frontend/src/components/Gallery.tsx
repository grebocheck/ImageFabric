import { useEffect, useMemo, useState } from "react";
import type { ImageItem } from "../types";

export function Gallery({ images, onSearch }: { images: ImageItem[]; onSearch: (q?: string) => void | Promise<void> }) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [searching, setSearching] = useState(false);
  const selected = useMemo(
    () => images.find((img) => img.id === selectedId) ?? images[0] ?? null,
    [images, selectedId],
  );

  useEffect(() => {
    if (!selectedId && images[0]) setSelectedId(images[0].id);
    if (selectedId && !images.some((img) => img.id === selectedId)) {
      setSelectedId(images[0]?.id ?? null);
    }
  }, [images, selectedId]);

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

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between gap-2 px-1 pb-2">
        <h2 className="text-sm font-semibold text-white/70">History</h2>
        <div className="grid grid-cols-[minmax(120px,1fr)_auto_auto] gap-1.5">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") void submitSearch();
            }}
            placeholder="search"
            className="w-full rounded-md border border-white/10 bg-black/30 px-2 py-1 text-xs outline-none focus:border-violet-500"
          />
          <button
            onClick={() => void submitSearch()}
            disabled={searching}
            className="rounded-md border border-white/15 px-2 py-1 text-xs hover:bg-white/10 disabled:opacity-30"
          >
            Search
          </button>
          <button
            onClick={() => void clearSearch()}
            disabled={searching || !query}
            className="rounded-md border border-white/15 px-2 py-1 text-xs hover:bg-white/10 disabled:opacity-30"
          >
            Clear
          </button>
        </div>
      </div>
      <div className="grid min-h-0 flex-1 grid-cols-[minmax(220px,1fr)_280px] gap-3">
        <div className="overflow-y-auto pr-1">
          {images.length === 0 && <div className="px-1 text-sm text-white/30">no images yet</div>}
          <div className="grid grid-cols-2 gap-2">
            {images.map((img) => (
              <button
                key={img.id}
                onClick={() => setSelectedId(img.id)}
                className={`group relative overflow-hidden rounded-md border text-left ${
                  selected?.id === img.id ? "border-violet-400/70" : "border-white/10"
                }`}
                title={String(img.params?.prompt ?? "")}
              >
                <img
                  src={img.thumb_url ?? img.url}
                  alt=""
                  loading="lazy"
                  className="aspect-square w-full object-cover transition group-hover:scale-105"
                />
                <span className="absolute bottom-1 left-1 rounded bg-black/60 px-1 text-[10px] text-white/80">
                  seed {img.seed ?? "?"}
                </span>
              </button>
            ))}
          </div>
        </div>
        <MetadataPanel image={selected} />
      </div>
    </div>
  );
}

function MetadataPanel({ image }: { image: ImageItem | null }) {
  if (!image) {
    return (
      <aside className="min-h-0 border-l border-white/10 pl-3 text-sm text-white/30">
        metadata
      </aside>
    );
  }

  const params = image.params ?? {};
  return (
    <aside className="min-h-0 overflow-y-auto border-l border-white/10 pl-3 text-sm">
      <div className="mb-2 flex items-center justify-between gap-2">
        <h3 className="font-semibold text-white/70">Metadata</h3>
        <div className="flex gap-1.5">
          <a
            href={image.url}
            target="_blank"
            rel="noreferrer"
            className="rounded border border-white/15 px-2 py-1 text-xs hover:bg-white/10"
          >
            Open
          </a>
          <a
            href={image.url}
            download={`${image.id}.png`}
            className="rounded border border-white/15 px-2 py-1 text-xs hover:bg-white/10"
          >
            PNG
          </a>
          <a
            href={`/api/images/${image.id}/metadata`}
            download
            className="rounded border border-white/15 px-2 py-1 text-xs hover:bg-white/10"
          >
            JSON
          </a>
        </div>
      </div>
      <img
        src={image.thumb_url ?? image.url}
        alt=""
        className="mb-3 aspect-square w-full rounded-md border border-white/10 object-cover"
      />
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
        <div className="text-xs uppercase tracking-wide text-white/35">Prompt</div>
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
