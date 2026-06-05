import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import { Select, type SelectOption } from "./Select";
import { toast } from "./Toast";
import type { ImageItem, ImageStats, Model } from "../types";

const PAGE = 60;

const DATE_RANGES = [
  { value: "all", label: "All time" },
  { value: "today", label: "Today" },
  { value: "7d", label: "Last 7 days" },
  { value: "30d", label: "Last 30 days" },
];

const SIZE_FILTERS = [
  { value: "", label: "All sizes" },
  { value: "square", label: "Square" },
  { value: "landscape", label: "Landscape" },
  { value: "portrait", label: "Portrait" },
  { value: "large", label: "1024+ px" },
  { value: "small", label: "Under 1024" },
];

function rangeStart(id: string): string | undefined {
  const now = Date.now();
  if (id === "today") {
    const d = new Date();
    d.setHours(0, 0, 0, 0);
    return d.toISOString();
  }
  if (id === "7d") return new Date(now - 7 * 86_400_000).toISOString();
  if (id === "30d") return new Date(now - 30 * 86_400_000).toISOString();
  return undefined;
}

export function Gallery({
  models,
  reloadSignal,
  onReproduce,
}: {
  models: Model[];
  reloadSignal: number;
  onReproduce: (image: ImageItem, opts: { keepSeed: boolean }) => void;
}) {
  // `applied` is what actually drives fetching; `query` is the live input box.
  const [query, setQuery] = useState("");
  const [applied, setApplied] = useState({ q: "", model: "", size: "", lora: "", range: "all" });
  const [items, setItems] = useState<ImageItem[]>([]);
  const [stats, setStats] = useState<ImageStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [hasMore, setHasMore] = useState(false);
  const [openId, setOpenId] = useState<string | null>(null);
  const [selectMode, setSelectMode] = useState(false);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [exporting, setExporting] = useState(false);

  const refreshStats = useCallback(() => {
    api.imageStats().then(setStats).catch(() => {});
  }, []);

  const fetchPage = useCallback(
    (offset: number) =>
      api.queryImages({
        q: applied.q || undefined,
        model: applied.model || undefined,
        size: applied.size || undefined,
        lora: applied.lora || undefined,
        date_from: rangeStart(applied.range),
        limit: PAGE,
        offset,
      }),
    [applied],
  );

  const reload = useCallback(async () => {
    setLoading(true);
    try {
      const rows = await fetchPage(0);
      setItems(rows);
      setHasMore(rows.length === PAGE);
    } catch {
      setItems([]);
      setHasMore(false);
    } finally {
      setLoading(false);
    }
  }, [fetchPage]);

  useEffect(() => {
    void reload();
  }, [reload, reloadSignal]);

  useEffect(() => {
    refreshStats();
  }, [refreshStats, reloadSignal]);

  const loadMore = async () => {
    setLoading(true);
    try {
      const rows = await fetchPage(items.length);
      setItems((prev) => [...prev, ...rows]);
      setHasMore(rows.length === PAGE);
    } finally {
      setLoading(false);
    }
  };

  const open = useMemo(() => items.find((i) => i.id === openId) ?? null, [items, openId]);

  const removeOne = useCallback(
    async (id: string) => {
      try {
        await api.deleteImage(id);
        setItems((prev) => prev.filter((i) => i.id !== id));
        setSelected((prev) => {
          const next = new Set(prev);
          next.delete(id);
          return next;
        });
        if (openId === id) setOpenId(null);
        refreshStats();
      } catch {
        toast.error("Could not delete image");
      }
    },
    [openId, refreshStats],
  );

  const removeSelected = async () => {
    const ids = [...selected];
    if (!ids.length) return;
    await Promise.allSettled(ids.map((id) => api.deleteImage(id)));
    setItems((prev) => prev.filter((i) => !selected.has(i.id)));
    setSelected(new Set());
    setSelectMode(false);
    refreshStats();
    toast.success(`Deleted ${ids.length} image${ids.length > 1 ? "s" : ""}`);
  };

  const exportSelected = async () => {
    const ids = [...selected];
    if (!ids.length || exporting) return;
    setExporting(true);
    try {
      const blob = await api.exportImages(ids);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `hfabric-images-${ids.length}.zip`;
      a.click();
      URL.revokeObjectURL(url);
      toast.success(`Exported ${ids.length} image${ids.length > 1 ? "s" : ""}`);
    } catch {
      toast.error("Could not export selected images");
    } finally {
      setExporting(false);
    }
  };

  const toggleSelected = (id: string) =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const submitSearch = () => setApplied((a) => ({ ...a, q: query }));

  const modelOptions: SelectOption[] = [
    { value: "", label: stats ? `All models (${stats.total})` : "All models" },
    ...(stats?.by_model ?? []).map((m) => ({ value: m.model, label: m.model, hint: String(m.count) })),
  ];
  const loraOptions: SelectOption[] = [
    { value: "", label: "All LoRAs" },
    ...(stats?.by_lora ?? []).map((lora) => ({ value: lora.id, label: lora.name, hint: String(lora.count) })),
  ];
  const selectedLoraName = loraOptions.find((option) => option.value === applied.lora)?.label ?? applied.lora;

  return (
    <div className="flex h-full min-h-0 flex-col gap-3">
      {/* --- header: counters + search + filters --- */}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <div className="flex items-baseline gap-2">
          <h2 className="text-sm font-semibold text-white/80">History</h2>
          {stats && (
            <span className="text-xs text-white/40">
              {stats.total} total · {stats.today} today
            </span>
          )}
        </div>
        <div className="ml-auto flex flex-wrap items-center gap-2">
          <div className="flex">
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") submitSearch();
              }}
              placeholder="search prompt / seed"
              className="w-44 rounded-l-md border border-white/10 bg-black/30 px-2 py-1 text-xs outline-none focus:border-violet-500"
            />
            <button
              onClick={submitSearch}
              className="rounded-r-md border border-l-0 border-white/10 px-2 py-1 text-xs text-white/70 hover:bg-white/10"
            >
              Go
            </button>
          </div>
          <div className="w-40">
            <Select value={applied.model} options={modelOptions} onChange={(v) => setApplied((a) => ({ ...a, model: v }))} />
          </div>
          <div className="w-32">
            <Select value={applied.size} options={SIZE_FILTERS} onChange={(v) => setApplied((a) => ({ ...a, size: v }))} />
          </div>
          <div className="w-40">
            <Select value={applied.lora} options={loraOptions} onChange={(v) => setApplied((a) => ({ ...a, lora: v }))} />
          </div>
          <div className="w-36">
            <Select value={applied.range} options={DATE_RANGES} onChange={(v) => setApplied((a) => ({ ...a, range: v }))} />
          </div>
          <button
            onClick={() => {
              setSelectMode((v) => !v);
              setSelected(new Set());
            }}
            className={`rounded-md border px-2.5 py-1 text-xs transition ${
              selectMode ? "border-violet-400/70 bg-violet-500/20 text-white" : "border-white/15 text-white/60 hover:bg-white/10"
            }`}
          >
            {selectMode ? "Done" : "Select"}
          </button>
        </div>
      </div>

      {(applied.q || applied.model || applied.size || applied.lora || applied.range !== "all") && (
        <div className="flex flex-wrap items-center gap-1.5 text-xs">
          {applied.q && <Chip onClear={() => { setQuery(""); setApplied((a) => ({ ...a, q: "" })); }}>“{applied.q}”</Chip>}
          {applied.model && <Chip onClear={() => setApplied((a) => ({ ...a, model: "" }))}>{applied.model}</Chip>}
          {applied.size && (
            <Chip onClear={() => setApplied((a) => ({ ...a, size: "" }))}>
              {SIZE_FILTERS.find((r) => r.value === applied.size)?.label}
            </Chip>
          )}
          {applied.lora && <Chip onClear={() => setApplied((a) => ({ ...a, lora: "" }))}>LoRA: {selectedLoraName}</Chip>}
          {applied.range !== "all" && (
            <Chip onClear={() => setApplied((a) => ({ ...a, range: "all" }))}>
              {DATE_RANGES.find((r) => r.value === applied.range)?.label}
            </Chip>
          )}
        </div>
      )}

      {/* --- bulk action bar --- */}
      {selectMode && (
        <div className="flex items-center gap-2 rounded-md border border-white/10 bg-black/30 px-3 py-1.5 text-xs">
          <span className="text-white/60">{selected.size} selected</span>
          <button
            onClick={removeSelected}
            disabled={!selected.size}
            className="rounded border border-red-400/30 px-2 py-0.5 text-red-300 hover:bg-red-400/10 disabled:opacity-30"
          >
            Delete selected
          </button>
          <button
            onClick={exportSelected}
            disabled={!selected.size || exporting}
            className="rounded border border-white/15 px-2 py-0.5 text-white/65 hover:bg-white/10 disabled:opacity-30"
          >
            {exporting ? "Exporting..." : "Export ZIP"}
          </button>
          <button onClick={() => setSelected(new Set())} disabled={!selected.size} className="rounded border border-white/15 px-2 py-0.5 text-white/60 hover:bg-white/10 disabled:opacity-30">
            Clear
          </button>
        </div>
      )}

      {/* --- grid --- */}
      <div className="min-h-0 flex-1 overflow-y-auto">
        {items.length === 0 ? (
          <div className="flex h-full items-center justify-center text-sm text-white/30">
            {loading ? "loading…" : "no images match"}
          </div>
        ) : (
          <>
            <div className="grid grid-cols-[repeat(auto-fill,minmax(120px,1fr))] gap-2">
              {items.map((img) => {
                const isSel = selected.has(img.id);
                return (
                  <button
                    key={img.id}
                    onClick={() => (selectMode ? toggleSelected(img.id) : setOpenId(img.id))}
                    title={String(img.params?.prompt ?? "")}
                    className={`group relative aspect-square animate-fade-in overflow-hidden rounded-md border transition ${
                      isSel ? "border-violet-400 ring-2 ring-violet-400/40" : "border-white/10 hover:border-white/30"
                    }`}
                  >
                    <img src={img.thumb_url ?? img.url} alt="" loading="lazy" className="h-full w-full object-cover" />
                    {selectMode && (
                      <span className={`absolute left-1.5 top-1.5 grid h-5 w-5 place-items-center rounded border text-[11px] ${
                        isSel ? "border-violet-300 bg-violet-500 text-white" : "border-white/40 bg-black/40 text-transparent"
                      }`}>
                        ✓
                      </span>
                    )}
                    {!selectMode && (
                      <span className="pointer-events-none absolute inset-x-0 bottom-0 truncate bg-gradient-to-t from-black/80 to-transparent px-1.5 pb-1 pt-4 text-left text-[10px] text-white/70 opacity-0 transition group-hover:opacity-100">
                        {String(img.params?.model ?? "")}
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
            {hasMore && (
              <div className="mt-3 flex justify-center">
                <button
                  onClick={loadMore}
                  disabled={loading}
                  className="rounded-md border border-white/15 px-4 py-1.5 text-xs text-white/70 hover:bg-white/10 disabled:opacity-40"
                >
                  {loading ? "loading…" : "Load more"}
                </button>
              </div>
            )}
          </>
        )}
      </div>

      {open && (
        <DetailModal
          image={open}
          models={models}
          onClose={() => setOpenId(null)}
          onReproduce={onReproduce}
          onDelete={() => void removeOne(open.id)}
        />
      )}
    </div>
  );
}

function Chip({ children, onClear }: { children: React.ReactNode; onClear: () => void }) {
  return (
    <span className="inline-flex items-center gap-1 rounded-full border border-white/15 bg-white/5 px-2 py-0.5 text-white/70">
      {children}
      <button onClick={onClear} className="text-white/40 hover:text-white" title="remove filter">×</button>
    </span>
  );
}

function DetailModal({
  image,
  models,
  onClose,
  onReproduce,
  onDelete,
}: {
  image: ImageItem;
  models: Model[];
  onClose: () => void;
  onReproduce: (image: ImageItem, opts: { keepSeed: boolean }) => void;
  onDelete: () => void;
}) {
  const params = image.params ?? {};
  const modelName = text(params.model);
  const knownModel = models.some((m) => m.job_type === "image" && m.name === modelName);

  const copyImage = async () => {
    try {
      const blob = await (await fetch(image.url)).blob();
      await navigator.clipboard.write([new ClipboardItem({ [blob.type || "image/png"]: blob })]);
      toast.success("Image copied");
    } catch {
      toast.error("Copy failed — browser blocked clipboard");
    }
  };

  const reveal = async () => {
    try {
      await api.revealImage(image.id);
      toast.success("Opened in file explorer");
    } catch {
      toast.error("Could not open explorer");
    }
  };

  return (
    <div className="fixed inset-0 z-30 flex bg-black/85" onClick={onClose}>
      <div
        className="m-auto flex max-h-[92vh] w-[min(1100px,94vw)] gap-4 overflow-hidden rounded-lg border border-white/10 bg-surface p-4"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex min-h-0 min-w-0 flex-1 items-center justify-center">
          <img src={image.url} alt="" className="max-h-[84vh] max-w-full rounded object-contain" />
        </div>
        <aside className="flex w-72 shrink-0 flex-col overflow-y-auto">
          <div className="mb-2 flex items-center justify-between">
            <h3 className="font-semibold text-white/80">Details</h3>
            <button onClick={onClose} className="text-white/40 hover:text-white">close</button>
          </div>

          <div className="grid grid-cols-2 gap-2">
            <button onClick={() => onReproduce(image, { keepSeed: true })} className={primaryBtn} title={knownModel ? "" : "model not loaded — params applied, model unchanged"}>
              Edit in composer
            </button>
            <button onClick={() => onReproduce(image, { keepSeed: false })} className={actionBtn}>
              Variation
            </button>
          </div>

          <dl className="mt-3 space-y-2">
            <Meta label="Model" value={modelName} />
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
              <button
                onClick={() => navigator.clipboard?.writeText(text(params.prompt)).catch(() => {})}
                className="text-[11px] text-white/40 hover:text-white/80"
              >
                copy
              </button>
            </div>
            <p className="mt-1 whitespace-pre-wrap break-words text-xs leading-5 text-white/70">{text(params.prompt) || "-"}</p>
          </div>
          {text(params.negative) ? (
            <div className="mt-3">
              <div className="text-xs uppercase tracking-wide text-white/35">Negative</div>
              <p className="mt-1 whitespace-pre-wrap break-words text-xs leading-5 text-white/55">{text(params.negative)}</p>
            </div>
          ) : null}

          <div className="mt-4 flex flex-wrap gap-1.5">
            <button onClick={copyImage} className={actionBtn}>Copy</button>
            <button onClick={reveal} className={actionBtn}>Show in folder</button>
            <a href={image.url} download={`${image.id}.png`} className={actionBtn}>PNG</a>
            <a href={`/api/images/${image.id}/metadata`} download className={actionBtn}>JSON</a>
            <button onClick={onDelete} className="rounded border border-red-400/25 px-2.5 py-1 text-xs text-red-300 hover:bg-red-400/10">
              Delete
            </button>
          </div>
        </aside>
      </div>
    </div>
  );
}

const actionBtn = "rounded border border-white/15 px-2.5 py-1 text-xs hover:bg-white/10";
const primaryBtn = "rounded bg-violet-600 px-2.5 py-1 text-xs font-medium text-white transition hover:bg-violet-500";

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
      const name = "name" in item ? text(item.name) : "id" in item ? text(item.id) : "";
      const weight = "weight" in item ? text(item.weight) : "";
      return weight ? `${name || "LoRA"} @ ${weight}` : name;
    })
    .filter(Boolean)
    .join(", ");
}
