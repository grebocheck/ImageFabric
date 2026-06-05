import { useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import { Select, type SelectOption } from "./Select";
import { Slider } from "./Slider";
import type { Lora, Model, Preset } from "../types";

const STORE_KEY = "hfabric.image.composer";

type LoraSelection = { id: string; weight: number };
type SavedComposer = {
  imgModel?: string;
  negative?: string;
  steps?: number;
  guidance?: number;
  width?: number;
  height?: number;
  seed?: number;
  batch?: number;
  selectedLoras?: LoraSelection[];
  presetId?: string;
};

const DEFAULT_STEPS = 28;
const DEFAULT_GUIDANCE = 3.5;
const DEFAULT_SIZE = 1024;
const FLUX2_STEPS = 6;
const FLUX2_GUIDANCE = 4.0;
const FLUX2_SIZE = 768;

const field =
  "w-full rounded-md border border-white/10 bg-black/30 px-2.5 py-1.5 text-[13px] outline-none transition placeholder:text-white/25 focus:border-violet-500";
const label = "text-[10px] font-medium uppercase tracking-wide text-white/45";
const section = "border-b border-white/10 p-3 last:border-b-0";
const subtleButton = "rounded-md border border-white/15 px-2.5 py-1.5 text-xs text-white/70 transition hover:bg-white/10 hover:text-white disabled:opacity-30";

const RATIOS: Array<{ label: string; w: number; h: number }> = [
  { label: "1:1", w: 1, h: 1 },
  { label: "3:4", w: 3, h: 4 },
  { label: "4:3", w: 4, h: 3 },
  { label: "16:9", w: 16, h: 9 },
  { label: "9:16", w: 9, h: 16 },
];

function readSaved(): SavedComposer {
  try {
    const raw = localStorage.getItem(STORE_KEY);
    return raw ? (JSON.parse(raw) as SavedComposer) : {};
  } catch {
    return {};
  }
}

export function ImageComposer({
  models,
  loras,
  presets,
  onPresetsChanged,
  promptDraft,
  setPromptDraft,
}: {
  models: Model[];
  loras: Lora[];
  presets: Preset[];
  onPresetsChanged: () => void;
  promptDraft: string;
  setPromptDraft: (v: string) => void;
}) {
  const imgModels = models
    .filter((m) => m.job_type === "image")
    .sort((a, b) => imageModelRank(a) - imageModelRank(b) || a.name.localeCompare(b.name));

  const saved = useMemo(readSaved, []);
  const [imgModel, setImgModel] = useState(saved.imgModel ?? "");
  const [negative, setNegative] = useState(saved.negative ?? "");
  const [steps, setSteps] = useState(saved.steps ?? DEFAULT_STEPS);
  const [guidance, setGuidance] = useState(saved.guidance ?? DEFAULT_GUIDANCE);
  const [width, setWidth] = useState(saved.width ?? DEFAULT_SIZE);
  const [height, setHeight] = useState(saved.height ?? DEFAULT_SIZE);
  const [seed, setSeed] = useState(saved.seed ?? -1);
  const [batch, setBatch] = useState(saved.batch ?? 1);
  const [selectedLoras, setSelectedLoras] = useState<LoraSelection[]>(saved.selectedLoras ?? []);
  const [loraId, setLoraId] = useState("");
  const [loraWeight, setLoraWeight] = useState(1);
  const [count, setCount] = useState(1);
  const [presetId, setPresetId] = useState(saved.presetId ?? "");
  const [presetName, setPresetName] = useState("");
  const [presetError, setPresetError] = useState("");

  const selectedImgModel = imgModels.find((m) => m.id === imgModel);
  const selectedFamily = selectedImgModel?.family;
  const imagePresets = presets.filter((p) => p.type === "image");
  const compatibleLoras = loras
    .filter((lora) => isLoraCompatible(lora, selectedImgModel))
    .sort((a, b) => a.name.localeCompare(b.name));

  useEffect(() => {
    if (!imgModel || !imgModels.some((m) => m.id === imgModel)) {
      const preferred = pickDefaultImageModel(imgModels);
      if (preferred) setImgModel(preferred.id);
    }
  }, [imgModels, imgModel]);

  useEffect(() => {
    const data: SavedComposer = { imgModel, negative, steps, guidance, width, height, seed, batch, selectedLoras, presetId };
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify(data));
    } catch {
      // Private-mode or quota errors should not break generation.
    }
  }, [imgModel, negative, steps, guidance, width, height, seed, batch, selectedLoras, presetId]);

  useEffect(() => {
    setSelectedLoras((current) =>
      current.filter((selected) => {
        const lora = loras.find((item) => item.id === selected.id);
        return lora ? isLoraCompatible(lora, selectedImgModel) : false;
      }),
    );
  }, [loras, selectedImgModel]);

  useEffect(() => {
    if (selectedFamily !== "flux2") return;
    setSteps((value) => value === DEFAULT_STEPS ? FLUX2_STEPS : value);
    setGuidance((value) => value === DEFAULT_GUIDANCE ? FLUX2_GUIDANCE : value);
    setWidth((value) => value === DEFAULT_SIZE ? FLUX2_SIZE : value);
    setHeight((value) => value === DEFAULT_SIZE ? FLUX2_SIZE : value);
  }, [selectedFamily]);

  const imageParams = () => ({
    prompt: promptDraft.trim(),
    negative: negative.trim() || undefined,
    steps,
    guidance,
    width,
    height,
    seed,
    batch_size: batch,
    loras: selectedLoras.length ? selectedLoras.map(({ id, weight }) => ({ id, weight })) : undefined,
  });

  const generate = async () => {
    if (!imgModel || !promptDraft.trim()) return;
    const params = imageParams();
    await api.createJobs(Array.from({ length: count }, () => ({ type: "image" as const, model_id: imgModel, params })));
  };

  const applyRatio = (rw: number, rh: number) => {
    const base = selectedFamily === "flux2" ? FLUX2_SIZE : DEFAULT_SIZE;
    const round64 = (n: number) => Math.max(64, Math.round(n / 64) * 64);
    if (rw >= rh) {
      setWidth(round64(base));
      setHeight(round64((base * rh) / rw));
    } else {
      setHeight(round64(base));
      setWidth(round64((base * rw) / rh));
    }
  };

  const addLora = () => {
    if (!loraId || selectedLoras.some((lora) => lora.id === loraId)) return;
    const lora = compatibleLoras.find((item) => item.id === loraId);
    if (!lora) return;
    setSelectedLoras((current) => [...current, { id: lora.id, weight: loraWeight }]);
    setLoraId("");
    setLoraWeight(1);
  };

  const updateLoraWeight = (id: string, weight: number) => {
    setSelectedLoras((current) => current.map((lora) => lora.id === id ? { ...lora, weight } : lora));
  };

  const removeLora = (id: string) => {
    setSelectedLoras((current) => current.filter((lora) => lora.id !== id));
  };

  const savePreset = async () => {
    const name = presetName.trim();
    if (!name) return;
    setPresetError("");
    try {
      await api.createPreset(name, "image", { ...imageParams(), model_id: imgModel });
      setPresetName("");
      onPresetsChanged();
    } catch (err) {
      setPresetError(err instanceof Error ? err.message : "Could not save preset");
    }
  };

  const applyPreset = () => {
    const preset = imagePresets.find((p) => p.id === presetId);
    if (!preset) return;
    const params = preset.params;
    if (typeof params.prompt === "string") setPromptDraft(params.prompt);
    setNegative(typeof params.negative === "string" ? params.negative : "");
    const presetModel = typeof params.model_id === "string"
      ? imgModels.find((m) => m.id === params.model_id)
      : undefined;
    if (presetModel) setImgModel(presetModel.id);
    setSteps(numberParam(params.steps, steps));
    setGuidance(numberParam(params.guidance, guidance));
    setWidth(numberParam(params.width, width));
    setHeight(numberParam(params.height, height));
    setSeed(numberParam(params.seed, seed));
    setBatch(numberParam(params.batch_size, batch));
    setSelectedLoras(parseLoraSelections(params.loras, loras, presetModel ?? selectedImgModel));
  };

  const deletePreset = async () => {
    if (!presetId) return;
    setPresetError("");
    try {
      await api.deletePreset(presetId);
      setPresetId("");
      onPresetsChanged();
    } catch (err) {
      setPresetError(err instanceof Error ? err.message : "Could not delete preset");
    }
  };

  const modelOptions: SelectOption[] = imgModels.map((m) => {
    const meta = modelMeta(m);
    return { value: m.id, label: meta.label, hint: meta.hint };
  });
  const loraOptions: SelectOption[] = [
    { value: "", label: "none" },
    ...compatibleLoras.map((lora) => ({ value: lora.id, label: lora.name, hint: loraHint(lora) })),
  ];
  const presetOptions: SelectOption[] = [
    { value: "", label: "unsaved" },
    ...imagePresets.map((p) => ({ value: p.id, label: p.name })),
  ];

  const canQueue = Boolean(imgModel) && Boolean(promptDraft.trim());
  const activeRatio = RATIOS.find((r) => isRatio(width, height, r.w, r.h))?.label ?? "custom";
  const promptChars = promptDraft.trim().length;
  const queueLabel = count > 1 ? `Queue ${count} jobs` : "Queue generation";

  return (
    <section className="flex h-full min-h-0 flex-col overflow-hidden rounded-lg border border-white/10 bg-surface max-[860px]:mb-4 max-[860px]:h-[760px]">
      <div className="border-b border-white/10 px-3 py-3">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <h2 className="text-sm font-semibold text-white/85">Generate</h2>
            <p className="mt-0.5 truncate text-xs text-white/40">{width}x{height} / {steps} steps / {selectedLoras.length || "no"} LoRA</p>
          </div>
          <span className="shrink-0 rounded-md border border-white/10 bg-black/25 px-2 py-1 text-[11px] uppercase text-white/50">
            {selectedFamily ?? "image"}
          </span>
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto">
        <section className={section}>
          <div className="flex items-center justify-between">
            <label htmlFor="image-prompt" className={label}>Prompt</label>
            <span className="text-[11px] text-white/30">{promptChars ? `${promptChars} chars` : "empty"}</span>
          </div>
          <textarea
            id="image-prompt"
            value={promptDraft}
            onChange={(e) => setPromptDraft(e.target.value)}
            rows={6}
            placeholder="describe the image..."
            className={`${field} mt-1.5 min-h-32 resize-y leading-5`}
          />
          <label className="mt-3 block">
            <div className={label}>Negative {selectedFamily === "flux2" ? "(ignored by FLUX.2)" : ""}</div>
            <input
              value={negative}
              onChange={(e) => setNegative(e.target.value)}
              placeholder="things to avoid..."
              className={`${field} mt-1.5`}
            />
          </label>
        </section>

        <section className={section}>
          <div className={label}>Model</div>
          <Select value={imgModel} options={modelOptions} onChange={setImgModel} placeholder="pick a model..." className="mt-1.5" />
          {selectedImgModel?.slow ? (
            <Notice tone="amber">
              Raw FLUX fp8 is slow and high-memory on 16 GB VRAM. Prefer a nunchaku FLUX entry when available.
            </Notice>
          ) : null}
          {selectedFamily === "flux2" && isNunchaku(selectedImgModel) ? (
            <Notice tone="emerald">
              FLUX.2 nunchaku uses the local SVDQuant transformer sidecar.
            </Notice>
          ) : selectedFamily === "flux2" ? (
            <Notice tone="sky">
              FLUX.2 klein is tuned here for 768x768, 6 steps, guidance 4.0. Negative prompt is ignored.
            </Notice>
          ) : null}
        </section>

        <section className={section}>
          <div className="flex items-center justify-between">
            <div className={label}>Canvas</div>
            <span className="text-[11px] text-white/35">{activeRatio}</span>
          </div>
          <div className="mt-1.5 flex flex-wrap gap-1.5">
            {RATIOS.map((r) => {
              const active = isRatio(width, height, r.w, r.h);
              return (
                <button
                  key={r.label}
                  onClick={() => applyRatio(r.w, r.h)}
                  className={`h-7 rounded-md border px-2.5 text-xs transition ${
                    active ? "border-violet-400/70 bg-violet-500/20 text-white" : "border-white/15 text-white/60 hover:bg-white/10"
                  }`}
                >
                  {r.label}
                </button>
              );
            })}
          </div>
          <div className="mt-2 grid grid-cols-2 gap-2">
            <Num label="Width" v={width} set={setWidth} step={64} />
            <Num label="Height" v={height} set={setHeight} step={64} />
          </div>
        </section>

        <section className={section}>
          <div className={label}>Sampling</div>
          <div className="mt-1.5 grid grid-cols-2 gap-2">
            <Num label="Steps" v={steps} set={setSteps} />
            <Num label="Guidance" v={guidance} set={setGuidance} step={0.1} />
            <Num label="Seed" v={seed} set={setSeed} />
            <Num label="Batch" v={batch} set={setBatch} />
          </div>
        </section>

        <section className={section}>
          <div className={label}>LoRA</div>
          <div className="mt-1.5 grid grid-cols-[minmax(0,1fr)_72px_auto] gap-2">
            <Select value={loraId} options={loraOptions} onChange={setLoraId} placeholder="none" />
            <input
              type="number"
              value={loraWeight}
              min={-2}
              max={2}
              step={0.05}
              onChange={(e) => setLoraWeight(Number(e.target.value))}
              className={field}
              aria-label="LoRA weight"
            />
            <button
              onClick={addLora}
              disabled={!loraId || selectedLoras.some((lora) => lora.id === loraId)}
              className={subtleButton}
            >
              Add
            </button>
          </div>
          {selectedLoras.length ? (
            <div className="mt-2 flex flex-col gap-1.5">
              {selectedLoras.map((selected) => {
                const lora = loras.find((item) => item.id === selected.id);
                return (
                  <div key={selected.id} className="rounded-md border border-white/10 bg-black/20 px-2 py-1.5">
                    <div className="flex items-center justify-between gap-2">
                      <div className="min-w-0 truncate text-xs text-white/75" title={lora?.name ?? selected.id}>{lora?.name ?? selected.id}</div>
                      <button
                        onClick={() => removeLora(selected.id)}
                        className="h-5 w-5 shrink-0 rounded border border-white/15 text-xs text-white/50 hover:bg-white/10 hover:text-white"
                        title="Remove LoRA"
                      >
                        x
                      </button>
                    </div>
                    <Slider
                      value={selected.weight}
                      min={-2}
                      max={2}
                      step={0.05}
                      onChange={(v) => updateLoraWeight(selected.id, v)}
                    />
                  </div>
                );
              })}
            </div>
          ) : null}
        </section>

        <section className={section}>
          <div className={label}>Preset</div>
          <div className="mt-1.5 grid grid-cols-[minmax(0,1fr)_auto_auto] gap-2">
            <Select value={presetId} options={presetOptions} onChange={setPresetId} placeholder="unsaved" />
            <button onClick={applyPreset} disabled={!presetId} className={subtleButton}>Apply</button>
            <button
              onClick={deletePreset}
              disabled={!presetId}
              className="rounded-md border border-red-400/25 px-2.5 py-1.5 text-xs text-red-300 transition hover:bg-red-400/10 disabled:opacity-30"
            >
              Delete
            </button>
          </div>
          <div className="mt-2 grid grid-cols-[minmax(0,1fr)_auto] gap-2">
            <input
              value={presetName}
              onChange={(e) => setPresetName(e.target.value)}
              placeholder="preset name"
              className={field}
            />
            <button onClick={savePreset} disabled={!presetName.trim()} className={subtleButton}>Save</button>
          </div>
          {presetError ? <div className="mt-1 truncate text-xs text-red-300" title={presetError}>{presetError}</div> : null}
        </section>
      </div>

      <div className="border-t border-white/10 bg-black/20 p-3">
        <div className="grid grid-cols-[76px_minmax(0,1fr)] gap-2">
          <label>
            <div className={label}>Jobs</div>
            <input
              type="number"
              value={count}
              min={1}
              onChange={(e) => setCount(Math.max(1, Number(e.target.value)))}
              className="mt-1 w-full rounded-md border border-white/10 bg-black/30 px-2 py-2 text-sm outline-none focus:border-violet-500"
            />
          </label>
          <button
            onClick={generate}
            disabled={!canQueue}
            className="mt-4 rounded-md bg-violet-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-violet-500 disabled:opacity-40"
          >
            {queueLabel}
          </button>
        </div>
        <div className="mt-2 flex items-center justify-between gap-2 text-[11px] text-white/35">
          <span className="truncate">{selectedImgModel?.name ?? "No image model"}</span>
          <span className="shrink-0">{seed === -1 ? "random seed" : `seed ${seed}`}</span>
        </div>
      </div>
    </section>
  );
}

function Notice({ tone, children }: { tone: "amber" | "emerald" | "sky"; children: string }) {
  const classes = {
    amber: "border-amber-500/30 bg-amber-500/10 text-amber-100",
    emerald: "border-emerald-500/30 bg-emerald-500/10 text-emerald-100",
    sky: "border-sky-500/30 bg-sky-500/10 text-sky-100",
  };
  return <div className={`mt-2 rounded-md border px-2.5 py-2 text-xs leading-5 ${classes[tone]}`}>{children}</div>;
}

function Num({ label: l, v, set, step = 1 }: { label: string; v: number; set: (n: number) => void; step?: number }) {
  return (
    <label className="block">
      <div className={label}>{l}</div>
      <input
        type="number"
        value={v}
        step={step}
        onChange={(e) => set(Number(e.target.value))}
        className="mt-1 w-full rounded-md border border-white/10 bg-black/30 px-2 py-1.5 text-sm outline-none focus:border-violet-500"
      />
    </label>
  );
}

function isRatio(w: number, h: number, rw: number, rh: number): boolean {
  if (!w || !h) return false;
  return Math.abs(w / h - rw / rh) < 0.02;
}

function parseLoraSelections(value: unknown, loras: Lora[], model: Model | undefined): LoraSelection[] {
  if (!Array.isArray(value)) return [];
  const seen = new Set<string>();
  const selections: LoraSelection[] = [];
  for (const item of value) {
    const id = typeof item === "string"
      ? item
      : item && typeof item === "object" && "id" in item && typeof item.id === "string"
        ? item.id
        : "";
    if (!id || seen.has(id)) continue;
    const lora = loras.find((candidate) => candidate.id === id);
    if (!lora || !isLoraCompatible(lora, model)) continue;
    const weight = item && typeof item === "object" && "weight" in item
      ? numberParam(item.weight, 1)
      : 1;
    selections.push({ id, weight });
    seen.add(id);
  }
  return selections;
}

function isLoraCompatible(lora: Lora, model: Model | undefined): boolean {
  return !model || !lora.family || lora.family === model.family;
}

function loraHint(lora: Lora): string {
  return [lora.family ?? "unknown", formatSize(lora.size_bytes)].filter(Boolean).join(" / ");
}

function formatSize(bytes: number): string {
  if (!bytes) return "";
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
  return `${Math.max(1, Math.round(bytes / 1024 ** 2))} MB`;
}

function numberParam(value: unknown, fallback: number): number {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function imageModelRank(model: Model): number {
  if (model.family === "flux2" && isNunchaku(model)) return -1;
  if (model.family === "flux2") return 0;
  if (model.family === "flux" && isNunchaku(model)) return 0;
  if (!model.slow) return 1;
  return 2;
}

function pickDefaultImageModel(models: Model[]): Model | undefined {
  return models.find((m) => m.family === "flux" && isNunchaku(m))
    ?? models.find((m) => !m.slow)
    ?? models[0];
}

function isNunchaku(model: Model | undefined): boolean {
  return Boolean(model?.quant?.startsWith("nunchaku"));
}

function modelMeta(model: Model): { label: string; hint?: string } {
  const tags: string[] = [];
  if (model.quant) tags.push(model.quant);
  if (model.estimated_vram_gb) {
    const prefix = model.slow ? ">=" : "~";
    tags.push(`${prefix}${model.estimated_vram_gb.toFixed(1)} GB`);
  }
  if (model.slow) tags.push("slow");
  return { label: model.name, hint: tags.length ? tags.join(" / ") : undefined };
}
