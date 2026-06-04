import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { Lora, Model, Preset } from "../types";

const field = "w-full rounded-md bg-black/30 border border-white/10 px-2.5 py-1.5 text-sm outline-none focus:border-violet-500";
const label = "text-xs uppercase tracking-wide text-white/40";
const DEFAULT_STEPS = 28;
const DEFAULT_GUIDANCE = 3.5;
const DEFAULT_SIZE = 1024;
const FLUX2_STEPS = 6;
const FLUX2_GUIDANCE = 4.0;
const FLUX2_SIZE = 768;

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

  const [imgModel, setImgModel] = useState("");
  const [negative, setNegative] = useState("");
  const [steps, setSteps] = useState(DEFAULT_STEPS);
  const [guidance, setGuidance] = useState(DEFAULT_GUIDANCE);
  const [width, setWidth] = useState(DEFAULT_SIZE);
  const [height, setHeight] = useState(DEFAULT_SIZE);
  const [seed, setSeed] = useState(-1);
  const [batch, setBatch] = useState(1);
  const [selectedLoras, setSelectedLoras] = useState<LoraSelection[]>([]);
  const [loraId, setLoraId] = useState("");
  const [loraWeight, setLoraWeight] = useState(1);
  const [count, setCount] = useState(1);
  const [presetId, setPresetId] = useState("");
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
    setSelectedLoras((current) =>
      current.filter((selected) => {
        const lora = loras.find((item) => item.id === selected.id);
        return lora ? isLoraCompatible(lora, selectedImgModel) : false;
      }),
    );
  }, [loras, selectedFamily]);

  useEffect(() => {
    if (selectedFamily !== "flux2") return;
    setSteps((value) => value === DEFAULT_STEPS ? FLUX2_STEPS : value);
    setGuidance((value) => value === DEFAULT_GUIDANCE ? FLUX2_GUIDANCE : value);
    setWidth((value) => value === DEFAULT_SIZE ? FLUX2_SIZE : value);
    setHeight((value) => value === DEFAULT_SIZE ? FLUX2_SIZE : value);
  }, [selectedFamily]);

  const generate = async () => {
    if (!imgModel || !promptDraft.trim()) return;
    const params = imageParams();
    const jobs = Array.from({ length: count }, () => ({
      type: "image" as const,
      model_id: imgModel,
      params,
    }));
    await api.createJobs(jobs);
  };

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
    if (presetModel) {
      setImgModel(presetModel.id);
    }
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

  return (
    <div className="flex flex-col gap-4">
      <section className="rounded-lg border border-white/10 p-3">
        <div className={label}>Prompt</div>
        <textarea
          value={promptDraft}
          onChange={(e) => setPromptDraft(e.target.value)}
          rows={5}
          placeholder="describe the image…"
          className={`${field} mt-1 resize-none`}
        />
        <div className={`${label} mt-3`}>Negative</div>
        <input value={negative} onChange={(e) => setNegative(e.target.value)} className={`${field} mt-1`} />

        <div className="mt-3 grid grid-cols-3 gap-2">
          <Num label="Steps" v={steps} set={setSteps} />
          <Num label="Guidance" v={guidance} set={setGuidance} step={0.1} />
          <Num label="Seed" v={seed} set={setSeed} />
          <Num label="Width" v={width} set={setWidth} step={64} />
          <Num label="Height" v={height} set={setHeight} step={64} />
          <Num label="Batch" v={batch} set={setBatch} />
        </div>

        <div className="mt-3 flex items-end gap-2">
          <label className="flex-1">
            <div className={label}>Model</div>
            <select value={imgModel} onChange={(e) => setImgModel(e.target.value)} className={`${field} mt-1`}>
              {imgModels.map((m) => (
                <option key={m.id} value={m.id}>{modelLabel(m)}</option>
              ))}
            </select>
            {selectedImgModel?.slow ? (
              <div className="mt-2 rounded-md border border-amber-500/30 bg-amber-500/10 px-2.5 py-2 text-xs text-amber-100">
                Raw FLUX fp8 is slow and high-mem on 16 GB VRAM. Prefer the nunchaku FLUX entry when available.
              </div>
            ) : null}
            {selectedFamily === "flux2" && isNunchaku(selectedImgModel) ? (
              <div className="mt-2 rounded-md border border-emerald-500/30 bg-emerald-500/10 px-2.5 py-2 text-xs text-emerald-100">
                FLUX.2 nunchaku is an experimental sidecar path using the local SVDQuant transformer.
              </div>
            ) : selectedFamily === "flux2" ? (
              <div className="mt-2 rounded-md border border-sky-500/30 bg-sky-500/10 px-2.5 py-2 text-xs text-sky-100">
                FLUX.2 klein was validated at 768x768, 6 steps, guidance 4.0 on this 16 GB GPU. Negative prompt is ignored.
              </div>
            ) : null}
          </label>
          <Num label="× jobs" v={count} set={setCount} />
          <button
            onClick={generate}
            disabled={!imgModel || !promptDraft.trim()}
            className="rounded-md bg-violet-600 px-4 py-1.5 text-sm font-medium hover:bg-violet-500 disabled:opacity-40"
          >
            Queue
          </button>
        </div>

        <div className="mt-3">
          <div className={label}>LoRA</div>
          <div className="mt-1 grid grid-cols-[1fr_84px_auto] gap-2">
            <select value={loraId} onChange={(e) => setLoraId(e.target.value)} className={field}>
              <option value="">none</option>
              {compatibleLoras.map((lora) => (
                <option key={lora.id} value={lora.id}>{loraLabel(lora)}</option>
              ))}
            </select>
            <input
              type="number"
              value={loraWeight}
              min={-2}
              max={2}
              step={0.05}
              onChange={(e) => setLoraWeight(Number(e.target.value))}
              className={field}
            />
            <button
              onClick={addLora}
              disabled={!loraId || selectedLoras.some((lora) => lora.id === loraId)}
              className="rounded-md border border-white/15 px-2.5 py-1 text-xs hover:bg-white/10 disabled:opacity-30"
            >
              Add
            </button>
          </div>
          {selectedLoras.length ? (
            <div className="mt-2 flex flex-col gap-1.5">
              {selectedLoras.map((selected) => {
                const lora = loras.find((item) => item.id === selected.id);
                return (
                  <div key={selected.id} className="grid grid-cols-[1fr_80px_auto] items-center gap-2 rounded-md border border-white/10 bg-white/[0.03] px-2 py-1.5">
                    <div className="min-w-0 truncate text-sm text-white/80">{lora?.name ?? selected.id}</div>
                    <input
                      type="number"
                      value={selected.weight}
                      min={-2}
                      max={2}
                      step={0.05}
                      onChange={(e) => updateLoraWeight(selected.id, Number(e.target.value))}
                      className="w-full rounded-md border border-white/10 bg-black/30 px-2 py-1 text-xs outline-none focus:border-violet-500"
                    />
                    <button
                      onClick={() => removeLora(selected.id)}
                      className="rounded-md border border-white/15 px-2 py-1 text-xs hover:bg-white/10"
                    >
                      Remove
                    </button>
                  </div>
                );
              })}
            </div>
          ) : null}
        </div>

        <div className="mt-3 grid grid-cols-[1fr_auto_auto] gap-2">
          <label className="min-w-0">
            <div className={label}>Preset</div>
            <select value={presetId} onChange={(e) => setPresetId(e.target.value)} className={`${field} mt-1`}>
              <option value="">unsaved</option>
              {imagePresets.map((p) => (
                <option key={p.id} value={p.id}>{p.name}</option>
              ))}
            </select>
          </label>
          <button
            onClick={applyPreset}
            disabled={!presetId}
            className="mt-5 rounded-md border border-white/15 px-2.5 py-1 text-xs hover:bg-white/10 disabled:opacity-30"
          >
            Apply
          </button>
          <button
            onClick={deletePreset}
            disabled={!presetId}
            className="mt-5 rounded-md border border-red-400/25 px-2.5 py-1 text-xs text-red-300 hover:bg-red-400/10 disabled:opacity-30"
          >
            Delete
          </button>
        </div>

        <div className="mt-2 grid grid-cols-[1fr_auto] gap-2">
          <input
            value={presetName}
            onChange={(e) => setPresetName(e.target.value)}
            placeholder="preset name"
            className={field}
          />
          <button
            onClick={savePreset}
            disabled={!presetName.trim()}
            className="rounded-md border border-white/15 px-2.5 py-1 text-xs hover:bg-white/10 disabled:opacity-30"
          >
            Save
          </button>
        </div>
        {presetError ? <div className="mt-1 truncate text-xs text-red-300">{presetError}</div> : null}
      </section>
    </div>
  );
}

type LoraSelection = { id: string; weight: number };

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

function loraLabel(lora: Lora): string {
  const tags = [lora.family ?? "unknown", formatSize(lora.size_bytes)].filter(Boolean);
  return `${lora.name} (${tags.join(", ")})`;
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

function modelLabel(model: Model): string {
  const tags: string[] = [];
  if (model.quant) tags.push(model.quant);
  if (model.estimated_vram_gb) {
    const prefix = model.slow ? ">=" : "~";
    tags.push(`${prefix}${model.estimated_vram_gb.toFixed(1)} GB VRAM`);
  }
  if (model.slow) tags.push("slow/high-mem");
  return tags.length ? `${model.name} (${tags.join(", ")})` : model.name;
}

function Num({ label: l, v, set, step = 1 }: { label: string; v: number; set: (n: number) => void; step?: number }) {
  return (
    <label className="block">
      <div className="text-xs uppercase tracking-wide text-white/40">{l}</div>
      <input
        type="number"
        value={v}
        step={step}
        onChange={(e) => set(Number(e.target.value))}
        className="mt-1 w-full rounded-md border border-white/10 bg-black/30 px-2 py-1 text-sm outline-none focus:border-violet-500"
      />
    </label>
  );
}
