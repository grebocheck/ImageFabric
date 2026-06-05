import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api/client";
import { Badge } from "./Badge";
import { Select, type SelectOption } from "./Select";
import { Slider } from "./Slider";
import { Toggle } from "./Toggle";
import type { ComposerApply, Lora, Model, Preset } from "../types";

const STORE_KEY = "hfabric.image.composer";
const PROMPT_HISTORY_KEY = "hfabric.image.promptHistory";
const promptHistoryLimit = 14;

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
  count?: number;
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

function loadPromptHistory(): string[] {
  try {
    const parsed = JSON.parse(localStorage.getItem(PROMPT_HISTORY_KEY) ?? "[]");
    return Array.isArray(parsed) ? parsed.filter((x): x is string => typeof x === "string").slice(0, promptHistoryLimit) : [];
  } catch {
    return [];
  }
}

export function ImageComposer({
  models,
  loras,
  presets,
  onPresetsChanged,
  promptDraft,
  setPromptDraft,
  apply,
}: {
  models: Model[];
  loras: Lora[];
  presets: Preset[];
  onPresetsChanged: () => void;
  promptDraft: string;
  setPromptDraft: (v: string) => void;
  apply?: ComposerApply | null;
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
  const [count, setCount] = useState(saved.count ?? 1);
  const [presetId, setPresetId] = useState(saved.presetId ?? "");
  const [presetName, setPresetName] = useState("");
  const [presetError, setPresetError] = useState("");
  const [promptHistory, setPromptHistory] = useState<string[]>(() => loadPromptHistory());
  const [promptHistoryOpen, setPromptHistoryOpen] = useState(false);
  const promptHistoryRef = useRef<HTMLDivElement>(null);

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
    const data: SavedComposer = { imgModel, negative, steps, guidance, width, height, seed, batch, count, selectedLoras, presetId };
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify(data));
    } catch {
      // Private-mode or quota errors should not break generation.
    }
  }, [imgModel, negative, steps, guidance, width, height, seed, batch, count, selectedLoras, presetId]);

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

  const rememberPrompt = useCallback((content: string) => {
    const text = content.trim();
    if (!text) return;
    setPromptHistory((prev) => [text, ...prev.filter((item) => item !== text)].slice(0, promptHistoryLimit));
  }, []);

  const generate = async () => {
    if (!imgModel || !promptDraft.trim()) return;
    const params = imageParams();
    rememberPrompt(params.prompt);
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

  const updateLoraWeight = (id: string, weight: number) => {
    setSelectedLoras((current) => current.map((lora) => lora.id === id ? { ...lora, weight } : lora));
  };

  const toggleLora = (lora: Lora, enabled: boolean) => {
    setSelectedLoras((current) => {
      const exists = current.some((selected) => selected.id === lora.id);
      if (enabled && !exists) return [...current, { id: lora.id, weight: 1 }];
      if (!enabled) return current.filter((selected) => selected.id !== lora.id);
      return current;
    });
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

  // Load a full param snapshot into the composer. Shared by presets (model
  // identified by id) and History reproduce (model id resolved by the caller).
  const applyParams = (params: Record<string, unknown>, modelId?: string) => {
    if (typeof params.prompt === "string") setPromptDraft(params.prompt);
    setNegative(typeof params.negative === "string" ? params.negative : "");
    const targetId = modelId ?? (typeof params.model_id === "string" ? params.model_id : undefined);
    const model = targetId ? imgModels.find((m) => m.id === targetId) : undefined;
    if (model) setImgModel(model.id);
    setSteps(numberParam(params.steps, steps));
    setGuidance(numberParam(params.guidance, guidance));
    setWidth(numberParam(params.width, width));
    setHeight(numberParam(params.height, height));
    setSeed(numberParam(params.seed, seed));
    setBatch(numberParam(params.batch_size, batch));
    setSelectedLoras(parseLoraSelections(params.loras, loras, model ?? selectedImgModel));
  };

  const applyPreset = () => {
    const preset = imagePresets.find((p) => p.id === presetId);
    if (preset) applyParams(preset.params);
  };

  // External "reproduce from History" request: apply once per nonce.
  const appliedNonce = useRef<number | null>(null);
  useEffect(() => {
    if (!apply || apply.nonce === appliedNonce.current) return;
    appliedNonce.current = apply.nonce;
    applyParams(apply.params, apply.model_id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [apply]);

  useEffect(() => {
    try {
      localStorage.setItem(PROMPT_HISTORY_KEY, JSON.stringify(promptHistory));
    } catch {
      // Private-mode or quota errors should not break recall.
    }
  }, [promptHistory]);

  useEffect(() => {
    if (!promptHistoryOpen) return;
    const onDoc = (e: MouseEvent) => {
      if (promptHistoryRef.current && !promptHistoryRef.current.contains(e.target as Node)) {
        setPromptHistoryOpen(false);
      }
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [promptHistoryOpen]);

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

  const presetOptions: SelectOption[] = [
    { value: "", label: "unsaved" },
    ...imagePresets.map((p) => ({ value: p.id, label: p.name })),
  ];

  const canQueue = Boolean(imgModel) && Boolean(promptDraft.trim());
  const activeRatio = RATIOS.find((r) => isRatio(width, height, r.w, r.h))?.label ?? "custom";
  const promptChars = promptDraft.trim().length;
  const queueLabel = count > 1 ? `Queue ${count} jobs` : "Queue generation";
  const visiblePromptHistory = promptHistory.filter((item) => item !== promptDraft.trim()).slice(0, 8);

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
            <div className="flex items-center gap-2">
              <div ref={promptHistoryRef} className="relative">
                <button
                  type="button"
                  onClick={() => setPromptHistoryOpen((open) => !open)}
                  disabled={visiblePromptHistory.length === 0}
                  title={visiblePromptHistory.length ? "Recall recent prompt" : "No recent image prompts"}
                  aria-label="Recall recent image prompt"
                  aria-expanded={promptHistoryOpen}
                  className="h-6 w-6 rounded-md border border-white/15 text-sm leading-none text-white/55 transition hover:bg-white/10 hover:text-white disabled:opacity-25"
                >
                  ↑
                </button>
                {promptHistoryOpen ? (
                  <div className="absolute right-0 z-30 mt-1 w-72 overflow-hidden rounded-md border border-white/10 bg-surface-2 py-1 shadow-xl shadow-black/60">
                    {visiblePromptHistory.length === 0 ? (
                      <div className="px-2.5 py-1.5 text-sm text-white/30">no recent prompts</div>
                    ) : (
                      visiblePromptHistory.map((prompt) => (
                        <button
                          key={prompt}
                          type="button"
                          onClick={() => {
                            setPromptDraft(prompt);
                            setPromptHistoryOpen(false);
                          }}
                          className="block w-full truncate px-2.5 py-1.5 text-left text-sm text-white/75 transition hover:bg-white/10 hover:text-white"
                          title={prompt}
                        >
                          {prompt}
                        </button>
                      ))
                    )}
                  </div>
                ) : null}
              </div>
              <span className="text-[11px] text-white/30">{promptChars ? `${promptChars} chars` : "empty"}</span>
            </div>
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
          <div className="mt-1.5 grid gap-2">
            {imgModels.length === 0 ? (
              <div className="rounded-md border border-white/10 bg-black/20 px-3 py-2 text-sm text-white/35">no image models</div>
            ) : (
              imgModels.map((model) => (
                <ModelCard
                  key={model.id}
                  model={model}
                  active={model.id === imgModel}
                  onSelect={() => setImgModel(model.id)}
                />
              ))
            )}
          </div>
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
          <div className="flex items-center justify-between gap-2">
            <div className={label}>LoRA</div>
            <span className="text-[11px] text-white/35">{selectedLoras.length ? `${selectedLoras.length} active` : "none active"}</span>
          </div>
          {compatibleLoras.length ? (
            <div className="mt-1.5 flex max-h-64 flex-col gap-2 overflow-y-auto pr-1">
              {compatibleLoras.map((lora) => {
                const selected = selectedLoras.find((item) => item.id === lora.id);
                return (
                  <LoraCard
                    key={lora.id}
                    lora={lora}
                    selected={selected}
                    onToggle={(enabled) => toggleLora(lora, enabled)}
                    onWeight={(weight) => updateLoraWeight(lora.id, weight)}
                  />
                );
              })}
            </div>
          ) : (
            <div className="mt-1.5 rounded-md border border-white/10 bg-black/20 px-3 py-2 text-sm text-white/35">
              {selectedImgModel ? "no compatible LoRA files" : "pick an image model first"}
            </div>
          )}
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

function ModelCard({ model, active, onSelect }: { model: Model; active: boolean; onSelect: () => void }) {
  return (
    <button
      type="button"
      aria-pressed={active}
      onClick={onSelect}
      className={`min-w-0 rounded-md border px-3 py-2 text-left transition ${
        active
          ? "border-violet-400/60 bg-violet-500/15 shadow-sm shadow-violet-950/40"
          : "border-white/10 bg-black/20 hover:border-white/20 hover:bg-white/[0.06]"
      }`}
    >
      <div className="flex min-w-0 items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="truncate text-sm font-medium text-white/85" title={model.name}>{model.name}</div>
          <div className="mt-1 flex flex-wrap gap-1.5">
            <Badge color={familyColor(model.family)}>{model.family}</Badge>
            {model.quant ? <Badge>{model.quant}</Badge> : null}
            {model.loaded ? <Badge color="bg-emerald-700/55 text-emerald-100">loaded</Badge> : model.warm ? <Badge color="bg-sky-700/50 text-sky-100">warm</Badge> : null}
          </div>
        </div>
        {model.estimated_vram_gb ? (
          <div className="shrink-0 text-right">
            <div className="font-mono text-xs text-white/70">{formatVram(model)}</div>
            {model.vram_measured ? <div className="mt-0.5 text-[10px] text-white/35">measured</div> : null}
          </div>
        ) : null}
      </div>
      <div className="mt-2 flex flex-wrap gap-1.5">
        {model.slow ? <Badge color="bg-amber-600/35 text-amber-100">slow</Badge> : null}
        {isNunchaku(model) ? <Badge color="bg-emerald-700/55 text-emerald-100">fast path</Badge> : null}
        {active ? <Badge color="bg-violet-600/45 text-violet-100">selected</Badge> : null}
      </div>
    </button>
  );
}

function LoraCard({
  lora,
  selected,
  onToggle,
  onWeight,
}: {
  lora: Lora;
  selected?: LoraSelection;
  onToggle: (enabled: boolean) => void;
  onWeight: (weight: number) => void;
}) {
  const enabled = Boolean(selected);
  const weight = selected?.weight ?? 1;

  return (
    <div className={`rounded-md border px-2.5 py-2 transition ${
      enabled ? "border-violet-400/45 bg-violet-500/10" : "border-white/10 bg-black/20"
    }`}>
      <div className="flex min-w-0 items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="truncate text-xs font-medium text-white/75" title={lora.name}>{lora.name}</div>
          <div className="mt-1 flex flex-wrap gap-1.5">
            <Badge color={familyColor(lora.family ?? "unknown")}>{lora.family ?? "any"}</Badge>
            <Badge>{formatSize(lora.size_bytes)}</Badge>
          </div>
        </div>
        <Toggle checked={enabled} onChange={onToggle} ariaLabel={`Toggle ${lora.name}`} />
      </div>
      <div className={`mt-2 ${enabled ? "" : "pointer-events-none opacity-35"}`}>
        <div className="mb-1 flex items-center justify-between text-[11px] text-white/40">
          <span>Weight</span>
          <span className="font-mono text-white/60">{weight.toFixed(2)}</span>
        </div>
        <Slider value={weight} min={-2} max={2} step={0.05} onChange={onWeight} />
      </div>
    </div>
  );
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

function formatSize(bytes: number): string {
  if (!bytes) return "";
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
  return `${Math.max(1, Math.round(bytes / 1024 ** 2))} MB`;
}

function formatVram(model: Model): string {
  if (!model.estimated_vram_gb) return "";
  const prefix = model.slow ? ">=" : "~";
  return `${prefix}${model.estimated_vram_gb.toFixed(1)} GB`;
}

function familyColor(family: string): string {
  if (family === "flux2") return "bg-sky-700/50 text-sky-100";
  if (family === "flux") return "bg-violet-700/55 text-violet-100";
  if (family === "sdxl") return "bg-emerald-700/55 text-emerald-100";
  if (family === "gguf") return "bg-amber-700/50 text-amber-100";
  return "bg-white/10 text-white/65";
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
