import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api/client";
import { Badge } from "./Badge";
import { MaskEditor } from "./MaskEditor";
import { ModelPicker } from "./ModelPicker";
import { Select, type SelectOption } from "./Select";
import { Slider } from "./Slider";
import { SkeletonLine, SkeletonRows } from "./WorkspaceChrome";
import { Toggle } from "./Toggle";
import type { ComposerApply, Lora, Model, Preset } from "../types";
import {
  DEFAULT_GUIDANCE,
  DEFAULT_SIZE,
  DEFAULT_STEPS,
  familyColor,
  formatSize,
  imageFamilyDefaults,
  imageModelRank,
  isKnownGuidanceDefault,
  isKnownSizeDefault,
  isKnownStepDefault,
  isLoraCompatible,
  isNunchaku,
  loadPromptHistory,
  numberParam,
  pickDefaultImageModel,
  PROMPT_HISTORY_KEY,
  promptHistoryLimit,
  readSaved,
  STORE_KEY,
  type LoraSelection,
  type SavedComposer,
} from "./imageComposerHelpers";

const field =
  "w-full rounded-md border border-white/10 bg-black/30 px-2.5 py-1.5 text-[13px] outline-none transition placeholder:text-white/25 focus:border-accent";
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

export function ImageComposer({
  models,
  modelsLoading = false,
  loras,
  lorasLoading = false,
  presets,
  presetsLoading = false,
  onPresetsChanged,
  promptDraft,
  setPromptDraft,
  apply,
}: {
  models: Model[];
  modelsLoading?: boolean;
  loras: Lora[];
  lorasLoading?: boolean;
  presets: Preset[];
  presetsLoading?: boolean;
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
  // img2img source (P13.4) — transient, not persisted. Only SDXL is wired so far.
  const [initImage, setInitImage] = useState<{ token: string; url: string } | null>(null);
  const [maskDraft, setMaskDraft] = useState<File | null>(null);
  const [strength, setStrength] = useState(0.6);
  const [uploadBusy, setUploadBusy] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const img2imgSupported = selectedFamily === "sdxl";
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
    const defaults = imageFamilyDefaults(selectedFamily);
    if (!defaults) return;
    setSteps((value) => isKnownStepDefault(value) ? defaults.steps : value);
    setGuidance((value) => isKnownGuidanceDefault(value) ? defaults.guidance : value);
    setWidth((value) => isKnownSizeDefault(value) ? defaults.width : value);
    setHeight((value) => isKnownSizeDefault(value) ? defaults.height : value);
  }, [selectedFamily]);

  const useImg2img = img2imgSupported && initImage !== null;
  const imageParams = (maskToken?: string) => ({
    prompt: promptDraft.trim(),
    negative: negative.trim() || undefined,
    steps,
    guidance,
    width,
    height,
    seed,
    batch_size: batch,
    loras: selectedLoras.length ? selectedLoras.map(({ id, weight }) => ({ id, weight })) : undefined,
    init_image: useImg2img ? initImage!.token : undefined,
    mask_image: useImg2img && maskToken ? maskToken : undefined,
    strength: useImg2img ? strength : undefined,
  });

  const rememberPrompt = useCallback((content: string) => {
    const text = content.trim();
    if (!text) return;
    setPromptHistory((prev) => [text, ...prev.filter((item) => item !== text)].slice(0, promptHistoryLimit));
  }, []);

  const generate = async () => {
    if (!imgModel || !promptDraft.trim()) return;
    let maskToken: string | undefined;
    if (useImg2img && maskDraft) {
      setUploadError("");
      try {
        const res = await api.uploadMaskImage(maskDraft);
        maskToken = res.mask_image;
      } catch {
        setUploadError("mask upload failed");
        return;
      }
    }
    const params = imageParams(maskToken);
    rememberPrompt(params.prompt);
    await api.createJobs(Array.from({ length: count }, () => ({ type: "image" as const, model_id: imgModel, params })));
  };

  const onPickInitImage = async (file: File | null | undefined) => {
    if (!file) return;
    setUploadError("");
    setUploadBusy(true);
    try {
      const res = await api.uploadInitImage(file);
      setInitImage({ token: res.init_image, url: res.url });
      setMaskDraft(null);
      // snap the canvas to the source aspect (rounded to 64) for a faithful result
      const round64 = (n: number) => Math.max(64, Math.round(n / 64) * 64);
      setWidth(round64(res.width));
      setHeight(round64(res.height));
    } catch {
      setUploadError("upload failed");
    } finally {
      setUploadBusy(false);
    }
  };

  const applyRatio = (rw: number, rh: number) => {
    const base = imageFamilyDefaults(selectedFamily)?.width ?? DEFAULT_SIZE;
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
          <div className="mt-1.5">
            {modelsLoading && imgModels.length === 0 ? (
              <SkeletonLine />
            ) : imgModels.length === 0 ? (
              <div className="rounded-md border border-white/10 bg-black/20 px-3 py-2 text-sm text-white/35">no image models</div>
            ) : (
              <ModelPicker models={imgModels} value={imgModel} onChange={setImgModel} />
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
          ) : selectedFamily === "qwen-image" ? (
            <Notice tone="sky">
              Qwen-Image-2512 is tuned here for 1328x1328, 50 steps, true CFG 4.0. The backend defaults to bnb-nf4.
            </Notice>
          ) : selectedFamily === "z-image" ? (
            <Notice tone="sky">
              Z-Image-Turbo is tuned here for 1024x1024, 9 steps, guidance 0.0.
            </Notice>
          ) : null}
        </section>

        {img2imgSupported ? (
          <section className={section}>
            <div className="flex items-center justify-between">
              <div className={label}>Source image (img2img)</div>
              {initImage ? (
                <button
                  onClick={() => {
                    setInitImage(null);
                    setMaskDraft(null);
                  }}
                  className="text-[11px] text-white/45 transition hover:text-white/80"
                >
                  clear
                </button>
              ) : null}
            </div>
            {initImage ? (
              <div className="mt-1.5 space-y-2">
                <img
                  src={initImage.url}
                  alt="source"
                  className="max-h-40 w-full rounded-md border border-white/10 bg-black/30 object-contain"
                />
                <div className="flex items-center justify-between text-[11px] text-white/40">
                  <span>Strength</span>
                  <span className="font-mono text-white/60">{strength.toFixed(2)}</span>
                </div>
                <Slider value={strength} min={0.05} max={1} step={0.05} onChange={setStrength} />
                <p className="text-[11px] text-white/35">Lower keeps the source; higher follows the prompt.</p>
                <MaskEditor src={initImage.url} onMaskChange={setMaskDraft} />
              </div>
            ) : (
              <label
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  onPickInitImage(e.dataTransfer.files?.[0]);
                }}
                className={`mt-1.5 flex cursor-pointer items-center justify-center rounded-md border border-dashed border-white/15 px-3 py-4 text-center text-xs text-white/45 transition hover:border-white/30 hover:text-white/70 ${
                  uploadBusy ? "pointer-events-none opacity-50" : ""
                }`}
              >
                {uploadBusy ? "uploading…" : "drop or click to add a source image"}
                <input
                  type="file"
                  accept="image/*"
                  className="hidden"
                  disabled={uploadBusy}
                  onChange={(e) => {
                    onPickInitImage(e.target.files?.[0]);
                    e.target.value = "";
                  }}
                />
              </label>
            )}
            {uploadError ? <Notice tone="amber">{uploadError}</Notice> : null}
          </section>
        ) : null}

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
                    active ? "border-accent/70 bg-accent/20 text-white" : "border-white/15 text-white/60 hover:bg-white/10"
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
          {lorasLoading && selectedImgModel && compatibleLoras.length === 0 ? (
            <div className="mt-1.5">
              <SkeletonRows rows={3} />
            </div>
          ) : compatibleLoras.length ? (
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
            {presetsLoading && imagePresets.length === 0 ? (
              <SkeletonLine className="h-9 w-full rounded-md" />
            ) : (
              <Select value={presetId} options={presetOptions} onChange={setPresetId} placeholder="unsaved" />
            )}
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
              className="mt-1 w-full rounded-md border border-white/10 bg-black/30 px-2 py-2 text-sm outline-none focus:border-accent"
            />
          </label>
          <button
            onClick={generate}
            disabled={!canQueue}
            className="mt-4 rounded-md bg-accent px-4 py-2 text-sm font-semibold text-white transition hover:bg-accent-hover disabled:opacity-40"
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
      enabled ? "border-accent/45 bg-accent/10" : "border-white/10 bg-black/20"
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
        className="mt-1 w-full rounded-md border border-white/10 bg-black/30 px-2 py-1.5 text-sm outline-none focus:border-accent"
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
