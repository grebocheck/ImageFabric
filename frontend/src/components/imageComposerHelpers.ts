// Pure, view-agnostic helpers extracted from ImageComposer (P11.1): persisted
// composer state, model ranking / default selection, LoRA compatibility, and
// small formatters. Side-effect-free (beyond localStorage) so they unit-test
// without rendering the composer.

import type { Lora, Model } from "../types";

export const STORE_KEY = "hfabric.image.composer";
export const PROMPT_HISTORY_KEY = "hfabric.image.promptHistory";
export const promptHistoryLimit = 14;

export const DEFAULT_STEPS = 28;
export const DEFAULT_GUIDANCE = 3.5;
export const DEFAULT_SIZE = 1024;
export const FLUX2_STEPS = 6;
export const FLUX2_GUIDANCE = 4.0;
export const FLUX2_SIZE = 768;
export const QWEN_IMAGE_STEPS = 50;
export const QWEN_IMAGE_GUIDANCE = 4.0;
export const QWEN_IMAGE_SIZE = 1328;
export const Z_IMAGE_STEPS = 9;
export const Z_IMAGE_GUIDANCE = 0.0;
export const Z_IMAGE_SIZE = 1024;

export type ImageFamilyDefaults = { steps: number; guidance: number; width: number; height: number };

export function imageFamilyDefaults(family: string | undefined): ImageFamilyDefaults | undefined {
  if (family === "flux2") {
    return { steps: FLUX2_STEPS, guidance: FLUX2_GUIDANCE, width: FLUX2_SIZE, height: FLUX2_SIZE };
  }
  if (family === "qwen-image") {
    return { steps: QWEN_IMAGE_STEPS, guidance: QWEN_IMAGE_GUIDANCE, width: QWEN_IMAGE_SIZE, height: QWEN_IMAGE_SIZE };
  }
  if (family === "z-image") {
    return { steps: Z_IMAGE_STEPS, guidance: Z_IMAGE_GUIDANCE, width: Z_IMAGE_SIZE, height: Z_IMAGE_SIZE };
  }
  return undefined;
}

const knownStepDefaults = [DEFAULT_STEPS, FLUX2_STEPS, QWEN_IMAGE_STEPS, Z_IMAGE_STEPS];
const knownGuidanceDefaults = [DEFAULT_GUIDANCE, FLUX2_GUIDANCE, QWEN_IMAGE_GUIDANCE, Z_IMAGE_GUIDANCE];
const knownSizeDefaults = [DEFAULT_SIZE, FLUX2_SIZE, QWEN_IMAGE_SIZE, Z_IMAGE_SIZE];

export const isKnownStepDefault = (value: number): boolean => knownStepDefaults.includes(value);
export const isKnownGuidanceDefault = (value: number): boolean => knownGuidanceDefaults.includes(value);
export const isKnownSizeDefault = (value: number): boolean => knownSizeDefaults.includes(value);

export type LoraSelection = { id: string; weight: number };
export type SavedComposer = {
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

export function readSaved(): SavedComposer {
  try {
    const raw = localStorage.getItem(STORE_KEY);
    return raw ? (JSON.parse(raw) as SavedComposer) : {};
  } catch {
    return {};
  }
}

export function loadPromptHistory(): string[] {
  try {
    const parsed = JSON.parse(localStorage.getItem(PROMPT_HISTORY_KEY) ?? "[]");
    return Array.isArray(parsed)
      ? parsed.filter((x): x is string => typeof x === "string").slice(0, promptHistoryLimit)
      : [];
  } catch {
    return [];
  }
}

export function isNunchaku(model: Model | undefined): boolean {
  return Boolean(model?.quant?.startsWith("nunchaku"));
}

export function isLoraCompatible(lora: Lora, model: Model | undefined): boolean {
  return !model || !lora.family || lora.family === model.family;
}

export function formatSize(bytes: number): string {
  if (!bytes) return "";
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
  return `${Math.max(1, Math.round(bytes / 1024 ** 2))} MB`;
}

export function formatVram(model: Model): string {
  if (!model.estimated_vram_gb) return "";
  const prefix = model.slow ? ">=" : "~";
  return `${prefix}${model.estimated_vram_gb.toFixed(1)} GB`;
}

export function familyColor(family: string): string {
  if (family === "flux2") return "bg-sky-700/50 text-sky-100";
  if (family === "qwen-image") return "bg-violet-700/50 text-violet-100";
  if (family === "z-image") return "bg-cyan-700/50 text-cyan-100";
  if (family === "flux") return "bg-accent/55 text-accent-fg";
  if (family === "sdxl") return "bg-emerald-700/55 text-emerald-100";
  if (family === "gguf") return "bg-amber-700/50 text-amber-100";
  return "bg-white/10 text-white/65";
}

export function numberParam(value: unknown, fallback: number): number {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

export function imageModelRank(model: Model): number {
  if (model.family === "flux2" && isNunchaku(model)) return -1;
  if (model.family === "flux2") return 0;
  if (model.family === "flux" && isNunchaku(model)) return 0;
  if (model.family === "z-image") return 0;
  if (model.family === "qwen-image") return 1;
  if (!model.slow) return 1;
  return 2;
}

export function pickDefaultImageModel(models: Model[]): Model | undefined {
  return models.find((m) => m.family === "flux" && isNunchaku(m))
    ?? models.find((m) => m.family === "z-image")
    ?? models.find((m) => !m.slow)
    ?? models[0];
}
