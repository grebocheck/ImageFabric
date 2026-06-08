import { afterEach, describe, expect, it } from "vitest";

import {
  formatSize,
  formatVram,
  imageFamilyDefaults,
  imageModelRank,
  isKnownGuidanceDefault,
  isKnownSizeDefault,
  isKnownStepDefault,
  isLoraCompatible,
  isNunchaku,
  numberParam,
  pickDefaultImageModel,
  readSaved,
  STORE_KEY,
} from "./imageComposerHelpers";
import type { Lora, Model } from "../types";

function model(over: Partial<Model> = {}): Model {
  return {
    id: "m", name: "M", family: "sdxl", job_type: "image",
    size_bytes: 0, loaded: false, warm: false, slow: false, ...over,
  } as Model;
}

describe("formatters", () => {
  it("formatSize switches MB/GB and blanks zero", () => {
    expect(formatSize(0)).toBe("");
    expect(formatSize(5 * 1024 ** 2)).toBe("5 MB");
    expect(formatSize(2 * 1024 ** 3)).toBe("2.0 GB");
  });

  it("formatVram marks slow models with >=", () => {
    expect(formatVram(model({ estimated_vram_gb: 9.8 }))).toBe("~9.8 GB");
    expect(formatVram(model({ estimated_vram_gb: 16, slow: true }))).toBe(">=16.0 GB");
    expect(formatVram(model({}))).toBe("");
  });

  it("numberParam falls back on non-finite input", () => {
    expect(numberParam("3", 0)).toBe(3);
    expect(numberParam("nope", 7)).toBe(7);
    expect(numberParam(undefined, 1)).toBe(1);
  });
});

describe("model ranking & selection", () => {
  it("isNunchaku detects the quant prefix", () => {
    expect(isNunchaku(model({ quant: "nunchaku-fp4" }))).toBe(true);
    expect(isNunchaku(model({ quant: "fp16" }))).toBe(false);
    expect(isNunchaku(undefined)).toBe(false);
  });

  it("imageModelRank orders flux2-nunchaku first and slow last", () => {
    expect(imageModelRank(model({ family: "flux2", quant: "nunchaku-fp4" }))).toBe(-1);
    expect(imageModelRank(model({ family: "flux2" }))).toBe(0);
    expect(imageModelRank(model({ family: "flux", quant: "nunchaku-fp4" }))).toBe(0);
    expect(imageModelRank(model({ family: "z-image" }))).toBe(0);
    expect(imageModelRank(model({ family: "qwen-image" }))).toBe(1);
    expect(imageModelRank(model({ family: "sdxl" }))).toBe(1);
    expect(imageModelRank(model({ family: "sdxl", slow: true }))).toBe(2);
  });

  it("pickDefaultImageModel prefers flux-nunchaku, then non-slow, then first", () => {
    const fluxNun = model({ id: "fn", family: "flux", quant: "nunchaku-fp4" });
    const slow = model({ id: "s", slow: true });
    expect(pickDefaultImageModel([slow, fluxNun])?.id).toBe("fn");
    expect(pickDefaultImageModel([slow, model({ id: "ok" })])?.id).toBe("ok");
    expect(pickDefaultImageModel([slow])?.id).toBe("s");
  });

  it("isLoraCompatible matches family or passes when unconstrained", () => {
    const sdxlLora = { id: "l", name: "L", family: "sdxl" } as Lora;
    const anyLora = { id: "a", name: "A", family: null } as unknown as Lora;
    expect(isLoraCompatible(sdxlLora, model({ family: "sdxl" }))).toBe(true);
    expect(isLoraCompatible(sdxlLora, model({ family: "flux" }))).toBe(false);
    expect(isLoraCompatible(anyLora, model({ family: "flux" }))).toBe(true);
    expect(isLoraCompatible(sdxlLora, undefined)).toBe(true);
  });

  it("exposes family defaults for Qwen and Z-Image", () => {
    expect(imageFamilyDefaults("qwen-image")).toMatchObject({ steps: 50, guidance: 4, width: 1328 });
    expect(imageFamilyDefaults("z-image")).toMatchObject({ steps: 9, guidance: 0, width: 1024 });
    expect(imageFamilyDefaults("sdxl")).toBeUndefined();
    expect(isKnownStepDefault(9)).toBe(true);
    expect(isKnownGuidanceDefault(0)).toBe(true);
    expect(isKnownSizeDefault(1328)).toBe(true);
  });
});

describe("readSaved", () => {
  afterEach(() => localStorage.clear());

  it("returns {} when absent or corrupt", () => {
    expect(readSaved()).toEqual({});
    localStorage.setItem(STORE_KEY, "{bad");
    expect(readSaved()).toEqual({});
  });

  it("round-trips a saved composer snapshot", () => {
    localStorage.setItem(STORE_KEY, JSON.stringify({ imgModel: "x", steps: 12, count: 3 }));
    expect(readSaved()).toMatchObject({ imgModel: "x", steps: 12, count: 3 });
  });
});
