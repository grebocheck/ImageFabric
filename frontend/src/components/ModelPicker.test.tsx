import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ModelPicker } from "./ModelPicker";
import type { Model } from "../types";

afterEach(cleanup);

function model(over: Partial<Model> = {}): Model {
  return {
    id: "m", name: "M", family: "sdxl", job_type: "image",
    size_bytes: 0, loaded: false, warm: false, slow: false, ...over,
  } as Model;
}

const MODELS = [
  model({ id: "sdxl", name: "SDXL base", family: "sdxl", estimated_vram_gb: 11 }),
  model({ id: "flux", name: "FLUX dev", family: "flux", quant: "nunchaku-fp4", estimated_vram_gb: 9.8 }),
];

describe("ModelPicker", () => {
  it("shows the selected model name in the trigger", () => {
    render(<ModelPicker models={MODELS} value="flux" onChange={() => {}} />);
    expect(screen.getByText("FLUX dev")).toBeTruthy();
  });

  it("renders styled options with family + fast-path badges and VRAM", async () => {
    const user = userEvent.setup();
    render(<ModelPicker models={MODELS} value="" onChange={() => {}} />);
    await user.click(screen.getByRole("button"));
    // family badge + fast-path badge for the nunchaku model + measured VRAM
    expect(screen.getByText("flux")).toBeTruthy();
    expect(screen.getByText("fast")).toBeTruthy();
    expect(screen.getByText("~9.8 GB")).toBeTruthy();
  });

  it("calls onChange with the chosen model id", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<ModelPicker models={MODELS} value="" onChange={onChange} />);
    await user.click(screen.getByRole("button"));
    await user.click(screen.getByText("SDXL base"));
    expect(onChange).toHaveBeenCalledWith("sdxl");
  });

  it("badges models recommended for the current hardware", async () => {
    const user = userEvent.setup();
    render(<ModelPicker
      models={[model({ id: "sdxl", name: "SDXL base", family: "sdxl", recommendation: "recommended" })]}
      value=""
      onChange={() => {}}
    />);
    await user.click(screen.getByRole("button"));
    expect(screen.getByText("recommended")).toBeTruthy();
  });

  it("renders unavailable models as disabled options", async () => {
    const user = userEvent.setup();
    render(<ModelPicker
      models={[...MODELS, model({
        id: "off",
        name: "Disabled model",
        available: false,
        unavailable_reason: "needs more VRAM",
      })]}
      value=""
      onChange={() => {}}
    />);
    await user.click(screen.getByRole("button"));
    const disabled = screen.getByText("Disabled model").closest("button") as HTMLButtonElement | null;
    expect(disabled?.disabled).toBe(true);
    expect(screen.getByText("disabled")).toBeTruthy();
    expect(screen.getByText("needs more VRAM")).toBeTruthy();
  });
});
