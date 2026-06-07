import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { Select, type SelectOption } from "./Select";

afterEach(cleanup);

// 7+ options makes the shared Select switch on its in-dropdown search field.
const OPTIONS: SelectOption[] = [
  { value: "flux", label: "FLUX dev" },
  { value: "flux2", label: "FLUX.2 klein" },
  { value: "sdxl", label: "SDXL base" },
  { value: "sdxl-turbo", label: "SDXL turbo" },
  { value: "gpt", label: "gpt-oss 20B" },
  { value: "qwen", label: "Qwen 7B" },
  { value: "gemma", label: "Gemma 12B" },
];

function setup(value = "") {
  const onChange = vi.fn();
  render(<Select value={value} options={OPTIONS} onChange={onChange} />);
  return { onChange };
}

describe("Select", () => {
  it("shows the placeholder when nothing is selected", () => {
    setup();
    expect(screen.getByText("select...")).toBeTruthy();
  });

  it("renders the selected option's label", () => {
    setup("sdxl");
    expect(screen.getByText("SDXL base")).toBeTruthy();
  });

  it("opens on click and lists the options", async () => {
    const user = userEvent.setup();
    setup();
    await user.click(screen.getByRole("button"));
    expect(screen.getByText("FLUX dev")).toBeTruthy();
    expect(screen.getByText("Gemma 12B")).toBeTruthy();
  });

  it("filters options by the search query", async () => {
    const user = userEvent.setup();
    setup();
    await user.click(screen.getByRole("button"));
    await user.type(screen.getByPlaceholderText("search..."), "sdxl");
    expect(screen.getByText("SDXL base")).toBeTruthy();
    expect(screen.getByText("SDXL turbo")).toBeTruthy();
    expect(screen.queryByText("FLUX dev")).toBeNull();
  });

  it("calls onChange with the chosen value and closes", async () => {
    const user = userEvent.setup();
    const { onChange } = setup();
    await user.click(screen.getByRole("button"));
    await user.click(screen.getByText("Qwen 7B"));
    expect(onChange).toHaveBeenCalledWith("qwen");
    expect(screen.queryByPlaceholderText("search...")).toBeNull();
  });

  it("shows a no-options state when the filter matches nothing", async () => {
    const user = userEvent.setup();
    setup();
    await user.click(screen.getByRole("button"));
    await user.type(screen.getByPlaceholderText("search..."), "zzz");
    expect(screen.getByText("no options")).toBeTruthy();
  });
});
