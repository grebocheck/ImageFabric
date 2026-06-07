import { describe, expect, it } from "vitest";

import { splitReasoning } from "./Thinking";

// splitReasoning is the frontend mirror of the backend's `_strip_reasoning`: it
// pulls <think>…</think> out of a streamed reply so the Thinking panel can show
// the reasoning separately. These cover the streaming states the chat relies on.
describe("splitReasoning", () => {
  it("returns plain content when there is no think block", () => {
    const r = splitReasoning("just an answer");
    expect(r.reasoning).toBeNull();
    expect(r.answer).toBe("just an answer");
    expect(r.active).toBe(false);
  });

  it("splits a completed think block from the answer", () => {
    const r = splitReasoning("<think>weighing options</think>The answer.");
    expect(r.reasoning).toBe("weighing options");
    expect(r.answer).toBe("The answer.");
    expect(r.active).toBe(false);
  });

  it("marks an unclosed block as actively reasoning", () => {
    const r = splitReasoning("<think>still going");
    expect(r.reasoning).toBe("still going");
    expect(r.answer).toBe("");
    expect(r.active).toBe(true);
  });

  it("handles the <thinking> alias case-insensitively", () => {
    const r = splitReasoning("<THINKING>hmm</Thinking>done");
    expect(r.reasoning).toBe("hmm");
    expect(r.answer).toBe("done");
  });

  it("stitches text before and after the block into the answer", () => {
    const r = splitReasoning("Intro <think>aside</think>outro");
    expect(r.reasoning).toBe("aside");
    expect(r.answer).toBe("Intro outro");
  });
});
