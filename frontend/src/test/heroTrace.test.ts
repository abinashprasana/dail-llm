import { describe, expect, it } from "vitest";
import {
  getTraceStageOffsets,
  TRACE_CAPTIONS,
  TRACE_SEQUENCE_DURATION_MS,
  TRACE_START_DELAY_MS,
  TRACE_STEPS,
} from "../components/heroTrace";

describe("shared parliamentary trace contract", () => {
  it("keeps the agreed stage order and timings", () => {
    expect(TRACE_START_DELAY_MS).toBe(250);
    expect(TRACE_STEPS.map(({ stage, durationMs }) => ({ stage, durationMs }))).toEqual([
      { stage: "speaker", durationMs: 500 },
      { stage: "attention", durationMs: 900 },
      { stage: "prediction", durationMs: 700 },
    ]);
    expect(getTraceStageOffsets()).toEqual([
      { stage: "speaker", atMs: 250 },
      { stage: "attention", atMs: 750 },
      { stage: "prediction", atMs: 1_650 },
    ]);
    expect(TRACE_SEQUENCE_DURATION_MS).toBe(2_350);
  });

  it("provides copy for every state from the same source of truth", () => {
    expect(TRACE_CAPTIONS.idle).toBeTruthy();
    for (const step of TRACE_STEPS) {
      expect(step.index).toMatch(/^0[1-3]$/);
      expect(step.label).toBeTruthy();
      expect(TRACE_CAPTIONS[step.stage]).toBe(step.caption);
    }
  });
});
