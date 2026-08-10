export type TraceStage = "idle" | "speaker" | "attention" | "prediction";

export type TraceMotion = "static" | "animate";

export interface TraceStep {
  stage: Exclude<TraceStage, "idle">;
  index: string;
  label: string;
  caption: string;
  durationMs: number;
}

export const TRACE_START_DELAY_MS = 250;

export const TRACE_STEPS: readonly TraceStep[] = [
  {
    stage: "speaker",
    index: "01",
    label: "Speaker",
    caption: "A character enters the model’s visible context.",
    durationMs: 500,
  },
  {
    stage: "attention",
    index: "02",
    label: "Eight heads",
    caption: "Eight attention heads compare the characters already in view.",
    durationMs: 900,
  },
  {
    stage: "prediction",
    index: "03",
    label: "Next character",
    caption: "The model resolves its next-character estimate.",
    durationMs: 700,
  },
] as const;

export const TRACE_SEQUENCE_DURATION_MS =
  TRACE_START_DELAY_MS + TRACE_STEPS.reduce((total, step) => total + step.durationMs, 0);

export const TRACE_CAPTIONS: Readonly<Record<TraceStage, string>> = {
  idle: "Follow one character through the model.",
  speaker: TRACE_STEPS[0].caption,
  attention: TRACE_STEPS[1].caption,
  prediction: TRACE_STEPS[2].caption,
};

export function getTraceStageOffsets(): ReadonlyArray<{ stage: TraceStep["stage"]; atMs: number }> {
  let atMs = TRACE_START_DELAY_MS;
  return TRACE_STEPS.map((step) => {
    const offset = { stage: step.stage, atMs };
    atMs += step.durationMs;
    return offset;
  });
}
