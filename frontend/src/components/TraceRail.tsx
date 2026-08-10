import { useRef } from "react";
import { RotateCcw } from "lucide-react";
import { TRACE_CAPTIONS, TRACE_STEPS } from "./heroTrace";
import type { TraceStage } from "./heroTrace";

interface TraceRailProps {
  stage: TraceStage;
  lockedStage: TraceStage;
  running: boolean;
  runId: number;
  onPreview: (stage: TraceStage) => void;
  onPreviewEnd: () => void;
  onSelect: (stage: TraceStage) => void;
  onReplay: () => void;
}

export function TraceRail({
  stage,
  lockedStage,
  running,
  runId,
  onPreview,
  onPreviewEnd,
  onSelect,
  onReplay,
}: TraceRailProps) {
  const hoveredStage = useRef<TraceStage | null>(null);
  const focusedStage = useRef<TraceStage | null>(null);

  const restoreRemainingInteraction = () => {
    const remainingStage = focusedStage.current ?? hoveredStage.current;
    if (remainingStage) onPreview(remainingStage);
    else onPreviewEnd();
  };

  return (
    <div
      className="trace-rail"
      data-trace-stage={stage}
      data-trace-running={running ? "true" : "false"}
      data-trace-run-id={runId}
    >
      <p className="trace-rail-heading">Trace a prediction</p>

      <div className="trace-stage-list" role="group" aria-label="Prediction trace stages" aria-describedby="trace-caption">
        {TRACE_STEPS.map((step) => (
          <button
            className={`trace-stage-button${stage === step.stage ? " is-active" : ""}`}
            type="button"
            key={step.stage}
            data-trace-stage={step.stage}
            aria-pressed={lockedStage === step.stage}
            onPointerEnter={(event) => {
              if (event.pointerType !== "mouse" && event.pointerType !== "pen") return;
              hoveredStage.current = step.stage;
              onPreview(step.stage);
            }}
            onPointerLeave={(event) => {
              if (event.pointerType !== "mouse" && event.pointerType !== "pen") return;
              if (hoveredStage.current === step.stage) hoveredStage.current = null;
              restoreRemainingInteraction();
            }}
            onFocus={() => {
              focusedStage.current = step.stage;
              onPreview(step.stage);
            }}
            onBlur={() => {
              if (focusedStage.current === step.stage) focusedStage.current = null;
              restoreRemainingInteraction();
            }}
            onClick={() => onSelect(step.stage)}
          >
            <span className="trace-stage-index" aria-hidden="true">{step.index}</span>
            <span>{step.label}</span>
          </button>
        ))}
      </div>

      <button className="trace-replay" type="button" onClick={onReplay} aria-label="Replay trace">
        <RotateCcw size={15} aria-hidden="true" />
        <span
          style={{
            position: "absolute",
            width: 1,
            height: 1,
            padding: 0,
            margin: -1,
            overflow: "hidden",
            clip: "rect(0, 0, 0, 0)",
            whiteSpace: "nowrap",
            border: 0,
          }}
        >
          Replay trace
        </span>
      </button>

      <p
        className="trace-caption"
        id="trace-caption"
        role="status"
        aria-live="polite"
        aria-atomic="true"
      >
        {TRACE_CAPTIONS[stage]}
      </p>
    </div>
  );
}
