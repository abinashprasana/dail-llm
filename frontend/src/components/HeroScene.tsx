import { Component, lazy, Suspense, useCallback, useEffect, useRef, useState } from "react";
import type { ComponentType, ReactNode } from "react";
import { useReducedMotion } from "motion/react";
import { SceneFallback } from "./SceneFallback";
import { TraceRail } from "./TraceRail";
import {
  getTraceStageOffsets,
  TRACE_SEQUENCE_DURATION_MS,
} from "./heroTrace";
import type { TraceMotion, TraceStage } from "./heroTrace";

interface ChamberCanvasProps {
  active: boolean;
  traceStage: TraceStage;
  traceRunId: number;
  traceMotion: TraceMotion;
  onReady?: () => void;
  onAssembled?: () => void;
  onError?: () => void;
}

const ChamberCanvas = lazy(async () => {
  const module = await import("./ChamberCanvas");
  return { default: module.default as ComponentType<ChamberCanvasProps> };
});

const MIN_VIEWPORT_WIDTH = 800;
const MIN_CPU_CORES = 4;
const MIN_DEVICE_MEMORY_GB = 4;
const IDLE_START_DELAY_MS = 250;
const IDLE_DEADLINE_MS = 1200;
const INTERSECTION_THRESHOLD = 0.05;
const CANVAS_FADE_MS = 1000;

type SceneStatus = "poster" | "loading" | "ready" | "failed";
type SceneMode = "checking" | "poster" | "webgl";

type NavigatorWithHints = Navigator & {
  connection?: {
    saveData?: boolean;
    addEventListener?: (type: "change", listener: () => void) => void;
    removeEventListener?: (type: "change", listener: () => void) => void;
  };
  deviceMemory?: number;
};

interface SceneErrorBoundaryProps {
  children: ReactNode;
  onError: () => void;
}

class SceneErrorBoundary extends Component<SceneErrorBoundaryProps, { failed: boolean }> {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  componentDidCatch() {
    this.props.onError();
  }

  render() {
    return this.state.failed ? null : this.props.children;
  }
}

function canUseWebGL(): boolean {
  try {
    const canvas = document.createElement("canvas");
    const context = canvas.getContext("webgl2") || canvas.getContext("webgl");
    if (!context) return false;
    context.getExtension?.("WEBGL_lose_context")?.loseContext();
    return true;
  } catch {
    return false;
  }
}

function prefersDataSaving(): boolean {
  return Boolean((navigator as NavigatorWithHints).connection?.saveData);
}

export function HeroScene() {
  const reducedMotion = Boolean(useReducedMotion());
  const traceMotion: TraceMotion = reducedMotion ? "static" : "animate";
  const container = useRef<HTMLDivElement>(null);
  const readyTimer = useRef<ReturnType<typeof globalThis.setTimeout> | null>(null);
  const traceTimers = useRef<Array<ReturnType<typeof globalThis.setTimeout>>>([]);
  const lockedStageRef = useRef<TraceStage>("idle");
  const autoplayedRef = useRef(false);
  const failureLatched = useRef(false);

  const [enabled, setEnabled] = useState(false);
  const [viewportCapable, setViewportCapable] = useState(() =>
    window.matchMedia(`(min-width: ${MIN_VIEWPORT_WIDTH}px)`).matches,
  );
  const [dataSaving, setDataSaving] = useState(prefersDataSaving);
  const [intersecting, setIntersecting] = useState(true);
  const [documentVisible, setDocumentVisible] = useState(() => document.visibilityState !== "hidden");
  const [sceneStatus, setSceneStatus] = useState<SceneStatus>("poster");
  const [sceneMode, setSceneMode] = useState<SceneMode>("checking");
  const [assembled, setAssembled] = useState(false);
  const [traceStage, setTraceStage] = useState<TraceStage>("idle");
  const [lockedStage, setLockedStage] = useState<TraceStage>("idle");
  const [traceRunning, setTraceRunning] = useState(false);
  const [traceRunId, setTraceRunId] = useState(0);

  const clearReadyTimer = useCallback(() => {
    if (readyTimer.current === null) return;
    globalThis.clearTimeout(readyTimer.current);
    readyTimer.current = null;
  }, []);

  const clearTraceTimers = useCallback(() => {
    traceTimers.current.forEach((timer) => globalThis.clearTimeout(timer));
    traceTimers.current = [];
  }, []);

  const cancelTraceSequence = useCallback((restoreLockedStage = true) => {
    clearTraceTimers();
    setTraceRunning(false);
    if (restoreLockedStage) setTraceStage(lockedStageRef.current);
  }, [clearTraceTimers]);

  const runTraceSequence = useCallback(() => {
    clearTraceTimers();
    setTraceRunning(true);
    setTraceStage("idle");
    setTraceRunId((current) => current + 1);

    getTraceStageOffsets().forEach(({ stage, atMs }) => {
      traceTimers.current.push(globalThis.setTimeout(() => setTraceStage(stage), atMs));
    });

    traceTimers.current.push(globalThis.setTimeout(() => {
      lockedStageRef.current = "prediction";
      setLockedStage("prediction");
      setTraceStage("prediction");
      setTraceRunning(false);
      traceTimers.current = [];
    }, TRACE_SEQUENCE_DURATION_MS));
  }, [clearTraceTimers]);

  const startAutoplayOnce = useCallback(() => {
    if (reducedMotion || autoplayedRef.current) return;
    autoplayedRef.current = true;
    runTraceSequence();
  }, [reducedMotion, runTraceSequence]);

  const handleSceneFailure = useCallback(() => {
    if (!failureLatched.current) failureLatched.current = true;
    clearReadyTimer();
    setAssembled(false);
    setSceneMode("poster");
    setSceneStatus("failed");
    setEnabled(false);
  }, [clearReadyTimer]);

  const handleCanvasCreated = useCallback(() => {
    if (failureLatched.current) return;
    clearReadyTimer();
    setSceneStatus("loading");
    readyTimer.current = globalThis.setTimeout(() => {
      readyTimer.current = null;
      if (!failureLatched.current) setSceneStatus("ready");
    }, CANVAS_FADE_MS);
  }, [clearReadyTimer]);

  const handleCanvasAssembled = useCallback(() => {
    if (!failureLatched.current) setAssembled(true);
  }, []);

  const handlePreview = useCallback((stage: TraceStage) => {
    autoplayedRef.current = true;
    cancelTraceSequence(false);
    setTraceStage(stage);
  }, [cancelTraceSequence]);

  const handlePreviewEnd = useCallback(() => {
    cancelTraceSequence(true);
  }, [cancelTraceSequence]);

  const handleSelectStage = useCallback((stage: TraceStage) => {
    autoplayedRef.current = true;
    cancelTraceSequence(false);
    lockedStageRef.current = stage;
    setLockedStage(stage);
    setTraceStage(stage);
  }, [cancelTraceSequence]);

  const handleReplay = useCallback(() => {
    autoplayedRef.current = true;
    cancelTraceSequence(false);
    runTraceSequence();
  }, [cancelTraceSequence, runTraceSequence]);

  useEffect(() => () => {
    clearReadyTimer();
    clearTraceTimers();
  }, [clearReadyTimer, clearTraceTimers]);

  useEffect(() => {
    const media = window.matchMedia(`(min-width: ${MIN_VIEWPORT_WIDTH}px)`);
    const updateViewportCapability = () => setViewportCapable(media.matches);
    updateViewportCapability();
    if (typeof media.addEventListener === "function") {
      media.addEventListener("change", updateViewportCapability);
      return () => media.removeEventListener("change", updateViewportCapability);
    }
    media.addListener(updateViewportCapability);
    return () => media.removeListener(updateViewportCapability);
  }, []);

  useEffect(() => {
    const connection = (navigator as NavigatorWithHints).connection;
    const updateDataSaving = () => setDataSaving(Boolean(connection?.saveData));
    updateDataSaving();
    connection?.addEventListener?.("change", updateDataSaving);
    return () => connection?.removeEventListener?.("change", updateDataSaving);
  }, []);

  useEffect(() => {
    if (!reducedMotion) return;
    autoplayedRef.current = true;
    cancelTraceSequence(true);
  }, [cancelTraceSequence, reducedMotion]);

  useEffect(() => {
    clearReadyTimer();
    setEnabled(false);
    setAssembled(false);

    if (failureLatched.current) {
      setSceneMode("poster");
      setSceneStatus("failed");
      return;
    }

    setSceneStatus("poster");
    const browserNavigator = navigator as NavigatorWithHints;
    const memory = browserNavigator.deviceMemory ?? 8;
    const cpuCores = browserNavigator.hardwareConcurrency || 8;
    const capable =
      !reducedMotion &&
      !dataSaving &&
      viewportCapable &&
      cpuCores >= MIN_CPU_CORES &&
      memory >= MIN_DEVICE_MEMORY_GB &&
      canUseWebGL();

    if (!capable) {
      setSceneMode("poster");
      return;
    }

    setSceneMode("webgl");
    let settled = false;
    let idleHandle: number | null = null;
    let deadlineHandle: ReturnType<typeof globalThis.setTimeout> | null = null;

    const enableScene = () => {
      if (settled || failureLatched.current) return;
      settled = true;
      setSceneStatus("loading");
      setEnabled(true);
    };

    if ("requestIdleCallback" in window) {
      idleHandle = window.requestIdleCallback(enableScene, { timeout: IDLE_DEADLINE_MS });
      deadlineHandle = globalThis.setTimeout(enableScene, IDLE_DEADLINE_MS);
    } else {
      deadlineHandle = globalThis.setTimeout(enableScene, IDLE_START_DELAY_MS);
    }

    return () => {
      settled = true;
      if (idleHandle !== null) window.cancelIdleCallback(idleHandle);
      if (deadlineHandle !== null) globalThis.clearTimeout(deadlineHandle);
    };
  }, [clearReadyTimer, dataSaving, reducedMotion, viewportCapable]);

  useEffect(() => {
    const node = container.current;
    if (!node || !("IntersectionObserver" in window)) return;

    const observer = new IntersectionObserver(
      ([entry]) => setIntersecting(entry?.isIntersecting ?? true),
      { threshold: INTERSECTION_THRESHOLD },
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const updateVisibility = () => setDocumentVisible(document.visibilityState !== "hidden");
    updateVisibility();
    document.addEventListener("visibilitychange", updateVisibility);
    return () => document.removeEventListener("visibilitychange", updateVisibility);
  }, []);

  useEffect(() => {
    const node = container.current;
    if (!node || !enabled) return;

    const attachedCanvases = new Set<HTMLCanvasElement>();
    const onContextLost = (event: Event) => {
      event.preventDefault();
      handleSceneFailure();
    };

    const attachToCanvas = (canvas: HTMLCanvasElement) => {
      if (attachedCanvases.has(canvas)) return;
      attachedCanvases.add(canvas);
      canvas.setAttribute("aria-hidden", "true");
      canvas.setAttribute("role", "presentation");
      canvas.tabIndex = -1;
      canvas.addEventListener("webglcontextlost", onContextLost);
    };

    const scanForCanvas = () => node.querySelectorAll("canvas").forEach(attachToCanvas);
    scanForCanvas();
    const observer = new MutationObserver(scanForCanvas);
    observer.observe(node, { childList: true, subtree: true });

    return () => {
      observer.disconnect();
      attachedCanvases.forEach((canvas) => canvas.removeEventListener("webglcontextlost", onContextLost));
    };
  }, [enabled, handleSceneFailure]);

  const active = intersecting && documentVisible;

  useEffect(() => {
    if (!active) {
      if (traceRunning) cancelTraceSequence(true);
      return;
    }

    const visualReady = sceneMode === "poster" || (sceneMode === "webgl" && assembled);
    if (visualReady) startAutoplayOnce();
  }, [active, assembled, cancelTraceSequence, sceneMode, startAutoplayOnce, traceRunning]);

  const posterVisible = sceneStatus !== "ready";

  return (
    <div
      className="hero-scene-shell"
      data-trace-stage={traceStage}
      data-trace-running={traceRunning ? "true" : "false"}
      data-trace-run-id={traceRunId}
      data-trace-motion={traceMotion}
    >
      <div
        className={`hero-scene${sceneStatus === "ready" ? " is-canvas-ready" : ""}`}
        ref={container}
        role="img"
        aria-label="A near-isometric parliamentary chamber with tiered benches, a central speaking floor, and character tokens tracing model attention"
        data-scene-status={sceneStatus}
        data-trace-stage={traceStage}
      >
        <SceneFallback
          visible={posterVisible}
          traceStage={traceStage}
          traceMotion={traceMotion}
          traceRunId={traceRunId}
          running={traceRunning}
        />
        {enabled && (
          <SceneErrorBoundary onError={handleSceneFailure}>
            <Suspense fallback={null}>
              <ChamberCanvas
                active={active}
                traceStage={traceStage}
                traceRunId={traceRunId}
                traceMotion={traceMotion}
                onReady={handleCanvasCreated}
                onAssembled={handleCanvasAssembled}
                onError={handleSceneFailure}
              />
            </Suspense>
          </SceneErrorBoundary>
        )}
      </div>

      <TraceRail
        stage={traceStage}
        lockedStage={lockedStage}
        running={traceRunning}
        runId={traceRunId}
        onPreview={handlePreview}
        onPreviewEnd={handlePreviewEnd}
        onSelect={handleSelectStage}
        onReplay={handleReplay}
      />
    </div>
  );
}
