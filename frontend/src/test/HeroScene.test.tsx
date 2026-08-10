import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { TraceMotion, TraceStage } from "../components/heroTrace";
import {
  TRACE_CAPTIONS,
  TRACE_SEQUENCE_DURATION_MS,
  TRACE_START_DELAY_MS,
  TRACE_STEPS,
} from "../components/heroTrace";
import { HeroScene } from "../components/HeroScene";

interface MockCanvasProps {
  active: boolean;
  traceStage: TraceStage;
  traceRunId: number;
  traceMotion: TraceMotion;
  onReady?: () => void;
  onAssembled?: () => void;
  onError?: () => void;
}

const sceneState = vi.hoisted(() => ({
  reducedMotion: false,
  reducedSubscribers: new Set<(value: boolean) => void>(),
  throwOnRender: false,
  autoAssemble: true,
  canvasMounts: 0,
  latestCanvasProps: null as MockCanvasProps | null,
}));

vi.mock("motion/react", async () => {
  const actual = await vi.importActual<typeof import("motion/react")>("motion/react");
  const React = await import("react");

  return {
    ...actual,
    useReducedMotion: () => {
      const [reduced, setReduced] = React.useState(sceneState.reducedMotion);
      React.useEffect(() => {
        const subscriber = (value: boolean) => setReduced(value);
        sceneState.reducedSubscribers.add(subscriber);
        return () => { sceneState.reducedSubscribers.delete(subscriber); };
      }, []);
      return reduced;
    },
  };
});

vi.mock("../components/ChamberCanvas", async () => {
  const React = await import("react");

  return {
    default: function MockChamberCanvas(props: MockCanvasProps) {
      const { active, traceStage, traceRunId, traceMotion, onReady, onAssembled } = props;
      const announced = React.useRef(false);
      sceneState.latestCanvasProps = props;

      React.useEffect(() => {
        sceneState.canvasMounts += 1;
        return () => { sceneState.canvasMounts -= 1; };
      }, []);

      React.useEffect(() => {
        if (!sceneState.autoAssemble || announced.current) return;
        announced.current = true;
        onReady?.();
        onAssembled?.();
      }, [onAssembled, onReady]);

      if (sceneState.throwOnRender) throw new Error("WebGL context failed");
      return (
        <canvas
          data-testid="chamber-canvas"
          data-active={String(active)}
          data-trace-stage={traceStage}
          data-trace-run-id={String(traceRunId)}
          data-trace-motion={traceMotion}
          aria-hidden="true"
          tabIndex={-1}
        />
      );
    },
  };
});

type ConnectionNavigator = Navigator & {
  connection?: {
    saveData?: boolean;
    addEventListener?: (type: "change", listener: EventListener) => void;
    removeEventListener?: (type: "change", listener: EventListener) => void;
  };
  deviceMemory?: number;
};

type MediaListener = (event: MediaQueryListEvent) => void;

const originalHardwareConcurrency = Object.getOwnPropertyDescriptor(navigator, "hardwareConcurrency");
const originalDeviceMemory = Object.getOwnPropertyDescriptor(navigator, "deviceMemory");
const originalConnection = Object.getOwnPropertyDescriptor(navigator, "connection");
const originalMatchMedia = Object.getOwnPropertyDescriptor(window, "matchMedia");
const originalPointerEvent = Object.getOwnPropertyDescriptor(window, "PointerEvent");
const originalCanvasContext = Object.getOwnPropertyDescriptor(HTMLCanvasElement.prototype, "getContext");

let viewportMatches = true;
let saveDataEnabled = false;
let loseProbeContext = vi.fn();
const viewportListeners = new Set<MediaListener>();
const connectionListeners = new Set<EventListener>();

function restoreOwnProperty(target: object, property: PropertyKey, descriptor?: PropertyDescriptor) {
  if (descriptor) Object.defineProperty(target, property, descriptor);
  else Reflect.deleteProperty(target, property);
}

function viewportMediaQuery(query: string): MediaQueryList {
  return {
    get matches() { return query.includes("min-width") ? viewportMatches : false; },
    media: query,
    onchange: null,
    addListener: vi.fn((listener: MediaListener) => viewportListeners.add(listener)),
    removeListener: vi.fn((listener: MediaListener) => viewportListeners.delete(listener)),
    addEventListener: vi.fn((type: string, listener: EventListenerOrEventListenerObject) => {
      if (type === "change" && typeof listener === "function") viewportListeners.add(listener as MediaListener);
    }),
    removeEventListener: vi.fn((type: string, listener: EventListenerOrEventListenerObject) => {
      if (type === "change" && typeof listener === "function") viewportListeners.delete(listener as MediaListener);
    }),
    dispatchEvent: vi.fn(),
  };
}

function configureBrowser({
  webgl = true,
  saveData = false,
  cores = 8,
  memory = 8,
  wide = true,
} = {}) {
  viewportMatches = wide;
  saveDataEnabled = saveData;
  Object.defineProperty(navigator, "hardwareConcurrency", { configurable: true, value: cores });
  Object.defineProperty(navigator, "deviceMemory", { configurable: true, value: memory });
  Object.defineProperty(navigator, "connection", {
    configurable: true,
    value: {
      get saveData() { return saveDataEnabled; },
      addEventListener: (type: "change", listener: EventListener) => {
        if (type === "change") connectionListeners.add(listener);
      },
      removeEventListener: (type: "change", listener: EventListener) => {
        if (type === "change") connectionListeners.delete(listener);
      },
    } satisfies ConnectionNavigator["connection"],
  });
  Object.defineProperty(window, "matchMedia", {
    configurable: true,
    value: vi.fn((query: string) => viewportMediaQuery(query)),
  });
  Object.defineProperty(window, "PointerEvent", {
    configurable: true,
    value: class PointerEventMock extends MouseEvent {
      readonly pointerType: string;

      constructor(type: string, init: PointerEventInit = {}) {
        super(type, init);
        this.pointerType = init.pointerType ?? "";
      }
    },
  });
  Object.defineProperty(HTMLCanvasElement.prototype, "getContext", {
    configurable: true,
    value: vi.fn((context: string) => (webgl && context.startsWith("webgl") ? {
      getExtension: (name: string) => name === "WEBGL_lose_context"
        ? { loseContext: loseProbeContext }
        : null,
    } : null)),
  });
  Reflect.deleteProperty(window, "requestIdleCallback");
  Reflect.deleteProperty(window, "cancelIdleCallback");
}

async function advance(milliseconds: number) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(milliseconds);
    await Promise.resolve();
  });
}

async function startCapableScene() {
  const rendered = render(<HeroScene />);
  await advance(300);
  return rendered;
}

function setReducedMotion(reduced: boolean) {
  sceneState.reducedMotion = reduced;
  act(() => sceneState.reducedSubscribers.forEach((subscriber) => subscriber(reduced)));
}

function setViewportCapability(wide: boolean) {
  viewportMatches = wide;
  const event = { matches: wide, media: "(min-width: 800px)" } as MediaQueryListEvent;
  act(() => viewportListeners.forEach((listener) => listener(event)));
}

function setDataSaving(enabled: boolean) {
  saveDataEnabled = enabled;
  act(() => connectionListeners.forEach((listener) => listener(new Event("change"))));
}

function shell() {
  return document.querySelector<HTMLElement>(".hero-scene-shell")!;
}

function fallback() {
  return document.querySelector<HTMLElement>(".scene-fallback")!;
}

function expectSynchronizedStage(stage: TraceStage, motion?: TraceMotion) {
  expect(shell()).toHaveAttribute("data-trace-stage", stage);
  expect(document.querySelector(".trace-rail")).toHaveAttribute("data-trace-stage", stage);
  expect(fallback()).toHaveAttribute("data-trace-stage", stage);
  if (motion) expect(fallback()).toHaveAttribute("data-trace-motion", motion);
  const canvas = screen.queryByTestId("chamber-canvas");
  if (canvas) {
    expect(canvas).toHaveAttribute("data-trace-stage", stage);
    if (motion) expect(canvas).toHaveAttribute("data-trace-motion", motion);
  }
}

beforeEach(() => {
  vi.useFakeTimers();
  sceneState.reducedMotion = false;
  sceneState.reducedSubscribers.clear();
  sceneState.throwOnRender = false;
  sceneState.autoAssemble = true;
  sceneState.canvasMounts = 0;
  sceneState.latestCanvasProps = null;
  loseProbeContext = vi.fn();
  viewportListeners.clear();
  connectionListeners.clear();
  configureBrowser();
});

afterEach(() => {
  cleanup();
  vi.clearAllTimers();
  vi.useRealTimers();
  vi.restoreAllMocks();
  restoreOwnProperty(navigator, "hardwareConcurrency", originalHardwareConcurrency);
  restoreOwnProperty(navigator, "deviceMemory", originalDeviceMemory);
  restoreOwnProperty(navigator, "connection", originalConnection);
  restoreOwnProperty(window, "matchMedia", originalMatchMedia);
  restoreOwnProperty(window, "PointerEvent", originalPointerEvent);
  restoreOwnProperty(HTMLCanvasElement.prototype, "getContext", originalCanvasContext);
  Reflect.deleteProperty(window, "requestIdleCallback");
  Reflect.deleteProperty(window, "cancelIdleCallback");
  Reflect.deleteProperty(document, "visibilityState");
});

describe("parliamentary hero scene", () => {
  it("releases its temporary WebGL capability probe context", async () => {
    await startCapableScene();

    expect(loseProbeContext).toHaveBeenCalledTimes(1);
  });

  it("announces trace captions politely and atomically", async () => {
    sceneState.reducedMotion = true;
    render(<HeroScene />);

    const caption = screen.getByRole("status");
    expect(caption).toHaveAttribute("aria-live", "polite");
    expect(caption).toHaveAttribute("aria-atomic", "true");
    expect(caption).toHaveTextContent(TRACE_CAPTIONS.idle);

    fireEvent.click(screen.getByRole("button", { name: "Eight heads" }));
    expect(caption).toHaveTextContent(TRACE_CAPTIONS.attention);
  });

  it("autoplays the shared trace once and leaves prediction locked", async () => {
    await startCapableScene();
    expect(shell()).toHaveAttribute("data-trace-running", "true");
    expectSynchronizedStage("idle", "animate");

    await advance(TRACE_START_DELAY_MS - 1);
    expectSynchronizedStage("idle", "animate");
    await advance(1);
    expectSynchronizedStage("speaker", "animate");
    await advance(TRACE_STEPS[0].durationMs);
    expectSynchronizedStage("attention", "animate");
    await advance(TRACE_STEPS[1].durationMs);
    expectSynchronizedStage("prediction", "animate");
    await advance(TRACE_STEPS[2].durationMs);

    expect(shell()).toHaveAttribute("data-trace-running", "false");
    expectSynchronizedStage("prediction", "animate");
    expect(screen.getByRole("button", { name: "Next character" })).toHaveAttribute("aria-pressed", "true");
    const completedRunId = shell().dataset.traceRunId;

    act(() => sceneState.latestCanvasProps?.onAssembled?.());
    await advance(TRACE_SEQUENCE_DURATION_MS + 1);
    expect(shell()).toHaveAttribute("data-trace-running", "false");
    expect(shell()).toHaveAttribute("data-trace-run-id", completedRunId);
    expectSynchronizedStage("prediction", "animate");
  });

  it("previews on hover and focus, restores the lock, and cancels autoplay on selection", async () => {
    await startCapableScene();
    const idle = screen.getByRole("button", { name: /Speaker/i });
    const attention = screen.getByRole("button", { name: /Eight heads/i });
    const prediction = screen.getByRole("button", { name: /Next character/i });

    fireEvent.pointerEnter(attention, { pointerType: "mouse" });
    expect(shell()).toHaveAttribute("data-trace-running", "false");
    expectSynchronizedStage("attention", "animate");
    expect(idle).toHaveAttribute("aria-pressed", "false");
    fireEvent.pointerLeave(attention, { pointerType: "mouse" });
    expectSynchronizedStage("idle", "animate");

    fireEvent.click(idle);
    expectSynchronizedStage("speaker", "animate");
    expect(idle).toHaveAttribute("aria-pressed", "true");
    fireEvent.focus(prediction);
    expectSynchronizedStage("prediction", "animate");
    expect(idle).toHaveAttribute("aria-pressed", "true");
    fireEvent.blur(prediction);
    expectSynchronizedStage("speaker", "animate");

    fireEvent.click(attention);
    await advance(TRACE_SEQUENCE_DURATION_MS + 1);
    expectSynchronizedStage("attention", "animate");
    expect(attention).toHaveAttribute("aria-pressed", "true");
  });

  it("ignores touch pointer hover while preserving click selection", async () => {
    await startCapableScene();
    const attention = screen.getByRole("button", { name: /Eight heads/i });

    fireEvent.pointerEnter(attention, { pointerType: "touch" });
    expect(shell()).toHaveAttribute("data-trace-running", "true");
    expectSynchronizedStage("idle", "animate");

    fireEvent.click(attention);
    expect(shell()).toHaveAttribute("data-trace-running", "false");
    expectSynchronizedStage("attention", "animate");
    expect(attention).toHaveAttribute("aria-pressed", "true");

    fireEvent.pointerLeave(attention, { pointerType: "touch" });
    expectSynchronizedStage("attention", "animate");
  });

  it("replay cancels the current state and follows the shared timings", async () => {
    await startCapableScene();
    fireEvent.click(screen.getByRole("button", { name: "Eight heads" }));
    const previousRunId = Number(shell().dataset.traceRunId);

    fireEvent.click(screen.getByRole("button", { name: "Replay trace" }));
    expect(shell()).toHaveAttribute("data-trace-running", "true");
    expect(Number(shell().dataset.traceRunId)).toBeGreaterThan(previousRunId);
    expectSynchronizedStage("idle", "animate");
    await advance(TRACE_START_DELAY_MS);
    expectSynchronizedStage("speaker", "animate");
    await advance(TRACE_STEPS[0].durationMs);
    expectSynchronizedStage("attention", "animate");
    await advance(TRACE_STEPS[1].durationMs);
    expectSynchronizedStage("prediction", "animate");
    await advance(TRACE_STEPS[2].durationMs);
    expect(shell()).toHaveAttribute("data-trace-running", "false");
    expect(screen.getByRole("button", { name: "Next character" })).toHaveAttribute("aria-pressed", "true");
  });

  it("keeps reduced motion static, skips autoplay, and still supports replay", async () => {
    sceneState.reducedMotion = true;
    render(<HeroScene />);
    await advance(TRACE_SEQUENCE_DURATION_MS + 1_000);

    expect(screen.queryByTestId("chamber-canvas")).not.toBeInTheDocument();
    expect(shell()).toHaveAttribute("data-trace-running", "false");
    expectSynchronizedStage("idle", "static");
    fireEvent.click(screen.getByRole("button", { name: "Replay trace" }));
    expect(shell()).toHaveAttribute("data-trace-running", "true");
    expectSynchronizedStage("idle", "static");
    await advance(TRACE_START_DELAY_MS);
    expectSynchronizedStage("speaker", "static");
    await advance(TRACE_STEPS[0].durationMs);
    expectSynchronizedStage("attention", "static");
    await advance(TRACE_STEPS[1].durationMs);
    expectSynchronizedStage("prediction", "static");
    await advance(TRACE_STEPS[2].durationMs);
    expect(shell()).toHaveAttribute("data-trace-running", "false");
    expect(screen.getByRole("button", { name: "Next character" })).toHaveAttribute("aria-pressed", "true");
  });

  it("responds to a live reduced-motion change without replaying autoplay", async () => {
    await startCapableScene();
    const firstRunId = shell().dataset.traceRunId;
    setReducedMotion(true);
    await advance(1);
    expect(screen.queryByTestId("chamber-canvas")).not.toBeInTheDocument();
    expect(shell()).toHaveAttribute("data-trace-running", "false");
    expect(fallback()).toHaveAttribute("data-trace-motion", "static");

    setReducedMotion(false);
    await advance(300);
    expect(screen.getByTestId("chamber-canvas")).toBeInTheDocument();
    act(() => sceneState.latestCanvasProps?.onAssembled?.());
    await advance(TRACE_SEQUENCE_DURATION_MS + 1);
    expect(shell()).toHaveAttribute("data-trace-running", "false");
    expect(shell()).toHaveAttribute("data-trace-run-id", firstRunId);
  });

  it.each([
    { label: "data saving", options: { saveData: true } },
    { label: "two CPU cores", options: { cores: 2 } },
    { label: "two gigabytes of memory", options: { memory: 2 } },
    { label: "unavailable WebGL", options: { webgl: false } },
  ])("keeps the SVG fallback for $label", async ({ options }) => {
    configureBrowser(options);
    render(<HeroScene />);
    await advance(1_500);
    expect(screen.queryByTestId("chamber-canvas")).not.toBeInTheDocument();
    expect(screen.getByRole("img", { name: /parliamentary chamber/i })).toBeVisible();
  });

  it("reacts to live viewport and data-saving capability changes", async () => {
    configureBrowser({ wide: false });
    render(<HeroScene />);
    await advance(300);
    expect(screen.queryByTestId("chamber-canvas")).not.toBeInTheDocument();

    setViewportCapability(true);
    await advance(300);
    expect(screen.getByTestId("chamber-canvas")).toBeInTheDocument();
    setDataSaving(true);
    await advance(1);
    expect(screen.queryByTestId("chamber-canvas")).not.toBeInTheDocument();
    setDataSaving(false);
    await advance(300);
    expect(screen.getByTestId("chamber-canvas")).toBeInTheDocument();
  });

  it("uses requestIdleCallback when it fires", async () => {
    let idleCallback: IdleRequestCallback | undefined;
    const cancelIdle = vi.fn();
    Object.defineProperty(window, "requestIdleCallback", {
      configurable: true,
      value: vi.fn((callback: IdleRequestCallback) => { idleCallback = callback; return 17; }),
    });
    Object.defineProperty(window, "cancelIdleCallback", { configurable: true, value: cancelIdle });
    const { unmount } = render(<HeroScene />);
    await advance(100);
    expect(screen.queryByTestId("chamber-canvas")).not.toBeInTheDocument();

    act(() => idleCallback?.({ didTimeout: false, timeRemaining: () => 8 }));
    await advance(1);
    expect(screen.getByTestId("chamber-canvas")).toBeInTheDocument();
    unmount();
    expect(cancelIdle).toHaveBeenCalledWith(17);
  });

  it("uses the idle deadline and cancels every pending timer on unmount", async () => {
    const cancelIdle = vi.fn();
    Object.defineProperty(window, "requestIdleCallback", {
      configurable: true,
      value: vi.fn(() => 21),
    });
    Object.defineProperty(window, "cancelIdleCallback", { configurable: true, value: cancelIdle });
    const rendered = render(<HeroScene />);
    await advance(1_199);
    expect(screen.queryByTestId("chamber-canvas")).not.toBeInTheDocument();
    await advance(1);
    expect(screen.getByTestId("chamber-canvas")).toBeInTheDocument();
    rendered.unmount();
    expect(cancelIdle).toHaveBeenCalledWith(21);
    expect(vi.getTimerCount()).toBe(0);
  });

  it("keeps the poster mounted after canvas readiness", async () => {
    await startCapableScene();
    expect(screen.getByTestId("chamber-canvas")).toBeInTheDocument();
    await advance(1_000);
    expect(document.querySelector(".hero-scene")).toHaveAttribute("data-scene-status", "ready");
    expect(fallback()).toBeInTheDocument();
    expect(fallback()).toHaveAttribute("data-poster-visible", "false");
  });

  it("latches a WebGL context loss and never retries in the same page session", async () => {
    await startCapableScene();
    await act(async () => { await Promise.resolve(); });
    const canvas = screen.getByTestId("chamber-canvas");
    const contextLost = new Event("webglcontextlost", { cancelable: true });
    fireEvent(canvas, contextLost);

    expect(contextLost.defaultPrevented).toBe(true);
    expect(document.querySelector(".hero-scene")).toHaveAttribute("data-scene-status", "failed");
    expect(screen.queryByTestId("chamber-canvas")).not.toBeInTheDocument();
    expect(fallback()).toHaveAttribute("data-poster-visible", "true");

    setViewportCapability(false);
    setViewportCapability(true);
    await advance(1_500);
    expect(screen.queryByTestId("chamber-canvas")).not.toBeInTheDocument();
    expect(document.querySelector(".hero-scene")).toHaveAttribute("data-scene-status", "failed");
  });

  it("retains the poster when the lazy scene throws", async () => {
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => undefined);
    sceneState.throwOnRender = true;
    render(<HeroScene />);
    await advance(300);

    expect(document.querySelector(".hero-scene")).toHaveAttribute("data-scene-status", "failed");
    expect(fallback()).toHaveAttribute("data-poster-visible", "true");
    expect(screen.queryByTestId("chamber-canvas")).not.toBeInTheDocument();
    expect(consoleError).toHaveBeenCalled();
  });

  it("pauses the canvas for page visibility and cleans timers on unmount", async () => {
    let visibility: DocumentVisibilityState = "visible";
    Object.defineProperty(document, "visibilityState", {
      configurable: true,
      get: () => visibility,
    });
    const rendered = await startCapableScene();
    expect(screen.getByTestId("chamber-canvas")).toHaveAttribute("data-active", "true");

    visibility = "hidden";
    act(() => document.dispatchEvent(new Event("visibilitychange")));
    expect(screen.getByTestId("chamber-canvas")).toHaveAttribute("data-active", "false");
    visibility = "visible";
    act(() => document.dispatchEvent(new Event("visibilitychange")));
    expect(screen.getByTestId("chamber-canvas")).toHaveAttribute("data-active", "true");

    expect(vi.getTimerCount()).toBeGreaterThan(0);
    rendered.unmount();
    expect(vi.getTimerCount()).toBe(0);
    expect(sceneState.canvasMounts).toBe(0);
  });
});
