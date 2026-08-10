import type { ComponentProps } from "react";
import { render } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { HomePage } from "../pages/HomePage";

const motionAudit = vi.hoisted(() => ({
  reduced: true,
  heroInitial: null as unknown,
  heroTransition: null as unknown,
}));

vi.mock("motion/react", async () => {
  const actual = await vi.importActual<typeof import("motion/react")>("motion/react");
  const React = await import("react");

  type MotionDivProps = ComponentProps<"div"> & {
    initial?: unknown;
    animate?: unknown;
    transition?: unknown;
    whileInView?: unknown;
    viewport?: unknown;
  };

  const AuditedMotionDiv = React.forwardRef<HTMLDivElement, MotionDivProps>((props, ref) => {
    if (props.className === "hero-copy") {
      motionAudit.heroInitial = props.initial;
      motionAudit.heroTransition = props.transition;
    }

    const domProps = { ...props };
    delete domProps.initial;
    delete domProps.animate;
    delete domProps.transition;
    delete domProps.whileInView;
    delete domProps.viewport;
    return React.createElement("div", { ...domProps, ref });
  });

  const motion = new Proxy(actual.motion, {
    get(target, property, receiver) {
      if (property === "div") return AuditedMotionDiv;
      return Reflect.get(target, property, receiver);
    },
  });

  return {
    ...actual,
    motion,
    useReducedMotion: () => motionAudit.reduced,
  };
});

beforeEach(() => {
  motionAudit.reduced = true;
  motionAudit.heroInitial = null;
  motionAudit.heroTransition = null;
  vi.stubGlobal("fetch", vi.fn(() => new Promise<Response>(() => undefined)));
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("home hero motion", () => {
  it("removes vertical travel from the reduced-motion copy reveal", () => {
    render(<MemoryRouter><HomePage /></MemoryRouter>);

    expect(motionAudit.heroInitial).toEqual({ opacity: 0, y: 0 });
    expect(motionAudit.heroTransition).toEqual({ duration: 0.18 });
  });
});
