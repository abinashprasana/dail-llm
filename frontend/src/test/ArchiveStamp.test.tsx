import type { ReactNode } from "react";
import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ArchiveStamp } from "../components/ArchiveStamp";

interface MotionCapture {
  initial?: unknown;
  whileInView?: unknown;
  viewport?: unknown;
  variants?: unknown;
}

const motionAudit = vi.hoisted(() => ({
  captures: {} as Record<string, MotionCapture>,
}));

vi.mock("motion/react", async () => {
  const React = await import("react");

  type MotionProps = Record<string, unknown> & {
    children?: ReactNode;
    className?: string;
  };

  const motionElement = (tag: "div" | "g" | "path" | "span") => function MockMotionElement(props: MotionProps) {
    const { className, initial, whileInView, viewport, variants } = props;
    if (className) {
      motionAudit.captures[className] = { initial, whileInView, viewport, variants };
    }

    const domProps = { ...props };
    delete domProps.initial;
    delete domProps.whileInView;
    delete domProps.viewport;
    delete domProps.variants;
    return React.createElement(tag, domProps);
  };

  return {
    motion: {
      div: motionElement("div"),
      g: motionElement("g"),
      path: motionElement("path"),
      span: motionElement("span"),
    },
  };
});

function variantsFor(className: string) {
  return motionAudit.captures[className].variants as {
    hidden: Record<string, unknown>;
    visible: Record<string, unknown> & {
      transition?: Record<string, unknown>;
    };
  };
}

function expectNoPositionalTransform(state: Record<string, unknown>) {
  for (const property of ["x", "y", "scale", "scaleX", "scaleY", "rotate"]) {
    expect(state).not.toHaveProperty(property);
  }
}

beforeEach(() => {
  motionAudit.captures = {};
});

describe("archive source stamp", () => {
  it("enters once when 65 percent is visible and keeps the final impression mounted", () => {
    render(<ArchiveStamp reducedMotion={false} />);

    const sequence = motionAudit.captures["archive-stamp-sequence"];
    expect(sequence.initial).toBe("hidden");
    expect(sequence.whileInView).toBe("visible");
    expect(sequence.viewport).toEqual({ once: true, amount: 0.65 });

    const impression = document.querySelector(".archive-stamp-impression");
    expect(impression).toBeInTheDocument();
    expect(variantsFor("archive-stamp-impression").visible).toMatchObject({ opacity: 1 });
  });

  it("stages a single press, ink border, and copy reveal on the normal-motion timeline", () => {
    render(<ArchiveStamp reducedMotion={false} />);

    const press = variantsFor("archive-stamp-press").visible;
    const transition = press.transition!;
    const x = press.x as number[];
    const y = press.y as number[];
    const scaleY = press.scaleY as number[];
    const opacity = press.opacity as number[];
    const times = transition.times as number[];

    expect(transition.duration).toBe(1.08);
    expect(times).toHaveLength(6);
    expect(times[0]).toBe(0);
    expect(times.at(-1)).toBe(1);
    expect(x[0]).toBeGreaterThan(0);
    expect(x.at(-1)).toBeLessThan(0);
    expect(y[0]).toBeGreaterThan(0);
    expect(Math.min(...y.slice(1, -1))).toBeLessThanOrEqual(2);
    expect(Math.max(...y.slice(1, -1))).toBeGreaterThanOrEqual(9);
    expect(y.at(-1)).toBeLessThan(0);
    expect(Math.min(...scaleY)).toBeLessThan(1);
    expect(opacity).toEqual([0, 1, 1, 1, 1, 0]);

    const pressShadow = variantsFor("archive-stamp-press-shadow").visible;
    expect(pressShadow.transition).toMatchObject({
      duration: 1.08,
      times,
    });
    expect(pressShadow.opacity).toEqual([0, 0.35, 0.35, 0.62, 0.4, 0]);
    expect(Math.min(...(pressShadow.scaleX as number[]))).toBeLessThan(1);

    const border = variantsFor("archive-stamp-border").visible;
    const copy = variantsFor("archive-stamp-copy").visible;
    expect(border).toMatchObject({
      opacity: 1,
      pathLength: 1,
      transition: { delay: 0.5, duration: 0.28 },
    });
    expect(copy).toMatchObject({
      opacity: 1,
      transition: { delay: 0.5, duration: 0.28 },
    });
  });

  it("uses a static 180 millisecond opacity reveal for reduced motion", () => {
    render(<ArchiveStamp reducedMotion />);

    expect(document.querySelector(".archive-stamp-press")).not.toBeInTheDocument();
    expect(document.querySelector(".archive-stamp-press-shadow")).not.toBeInTheDocument();

    const impression = variantsFor("archive-stamp-impression");
    expect(impression.hidden).toEqual({ opacity: 0 });
    expect(impression.visible).toEqual({ opacity: 1, transition: { duration: 0.18 } });

    for (const className of [
      "archive-stamp-impression",
      "archive-stamp-border",
      "archive-stamp-defects",
      "archive-stamp-copy",
    ]) {
      const variants = variantsFor(className);
      expectNoPositionalTransform(variants.hidden);
      expectNoPositionalTransform(variants.visible);
    }
  });

  it("exposes one concise source note while keeping the physical stamp decorative", () => {
    render(<ArchiveStamp reducedMotion={false} />);

    expect(screen.getByRole("note", { name: /Harvard Dataverse.*DVN\/6MZN76/i })).toBeVisible();
    expect(document.querySelector(".archive-stamp-impression")).toHaveAttribute("aria-hidden", "true");
    expect(document.querySelector(".archive-stamp-press")).toHaveAttribute("aria-hidden", "true");
    expect(document.querySelector(".archive-stamp-press-shadow")).toBeInTheDocument();
    expect(document.querySelector(".archive-stamp-border")).toBeInTheDocument();
    expect(document.querySelector(".archive-stamp-defects")).toBeInTheDocument();
    expect(document.querySelector(".archive-stamp svg"))
      .toHaveAttribute("focusable", "false");
    expect(document.querySelector(".archive-stamp svg"))
      .toHaveAttribute("aria-hidden", "true");
  });
});
