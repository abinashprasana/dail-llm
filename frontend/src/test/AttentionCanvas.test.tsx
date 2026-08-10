import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AttentionCanvas } from "../components/AttentionCanvas";

const context = {
  scale: vi.fn(),
  fillRect: vi.fn(),
  set fillStyle(_value: string) {},
};

afterEach(() => {
  vi.restoreAllMocks();
});

describe("AttentionCanvas", () => {
  it("lets keyboard users inspect attention cells with bounded arrow navigation", () => {
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(context as unknown as CanvasRenderingContext2D);

    render(
      <AttentionCanvas
        matrices={[[[0.1, 0.2], [0.3, 0.4]]]}
        labels={["A", "B"]}
        layer={0}
        head={0}
      />,
    );

    const matrix = screen.getByRole("img", { name: /2 query characters and 2 key characters/i });
    vi.spyOn(matrix, "getBoundingClientRect").mockReturnValue({
      bottom: 200,
      height: 200,
      left: 0,
      right: 200,
      top: 0,
      width: 200,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    });
    fireEvent.pointerMove(matrix, { clientX: 25, clientY: 175 });
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
    fireEvent.focus(matrix);
    fireEvent.keyDown(matrix, { key: "ArrowRight" });
    expect(screen.getByRole("status")).toHaveTextContent("Query A · Key B");
    expect(screen.getByRole("status")).toHaveTextContent("0.2000");
    fireEvent.keyDown(matrix, { key: "ArrowDown" });
    expect(screen.getByRole("status")).toHaveTextContent("Query B · Key B");
    expect(screen.getByRole("status")).toHaveTextContent("0.4000");

    fireEvent.keyDown(matrix, { key: "End" });
    fireEvent.keyDown(matrix, { key: "ArrowRight" });
    fireEvent.keyDown(matrix, { key: "ArrowDown" });
    expect(screen.getByRole("status")).toHaveTextContent("Query B · Key B");

    fireEvent.keyDown(matrix, { key: "Home" });
    expect(screen.getByRole("status")).toHaveTextContent("Query A · Key A");
  });
});
