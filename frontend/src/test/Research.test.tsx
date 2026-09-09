import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { readFileSync } from "node:fs";
import { webcrypto } from "node:crypto";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import ResearchPage from "../pages/ResearchPage";
import {
  characterLabel,
  number,
  recordedResult,
  sourceSegments,
  visibleDistributions,
} from "../research";
import type { Recording, Summary } from "../research";

const summary = JSON.parse(
  readFileSync("public/research-data/pilot/summary.json", "utf8"),
) as Summary;
const recording = JSON.parse(
  readFileSync(
    "public/research-data/pilot/example-1-context_diverse.json",
    "utf8",
  ),
) as Recording;

beforeEach(() => {
  vi.stubGlobal("crypto", webcrypto);
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => {
      if (url.endsWith("capabilities"))
        return Response.json({ live: false, release_id: null });
      const name = url.split("/").pop()!;
      return new Response(readFileSync(`public/research-data/pilot/${name}`));
    }),
  );
});
afterEach(() => vi.unstubAllGlobals());
const open = (view: string) =>
  render(
    <MemoryRouter initialEntries={[`/research?view=${view}`]}>
      <ResearchPage />
    </MemoryRouter>,
  );

describe("research measurements", () => {
  it("highlights source offsets in Unicode code points rather than UTF-16 units", () => {
    expect(
      sourceSegments({ source_span: "🐈aé", span_start: 10, offset: 12 }),
    ).toEqual({ before: "🐈a", target: "é", after: "" });
  });
  it("keeps zero, unavailable, whitespace and original probability mass distinct", () => {
    expect(number(0)).toBe("0.0000");
    expect(number(null)).toBe("Unavailable");
    expect(characterLabel(" ")).toBe("Space");
    expect(characterLabel("<unk>")).toBe("Unknown");
    const original = recording.original;
    const result = recordedResult(recording, "4011093");
    expect(result.removed_entries).toBe(6);
    expect(
      result.neighbours_after.every((n) => n.speech_id !== "4011093"),
    ).toBe(true);
    const space = result.distributions.find((d) => d.character === " ")!;
    expect(space.before).toBeCloseTo(0.146419, 5);
    expect(space.after).toBeCloseTo(0.13134, 5);
    expect(result.distributions.reduce((s, d) => s + d.after, 0)).toBeCloseTo(
      1,
      6,
    );
    expect(
      visibleDistributions(result.distributions).every((d) =>
        result.distributions.includes(d),
      ),
    ).toBe(true);
    expect(recordedResult(recording, null)).toBe(original);
    expect(() => recordedResult(recording, "absent")).toThrow("unavailable");
  });

  it("renders verified overview evidence without an execution date", async () => {
    open("overview");
    expect(await screen.findByText("1.8572")).toBeInTheDocument();
    expect(
      screen.getByText(/Both memory-selection intervals include zero/),
    ).toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(
      /20260908|September 2026|generated_at/,
    );
  });

  it("supports keyboard tabs and keeps the small historical cohort explicit", async () => {
    open("overview");
    await screen.findByText("1.8572");
    const tab = screen.getByRole("tab", { name: "Overview" });
    tab.focus();
    fireEvent.keyDown(tab, { key: "End" });
    expect(screen.getByRole("tab", { name: "Methods" })).toHaveFocus();
    fireEvent.click(screen.getByRole("tab", { name: "Historical evaluation" }));
    fireEvent.change(screen.getByLabelText("Comparison"), {
      target: { value: "matched" },
    });
    expect(screen.getByText("Only 1 matched pair.")).toBeInTheDocument();
    expect(screen.getByText(/46 earlier and 34 later/)).toBeInTheDocument();
    expect(
      screen.getByText(
        number(
          summary.history.ngram.splits.earlier.matched!.supported_target_bpc,
        ),
      ),
    ).toBeInTheDocument();
  });

  it("verifies recorded bytes and performs a source removal without a live backend", async () => {
    open("memory");
    await screen.findByText("What comes next?");
    fireEvent.change(screen.getByLabelText("Memory selection"), {
      target: { value: "context_diverse" },
    });
    await waitFor(() =>
      expect(
        screen.getAllByRole("button", { name: "Exclude speech" }).length,
      ).toBeGreaterThan(0),
    );
    const buttons = screen.getAllByRole("button", { name: "Exclude speech" });
    fireEvent.click(buttons[0]);
    expect(
      screen.getByRole("button", { name: "Restore source" }),
    ).toBeEnabled();
    fireEvent.click(screen.getByRole("button", { name: "Restore source" }));
    expect(
      screen.getByRole("button", { name: "Restore source" }),
    ).toBeDisabled();
    expect(screen.queryByText("Try your own prefix")).not.toBeInTheDocument();
  });

  it("rejects a corrupted example rather than showing its values", async () => {
    const original = vi.mocked(fetch).getMockImplementation()!;
    vi.mocked(fetch).mockImplementation(async (...args) =>
      String(args[0]).endsWith(".json") &&
      !String(args[0]).endsWith("summary.json")
        ? new Response("{}")
        : original(...args),
    );
    open("memory");
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "does not match",
    );
    expect(screen.queryByText("What comes next?")).not.toBeInTheDocument();
  });
});
