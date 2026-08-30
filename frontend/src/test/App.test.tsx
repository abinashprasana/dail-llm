import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { MotionConfig } from "motion/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App";

const health = { status: "ready", version: "0.2.0", model_loaded: true, device: "cpu" };

function renderRoute(route: string) {
  return render(
    <MotionConfig reducedMotion="always">
      <MemoryRouter initialEntries={[route]}><App /></MemoryRouter>
    </MotionConfig>,
  );
}

beforeEach(() => {
  Object.defineProperty(navigator, "clipboard", {
    configurable: true,
    value: { writeText: vi.fn().mockResolvedValue(undefined) },
  });
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    const payload = url.includes("health") ? health : url.includes("evaluation")
      ? { checkpoint: { name: "model_best.pt" }, metrics: { cross_entropy: 1.4, perplexity: 4.1, bits_per_character: 2.02, next_character_accuracy: 0.55 }, samples: [] }
      : { name: "Dáil LLM", architecture: { block_size: 256, embed_dim: 256, n_layers: 4, n_heads: 8, dropout: 0.1, vocab_size: 80, parameters: 3270000, type: "character model" }, dataset: null };
    return new Response(JSON.stringify(payload), { status: 200, headers: { "Content-Type": "application/json" } });
  }));
});

describe("Dáil LLM application", () => {
  it("renders the product home and primary action", async () => {
    renderRoute("/");
    expect(screen.getByRole("heading", { name: /Debate, modeled/i })).toBeInTheDocument();
    expect(screen.getAllByRole("link", { name: /Open model lab/i })[0]).toHaveAttribute("href", "/lab");
  });

  it("switches model lab panels with accessible tabs", async () => {
    renderRoute("/lab");
    const attention = screen.getByRole("tab", { name: "Attention" });
    fireEvent.click(attention);
    expect(attention).toHaveAttribute("aria-selected", "true");
    expect(await screen.findByRole("heading", { name: "See what each character reads" })).toBeInTheDocument();
  });

  it("validates an empty generation prompt", async () => {
    renderRoute("/lab");
    fireEvent.change(screen.getByLabelText(/Seed prompt/i), { target: { value: "" } });
    fireEvent.click(screen.getByRole("button", { name: /Generate text/i }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Enter a prompt");
  });

  it("shows the running state, reveals returned text, and copies it", async () => {
    let resolveGeneration: ((response: Response) => void) | undefined;
    const originalFetch = vi.mocked(fetch);
    originalFetch.mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/generate")) {
        return new Promise<Response>((resolve) => { resolveGeneration = resolve; });
      }
      const payload = url.includes("health") ? health : url.includes("evaluation")
        ? { checkpoint: { name: "model_best.pt" }, metrics: {}, samples: [] }
        : { name: "Dáil LLM", architecture: { block_size: 256, embed_dim: 256, n_layers: 4, n_heads: 8, parameters: 3271168 }, dataset: null };
      return new Response(JSON.stringify(payload), { status: 200, headers: { "Content-Type": "application/json" } });
    });

    renderRoute("/lab");
    fireEvent.click(screen.getByRole("button", { name: /Generate text/i }));
    expect(await screen.findByRole("button", { name: /Model is running/i })).toBeDisabled();
    resolveGeneration?.(new Response(JSON.stringify({
      text: "The Minister for Finance",
      prompt: "The Minister for",
      generated_characters: 8,
      elapsed_ms: 125,
      filtered_characters: [],
    }), { status: 200, headers: { "Content-Type": "application/json" } }));

    expect(await screen.findByText("The Minister for Finance")).toBeInTheDocument();
    expect(document.querySelectorAll(".output-card [aria-live='polite']")).toHaveLength(1);
    expect(document.querySelector(".output-card pre")).not.toHaveAttribute("aria-live");
    fireEvent.click(screen.getByRole("button", { name: "Copy generated text" }));
    await waitFor(() => expect(navigator.clipboard.writeText).toHaveBeenCalledWith("The Minister for Finance"));
    expect(await screen.findByText("Copied")).toBeInTheDocument();
  });

  it("surfaces an API error in the generation panel", async () => {
    vi.mocked(fetch).mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/generate")) {
        return new Response(JSON.stringify({ detail: "The model service is busy. Try again shortly." }), {
          status: 429,
          headers: { "Content-Type": "application/json" },
        });
      }
      return new Response(JSON.stringify(url.includes("health") ? health : url.includes("evaluation")
        ? { checkpoint: { name: "model_best.pt" }, metrics: {}, samples: [] }
        : { name: "Dáil LLM", architecture: { block_size: 256, embed_dim: 256, n_layers: 4, n_heads: 8, parameters: 3271168 }, dataset: null }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    });

    renderRoute("/lab");
    fireEvent.click(screen.getByRole("button", { name: /Generate text/i }));
    expect(await screen.findByRole("alert")).toHaveTextContent("model service is busy");
  });

  it("supports arrow-key navigation across model tools", async () => {
    renderRoute("/lab");
    const generate = screen.getByRole("tab", { name: "Generate" });
    generate.focus();
    fireEvent.keyDown(generate, { key: "ArrowRight" });
    const evaluation = screen.getByRole("tab", { name: "Evaluation" });
    expect(evaluation).toHaveAttribute("aria-selected", "true");
    expect(evaluation).toHaveFocus();
  });
});
