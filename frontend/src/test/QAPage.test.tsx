import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, expect, it, vi } from "vitest";
import QAPage from "../pages/QAPage";

beforeEach(() => {
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    const payload = url.endsWith("/capabilities")
      ? { available: true, generation_configured: false,
          coverage: { first_date: "2014-01-15", last_date: "2026-10-01", days: 1298, passages: 400000 } }
      : { status: "sources_only", answer: "A cited excerpt", citation_ids: ["dail:2026-10-01:a:b"],
          sources: [{ passage_id: "dail:2026-10-01:a:b", date: "2026-10-01",
            speaker: "Deputy Example", title: "Housing", text: "A cited excerpt",
            source_url: "https://www.oireachtas.ie/en/debates/debate/dail/2026-10-01/" }] };
    return new Response(JSON.stringify(payload), { status: 200,
      headers: { "Content-Type": "application/json" } });
  }));
});

it("shows indexed coverage and links a returned excerpt to the Official Report", async () => {
  render(<MemoryRouter><QAPage /></MemoryRouter>);
  expect(await screen.findByText(/passages from 15 Jan 2014 to 1 Oct 2026/)).toBeInTheDocument();
  fireEvent.change(screen.getByLabelText("Your question"), {
    target: { value: "What was said about housing?" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Search debates" }));
  expect(await screen.findByRole("heading", { name: "Relevant passages" })).toBeInTheDocument();
  expect(screen.getByRole("link", { name: /Read the Official Report/ })).toHaveAttribute(
    "href", "https://www.oireachtas.ie/en/debates/debate/dail/2026-10-01/",
  );
});

it("links only citations returned with a matching source and expands long passages", async () => {
  const longText = "A detailed parliamentary passage. ".repeat(30);
  vi.mocked(fetch).mockImplementation(async (input: RequestInfo | URL) => {
    const payload = String(input).endsWith("/capabilities")
      ? { available: true, generation_configured: true, coverage: { first_date: "2014-01-15", last_date: "2026-10-01", days: 1, passages: 1 } }
      : { status: "answered", answer: "A checked claim [valid-id] and an unknown reference [unknown-id].", citation_ids: ["valid-id", "unknown-id"],
          sources: [{ passage_id: "valid-id", date: "2026-10-01", speaker: "Deputy Example", title: "Housing", text: longText,
            source_url: "https://www.oireachtas.ie/en/debates/debate/dail/2026-10-01/" }] };
    return new Response(JSON.stringify(payload), { status: 200, headers: { "Content-Type": "application/json" } });
  });
  render(<MemoryRouter><QAPage /></MemoryRouter>);
  fireEvent.change(screen.getByLabelText("Your question"), { target: { value: "What was said about housing?" } });
  await screen.findByText(/passages from/);
  fireEvent.click(screen.getByRole("button", { name: "Search debates" }));
  expect(await screen.findByRole("link", { name: "Jump to source 1" })).toHaveAttribute("href", "#qa-source-1");
  expect(screen.getByText(/unknown-id/)).toBeInTheDocument();
  expect(screen.queryByRole("link", { name: /unknown-id/ })).not.toBeInTheDocument();
  const expand = screen.getByRole("button", { name: "Read more of this passage" });
  fireEvent.click(expand);
  expect(screen.getByRole("button", { name: "Show less" })).toHaveAttribute("aria-expanded", "true");
  expect(document.querySelector(".qa-source p")).toHaveTextContent(longText.trim());
});

it("explains unavailable search and validates the date range", async () => {
  render(<MemoryRouter><QAPage /></MemoryRouter>);
  await screen.findByText(/passages from/);
  fireEvent.change(screen.getByLabelText("From"), { target: { value: "2026-10-01" } });
  fireEvent.change(screen.getByLabelText("To"), { target: { value: "2025-01-01" } });
  expect(screen.getByRole("alert")).toHaveTextContent("start date must be before");
  expect(screen.getByRole("button", { name: "Search debates" })).toBeDisabled();

  vi.mocked(fetch).mockImplementation(async () => new Response(JSON.stringify({ available: false, generation_configured: false, coverage: null }), {
    status: 200, headers: { "Content-Type": "application/json" },
  }));
  const second = render(<MemoryRouter><QAPage /></MemoryRouter>);
  await waitFor(() => expect(second.container).toHaveTextContent("Debate search is unavailable on this deployment."));
  second.unmount();
});
