import { fireEvent, render, screen } from "@testing-library/react";
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
  expect(await screen.findByText(/passages from 2014-01-15 to 2026-10-01/)).toBeInTheDocument();
  fireEvent.change(screen.getByLabelText("Your question"), {
    target: { value: "What was said about housing?" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Search debates" }));
  expect(await screen.findByRole("heading", { name: "Relevant passages" })).toBeInTheDocument();
  expect(screen.getByRole("link", { name: /Read the Official Report/ })).toHaveAttribute(
    "href", "https://www.oireachtas.ie/en/debates/debate/dail/2026-10-01/",
  );
});
