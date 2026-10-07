import { expect, test } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

test("current-debate excerpts retain source links and coverage", async ({ page }) => {
  await page.route("**/api/v1/qa/capabilities", route => route.fulfill({ json: {
    available: true, generation_configured: false,
    coverage: { first_date: "2014-01-15", last_date: "2026-10-01", days: 1298, passages: 496826 },
  } }));
  await page.route("**/api/v1/qa/ask", route => route.fulfill({ json: {
    status: "sources_only", answer: "Deputy Example, 2026-10-01: housing was discussed.",
    citation_ids: ["dail:2026-10-01:dbsect_1:spk_1"],
    sources: [{ passage_id: "dail:2026-10-01:dbsect_1:spk_1", date: "2026-10-01",
      speaker: "Deputy Example", title: "Housing", text: "Housing was discussed.",
      source_url: "https://www.oireachtas.ie/en/debates/debate/dail/2026-10-01/" }],
  } }));
  await page.goto("/ask");
  await expect(page.getByText(/passages from 15 Jan 2014 to 1 Oct 2026/)).toBeVisible();
  await page.getByLabel("Your question").fill("What was said about housing?");
  await page.getByRole("button", { name: "Search debates" }).click();
  await expect(page.getByRole("heading", { name: "Relevant passages" })).toBeVisible();
  await expect(page.getByRole("link", { name: /Read the Official Report/ })).toHaveAttribute(
    "href", "https://www.oireachtas.ie/en/debates/debate/dail/2026-10-01/",
  );
  expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
});
