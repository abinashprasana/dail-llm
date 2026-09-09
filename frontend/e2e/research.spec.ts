import { expect, test } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { readFileSync } from "node:fs";

const sample = JSON.parse(
  readFileSync("public/research-data/pilot/example-1-uniform.json", "utf8"),
).original;

test.beforeEach(async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.route("**/api/v1/**", (route) =>
    route.fulfill({
      status: 503,
      contentType: "application/json",
      body: JSON.stringify({ detail: "Live service unavailable" }),
    }),
  );
});

for (const view of ["overview", "memory", "history", "methods"]) {
  test(`research ${view}: accessible, responsive and traceable`, async ({
    page,
  }, testInfo) => {
    testInfo.snapshotSuffix = "";
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto(`/research?view=${view}`);
    await expect(
      page.getByRole("heading", { name: "Research", exact: true }),
    ).toBeVisible();
    await expect(page.locator(".research-intro h2")).toBeVisible();
    if (view === "memory")
      await expect(
        page.getByRole("heading", { name: "What comes next?" }),
      ).toBeVisible();
    await page.evaluate(() => document.fonts.ready);
    const dimensions = await page.evaluate(() => ({
      width: innerWidth,
      scroll: document.documentElement.scrollWidth,
    }));
    expect(dimensions.scroll).toBeLessThanOrEqual(dimensions.width + 1);
    expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
    expect(await page.locator("main").innerText()).not.toMatch(
      /September 2026|20260908|portfolio|student project/i,
    );
    if (["desktop", "mobile"].includes(testInfo.project.name))
      await expect(page).toHaveScreenshot(`research-${view}.png`, {
        animations: "disabled",
        maxDiffPixelRatio: 0.015,
      });
    expect(errors).toEqual([]);
  });
}

test("recorded source removal and restored probabilities", async ({ page }) => {
  await page.goto("/research?view=memory");
  await page
    .getByRole("combobox", { name: /^Memory selection/ })
    .selectOption("context_diverse");
  await expect(
    page.getByRole("heading", { name: "What comes next?" }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: /Show all .* contributing speeches/ })
    .click();
  const card = page.locator(".source-card").filter({ hasText: "4011093" });
  await card.getByRole("button", { name: "Exclude speech" }).click();
  await expect(page.locator(".research-status")).toContainText(
    "6 entries removed",
  );
  await expect(page.getByText("After removal", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Restore source" }).click();
  await expect(page.locator(".research-status")).toContainText(
    "0 entries removed",
  );
});

test("tabs, history and mobile navigation preserve deep links", async ({
  page,
}) => {
  await page.goto("/research?view=history");
  await page.getByLabel("Comparison").selectOption("matched");
  await expect(page.getByText("Only 1 matched pair.")).toBeVisible();
  await page.getByRole("tab", { name: "Methods" }).click();
  await expect(page).toHaveURL(/view=methods/);
  await page.goBack();
  await expect(
    page.getByRole("tab", { name: "Historical evaluation" }),
  ).toHaveAttribute("aria-selected", "true");
  await page.getByRole("tab", { name: "Historical evaluation" }).focus();
  await page.keyboard.press("Home");
  await expect(page.getByRole("tab", { name: "Overview" })).toBeFocused();
  if ((page.viewportSize()?.width ?? 9999) <= 780)
    await page.getByRole("button", { name: "Open navigation" }).click();
  await page
    .getByRole("navigation", { name: "Primary navigation" })
    .getByRole("link", { name: "Data", exact: true })
    .click();
  await expect(page).toHaveURL(/\/#data$/);
  await expect(page.locator("#data")).toBeInViewport();
});

test("narrow layout, enlarged text and memory explanation retain their content", async ({
  page,
}) => {
  await page.setViewportSize({ width: 320, height: 780 });
  await page.goto("/research?view=memory");
  await expect(
    page.getByRole("heading", { name: "What comes next?" }),
  ).toBeVisible();
  await page.addStyleTag({ content: "html { font-size: 200%; }" });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth + 1,
    ),
  ).toBe(true);
  await page.goto("/");
  await page.locator(".memory-section-slot").scrollIntoViewIfNeeded();
  await expect(
    page.getByRole("heading", { name: "Follow a speech into memory" }),
  ).toBeVisible();
  await expect(page.locator(".memory-story-steps article")).toHaveCount(4);
  await expect(page.locator(".hero-scene canvas")).toHaveCount(0);
});

test("live inspection handles busy responses, Unicode and stale requests", async ({
  page,
}) => {
  await page.route("**/api/v1/research/capabilities", (route) =>
    route.fulfill({ json: { live: true, release_id: sample.release_id } }),
  );
  let calls = 0;
  await page.route("**/api/v1/research/inspect", async (route) => {
    calls++;
    if (calls === 1)
      return route.fulfill({
        status: 429,
        headers: { "Retry-After": "3" },
        json: { detail: "Busy" },
      });
    const payload = route.request().postDataJSON();
    return route.fulfill({
      json: {
        ...sample,
        prefix: payload.prefix,
        unknown_characters: 1,
        neighbours_before: [
          {
            ...sample.neighbours_before[0],
            member: "<img src=x onerror=alert(1)>",
          },
        ],
      },
    });
  });
  await page.goto("/research?view=memory");
  await page.getByRole("button", { name: "Try your own prefix" }).click();
  await page.getByLabel("Your prefix").fill("The Minister 🐈");
  await page.getByRole("button", { name: "Inspect prefix" }).click();
  await expect(page.getByRole("alert")).toContainText("Try again in 3 seconds");
  await expect(page.getByLabel("Your prefix")).toHaveValue("The Minister 🐈");
  await page.getByRole("button", { name: "Inspect prefix" }).click();
  await expect(page.locator(".research-status")).toContainText(
    "1 unknown input characters",
  );
  await expect(
    page.getByText("<img src=x onerror=alert(1)>", { exact: true }),
  ).toBeVisible();
  await expect(page.locator(".source-card img")).toHaveCount(0);
  await page.getByLabel("Your prefix").fill("🐈".repeat(257));
  await expect(
    page.getByRole("button", { name: "Inspect prefix" }),
  ).toBeDisabled();
  await page.getByLabel("Your prefix").fill("Another prefix");
  let release: () => void = () => {};
  const pending = new Promise<void>((resolve) => {
    release = resolve;
  });
  await page.route("**/api/v1/research/inspect", async (route) => {
    await pending;
    await route
      .fulfill({ json: { ...sample, prefix: "Another prefix" } })
      .catch(() => {});
  });
  await page.getByRole("button", { name: "Inspect prefix" }).click();
  await page.getByRole("button", { name: "Use recorded examples" }).click();
  release();
  await expect(
    page.getByRole("heading", { name: "Recorded example" }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "What comes next?" }),
  ).toBeVisible();
  await expect(page.locator(".research-status")).toContainText(
    "0 unknown input characters",
  );
});

test("memory workspace visual baseline", async ({ page }, testInfo) => {
  test.skip(!["desktop", "mobile"].includes(testInfo.project.name));
  testInfo.snapshotSuffix = "";
  await page.goto("/research?view=memory");
  await expect(
    page.getByRole("heading", { name: "What comes next?" }),
  ).toBeVisible();
  await page.addStyleTag({ content: ".site-header { visibility: hidden !important; }" });
  await expect(page.locator(".memory-workspace")).toHaveScreenshot(
    "memory-workspace.png",
    {
      animations: "disabled",
      maxDiffPixelRatio: 0.015,
    },
  );
});

test("memory explanation responds to native scroll without trapping it", async ({
  page,
}, testInfo) => {
  test.skip(testInfo.project.name !== "desktop");
  testInfo.snapshotSuffix = "";
  await page.emulateMedia({ reducedMotion: "no-preference" });
  await page.goto("/");
  await page.locator(".memory-section-slot").scrollIntoViewIfNeeded();
  const sequence = page.locator(".memory-sequence");
  await expect(sequence).toBeVisible();
  for (const stage of [0, 2, 3]) {
    await sequence.evaluate((element, step) => {
      const bounds = element.getBoundingClientRect();
      window.scrollTo({
        top:
          bounds.top +
          scrollY -
          innerHeight / 2 +
          (bounds.height * (step + 0.5)) / 4,
        behavior: "instant",
      });
    }, stage);
    await expect(
      sequence.locator(".memory-story-steps article").nth(stage),
    ).toHaveClass("is-active");
    await expect(page).toHaveScreenshot(`memory-sequence-${stage}.png`, {
      animations: "disabled",
      maxDiffPixelRatio: 0.015,
    });
  }
  const before = await page.evaluate(() => scrollY);
  await page.mouse.wheel(0, 350);
  await expect.poll(() => page.evaluate(() => scrollY)).toBeGreaterThan(before);
});
