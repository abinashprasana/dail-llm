import { expect, test } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

async function openReducedMotionHome(page: import("@playwright/test").Page) {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/");
  await page.evaluate(() => document.fonts.ready);
}

async function expectTraceStage(
  page: import("@playwright/test").Page,
  stage: "idle" | "speaker" | "attention" | "prediction",
  options?: { timeout?: number },
) {
  await expect(page.locator(".hero-scene-shell")).toHaveAttribute("data-trace-stage", stage, options);
  await expect(page.locator(".trace-rail")).toHaveAttribute("data-trace-stage", stage, options);
  await expect(page.locator(".scene-fallback")).toHaveAttribute("data-trace-stage", stage, options);
}

test.beforeEach(async ({ page }) => {
  await page.route("**/api/v1/**", async (route) => {
    const url = route.request().url();
    if (url.endsWith("/health")) {
      return route.fulfill({ json: { status: "ready", version: "0.2.0", model_loaded: true, device: "cpu" } });
    }
    if (url.endsWith("/evaluation")) {
      return route.fulfill({ json: { checkpoint: { name: "model_best.pt" }, metrics: { cross_entropy: 1.4, perplexity: 4.07, bits_per_character: 2.02, next_character_accuracy: 0.55 }, samples: [{ text: "The Minister for the lay pig. There arrangements who are in principles." }] } });
    }
    if (url.endsWith("/model")) {
      return route.fulfill({ json: { name: "Dáil LLM", checkpoint: { name: "model_best.pt", sha256: "abc" }, architecture: { block_size: 256, embed_dim: 256, n_layers: 4, n_heads: 8, dropout: 0.1, parameters: 3271168 } } });
    }
    return route.continue();
  });
});

test("home has no horizontal overflow", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: /Debate, modeled/i })).toBeVisible();
  const dimensions = await page.evaluate(() => ({ width: document.documentElement.clientWidth, scroll: document.documentElement.scrollWidth }));
  expect(dimensions.scroll).toBeLessThanOrEqual(dimensions.width + 1);
});

test("compact navigation reaches the model lab", async ({ page }) => {
  test.skip((page.viewportSize()?.width ?? 9999) > 780, "Compact navigation behavior");
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.goto("/");
  await page.getByRole("button", { name: "Open navigation" }).click();
  const navigation = page.getByRole("navigation", { name: "Primary navigation" });
  const labLink = navigation.getByRole("link", { name: "Open model lab" });
  await expect(labLink).toBeVisible();
  await Promise.all([
    page.waitForURL("**/lab"),
    labLink.click(),
  ]);
  await page.waitForTimeout(500);
  expect(pageErrors).toEqual([]);
  await expect(page.getByRole("heading", { name: "Model lab" })).toBeVisible();
});

test("home has no automatically detectable accessibility violations", async ({ page }) => {
  await openReducedMotionHome(page);
  const results = await new AxeBuilder({ page }).analyze();
  expect(results.violations).toEqual([]);
});

test("reduced motion keeps the designed hero fallback", async ({ page }) => {
  const chamberRequests: string[] = [];
  page.on("request", (request) => {
    if (/\/assets\/ChamberCanvas-[^/]+\.js$/.test(new URL(request.url()).pathname)) {
      chamberRequests.push(request.url());
    }
  });
  await openReducedMotionHome(page);
  const poster = page.getByRole("img", { name: /parliamentary chamber/i });
  await expect(poster).toBeVisible();
  await expect(poster).toHaveAttribute("aria-label", /parliamentary chamber/i);
  await expect(page.locator(".hero-scene canvas")).toHaveCount(0);
  await expectTraceStage(page, "idle");
  await expect(page.locator(".hero-scene-shell")).toHaveAttribute("data-trace-running", "false");
  await page.waitForTimeout(2_750);
  expect(chamberRequests).toEqual([]);
  await expectTraceStage(page, "idle");
});

test("the hero poster remains beneath an optional decorative canvas", async ({ page }) => {
  test.skip((page.viewportSize()?.width ?? 0) < 800, "Desktop WebGL lifecycle behavior");
  await page.goto("/");
  const poster = page.getByRole("img", { name: /parliamentary chamber/i });
  const fallback = page.locator(".scene-fallback");
  await expect(poster).toBeVisible();
  await expect(fallback).toHaveCount(1);
  await page.waitForTimeout(1_250);
  await expect(poster).toBeVisible();
  await expect(fallback).toHaveCount(1);

  const canvas = page.locator(".hero-scene canvas");
  if (await canvas.count()) {
    await expect(canvas).toHaveAttribute("aria-hidden", "true");
  }
});

test("mobile presents the argument and primary action before the chamber", async ({ page }) => {
  test.skip((page.viewportSize()?.width ?? 9999) > 480, "Phone hero order");
  await openReducedMotionHome(page);

  const heading = page.getByRole("heading", { name: /Parliamentary debate, modeled character by character/i });
  const primaryAction = page.getByRole("link", { name: /Open model lab/i }).first();
  const scene = page.locator(".hero-scene");
  await expect(heading).toBeVisible();
  await expect(primaryAction).toBeVisible();
  await expect(scene).toBeVisible();

  const [headingBox, actionBox, sceneBox] = await Promise.all([
    heading.boundingBox(),
    primaryAction.boundingBox(),
    scene.boundingBox(),
  ]);
  expect(headingBox).not.toBeNull();
  expect(actionBox).not.toBeNull();
  expect(sceneBox).not.toBeNull();
  expect(headingBox!.y).toBeLessThan(sceneBox!.y);
  expect(actionBox!.y).toBeLessThan(sceneBox!.y);
});

test("a capable desktop requires the decorative WebGL chamber", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop", "Desktop WebGL integration only");
  await page.goto("/");

  const canvas = page.locator(".hero-scene canvas");
  await expect(canvas).toHaveCount(1, { timeout: 5_000 });
  await expect(page.locator(".hero-scene")).toHaveAttribute("data-scene-status", "ready", { timeout: 5_000 });
  await expect(canvas).toHaveAttribute("aria-hidden", "true");
  await expect(canvas).toHaveAttribute("tabindex", "-1");
  await expect(page.locator(".scene-fallback")).toHaveCount(1);
});

test("mouse preview restores the lock and click commits a trace stage", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop", "Desktop pointer controls only");
  await page.goto("/");

  const shell = page.locator(".hero-scene-shell");
  const speaker = page.locator('.trace-stage-button[data-trace-stage="speaker"]');
  const attention = page.locator('.trace-stage-button[data-trace-stage="attention"]');
  const prediction = page.locator('.trace-stage-button[data-trace-stage="prediction"]');

  await prediction.click();
  await expect(shell).toHaveAttribute("data-trace-running", "false");
  await expect(prediction).toHaveAttribute("aria-pressed", "true");
  await attention.hover();
  await expectTraceStage(page, "attention");
  await expect(prediction).toHaveAttribute("aria-pressed", "true");
  await page.locator(".hero-copy h1").hover();
  await expectTraceStage(page, "prediction");

  await speaker.click();
  await expectTraceStage(page, "speaker");
  await expect(speaker).toHaveAttribute("aria-pressed", "true");
  await expect(prediction).toHaveAttribute("aria-pressed", "false");
});

test("trace stages and replay work from the keyboard without focusing the canvas", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop", "Desktop keyboard controls only");
  await page.goto("/");

  const shell = page.locator(".hero-scene-shell");
  const attention = page.locator('.trace-stage-button[data-trace-stage="attention"]');
  const prediction = page.locator('.trace-stage-button[data-trace-stage="prediction"]');
  const replay = page.locator(".trace-replay");
  const canvas = page.locator(".hero-scene canvas");

  await prediction.click();
  await attention.focus();
  await expectTraceStage(page, "attention");
  await page.keyboard.press("Enter");
  await expect(attention).toHaveAttribute("aria-pressed", "true");
  await prediction.focus();
  await page.keyboard.press("Space");
  await expect(prediction).toHaveAttribute("aria-pressed", "true");

  await replay.focus();
  await page.keyboard.press("Enter");
  await expect(shell).toHaveAttribute("data-trace-running", "true");
  // Add brief delay for state propagation
  await page.waitForTimeout(50);
  await expectTraceStage(page, "speaker", { timeout: 1_000 });
  await expectTraceStage(page, "attention", { timeout: 1_250 });
  await expectTraceStage(page, "prediction", { timeout: 1_600 });
  await expect(shell).toHaveAttribute("data-trace-running", "false", { timeout: 1_200 });
  await expect(prediction).toHaveAttribute("aria-pressed", "true");

  if (await canvas.count()) {
    await expect(canvas).toHaveAttribute("tabindex", "-1");
  }
  await expect(page.locator("canvas:focus")).toHaveCount(0);
});

test("the static mobile chamber stays synchronized with manual trace controls", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "mobile", "Phone SVG interaction only");
  await openReducedMotionHome(page);
  await expect(page.locator(".hero-scene canvas")).toHaveCount(0);
  await expectTraceStage(page, "idle");

  const attention = page.locator('.trace-stage-button[data-trace-stage="attention"]');
  await attention.click();
  await expectTraceStage(page, "attention");
  await expect(attention).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator(".scene-fallback")).toHaveAttribute("data-trace-motion", "static");
});

test("hero controls preserve native page scrolling", async ({ page }) => {
  await openReducedMotionHome(page);
  const initialScroll = await page.evaluate(() => window.scrollY);
  await page.mouse.wheel(0, 700);
  await expect.poll(() => page.evaluate(() => window.scrollY)).toBeGreaterThan(initialScroll);
  await page.locator("#model").scrollIntoViewIfNeeded();
  await expect(page.locator("#model")).toBeInViewport();
});

test("trace controls remain visible without introducing overflow", async ({ page }) => {
  await openReducedMotionHome(page);
  await expect(page.locator(".trace-rail")).toBeVisible();
  await expect(page.locator(".trace-stage-button")).toHaveCount(3);
  await expect(page.locator(".trace-stage-button").first()).toBeVisible();
  await expect(page.locator(".trace-replay")).toBeVisible();

  const dimensions = await page.evaluate(() => ({
    client: document.documentElement.clientWidth,
    scroll: document.documentElement.scrollWidth,
  }));
  expect(dimensions.scroll).toBeLessThanOrEqual(dimensions.client + 1);
});

test("archive metadata and source stamp remain distinct inside the card", async ({ page }) => {
  await openReducedMotionHome(page);
  const card = page.locator(".archive-card");
  const stamp = card.getByRole("note", { name: /Harvard Dataverse.*DVN\/6MZN76/i });
  await stamp.scrollIntoViewIfNeeded();
  await expect(stamp).toBeVisible();
  await expect(stamp.locator(".archive-stamp-impression")).toHaveCSS("opacity", "1");

  const geometry = await card.evaluate((cardElement) => {
    const metadata = cardElement.querySelector("dl");
    const sourceStamp = cardElement.querySelector(".archive-stamp");
    if (!metadata || !sourceStamp) throw new Error("Archive geometry hooks are missing");
    const cardBox = cardElement.getBoundingClientRect();
    const metadataBox = metadata.getBoundingClientRect();
    const stampBox = sourceStamp.getBoundingClientRect();
    const metadataOverlapsStamp = metadataBox.left < stampBox.right
      && metadataBox.right > stampBox.left
      && metadataBox.top < stampBox.bottom
      && metadataBox.bottom > stampBox.top;

    return {
      card: { left: cardBox.left, right: cardBox.right, top: cardBox.top, bottom: cardBox.bottom },
      stamp: { left: stampBox.left, right: stampBox.right, top: stampBox.top, bottom: stampBox.bottom },
      metadataOverlapsStamp,
      documentWidth: document.documentElement.clientWidth,
      documentScrollWidth: document.documentElement.scrollWidth,
    };
  });

  const tolerance = 2;
  expect(geometry.stamp.left).toBeGreaterThanOrEqual(geometry.card.left - tolerance);
  expect(geometry.stamp.right).toBeLessThanOrEqual(geometry.card.right + tolerance);
  expect(geometry.stamp.top).toBeGreaterThanOrEqual(geometry.card.top - tolerance);
  expect(geometry.stamp.bottom).toBeLessThanOrEqual(geometry.card.bottom + tolerance);
  expect(geometry.metadataOverlapsStamp).toBe(false);
  expect(geometry.documentScrollWidth).toBeLessThanOrEqual(geometry.documentWidth + 1);
});

test("the mobile archive stamp waits offscreen and reveals when reached", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "mobile", "Phone viewport reveal only");
  await openReducedMotionHome(page);

  const stamp = page.getByRole("note", { name: /Harvard Dataverse.*DVN\/6MZN76/i });
  const impression = stamp.locator(".archive-stamp-impression");
  await expect(stamp).not.toBeInViewport();
  await expect(impression).toHaveCSS("opacity", "0");

  await stamp.scrollIntoViewIfNeeded();
  await expect(stamp).toBeInViewport();
  await expect(impression).toHaveCSS("opacity", "1");
});

test("matches the reduced-motion archive visual baseline", async ({ page }, testInfo) => {
  const width = page.viewportSize()?.width;
  test.skip(width !== 1440 && width !== 390, "Desktop and phone archive captures only");
  testInfo.snapshotSuffix = "";
  await openReducedMotionHome(page);

  const card = page.locator(".archive-card");
  await card.scrollIntoViewIfNeeded();
  await expect(card.getByRole("note", { name: /Harvard Dataverse.*DVN\/6MZN76/i }))
    .toBeVisible();
  await expect(card.locator(".archive-stamp-impression")).toHaveCSS("opacity", "1");
  await expect(card).toHaveScreenshot("archive-reduced-motion.png", {
    animations: "disabled",
    caret: "hide",
    maxDiffPixelRatio: 0.01,
  });
});

test("matches the reduced-motion hero visual baseline", async ({ page }, testInfo) => {
  const width = page.viewportSize()?.width;
  test.skip(width !== 1440 && width !== 390, "Desktop and phone hero captures only");
  // Keep one baseline per viewport instead of one per operating system. The
  // locally bundled fonts and a small pixel-difference allowance make these
  // snapshots portable to the Linux CI runner.
  testInfo.snapshotSuffix = "";
  await openReducedMotionHome(page);

  const hero = page.locator(".hero-section");
  await expect(hero).toBeVisible();
  await expect(hero).toHaveScreenshot("hero-reduced-motion.png", {
    animations: "disabled",
    caret: "hide",
    maxDiffPixelRatio: 0.01,
  });
});
