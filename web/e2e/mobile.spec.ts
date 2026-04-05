import { test, expect } from "@playwright/test";

const MOBILE_VIEWPORT = { width: 375, height: 667 };

test.describe("Mobile viewport tests (375x667)", () => {
  test.use({ viewport: MOBILE_VIEWPORT });

  test("landing page renders without horizontal scroll", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator("main")).toBeVisible({ timeout: 10_000 });
    const scrollWidth = await page.evaluate(() => document.documentElement.scrollWidth);
    expect(scrollWidth).toBeLessThanOrEqual(MOBILE_VIEWPORT.width + 5);
  });

  test("landing page renders main content on mobile", async ({ page }) => {
    await page.goto("/");
    // The scrollytelling homepage may render h1 off-screen — verify main content exists in the DOM
    const main = page.locator("main").first();
    await expect(main).toBeVisible({ timeout: 10_000 });
  });

  test("navigation is rendered on mobile", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator("main")).toBeVisible({ timeout: 10_000 });
    // On mobile, either a hamburger button OR a nav element is present
    const hasHamburger =
      (await page.locator('[aria-label*="menu" i], [aria-label*="navigation" i], button[type="button"]').count()) > 0;
    const hasNav = (await page.locator("nav").count()) > 0;
    expect(hasHamburger || hasNav).toBe(true);
  });

  test("forecast senate page loads on mobile", async ({ page }) => {
    await page.goto("/forecast/senate");
    const h1 = page.locator("h1");
    await expect(h1).toBeVisible({ timeout: 30_000 });
  });

  test("forecast senate page has no horizontal overflow on mobile", async ({ page }) => {
    await page.goto("/forecast/senate");
    await expect(page.locator("h1")).toBeVisible({ timeout: 30_000 });
    const scrollWidth = await page.evaluate(() => document.documentElement.scrollWidth);
    expect(scrollWidth).toBeLessThanOrEqual(MOBILE_VIEWPORT.width + 5);
  });

  test("race detail page loads on mobile", async ({ page }) => {
    await page.goto("/forecast/2026-ga-senate");
    const article = page.locator("article#main-content");
    await expect(article).toBeVisible({ timeout: 15_000 });
  });

  test("race detail page content is accessible on mobile (scrollable)", async ({ page }) => {
    await page.goto("/forecast/2026-ga-senate");
    await expect(page.locator("article#main-content")).toBeVisible({ timeout: 15_000 });
    // Page should be scrollable (not trapped) — verify body height > viewport height
    const bodyHeight = await page.evaluate(() => document.body.scrollHeight);
    expect(bodyHeight).toBeGreaterThan(MOBILE_VIEWPORT.height);
  });

  test("methodology page loads on mobile", async ({ page }) => {
    await page.goto("/methodology");
    // Use h1 as the load signal — avoids strict mode violation from two <main> elements
    const h1 = page.locator("h1").first();
    await expect(h1).toBeVisible({ timeout: 10_000 });
  });

  test("type detail page loads on mobile", async ({ page }) => {
    await page.goto("/type/0");
    const main = page.locator("main");
    await expect(main).toBeVisible({ timeout: 10_000 });
  });
});
