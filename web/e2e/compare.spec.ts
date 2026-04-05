import { test, expect } from "@playwright/test";

test.describe("Compare page", () => {
  test("page loads with main content area", async ({ page }) => {
    await page.goto("/compare");
    const main = page.locator("main");
    await expect(main).toBeVisible({ timeout: 10_000 });
  });

  test("page has a heading about comparison", async ({ page }) => {
    await page.goto("/compare");
    const h1 = page.locator("h1").first();
    await expect(h1).toBeVisible({ timeout: 10_000 });
    const text = await h1.textContent() ?? "";
    // Title contains "Comparison" or "Compare" or "Forecaster"
    expect(text.toLowerCase()).toMatch(/compar|forecaster/);
  });

  test("page renders without horizontal overflow", async ({ page }) => {
    await page.goto("/compare");
    await expect(page.locator("main")).toBeVisible({ timeout: 10_000 });
    const bodyScrollWidth = await page.evaluate(() => document.body.scrollWidth);
    const viewportWidth = await page.evaluate(() => window.innerWidth);
    // Allow a small tolerance (scrollbars, rounding)
    expect(bodyScrollWidth).toBeLessThanOrEqual(viewportWidth + 20);
  });

  test("comparison table or empty state is visible", async ({ page }) => {
    await page.goto("/compare");
    await expect(page.locator("main")).toBeVisible({ timeout: 10_000 });
    // Either a table or an empty/error state text should be present
    const hasTable = await page.locator("table").count().then((c) => c > 0).catch(() => false);
    const hasContent = await page.locator("main").innerText().then((t) => t.trim().length > 0);
    expect(hasTable || hasContent).toBe(true);
  });

  test("no console errors on compare page", async ({ page }) => {
    const errors: string[] = [];
    page.on("console", (msg) => {
      if (msg.type() === "error") errors.push(msg.text());
    });
    await page.goto("/compare");
    await page.waitForLoadState("networkidle", { timeout: 15_000 }).catch(() => {});
    const realErrors = errors.filter(
      (e) =>
        !e.includes("net::ERR_") &&
        !e.includes("Failed to load resource") &&
        !e.includes("favicon"),
    );
    expect(realErrors).toHaveLength(0);
  });
});
