import { test, expect } from "@playwright/test";

test.describe("Types hub page (/types)", () => {
  test("page loads with main content", async ({ page }) => {
    await page.goto("/types");
    const main = page.locator("main");
    await expect(main).toBeVisible({ timeout: 10_000 });
  });

  test("page has Electoral Types heading", async ({ page }) => {
    await page.goto("/types");
    const h1 = page.locator("h1").first();
    await expect(h1).toBeVisible({ timeout: 10_000 });
    const text = await h1.textContent() ?? "";
    expect(text).toContain("Electoral Types");
  });

  test("page contains links to individual type pages", async ({ page }) => {
    await page.goto("/types");
    await expect(page.locator("main")).toBeVisible({ timeout: 10_000 });
    // Type detail pages are at /type/[id]
    const typeLinks = page.locator('a[href^="/type/"]');
    await typeLinks.first().waitFor({ state: "attached", timeout: 10_000 });
    const count = await typeLinks.count();
    expect(count).toBeGreaterThan(0);
  });

  test("page URL matches /types", async ({ page }) => {
    await page.goto("/types");
    await expect(page).toHaveURL(/\/types/);
  });

  test("no console errors on types page", async ({ page }) => {
    const errors: string[] = [];
    page.on("console", (msg) => {
      if (msg.type() === "error") errors.push(msg.text());
    });
    await page.goto("/types");
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
