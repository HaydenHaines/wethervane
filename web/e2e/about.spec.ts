import { test, expect } from "@playwright/test";

test.describe("About page", () => {
  test("page loads with main content", async ({ page }) => {
    await page.goto("/about");
    // About page uses a (map) layout — the panel's main is the second <main> on the page
    const main = page.locator("main").last();
    await expect(main).toBeVisible({ timeout: 10_000 });
  });

  test("page has About WetherVane heading", async ({ page }) => {
    await page.goto("/about");
    const h1 = page.locator("h1").first();
    await expect(h1).toBeVisible({ timeout: 10_000 });
    const text = await h1.textContent() ?? "";
    expect(text).toContain("WetherVane");
  });

  test("page contains WetherVane description text", async ({ page }) => {
    await page.goto("/about");
    await page.locator("h1").first().waitFor({ timeout: 10_000 });
    // About page has a structural model description
    const bodyText = await page.locator("body").innerText();
    expect(bodyText.toLowerCase()).toMatch(/structural|electoral|model|political/);
  });

  test("page has a link to the methodology page", async ({ page }) => {
    await page.goto("/about");
    await page.locator("h1").first().waitFor({ timeout: 10_000 });
    const methodologyLink = page.locator('a[href*="methodology"]');
    await expect(methodologyLink.first()).toBeVisible({ timeout: 10_000 });
  });

  test("page URL matches /about", async ({ page }) => {
    await page.goto("/about");
    await expect(page).toHaveURL(/\/about/);
  });

  test("no console errors on about page", async ({ page }) => {
    const errors: string[] = [];
    page.on("console", (msg) => {
      if (msg.type() === "error") errors.push(msg.text());
    });
    await page.goto("/about");
    await page.locator("h1").first().waitFor({ timeout: 10_000 });
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
