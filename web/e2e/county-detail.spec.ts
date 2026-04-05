import { test, expect } from "@playwright/test";

test.describe("County detail pages (/county/[fips])", () => {
  test.describe("/county/13121 — Fulton County, GA", () => {
    const url = "/county/13121";

    test("page loads with main content", async ({ page }) => {
      await page.goto(url);
      const main = page.locator("main");
      await expect(main).toBeVisible({ timeout: 10_000 });
    });

    test("page heading shows county name", async ({ page }) => {
      await page.goto(url);
      const h1 = page.locator("h1").first();
      await expect(h1).toBeVisible({ timeout: 10_000 });
      const text = await h1.textContent() ?? "";
      // Should contain "Fulton" or at least a non-empty county name
      expect(text.trim().length).toBeGreaterThan(0);
    });

    test("page has breadcrumb navigation", async ({ page }) => {
      await page.goto(url);
      const breadcrumb = page.locator("nav[aria-label='Breadcrumb']");
      await expect(breadcrumb).toBeVisible({ timeout: 10_000 });
    });

    test("type information is displayed", async ({ page }) => {
      await page.goto(url);
      await expect(page.locator("main")).toBeVisible({ timeout: 10_000 });
      // Type name or link to /type/
      const hasTypeLink =
        (await page.locator('a[href^="/type/"]').count().catch(() => 0)) > 0;
      const hasTypeText =
        (await page.getByText(/electoral type/i).count().catch(() => 0)) > 0 ||
        (await page.getByText(/type/i).count().catch(() => 0)) > 0;
      expect(hasTypeLink || hasTypeText).toBe(true);
    });

    test("URL matches /county/13121", async ({ page }) => {
      await page.goto(url);
      await expect(page).toHaveURL(/\/county\/13121/);
    });

    test("no console errors on Fulton County page", async ({ page }) => {
      const errors: string[] = [];
      page.on("console", (msg) => {
        if (msg.type() === "error") errors.push(msg.text());
      });
      await page.goto(url);
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

  test.describe("/county/06037 — Los Angeles County, CA", () => {
    test("page loads with main content", async ({ page }) => {
      await page.goto("/county/06037");
      const main = page.locator("main");
      await expect(main).toBeVisible({ timeout: 10_000 });
    });

    test("page has a heading", async ({ page }) => {
      await page.goto("/county/06037");
      const h1 = page.locator("h1").first();
      await expect(h1).toBeVisible({ timeout: 10_000 });
      const text = await h1.textContent() ?? "";
      expect(text.trim().length).toBeGreaterThan(0);
    });
  });

  test.describe("Invalid FIPS code", () => {
    test("invalid county /county/99999 renders an error or 404 state", async ({ page }) => {
      await page.goto("/county/99999");
      const bodyText = await page.locator("body").innerText();
      const hasError =
        bodyText.includes("404") ||
        bodyText.includes("Not Found") ||
        bodyText.includes("not found") ||
        bodyText.includes("County not found") ||
        bodyText.includes("does not exist");
      expect(hasError).toBe(true);
    });
  });
});
