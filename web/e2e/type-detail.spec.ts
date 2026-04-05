import { test, expect } from "@playwright/test";

test.describe("Type detail pages (/type/[id])", () => {
  test.describe("Type 0 — first type", () => {
    const url = "/type/0";

    test("page loads with main content", async ({ page }) => {
      await page.goto(url);
      const main = page.locator("main");
      await expect(main).toBeVisible({ timeout: 10_000 });
    });

    test("page has a heading", async ({ page }) => {
      await page.goto(url);
      const h1 = page.locator("h1").first();
      await expect(h1).toBeVisible({ timeout: 10_000 });
      const text = await h1.textContent() ?? "";
      expect(text.trim().length).toBeGreaterThan(0);
    });

    test("page has breadcrumb navigation", async ({ page }) => {
      await page.goto(url);
      const breadcrumb = page.locator("nav[aria-label='Breadcrumb']");
      await expect(breadcrumb).toBeVisible({ timeout: 10_000 });
    });

    test("demographics table or section is present", async ({ page }) => {
      await page.goto(url);
      await expect(page.locator("main")).toBeVisible({ timeout: 10_000 });
      // Demographics renders as a table or definition list
      const hasDemoTable = await page.locator("table").count().then((c) => c > 0).catch(() => false);
      const hasDemoSection = await page.getByText(/demographics/i).count().then((c) => c > 0).catch(() => false);
      expect(hasDemoTable || hasDemoSection).toBe(true);
    });

    test("has a back link to /types", async ({ page }) => {
      await page.goto(url);
      const backLink = page.locator('a[href="/types"]').first();
      await expect(backLink).toBeVisible({ timeout: 10_000 });
    });

    test("county list or county section is present", async ({ page }) => {
      await page.goto(url);
      await expect(page.locator("main")).toBeVisible({ timeout: 10_000 });
      // Counties section contains county links or table rows
      const countyLinks = page.locator('a[href^="/county/"]');
      await countyLinks.first().waitFor({ state: "attached", timeout: 10_000 });
      const count = await countyLinks.count();
      expect(count).toBeGreaterThan(0);
    });

    test("no console errors on type 0 page", async ({ page }) => {
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

  test.describe("Type 99 — last type", () => {
    test("type 99 page loads with main content", async ({ page }) => {
      await page.goto("/type/99");
      const main = page.locator("main");
      await expect(main).toBeVisible({ timeout: 10_000 });
    });

    test("type 99 page has a heading", async ({ page }) => {
      await page.goto("/type/99");
      const h1 = page.locator("h1").first();
      await expect(h1).toBeVisible({ timeout: 10_000 });
      const text = await h1.textContent() ?? "";
      expect(text.trim().length).toBeGreaterThan(0);
    });
  });

  test.describe("Invalid type ID", () => {
    test("invalid type /type/999 renders an error or 404 state", async ({ page }) => {
      await page.goto("/type/999");
      // Either a 404 page or a not-found message within the page
      const bodyText = await page.locator("body").innerText();
      const hasError =
        bodyText.includes("404") ||
        bodyText.includes("Not Found") ||
        bodyText.includes("not found") ||
        bodyText.includes("Type not found") ||
        bodyText.includes("does not exist");
      expect(hasError).toBe(true);
    });
  });
});
