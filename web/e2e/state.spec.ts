import { test, expect } from "@playwright/test";

test.describe("State detail pages (/state/[abbr])", () => {
  test.describe("/state/GA — Georgia", () => {
    const url = "/state/GA";

    test("page loads with main content", async ({ page }) => {
      await page.goto(url);
      const main = page.locator("main");
      await expect(main).toBeVisible({ timeout: 10_000 });
    });

    test("page heading contains 'Georgia'", async ({ page }) => {
      await page.goto(url);
      const h1 = page.locator("h1").first();
      await expect(h1).toBeVisible({ timeout: 10_000 });
      const text = await h1.textContent() ?? "";
      expect(text).toContain("Georgia");
    });

    test("page URL matches /state/GA", async ({ page }) => {
      await page.goto(url);
      await expect(page).toHaveURL(/\/state\/GA/i);
    });

    test("county table or county section is present", async ({ page }) => {
      await page.goto(url);
      await expect(page.locator("main")).toBeVisible({ timeout: 10_000 });
      // County table rows or county links
      const hasCounts = await page.locator("table").count().then((c) => c > 0).catch(() => false);
      const hasCountyLinks = await page
        .locator('a[href^="/county/"]')
        .count()
        .then((c) => c > 0)
        .catch(() => false);
      expect(hasCounts || hasCountyLinks).toBe(true);
    });

    test("type distribution section is present", async ({ page }) => {
      await page.goto(url);
      await expect(page.locator("main")).toBeVisible({ timeout: 10_000 });
      // Type distribution section title or type links
      const hasTypeSection =
        (await page.getByText(/type distribution/i).count().catch(() => 0)) > 0 ||
        (await page.locator('a[href^="/type/"]').count().catch(() => 0)) > 0;
      expect(hasTypeSection).toBe(true);
    });

    test("no console errors on GA state page", async ({ page }) => {
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

  test.describe("/state/CA — California", () => {
    test("California state page loads", async ({ page }) => {
      await page.goto("/state/CA");
      const main = page.locator("main");
      await expect(main).toBeVisible({ timeout: 10_000 });
    });

    test("California heading is present", async ({ page }) => {
      await page.goto("/state/CA");
      const h1 = page.locator("h1").first();
      await expect(h1).toBeVisible({ timeout: 10_000 });
      const text = await h1.textContent() ?? "";
      expect(text).toContain("California");
    });
  });

  test.describe("Invalid state abbr", () => {
    test("invalid state /state/XX renders an error or 404 state", async ({ page }) => {
      await page.goto("/state/XX");
      const bodyText = await page.locator("body").innerText();
      const hasError =
        bodyText.includes("404") ||
        bodyText.includes("Not Found") ||
        bodyText.includes("not found") ||
        bodyText.includes("State not found") ||
        bodyText.includes("does not exist");
      expect(hasError).toBe(true);
    });
  });
});
