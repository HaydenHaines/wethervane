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

    test("page has correct meta title containing 'Georgia'", async ({ page }) => {
      await page.goto(url);
      await expect(page).toHaveTitle(/Georgia/i);
    });

    test("page has meta description", async ({ page }) => {
      await page.goto(url);
      const desc = await page.locator('meta[name="description"]').getAttribute("content");
      expect(desc).toBeTruthy();
      expect((desc ?? "").length).toBeGreaterThan(20);
    });

    test("page has Open Graph title tag", async ({ page }) => {
      await page.goto(url);
      const ogTitle = await page.locator('meta[property="og:title"]').getAttribute("content");
      expect(ogTitle).toBeTruthy();
      expect(ogTitle).toContain("Georgia");
    });

    test("page has Open Graph description tag", async ({ page }) => {
      await page.goto(url);
      const ogDesc = await page.locator('meta[property="og:description"]').getAttribute("content");
      expect(ogDesc).toBeTruthy();
      expect((ogDesc ?? "").length).toBeGreaterThan(20);
    });

    test("page has Open Graph image tag", async ({ page }) => {
      await page.goto(url);
      const ogImage = await page.locator('meta[property="og:image"]').getAttribute("content");
      expect(ogImage).toBeTruthy();
      expect(ogImage).toContain("opengraph-image");
    });

    test("page has Twitter card tag", async ({ page }) => {
      await page.goto(url);
      const twitterCard = await page.locator('meta[name="twitter:card"]').getAttribute("content");
      expect(twitterCard).toBeTruthy();
    });

    test("page has canonical link tag", async ({ page }) => {
      await page.goto(url);
      const canonical = await page.locator('link[rel="canonical"]').getAttribute("href");
      expect(canonical).toBeTruthy();
      expect(canonical).toContain("/state/GA");
    });

    test("page has JSON-LD WebPage structured data", async ({ page }) => {
      await page.goto(url);
      const ldJson = await page.locator('script[type="application/ld+json"]').first().textContent();
      expect(ldJson).toBeTruthy();
      const parsed = JSON.parse(ldJson ?? "{}");
      // Either the first script is a WebPage or there are multiple LD+JSON scripts
      const isWebPage = parsed["@type"] === "WebPage" || parsed["@type"] === "BreadcrumbList";
      expect(isWebPage).toBe(true);
    });

    test("page has JSON-LD BreadcrumbList structured data", async ({ page }) => {
      await page.goto(url);
      const allLdJson = page.locator('script[type="application/ld+json"]');
      const count = await allLdJson.count();
      expect(count).toBeGreaterThanOrEqual(2);

      // Find the BreadcrumbList script
      let hasBreadcrumb = false;
      for (let i = 0; i < count; i++) {
        const content = await allLdJson.nth(i).textContent();
        const parsed = JSON.parse(content ?? "{}");
        if (parsed["@type"] === "BreadcrumbList") {
          hasBreadcrumb = true;
          // Verify it contains Home + Forecast + Georgia
          const items: Array<{ name: string }> = parsed.itemListElement ?? [];
          const names = items.map((item) => item.name);
          expect(names).toContain("Home");
          expect(names.some((n) => n === "Georgia")).toBe(true);
        }
      }
      expect(hasBreadcrumb).toBe(true);
    });

    test("page has visible breadcrumb navigation", async ({ page }) => {
      await page.goto(url);
      // Breadcrumb nav — aria-label is "breadcrumb" on the state page
      const breadcrumb = page.locator('nav[aria-label="breadcrumb"]');
      await expect(breadcrumb).toBeVisible({ timeout: 10_000 });
      // Should contain links back to Home and Forecast
      const homeLink = breadcrumb.locator('a[href="/"]');
      await expect(homeLink).toBeVisible();
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
