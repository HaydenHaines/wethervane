import { test, expect } from "@playwright/test";

/**
 * E2e tests for the FundamentalsCard — "National Environment" section.
 *
 * The card is a client component that fetches /api/v1/forecast/fundamentals
 * and renders on both the senate overview page and race detail pages.
 */
test.describe("FundamentalsCard", () => {
  test.describe("Senate overview page", () => {
    test("National Environment card appears on senate overview", async ({ page }) => {
      await page.goto("/forecast/senate");

      // Wait for page to fully load (heading must be visible first)
      await expect(page.locator("h1")).toBeVisible({ timeout: 30_000 });

      // The card uses aria-label="National Environment"
      const card = page.locator('[aria-label="National Environment"]');
      await expect(card).toBeVisible({ timeout: 15_000 });
    });

    test("National Environment card displays a combined shift value", async ({
      page,
    }) => {
      await page.goto("/forecast/senate");

      await expect(page.locator("h1")).toBeVisible({ timeout: 30_000 });

      const card = page.locator('[aria-label="National Environment"]');
      await expect(card).toBeVisible({ timeout: 15_000 });

      // The combined shift is formatted as "D+X.X", "R+X.X", or "EVEN"
      // and has an aria-label starting with "Combined forecast shift:"
      const shiftBadge = page.locator(
        '[aria-label^="Combined forecast shift:"]',
      );
      await expect(shiftBadge).toBeVisible({ timeout: 10_000 });

      const shiftText = await shiftBadge.textContent();
      // Must match D+N.N, R+N.N, or EVEN
      expect(shiftText).toMatch(/^(D\+\d+\.\d|R\+\d+\.\d|EVEN)$/);
    });

    test("National Environment card shows indicator contributions", async ({
      page,
    }) => {
      await page.goto("/forecast/senate");

      await expect(page.locator("h1")).toBeVisible({ timeout: 30_000 });

      const card = page.locator('[aria-label="National Environment"]');
      await expect(card).toBeVisible({ timeout: 15_000 });

      // Each indicator has an aria-label on its contribution span
      const approvalContrib = page.locator(
        '[aria-label^="Presidential Approval contribution:"]',
      );
      await expect(approvalContrib).toBeVisible({ timeout: 10_000 });
    });

    test("National Environment card shows model uncertainty", async ({
      page,
    }) => {
      await page.goto("/forecast/senate");

      await expect(page.locator("h1")).toBeVisible({ timeout: 30_000 });

      const card = page.locator('[aria-label="National Environment"]');
      await expect(card).toBeVisible({ timeout: 15_000 });

      // Uncertainty text includes LOO RMSE and training cycle count
      await expect(card.getByText(/LOO RMSE/)).toBeVisible({ timeout: 10_000 });
    });
  });

  test.describe("Race detail page", () => {
    const raceSlug = "/forecast/2026-ga-senate";

    test("National Environment card appears on race detail page", async ({
      page,
    }) => {
      await page.goto(raceSlug);

      // Wait for the article to render
      await expect(page.locator("article#main-content")).toBeVisible({
        timeout: 15_000,
      });

      // The card may be below the fold — check it's attached to the DOM
      const card = page.locator('[aria-label="National Environment"]');
      await expect(card).toBeAttached({ timeout: 15_000 });
    });

    test("National Environment card on race detail shows a shift value", async ({
      page,
    }) => {
      await page.goto(raceSlug);

      await expect(page.locator("article#main-content")).toBeVisible({
        timeout: 15_000,
      });

      const shiftBadge = page.locator(
        '[aria-label^="Combined forecast shift:"]',
      );
      await expect(shiftBadge).toBeAttached({ timeout: 15_000 });

      const shiftText = await shiftBadge.textContent();
      expect(shiftText).toMatch(/^(D\+\d+\.\d|R\+\d+\.\d|EVEN)$/);
    });
  });
});
