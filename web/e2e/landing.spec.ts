import { test, expect } from "@playwright/test";

test.describe("Landing page", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
  });

  test("page renders with main content", async ({ page }) => {
    const main = page.locator("main");
    await expect(main).toBeVisible({ timeout: 10_000 });
  });

  test("closest races heading appears", async ({ page }) => {
    const heading = page.getByText("Closest Races");
    await expect(heading).toBeVisible({ timeout: 10_000 });
  });

  test("race ticker section is present", async ({ page }) => {
    const tickerHeading = page.getByText("Closest Races");
    await expect(tickerHeading).toBeVisible({ timeout: 10_000 });
  });

  test("entry point cards link to correct pages", async ({ page }) => {
    const forecastLink = page.locator('a[href="/forecast"]');
    const typesLink = page.locator('a[href="/types"]');
    const methodologyLink = page.locator('a[href="/methodology"]');

    await expect(forecastLink.first()).toBeVisible({ timeout: 10_000 });
    await expect(typesLink.first()).toBeVisible();
    await expect(methodologyLink.first()).toBeVisible();
  });

  test("entry point cards have descriptive text", async ({ page }) => {
    // Entry cards contain descriptive text about each section
    await expect(page.getByText("See the full forecast")).toBeVisible({ timeout: 10_000 });
    await expect(page.getByText("Explore electoral types")).toBeVisible();
    await expect(page.getByText("How the model works")).toBeVisible();
  });

  test("footer renders with navigation links", async ({ page }) => {
    const footer = page.locator("footer");
    await expect(footer).toBeVisible({ timeout: 10_000 });
    const footerNav = footer.locator("nav");
    await expect(footerNav).toBeVisible();
    const links = footerNav.locator("a");
    const linkCount = await links.count();
    expect(linkCount).toBeGreaterThan(0);
  });

  test("footer contains WetherVane branding text", async ({ page }) => {
    const footer = page.locator("footer");
    await expect(footer).toBeVisible({ timeout: 10_000 });
    await expect(footer).toContainText("WetherVane");
  });
});
