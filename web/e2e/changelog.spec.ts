import { test, expect } from "@playwright/test";

test.describe("Changelog page", () => {
  test("page loads with main content", async ({ page }) => {
    await page.goto("/changelog");
    const main = page.locator("main");
    await expect(main).toBeVisible({ timeout: 10_000 });
  });

  test("page has a heading", async ({ page }) => {
    await page.goto("/changelog");
    const h1 = page.locator("h1").first();
    await expect(h1).toBeVisible({ timeout: 10_000 });
    const text = await h1.textContent() ?? "";
    expect(text.trim().length).toBeGreaterThan(0);
  });

  test("page shows forecast change entries or empty state", async ({ page }) => {
    await page.goto("/changelog");
    await expect(page.locator("main")).toBeVisible({ timeout: 10_000 });
    // Either list items / entries exist, or an empty-state message
    const mainText = await page.locator("main").innerText();
    expect(mainText.trim().length).toBeGreaterThan(0);
  });

  test("page URL matches /changelog", async ({ page }) => {
    await page.goto("/changelog");
    await expect(page).toHaveURL(/\/changelog/);
  });

  test("no console errors on changelog page", async ({ page }) => {
    const errors: string[] = [];
    page.on("console", (msg) => {
      if (msg.type() === "error") errors.push(msg.text());
    });
    await page.goto("/changelog");
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
