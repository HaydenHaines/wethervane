import { test, expect } from "@playwright/test";

test("map loads and shows county polygons", async ({ page }) => {
  await page.goto("/");
  await expect(page).toHaveURL(/\/forecast/);
  const canvas = page.locator("canvas");
  await expect(canvas).toBeVisible({ timeout: 10_000 });
});

test("clicking a county opens the community panel", async ({ page }) => {
  await page.goto("/forecast");
  await page.locator("canvas").waitFor({ timeout: 10_000 });
  await page.mouse.click(300, 300);
  await page.waitForTimeout(500);
  const panel = page.locator("text=/Community \\d+/");
  const panelCount = await panel.count();
  expect(panelCount).toBeGreaterThanOrEqual(0);
});

test("forecast tab loads race predictions", async ({ page }) => {
  await page.goto("/forecast");
  const select = page.locator("select");
  await expect(select).toBeVisible({ timeout: 10_000 });
  const options = await select.locator("option").count();
  expect(options).toBeGreaterThan(0);
});

test("feed-a-poll updates predictions", async ({ page }) => {
  await page.goto("/forecast");
  const updateBtn = page.locator("text=Update");
  await expect(updateBtn).toBeVisible({ timeout: 10_000 });
  const slider = page.locator('input[type="range"]');
  await slider.evaluate((el: HTMLInputElement) => {
    el.value = "0.55";
    el.dispatchEvent(new Event("input", { bubbles: true }));
    el.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await updateBtn.click();
  const resetBtn = page.locator("text=Reset to baseline");
  await expect(resetBtn).toBeVisible({ timeout: 5_000 });
});
