import { expect, test } from "@playwright/test";

test("race list starts with ten, expands on demand, and searches beyond the initial ten", async ({ page }) => {
  await page.goto("/");
  const rows = page.locator(".race-table tbody tr");
  const toggle = page.getByRole("button", { name: "See all 35 races", exact: true });
  await expect(rows).toHaveCount(10);
  await expect(toggle).toHaveAttribute("aria-expanded", "false");
  await expect(page.getByRole("region", { name: "The races", exact: true })).toContainText("10 of 35 elections");
  await toggle.click();
  await expect(rows).toHaveCount(35);
  const collapse = page.getByRole("button", { name: "Show fewer races", exact: true });
  await expect(collapse).toHaveAttribute("aria-expanded", "true");
  await collapse.click();
  await expect(rows).toHaveCount(10);
  await page.getByLabel("Search states or candidates", { exact: true }).fill("Wyoming");
  await expect(rows).toHaveCount(1);
  await expect(rows).toContainText("Wyoming");
  await expect(page.locator(".race-list-toggle")).toHaveCount(0);
  await page.getByLabel("Search states or candidates", { exact: true }).fill("");
  await expect(rows).toHaveCount(10);
});

test("race details scroll independently before the page bottom on desktop and flow naturally on mobile", async ({ page, isMobile }) => {
  if (!isMobile) await page.setViewportSize({ width: 1280, height: 700 });
  await page.goto("/?run=current-polling&race=2026%3ANE%3Asenate%3Aregular");
  const detail = page.getByRole("complementary", { name: "Selected race detail", exact: true });
  await expect(detail.getByRole("heading", { name: "Nebraska", exact: true })).toBeVisible();
  if (isMobile) {
    await expect(detail).toHaveCSS("position", "static");
    await expect(detail).toHaveCSS("max-height", "none");
    await detail.locator(".source-date").scrollIntoViewIfNeeded();
    await expect(detail.locator(".source-date")).toBeInViewport();
    return;
  }
  await expect(page.locator(".map-svg [role=button]")).toHaveCount(35);
  await page.locator(".workspace").evaluate(node => window.scrollTo({ top: node.getBoundingClientRect().top + window.scrollY + 120, behavior: "instant" }));
  const dimensions = await detail.evaluate(node => ({ height: node.clientHeight, scrollHeight: node.scrollHeight, top: node.getBoundingClientRect().top }));
  expect(dimensions.height).toBeLessThanOrEqual(660);
  expect(dimensions.scrollHeight).toBeGreaterThan(dimensions.height);
  expect(dimensions.top).toBeCloseTo(20, 0);
  const pageScroll = await page.evaluate(() => window.scrollY);
  expect(await page.evaluate(() => scrollY + innerHeight < document.documentElement.scrollHeight - 200)).toBeTruthy();
  await detail.hover();
  // Stay inside the panel's range: scrolling past its bottom may naturally continue the page.
  await page.mouse.wheel(0, Math.min(200, Math.floor((dimensions.scrollHeight - dimensions.height) / 2)));
  await expect.poll(() => detail.evaluate(node => node.scrollTop)).toBeGreaterThan(0);
  expect(await page.evaluate(() => window.scrollY)).toBe(pageScroll);
  await detail.focus();
  await page.keyboard.press("End");
  await expect(detail.locator(".source-date")).toBeInViewport();
  await page.getByRole("button", { name: /^Texas, 1 election,/ }).click();
  await expect(detail.getByRole("heading", { name: "Texas", exact: true })).toBeVisible();
  await expect.poll(() => detail.evaluate(node => node.scrollTop)).toBe(0);
});

test("chamber heading and active tab track forecast control and keep a House race subset neutral", async ({ page }) => {
  await page.goto("/");
  const heading = page.locator("h1 .chamber-name");
  const senate = page.getByRole("tab", { name: "Senate", exact: true });
  await expect(heading).toHaveAttribute("data-chamber-outlook", "tossup");
  await expect(heading).toHaveCSS("color", "rgb(123, 101, 142)");
  await expect(senate).toHaveCSS("color", "rgb(123, 101, 142)");
  await page.locator("details.stack-builder > summary").click();
  await page.getByRole("checkbox", { name: "Include polling evidence", exact: true }).uncheck();
  await expect(heading).toHaveAttribute("data-chamber-outlook", "R");
  await expect(heading).toHaveCSS("color", "rgb(163, 85, 63)");
  await expect(senate).toHaveCSS("color", "rgb(163, 85, 63)");
  await page.goto("/?run=history-2022-FP-1");
  await expect(heading).toHaveAttribute("data-chamber-outlook", "D");
  await expect(heading).toHaveCSS("color", "rgb(62, 102, 131)");
  await expect(senate).toHaveCSS("color", "rgb(62, 102, 131)");
  await page.getByRole("tab", { name: "House", exact: true }).click();
  await expect(page.getByRole("heading", { name: /The House,.*in perspective/ })).toBeVisible();
  await expect(heading).toHaveAttribute("data-chamber-outlook", "unavailable");
  await expect(heading).toHaveCSS("color", "rgb(33, 57, 75)");
  await expect(page.getByRole("tab", { name: "House", exact: true })).toHaveCSS("color", "rgb(33, 57, 75)");
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBeTruthy();
});
