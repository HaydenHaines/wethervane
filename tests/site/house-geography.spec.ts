import { expect, test } from "@playwright/test";

test("published district maps show modeled and excluded races, including states without forecasts", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto("/?chamber=house");
  const map = page.getByRole("region", { name: "House map", exact: true });
  await expect(map.locator("svg path[data-district-key]")).toHaveCount(435);
  await expect(map.locator("svg path[data-forecast=available]")).toHaveCount(375);
  await expect(map.locator("svg path[data-forecast=unavailable]")).toHaveCount(60);
  const focus = page.getByLabel("Focus on a state", { exact: true });
  await expect(focus.locator("option")).toHaveCount(51);
  await expect(map.locator(".map-legend")).toContainText("Not in this model");
  expect(await map.locator("svg path").evaluateAll(nodes => nodes.every(node => {
    const d = node.getAttribute("d"); return d && !/NaN|Infinity/.test(d);
  }))).toBe(true);
  // Alaska is outside this legacy two-candidate study, but its map must still be explorable.
  await focus.selectOption("AK");
  await expect(map.locator("svg path[data-district-key]")).toHaveCount(1);
  await expect(map.locator("svg path[data-district-key=AK-00]")).toHaveAttribute("data-forecast", "unavailable");
  await focus.selectOption("CA");
  await expect(map.locator("svg path[data-district-key]")).toHaveCount(52);
  const race = map.locator('[data-district-key="CA-22"]');
  await race.focus();
  await page.keyboard.press("Enter");
  await expect(page.getByRole("complementary", { name: "Selected House race detail" }).getByRole("heading", { name: "CA-22", exact: true })).toBeVisible();
  await expect(race).toHaveAttribute("aria-pressed", "true");
  await page.getByRole("group", { name: "Map view", exact: true }).getByRole("button", { name: "Actual winners", exact: true }).click();
  await expect(map).toContainText("Actual winners are shown only for the included races.");
  await expect(race).toHaveAttribute("aria-label", /observed winner/);
  await expect(map.locator(".map-legend")).toContainText("Not in this model");
  await map.getByText("District boundary source", { exact: true }).click();
  await expect(map.locator(".district-map-provenance a")).toHaveAttribute("href", /^https:\/\/www2.census.gov\//);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  expect(errors).toEqual([]);
});

test("changing election year replaces geometry as well as forecasts", async ({ page }) => {
  await page.goto("/?chamber=house");
  await expect(page.locator(".district-map-svg")).toBeVisible();
  const configuration = page.locator("details.house-configuration");
  await configuration.locator(":scope > summary").click();
  const fingerprints = new Set<string>();
  for (const [year, california, modeled] of [[2018, 53, 371], [2020, 53, 383], [2022, 52, 373], [2024, 52, 375]]) {
    await configuration.getByRole("group", { name: "Election year", exact: true }).getByRole("button", { name: String(year), exact: true }).click();
    await expect(page.locator(".district-map-toolbar h3")).toHaveText(`${year} House districts`);
    const focus = page.getByLabel("Focus on a state", { exact: true });
    await focus.selectOption("");
    await expect(page.locator(".district-map-svg path[data-district-key]")).toHaveCount(435);
    await expect(page.locator(".district-map-svg path[data-forecast=available]")).toHaveCount(modeled);
    await focus.selectOption("CA");
    await expect(page.locator(".district-map-svg path[data-district-key]")).toHaveCount(california);
    fingerprints.add((await page.locator(".district-map-provenance code").textContent())!);
  }
  expect(fingerprints.size).toBe(4);
});
