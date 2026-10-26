import { test, expect } from "@playwright/test";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

const read = (path: string) => JSON.parse(readFileSync(resolve(path), "utf8"));
const senateRegistry = read("public/data/explorer/registry.json");
const houseCatalog = read("data/house-live/catalog.json");

for (const chamber of ["senate", "house"] as const) {
  test(`${chamber}: same-page guide starts standard, renders every exact model, and preserves the visitor's forecast`, async ({ page }, testInfo) => {
    test.setTimeout(240_000);
    const errors: string[] = [];
    page.on("pageerror", error => errors.push(error.message));
    const initialRun = chamber === "senate" ? "current-fundamentals" : "house-current-fundamentals";
    const defaultId = chamber === "senate" ? senateRegistry.defaults["2026"] : houseCatalog.default_id;
    const entries = chamber === "senate" ? senateRegistry.runs.filter((entry: { year: number }) => entry.year === 2026) : houseCatalog.entries;
    await page.goto(chamber === "senate" ? `/?run=${initialRun}` : `/?chamber=house&edition=current&run=${initialRun}`);
    await expect(page.locator(".hero")).toBeVisible();
    const originalUrl = page.url();
    const originalHero = await page.locator(".hero").textContent();
    const trigger = page.getByRole("button", { name: "Election night scorecard", exact: true });
    await trigger.click();
    const panel = page.locator("#election-night-scorecard");
    await expect(panel.getByRole("heading", { name: "Election night scorecard", exact: true })).toBeFocused();
    await expect(panel.locator(".night-edition")).toContainText("Standard forecast");
    await expect(panel).toHaveAttribute("aria-busy", "false");
    await expect(panel.locator(".night-featured .night-race")).toHaveCount(6);
    const watchIds = await panel.locator(".night-featured .night-race").evaluateAll(nodes => nodes.map(node => node.getAttribute("data-scorecard-race")));
    await panel.locator(".night-models summary").click();
    const select = panel.getByLabel("Scorecard model");
    await expect(select).toHaveValue(defaultId);
    await expect(select.locator("option")).toHaveCount(16);
    for (const entry of entries) {
      const raw = chamber === "senate" ? read(`public${entry.href}`) : read(`data/house-live/${entry.result.path}`);
      const digest = chamber === "senate" ? entry.sha256 : entry.result.sha256;
      const missed = raw.races.reduce((sum: number, race: { outcomes: { winProbability?: number; p_win?: number }[] }) => sum +
        1 - Math.max(...race.outcomes.map(outcome => chamber === "senate" ? outcome.winProbability! : outcome.p_win!)), 0);
      await select.selectOption(entry.id);
      await expect(panel.locator(".night-method code")).toHaveText(digest);
      await expect(panel.locator(".night-edition strong")).toContainText(entry.label);
      await expect(panel.locator(".night-big-number")).toContainText(missed.toFixed(1));
      expect(await panel.locator(".night-featured .night-race").evaluateAll(nodes => nodes.map(node => node.getAttribute("data-scorecard-race")))).toEqual(watchIds);
      expect(page.url()).toBe(originalUrl);
      expect(await page.locator(".hero").textContent()).toBe(originalHero);
    }
    await expect(panel.locator(".night-rubric")).toContainText("No letter grade assigned");
    await expect(panel.locator(".night-edition")).toContainText("have not yet been locked");
    await expect(panel.locator(".night-uncertainty-grid")).toContainText("not a limit or a measured accuracy record");
    await panel.locator(".night-featured").screenshot({ path: testInfo.outputPath(`${chamber}-featured.png`) });
    await panel.locator(".night-uncertainty-grid").screenshot({ path: testInfo.outputPath(`${chamber}-uncertainty.png`) });
    await panel.locator(".night-all-races summary").click();
    await expect(panel.locator(".night-predictions .night-race")).toHaveCount(10);
    if (chamber === "house") {
      const defaultEntry = entries.find((entry: { id: string }) => entry.id === defaultId);
      const unsupported = read(`data/house-live/${defaultEntry.result.path}`).races.find((race: { conditional_shares: unknown }) => race.conditional_shares === null);
      await panel.getByLabel("Find a race or candidate").fill(unsupported.district_key);
      await expect(panel.locator(".night-predictions .night-race")).toHaveCount(1);
      await expect(panel.locator(".night-predictions")).toContainText("No supported two-candidate margin");
    }
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await panel.getByRole("button", { name: "Close scorecard", exact: true }).click();
    await expect(panel).toHaveCount(0);
    await expect(trigger).toBeFocused();
    expect(page.url()).toBe(originalUrl);
    await trigger.click();
    await expect(panel.locator(".night-edition")).toContainText("Standard forecast");
    await expect(panel.locator(".night-models select")).toHaveValue(defaultId);
    expect(errors).toEqual([]);
  });
}
