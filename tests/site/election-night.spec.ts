import { test, expect } from "@playwright/test";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

const read = (path: string) => JSON.parse(readFileSync(resolve(path), "utf8"));
const senateRegistry = read("public/data/explorer/registry.json");
const houseCatalog = read("data/house-live/catalog.json");
const standardReference = read("public/data/election-night/references.json").entries[0].reference;
const coreReferences = read("public/data/election-night/core-references.json").entries;
const displayProbability = (p: number) => p > 0 && p < .001 ? "<0.1%" : p < 1 && p > .999 ? ">99.9%" : `${(100 * p).toFixed(1)}%`;

test("reviewed expected errors follow only their exact Senate edition and remain distinct from observed scores", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Election night scorecard", exact: true }).click();
  const panel = page.locator("#election-night-scorecard");
  await expect(panel).toHaveAttribute("aria-busy", "false");
  const margin = panel.locator(".night-margin-reference");
  await expect(margin.locator(".night-reference-value")).toContainText("5.6");
  await expect(margin.locator(".night-reference-band")).toContainText("4.0 to 7.9 points");
  await expect(margin).toContainText("not measured accuracy");
  await expect(panel.locator(".night-miss-reference .night-reference-band")).toContainText("1 to 7 wrong choices");
  await expect(panel.locator(".night-featured .night-race-expected-error")).toHaveCount(6);
  await expect(panel.locator(".night-featured .night-margin-chances")).toHaveCount(6);
  for (const card of await panel.locator(".night-featured .night-race").all()) {
    const id = await card.getAttribute("data-scorecard-race");
    const row = standardReference.cohort.find((item: { race_id: string }) => item.race_id === id);
    await expect(card.locator(".night-close-chance strong")).toHaveText(displayProbability(row.close_margin_probability_2pp));
    await expect(card.locator(".night-margin-chances dd")).toHaveText([
      displayProbability(row.first_decisive_margin_probability_10pp), displayProbability(row.second_decisive_margin_probability_10pp),
    ]);
  }
  await expect(panel.locator(".night-rubric-table tbody tr")).toHaveCount(5);
  await expect(panel.locator(".night-rubric-table tbody td")).toHaveText([
    "At least 20% lower", "10% to less than 20% lower", "0% to less than 10% lower", "More than 0%, up to 10% higher", "More than 10% higher",
  ]);
  await expect(panel.locator(".night-baseline-note")).toContainText("October 20 edition (earlier issued)");
  await expect(panel.getByRole("complementary", { name: "Senate evaluation coverage" })).toContainText("All 35 predictions stay visible. The primary grade covers 32 candidate pairs.");
  await expect(panel.locator(".night-baseline-note")).toContainText("all-35 forecast reference remains separate");
  await expect(panel.locator(".night-rubric")).toContainText("no better than D");
  await expect(panel.locator(".night-rubric")).toContainText("An F stays an F");
  await panel.locator(".night-reference-details > summary").click();
  await expect(panel.locator(".night-reference-details")).toContainText("31.5 of 35");
  await expect(panel.locator(".night-reference-details")).toContainText("26 to 35");
  await expect(panel.locator(".night-reference-details")).toContainText("−5.5 to +5.8 points");
  await expect(panel.locator(".night-reference-details")).toContainText("near zero by construction");
  await panel.locator(".night-models > summary").click();
  await panel.getByLabel("Scorecard model").selectOption("current-fundamentals");
  await expect(margin).toContainText("No reviewed error reference for this edition");
  await expect(panel.locator(".night-race-expected-error")).toHaveCount(0);
  await expect(panel.locator(".night-reference-details")).toHaveCount(0);
  await expect(panel.locator(".night-margin-chances")).toHaveCount(0);
  await expect(panel.locator(".night-baseline-note")).toHaveCount(0);
  await expect(panel.locator(".night-cohort-note")).toHaveCount(0);
  await panel.getByLabel("Scorecard model").selectOption("current-polling");
  await expect(margin.locator(".night-reference-value")).toContainText("5.6");
  await expect(panel.locator(".night-summary")).toContainText("Not scored");
  await expect(panel.locator(".night-featured .night-margin-chances")).toHaveCount(6);
  await expect(panel.locator(".night-edition")).toContainText("have not yet been locked");
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

test("an invalid uncertainty reference leaves the forecast available and can be retried", async ({ page }) => {
  const path = "**/data/election-night/references.json";
  await page.route(path, async route => {
    const response = await route.fetch();
    await route.fulfill({ response, body: (await response.text()) + " " });
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Election night scorecard", exact: true }).click();
  const panel = page.locator("#election-night-scorecard");
  await expect(panel.locator(".night-reference-error")).toContainText("could not be verified");
  await expect(panel.locator(".night-featured .night-race")).toHaveCount(6);
  await expect(panel.locator(".night-race-expected-error")).toHaveCount(0);
  await expect(panel.locator(".night-reference-value")).toHaveCount(0);
  await expect(panel.locator(".night-primary-reference")).toHaveAttribute("data-reference-run", "current-polling");
  await expect(panel.locator(".night-primary-margin-value")).toContainText("5.7 percentage points");
  await page.unroute(path);
  await panel.getByRole("button", { name: "Retry uncertainty reference", exact: true }).click();
  await expect(panel.locator(".night-reference-value")).toContainText("5.6");
  await expect(panel.locator(".night-reference-error")).toHaveCount(0);
});

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
    await panel.locator(".night-featured .night-race-definitions > summary").first().click();
    await expect(panel.locator(".night-featured .night-race").first()).toContainText(chamber === "senate"
      ? "the winning event used by this model’s two-contender approximation" : "pair event is separate from the winner chances");
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
      if (chamber === "house") await expect(panel.locator(":scope > .night-input-notice")).toContainText("Christopher Schmidt");
      expect(await panel.locator(".night-featured .night-race").evaluateAll(nodes => nodes.map(node => node.getAttribute("data-scorecard-race")))).toEqual(watchIds);
      expect(page.url()).toBe(originalUrl);
      expect(await page.locator(".hero").textContent()).toBe(originalHero);
    }
    await expect(panel.locator(".night-rubric")).toContainText("No letter grade assigned");
    await expect(panel.locator(".night-edition")).toContainText("have not yet been locked");
    await expect(panel.locator(".night-miss-reference")).toContainText("not a limit or a measured accuracy record");
    await panel.locator(".night-featured").screenshot({ path: testInfo.outputPath(`${chamber}-featured.png`) });
    await panel.locator(".night-uncertainty-grid").first().screenshot({ path: testInfo.outputPath(`${chamber}-uncertainty.png`) });
    await panel.locator(".night-all-races > summary").click();
    await expect(panel.locator(".night-predictions .night-race")).toHaveCount(10);
    if (chamber === "house") {
      const defaultEntry = entries.find((entry: { id: string }) => entry.id === defaultId);
      const unsupported = read(`data/house-live/${defaultEntry.result.path}`).races.find((race: { conditional_shares: unknown }) => race.conditional_shares === null);
      await panel.getByLabel("Find a race or candidate").fill(unsupported.district_key);
      await expect(panel.locator(".night-predictions .night-race")).toHaveCount(1);
      await expect(panel.locator(".night-predictions")).toContainText("No supported two-candidate margin");
      await panel.getByLabel("Find a race or candidate").fill("NY-21");
      await expect(panel.locator(".night-predictions .night-input-notice")).toContainText("excludes two polls");
      await expect(panel.locator(".night-predictions .night-input-notice a")).toHaveCount(3);
      await panel.locator(".night-method > summary").click();
      await expect(panel.locator(".night-method .night-input-notice")).toContainText("October 25 estimates");
      await expect(panel.locator(".night-method")).toContainText("Source capture notes");
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

test("an open scorecard follows chamber tabs and browser history with each chamber's standard forecast", async ({ page }) => {
  await page.goto("/?run=current-fundamentals");
  await page.getByRole("button", { name: "Election night scorecard", exact: true }).click();
  const panel = page.locator("#election-night-scorecard");
  await expect(panel).toHaveAttribute("data-scorecard-chamber", "senate");
  await expect(panel.locator(".night-models select")).toHaveValue(senateRegistry.defaults["2026"]);
  await page.getByRole("tab", { name: "House", exact: true }).click();
  await expect(panel).toHaveAttribute("data-scorecard-chamber", "house");
  await expect(panel.locator(".night-models select")).toHaveValue(houseCatalog.default_id);
  await expect(panel.getByRole("heading", { name: "Election night scorecard", exact: true })).toBeFocused();
  await page.goBack();
  await expect(panel).toHaveAttribute("data-scorecard-chamber", "senate");
  await expect(panel.locator(".night-models select")).toHaveValue(senateRegistry.defaults["2026"]);
  await expect(page).toHaveURL(/run=current-fundamentals/);
  await page.goForward();
  await expect(panel).toHaveAttribute("data-scorecard-chamber", "house");
  await expect(panel.locator(".night-models select")).toHaveValue(houseCatalog.default_id);
  await panel.getByRole("button", { name: "Close scorecard", exact: true }).click();
  await expect(page.getByRole("button", { name: "Election night scorecard", exact: true })).toBeFocused();
});


test("the separate fixed32 grading-group reference keeps exact F and FP values beside the full35 forecast", async ({ page }, testInfo) => {
  await page.goto("/?run=current-fundamentals");
  await expect(page.locator(".hero")).toBeVisible();
  const originalUrl = page.url();
  await page.getByRole("button", { name: "Election night scorecard", exact: true }).click();
  const panel = page.locator("#election-night-scorecard");
  await expect(panel).toHaveAttribute("aria-busy", "false");
  await panel.locator(".night-models > summary").click();
  for (const id of ["current-polling", "current-fundamentals", "current-polling"]) {
    await panel.getByLabel("Scorecard model").selectOption(id);
    const source = coreReferences.find((entry: { runId: string }) => entry.runId === id);
    const native = source.reference, primary = panel.getByRole("article", { name: "Primary grading group expected errors" });
    await expect(primary).toHaveAttribute("data-reference-run", id);
    await expect(primary.getByRole("heading")).toHaveText("The 32 candidate pairs used for the primary grade");
    await expect(primary.locator(".night-primary-margin-value")).toHaveText(`${native.distributions.mean_absolute_margin_error_pp.mean.toFixed(1)} percentage points`);
    await expect(primary.locator(".night-primary-margin-range")).toContainText(`${native.distributions.mean_absolute_margin_error_pp.quantiles.p05.toFixed(1)} to ${native.distributions.mean_absolute_margin_error_pp.quantiles.p95.toFixed(1)} points`);
    await expect(primary.locator(".night-primary-miss-value")).toHaveText(`${native.distributions.modal_pair_winner_miss_count.mean.toFixed(1)} wrong top choices out of 32`);
    await expect(primary.locator(".night-primary-miss-range")).toContainText(`${native.distributions.modal_pair_winner_miss_count.quantiles.p05} to ${native.distributions.modal_pair_winner_miss_count.quantiles.p95} wrong choices`);
    await expect(primary).toContainText("not observed errors or grades");
    await expect(primary).toContainText("does not combine simulations from different models");
    await expect(panel.getByRole("heading", { name: "All 35 modeled contests", exact: true })).toBeVisible();
    await expect(panel.locator(".night-featured .night-race")).toHaveCount(6);
    if (id === "current-polling") {
      await expect(panel.locator(".night-margin-reference .night-reference-value")).toContainText("5.6");
      await expect(panel.locator(".night-margin-reference")).toContainText("across 35 races");
    } else {
      await expect(panel.locator(".night-margin-reference")).toContainText("No reviewed error reference for this edition");
      await expect(panel.locator(".night-reference-value")).toHaveCount(0);
      await expect(panel.locator(".night-method code")).toHaveText(source.resultSha256);
    }
    expect(page.url()).toBe(originalUrl);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  }
  await panel.locator(".night-primary-reference").screenshot({ path: testInfo.outputPath("senate-primary32-reference.png") });
  await panel.getByLabel("Scorecard model").selectOption("current-cycle-low");
  await expect(panel.locator(".night-primary-reference")).toHaveCount(0);
  await expect(panel.locator(".night-reference-error")).toHaveCount(0);
});

test("an invalid fixed32 attachment preserves the forecast and35 reference, then retries independently", async ({ page }) => {
  const path = "**/data/election-night/core-references.json";
  await page.route(path, async route => {
    const response = await route.fetch();
    await route.fulfill({ response, body: (await response.text()) + " " });
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Election night scorecard", exact: true }).click();
  const panel = page.locator("#election-night-scorecard");
  await expect(panel.locator(".night-reference-error")).toContainText("32-race expected-error reference could not be verified");
  await expect(panel.locator(".night-primary-reference")).toHaveCount(0);
  await expect(panel.locator(".night-margin-reference .night-reference-value")).toContainText("5.6");
  await expect(panel.locator(".night-featured .night-race")).toHaveCount(6);
  await expect(panel.locator(".night-summary")).toContainText("Not scored");
  await page.unroute(path);
  await panel.getByRole("button", { name: "Retry grading-group reference", exact: true }).click();
  await expect(panel.locator(".night-primary-margin-value")).toHaveText("5.7 percentage points");
  await expect(panel.locator(".night-primary-reference")).toHaveAttribute("data-reference-run", "current-polling");
  await expect(panel.locator(".night-margin-reference .night-reference-value")).toContainText("5.6");
  await expect(panel.locator(".night-reference-error")).toHaveCount(0);
});
