# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: election-night.spec.ts >> senate: same-page guide starts standard, renders every exact model, and preserves the visitor's forecast
- Location: tests/site/election-night.spec.ts:10:7

# Error details

```
Error: expect(locator).toContainText(expected) failed

Locator: locator('#election-night-scorecard').locator('.night-uncertainty-grid')
Expected substring: "not a measured accuracy record"
Received string:    "Wrong top choices4.2 of 35The expected number of incorrect top choices under the model’s own probabilities. This is an average, not a limit or a measured accuracy record.These choices cover the two named principal contenders under each race’s stated assumptions. How far the margins could missAwaiting a reviewed uncertainty summaryThe margin ranges above are available. The model’s expected average margin error and the range for the number of misses have not yet been published."
Timeout: 5000ms

Call log:
  - Expect "toContainText" locator('#election-night-scorecard').locator('.night-uncertainty-grid') with timeout 5000ms
  - waiting for locator('#election-night-scorecard').locator('.night-uncertainty-grid')
    14 × locator resolved to <div class="night-uncertainty-grid">…</div>
       - unexpected value "Wrong top choices4.2 of 35The expected number of incorrect top choices under the model’s own probabilities. This is an average, not a limit or a measured accuracy record.These choices cover the two named principal contenders under each race’s stated assumptions. How far the margins could missAwaiting a reviewed uncertainty summaryThe margin ranges above are available. The model’s expected average margin error and the range for the number of misses have not yet been published."

```

```yaml
- article:
  - heading "Wrong top choices" [level=4]
  - text: 4.2 of 35
  - paragraph: The expected number of incorrect top choices under the model’s own probabilities. This is an average, not a limit or a measured accuracy record.
  - paragraph: These choices cover the two named principal contenders under each race’s stated assumptions.
- article:
  - heading "How far the margins could miss" [level=4]
  - paragraph: Awaiting a reviewed uncertainty summary
  - paragraph: The margin ranges above are available. The model’s expected average margin error and the range for the number of misses have not yet been published.
```

# Test source

```ts
  1  | import { test, expect } from "@playwright/test";
  2  | import { readFileSync } from "node:fs";
  3  | import { resolve } from "node:path";
  4  | 
  5  | const read = (path: string) => JSON.parse(readFileSync(resolve(path), "utf8"));
  6  | const senateRegistry = read("public/data/explorer/registry.json");
  7  | const houseCatalog = read("data/house-live/catalog.json");
  8  | 
  9  | for (const chamber of ["senate", "house"] as const) {
  10 |   test(`${chamber}: same-page guide starts standard, renders every exact model, and preserves the visitor's forecast`, async ({ page }, testInfo) => {
  11 |     test.setTimeout(240_000);
  12 |     const errors: string[] = [];
  13 |     page.on("pageerror", error => errors.push(error.message));
  14 |     const initialRun = chamber === "senate" ? "current-fundamentals" : "house-current-fundamentals";
  15 |     const defaultId = chamber === "senate" ? senateRegistry.defaults["2026"] : houseCatalog.default_id;
  16 |     const entries = chamber === "senate" ? senateRegistry.runs.filter((entry: { year: number }) => entry.year === 2026) : houseCatalog.entries;
  17 |     await page.goto(chamber === "senate" ? `/?run=${initialRun}` : `/?chamber=house&edition=current&run=${initialRun}`);
  18 |     await expect(page.locator(".hero")).toBeVisible();
  19 |     const originalUrl = page.url();
  20 |     const originalHero = await page.locator(".hero").textContent();
  21 |     const trigger = page.getByRole("button", { name: "Election night scorecard", exact: true });
  22 |     await trigger.click();
  23 |     const panel = page.locator("#election-night-scorecard");
  24 |     await expect(panel.getByRole("heading", { name: "Election night scorecard", exact: true })).toBeFocused();
  25 |     await expect(panel.locator(".night-edition")).toContainText("Standard forecast");
  26 |     await expect(panel).toHaveAttribute("aria-busy", "false");
  27 |     await expect(panel.locator(".night-featured .night-race")).toHaveCount(6);
  28 |     const watchIds = await panel.locator(".night-featured .night-race").evaluateAll(nodes => nodes.map(node => node.getAttribute("data-scorecard-race")));
  29 |     await panel.locator(".night-models summary").click();
  30 |     const select = panel.getByLabel("Scorecard model");
  31 |     await expect(select).toHaveValue(defaultId);
  32 |     await expect(select.locator("option")).toHaveCount(16);
  33 |     for (const entry of entries) {
  34 |       const raw = chamber === "senate" ? read(`public${entry.href}`) : read(`data/house-live/${entry.result.path}`);
  35 |       const digest = chamber === "senate" ? entry.sha256 : entry.result.sha256;
  36 |       const missed = raw.races.reduce((sum: number, race: { outcomes: { winProbability?: number; p_win?: number }[] }) => sum +
  37 |         1 - Math.max(...race.outcomes.map(outcome => chamber === "senate" ? outcome.winProbability! : outcome.p_win!)), 0);
  38 |       await select.selectOption(entry.id);
  39 |       await expect(panel.locator(".night-method code")).toHaveText(digest);
  40 |       await expect(panel.locator(".night-edition strong")).toContainText(entry.label);
  41 |       await expect(panel.locator(".night-big-number")).toContainText(missed.toFixed(1));
  42 |       expect(await panel.locator(".night-featured .night-race").evaluateAll(nodes => nodes.map(node => node.getAttribute("data-scorecard-race")))).toEqual(watchIds);
  43 |       expect(page.url()).toBe(originalUrl);
  44 |       expect(await page.locator(".hero").textContent()).toBe(originalHero);
  45 |     }
  46 |     await expect(panel.locator(".night-rubric")).toContainText("No letter grade assigned");
  47 |     await expect(panel.locator(".night-edition")).toContainText("have not yet been locked");
> 48 |     await expect(panel.locator(".night-uncertainty-grid")).toContainText("not a measured accuracy record");
     |                                                            ^ Error: expect(locator).toContainText(expected) failed
  49 |     await panel.locator(".night-featured").screenshot({ path: testInfo.outputPath(`${chamber}-featured.png`) });
  50 |     await panel.locator(".night-uncertainty-grid").screenshot({ path: testInfo.outputPath(`${chamber}-uncertainty.png`) });
  51 |     await panel.locator(".night-all-races summary").click();
  52 |     await expect(panel.locator(".night-predictions .night-race")).toHaveCount(10);
  53 |     if (chamber === "house") {
  54 |       const defaultEntry = entries.find((entry: { id: string }) => entry.id === defaultId);
  55 |       const unsupported = read(`data/house-live/${defaultEntry.result.path}`).races.find((race: { conditional_shares: unknown }) => race.conditional_shares === null);
  56 |       await panel.getByLabel("Find a race or candidate").fill(unsupported.district_key);
  57 |       await expect(panel.locator(".night-predictions .night-race")).toHaveCount(1);
  58 |       await expect(panel.locator(".night-predictions")).toContainText("No supported two-candidate margin");
  59 |     }
  60 |     expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  61 |     await panel.getByRole("button", { name: "Close scorecard", exact: true }).click();
  62 |     await expect(panel).toHaveCount(0);
  63 |     await expect(trigger).toBeFocused();
  64 |     expect(page.url()).toBe(originalUrl);
  65 |     await trigger.click();
  66 |     await expect(panel.locator(".night-edition")).toContainText("Standard forecast");
  67 |     await expect(panel.locator(".night-models select")).toHaveValue(defaultId);
  68 |     expect(errors).toEqual([]);
  69 |   });
  70 | }
  71 | 
```