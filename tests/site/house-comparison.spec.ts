import { expect, test, type APIRequestContext, type Page } from "@playwright/test";
import type { HouseResult } from "../../lib/house-result";
import type { HouseCatalog } from "../../lib/house-catalog";

// Use the published reviewed records, not synthetic scientific compatibility.
type Publication = { status: "available"; result: HouseResult; resultSha256: string; catalog: HouseCatalog };
const FIXED = "house_history_refreshed_v1-2024-h14";
const CONTEXT = "house_relative_context_v1-2024-h14";
const RELATIVE = "house_relative_lean_v1-2024-h14";
const STORE = "ballotmancy.house.saved-views.v1";
const RACE = "2024:CA-02:house:regular";

async function publication(request: APIRequestContext, id: string): Promise<Publication> {
  const response = await request.get(`/api/chambers/house?${new URLSearchParams({ run: id })}`);
  expect(response.ok()).toBeTruthy();
  const value = await response.json();
  expect(value.status).toBe("available");
  expect(value.result.run.id).toBe(id);
  return value;
}

function scores(result: HouseResult) {
  const known = result.races.filter(race => race.observed);
  const decisive = known.filter(race => {
    const maximum = Math.max(...race.outcomes.map(outcome => outcome.winProbability));
    return race.outcomes.filter(outcome => Math.abs(outcome.winProbability - maximum) < 1e-12).length === 1;
  });
  return {
    observed: known.length,
    decisive: decisive.length,
    correct: decisive.filter(race => race.outcomes.reduce((a, b) => a.winProbability > b.winProbability ? a : b).id === race.observed!.outcomeId).length,
    brier: known.reduce((sum, race) => sum + race.outcomes.reduce((value, outcome) => value + (outcome.winProbability - Number(outcome.id === race.observed!.outcomeId)) ** 2, 0), 0) / known.length,
  };
}

async function builder(page: Page) {
  const panel = page.locator("details.house-configuration");
  await expect(panel).toBeVisible();
  if (await panel.getAttribute("open") === null) await panel.locator(":scope > summary").click();
  return panel;
}

async function saved(page: Page) {
  const panel = (await builder(page)).locator("details.house-saved-views");
  if (await panel.getAttribute("open") === null) await panel.locator(":scope > summary").click();
  return panel;
}

async function open(page: Page, compare?: string) {
  const params = new URLSearchParams({ chamber: "house", run: FIXED, race: RACE, view: "actual" });
  if (compare) params.set("compare", compare);
  await page.goto(`/?${params}`);
  await expect(page.getByRole("heading", { name: "How this model did", exact: true })).toBeVisible();
  return builder(page);
}

async function assertMain(page: Page, id = FIXED) {
  await expect(page.getByLabel("House model", { exact: true })).toHaveValue(id);
  expect(new URL(page.url()).searchParams.get("run")).toBe(id);
  await expect(page.getByRole("heading", { name: "CA-02", exact: true })).toBeVisible();
  await expect(page.locator(".district-map-svg, .house-tipping-summary, .majority-position, .chamber-seat-summary")).toHaveCount(0);
}

async function assertComparison(page: Page, main: Publication, reference: Publication) {
  await expect(page.getByRole("region", { name: "House model comparison", exact: true })).toBeVisible();
  const first = scores(main.result), second = scores(reference.result);
  const cards = page.locator(".house-scoreboard-cards .scoreboard-card");
  await expect(cards.nth(0).locator(".scoreboard-value")).toHaveText(new RegExp(`${first.correct}\\s*/\\s*${first.decisive}`));
  await expect(cards.nth(0).locator(".scoreboard-comparison")).toContainText(`${second.correct} / ${second.decisive}`);
  await expect(cards.nth(1).locator(".scoreboard-value")).toHaveText(first.brier.toFixed(3));
  await expect(cards.nth(1).locator(".scoreboard-comparison strong")).toHaveText(second.brier.toFixed(3));
  const difference = first.brier - second.brier;
  if (Math.abs(difference) > 1e-12) {
    await expect(cards.nth(1).locator(".scoreboard-comparison")).toContainText(`${Math.abs(difference) < 0.0005 ? "Less than 0.001" : Math.abs(difference).toFixed(3)} ${difference < 0 ? "lower" : "higher"} probability error`);
  }
  await expect(cards.nth(2).locator(".scoreboard-comparison")).toContainText(`same ${first.observed} districts`);
  const race = main.result.races.find(item => item.id === RACE)!;
  const other = reference.result.races.find(item => item.id === RACE)!;
  const candidates = page.getByRole("complementary", { name: "Selected House race detail" }).locator(".candidate");
  for (let i = 0; i < race.outcomes.length; i++) {
    const outcome = race.outcomes[i];
    const probability = other.outcomes.find(item => item.id === outcome.id)!.winProbability;
    const delta = outcome.winProbability - probability;
    await expect(candidates.nth(i).locator(".candidate-name")).toHaveText(outcome.name);
    const deltaLabel = delta === 0 ? "No difference"
      : Math.abs(delta * 100) < 0.05 ? `Less than 0.1 percentage point ${delta > 0 ? "higher" : "lower"}`
        : `${Math.abs(delta * 100).toFixed(1)} percentage points ${delta > 0 ? "higher" : "lower"}`;
    await expect(candidates.nth(i).locator(".house-candidate-comparison")).toContainText(`Selected minus compared: ${deltaLabel}`);
  }
}

test("real same-year comparison preserves district, actual view, share links and history, with independently checked scores", async ({ page, request }) => {
  const [main, reference] = await Promise.all([publication(request, FIXED), publication(request, CONTEXT)]);
  const panel = await open(page);
  const select = panel.getByLabel("Compare with", { exact: true });
  await expect(select.locator("option")).toHaveCount(4);
  const options = await select.locator("option").evaluateAll(nodes => nodes.map(node => (node as HTMLOptionElement).value));
  expect(options.filter(Boolean).sort()).toEqual(main.catalog.runs.filter(run => run.year === 2024 && run.id !== FIXED).map(run => run.id).sort());
  await select.selectOption(CONTEXT);
  await assertComparison(page, main, reference);
  await assertMain(page);
  expect(new URL(page.url()).searchParams.get("race")).toBe(RACE);
  expect(new URL(page.url()).searchParams.get("view")).toBe("actual");
  await panel.getByRole("button", { name: "Share House view ↗", exact: true }).click();
  expect(new URL(await panel.getByLabel("Shareable House view", { exact: true }).inputValue()).searchParams.get("compare")).toBe(CONTEXT);
  await page.reload();
  await builder(page);
  await assertComparison(page, main, reference);
  await page.getByLabel("Compare with", { exact: true }).selectOption("");
  await expect(page.locator(".scoreboard-comparison, .house-candidate-comparison")).toHaveCount(0);
  await page.goBack();
  await assertComparison(page, main, reference);
  await page.goForward();
  await expect(page.getByLabel("Compare with", { exact: true })).toHaveValue("");
  await expect(page.locator(".scoreboard-comparison, .house-candidate-comparison")).toHaveCount(0);
});

test("changing the selected model keeps a same-year reference and changing year clears it", async ({ page, request }) => {
  const [main, reference] = await Promise.all([publication(request, RELATIVE), publication(request, CONTEXT)]);
  await open(page, CONTEXT);
  await expect(page.getByRole("region", { name: "House model comparison", exact: true })).toBeVisible();
  await page.getByLabel("House model", { exact: true }).selectOption(RELATIVE);
  await expect(page.getByLabel("House model", { exact: true })).toHaveValue(RELATIVE);
  await assertComparison(page, main, reference);
  expect(new URL(page.url()).searchParams.get("compare")).toBe(CONTEXT);
  expect(new URL(page.url()).searchParams.get("race")).toBe(RACE);
  await page.getByRole("group", { name: "Election year", exact: true }).getByRole("button", { name: "2022", exact: true }).click();
  await expect(page.getByLabel("House model", { exact: true })).toHaveValue("house_relative_lean_v1-2022-h14");
  expect(new URL(page.url()).searchParams.has("compare")).toBe(false);
  await expect(page.getByLabel("Compare with", { exact: true })).toHaveValue("");
  await expect(page.locator(".scoreboard-comparison, .house-candidate-comparison")).toHaveCount(0);
});

test("a shared cross-year reference is visibly rejected and can be removed without changing the selected model", async ({ page }) => {
  await open(page, "house_relative_context_v1-2022-h14");
  await expect(page.getByRole("button", { name: "Remove comparison", exact: true })).toBeVisible();
  await expect(page.getByRole("alert").filter({ hasText: "Comparison unavailable" })).toContainText("same election year");
  await expect(page.locator(".scoreboard-comparison, .house-candidate-comparison")).toHaveCount(0);
  await assertMain(page);
  await page.getByRole("button", { name: "Remove comparison", exact: true }).click();
  expect(new URL(page.url()).searchParams.has("compare")).toBe(false);
  await expect(page.getByLabel("Compare with", { exact: true })).toHaveValue("");
  await assertMain(page);
});

test("saved comparison binds both exact results, reloads, and restores even when its primary is already displayed", async ({ page, request }) => {
  const [main, reference] = await Promise.all([publication(request, FIXED), publication(request, CONTEXT)]);
  await open(page, CONTEXT);
  await assertComparison(page, main, reference);
  let views = await saved(page);
  await views.getByLabel("Name this House view", { exact: true }).fill("California comparison");
  await views.getByRole("button", { name: "Save House view", exact: true }).click();
  const stored = await page.evaluate(key => JSON.parse(localStorage.getItem(key) || "{}"), STORE);
  expect(stored.views[0]).toMatchObject({ runId: FIXED, resultSha256: main.resultSha256, raceId: RACE, mapView: "actual", comparison: { runId: CONTEXT, resultSha256: reference.resultSha256, configurationSha256: reference.result.run.configurationSha256, year: 2024, mapVintage: reference.result.metadata.mapVintage } });
  expect(JSON.stringify(stored)).not.toContain('"outcomes"');
  await page.reload();
  await builder(page);
  await assertComparison(page, main, reference);
  await page.getByRole("group", { name: "Election year", exact: true }).getByRole("button", { name: "2018", exact: true }).click();
  await expect(page.getByLabel("House model", { exact: true })).toHaveValue("house_history_refreshed_v1-2018-h14");
  views = await saved(page);
  await views.getByRole("button", { name: "Restore California comparison", exact: true }).click();
  await assertComparison(page, main, reference);
  await assertMain(page);
  await page.getByLabel("Compare with", { exact: true }).selectOption("");
  await expect(page.locator(".house-candidate-comparison")).toHaveCount(0);
  await (await saved(page)).getByRole("button", { name: "Restore California comparison", exact: true }).click();
  await assertComparison(page, main, reference);
  expect(new URL(page.url()).searchParams.get("compare")).toBe(CONTEXT);
  expect(new URL(page.url()).searchParams.get("view")).toBe("actual");
});

test("loading and failed secondary requests never show stale deltas or replace the primary, and retry recovers", async ({ page, request }) => {
  const [main, previous, reference] = await Promise.all([publication(request, FIXED), publication(request, CONTEXT), publication(request, RELATIVE)]);
  await open(page, CONTEXT);
  await assertComparison(page, main, previous);
  let release!: () => void;
  const gate = new Promise<void>(resolve => { release = resolve; });
  await page.route(`**/api/chambers/house?run=${RELATIVE}`, async route => {
    await gate;
    await route.fulfill({ status: 503, json: { error: "Private test failure" } });
  });
  await page.getByLabel("Compare with", { exact: true }).selectOption(RELATIVE);
  await expect(page.getByRole("status").filter({ hasText: "Loading comparison:" })).toBeVisible();
  await expect(page.locator(".scoreboard-comparison, .house-candidate-comparison")).toHaveCount(0);
  await assertMain(page);
  release();
  await expect(page.getByRole("button", { name: "Retry comparison", exact: true })).toBeVisible();
  await assertMain(page);
  await expect(page.locator(".scoreboard-comparison, .house-candidate-comparison")).toHaveCount(0);
  await page.unroute(`**/api/chambers/house?run=${RELATIVE}`);
  await page.getByRole("button", { name: "Retry comparison", exact: true }).click();
  await assertComparison(page, main, reference);
});

for (const invalid of ["wrong-result", "tampered-content"] as const) {
  test(`a ${invalid} secondary is rejected despite its claimed original hash`, async ({ page, request }) => {
    const [main, reference] = await Promise.all([publication(request, FIXED), publication(request, CONTEXT)]);
    await open(page);
    const response = structuredClone(invalid === "wrong-result" ? main : reference);
    if (invalid === "tampered-content") {
      const outcomes = response.result.races[0].outcomes;
      outcomes[0].winProbability = 0.37;
      outcomes[1].winProbability = 0.63;
      expect(response.resultSha256).toBe(reference.resultSha256);
    }
    await page.route(`**/api/chambers/house?run=${CONTEXT}`, route => route.fulfill({ json: response }));
    await page.getByLabel("Compare with", { exact: true }).selectOption(CONTEXT);
    await expect(page.getByRole("button", { name: "Remove comparison", exact: true })).toBeVisible();
    await expect(page.getByRole("region", { name: "House model comparison", exact: true })).toHaveCount(0);
    await expect(page.locator(".scoreboard-comparison, .house-candidate-comparison")).toHaveCount(0);
    await assertMain(page);
    await expect(page.locator(".house-scoreboard-cards .scoreboard-value").nth(1)).toHaveText(scores(main.result).brier.toFixed(3));
  });
}

test("failed saved secondary restore is atomic, while stale or withdrawn saved secondaries remain visible and disabled", async ({ page, request }) => {
  await open(page, CONTEXT);
  await expect(page.getByRole("region", { name: "House model comparison", exact: true })).toBeVisible();
  let views = await saved(page);
  await views.getByLabel("Name this House view", { exact: true }).fill("Exact pair");
  await views.getByRole("button", { name: "Save House view", exact: true }).click();
  const stored = await page.evaluate(key => JSON.parse(localStorage.getItem(key) || "{}"), STORE);
  await page.getByRole("group", { name: "Election year", exact: true }).getByRole("button", { name: "2022", exact: true }).click();
  await expect(page.getByLabel("House model", { exact: true })).toHaveValue("house_history_refreshed_v1-2022-h14");
  const prior = page.url();
  await page.route(`**/api/chambers/house?run=${CONTEXT}`, route => route.fulfill({ status: 503, json: {} }));
  await (await saved(page)).getByRole("button", { name: "Restore Exact pair", exact: true }).click();
  await expect(page.getByRole("button", { name: "Try selection again", exact: true })).toBeVisible();
  expect(page.url()).toBe(prior);
  await expect(page.getByLabel("House model", { exact: true })).toHaveValue("house_history_refreshed_v1-2022-h14");
  await page.unroute(`**/api/chambers/house?run=${CONTEXT}`);

  const stale = structuredClone(stored);
  stale.views[0].comparison.resultSha256 = "0".repeat(64);
  await page.evaluate(({ key, value }) => localStorage.setItem(key, JSON.stringify(value)), { key: STORE, value: stale });
  await page.reload();
  views = await saved(page);
  await expect(views.getByRole("button", { name: "Restore Exact pair", exact: true })).toBeDisabled();
  await expect(views.locator(".house-saved-reason")).toContainText("comparison result has changed");
  await expect(views.getByRole("button", { name: "Delete Exact pair", exact: true })).toBeEnabled();

  await page.evaluate(({ key, value }) => localStorage.setItem(key, JSON.stringify(value)), { key: STORE, value: stored });
  const current = await publication(request, "house_history_refreshed_v1-2022-h14");
  const withdrawn = structuredClone(current);
  const entry = withdrawn.catalog.runs.find(run => run.id === CONTEXT)!;
  entry.available = false;
  entry.resultSha256 = null;
  entry.unavailableReason = "This reviewed result has been withdrawn.";
  // Only publication availability changes; reviewed scientific manifest/results stay intact.
  await page.route("**/api/chambers/house*", route => route.fulfill({ json: withdrawn }));
  await page.reload();
  views = await saved(page);
  await expect(views.getByRole("button", { name: "Restore Exact pair", exact: true })).toBeDisabled();
  await expect(views.locator(".house-saved-reason")).toContainText("withdrawn");
  await expect(page.getByLabel("House model", { exact: true })).toHaveValue(current.result.run.id);
});
