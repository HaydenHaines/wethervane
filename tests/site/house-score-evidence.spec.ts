import { expect, test, type Page } from "@playwright/test";
import { parseHouseResult } from "../../lib/house-result";

const FIXED = "house_history_refreshed_v1-2024-h14";
const CONTEXT = "house_relative_context_v1-2024-h14";
const RELATIVE = "house_relative_lean_v1-2024-h14";
const RACE = "2024:CA-02:house:regular";

async function builder(page: Page) {
  const panel = page.locator("details.house-configuration");
  await expect(panel).toBeVisible();
  if (await panel.getAttribute("open") === null) await panel.locator(":scope > summary").click();
  return panel;
}

async function open(page: Page, compare?: string) {
  const params = new URLSearchParams({ chamber: "house", run: FIXED, race: RACE });
  if (compare) params.set("compare", compare);
  await page.goto(`/?${params}`);
  await expect(page.getByRole("heading", { name: "How this model did", exact: true })).toBeVisible();
  return page.locator("details.house-score-evidence");
}

async function expandEvidence(page: Page) {
  const evidence = page.locator("details.house-score-evidence");
  if (await evidence.getAttribute("open") === null) await evidence.getByText("More about vote-share tests", { exact: true }).click();
  return evidence;
}

async function fixed2024(page: Page) {
  const evidence = page.locator("details.house-score-evidence");
  // Independently audited source values, deliberately not read from the UI manifest.
  await expect(evidence.locator(".house-evidence-mae dd > strong")).toHaveText("2.1 percentage points");
  await expect(evidence.locator(".house-evidence-coverage dd > strong")).toHaveText("97.9% · 367 / 375 districts");
  await expect(evidence.locator(".house-evidence-coverage dd > span").first()).toHaveText("Average range width: 19.1 percentage points");
  await expect(evidence.locator(".house-evidence-crps dd > strong")).toHaveText("1.91 points");
}

test("reviewed vote-share evidence starts collapsed and displays actual Fixed2024 metrics and qualifications", async ({ page }) => {
  const evidence = await open(page);
  await expect(page.locator("details.house-configuration")).not.toHaveAttribute("open");
  await expect(evidence).not.toHaveAttribute("open");
  await expandEvidence(page);
  await fixed2024(page);
  await expect(evidence).toContainText("same 375 included districts");
  await expect(evidence).toContainText("votes for other candidates are excluded");
  await expect(evidence).toContainText("Wider ranges can contain more results");
  await expect(evidence).toContainText("successful calculation checks do not establish future forecasting accuracy");
  await expect(evidence).toContainText("Districts in the same election can move together");
  await expect(page.locator(".chamber-seat-summary, .house-tipping-summary")).toHaveCount(0);
});

test("Context2024 compares exact matched vote-share tests without replacing selected-model evidence", async ({ page }) => {
  await open(page);
  await builder(page);
  await page.getByLabel("Compare with", { exact: true }).selectOption(CONTEXT);
  await expect(page.getByRole("region", { name: "House model comparison", exact: true })).toBeVisible();
  const evidence = await expandEvidence(page);
  await fixed2024(page);
  await expect(evidence.locator(".house-evidence-reference")).toContainText("Relative lean with election context");
  await expect(evidence.locator(".house-evidence-mae dd > span")).toHaveText("Compared: 2.5 percentage points");
  await expect(evidence.locator(".house-evidence-coverage dd > span").nth(1)).toHaveText("Compared: 96.8% · 363 / 375 districtsAverage range width: 15.4 percentage points");
  await expect(evidence.locator(".house-evidence-crps dd > span")).toHaveText("Compared: 1.87 points");
  await expect(page.getByLabel("House model", { exact: true })).toHaveValue(FIXED);
  // Context improves these probability/distribution scores while worsening mean-share error.
  await expect(page.locator(".house-scoreboard-cards .scoreboard-comparison").nth(1)).toContainText("0.035");
});

test("changing to2018 replaces the exact cohort and metrics and clears comparison evidence", async ({ page }) => {
  await open(page, CONTEXT);
  await expect(page.getByRole("region", { name: "House model comparison", exact: true })).toBeVisible();
  const evidence = await expandEvidence(page);
  await expect(evidence.locator(".house-evidence-mae dd > span")).toHaveText("Compared: 2.5 percentage points");
  await builder(page);
  await page.getByRole("group", { name: "Election year", exact: true }).getByRole("button", { name: "2018", exact: true }).click();
  await expect(page.getByLabel("House model", { exact: true })).toHaveValue("house_history_refreshed_v1-2018-h14");
  await expandEvidence(page);
  await expect(evidence).toContainText("Fixed presidential carryover · 2018");
  await expect(evidence).toContainText("same 371 included districts");
  await expect(evidence.locator(".house-evidence-mae dd > strong")).toHaveText("7.0 percentage points");
  await expect(evidence.locator(".house-evidence-coverage dd > strong")).toHaveText("79.2% · 294 / 371 districts");
  await expect(evidence.locator(".house-evidence-coverage dd > span")).toHaveText("Average range width: 19.6 percentage points");
  await expect(evidence.locator(".house-evidence-crps dd > strong")).toHaveText("4.95 points");
  await expect(evidence.locator(".house-evidence-reference, .house-evidence-mae dd > span, .house-evidence-crps dd > span")).toHaveCount(0);
  expect(new URL(page.url()).searchParams.has("compare")).toBe(false);
});

test("a structurally valid primary with modified complementary mean shares cannot inherit the original evidence", async ({ page, request }) => {
  const response = await request.get(`/api/chambers/house?run=${FIXED}`);
  expect(response.ok()).toBeTruthy();
  const publication = await response.json();
  const originalHash = publication.resultSha256;
  const race = publication.result.races.find((item: { id: string }) => item.id === RACE);
  race.outcomes[0].share.mean += 0.001;
  race.outcomes[1].share.mean = 1 - race.outcomes[0].share.mean;
  expect(parseHouseResult(publication.result)).not.toBeNull();
  expect(publication.resultSha256).toBe(originalHash);
  await page.route(`**/api/chambers/house?run=${FIXED}`, route => route.fulfill({ json: publication }));
  await open(page);
  const evidence = await expandEvidence(page);
  await expect(evidence).toContainText("could not be verified for this exact saved result");
  await expect(evidence.locator(".house-evidence-metrics")).toHaveCount(0);
  await expect(page.getByRole("heading", { name: "CA-02", exact: true })).toBeVisible();
  await expect(page.locator(".house-scoreboard-cards .scoreboard-value").nth(0)).toHaveText(/365\s*\/\s*375/);
  await expect(page.locator(".house-scoreboard-cards .scoreboard-value").nth(1)).toHaveText("0.045");
});

test("loading or failing a new secondary removes old reference evidence while keeping primary tests", async ({ page }) => {
  await open(page, CONTEXT);
  await expect(page.getByRole("region", { name: "House model comparison", exact: true })).toBeVisible();
  const evidence = await expandEvidence(page);
  await expect(evidence.locator(".house-evidence-mae dd > span")).toHaveText("Compared: 2.5 percentage points");
  let release!: () => void;
  const gate = new Promise<void>(resolve => { release = resolve; });
  await page.route(`**/api/chambers/house?run=${RELATIVE}`, async route => {
    await gate;
    await route.fulfill({ status: 503, json: { error: "Private failure-isolation test" } });
  });
  await builder(page);
  await page.getByLabel("Compare with", { exact: true }).selectOption(RELATIVE);
  await expect(page.getByRole("status").filter({ hasText: "Loading comparison:" })).toBeVisible();
  await fixed2024(page);
  await expect(evidence.locator(".house-evidence-reference, .house-evidence-mae dd > span, .house-evidence-crps dd > span")).toHaveCount(0);
  await expect(evidence.locator(".house-evidence-coverage dd > span")).toHaveCount(1);
  release();
  await expect(page.getByRole("button", { name: "Retry comparison", exact: true })).toBeVisible();
  await fixed2024(page);
  await expect(evidence.locator(".house-evidence-reference, .house-evidence-mae dd > span, .house-evidence-crps dd > span")).toHaveCount(0);
  await expect(page.getByLabel("House model", { exact: true })).toHaveValue(FIXED);
  expect(new URL(page.url()).searchParams.get("run")).toBe(FIXED);
});
