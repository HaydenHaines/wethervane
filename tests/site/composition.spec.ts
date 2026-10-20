import { expect, test, type Page } from "@playwright/test";
import type { CompositionCatalog } from "../../lib/composition-types";

async function expandControls(page: Page) {
  const controls = page.locator("details.stack-builder");
  await expect(controls).toBeVisible();
  if (await controls.getAttribute("open") === null) await controls.locator(":scope > summary").click();
  await expect(controls).toHaveAttribute("open", "");
}

test("the current saved forecast starts with component controls collapsed", async ({ page }) => {
  await page.goto("/");
  await expect(page).toHaveURL(/run=current-polling/);
  const controls = page.locator("details.stack-builder");
  await expect(controls).toBeVisible();
  await expect(controls).not.toHaveAttribute("open", "");
  await expect(controls.locator(":scope > summary")).toContainText("Build your perspective");
  await expect(page.getByLabel("Base model", { exact: true })).not.toBeVisible();
  await expect(page.getByRole("region", { name: "The path to a majority", exact: true })).toBeVisible();
  await expect(page.getByRole("region", { name: "How this model did", exact: true })).toHaveCount(0);
  await expandControls(page);
  await expect(page.getByLabel("Polling component", { exact: true })).toHaveValue("standard");
  await expect(page.locator(".composition-state")).toContainText("Ready");
  await controls.locator(":scope > summary").click();
  await expect(controls).not.toHaveAttribute("open", "");
  await expect(page).toHaveURL(/run=current-polling/);
});

test("historical scorecard distinguishes outcome hits, probability error, and actual chamber control", async ({ page }) => {
  await page.goto("/?run=history-2018-FP-1");
  const scoreboard = page.getByRole("region", { name: "How this model did", exact: true });
  const favorites = scoreboard.getByRole("article", { name: "Favorites that won", exact: true });
  await expect(favorites.locator(".scoreboard-value")).toHaveText(/31\s*\/\s*35/);
  await expect(favorites).toContainText("31.2");
  await favorites.getByText("Explore 4 misses", { exact: true }).click();
  await favorites.getByRole("button", { name: "Maine", exact: true }).click();
  await expect(page.getByRole("complementary").getByRole("heading", { name: "Maine", exact: true })).toBeVisible();
  const quality = scoreboard.getByRole("article", { name: "Probability quality", exact: true });
  await quality.getByText("Probability score details", { exact: true }).click();
  await expect(quality).toContainText("Average Brier score across races: 0.1352");
  await expect(quality).toContainText("0–2 scale");
  await expect(quality).toContainText("35 contests scored.");
  await expect(scoreboard.getByRole("img", { name: /^D caucus: forecast median .*actual 47\./ })).toBeVisible();
  await expect(scoreboard.getByRole("img", { name: /^R caucus: forecast median .*actual 53\./ })).toBeVisible();
  const control = scoreboard.getByRole("article", { name: "Chamber control", exact: true });
  await expect(control.locator(".observed")).toContainText("R control · Actual");
  await expect(control.locator(".scoreboard-value")).toHaveText("99.1%");
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy();
});

test("an experimental request stays separate from the saved result and cannot be saved as ready", async ({ page, request }) => {
  const catalog = await (await request.get("/api/configurations/catalog")).json() as CompositionCatalog;
  const experimental = catalog.computation.find((entry) => entry.years.includes(2026))!;
  await page.goto("/?run=current-polling&race=2026%3ANE%3Asenate%3Aregular");
  const chamber = page.getByRole("region", { name: "Who could control the chamber?", exact: true });
  await expect(chamber).toContainText("8.3%");
  const savedChamber = await chamber.innerText();
  await expandControls(page);
  await page.getByLabel("Base model", { exact: true }).selectOption(experimental.baseId);
  const state = page.locator(".composition-state");
  await expect(state).toContainText("Requires computation");
  await expect(state).toContainText("Still displaying");
  await expect(page).toHaveURL(/run=current-polling/);
  await expect(chamber).toHaveText(savedChamber, {useInnerText:true});
  await expect(page.getByRole("complementary").getByText("14.2%", { exact: true })).toBeVisible();
  await expect(page.locator(".stack-builder-summary")).toContainText("Requires computation");
  await page.locator("details.composition-saved > summary").click();
  await page.getByLabel("Name this view", { exact: true }).fill("Uncomputed request");
  await expect(page.getByRole("button", { name: "Save view", exact: true })).toBeDisabled();
  await page.getByRole("button", { name: "Reset stack", exact: true }).click();
  await expect(page.getByLabel("Base model", { exact: true })).toHaveValue("fundamentals");
  await expect(state).toContainText("Ready");
});

test("named local views reload and restore the exact stack, race, scenario, and comparison", async ({ page }) => {
  await page.goto("/?run=history-2018-F-1&scenario=unresolved_other_R&race=2018%3AMN%3Asenate%3Aspecial&compare=1&against=history-2018-W-1&view=actual");
  await expandControls(page);
  await page.locator("details.composition-saved > summary").click();
  await page.getByLabel("Name this view", { exact: true }).fill("Midterm reference");
  await page.getByRole("button", { name: "Save view", exact: true }).click();
  await expect(page.getByRole("button", { name: "Restore Midterm reference", exact: true })).toBeVisible();
  const stored = await page.evaluate(() => JSON.parse(localStorage.getItem("ballotmancy.saved-views.v1") || "[]"));
  expect(stored).toHaveLength(1);
  expect(stored[0]).toMatchObject({ name: "Midterm reference", runId: "history-2018-F-1", scenarioId: "unresolved_other_R", raceId: "2018:MN:senate:special", compare: true, baselineRunId: "history-2018-W-1", mapView: "actual" });
  expect(stored[0].resultSha256).toMatch(/^[a-f0-9]{64}$/);
  await page.getByRole("group", { name: "Election year", exact: true }).getByRole("button", { name: "2026", exact: true }).click();
  await expect(page).toHaveURL(/run=current-/);
  await page.reload();
  await expandControls(page);
  await page.locator("details.composition-saved > summary").click();
  await page.getByRole("button", { name: "Restore Midterm reference", exact: true }).click();
  await expect(page).toHaveURL(/run=history-2018-F-1/);
  const restored = new URL(page.url()).searchParams;
  expect(restored.get("scenario")).toBe("unresolved_other_R");
  expect(restored.get("race")).toBe("2018:MN:senate:special");
  expect(restored.get("against")).toBe("history-2018-W-1");
  expect(restored.get("view")).toBe("actual");
  await expect(page.getByLabel("Comparison baseline", { exact: true })).toHaveValue("history-2018-W-1");
  await expect(page.getByRole("checkbox", { name: "Compare configuration", exact: true })).toBeChecked();
  await expect(page.getByLabel("Choose election in Minnesota", { exact: true })).toHaveValue("2018:MN:senate:special");
  await page.getByRole("button", { name: "Delete Midterm reference", exact: true }).click();
  await expect(page.getByRole("button", { name: "Restore Midterm reference", exact: true })).toHaveCount(0);
});

test("custom comparison identities survive URL reload and change probability comparisons", async ({ page }) => {
  await page.goto("/?run=history-2018-F-1");
  await expandControls(page);
  const baseline = page.getByLabel("Comparison baseline", { exact: true });
  await expect(page.getByRole("checkbox", {name:"Compare default",exact:true})).not.toBeChecked();
  await baseline.selectOption("history-2018-W-1");
  await expect(baseline).toHaveValue("history-2018-W-1");
  await expect(page.getByRole("checkbox", {name:"Compare configuration",exact:true})).not.toBeChecked();
  await expect(baseline.locator('option[value^="history-2022"]')).toHaveCount(0);
  const quality = page.getByRole("region", { name: "How this model did", exact: true }).getByRole("article", { name: "Probability quality", exact: true });
  await expect(quality).toContainText("higher error");
  await quality.getByText("Probability score details", { exact: true }).click();
  await expect(quality).toContainText("Average Brier score across races: 0.2418; comparison 0.2272");
  await baseline.selectOption("history-2018-F-1");
  await expect(page).toHaveURL(/against=history-2018-F-1/);
  await expect(quality).toContainText("Same error");
  await page.reload();
  await expandControls(page);
  await expect(baseline).toHaveValue("history-2018-F-1");
  await expect(quality).toContainText("Same error");
});

test("composition API binds ready presets and reports pending or invalid requests without execution", async ({ request }) => {
  const response = await request.get("/api/configurations/catalog");
  expect(response.ok()).toBeTruthy();
  const catalog = await response.json() as CompositionCatalog;
  const preset = catalog.presets.find((entry) => entry.runId === "current-polling")!;
  const configuration = catalog.saved.find((entry) => entry.runId === preset.runId)!;
  const base = { catalogId: catalog.id, year: configuration.year, dateId: configuration.dateId, components: configuration.components };
  const readyResponse = await request.post("/api/configurations/resolve", { data: { ...base, presetId: preset.id } });
  expect(readyResponse.status()).toBe(200);
  const ready = await readyResponse.json();
  expect(ready.state).toBe("ready");
  expect(ready.configurationId).toBe(configuration.id);
  expect(ready.resultId).toBe(configuration.resultSha256);
  expect(ready.run.id).toBe(configuration.runId);
  const experimental = catalog.computation.find((entry) => entry.years.includes(configuration.year))!;
  const pendingResponse = await request.post("/api/configurations/resolve", { data: { ...base, components: { ...base.components, base: experimental.baseId, polling: "standard", uncertainty: "standard" } } });
  expect(pendingResponse.status()).toBe(200);
  expect(await pendingResponse.json()).toMatchObject({ state: "requires_computation", resultId: null, configurationId: null, outputs: ["race-probabilities"], executionAvailable: false });
  const incompatible = await request.post("/api/configurations/resolve", { data: { ...base, components: { ...base.components, polling: "none", uncertainty: "error-high" } } });
  expect(incompatible.status()).toBe(200);
  expect((await incompatible.json()).state).toBe("incompatible");
  const injected = await request.post("/api/configurations/resolve", { data: { ...base, bindings: { artifact: "client-chosen" } } });
  expect(injected.status()).toBe(400);
  const oversized = await request.post("/api/configurations/resolve", { headers: { "Content-Type": "application/json" }, data: " ".repeat(9000) });
  expect(oversized.status()).toBe(400);
  expect((await oversized.json()).error).toMatch(/too large/i);
});
