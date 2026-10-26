/** Actual October 26 release candidate: real HTTP artifacts and UI, with no interception. */
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { expect, test, type APIRequestContext, type Page } from "@playwright/test";
import type { ExplorerBundle, ExplorerRegistry, ExplorerRunEntry } from "../../lib/explorer-types";
import type { CompositionCatalog, CompositionResolution } from "../../lib/composition-types";

// Root can reuse this exact test against the served release with a config whose
// webServer is disabled. No route interception or local-origin assumptions follow.
if (process.env.BALLOTMANCY_SERVED_ORIGIN) test.use({ baseURL: process.env.BALLOTMANCY_SERVED_ORIGIN });

const registry = JSON.parse(readFileSync("public/data/explorer/registry.json", "utf8")) as ExplorerRegistry;
const catalog = JSON.parse(readFileSync("public/data/explorer/capabilities.json", "utf8")) as CompositionCatalog;
const current = registry.runs.filter(run => run.year === 2026);
const DEFAULT_SHA = "905ce512c44104d3e48900745cec0f430c9e3954736614468e1a8b728b6da5f0";
const POLLING_CUTOFF = "2026-10-26T15:03:57.212405+00:00";
const DEFAULT_PRESET = "best_guess_polling_20261026_v1";
const primary = current.find(run => run.id === registry.defaults["2026"])!;
const historicalBase = current.find(run => run.id === "current-fundamentals")!;
const texas = "2026:TX:senate:regular";
const escapeRegex = (value: string) => value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

async function expandControls(page: Page) {
  const controls = page.locator("details.stack-builder");
  await expect(controls).toBeVisible();
  if (await controls.getAttribute("open") === null) await controls.locator(":scope > summary").click();
  await expect(controls).toHaveAttribute("open", "");
}
async function exactArtifact(request: APIRequestContext, entry: ExplorerRunEntry): Promise<ExplorerBundle> {
  const response = await request.get(entry.href);
  expect(response.ok(), entry.id).toBeTruthy();
  const bytes = await response.body();
  expect(createHash("sha256").update(bytes).digest("hex"), entry.id).toBe(entry.sha256);
  expect(bytes.equals(readFileSync(`public${entry.href}`)), `${entry.id} served bytes match candidate`).toBeTruthy();
  const bundle = JSON.parse(bytes.toString("utf8")) as ExplorerBundle;
  const { href: _href, sha256: _sha, ...metadata } = entry;
  expect(bundle.run, entry.id).toEqual(metadata);
  expect(bundle.races, entry.id).toHaveLength(35);
  expect(new Set(bundle.races.map(race => race.id)).size, entry.id).toBe(35);
  return bundle;
}
async function actualHealth(request: APIRequestContext) {
  const response = await request.get("/api/health");
  expect(response.ok()).toBeTruthy();
  expect(await response.json()).toEqual({ status: "ok", application: "ballotmancy", runId: primary.id,
    resultSha256: DEFAULT_SHA, issuedAt: primary.issuedAt, cutoffAt: POLLING_CUTOFF, races: 35 });
}
async function displayed(page: Page, entry: ExplorerRunEntry) {
  await expect(page).toHaveURL(new RegExp(`[?&]run=${escapeRegex(entry.id)}(?:&|$)`));
  await expect(page.locator("details.stack-builder")).toHaveAttribute("aria-busy", "false");
  await expandControls(page);
  await expect(page.locator(".composition-state")).toContainText("Ready");
  await expect(page.locator(".composition-state")).toContainText(entry.dateId);
  await expect(page.getByLabel("Forecast date", { exact: true })).toHaveValue(entry.dateId);
  await expect(page.getByLabel("Base model", { exact: true })).toHaveValue(entry.components.base);
  const polls = page.getByRole("checkbox", { name: "Include polling evidence", exact: true });
  if (entry.components.polling === "none") await expect(polls).not.toBeChecked();
  else await expect(polls).toBeChecked();
  const configuration = page.locator("details.composition-details");
  if (await configuration.getAttribute("open") === null) await configuration.locator(":scope > summary").click();
  await expect(configuration.locator(".configuration-display")).toContainText(entry.sha256);
  await expect(configuration.locator(".configuration-display")).toContainText(entry.id);
}
async function expectedView(page: Page, bundle: ExplorerBundle) {
  const ribbon = page.getByRole("region", { name: "The path to a majority", exact: true });
  await expect(ribbon).toBeVisible();
  await expect(ribbon.locator("button[data-race-id]")).toHaveCount(35);
  expect(await ribbon.locator("button[data-race-id]").evaluateAll(nodes => nodes.map(n => (n as HTMLElement).dataset.raceId)))
    .toEqual(bundle.tipping!.raceOrder);
  await expect(page.locator(".map-card")).toBeVisible();
  await expect(page.locator(".map-card svg [role=button]")).toHaveCount(35);
  await expect(page.getByRole("region", { name: "Who could control the chamber?", exact: true })).toBeVisible();
  await expect(page.getByRole("complementary", { name: "Selected race detail", exact: true })).toBeVisible();
  await expect(page.getByRole("region", { name: "How this model did", exact: true })).toHaveCount(0);
}

test("actual refreshed candidate serves all 16 exact current artifacts and coherent URL-selected map, snake and control views", async ({ page, request }) => {
  test.setTimeout(180_000);
  const errors: string[] = []; page.on("pageerror", error => errors.push(error.message));
  const servedRegistry = await request.get("/data/explorer/registry.json");
  expect(servedRegistry.ok()).toBeTruthy(); expect(await servedRegistry.json()).toEqual(registry);
  expect(current).toHaveLength(16);
  expect(current.filter(run => run.dateId === "2026-10-26")).toHaveLength(10);
  expect(current.filter(run => run.dateId === "2026-10-20")).toHaveLength(6);
  expect(primary.id).toBe("current-polling"); expect(primary.sha256).toBe(DEFAULT_SHA);
  expect(primary.cutoffAt).toBe(POLLING_CUTOFF);
  await actualHealth(request);
  for (const entry of current) {
    const bundle = await exactArtifact(request, entry);
    expect(entry.dateId).toBe(entry.components.polling === "none" ? "2026-10-20" : "2026-10-26");
    if (entry.components.polling === "standard") {
      expect(bundle.races.reduce((sum, race) => sum + (race.pollCount ?? 0), 0)).toBe(131);
      expect(bundle.races.filter(race => (race.pollCount ?? 0) > 0)).toHaveLength(25);
      expect(entry.strictAsOf).toBe(false);
    } else expect(bundle.races.every(race => (race.pollCount ?? 0) === 0)).toBeTruthy();
    await page.goto(`/?run=${encodeURIComponent(entry.id)}&race=${encodeURIComponent(texas)}`);
    await displayed(page, entry); await expectedView(page, bundle);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBeTruthy();
  }
  expect(errors).toEqual([]);
});

test("actual new default preset and F-to-FP switch preserve edition dates, exact hashes and current poll counts", async ({ page, request }) => {
  test.setTimeout(90_000);
  const response = await request.get("/api/configurations/catalog");
  expect(response.ok()).toBeTruthy(); expect(await response.json()).toEqual(catalog);
  expect(catalog.defaultPresetId).toBe(DEFAULT_PRESET);
  const preset = catalog.presets.find(value => value.id === DEFAULT_PRESET)!;
  expect(preset.runId).toBe(primary.id);
  const bundle = await exactArtifact(request, primary);
  expect(preset.forecastSha256).toBe(bundle.provenance.sources.forecastSha256);
  const configuration = catalog.saved.find(value => value.runId === preset.runId)!;
  const resolved = await request.post("/api/configurations/resolve", { data: { catalogId: catalog.id,
    year: configuration.year, dateId: configuration.dateId, components: configuration.components, presetId: preset.id } });
  expect(resolved.ok()).toBeTruthy();
  const ready = await resolved.json() as CompositionResolution;
  expect(ready.state).toBe("ready"); expect(ready.run?.id).toBe(primary.id);
  expect(ready.run?.sha256).toBe(DEFAULT_SHA); expect(ready.request.dateId).toBe("2026-10-26");
  const detail = page.getByRole("complementary", { name: "Selected race detail", exact: true });
  const polling = page.getByRole("region", { name: "Why some polls are included—and others are not", exact: true });
  await page.goto(`/?run=${historicalBase.id}&race=${encodeURIComponent(texas)}`);
  await displayed(page, historicalBase);
  await expect(detail).toContainText("This setup uses electoral history without polls.");
  await expect(polling).toContainText("No polls are included.");
  await page.getByRole("group", { name: "Preserved forecast presets", exact: true })
    .getByRole("button", { name: new RegExp(escapeRegex(preset.label)) }).click();
  await displayed(page, primary);
  await expect(polling).toContainText("131 poll records across 25 of 35 races");
  const texasPolls = bundle.races.find(race => race.id === texas)!.pollCount!;
  expect(texasPolls).toBe(17);
  await expect(detail.locator(".metric").filter({ hasText: "Poll records included" }).locator("strong")).toHaveText(String(texasPolls));
  const toggle = page.getByRole("checkbox", { name: "Include polling evidence", exact: true });
  await toggle.uncheck(); await displayed(page, historicalBase);
  await expect(polling).toContainText("No polls are included.");
  await toggle.check(); await displayed(page, primary);
  await expect(page.locator(".composition-state")).toContainText("original input cutoff is preserved");
  await expect(polling).toContainText("131 poll records across 25 of 35 races");
  await expect(detail.locator(".metric").filter({ hasText: "Poll records included" }).locator("strong")).toHaveText("17");
  await expectedView(page, bundle); await actualHealth(request);
});
