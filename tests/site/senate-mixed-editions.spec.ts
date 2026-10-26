import { expect, test, type Page } from "@playwright/test";
import type { CompositionCatalog, CompositionRequest } from "../../lib/composition-types";
import type { ExplorerBundle, ExplorerRegistry } from "../../lib/explorer-types";
import { resolveComposition } from "../../lib/composition-resolver";
import { mixedSenateEditionFixture } from "../fixtures/senate-mixed-edition.mjs";

async function setup(page: Page) {
  // Synthetic metadata split around saved software fixtures. No native forecasts
  // or scientific data are written. Exact resolver semantics remain unchanged.
  const fixture = mixedSenateEditionFixture() as { registry: ExplorerRegistry; catalog: CompositionCatalog;
    bundles: Map<string, ExplorerBundle>; bytes: Map<string, string> };
  const requests: CompositionRequest[] = [];
  await page.route("**/data/explorer/*.json", async route => {
    const path = new URL(route.request().url()).pathname;
    if (path.endsWith("/registry.json")) return route.fulfill({ json: fixture.registry });
    const bytes = fixture.bytes.get(path);
    if (bytes !== undefined) return route.fulfill({ body: bytes, contentType: "application/json" });
    return route.continue();
  });
  await page.route("**/api/configurations/catalog", route => route.fulfill({ json: fixture.catalog }));
  await page.route("**/api/configurations/resolve", async route => {
    const request = route.request().postDataJSON() as CompositionRequest;
    requests.push(request);
    await route.fulfill({ json: resolveComposition(fixture.catalog, fixture.registry, request) });
  });
  return { ...fixture, requests };
}
async function expand(page: Page) {
  const controls = page.locator("details.stack-builder");
  await expect(controls).toBeVisible();
  if (await controls.getAttribute("open") === null) await controls.locator(":scope > summary").click();
}
async function displayed(page: Page, runId: string, date: string) {
  await expect(page).toHaveURL(new RegExp(`run=${runId}(?:&|$)`));
  await expect(page.locator(".composition-state")).toContainText("Ready");
  await expect(page.locator(".composition-state")).toContainText(date);
  await expect(page.getByLabel("Forecast date", { exact: true })).toHaveValue(date);
}

test("all 16 current choices remain reachable with truthful saved dates and exhaustive F/FP switches", async ({ page }) => {
  test.setTimeout(120_000);
  const fixture = await setup(page), visited = new Set<string>();
  await page.goto("/?run=current-polling"); await expand(page);
  const base = page.getByLabel("Base model", { exact: true });
  const polling = page.getByRole("checkbox", { name: "Include polling evidence", exact: true });
  for (const off of fixture.registry.runs.filter(r => r.year === 2026 && r.components.polling === "none")) {
    await base.selectOption(off.components.base);
    await polling.uncheck();
    await displayed(page, off.id, "2026-10-20"); visited.add(off.id);
    const on = fixture.registry.runs.find(r => r.year === 2026 && r.components.base === off.components.base &&
      r.components.polling === "standard" && r.components.uncertainty === "standard")!;
    await polling.check();
    await displayed(page, on.id, "2026-10-26"); visited.add(on.id);
    await expect(page.locator(".composition-state")).toContainText("original input cutoff is preserved");
    expect(fixture.requests.at(-1)?.dateId).toBe("2026-10-26");
    await polling.uncheck();
    await displayed(page, off.id, "2026-10-20");
    expect(fixture.requests.at(-1)?.dateId).toBe("2026-10-20");
    await polling.check(); await displayed(page, on.id, "2026-10-26");
  }
  await base.selectOption("fundamentals"); await displayed(page, "current-polling", "2026-10-26");
  for (const run of fixture.registry.runs.filter(r => r.year === 2026 && r.components.uncertainty !== "standard")) {
    await page.getByLabel("Uncertainty", { exact: true }).selectOption(run.components.uncertainty);
    await displayed(page, run.id, "2026-10-26"); visited.add(run.id);
  }
  expect([...visited].sort()).toEqual(fixture.registry.runs.filter(r => r.year === 2026).map(r => r.id).sort());
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBeTruthy();
});

test("date control stays on its requested date, explains component changes and keeps presets exact", async ({ page }) => {
  const fixture = await setup(page);
  await page.goto("/?run=current-polling"); await expand(page);
  await page.getByLabel("Forecast date", { exact: true }).selectOption("2026-10-20");
  await displayed(page, "current-fundamentals", "2026-10-20");
  await expect(page.getByRole("checkbox", { name: "Include polling evidence", exact: true })).not.toBeChecked();
  await expect(page.locator(".composition-state")).toContainText("Selected components:");
  const noneLabel = fixture.catalog.slots.find(s => s.id === "polling")!.options.find(o => o.id === "none")!.label;
  await expect(page.locator(".composition-state")).toContainText(noneLabel);
  await page.getByLabel("Forecast date", { exact: true }).selectOption("2026-10-26");
  await displayed(page, "current-polling", "2026-10-26");
  const entry = fixture.registry.runs.find(r => r.id === "current-polling")!;
  const preset = fixture.catalog.presets.find(p => p.runId === entry.id)!;
  const wrong = { catalogId: fixture.catalog.id, year: 2026, dateId: "2026-10-20", components: entry.components, presetId: preset.id };
  const rejected = await page.evaluate(async request => (await fetch("/api/configurations/resolve", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(request),
  })).json(), wrong);
  expect(rejected.state).toBe("incompatible"); expect(rejected.request.dateId).toBe("2026-10-20");
  const basic = fixture.catalog.presets.find(p => p.runId === "current-fundamentals")!;
  await page.getByRole("group", { name: "Preserved forecast presets", exact: true }).getByRole("button", { name: new RegExp(basic.label) }).click();
  await displayed(page, "current-fundamentals", "2026-10-20");
  await page.getByRole("group", { name: "Election year", exact: true }).getByRole("button", { name: "2022", exact: true }).click();
  await expect(page).toHaveURL(/run=history-2022-F-1/);
  await page.getByRole("group", { name: "Election year", exact: true }).getByRole("button", { name: "2026", exact: true }).click();
  await displayed(page, "current-fundamentals", "2026-10-20");
});

test("mixed-date controls never offer or retain a comparison across dates or forecast laws", async ({ page }) => {
  await setup(page);
  await page.goto("/?run=current-comparison-senate_relative_lean_v1-fp"); await expand(page);
  const comparison = page.getByLabel("Comparison baseline", { exact: true });
  await expect(comparison.locator('option[value="current-comparison-senate_relative_lean_v1-f"]')).toHaveCount(0);
  await comparison.selectOption("current-comparison-senate_relative_context_v1-fp");
  await page.getByRole("checkbox", { name: "Compare configuration", exact: true }).check();
  await expect(page.locator(".compare-banner").filter({ hasText: "Comparing with" })).toContainText("Changes compare the same races");
  await page.getByRole("checkbox", { name: "Include polling evidence", exact: true }).uncheck();
  await displayed(page, "current-comparison-senate_relative_lean_v1-f", "2026-10-20");
  expect(new URL(page.url()).searchParams.get("against")).toBeNull();
  await expect(comparison.locator('option[value="current-comparison-senate_relative_context_v1-fp"]')).toHaveCount(0);
  await expect(comparison.locator('option[value=""]')).toBeDisabled();
  await expect(page.locator(".compare-banner").filter({ hasText: "Comparing with" })).toContainText("saved dates");
});
