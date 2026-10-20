import { expect, test, type Page } from "@playwright/test";
import { makeHousePublication, fixtureSha256 } from "../fixtures/house.mjs";
import type { HouseCatalog } from "../../lib/house-catalog";

// Synthetic data only: exercise the confirmed multi-result publication contract.
function collection() {
  const publications = [2022, 2018].flatMap((year) => ["fixed", "context"].map((family) => {
    const publication = makeHousePublication({ year, eligibleCount: family === "fixed" ? 16 : 20, withGeography: false });
    const result = publication.result;
    result.run.id = `fixture-${family}-${year}`;
    result.run.label = `${family === "fixed" ? "Fixed reference" : "Context alternative"} (${year})`;
    result.run.configurationId = family;
    result.run.configurationSha256 = fixtureSha256(family);
    result.metadata.configurationId = family;
    publication.resultSha256 = fixtureSha256(result);
    return publication;
  }));
  const catalog: HouseCatalog = {
    schemaVersion: 1, defaultRunId: publications[0].result.run.id,
    selectionNote: "Fixed reference is the initial selection.",
    capabilities: { historical_races: true, polling: false, district_geometry: false, full_chamber: false, chamber_control: false, chamber_tipping: false, current_2026: false },
    runs: publications.map((publication) => ({ ...publication.result.run, informationMode: "retrospective_proxy_strictly_earlier_fitted_labels", resultSha256: publication.resultSha256, raceCount: publication.result.races.length, available: true, unavailableReason: null })),
  };
  return publications.map((publication) => ({ ...publication, catalog }));
}

async function controls(page: Page) {
  const panel = page.locator("details.house-configuration");
  await expect(panel).toBeVisible();
  if (await panel.getAttribute("open") === null) await panel.locator(":scope > summary").click();
  return panel;
}

test("House builder shows all eight roles with only its own saved fundamentals selector active", async ({ page }) => {
  const publications = collection();
  await page.route("**/api/chambers/house*", route => route.fulfill({ json: publications[0] }));
  await page.goto("/?chamber=house");
  const panel = page.locator("details.house-configuration");
  await expect(panel).toBeVisible();
  await expect(panel).not.toHaveAttribute("open", "");
  await controls(page);
  await expect(panel.locator(".model-slice")).toHaveCount(8);
  await expect(panel.locator(".slice-control")).toHaveCount(1);
  await expect(panel.locator(".model-slice.is-unavailable")).toHaveCount(7);
  for (const label of ["Population & geography", "Community composition", "Electoral history", "National environment", "Polls & surveys", "Turnout & participation", "Candidate choice", "Vote counts & election rules"]) {
    await expect(panel.getByRole("heading", { name: label, exact: true })).toBeVisible();
  }
  for (const slice of await panel.locator(".model-slice.is-unavailable").all()) {
    const checkbox = slice.getByRole("checkbox");
    const reason = slice.locator(".slice-unavailable-reason");
    await expect(checkbox).toBeDisabled();
    await expect(checkbox).not.toBeChecked();
    await expect(reason).toBeVisible();
    await expect(checkbox).toHaveAttribute("aria-describedby", (await reason.getAttribute("id"))!);
  }
  const fundamentals = panel.locator('[data-slice="fundamentals"]');
  await expect(fundamentals.getByText("Always included", { exact: true })).toBeVisible();
  await expect(fundamentals.getByLabel("House model", { exact: true })).toHaveValue("fixture-fixed-2022");
  await expect(panel.locator('[data-slice="measurement"]')).toContainText("Polling is not connected");
  await expect(panel.locator('[data-slice="environment"]')).toContainText("bundled into each saved House model");
  await expect(panel.locator('[data-slice="counts_rules"]')).toContainText("Full-chamber coverage is missing");
  await expect(panel).not.toContainText("Saved Senate results");
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBeTruthy();
});

test("an unavailable House variant keeps its reason and future capability flags do not admit controls", async ({ page }) => {
  const publications = collection();
  const publication = publications[0];
  publication.catalog.runs[1] = { ...publication.catalog.runs[1], available: false, resultSha256: null, unavailableReason: "Synthetic diagnostic failure; result withheld." };
  publication.catalog.capabilities!.polling = true;
  publication.catalog.capabilities!.full_chamber = true;
  publication.catalog.capabilities!.chamber_control = true;
  await page.route("**/api/chambers/house*", route => route.fulfill({ json: publication }));
  await page.goto("/?chamber=house");
  const panel = await controls(page);
  await expect(page.getByLabel("House model", { exact: true }).locator('option[value="fixture-context-2022"]')).toBeDisabled();
  await panel.getByText("Why some choices are unavailable", { exact: true }).click();
  await expect(panel).toContainText("Synthetic diagnostic failure; result withheld.");
  await expect(panel.locator(".model-slice.is-unavailable")).toHaveCount(7);
  await expect(panel.locator('[data-slice="measurement"]').getByRole("checkbox")).toBeDisabled();
  await expect(page.getByRole("region", { name: "House control availability", exact: true })).toContainText("Not available for this result");
});

test("a fixed legacy House result shows its model without inventing a selector", async ({ page }) => {
  const publication = makeHousePublication({ eligibleCount: 16, withGeography: false });
  await page.route("**/api/chambers/house*", route => route.fulfill({ json: publication }));
  await page.goto("/?chamber=house");
  const panel = await controls(page);
  await expect(panel.locator(".model-slice")).toHaveCount(8);
  await expect(panel.locator('[data-slice="fundamentals"] .slice-fixed-choice')).toHaveText(publication.result.run.label);
  await expect(panel.getByRole("combobox")).toHaveCount(0);
  await expect(panel.getByLabel("House model", { exact: true })).toHaveCount(0);
  await expect(panel.locator(".model-slice.is-unavailable")).toHaveCount(7);
});

test("House model/year selections restore exact results, scope and history without carrying districts across maps", async ({ page }) => {
  const publications = collection();
  await page.route("**/api/chambers/house*", (route) => {
    const id = new URL(route.request().url()).searchParams.get("run");
    return route.fulfill({ json: publications.find((p) => p.result.run.id === id) || publications[0] });
  });
  await page.goto("/?chamber=house&race=2022%3ACA-02%3Ahouse%3Aregular&view=actual");
  const panel = page.locator("details.house-configuration");
  await expect(panel).toBeVisible();
  await expect(panel).not.toHaveAttribute("open", "");
  await controls(page);
  await expect(page.getByLabel("House model", { exact: true })).toHaveValue("fixture-fixed-2022");
  await page.getByLabel("House model", { exact: true }).selectOption("fixture-context-2022");
  await expect(page).toHaveURL(/run=fixture-context-2022/);
  await expect(panel).toHaveAttribute("open", "");
  expect(new URL(page.url()).searchParams.get("race")).toBe("2022:CA-02:house:regular");
  await expect(page.getByRole("region", { name: "How this model did", exact: true })).toContainText("20 / 435");
  await panel.getByRole("button", { name: "2018", exact: true }).click();
  await expect(page).toHaveURL(/run=fixture-context-2018/);
  expect(new URL(page.url()).searchParams.get("race")).toBeNull();
  expect(new URL(page.url()).searchParams.get("view")).toBe("actual");
  await page.goBack();
  await expect(page.getByLabel("House model", { exact: true })).toHaveValue("fixture-context-2022");
  expect(new URL(page.url()).searchParams.get("race")).toBe("2022:CA-02:house:regular");
  await page.reload();
  await controls(page);
  await expect(page.getByLabel("House model", { exact: true })).toHaveValue("fixture-context-2022");
  await expect(page.getByRole("complementary", { name: "Selected House race detail", exact: true })).toContainText("CA-02");
  await page.getByRole("button", { name: "Share House view ↗", exact: true }).click();
  await expect(page.getByLabel("Shareable House view", { exact: true })).toHaveValue(page.url());
  await page.getByRole("button", { name: "Reset House view", exact: true }).click();
  await expect(page).toHaveURL(/run=fixture-fixed-2022/);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBeTruthy();
});

for (const failure of ["service", "wrong result"] as const) {
  test(`a House ${failure} failure preserves the displayed model and permits retry`, async ({ page }) => {
    const publications = collection();
    let fail = true;
    await page.route("**/api/chambers/house*", (route) => {
      const id = new URL(route.request().url()).searchParams.get("run");
      if (id === "fixture-context-2022" && fail) return failure === "service"
        ? route.fulfill({ status: 503, json: { error: "Test failure" } })
        : route.fulfill({ json: publications[0] });
      return route.fulfill({ json: publications.find((p) => p.result.run.id === id) || publications[0] });
    });
    await page.goto("/?chamber=house");
    await controls(page);
    await page.getByLabel("House model", { exact: true }).selectOption("fixture-context-2022");
    await expect(page.locator('.compare-banner[role="alert"]')).toContainText("previous forecast remains visible");
    await expect(page).toHaveURL(/run=fixture-fixed-2022/);
    await expect(page.getByLabel("House model", { exact: true })).toHaveValue("fixture-fixed-2022");
    await expect(page.getByRole("region", { name: "How this model did", exact: true })).toContainText("16 / 435");
    fail = false;
    await page.getByRole("button", { name: "Try selection again", exact: true }).click();
    await expect(page).toHaveURL(/run=fixture-context-2022/);
    await expect(page.locator('.compare-banner[role="alert"]')).toHaveCount(0);
    await expect(page.getByRole("region", { name: "How this model did", exact: true })).toContainText("20 / 435");
  });
}

test("an unknown House deep link can recover to the explicitly published default", async ({ page }) => {
  const publications = collection();
  await page.route("**/api/chambers/house*", (route) => new URL(route.request().url()).searchParams.has("run")
    ? route.fulfill({ status: 400, json: { error: "Unknown model" } })
    : route.fulfill({ json: publications[0] }));
  await page.goto("/?chamber=house&run=unknown");
  await expect(page.getByRole("heading", { name: "House view unavailable", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Load published default", exact: true }).click();
  await expect(page).toHaveURL(/run=fixture-fixed-2022/);
  await expect(page.locator("details.house-configuration")).toBeVisible();
});
