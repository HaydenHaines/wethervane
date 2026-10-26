import { expect, test, type Page } from "@playwright/test";
import { makeHousePublication, fixtureSha256 } from "../fixtures/house.mjs";
import { parseHousePublicationWithCatalog, type HouseCatalog } from "../../lib/house-catalog";
import { createHouseSavedView, serializeHouseSavedViews, HOUSE_SAVED_VIEWS_STORAGE_KEY } from "../../lib/house-saved-views";

const historicalApi = /\/api\/chambers\/house(?:\?.*)?$/;
// Bare House links should deterministically take the historical path in these fixtures.
test.beforeEach(async ({ page }) => {
  await page.route("**/api/chambers/house/live/availability", route =>
    route.fulfill({ json: { schema: "HouseLiveAvailabilityV1", available: false } }));
});

// Synthetic data only: exercise the confirmed multi-result publication contract.
function collection() {
  const publications = [2022, 2018].flatMap((year) => ["fixed", "context"].map((family) => {
    const publication = makeHousePublication({ year, eligibleCount: family === "fixed" ? 16 : 20, withGeography: false });
    const result = publication.result;
    result.run.id = `fixture-${family}-${year}-h14`;
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
  return publications.map((publication) => {
    const parsed = parseHousePublicationWithCatalog({ ...publication, catalog });
    if (!parsed || parsed.status !== "available") throw Error("Invalid House selection fixture");
    return { ...parsed, catalog };
  });
}

async function controls(page: Page) {
  const panel = page.locator("details.house-configuration");
  await expect(panel).toBeVisible();
  if (await panel.getAttribute("open") === null) await panel.locator(":scope > summary").click();
  return panel;
}

async function savedViews(page: Page) {
  const panel = await controls(page);
  const views = panel.locator("details.house-saved-views");
  if (await views.getAttribute("open") === null) await views.locator(":scope > summary").click();
  return views;
}

test("House views save exact results locally and restore district, map mode and history across years", async ({ page }) => {
  const publications = collection();
  await page.route(historicalApi, route => route.fulfill({ json: publications.find(p => p.result.run.id === new URL(route.request().url()).searchParams.get("run")) || publications[0] }));
  await page.goto("/?chamber=house&run=fixture-fixed-2022-h14&race=2022%3ACA-02%3Ahouse%3Aregular&view=actual");
  const views = await savedViews(page);
  const before = page.url();
  await views.getByLabel("Name this House view", { exact: true }).fill("California reference");
  await views.getByRole("button", { name: "Save House view", exact: true }).click();
  await expect(views.getByRole("button", { name: "Restore California reference", exact: true })).toBeEnabled();
  expect(page.url()).toBe(before);
  const stored = await page.evaluate(key => JSON.parse(localStorage.getItem(key) || "{}"), HOUSE_SAVED_VIEWS_STORAGE_KEY);
  expect(stored.schemaVersion).toBe(1);
  expect(stored.views).toHaveLength(1);
  expect(stored.views[0]).toMatchObject({ runId: "fixture-fixed-2022-h14", resultSha256: publications[0].resultSha256, configurationSha256: publications[0].result.run.configurationSha256,
    year: 2022, mapVintage: "house_election_map_2022", raceId: "2022:CA-02:house:regular", mapView: "actual", label: "California reference" });
  await page.getByLabel("House model", { exact: true }).selectOption("fixture-context-2022-h14");
  await (await controls(page)).getByRole("button", { name: "2018", exact: true }).click();
  await expect(page).toHaveURL(/run=fixture-context-2018-h14/);
  await (await savedViews(page)).getByRole("button", { name: "Restore California reference", exact: true }).click();
  await expect(page).toHaveURL(/run=fixture-fixed-2022-h14&race=2022%3ACA-02%3Ahouse%3Aregular&view=actual/);
  await expect(page.getByRole("complementary", { name: "Selected House race detail", exact: true })).toContainText("CA-02");
  await expect(page.getByRole("group", { name: "Map view", exact: true }).getByRole("button", { name: "Actual winners", exact: true })).toHaveAttribute("aria-pressed", "true");
  await page.goBack();
  await expect(page).toHaveURL(/run=fixture-context-2018-h14/);
  await expect(page.getByLabel("House model", { exact: true })).toHaveValue("fixture-context-2018-h14");
  await page.goForward();
  await expect(page.getByLabel("House model", { exact: true })).toHaveValue("fixture-fixed-2022-h14");
  await expect(page.getByRole("complementary", { name: "Selected House race detail", exact: true })).toContainText("CA-02");
  await page.reload();
  const restored = await savedViews(page);
  await expect(restored.getByRole("button", { name: "Restore California reference", exact: true })).toBeEnabled();
  await restored.getByRole("button", { name: "Delete California reference", exact: true }).click();
  await expect(restored.getByRole("button", { name: "Restore California reference", exact: true })).toHaveCount(0);
  expect(await page.evaluate(key => JSON.parse(localStorage.getItem(key) || "{}").views, HOUSE_SAVED_VIEWS_STORAGE_KEY)).toEqual([]);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBeTruthy();
});

test("a saved House restore rejects a changed result and retry retains the exact saved identity", async ({ page }) => {
  const publications = collection();
  const saved = createHouseSavedView("Original result", publications[1], { raceId: "2022:CA-02:house:regular", mapView: "actual" });
  await page.addInitScript(({ key, raw }) => localStorage.setItem(key, raw), { key: HOUSE_SAVED_VIEWS_STORAGE_KEY, raw: serializeHouseSavedViews([saved]) });
  let changed = true;
  let release: (() => void) | undefined;
  await page.route(historicalApi, async route => {
    const id = new URL(route.request().url()).searchParams.get("run");
    if (id !== publications[1].result.run.id) return route.fulfill({ json: publications[0] });
    if (!changed) return route.fulfill({ json: publications[1] });
    await new Promise<void>(resolve => { release = resolve; });
    const replacement = structuredClone(publications[1]);
    replacement.result.limitations.push("Synthetic updated result, not the saved artifact.");
    replacement.resultSha256 = fixtureSha256(replacement.result);
    replacement.catalog!.runs.find(run => run.id === id)!.resultSha256 = replacement.resultSha256;
    return route.fulfill({ json: replacement });
  });
  await page.goto("/?chamber=house&edition=historical&year=2022&race=2022%3ACA-01%3Ahouse%3Aregular");
  const views = await savedViews(page), before = page.url();
  await views.getByLabel("Name this House view", { exact: true }).fill("Pending view");
  await expect(views.getByRole("button", { name: "Save House view", exact: true })).toBeEnabled();
  await views.getByRole("button", { name: "Restore Original result", exact: true }).click();
  await expect(page.locator(".compare-banner[role=status]")).toContainText("Still displaying");
  await expect(views.getByRole("button", { name: "Save House view", exact: true })).toBeDisabled();
  expect(page.url()).toBe(before);
  await expect.poll(() => typeof release).toBe("function");
  release!();
  await expect(page.locator(".compare-banner[role=alert]")).toContainText("previous forecast remains visible");
  expect(page.url()).toBe(before);
  await expect(page.getByLabel("House model", { exact: true })).toHaveValue("fixture-fixed-2022-h14");
  await expect(views.getByRole("button", { name: "Save House view", exact: true })).toBeDisabled();
  changed = false;
  await page.getByRole("button", { name: "Try selection again", exact: true }).click();
  await expect(page).toHaveURL(/run=fixture-context-2022-h14&race=2022%3ACA-02%3Ahouse%3Aregular&view=actual/);
  await expect(page.getByRole("group", { name: "Map view", exact: true }).getByRole("button", { name: "Actual winners", exact: true })).toHaveAttribute("aria-pressed", "true");
});

test("withdrawn and changed House views remain listed, explained and deletable", async ({ page }) => {
  const publications = collection();
  const entries = [createHouseSavedView("Withdrawn model", publications[1], { raceId: "2022:CA-02:house:regular", mapView: "forecast" }),
    createHouseSavedView("Changed result", publications[2], { raceId: "2018:CA-01:house:regular", mapView: "forecast" })];
  publications[0].catalog!.runs[1] = { ...publications[0].catalog!.runs[1], available: false, resultSha256: null, unavailableReason: "Model withdrawn pending review." };
  publications[0].catalog!.runs[2].resultSha256 = "a".repeat(64);
  await page.addInitScript(({ key, raw }) => localStorage.setItem(key, raw), { key: HOUSE_SAVED_VIEWS_STORAGE_KEY, raw: serializeHouseSavedViews(entries) });
  await page.route(historicalApi, route => route.fulfill({ json: publications[0] }));
  await page.goto("/?chamber=house");
  const views = await savedViews(page);
  await expect(views.getByRole("button", { name: "Restore Withdrawn model", exact: true })).toBeDisabled();
  await expect(views).toContainText("Model withdrawn pending review.");
  await expect(views.getByRole("button", { name: "Restore Changed result", exact: true })).toBeDisabled();
  await expect(views).toContainText(/changed|differs/);
  await views.getByRole("button", { name: "Delete Withdrawn model", exact: true }).click();
  await expect(views.getByRole("button", { name: "Restore Withdrawn model", exact: true })).toHaveCount(0);
  await expect(views.getByRole("button", { name: "Restore Changed result", exact: true })).toBeDisabled();
});

test("a browser storage failure does not claim a House view was saved", async ({ page }) => {
  await page.addInitScript(() => { Storage.prototype.setItem = () => { throw new DOMException("Test storage limit", "QuotaExceededError"); }; });
  const publications = collection();
  await page.route(historicalApi, route => route.fulfill({ json: publications[0] }));
  await page.goto("/?chamber=house");
  const views = await savedViews(page), before = page.url();
  await views.getByLabel("Name this House view", { exact: true }).fill("Unsaved example");
  await views.getByRole("button", { name: "Save House view", exact: true }).click();
  await expect(views).toContainText(/Couldn.t save|could not be saved|Unable to save/i);
  await expect(views.getByRole("button", { name: "Restore Unsaved example", exact: true })).toHaveCount(0);
  expect(page.url()).toBe(before);
});

test("an actual-winner view can retain an unknown selected result without substituting a forecast", async ({ page }) => {
  const publications = collection(), publication = publications[0];
  delete publication.result.races.find(race => race.id === "2022:CA-02:house:regular")!.observed;
  publication.resultSha256 = fixtureSha256(publication.result);
  publication.catalog.runs[0].resultSha256 = publication.resultSha256;
  await page.route(historicalApi, route => route.fulfill({ json: publication }));
  await page.goto("/?chamber=house&race=2022%3ACA-02%3Ahouse%3Aregular&view=actual");
  const views = await savedViews(page);
  await views.getByLabel("Name this House view", { exact: true }).fill("Unknown actual winner");
  await views.getByRole("button", { name: "Save House view", exact: true }).click();
  await expect(views.getByRole("button", { name: "Restore Unknown actual winner", exact: true })).toBeEnabled();
  await page.getByRole("group", { name: "Map view", exact: true }).getByRole("button", { name: "Forecast", exact: true }).click();
  await views.getByRole("button", { name: "Restore Unknown actual winner", exact: true }).click();
  await expect(page).toHaveURL(/race=2022%3ACA-02%3Ahouse%3Aregular&view=actual/);
  await expect(page.getByRole("group", { name: "Map view", exact: true }).getByRole("button", { name: "Actual winners", exact: true })).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByRole("complementary", { name: "Selected House race detail", exact: true }).locator(".observed-result")).toHaveCount(0);
});

test("House builder shows all eight roles with only its own saved fundamentals selector active", async ({ page }) => {
  const publications = collection();
  await page.route(historicalApi, route => route.fulfill({ json: publications[0] }));
  await page.goto("/?chamber=house");
  const panel = page.locator("details.house-configuration");
  await expect(panel).toBeVisible();
  await expect(panel).not.toHaveAttribute("open", "");
  await controls(page);
  await expect(panel.locator(".model-slice")).toHaveCount(8);
  await expect(panel.locator(".slice-control")).toHaveCount(1);
  await expect(panel.locator(".model-slice.is-unavailable")).toHaveCount(4);
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
  await expect(panel.locator(".model-slice.is-fixed")).toHaveCount(3);
  for (const slice of await panel.locator(".model-slice.is-fixed").all()) {
    await expect(slice.getByRole("checkbox")).toHaveCount(0);
    await expect(slice).toContainText("Built into this model");
  }
  const fundamentals = panel.locator('[data-slice="fundamentals"]');
  await expect(fundamentals.getByText("Always included", { exact: true })).toBeVisible();
  await expect(fundamentals.getByLabel("House model", { exact: true })).toHaveValue("fixture-fixed-2022-h14");
  await expect(panel.locator('[data-slice="measurement"]')).toContainText("Polling is not connected");
  await expect(panel.locator('[data-slice="environment"]')).toContainText("bundled into each saved House model");
  await expect(panel.locator('[data-slice="counts_rules"]')).toContainText("does not model vote counts or cover all 435 districts");
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
  await page.route(historicalApi, route => route.fulfill({ json: publication }));
  await page.goto("/?chamber=house");
  const panel = await controls(page);
  await expect(page.getByLabel("House model", { exact: true }).locator('option[value="fixture-context-2022-h14"]')).toBeDisabled();
  await panel.getByText("Why some choices are unavailable", { exact: true }).click();
  await expect(panel).toContainText("Synthetic diagnostic failure; result withheld.");
  await expect(panel.locator(".model-slice.is-unavailable")).toHaveCount(4);
  await expect(panel.locator('[data-slice="measurement"]').getByRole("checkbox")).toBeDisabled();
  await expect(page.getByRole("region", { name: "House control availability", exact: true })).toContainText("Not available for this result");
});

test("a fixed legacy House result shows its model without inventing a selector", async ({ page }) => {
  const publication = makeHousePublication({ eligibleCount: 16, withGeography: false });
  publication.result.run.id = "fixture-fixed-2022-h14";
  publication.resultSha256 = fixtureSha256(publication.result);
  await page.route(historicalApi, route => route.fulfill({ json: publication }));
  await page.goto("/?chamber=house");
  const panel = await controls(page);
  await expect(panel.locator(".model-slice")).toHaveCount(8);
  await expect(panel.locator('[data-slice="fundamentals"] .slice-fixed-choice')).toHaveText(publication.result.run.label);
  await expect(panel.getByRole("combobox")).toHaveCount(0);
  await expect(panel.getByLabel("House model", { exact: true })).toHaveCount(0);
  await expect(panel.locator(".model-slice.is-unavailable")).toHaveCount(4);
});

test("House model/year selections restore exact results, scope and history without carrying districts across maps", async ({ page }) => {
  const publications = collection();
  await page.route(historicalApi, (route) => {
    const id = new URL(route.request().url()).searchParams.get("run");
    return route.fulfill({ json: publications.find((p) => p.result.run.id === id) || publications[0] });
  });
  await page.goto("/?chamber=house&race=2022%3ACA-02%3Ahouse%3Aregular&view=actual");
  const panel = page.locator("details.house-configuration");
  await expect(panel).toBeVisible();
  await expect(panel).not.toHaveAttribute("open", "");
  await controls(page);
  await expect(page.getByLabel("House model", { exact: true })).toHaveValue("fixture-fixed-2022-h14");
  await page.getByLabel("House model", { exact: true }).selectOption("fixture-context-2022-h14");
  await expect(page).toHaveURL(/run=fixture-context-2022-h14/);
  await expect(panel).toHaveAttribute("open", "");
  expect(new URL(page.url()).searchParams.get("race")).toBe("2022:CA-02:house:regular");
  await expect(page.getByRole("region", { name: "How this model did", exact: true })).toContainText("20 / 435");
  await panel.getByRole("button", { name: "2018", exact: true }).click();
  await expect(page).toHaveURL(/run=fixture-context-2018-h14/);
  expect(new URL(page.url()).searchParams.get("race")).toBeNull();
  expect(new URL(page.url()).searchParams.get("view")).toBe("actual");
  await page.goBack();
  await expect(page.getByLabel("House model", { exact: true })).toHaveValue("fixture-context-2022-h14");
  expect(new URL(page.url()).searchParams.get("race")).toBe("2022:CA-02:house:regular");
  await page.reload();
  await controls(page);
  await expect(page.getByLabel("House model", { exact: true })).toHaveValue("fixture-context-2022-h14");
  await expect(page.getByRole("complementary", { name: "Selected House race detail", exact: true })).toContainText("CA-02");
  await page.getByRole("button", { name: "Share House view ↗", exact: true }).click();
  await expect(page.getByLabel("Shareable House view", { exact: true })).toHaveValue(page.url());
  await page.getByRole("button", { name: "Reset House view", exact: true }).click();
  await expect(page).toHaveURL(/run=fixture-fixed-2022-h14/);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBeTruthy();
});

for (const failure of ["service", "wrong result"] as const) {
  test(`a House ${failure} failure preserves the displayed model and permits retry`, async ({ page }) => {
    const publications = collection();
    let fail = true;
    await page.route(historicalApi, (route) => {
      const id = new URL(route.request().url()).searchParams.get("run");
      if (id === "fixture-context-2022-h14" && fail) return failure === "service"
        ? route.fulfill({ status: 503, json: { error: "Test failure" } })
        : route.fulfill({ json: publications[0] });
      return route.fulfill({ json: publications.find((p) => p.result.run.id === id) || publications[0] });
    });
    await page.goto("/?chamber=house");
    await controls(page);
    await page.getByLabel("House model", { exact: true }).selectOption("fixture-context-2022-h14");
    await expect(page.locator('.compare-banner[role="alert"]')).toContainText("previous forecast remains visible");
    await expect(page).toHaveURL(/run=fixture-fixed-2022-h14/);
    await expect(page.getByLabel("House model", { exact: true })).toHaveValue("fixture-fixed-2022-h14");
    await expect(page.getByRole("region", { name: "How this model did", exact: true })).toContainText("16 / 435");
    fail = false;
    await page.getByRole("button", { name: "Try selection again", exact: true }).click();
    await expect(page).toHaveURL(/run=fixture-context-2022-h14/);
    await expect(page.locator('.compare-banner[role="alert"]')).toHaveCount(0);
    await expect(page.getByRole("region", { name: "How this model did", exact: true })).toContainText("20 / 435");
  });
}

test("an unknown House deep link explains its missing edition and can recover to historical research", async ({ page }) => {
  const publications = collection();
  await page.route(historicalApi, (route) => new URL(route.request().url()).searchParams.has("run")
    ? route.fulfill({ status: 400, json: { error: "Unknown model" } })
    : route.fulfill({ json: publications[0] }));
  await page.goto("/?chamber=house&run=unknown");
  await expect(page.getByRole("heading", { name: "This House link could not be opened", exact: true })).toBeVisible();
  await expect(page.getByRole("alert")).toContainText("does not identify a current or historical edition");
  await page.getByRole("button", { name: "Explore historical House research", exact: true }).click();
  await expect(page).toHaveURL(/run=fixture-fixed-2022-h14/);
  await expect(page.locator("details.house-configuration")).toBeVisible();
});
