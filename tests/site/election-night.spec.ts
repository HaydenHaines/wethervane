import { test, expect } from "@playwright/test";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { createHash } from "node:crypto";

const read = (path: string) => JSON.parse(readFileSync(resolve(path), "utf8"));
const senateRegistry = read("public/data/explorer/registry.json");
const houseCatalog = read("data/house-live/catalog.json");
const standardReference = read("public/data/election-night/references.json").entries[0].reference;
const coreReferences = read("public/data/election-night/core-references.json").entries;
const displayProbability = (p: number) => p > 0 && p < .001 ? "<0.1%" : p < 1 && p > .999 ? ">99.9%" : `${(100 * p).toFixed(1)}%`;

test("locked evaluation follows admitted updates without changing the live forecast or assigning a partial grade", async ({ page }) => {
  // Every result, candidate and score in these mocked responses is invented.
  const noLock = read("tests/fixtures/election-night/noLock.json");
  const called = read("tests/fixtures/election-night/called.json");
  const corrected = read("tests/fixtures/election-night/corrected.json");
  const graded = read("tests/fixtures/election-night/graded.json");
  const defaultRun = senateRegistry.runs.find((row: { id: string }) => row.id === senateRegistry.defaults["2026"]);
  let response = noLock;
  await page.route(url => url.pathname === "/api/election-night/evaluation", route => route.fulfill({
    status: response.status === "unavailable" ? 503 : 200, contentType: "application/json", body: JSON.stringify(response),
  }));
  await page.goto("/");
  await page.getByRole("button", { name: "Election night scorecard", exact: true }).click();
  const panel = page.locator("#election-night-scorecard");
  await expect(panel.locator(".night-edition")).toContainText("No locked election-night evaluation has been published");
  const firstRace = await panel.locator(".night-featured .night-race").first().getAttribute("data-scorecard-race");
  expect(firstRace).toBeTruthy();
  const adapt = (value: typeof called) => {
    // Map one invented row to a displayed race ID to exercise per-race status;
    // the API response is mocked and does not admit any actual result.
    const original = value.rows[0].raceId;
    const result = JSON.parse(JSON.stringify(value).split(original).join(firstRace!));
    result.lock.model.resultSha256 = defaultRun.sha256;
    return result;
  };
  response = adapt(called);
  await page.evaluate(() => window.dispatchEvent(new Event("focus")));
  const evaluation = panel.getByRole("region", { name: "Evaluation of the locked forecast" });
  await expect(evaluation).toContainText("This is the forecast currently shown above");
  await expect(evaluation).toContainText("0 of 2 graded winner results");
  await expect(panel.locator(".night-featured .night-race").first()).toContainText("See the locked evaluation above");
  await expect(panel.locator(".night-featured .night-race").nth(1)).toContainText("Awaiting verified result");
  await evaluation.getByText("Results reported for the locked forecast", { exact: false }).click();
  await expect(evaluation).toContainText("INVENTED First Candidate · called");
  await expect(evaluation).toContainText("preliminary");
  await expect(evaluation).toContainText("Not scored");
  await expect(panel.locator(".night-rubric")).toContainText("No final letter grade assigned");

  response = adapt(corrected);
  await page.evaluate(() => window.dispatchEvent(new Event("focus")));
  await expect(evaluation).toContainText("1 of 2 graded winner results");
  const results = evaluation.getByText("Results reported for the locked forecast", { exact: false });
  if (!(await results.evaluate(node => (node.parentElement as HTMLDetailsElement).open))) await results.click();
  await expect(evaluation).toContainText("9.0 points from the average; outside the 90% range");
  await expect(panel.locator(".night-rubric")).toContainText("No final letter grade assigned");

  response = adapt(graded);
  await page.evaluate(() => window.dispatchEvent(new Event("focus")));
  await expect(panel.locator(".night-rubric")).toContainText(`Overall grade: ${graded.rubric.overall.grade}`);
  await expect(panel.locator(".night-grade-active")).toHaveText(graded.rubric.overall.grade);
  await panel.locator(".night-models > summary").click();
  await panel.getByLabel("Scorecard model").selectOption("current-fundamentals");
  await expect(evaluation).toContainText("The forecast shown above is a different edition");
  await expect(panel.locator(".night-rubric")).toContainText(`Overall grade: ${graded.rubric.overall.grade}`);

  response = { ...noLock, status: "unavailable", errorCode: "ledger_verification_failed" };
  await page.evaluate(() => window.dispatchEvent(new Event("focus")));
  await expect(panel.locator(".night-rubric")).toContainText("evaluation status could not be verified");
  await expect(panel.locator(".night-grade-active")).toHaveCount(0);
  await expect(panel.locator(".night-evaluation")).toHaveCount(0);
  await expect(panel.locator(".night-featured .night-race")).toHaveCount(6);
});

test("House expected errors stay separate for 435 winner groups and 412 named pairs across a same-run revision", async ({ page }, testInfo) => {
  // Compatibility sidecar is a synthetic browser fixture, never an admitted publication.
  const sidecar = read("tests/fixtures/election-night/house-compatibility-sidecar.json");
  const run = houseCatalog.entries.find((row: { id: string }) => row.id === "house-current-fundamentals");
  const digest = (body: string) => createHash("sha256").update(body).digest("hex");
  const h = (digit: string) => digit.repeat(64);
  const pins = { collectionSha256: h("a"), collectionReceiptSha256: h("b"), nativeCatalogSha256: sidecar.catalog.sha256,
    readyPacketSha256: sidecar.ready_packet.sha256, referenceSha256: h("c"), sidecarReceiptSha256: h("d"),
    reviewSha256: h("e"), auditorSha256: h("f"), sourceReceiptSha256: sidecar.source_receipt.sha256,
    jointDrawsSha256: sidecar.joint_draws.sha256, sourcePlanSha256: sidecar.source_plan.sha256,
    conditionalSharesSha256: sidecar.conditional_shares.sha256, jointMetricArchiveSha256: sidecar.joint_metric_archive.sha256,
    scientificReviewSha256: sidecar.scientific_review.sha256, nativeNumericalReviewSha256: sidecar.native_numerical_review.sha256,
    qualificationSha256: sidecar.qualification.sha256, selectionSha256: sidecar.selection.sha256,
    builderSha256: sidecar.builder.sha256, operatorReviewSha256: h("1") };
  const record = { chamber: "house", runId: run.id, resultSha256: run.result.sha256,
    configuration: run.configuration, forecastIssuedAt: sidecar.forecast_issued_at,
    forecastCutoffAt: sidecar.forecast_cutoff, referenceIssuedAt: sidecar.issued_at,
    rows: sidecar.draw_layout.rows, chains: sidecar.draw_layout.chains,
    winnerRaceCount: 435, pairRaceCount: 412, pairInapplicableRaceCount: 23, sourcePins: pins, reference: sidecar };
  const catalog = { schema: "HouseExpectedErrorPublicCatalogV1", schemaVersion: 1, chamber: "house",
    defaultRunId: run.id, sourceCatalogSha256: pins.collectionSha256,
    nativeCatalogSha256: pins.nativeCatalogSha256, readyPacketSha256: pins.readyPacketSha256,
    evaluationLock: false, observedResultsConsumed: false, entries: [record] };
  const { reference: unused, chamber, runId, resultSha256, ...qualification } = record;
  const index = read("public/data/election-night/publication-index.json");
  const catalogBody = () => JSON.stringify(catalog);
  const updateIndex = () => { index.house = { schemaVersion: 1, href: "/data/election-night/house-references.json",
    sha256: digest(catalogBody()), editions: [{ chamber, runId, resultSha256: record.resultSha256, qualification }] }; };
  updateIndex();
  let revised = false;
  await page.route("**/data/election-night/publication-index.json", route => route.fulfill({ contentType: "application/json", body: JSON.stringify(index) }));
  await page.route("**/data/election-night/house-references.json", route => route.fulfill({ contentType: "application/json", body: catalogBody() }));
  await page.route("**/api/chambers/house/live**", async route => {
    const response = await route.fetch();
    if (!revised) return route.fulfill({ response });
    const body = await response.json();
    if (!body.view) return route.fulfill({ response });
    body.view.run.resultSha256 = h("2");
    body.catalog.entries.find((entry: { id: string }) => entry.id === run.id).resultSha256 = h("2");
    if (body.polling) body.polling.resultSha256 = h("2");
    return route.fulfill({ response, body: JSON.stringify(body) });
  });
  await page.goto("/?chamber=house");
  await page.getByRole("button", { name: "Election night scorecard", exact: true }).click();
  const panel = page.locator("#election-night-scorecard");
  await panel.locator(".night-models summary").click();
  await panel.getByLabel("Scorecard model").selectOption(run.id);
  await expect(panel.locator(".night-edition")).toHaveAttribute("data-scorecard-result-sha", run.result.sha256);
  await expect(panel.locator(".night-reference-value")).toContainText("9.1");
  await expect(panel.locator(".night-miss-reference .night-reference-band")).toContainText("17 to 66 wrong choices");
  await expect(panel.locator(".night-race-expected-error")).not.toHaveCount(0);
  await panel.locator(".night-house-reference-details > summary").click();
  await expect(panel.locator(".night-house-reference-details")).toContainText("412 supported");
  await expect(panel.locator(".night-house-reference-details")).toContainText("23 contests have no pair error");
  await expect(panel.locator(".night-house-reference-details")).toContainText("separately qualified against a precision threshold");
  await panel.locator(".night-uncertainty-grid").first().screenshot({ path: testInfo.outputPath("house-expected-errors-synthetic.png") });
  revised = true;
  await page.evaluate(() => window.dispatchEvent(new Event("focus")));
  await expect(panel.locator(".night-edition")).toHaveAttribute("data-scorecard-result-sha", h("2"));
  await expect(panel.locator(".night-reference-value")).toHaveCount(0);
  record.resultSha256 = h("2"); sidecar.native_result.sha256 = h("2");
  updateIndex();
  await page.evaluate(() => window.dispatchEvent(new Event("focus")));
  await expect(panel.locator(".night-reference-value")).toContainText("9.1");
  await expect(panel.locator(".night-edition")).toHaveAttribute("data-scorecard-result-sha", h("2"));
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

test("open scorecard follows a revised same-ID public forecast, then its separately published exact references", async ({ page }) => {
  const old = senateRegistry.runs.find((row: { id: string }) => row.id === senateRegistry.defaults["2026"]);
  const nextRegistry = structuredClone(senateRegistry);
  const next = nextRegistry.runs.find((row: { id: string }) => row.id === old.id);
  const bundle = read(`public${old.href}`);
  const cutoff = "2026-10-27T12:00:00Z", issued = "2026-10-27T12:05:00Z";
  for (const run of [next, bundle.run]) {
    run.dateId = "2026-10-27"; run.dateLabel = "October 27, 2026";
    run.forecastAt = cutoff; run.cutoffAt = cutoff; run.issuedAt = issued;
    run.inputCutoffs.find((row: { label: string }) => row.label === "Polling inputs").at = cutoff;
  }
  const resultBody = JSON.stringify(bundle);
  const digest = (body: string) => createHash("sha256").update(body).digest("hex");
  next.sha256 = digest(resultBody);
  const full = read("public/data/election-night/references.json");
  const fullEntry = full.entries.find((row: { runId: string }) => row.runId === old.id);
  fullEntry.resultSha256 = next.sha256; fullEntry.forecastCutoffAt = cutoff; fullEntry.forecastIssuedAt = issued;
  const fullBody = JSON.stringify(full);
  const core = read("public/data/election-night/core-references.json");
  const coreEntry = core.entries.find((row: { runId: string }) => row.runId === old.id);
  coreEntry.resultSha256 = next.sha256; coreEntry.forecastCutoffAt = cutoff; coreEntry.forecastIssuedAt = issued;
  coreEntry.reference.edition = "2026-10-27"; coreEntry.reference.input_cutoff_utc = cutoff;
  const coreBody = JSON.stringify(core);
  const index = read("public/data/election-night/publication-index.json");
  index.full.sha256 = digest(fullBody); index.primary.sha256 = digest(coreBody);
  for (const publication of [index.full, index.primary]) {
    const qualified = publication.editions.find((row: { runId: string }) => row.runId === old.id);
    qualified.resultSha256 = next.sha256;
    qualified.qualification.forecastIssuedAt = issued;
    qualified.qualification.forecastCutoffAt = cutoff;
  }
  let revised = false, referencesPublished = false;
  await page.route("**/data/explorer/registry.json", route => revised
    ? route.fulfill({ contentType: "application/json", body: JSON.stringify(nextRegistry) }) : route.continue());
  await page.route(`**${old.href}`, route => revised
    ? route.fulfill({ contentType: "application/json", body: resultBody }) : route.continue());
  await page.route("**/data/election-night/publication-index.json", route => referencesPublished
    ? route.fulfill({ contentType: "application/json", body: JSON.stringify(index) }) : route.continue());
  await page.route("**/data/election-night/references.json", route => referencesPublished
    ? route.fulfill({ contentType: "application/json", body: fullBody }) : route.continue());
  await page.route("**/data/election-night/core-references.json", route => referencesPublished
    ? route.fulfill({ contentType: "application/json", body: coreBody }) : route.continue());
  await page.goto("/");
  await page.getByRole("button", { name: "Election night scorecard", exact: true }).click();
  const panel = page.locator("#election-night-scorecard");
  await expect(panel.locator(".night-edition")).toHaveAttribute("data-scorecard-result-sha", old.sha256);
  await expect(panel.locator(".night-reference-value")).toBeVisible();
  revised = true;
  await page.evaluate(() => window.dispatchEvent(new Event("focus")));
  await expect(panel.locator(".night-edition")).toHaveAttribute("data-scorecard-result-sha", next.sha256);
  await expect(panel.locator(".night-reference-value")).toHaveCount(0);
  await expect(panel.locator(".night-primary-reference")).toHaveCount(0);
  await expect(panel.locator(".night-margin-reference")).toContainText("No reviewed error reference for this edition");
  referencesPublished = true;
  await page.evaluate(() => window.dispatchEvent(new Event("focus")));
  await expect(panel.locator(".night-reference-value")).toContainText("5.6");
  await expect(panel.locator(".night-primary-reference")).toHaveAttribute("data-reference-run", old.id);
  await expect(panel.locator(".night-method code")).toHaveText(next.sha256);
  await expect(panel.getByLabel("Scorecard model")).toHaveValue(old.id);
  await expect(panel).toHaveAttribute("aria-busy", "false");
  nextRegistry.defaults["2026"] = "current-fundamentals";
  await page.evaluate(() => window.dispatchEvent(new Event("focus")));
  await expect(panel.getByLabel("Scorecard model")).toHaveValue("current-fundamentals");
  await expect(panel.locator(".night-edition")).toContainText("Standard forecast");
  await expect(panel.locator(".night-reference-value")).toHaveCount(0);
  await expect(panel.locator(".night-primary-reference")).toHaveAttribute("data-reference-run", "current-fundamentals");
});

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
  await expect(panel.locator(".night-edition")).toContainText("No locked election-night evaluation has been published");
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
    await expect(panel.locator(".night-edition")).toContainText("No locked election-night evaluation has been published");
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
      await panel.getByLabel("Find a race or candidate").fill("Schmidt");
      await expect(panel.locator(".night-predictions .night-race")).toHaveCount(1);
      await expect(panel.locator(".night-predictions .night-input-notice")).toContainText("NY-21 is under review");
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
      await expect(panel.locator(".night-margin-reference")).toContainText("across 35 supported named pairs");
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
