import { expect, test, type Page } from "@playwright/test";

const defaultRun = "house_history_refreshed_v1-2024-h14";
const defaultResultSha =
  "0548d05a65b7575780da7e1e0a6d4c3cdb4d356a300fd877e8dd019c4d302a0c";
const defaultConfigurationSha =
  "6fb45986f5acdeefe90bad6aaa594ccb0a872f196a4f275467d76d8809b72cd0";
const readyHeading = /The House,\s*in perspective\./;
const storageKey = "ballotmancy.house.saved-views.v1";

const scoreboard = (page: Page) =>
  page.getByRole("region", { name: "How this model did", exact: true });
const scoreValue = (page: Page, name: string) =>
  scoreboard(page)
    .getByRole("article")
    .filter({ has: page.getByRole("heading", { name, exact: true }) })
    .locator(".scoreboard-value");
async function openControls(page: Page) {
  const controls = page.locator("details.house-configuration");
  if (!(await controls.evaluate((node) => (node as HTMLDetailsElement).open)))
    await controls.locator(":scope > summary").click();
  return controls;
}
async function openSavedViews(page: Page) {
  await openControls(page);
  const saved = page.locator("details.house-saved-views");
  if (!(await saved.evaluate((node) => (node as HTMLDetailsElement).open)))
    await saved.locator(":scope > summary").click();
  return saved;
}

test("the real published API offers sixteen exact historical race results with election-specific maps and without chamber control", async ({
  request,
}) => {
  const response = await request.get("/api/chambers/house");
  expect(response.status()).toBe(200);
  const publication = await response.json();
  expect(publication.status).toBe("available");
  expect(publication.result.run.id).toBe(defaultRun);
  expect(publication.resultSha256).toBe(defaultResultSha);
  const catalog = publication.catalog;
  expect(catalog.defaultRunId).toBe(defaultRun);
  expect(catalog.runs).toHaveLength(16);
  expect(catalog.capabilities).toMatchObject({
    historical_races: true,
    polling: false,
    district_geometry: true,
    full_chamber: false,
    chamber_control: false,
    chamber_tipping: false,
    current_2026: false,
  });
  const expectedCoverage: Record<number, number> = {
    2018: 371,
    2020: 383,
    2022: 373,
    2024: 375,
  };
  for (const year of Object.keys(expectedCoverage).map(Number))
    expect(
      catalog.runs.filter((run: { year: number }) => run.year === year),
    ).toHaveLength(4);
  for (const entry of catalog.runs) {
    expect(entry.available).toBe(true);
    expect(entry.unavailableReason).toBeNull();
    const selectedResponse = await request.get(
      `/api/chambers/house?${new URLSearchParams({ run: entry.id })}`,
    );
    expect(selectedResponse.status()).toBe(200);
    const selected = await selectedResponse.json();
    const result = selected.result;
    expect(selected.status).toBe("available");
    expect(result.run.id).toBe(entry.id);
    expect(selected.resultSha256).toBe(entry.resultSha256);
    expect(result.run.configurationSha256).toBe(entry.configurationSha256);
    expect(result.run.historical).toBe(true);
    expect(result.run.experimental).toBe(true);
    expect(result.races).toHaveLength(expectedCoverage[entry.year]);
    expect(result.coverage).toHaveLength(435);
    expect(
      result.coverage.filter((row: { eligible: boolean }) => row.eligible),
    ).toHaveLength(result.races.length);
    expect(result.metadata).toMatchObject({
      scope: "eligible_race_subset",
      full_chamber: false,
      chamber_size: 435,
      majority_threshold: 218,
      mapVintage: `house_election_map_${entry.year}`,
    });
    expect(selected.geography.metadata).toMatchObject({ cycle: entry.year, mapVintage: `house_election_map_${entry.year}`, admission: "verified" });
    expect(selected.geography.geojson.features).toHaveLength(435);
    expect(result.chamberProof).toBeUndefined();
    expect(result.tippingProof).toBeUndefined();
    for (const race of result.races) {
      const winner = race.outcomes.find(
        (outcome: { id: string }) => outcome.id === race.observed.outcomeId,
      );
      expect(winner).toBeTruthy();
      expect(winner.name).toBe(race.observed.winnerName);
      expect(winner.party).toBe(race.observed.winnerParty);
      expect(race.district).toMatchObject({
        cycle: entry.year,
        mapVintage: `house_election_map_${entry.year}`,
      });
    }
    expect(result.limitations.join(" ")).toMatch(/retrospective/i);
    expect(result.limitations.join(" ")).toMatch(/subtotal.*precision/i);
    expect(JSON.stringify(selected)).not.toMatch(
      /\/home\/|\.redesign-runtime\/|source-snapshots|file:\/\//,
    );
  }
  expect(
    (
      await request.get("/api/chambers/house?run=not-a-published-house-result")
    ).status(),
  ).toBe(400);
});

test("the actual default 2024 scorecard has independently audited scores and a matching map without chamber results", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/?chamber=house");
  await expect(page.getByRole("heading", { name: readyHeading })).toBeVisible();
  await expect(page).toHaveURL(new RegExp(`run=${defaultRun}`));
  await expect(page.locator("details.house-configuration")).not.toHaveAttribute(
    "open",
    "",
  );
  await expect(scoreValue(page, "Favorites that won")).toHaveText("365 / 375");
  await expect(scoreValue(page, "Probability error")).toHaveText("0.045");
  await expect(scoreValue(page, "Tested coverage")).toHaveText("375 / 435");
  await expect(scoreboard(page)).toContainText("0 exact ties excluded");
  const ribbon = page.getByRole("region", {
    name: "House race ribbon",
    exact: true,
  });
  await expect(ribbon.locator("button[data-race-id]")).toHaveCount(375);
  await expect(page.locator(".district-map-svg")).toHaveCount(1);
  await expect(page.locator(".district-map-svg [data-forecast=available]")).toHaveCount(375);
  await expect(page.locator(".district-map-svg [data-forecast=unavailable]")).toHaveCount(60);
  await expect(
    page.getByRole("region", { name: "House map", exact: true }),
  ).toContainText("375 of 375 forecast districts match this election’s district map");
  await expect(
    page.getByRole("region", {
      name: "House control availability",
      exact: true,
    }),
  ).toContainText("Not available for this result");
  await expect(
    page.locator(
      ".prob-number,.house-tipping-summary,.house-ribbon-cell.majority-position",
    ),
  ).toHaveCount(0);
  await openControls(page);
  await expect(
    page.getByLabel("House model", { exact: true }).locator("option"),
  ).toHaveCount(4);
  await expect(
    page
      .getByRole("group", { name: "Election year", exact: true })
      .getByRole("button", { name: "2026", exact: true }),
  ).toHaveCount(0);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBeTruthy();
  expect(errors).toEqual([]);
});

test("real 2018 year selection loads its own 371 races and retains all four reviewed models", async ({
  page,
}) => {
  await page.goto("/?chamber=house");
  await expect(scoreValue(page, "Tested coverage")).toHaveText("375 / 435");
  const controls = await openControls(page);
  await controls
    .getByRole("group", { name: "Election year", exact: true })
    .getByRole("button", { name: "2018", exact: true })
    .click();
  await expect(page).toHaveURL(/run=house_history_refreshed_v1-2018-h14/);
  await expect(scoreValue(page, "Tested coverage")).toHaveText("371 / 435");
  await expect(scoreValue(page, "Favorites that won")).toHaveText("328 / 371");
  await expect(scoreValue(page, "Probability error")).toHaveText("0.173");
  await expect(
    page.getByLabel("House model", { exact: true }).locator("option"),
  ).toHaveCount(4);
  await page
    .getByLabel("House model", { exact: true })
    .selectOption("house_relative_context_v1-2018-h14");
  await expect(page).toHaveURL(/run=house_relative_context_v1-2018-h14/);
  await expect(scoreValue(page, "Favorites that won")).toHaveText("348 / 371");
  await expect(scoreValue(page, "Probability error")).toHaveText("0.098");
  await expect(
    page
      .getByRole("region", { name: "House race ribbon", exact: true })
      .locator("button[data-race-id]"),
  ).toHaveCount(371);
});

test("a real saved House view restores exact model, race, map edition and actual mode after another model loads", async ({
  page,
}) => {
  await page.goto("/?chamber=house");
  await expect(scoreValue(page, "Tested coverage")).toHaveText("375 / 435");
  await page
    .getByRole("region", { name: "House race ribbon", exact: true })
    .locator('[data-race-id="2024:CA-22:house:regular"]')
    .click();
  await expect(
    page
      .getByRole("complementary", {
        name: "Selected House race detail",
        exact: true,
      })
      .getByRole("heading", { name: "CA-22", exact: true }),
  ).toBeVisible();
  await page
    .getByRole("group", { name: "Map view", exact: true })
    .getByRole("button", { name: "Actual winners", exact: true })
    .click();
  await openSavedViews(page);
  await page
    .getByLabel("Name this House view", { exact: true })
    .fill("Audited 2024 district");
  await page
    .getByRole("button", { name: "Save House view", exact: true })
    .click();
  await expect(
    page.getByRole("button", {
      name: "Restore Audited 2024 district",
      exact: true,
    }),
  ).toBeEnabled();
  const record = await page.evaluate(
    (key) => JSON.parse(localStorage.getItem(key)!).views[0],
    storageKey,
  );
  expect(record).toMatchObject({
    runId: defaultRun,
    resultSha256: defaultResultSha,
    configurationSha256: defaultConfigurationSha,
    year: 2024,
    mapVintage: "house_election_map_2024",
    raceId: "2024:CA-22:house:regular",
    mapView: "actual",
  });
  await page
    .getByLabel("House model", { exact: true })
    .selectOption("house_relative_context_v1-2024-h14");
  await expect(page).toHaveURL(/run=house_relative_context_v1-2024-h14/);
  await expect(scoreValue(page, "Probability error")).toHaveText("0.035");
  await page
    .getByRole("button", { name: "Restore Audited 2024 district", exact: true })
    .click();
  await expect(page).toHaveURL(new RegExp(`run=${defaultRun}`));
  expect(new URL(page.url()).searchParams.get("race")).toBe(
    "2024:CA-22:house:regular",
  );
  expect(new URL(page.url()).searchParams.get("view")).toBe("actual");
  await expect(scoreValue(page, "Probability error")).toHaveText("0.045");
  await page.reload();
  await expect(page.getByRole("heading", { name: readyHeading })).toBeVisible();
  await openSavedViews(page);
  await expect(
    page.getByRole("button", {
      name: "Restore Audited 2024 district",
      exact: true,
    }),
  ).toBeEnabled();
  await expect(
    page
      .getByRole("complementary", {
        name: "Selected House race detail",
        exact: true,
      })
      .getByRole("heading", { name: "CA-22", exact: true }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBeTruthy();
});
