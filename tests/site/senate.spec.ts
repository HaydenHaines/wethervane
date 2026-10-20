import { expect, test } from "@playwright/test";
import type { SiteForecastSnapshot } from "../../lib/site-types";
import type { TippingSnapshot } from "../../lib/tipping-types";

test("public forecast preserves candidate parties and chamber-control assumptions", async ({ request }) => {
  const response = await request.get("/data/forecast.json");
  expect(response.ok()).toBeTruthy();
  const forecast = await response.json() as SiteForecastSnapshot;

  for (const model of Object.values(forecast.models)) {
    expect(model.races).toHaveLength(35);
    expect(new Set(model.races.map((race) => race.id)).size).toBe(35);
    expect(model.races.filter((race) => race.id.endsWith(":special")).map((race) => race.state).sort()).toEqual(["FL", "OH"]);
    expect(model.races.filter((race) => race.candidates[0].party === "I").map((race) => race.state).sort()).toEqual(["ID", "MT", "NE", "SD"]);

    for (const race of model.races) {
      expect(race.firstWinProbability + race.secondWinProbability).toBeCloseTo(1, 10);
      expect(race.shareQuantiles.p05).toBeLessThanOrEqual(race.shareQuantiles.p50);
      expect(race.shareQuantiles.p50).toBeLessThanOrEqual(race.shareQuantiles.p95);
    }

    const partySeats = Object.values(model.chamberDistributions.party)
      .reduce((total, distribution) => total + distribution.expectedSeats, 0);
    expect(partySeats).toBeCloseTo(100, 8);
    const unaligned = model.chamberDistributions.caucus.campaign_pledge_unaligned;
    // King and Sanders retain independent party labels while joining the D caucus.
    expect(unaligned.D.expectedSeats - model.chamberDistributions.party.D.expectedSeats).toBeCloseTo(2, 8);

    for (const [scenario, control] of Object.entries(model.controls)) {
      const seats = model.chamberDistributions.caucus[scenario];
      const democraticControl = seats.D.pmf.slice(51).reduce((sum, probability) => sum + probability, 0);
      // The saved forecast assigns the vice-presidential tie-break to Republicans.
      const republicanControl = seats.R.pmf.slice(50).reduce((sum, probability) => sum + probability, 0);
      expect(control.democraticMajorityProbability).toBeCloseTo(democraticControl, 10);
      expect(control.republicanMajorityProbability).toBeCloseTo(republicanControl, 10);
      expect(democraticControl + republicanControl + control.unresolvedOrganizingProbability).toBeCloseTo(1, 10);
    }
  }

  // Frozen source-artifact values independently checked against the October 20 runs.
  expect(forecast.models.polling.controls.campaign_pledge_unaligned.unresolvedOrganizingProbability).toBe(0.08305);
  expect(forecast.models.polling.controls.independent_opposition_caucuses_D.democraticMajorityProbability).toBe(0.541725);
  expect(forecast.models.fundamentals.controls.campaign_pledge_unaligned.republicanMajorityProbability).toBe(0.89352);
});

test("model and caucus controls preserve independent candidate identity and share intervals", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  const chamber = page.getByRole("region", { name: "Who could control the chamber?" });
  const detail = page.getByRole("complementary", { name: "Selected race detail" });
  await expect(chamber.getByText("8.3%", { exact: true })).toBeVisible();
  await expect(page.locator(".race-table tbody tr")).toHaveCount(35);
  await expect(page.locator("svg [role=button]")).toHaveCount(35);

  await page.getByRole("button", { name: /^Nebraska, 1 election,/ }).click();
  await expect(detail.getByRole("heading", { name: "Nebraska", exact: true })).toBeVisible();
  await expect(detail.getByText("Independent", { exact: true })).toBeVisible();
  await expect(detail.getByText("14.2%", { exact: true })).toBeVisible();
  await expect(detail.locator('[aria-label="Dan Osborn: mean 46.6%, 90% interval 41.4% to 51.8%"]')).toBeVisible();
  await expect(detail.locator('[aria-label="Pete Ricketts: mean 53.4%, 90% interval 48.2% to 58.6%"]')).toBeVisible();

  await page.getByLabel("Independent alignment").selectOption("independent_opposition_caucuses_D");
  await expect(chamber.getByText("54.2%", { exact: true })).toBeVisible();
  await expect(chamber.getByText("0.0%", { exact: true })).toBeVisible();
  await expect(detail.getByText("Independent", { exact: true })).toBeVisible();

  await page.getByRole("button", { name: "Fundamentals model", exact: true }).click();
  await expect(detail.getByRole("heading", { name: "Nebraska", exact: true })).toBeVisible();
  await expect(chamber.getByText("10.6%", { exact: true })).toBeVisible();
  await expect(detail.getByText("3.6%", { exact: true })).toBeVisible();
  await expect(detail.locator('[aria-label="Dan Osborn: mean 38.1%, 90% interval 28.2% to 48.7%"]')).toBeVisible();
  await expect(detail.locator('[aria-label="Pete Ricketts: mean 61.9%, 90% interval 51.3% to 71.8%"]')).toBeVisible();
  await page.getByLabel("Independent alignment").selectOption("independent_opposition_caucuses_R");
  await expect(chamber.getByText("90.6%", { exact: true })).toBeVisible();
  await page.getByLabel("Independent alignment").selectOption("campaign_pledge_unaligned");
  await expect(chamber.getByText("1.3%", { exact: true })).toBeVisible();
  expect(errors).toEqual([]);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy();
});

test("race search, special elections, and map keyboard selection work together", async ({ page }) => {
  await page.goto("/");
  const detail = page.getByRole("complementary", { name: "Selected race detail" });
  const search = page.getByRole("textbox", { name: "Search states or candidates" });
  await search.fill("special");
  await expect(page.locator(".race-table tbody tr")).toHaveCount(2);
  await page.getByRole("button", { name: "Florida Special election", exact: true }).click();
  await expect(detail.getByRole("heading", { name: "Florida", exact: true })).toBeVisible();
  await expect(detail.getByText("U.S. Senate · Special election", { exact: true })).toBeVisible();
  await search.fill("no such contender");
  await expect(page.getByText("No matching races. Try another state or candidate.", { exact: true })).toBeVisible();
  await search.fill("");
  await page.getByLabel("Sort races").selectOption("state");
  await expect(page.locator(".race-table tbody tr").first()).toContainText("Alabama");
  const alaska = page.getByRole("button", { name: /^Alaska, 1 election,/ });
  await alaska.focus();
  await page.keyboard.press("Enter");
  await expect(detail.getByRole("heading", { name: "Alaska", exact: true })).toBeVisible();
  await expect(alaska).toHaveAttribute("aria-pressed", "true");
});

test("two races in a state remain separately selectable", async ({ page }) => {
  await page.route("**/data/forecast.json", async (route) => {
    const response = await route.fetch();
    const forecast = await response.json() as SiteForecastSnapshot;
    for (const model of Object.values(forecast.models)) {
      const special = model.races.find((race) => race.state === "OH")!;
      model.races.push({ ...special, id: "2026:OH:senate:regular", seatId: "US-SEN-OH-CLASS-2" });
    }
    await route.fulfill({ response, json: forecast });
  });
  await page.goto("/");
  await expect(page.getByText("The tipping-point snapshot is unavailable. Map and race summaries remain available below.", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: /^Ohio, 2 elections,/ }).click();
  const seats = page.getByLabel("Choose election in Ohio");
  await expect(seats.locator("option")).toHaveCount(2);
  await seats.selectOption("2026:OH:senate:regular");
  await expect(page.getByRole("button", { name: "Ohio Regular election", exact: true })).toHaveAttribute("aria-pressed", "true");
  await page.getByRole("button", { name: "Ohio Special election", exact: true }).click();
  await expect(seats).toHaveValue("2026:OH:senate:special");
});

test("the race list remains usable when map geography fails", async ({ page }) => {
  await page.route("**/states-us.geojson", (route) => route.abort());
  await page.goto("/");
  await expect(page.getByText("Map unavailable. Explore every race in the list below.", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Texas Regular election", exact: true }).click();
  await expect(page.getByRole("complementary").getByRole("heading", { name: "Texas", exact: true })).toBeVisible();
});

test("joint-draw tipping frequencies retain the correct caucus thresholds and denominator", async ({ request }) => {
  const response = await request.get("/data/tipping-points.json");
  expect(response.ok()).toBeTruthy();
  const tipping = await response.json() as TippingSnapshot;
  const forecast = await (await request.get("/data/forecast.json")).json() as SiteForecastSnapshot;
  expect(tipping.method).toBe("caucus_rank_threshold_v1");
  expect(tipping.sources.pollingForecastSha256).toBe(forecast.sources.pollingSha256);
  expect(tipping.sources.fundamentalsForecastSha256).toBe(forecast.sources.fundamentalsSha256);
  const texas = "2026:TX:senate:regular";
  const independentIds = ["ID", "MT", "NE", "SD"].map((state) => `2026:${state}:senate:regular`);
  for (const modelKey of ["polling", "fundamentals"] as const) {
    const model = tipping.models[modelKey];
    expect(model.simulationCount).toBe(modelKey === "polling" ? 40000 : 100000);
    const expectedOrder = [...forecast.models[modelKey].races]
      .sort((a, b) => b.firstShareMean - a.firstShareMean || a.id.localeCompare(b.id))
      .map((race) => race.id);
    expect(model.raceOrder).toEqual(expectedOrder);
    for (const scenario of Object.values(model.scenarios)) {
      expect(scenario.D.thresholdSeats).toBe(51);
      expect(scenario.R.thresholdSeats).toBe(50);
      expect(scenario.D.retainedSeats).toBe(34);
      expect(scenario.R.retainedSeats).toBe(31);
      for (const caucus of [scenario.D, scenario.R]) {
        expect(caucus.projectedTippingRaceId).toBe(texas);
        expect(Object.keys(caucus.frequencyByRace).sort()).toEqual([...expectedOrder].sort());
        expect(Object.values(caucus.frequencyByRace).reduce((sum, p) => sum + p, 0)).toBeCloseTo(1, 10);
      }
    }
    const unaligned = model.scenarios.campaign_pledge_unaligned;
    expect(unaligned.D.requiredContestedWins).toBe(17);
    expect(unaligned.R.requiredContestedWins).toBe(19);
    expect(unaligned.D.eligibleRaceIds).toHaveLength(31);
    expect(unaligned.R.eligibleRaceIds).toHaveLength(35);
    expect(unaligned.D.frequencyByRace).not.toEqual(unaligned.R.frequencyByRace);
    for (const id of independentIds) expect(unaligned.D.frequencyByRace[id]).toBe(0);
    const alignedD = model.scenarios.independent_opposition_caucuses_D;
    expect(alignedD.D.frequencyByRace).toEqual(alignedD.R.frequencyByRace);
    expect(alignedD.D.frequencyByRace).toEqual(unaligned.R.frequencyByRace);
    const alignedR = model.scenarios.independent_opposition_caucuses_R;
    expect(alignedR.R.guaranteedRaceIds.sort()).toEqual([...independentIds].sort());
    expect(alignedR.R.requiredContestedWins).toBe(15);
    expect(alignedR.D.frequencyByRace).toEqual(alignedR.R.frequencyByRace);
    expect(alignedR.D.frequencyByRace).toEqual(unaligned.D.frequencyByRace);
    for (const id of independentIds) expect(alignedR.R.frequencyByRace[id]).toBe(0);
  }
  // Independently computed directly from the immutable 40,000 / 100,000 saved draw rows.
  expect(tipping.models.polling.scenarios.campaign_pledge_unaligned.D.frequencyByRace[texas]).toBe(0.128225);
  expect(tipping.models.polling.scenarios.campaign_pledge_unaligned.R.frequencyByRace[texas]).toBe(0.102975);
  expect(tipping.models.fundamentals.scenarios.campaign_pledge_unaligned.D.frequencyByRace[texas]).toBe(0.11148);
  expect(tipping.models.fundamentals.scenarios.campaign_pledge_unaligned.R.frequencyByRace[texas]).toBe(0.09966);
});

test("ribbon displays all margin-ordered races, coincident markers, and scenario-specific tipping frequencies", async ({ page, request }) => {
  const tipping = await (await request.get("/data/tipping-points.json")).json() as TippingSnapshot;
  await page.goto("/");
  const ribbon = page.getByRole("region", { name: "The path to a majority", exact: true });
  const tiles = ribbon.locator("button[data-race-id]");
  await expect(tiles).toHaveCount(35);
  expect(await tiles.evaluateAll((elements) => elements.map((element) => (element as HTMLElement).dataset.raceId))).toEqual(tipping.models.polling.raceOrder);
  await expect(ribbon.locator(".retained-endcap")).toHaveCount(2);
  await expect(ribbon.locator(".retained-endcap").first()).toContainText("34");
  await expect(ribbon.locator(".retained-endcap").last()).toContainText("31");
  await expect(ribbon.getByText("D51 / R50 · TX", { exact: true })).toBeVisible();
  const texas = ribbon.locator('[data-race-id="2026:TX:senate:regular"]');
  await expect(texas).toHaveAttribute("aria-label", /projected D control threshold, projected R control threshold/);
  await texas.click();
  const selectedFrequencies = ribbon.locator(".selected-threshold");
  await expect(selectedFrequencies).toContainText("At the D control threshold in 12.8% of simulations.");
  await expect(selectedFrequencies).toContainText("At the R control threshold in 10.3% of simulations.");

  await page.getByLabel("Independent alignment").selectOption("independent_opposition_caucuses_D");
  await expect(selectedFrequencies).toContainText("At the D control threshold in 10.3% of simulations.");
  const nebraska = ribbon.locator('[data-race-id="2026:NE:senate:regular"]');
  await nebraska.focus();
  await page.keyboard.press("Enter");
  await expect(nebraska).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByRole("complementary").getByText("Independent", { exact: true })).toBeVisible();
  await expect(selectedFrequencies).toContainText("At the D control threshold in 3.3% of simulations.");
  await expect(selectedFrequencies).toContainText("At the R control threshold in 3.3% of simulations.");
  await page.getByLabel("Independent alignment").selectOption("independent_opposition_caucuses_R");
  await expect(nebraska).toHaveAttribute("aria-label", /R caucus regardless of winner/);
  await expect(selectedFrequencies).toContainText("At the D control threshold in 0.0% of simulations.");
  await expect(selectedFrequencies).toContainText("At the R control threshold in 0.0% of simulations.");
  await expect(page.getByRole("complementary").getByText("Independent", { exact: true })).toBeVisible();
  await expect(ribbon.locator(".ribbon-guarantee")).toContainText("4 independent–Republican contests");

  await page.getByRole("button", { name: "Fundamentals model", exact: true }).click();
  expect(await tiles.evaluateAll((elements) => elements.map((element) => (element as HTMLElement).dataset.raceId))).toEqual(tipping.models.fundamentals.raceOrder);
  await texas.click();
  await expect(selectedFrequencies).toContainText("At the D control threshold in 11.1% of simulations.");
  await expect(selectedFrequencies).toContainText("At the R control threshold in 11.1% of simulations.");
  await ribbon.getByText("How to read the thresholds", { exact: true }).click();
  await expect(ribbon.getByText(/Draws where the caucus falls short still count/)).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy();
  expect(await ribbon.evaluate((element) => Boolean(element.compareDocumentPosition(document.querySelector(".map-card")!) & Node.DOCUMENT_POSITION_FOLLOWING))).toBeTruthy();
  expect(await page.locator(".map-card").evaluate((element) => Boolean(element.compareDocumentPosition(document.querySelector(".control-card")!) & Node.DOCUMENT_POSITION_FOLLOWING))).toBeTruthy();
});

for (const failure of ["missing", "mismatched"] as const) {
  test(`a ${failure} tipping artifact preserves usable map and race details`, async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.route("**/data/tipping-points.json", async (route) => {
      if (failure === "missing") {
        await route.fulfill({ status: 404, body: "Unavailable" });
      } else {
        const response = await route.fetch();
        const tipping = await response.json() as TippingSnapshot;
        tipping.sources.pollingForecastSha256 = "wrong-source";
        await route.fulfill({ response, json: tipping });
      }
    });
    await page.goto("/");
    await expect(page.getByText("The tipping-point snapshot is unavailable. Map and race summaries remain available below.", { exact: true })).toBeVisible();
    await expect(page.locator(".race-table tbody tr")).toHaveCount(35);
    await expect(page.locator("svg [role=button]")).toHaveCount(35);
    await page.getByRole("button", { name: /^Texas, 1 election,/ }).click();
    await expect(page.getByRole("complementary").getByRole("heading", { name: "Texas", exact: true })).toBeVisible();
    expect(errors).toEqual([]);
  });
}
