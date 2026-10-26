import { expect, test, type Page } from "@playwright/test";
import type { SiteForecastSnapshot } from "../../lib/site-types";
import type { TippingSnapshot } from "../../lib/tipping-types";

async function expandControls(page: Page) {
  const controls = page.locator("details.stack-builder");
  await expect(controls).toBeVisible();
  if (await controls.getAttribute("open") === null) await controls.locator(":scope > summary").click();
  await expect(controls).toHaveAttribute("open", "");
}

const percent = (value: number) => `${(value * 100).toFixed(1)}%`;
const shareLabel = (outcome: { name: string; share: { mean: number; quantiles: { p05: number; p95: number } } }) =>
  `${outcome.name}: mean ${percent(outcome.share.mean)}, Modeled 90% range ${percent(outcome.share.quantiles.p05)} to ${percent(outcome.share.quantiles.p95)}`;

test("the refreshed current edition matches its frozen polling inputs and independent-candidate outcomes", async ({ request }) => {
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

  // Keep the separately stored October 20 snapshot's own reproducibility pins.
  expect(forecast.models.polling.controls.campaign_pledge_unaligned.unresolvedOrganizingProbability).toBe(0.08305);
  expect(forecast.models.polling.controls.independent_opposition_caucuses_D.democraticMajorityProbability).toBe(0.541725);
  expect(forecast.models.fundamentals.controls.campaign_pledge_unaligned.republicanMajorityProbability).toBe(0.89352);

  const activeResponse = await request.get("/data/explorer/current-polling.json");
  expect(activeResponse.ok()).toBeTruthy();
  const active = await activeResponse.json();
  expect(active.run.dateId).toBe("2026-10-26");
  expect(active.run.cutoffAt).toBe("2026-10-26T15:03:57.212405+00:00");
  expect(active.run.inputCutoffs.find((item: { label: string }) => item.label === "Polling inputs")?.at)
    .toBe(active.run.cutoffAt);
  const defaultControl = active.scenarios.find((item: { id: string }) => item.id === active.defaultScenarioId).control;
  expect(defaultControl.unresolvedOrganizingProbability).toBeCloseTo(0.03335, 10);
  const nebraska = active.races.find((race: { id: string }) => race.id === "2026:NE:senate:regular");
  expect(nebraska.pollCount).toBe(5);
  expect(nebraska.outcomes.find((outcome: { name: string }) => outcome.name === "Dan Osborn").winProbability)
    .toBeCloseTo(0.155625, 10);
});

test("model and caucus controls preserve independent candidate identity and share intervals", async ({ page, request }) => {
  const polling = await (await request.get("/data/explorer/current-polling.json")).json();
  const fundamentals = await (await request.get("/data/explorer/current-fundamentals.json")).json();
  const pollingUnaligned = polling.scenarios.find((item: { id: string }) => item.id === "campaign_pledge_unaligned");
  const pollingAlignedD = polling.scenarios.find((item: { id: string }) => item.id === "independent_opposition_caucuses_D");
  const fundamentalsAlignedD = fundamentals.scenarios.find((item: { id: string }) => item.id === "independent_opposition_caucuses_D");
  const fundamentalsAlignedR = fundamentals.scenarios.find((item: { id: string }) => item.id === "independent_opposition_caucuses_R");
  const nebraskaPolling = polling.races.find((race: { id: string }) => race.id === "2026:NE:senate:regular");
  const nebraskaFundamentals = fundamentals.races.find((race: { id: string }) => race.id === "2026:NE:senate:regular");
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await expandControls(page);
  const chamber = page.getByRole("region", { name: "Who could control the chamber?" });
  const detail = page.getByRole("complementary", { name: "Selected race detail" });
  await expect(chamber.getByText(percent(pollingUnaligned.control.unresolvedOrganizingProbability), { exact: true })).toBeVisible();
  await expect(page.locator(".race-table tbody tr")).toHaveCount(10);
  await expect(page.locator("svg [role=button]")).toHaveCount(35);

  await page.getByRole("button", { name: /^Nebraska, 1 election,/ }).click();
  await expect(detail.getByRole("heading", { name: "Nebraska", exact: true })).toBeVisible();
  await expect(detail.getByText("Independent", { exact: true })).toBeVisible();
  const osborn = nebraskaPolling.outcomes.find((outcome: { name: string }) => outcome.name === "Dan Osborn");
  const ricketts = nebraskaPolling.outcomes.find((outcome: { name: string }) => outcome.name === "Pete Ricketts");
  await expect(detail.getByText(percent(osborn.winProbability), { exact: true })).toBeVisible();
  await expect(detail.locator(`[aria-label="${shareLabel(osborn)}"]`)).toBeVisible();
  await expect(detail.locator(`[aria-label="${shareLabel(ricketts)}"]`)).toBeVisible();

  await page.getByLabel("Who joins each party?").selectOption("independent_opposition_caucuses_D");
  await expect(chamber.getByText(percent(pollingAlignedD.control.democraticMajorityProbability), { exact: true })).toBeVisible();
  await expect(chamber.getByText(percent(pollingAlignedD.control.unresolvedOrganizingProbability), { exact: true })).toBeVisible();
  await expect(detail.getByText("Independent", { exact: true })).toBeVisible();

  await page.getByRole("checkbox", { name: "Include polling evidence", exact: true }).uncheck();
  await expect(page.locator(".stack-builder")).toHaveAttribute("aria-busy", "false");
  await expect(detail.getByRole("heading", { name: "Nebraska", exact: true })).toBeVisible();
  await expect(chamber.getByText(percent(fundamentalsAlignedD.control.democraticMajorityProbability), { exact: true })).toBeVisible();
  const osbornFundamentals = nebraskaFundamentals.outcomes.find((outcome: { name: string }) => outcome.name === "Dan Osborn");
  const rickettsFundamentals = nebraskaFundamentals.outcomes.find((outcome: { name: string }) => outcome.name === "Pete Ricketts");
  await expect(detail.getByText(percent(osbornFundamentals.winProbability), { exact: true })).toBeVisible();
  await expect(detail.locator(`[aria-label="${shareLabel(osbornFundamentals)}"]`)).toBeVisible();
  await expect(detail.locator(`[aria-label="${shareLabel(rickettsFundamentals)}"]`)).toBeVisible();
  await page.getByLabel("Who joins each party?").selectOption("independent_opposition_caucuses_R");
  await expect(chamber.getByText(percent(fundamentalsAlignedR.control.republicanMajorityProbability), { exact: true })).toBeVisible();
  await page.getByLabel("Who joins each party?").selectOption("campaign_pledge_unaligned");
  const fundamentalsUnaligned = fundamentals.scenarios.find((item: { id: string }) => item.id === "campaign_pledge_unaligned");
  await expect(chamber.getByText(percent(fundamentalsUnaligned.control.unresolvedOrganizingProbability), { exact: true })).toBeVisible();
  expect(errors).toEqual([]);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy();
});

test("race search, special elections, and map keyboard selection work together", async ({ page }) => {
  await page.goto("/");
  await expandControls(page);
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

test("the race list remains usable when map geography fails", async ({ page }) => {
  await page.route("**/states-us.geojson", (route) => route.abort());
  await page.goto("/");
  await expandControls(page);
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
  const polling = (await (await request.get("/data/explorer/current-polling.json")).json()).tipping as TippingSnapshot["models"]["polling"];
  const fundamentals = (await (await request.get("/data/explorer/current-fundamentals.json")).json()).tipping as TippingSnapshot["models"]["fundamentals"];
  const txId = "2026:TX:senate:regular";
  const neId = "2026:NE:senate:regular";
  const thresholdText = (caucus: "D" | "R", probability: number) =>
    `At the ${caucus} control threshold in ${percent(probability)} of simulations.`;
  await page.goto("/");
  await expandControls(page);
  const ribbon = page.getByRole("region", { name: "The path to a majority", exact: true });
  const tiles = ribbon.locator("button[data-race-id]");
  await expect(tiles).toHaveCount(35);
  expect(await tiles.evaluateAll((elements) => elements.map((element) => (element as HTMLElement).dataset.raceId))).toEqual(polling.raceOrder);
  await expect(ribbon.locator(".retained-endcap")).toHaveCount(2);
  await expect(ribbon.locator(".retained-endcap").first()).toContainText("34");
  await expect(ribbon.locator(".retained-endcap").last()).toContainText("31");
  await expect(ribbon.getByText("D51 / R50 · TX", { exact: true })).toBeVisible();
  const texas = ribbon.locator('[data-race-id="2026:TX:senate:regular"]');
  await expect(texas).toHaveAttribute("aria-label", /projected D control threshold, projected R control threshold/);
  await texas.click();
  const selectedFrequencies = ribbon.locator(".selected-threshold");
  await expect(selectedFrequencies).toContainText(thresholdText("D", polling.scenarios.campaign_pledge_unaligned.D.frequencyByRace[txId]));
  await expect(selectedFrequencies).toContainText(thresholdText("R", polling.scenarios.campaign_pledge_unaligned.R.frequencyByRace[txId]));

  await page.getByLabel("Who joins each party?").selectOption("independent_opposition_caucuses_D");
  await expect(selectedFrequencies).toContainText(thresholdText("D", polling.scenarios.independent_opposition_caucuses_D.D.frequencyByRace[txId]));
  const nebraska = ribbon.locator('[data-race-id="2026:NE:senate:regular"]');
  await nebraska.focus();
  await page.keyboard.press("Enter");
  await expect(nebraska).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByRole("complementary").getByText("Independent", { exact: true })).toBeVisible();
  await expect(selectedFrequencies).toContainText(thresholdText("D", polling.scenarios.independent_opposition_caucuses_D.D.frequencyByRace[neId]));
  await expect(selectedFrequencies).toContainText(thresholdText("R", polling.scenarios.independent_opposition_caucuses_D.R.frequencyByRace[neId]));
  await page.getByLabel("Who joins each party?").selectOption("independent_opposition_caucuses_R");
  await expect(nebraska).toHaveAttribute("aria-label", /R caucus regardless of winner/);
  await expect(selectedFrequencies).toContainText("At the D control threshold in 0.0% of simulations.");
  await expect(selectedFrequencies).toContainText("At the R control threshold in 0.0% of simulations.");
  await expect(page.getByRole("complementary").getByText("Independent", { exact: true })).toBeVisible();
  await expect(ribbon.locator(".ribbon-guarantee")).toContainText("4 contests count toward the R caucus");

  await page.getByRole("checkbox", { name: "Include polling evidence", exact: true }).uncheck();
  await expect(page.locator(".stack-builder")).toHaveAttribute("aria-busy", "false");
  expect(await tiles.evaluateAll((elements) => elements.map((element) => (element as HTMLElement).dataset.raceId))).toEqual(fundamentals.raceOrder);
  await texas.click();
  await expect(selectedFrequencies).toContainText(thresholdText("D", fundamentals.scenarios.independent_opposition_caucuses_R.D.frequencyByRace[txId]));
  await expect(selectedFrequencies).toContainText(thresholdText("R", fundamentals.scenarios.independent_opposition_caucuses_R.R.frequencyByRace[txId]));
  await ribbon.getByText("How to read the thresholds", { exact: true }).click();
  await expect(ribbon.getByText(/Simulations where the party group falls short still count/)).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy();
  expect(await ribbon.evaluate((element) => Boolean(element.compareDocumentPosition(document.querySelector(".map-card")!) & Node.DOCUMENT_POSITION_FOLLOWING))).toBeTruthy();
  expect(await page.locator(".map-card").evaluate((element) => Boolean(element.compareDocumentPosition(document.querySelector(".control-card")!) & Node.DOCUMENT_POSITION_FOLLOWING))).toBeTruthy();
});

// Multi-contest historical states and atomic bundle failures are covered in explorer.spec.ts.
