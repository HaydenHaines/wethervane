import { expect, test } from "@playwright/test";
import type { SiteForecastSnapshot } from "../../lib/site-types";

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
