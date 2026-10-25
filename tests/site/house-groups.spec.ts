import { expect, test, type Page } from "@playwright/test";

const run = (year = 2024, model = "history_refreshed") => `house_${model}_v1-${year}-h14-full435-v1`;
const builder = (page: Page) => page.locator("details.house-configuration");
const map = (page: Page) => page.getByRole("region", { name: "House map", exact: true });
const detail = (page: Page) => page.getByRole("complementary", { name: "Selected House race detail", exact: true });
async function openBuilder(page: Page) {
  await expect(builder(page)).toBeVisible();
  if (await builder(page).getAttribute("open") === null) await builder(page).locator(":scope > summary").click();
}

test("one-page winner selection preserves model/year/race, maps all435, and restores legacy history", async ({ page }) => {
  const errors: string[] = []; page.on("pageerror", error => errors.push(error.message));
  await page.goto("/?chamber=house&run=house_relative_lean_v1-2022-h14&race=2022%3ACA-22%3Ahouse%3Aregular&view=actual");
  await expect(map(page).locator("svg path[data-forecast=available]")).toHaveCount(373);
  await expect(builder(page)).not.toHaveAttribute("open", "");
  await openBuilder(page);
  await page.getByLabel("Winner outcome set", { exact: true }).selectOption("full435");
  await expect(page).toHaveURL(new RegExp(`run=${run(2022,"relative_lean")}`));
  await expect(builder(page)).toHaveAttribute("open", "");
  await expect(detail(page).getByRole("heading", {name:"CA-22",exact:true})).toBeVisible();
  await expect(map(page).locator("svg path[data-forecast=available]")).toHaveCount(435);
  await expect(page.getByLabel("House model", {exact:true})).toHaveValue(run(2022,"relative_lean"));
  await expect(page.getByRole("group",{name:"Map view",exact:true}).getByRole("button",{name:"Actual winners",exact:true})).toHaveAttribute("aria-pressed","true");
  await expect(builder(page).locator(".model-slice")).toHaveCount(8);
  await expect(builder(page).locator(".model-slice.is-unavailable")).toHaveCount(3);
  await expect(builder(page).locator(".model-slice.is-fixed")).toHaveCount(2);
  await expect(page.getByRole("region",{name:"House group-win probability ribbon"}).locator(".house-ribbon-cell")).toHaveCount(435);
  await page.goBack();
  await expect(page).toHaveURL(/run=house_relative_lean_v1-2022-h14&/);
  await expect(map(page).locator("svg path[data-forecast=available]")).toHaveCount(373);
  await page.goForward();
  await expect(map(page).locator("svg path[data-forecast=available]")).toHaveCount(435);
  await page.getByLabel("Winner outcome set",{exact:true}).selectOption("legacy");
  await expect(map(page).locator("svg path[data-forecast=available]")).toHaveCount(373);
  expect(errors).toEqual([]);
});

test("all historical group years use matching maps; NC09 actual remains missing and unscored", async ({ page }) => {
  await page.goto(`/?chamber=house&run=${run(2018)}&race=2018%3ANC-09%3Ahouse%3Aregular`);
  await expect(page.locator(".historical-scoreboard")).toContainText("434 / 435");
  await expect(detail(page)).toContainText("this election was annulled");
  await expect(map(page).locator("path[data-forecast=available]")).toHaveCount(435);
  await page.getByRole("group",{name:"Map view",exact:true}).getByRole("button",{name:"Actual winners",exact:true}).click();
  await expect(map(page).locator('path[data-district-key="NC-09"]')).toHaveAttribute("fill","#d8ddd4");
  await expect(map(page)).toContainText("no certified winner");
  await openBuilder(page);
  const hashes = new Set<string>();
  for (const [year, seats] of [[2018,53],[2020,53],[2022,52],[2024,52]]) {
    if(year!==2018) await page.getByRole("group",{name:"Election year",exact:true}).getByRole("button",{name:String(year),exact:true}).click();
    await expect(page.locator(".district-map-toolbar h3")).toHaveText(`${year} House districts`);
    await page.getByLabel("Focus on a state",{exact:true}).selectOption("CA");
    await expect(map(page).locator("path[data-district-key]")).toHaveCount(seats);
    hashes.add((await page.locator(".district-map-provenance code").textContent())!);
    if(year!==2018) await expect(page).not.toHaveURL(/race=/);
  }
  expect(hashes.size).toBe(4);
});

test("group comparisons and exact saved pairs restore together, leaving legacy storage intact", async ({ page }) => {
  await page.goto(`/?chamber=house&run=${run()}&race=2024%3AAK-00%3Ahouse%3Aregular&view=actual`);
  await openBuilder(page);
  await page.evaluate(()=>localStorage.setItem("ballotmancy.house.saved-views.v1","legacy-marker"));
  await page.getByLabel("Compare with",{exact:true}).selectOption(run(2024,"relative_lean"));
  await expect(page.getByRole("region",{name:"House model comparison",exact:true})).toContainText("Relative lean");
  await expect(page.locator(".historical-scoreboard")).toContainText("Compared result:");
  await builder(page).getByText("Saved group views",{exact:false}).first().click();
  await page.getByLabel("Name this group view",{exact:true}).fill("Alaska group pair");
  await page.getByRole("button",{name:"Save group view",exact:true}).click();
  await expect(builder(page)).toContainText("Saved Alaska group pair in this browser.");
  await page.getByLabel("House model",{exact:true}).selectOption(run(2024,"relative_context"));
  await expect(page.getByLabel("House model",{exact:true})).toHaveValue(run(2024,"relative_context"));
  await page.getByRole("button",{name:"Alaska group pair",exact:true}).click();
  await expect(page.getByLabel("House model",{exact:true})).toHaveValue(run());
  await expect(detail(page).getByRole("heading",{name:"AK-00",exact:true})).toBeVisible();
  await expect(page).toHaveURL(new RegExp(`compare=${run(2024,"relative_lean")}`));
  await expect(page.getByRole("region",{name:"House model comparison",exact:true})).toBeVisible();
  expect(await page.evaluate(()=>localStorage.getItem("ballotmancy.house.saved-views.v1"))).toBe("legacy-marker");
  await page.getByRole("group",{name:"Election year",exact:true}).getByRole("button",{name:"2022",exact:true}).click();
  await expect(page).toHaveURL(new RegExp(`run=${run(2022)}&view=actual$`));
  await expect(page.getByRole("region",{name:"House model comparison",exact:true})).toHaveCount(0);
});

test("failed new selections preserve prior forecast and retries recover; invalid comparisons explain why", async ({ page }) => {
  let fail=true;
  await page.route("**/api/chambers/house/groups?*", route=>fail?route.fulfill({status:503,body:"unavailable"}):route.continue());
  await page.goto("/?chamber=house"); await openBuilder(page);
  await page.getByLabel("Winner outcome set",{exact:true}).selectOption("full435");
  await expect(page.locator(".compare-banner[role=alert]")).toContainText("Unable to change House models");
  await expect(map(page).locator("path[data-forecast=available]")).toHaveCount(375);
  await expect(page).not.toHaveURL(/full435/);
  fail=false; await page.getByRole("button",{name:"Try selection again",exact:true}).click();
  await expect(map(page).locator("path[data-forecast=available]")).toHaveCount(435);
  await page.goto(`/?chamber=house&run=${run()}&compare=${run(2018,"relative_lean")}`);
  await expect(page.locator(".compare-banner[role=alert]")).toContainText("same election year");
  await expect(map(page).locator("path[data-forecast=available]")).toHaveCount(435);
  await page.getByRole("button",{name:"Remove comparison",exact:true}).click();
  await expect(page.locator(".compare-banner[role=alert]")).toHaveCount(0);
});

test("same-party and Other outcomes stay grouped; responsive map/ribbon selection and ten-row list work", async ({ page }) => {
  const errors:string[]=[];page.on("pageerror",error=>errors.push(error.message));
  await page.goto(`/?chamber=house&run=${run()}&race=2024%3ALA-04%3Ahouse%3Aregular`);
  await expect(detail(page)).toContainText(/Only (this|one) group/);
  await expect(detail(page)).toContainText("does not assign individual chances");
  await expect(page.locator("#house-race-list > button")).toHaveCount(10);
  await page.getByRole("button",{name:/See all 435 races/}).click();
  await expect(page.locator("#house-race-list > button")).toHaveCount(435);
  await page.getByLabel("Focus on a state",{exact:true}).selectOption("AK");
  await map(page).locator('path[data-district-key="AK-00"]').focus();await page.keyboard.press("Enter");
  await expect(detail(page).getByRole("heading",{name:"AK-00",exact:true})).toBeVisible();
  await expect(detail(page)).toContainText("Other-group probability");
  const ribbon=page.getByRole("region",{name:"House group-win probability ribbon"});
  const first=ribbon.locator("button.house-ribbon-cell").first();await first.focus();await page.keyboard.press("End");await page.keyboard.press("Enter");
  await expect(ribbon.locator('button.house-ribbon-cell[aria-pressed="true"]')).toHaveCount(1);
  await page.getByText("Group-win score by district group",{exact:true}).click();
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
  expect(errors).toEqual([]);
});

test("failed Back navigation preserves the displayed district and a failed saved-pair restore is atomic", async ({ page }) => {
  await page.goto(`/?chamber=house&run=${run(2018)}&race=2018%3ANC-09%3Ahouse%3Aregular`);
  await openBuilder(page);
  await page.getByRole("group",{name:"Election year",exact:true}).getByRole("button",{name:"2024",exact:true}).click();
  await expect(page.locator(".district-map-toolbar h3")).toHaveText("2024 House districts");
  const selected = await detail(page).locator("h2").innerText();
  await page.route(`**/api/chambers/house/groups?run=${run(2018)}`,route=>route.fulfill({status:503,body:"unavailable"}));
  await page.goBack();
  await expect(page.locator(".compare-banner[role=alert]")).toContainText("Unable to change House group forecasts");
  await expect(page.locator(".district-map-toolbar h3")).toHaveText("2024 House districts");
  await expect(detail(page).locator("h2")).toHaveText(selected);
  await page.unrouteAll();
  await page.goto(`/?chamber=house&run=${run()}&race=2024%3AAK-00%3Ahouse%3Aregular&compare=${run(2024,"relative_lean")}`);
  await openBuilder(page);
  await expect(page.getByRole("region",{name:"House model comparison",exact:true})).toBeVisible();
  await builder(page).getByText("Saved group views",{exact:false}).first().click();
  await page.getByLabel("Name this group view",{exact:true}).fill("Exact pair");
  await page.getByRole("button",{name:"Save group view",exact:true}).click();
  await expect(builder(page)).toContainText("Saved Exact pair in this browser.");
  await page.getByLabel("Compare with",{exact:true}).selectOption("");
  await page.getByLabel("House model",{exact:true}).selectOption(run(2024,"relative_context"));
  await expect(page.getByLabel("House model",{exact:true})).toHaveValue(run(2024,"relative_context"));
  await page.route(`**/api/chambers/house/groups?run=${run(2024,"relative_lean")}`,route=>route.fulfill({status:503,body:"unavailable"}));
  await page.getByRole("button",{name:"Exact pair",exact:true}).click();
  await expect(page.locator(".compare-banner[role=alert]")).toContainText("Unable to change House group forecasts");
  await expect(page.getByLabel("House model",{exact:true})).toHaveValue(run(2024,"relative_context"));
  await expect(page).toHaveURL(new RegExp(`run=${run(2024,"relative_context")}`));
  await expect(page.getByRole("region",{name:"House model comparison",exact:true})).toHaveCount(0);
});
