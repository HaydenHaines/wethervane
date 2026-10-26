# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: house-groups.spec.ts >> failed new selections preserve prior forecast and retries recover; invalid comparisons explain why
- Location: tests/site/house-groups.spec.ts:84:5

# Error details

```
Error: expect(locator).toContainText(expected) failed

Locator: locator('.compare-banner[role=alert]')
Expected substring: "same election year"
Timeout: 5000ms
Error: element(s) not found

Call log:
  - Expect "toContainText" locator('.compare-banner[role=alert]') with timeout 5000ms
  - waiting for locator('.compare-banner[role=alert]')

```

```yaml
- link "Skip to the races":
  - /url: "#races"
- banner:
  - link "Ballotmancy home":
    - /url: /
    - text: ballotmancy
  - navigation "Main navigation": Research preview
- tablist "Chamber":
  - tab "Senate"
  - tab "House" [selected]
- text: Two chambers. Your perspective.
- tabpanel "House":
  - group "House election edition":
    - button "Current · 2026" [disabled]
    - button "Historical elections"
  - paragraph: The 2026 forecast is awaiting publication review. Historical results are available below.
  - main:
    - heading "This House link could not be opened" [level=1]
    - alert: The comparison run belongs to a different historical House year.
    - button "Open current House forecast"
    - button "Explore historical House research"
    - button "Explore the Senate"
- alert
```

# Test source

```ts
  1   | import { expect, test, type Page } from "@playwright/test";
  2   | 
  3   | const run = (year = 2024, model = "history_refreshed") => `house_${model}_v1-${year}-h14-full435-v1`;
  4   | const builder = (page: Page) => page.locator("details.house-configuration");
  5   | const map = (page: Page) => page.getByRole("region", { name: "House map", exact: true });
  6   | const detail = (page: Page) => page.getByRole("complementary", { name: "Selected House race detail", exact: true });
  7   | async function openBuilder(page: Page) {
  8   |   await expect(builder(page)).toBeVisible();
  9   |   if (await builder(page).getAttribute("open") === null) await builder(page).locator(":scope > summary").click();
  10  | }
  11  | 
  12  | test("one-page winner selection preserves model/year/race, maps all435, and restores legacy history", async ({ page }) => {
  13  |   const errors: string[] = []; page.on("pageerror", error => errors.push(error.message));
  14  |   await page.goto("/?chamber=house&run=house_relative_lean_v1-2022-h14&race=2022%3ACA-22%3Ahouse%3Aregular&view=actual");
  15  |   await expect(map(page).locator("svg path[data-forecast=available]")).toHaveCount(373);
  16  |   await expect(builder(page)).not.toHaveAttribute("open", "");
  17  |   await openBuilder(page);
  18  |   await page.getByLabel("Winner outcome set", { exact: true }).selectOption("full435");
  19  |   await expect(page).toHaveURL(new RegExp(`run=${run(2022,"relative_lean")}`));
  20  |   await expect(builder(page)).toHaveAttribute("open", "");
  21  |   await expect(detail(page).getByRole("heading", {name:"CA-22",exact:true})).toBeVisible();
  22  |   await expect(map(page).locator("svg path[data-forecast=available]")).toHaveCount(435);
  23  |   await expect(page.getByLabel("House model", {exact:true})).toHaveValue(run(2022,"relative_lean"));
  24  |   await expect(page.getByRole("group",{name:"Map view",exact:true}).getByRole("button",{name:"Actual winners",exact:true})).toHaveAttribute("aria-pressed","true");
  25  |   await expect(builder(page).locator(".model-slice")).toHaveCount(8);
  26  |   await expect(builder(page).locator(".model-slice.is-unavailable")).toHaveCount(3);
  27  |   await expect(builder(page).locator(".model-slice.is-fixed")).toHaveCount(2);
  28  |   await expect(page.getByRole("region",{name:"House group-win probability ribbon"}).locator(".house-ribbon-cell")).toHaveCount(435);
  29  |   await page.goBack();
  30  |   await expect(page).toHaveURL(/run=house_relative_lean_v1-2022-h14&/);
  31  |   await expect(map(page).locator("svg path[data-forecast=available]")).toHaveCount(373);
  32  |   await page.goForward();
  33  |   await expect(map(page).locator("svg path[data-forecast=available]")).toHaveCount(435);
  34  |   await page.getByLabel("Winner outcome set",{exact:true}).selectOption("legacy");
  35  |   await expect(map(page).locator("svg path[data-forecast=available]")).toHaveCount(373);
  36  |   expect(errors).toEqual([]);
  37  | });
  38  | 
  39  | test("all historical group years use matching maps; NC09 actual remains missing and unscored", async ({ page }) => {
  40  |   await page.goto(`/?chamber=house&run=${run(2018)}&race=2018%3ANC-09%3Ahouse%3Aregular`);
  41  |   await expect(page.locator(".historical-scoreboard")).toContainText("434 / 435");
  42  |   await expect(detail(page)).toContainText("this election was annulled");
  43  |   await expect(map(page).locator("path[data-forecast=available]")).toHaveCount(435);
  44  |   await page.getByRole("group",{name:"Map view",exact:true}).getByRole("button",{name:"Actual winners",exact:true}).click();
  45  |   await expect(map(page).locator('path[data-district-key="NC-09"]')).toHaveAttribute("fill","#d8ddd4");
  46  |   await expect(map(page)).toContainText("no certified winner");
  47  |   await openBuilder(page);
  48  |   const hashes = new Set<string>();
  49  |   for (const [year, seats] of [[2018,53],[2020,53],[2022,52],[2024,52]]) {
  50  |     if(year!==2018) await page.getByRole("group",{name:"Election year",exact:true}).getByRole("button",{name:String(year),exact:true}).click();
  51  |     await expect(page.locator(".district-map-toolbar h3")).toHaveText(`${year} House districts`);
  52  |     await page.getByLabel("Focus on a state",{exact:true}).selectOption("CA");
  53  |     await expect(map(page).locator("path[data-district-key]")).toHaveCount(seats);
  54  |     hashes.add((await page.locator(".district-map-provenance code").textContent())!);
  55  |     if(year!==2018) await expect(page).not.toHaveURL(/race=/);
  56  |   }
  57  |   expect(hashes.size).toBe(4);
  58  | });
  59  | 
  60  | test("group comparisons and exact saved pairs restore together, leaving legacy storage intact", async ({ page }) => {
  61  |   await page.goto(`/?chamber=house&run=${run()}&race=2024%3AAK-00%3Ahouse%3Aregular&view=actual`);
  62  |   await openBuilder(page);
  63  |   await page.evaluate(()=>localStorage.setItem("ballotmancy.house.saved-views.v1","legacy-marker"));
  64  |   await page.getByLabel("Compare with",{exact:true}).selectOption(run(2024,"relative_lean"));
  65  |   await expect(page.getByRole("region",{name:"House model comparison",exact:true})).toContainText("Relative lean");
  66  |   await expect(page.locator(".historical-scoreboard")).toContainText("Compared result:");
  67  |   await builder(page).getByText("Saved group views",{exact:false}).first().click();
  68  |   await page.getByLabel("Name this group view",{exact:true}).fill("Alaska group pair");
  69  |   await page.getByRole("button",{name:"Save group view",exact:true}).click();
  70  |   await expect(builder(page)).toContainText("Saved Alaska group pair in this browser.");
  71  |   await page.getByLabel("House model",{exact:true}).selectOption(run(2024,"relative_context"));
  72  |   await expect(page.getByLabel("House model",{exact:true})).toHaveValue(run(2024,"relative_context"));
  73  |   await page.getByRole("button",{name:"Alaska group pair",exact:true}).click();
  74  |   await expect(page.getByLabel("House model",{exact:true})).toHaveValue(run());
  75  |   await expect(detail(page).getByRole("heading",{name:"AK-00",exact:true})).toBeVisible();
  76  |   await expect(page).toHaveURL(new RegExp(`compare=${run(2024,"relative_lean")}`));
  77  |   await expect(page.getByRole("region",{name:"House model comparison",exact:true})).toBeVisible();
  78  |   expect(await page.evaluate(()=>localStorage.getItem("ballotmancy.house.saved-views.v1"))).toBe("legacy-marker");
  79  |   await page.getByRole("group",{name:"Election year",exact:true}).getByRole("button",{name:"2022",exact:true}).click();
  80  |   await expect(page).toHaveURL(new RegExp(`run=${run(2022)}&view=actual$`));
  81  |   await expect(page.getByRole("region",{name:"House model comparison",exact:true})).toHaveCount(0);
  82  | });
  83  | 
  84  | test("failed new selections preserve prior forecast and retries recover; invalid comparisons explain why", async ({ page }) => {
  85  |   let fail=true;
  86  |   await page.route("**/api/chambers/house/groups?*", route=>fail?route.fulfill({status:503,body:"unavailable"}):route.continue());
  87  |   await page.goto("/?chamber=house&edition=historical"); await openBuilder(page);
  88  |   await page.getByLabel("Winner outcome set",{exact:true}).selectOption("full435");
  89  |   await expect(page.locator(".compare-banner[role=alert]")).toContainText("Unable to change House models");
  90  |   await expect(map(page).locator("path[data-forecast=available]")).toHaveCount(375);
  91  |   await expect(page).not.toHaveURL(/full435/);
  92  |   fail=false; await page.getByRole("button",{name:"Try selection again",exact:true}).click();
  93  |   await expect(map(page).locator("path[data-forecast=available]")).toHaveCount(435);
  94  |   await page.goto(`/?chamber=house&run=${run()}&compare=${run(2018,"relative_lean")}`);
> 95  |   await expect(page.locator(".compare-banner[role=alert]")).toContainText("same election year");
      |                                                             ^ Error: expect(locator).toContainText(expected) failed
  96  |   await expect(map(page).locator("path[data-forecast=available]")).toHaveCount(435);
  97  |   await page.getByRole("button",{name:"Remove comparison",exact:true}).click();
  98  |   await expect(page.locator(".compare-banner[role=alert]")).toHaveCount(0);
  99  | });
  100 | 
  101 | test("same-party and Other outcomes stay grouped; responsive map/ribbon selection and ten-row list work", async ({ page }) => {
  102 |   const errors:string[]=[];page.on("pageerror",error=>errors.push(error.message));
  103 |   await page.goto(`/?chamber=house&run=${run()}&race=2024%3ALA-04%3Ahouse%3Aregular`);
  104 |   await expect(detail(page)).toContainText(/Only (this|one) group/);
  105 |   await expect(detail(page)).toContainText("does not assign individual chances");
  106 |   await expect(page.locator("#house-race-list > button")).toHaveCount(10);
  107 |   await page.getByRole("button",{name:/See all 435 races/}).click();
  108 |   await expect(page.locator("#house-race-list > button")).toHaveCount(435);
  109 |   await page.getByLabel("Focus on a state",{exact:true}).selectOption("AK");
  110 |   await map(page).locator('path[data-district-key="AK-00"]').focus();await page.keyboard.press("Enter");
  111 |   await expect(detail(page).getByRole("heading",{name:"AK-00",exact:true})).toBeVisible();
  112 |   await expect(detail(page)).toContainText("Other-group probability");
  113 |   const ribbon=page.getByRole("region",{name:"House group-win probability ribbon"});
  114 |   const first=ribbon.locator("button.house-ribbon-cell").first();await first.focus();await page.keyboard.press("End");await page.keyboard.press("Enter");
  115 |   await expect(ribbon.locator('button.house-ribbon-cell[aria-pressed="true"]')).toHaveCount(1);
  116 |   await page.getByText("Group-win score by district group",{exact:true}).click();
  117 |   expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
  118 |   expect(errors).toEqual([]);
  119 | });
  120 | 
  121 | test("failed Back navigation preserves the displayed district and a failed saved-pair restore is atomic", async ({ page }) => {
  122 |   await page.goto(`/?chamber=house&run=${run(2018)}&race=2018%3ANC-09%3Ahouse%3Aregular`);
  123 |   await openBuilder(page);
  124 |   await page.getByRole("group",{name:"Election year",exact:true}).getByRole("button",{name:"2024",exact:true}).click();
  125 |   await expect(page.locator(".district-map-toolbar h3")).toHaveText("2024 House districts");
  126 |   const selected = await detail(page).locator("h2").innerText();
  127 |   await page.route(`**/api/chambers/house/groups?run=${run(2018)}`,route=>route.fulfill({status:503,body:"unavailable"}));
  128 |   await page.goBack();
  129 |   await expect(page.locator(".compare-banner[role=alert]")).toContainText("Unable to change House group forecasts");
  130 |   await expect(page.locator(".district-map-toolbar h3")).toHaveText("2024 House districts");
  131 |   await expect(detail(page).locator("h2")).toHaveText(selected);
  132 |   await page.unrouteAll();
  133 |   await page.goto(`/?chamber=house&run=${run()}&race=2024%3AAK-00%3Ahouse%3Aregular&compare=${run(2024,"relative_lean")}`);
  134 |   await openBuilder(page);
  135 |   await expect(page.getByRole("region",{name:"House model comparison",exact:true})).toBeVisible();
  136 |   await builder(page).getByText("Saved group views",{exact:false}).first().click();
  137 |   await page.getByLabel("Name this group view",{exact:true}).fill("Exact pair");
  138 |   await page.getByRole("button",{name:"Save group view",exact:true}).click();
  139 |   await expect(builder(page)).toContainText("Saved Exact pair in this browser.");
  140 |   await page.getByLabel("Compare with",{exact:true}).selectOption("");
  141 |   await page.getByLabel("House model",{exact:true}).selectOption(run(2024,"relative_context"));
  142 |   await expect(page.getByLabel("House model",{exact:true})).toHaveValue(run(2024,"relative_context"));
  143 |   await page.route(`**/api/chambers/house/groups?run=${run(2024,"relative_lean")}`,route=>route.fulfill({status:503,body:"unavailable"}));
  144 |   await page.getByRole("button",{name:"Exact pair",exact:true}).click();
  145 |   await expect(page.locator(".compare-banner[role=alert]")).toContainText("Unable to change House group forecasts");
  146 |   await expect(page.getByLabel("House model",{exact:true})).toHaveValue(run(2024,"relative_context"));
  147 |   await expect(page).toHaveURL(new RegExp(`run=${run(2024,"relative_context")}`));
  148 |   await expect(page.getByRole("region",{name:"House model comparison",exact:true})).toHaveCount(0);
  149 | });
  150 | 
```