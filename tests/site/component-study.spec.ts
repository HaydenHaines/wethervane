import { expect, test, type Page } from "@playwright/test";

const scoreboard = (page: Page) => page.getByRole("region", { name: "Historical study scorecard", exact: true });
const savedStudies = (page: Page) => page.getByRole("group", { name: "Saved study results", exact: true });
const detail = (page: Page) => page.getByRole("complementary", { name: "Selected study race detail", exact: true });
async function openStudyControls(page: Page) {
  const controls = page.locator("details.stack-builder");
  await expect(controls).toBeVisible();
  if (await controls.getAttribute("open") === null) await controls.locator(":scope > summary").click();
  await expect(controls).toHaveAttribute("open", "");
}

test("the collapsed full forecast opens a scoped study with honest model roles and race coverage", async ({ page }) => {
  await page.goto("/?run=current-polling");
  const controls = page.locator("details.stack-builder");
  await expect(controls).not.toHaveAttribute("open", "");
  await controls.locator(":scope > summary").click();
  await page.getByRole("button", { name: "Explore the 2018 component study", exact: true }).click();
  await expect(page).toHaveURL(/study=ordinary-2018-fp/);
  await expect(page.locator("details.stack-builder")).not.toHaveAttribute("open", "");
  await openStudyControls(page);
  await expect(page.locator(".study-scope")).toContainText("30 regular Democratic–Republican races");
  await expect(scoreboard(page)).toContainText("Scored on 30 of 30 study races");
  await expect(scoreboard(page).getByRole("heading", { name: "Two-party favorites right", exact: true })).toBeVisible();
  await expect(scoreboard(page).locator(".scoreboard-value").first()).toHaveText(/27\s*\/\s*30/);
  await expect(scoreboard(page).locator(".scoreboard-value").nth(1)).toHaveText("0.145");
  await expect(page.locator(".model-slice")).toHaveCount(8);
  await expect(page.locator(".slice-control")).toHaveCount(7);
  await expect(page.locator(".model-slice.is-unavailable")).toHaveCount(5);
  for (const checkbox of await page.locator(".model-slice.is-unavailable").getByRole("checkbox").all()) {
    await expect(checkbox).toBeDisabled();
    await expect(checkbox).not.toBeChecked();
  }
  const ribbon = page.getByRole("region", { name: "Study races ordered by Democratic win probability", exact: true });
  await expect(ribbon.getByRole("button")).toHaveCount(30);
  await expect(page.locator(".threshold-combined, .threshold-marker")).toHaveCount(0);
  await expect(page.locator(".map-svg [role=button]")).toHaveCount(30);
  const excluded = page.locator('svg path[aria-label="California, not included in this study"]');
  await expect(excluded).toBeVisible();
  await expect(excluded).not.toHaveAttribute("role", "button");
  await expect(page.locator(".race-table tbody tr")).toHaveCount(10);
  await page.getByRole("button", { name: "See all 30 races", exact: true }).click();
  await expect(page.locator(".race-table tbody tr")).toHaveCount(30);
  await page.getByRole("button", { name: "Show fewer races", exact: true }).click();
  await expect(page.locator(".race-table tbody tr")).toHaveCount(10);
  await page.getByRole("button", { name: "Return to the full Senate forecast", exact: true }).click();
  await expect(page).toHaveURL(/run=current-polling/);
  await expect(page.getByRole("region", { name: "The path to a majority", exact: true })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBeTruthy();
});

test("named saved studies switch complete bindings and candidate removal never relabels the old result", async ({ page }) => {
  await page.goto("/?study=ordinary-2018-f&race=2018%3AAZ%3Asenate%3Aregular");
  await openStudyControls(page);
  await expect(scoreboard(page).locator(".scoreboard-value").first()).toHaveText(/24\s*\/\s*30/);
  await expect(page.getByRole("checkbox", { name: "Include polling evidence", exact: true })).not.toBeChecked();
  await savedStudies(page).getByRole("button", { name: "With polls", exact: true }).click();
  await expect(page).toHaveURL(/study=ordinary-2018-fp/);
  await expect(scoreboard(page).locator(".scoreboard-value").first()).toHaveText(/27\s*\/\s*30/);
  await expect(page.getByRole("checkbox", { name: "Include polling evidence", exact: true })).toBeChecked();
  const oldScore = await scoreboard(page).innerText();
  const oldProbabilities = await detail(page).locator(".candidate-prob").allTextContents();
  await page.getByLabel("Incumbency adjustment", { exact: true }).selectOption("off");
  await expect(page.locator(".composition-state")).toContainText("Requires a new calculation");
  await expect(page.locator(".composition-state")).toContainText("Still displaying");
  await expect(page.locator(".composition-state")).toContainText("No saved result exists for this setup");
  await expect(page.locator(".composition-state")).toContainText("This site cannot run that calculation");
  await expect(page.locator(".composition-state")).toContainText("No calculation has been scheduled");
  await expect(page).toHaveURL(/study=ordinary-2018-fp/);
  await expect(scoreboard(page)).toHaveText(oldScore, { useInnerText: true });
  await expect(detail(page).locator(".candidate-prob")).toHaveText(oldProbabilities);
  await page.locator("details.composition-saved > summary").click();
  await page.getByLabel("Name this study view", { exact: true }).fill("Uncomputed choice");
  await expect(page.getByRole("button", { name: "Save view", exact: true })).toBeDisabled();
  await page.getByRole("button", { name: "Reset to displayed result", exact: true }).click();
  await expect(page.getByLabel("Incumbency adjustment", { exact: true })).toHaveValue("signed_incumbency");
  await expect(page.locator(".composition-state")).toContainText("Saved result ready");
});

test("race detail preserves the supplied 80% and 95% ranges and accessible map selection", async ({ page }) => {
  await page.goto("/?study=ordinary-2018-f&race=2018%3AAZ%3Asenate%3Aregular");
  await expect(detail(page).getByRole("heading", { name: "Arizona", exact: true })).toBeVisible();
  await expect(detail(page).getByRole("img", { name: /modeled 80% range .*modeled 95% range/ })).toHaveCount(2);
  await expect(detail(page)).toContainText("Modeled 80% range:");
  await expect(detail(page)).toContainText("Modeled 95% range:");
  await expect(detail(page)).not.toContainText("Modeled 90% range");
  await expect(detail(page).getByRole("heading", { name: "Share of Democratic–Republican votes", exact: true })).toBeVisible();
  await expect(detail(page)).toContainText("Other candidates' votes are excluded from this comparison.");
  const texas = page.getByRole("button", { name: /^Texas, 1 election,/ });
  await texas.focus();
  await page.keyboard.press("Enter");
  await expect(detail(page).getByRole("heading", { name: "Texas", exact: true })).toBeVisible();
  await expect(page).toHaveURL(/race=2018%3ATX%3Asenate%3Aregular/);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBeTruthy();
});

test("study comparison, sharing, reload, Back and local saved views restore identified results", async ({ page }) => {
  await page.goto("/?study=ordinary-2018-f&race=2018%3AAZ%3Asenate%3Aregular");
  await openStudyControls(page);
  await expect(scoreboard(page)).toBeVisible();
  await savedStudies(page).getByRole("button", { name: "With polls", exact: true }).click();
  await expect(page).toHaveURL(/study=ordinary-2018-fp/);
  await page.goBack();
  await expect(page).toHaveURL(/study=ordinary-2018-f&/);
  await expect(scoreboard(page).locator(".scoreboard-value").first()).toHaveText(/24\s*\/\s*30/);
  await page.getByRole("checkbox", { name: "Compare with the other study result", exact: true }).check();
  await expect(scoreboard(page)).toContainText("on the same races, two-party outcomes and forecast date");
  await expect(scoreboard(page).locator(".scoreboard-comparison").first()).toContainText("0.090");
  await page.getByRole("button", { name: "Share view ↗", exact: true }).click();
  await expect(page.getByLabel("Shareable displayed result", { exact: true })).toHaveValue(page.url());
  await page.reload();
  await openStudyControls(page);
  await expect(page.getByRole("checkbox", { name: "Compare with the other study result", exact: true })).toBeChecked();
  await expect(detail(page).getByRole("heading", { name: "Arizona", exact: true })).toBeVisible();
  await page.locator("details.composition-saved > summary").click();
  await page.getByLabel("Name this study view", { exact: true }).fill("Arizona reference");
  await page.getByRole("button", { name: "Save view", exact: true }).click();
  const stored = await page.evaluate(() => JSON.parse(localStorage.getItem("ballotmancy.component-study-views.v1") || "[]"));
  expect(stored).toHaveLength(1);
  expect(stored[0]).toMatchObject({ name: "Arizona reference", studyId: "ordinary-2018-f", raceId: "2018:AZ:senate:regular", compare: true });
  expect(stored[0].sha256).toMatch(/^[a-f0-9]{64}$/);
  await page.reload();
  await openStudyControls(page);
  await page.locator("details.composition-saved > summary").click();
  await expect(page.getByRole("button", { name: "Restore Arizona reference", exact: true })).toBeVisible();
  await savedStudies(page).getByRole("button", { name: "With polls", exact: true }).click();
  await expect(page).toHaveURL(/study=ordinary-2018-fp/);
  await page.getByRole("button", { name: "Restore Arizona reference", exact: true }).click();
  await expect(page).toHaveURL(/study=ordinary-2018-f&/);
  await expect(detail(page).getByRole("heading", { name: "Arizona", exact: true })).toBeVisible();
  await expect(page.getByRole("checkbox", { name: "Compare with the other study result", exact: true })).toBeChecked();
  await page.getByRole("button", { name: "Delete Arizona reference", exact: true }).click();
  await expect(page.getByRole("button", { name: "Restore Arizona reference", exact: true })).toHaveCount(0);
});

test("unknown study links recover to the full forecast and failed result loads preserve the prior study", async ({ page }) => {
  await page.goto("/?study=unknown");
  await expect(page.getByRole("heading", { name: "Study unavailable", exact: true })).toBeVisible();
  await expect(page.getByRole("main").getByRole("alert")).toContainText("does not identify an available result");
  await page.getByRole("button", { name: "Return to the full Senate forecast", exact: true }).click();
  await expect(page.getByRole("region", { name: "The path to a majority", exact: true })).toBeVisible();
  await page.route("**/api/component-studies?result=ordinary-2018-fp", route => route.fulfill({ status: 503, json: { error: "Synthetic unavailable result" } }));
  await page.goto("/?study=ordinary-2018-f");
  await openStudyControls(page);
  await expect(scoreboard(page).locator(".scoreboard-value").first()).toHaveText(/24\s*\/\s*30/);
  await savedStudies(page).getByRole("button", { name: "With polls", exact: true }).click();
  await expect(page.getByRole("main").getByRole("alert")).toContainText("previous result remains displayed");
  await expect(page).toHaveURL(/study=ordinary-2018-f&/);
  await expect(scoreboard(page).locator(".scoreboard-value").first()).toHaveText(/24\s*\/\s*30/);
  await page.unroute("**/api/component-studies?result=ordinary-2018-fp");
  await savedStudies(page).getByRole("button", { name: "With polls", exact: true }).click();
  await expect(page).toHaveURL(/study=ordinary-2018-fp/);
  await expect(scoreboard(page).locator(".scoreboard-value").first()).toHaveText(/27\s*\/\s*30/);
});
