import { expect, test, type Page } from "@playwright/test";

const run = (year = 2020, polls = false, model = "relative_context") => `house_${model}_v1-${year}-h14-full435-${polls ? "polls-" : ""}v1`;
const builder = (page: Page) => page.locator("details.house-configuration");
const map = (page: Page) => page.getByRole("region", { name: "House map", exact: true });
const detail = (page: Page) => page.getByRole("complementary", { name: "Selected House race detail", exact: true });
async function openBuilder(page: Page) {
  await expect(builder(page)).toBeVisible();
  if (await builder(page).getAttribute("open") === null) await builder(page).locator(":scope > summary").click();
}

test("published default remains unchanged and exposes the all-district polling builder", async ({ page }) => {
  await page.goto("/?chamber=house&edition=historical");
  await expect(map(page).locator("path[data-forecast=available]")).toHaveCount(375);
  await expect(builder(page)).not.toHaveAttribute("open", "");
  await openBuilder(page);
  await page.getByRole("button", { name: "Open all-district polling options", exact: true }).click();
  await expect(map(page).locator("path[data-forecast=available]")).toHaveCount(435);
  await expect(page.getByRole("checkbox", { name: "Include historical polls", exact: true })).not.toBeChecked();
  await expect(builder(page)).toHaveAttribute("open", "");
});

test("polling checkbox preserves district and map, compares matched baseline, and restores exact saved pair", async ({ page }) => {
  const errors: string[] = []; page.on("pageerror", error => errors.push(error.message));
  await page.goto(`/?chamber=house&run=${run()}&race=2020%3AAK-00%3Ahouse%3Aregular&view=actual`);
  await expect(map(page).locator("path[data-forecast=available]")).toHaveCount(435);
  await expect(builder(page)).not.toHaveAttribute("open", "");
  await openBuilder(page);
  const toggle = page.getByRole("checkbox", { name: "Include historical polls", exact: true });
  await expect(toggle).not.toBeChecked();
  await toggle.click();
  await expect(page).toHaveURL(new RegExp(`run=${run(2020, true)}&race=2020%3AAK-00%3Ahouse%3Aregular&view=actual`));
  await expect(toggle).toBeChecked();
  await expect(map(page).locator("path[data-forecast=available]")).toHaveCount(435);
  await expect(page.getByRole("region", { name: "House group-win probability ribbon" }).locator(".house-ribbon-cell")).toHaveCount(435);
  await expect(detail(page).getByRole("region", { name: "District polling information" })).toContainText("2 local polls");
  await page.getByRole("button", { name: "Compare with the same model without polls", exact: true }).click();
  await expect(page.getByRole("region", { name: "House model comparison", exact: true })).toBeVisible();
  await expect(page.locator(".historical-scoreboard")).toContainText("Compared result:");
  await detail(page).getByText("Vote shares with polling", { exact: true }).click();
  await expect(detail(page).locator(".house-polling-shares")).toContainText("Middle 90% of modeled shares");
  await expect(detail(page).locator(".house-polling-shares")).toContainText("excluding Other");
  await builder(page).getByText("Saved group views", { exact: false }).first().click();
  await page.getByLabel("Name this group view", { exact: true }).fill("Polling pair");
  await page.getByRole("button", { name: "Save group view", exact: true }).click();
  await expect(builder(page)).toContainText("Saved Polling pair in this browser.");
  await page.getByRole("group", { name: "Election year", exact: true }).getByRole("button", { name: "2018", exact: true }).click();
  await expect(page).toHaveURL(new RegExp(`run=${run(2018, true)}&view=actual$`));
  await expect(toggle).toBeChecked();
  await expect(page.locator(".historical-scoreboard")).toContainText("434 / 435");
  await page.getByRole("button", { name: "Polling pair", exact: true }).click();
  await expect(page).toHaveURL(new RegExp(`compare=${run()}$`));
  await expect(detail(page).getByRole("heading", { name: "AK-00", exact: true })).toBeVisible();
  await expect(toggle).toBeChecked();
  expect(errors).toEqual([]);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

test("failed polling and share loads preserve the displayed forecast and recover explicitly", async ({ page }) => {
  await page.goto(`/?chamber=house&run=${run()}`);
  await openBuilder(page);
  let fail = true;
  await page.route(`**/api/chambers/house/groups?run=${run(2020, true)}`, route => fail ? route.fulfill({ status: 503, body: "unavailable" }) : route.continue());
  const toggle = page.getByRole("checkbox", { name: "Include historical polls", exact: true });
  await toggle.click();
  await expect(page.locator(".compare-banner[role=alert]")).toContainText("Unable to change House group forecasts");
  await expect(toggle).not.toBeChecked();
  await expect(page).not.toHaveURL(/full435-polls/);
  await expect(map(page).locator("path[data-forecast=available]")).toHaveCount(435);
  fail = false;
  await page.getByRole("button", { name: "Try selection again", exact: true }).click();
  await expect(toggle).toBeChecked();
  await page.goto(`/?chamber=house&run=${run(2020, true)}&race=2020%3AAK-00%3Ahouse%3Aregular`);
  let failShares = true;
  await page.route("**/api/chambers/house/groups/shares?*", route => failShares ? route.fulfill({ status: 503, body: "unavailable" }) : route.continue());
  await detail(page).getByText("Vote shares with polling", { exact: true }).click();
  await expect(detail(page).locator(".house-polling-shares [role=alert]")).toBeVisible();
  await expect(map(page).locator("path[data-forecast=available]")).toHaveCount(435);
  failShares = false;
  await page.getByRole("button", { name: "Retry vote shares", exact: true }).click();
  await expect(detail(page).locator(".house-polling-shares")).toContainText("Middle 90% of modeled shares");
});

test("unpolled and out-of-share-study districts retain honest coverage explanations", async ({ page }) => {
  await page.goto(`/?chamber=house&run=${run(2024, true)}&race=2024%3ALA-04%3Ahouse%3Aregular`);
  await expect(detail(page).getByRole("region", { name: "District polling information" })).toContainText("No local polls were included");
  await detail(page).getByText("Vote shares with polling", { exact: true }).click();
  await expect(detail(page).locator(".house-polling-shares")).toContainText("outside the two-party vote-share study");
  await expect(page.getByRole("region", { name: "House control availability" })).toContainText("remain unavailable");
});
