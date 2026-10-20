import { expect, test } from "@playwright/test";

test("poll explanations connect race counts to included sources and specific omissions", async ({ page }) => {
  await page.goto("/?run=current-polling&race=2026%3ATX%3Asenate%3Aregular");
  const detail = page.getByRole("complementary", { name: "Selected race detail" });
  await expect(detail.getByText("Poll records included", { exact: true })).toBeVisible();
  await detail.getByRole("link", { name: /Why some polls are included/ }).click();
  const policy = page.locator("#poll-selection");
  await expect(policy).toContainText("49 poll records across 23 of 35 races");
  const disclosure = policy.locator(".poll-records > summary");
  await disclosure.focus();
  await page.keyboard.press("Enter");
  await expect(page.getByLabel("Poll records for", { exact: true })).toHaveValue("TX");
  await expect(policy.locator(".poll-record-status.excluded")).toHaveCount(2);
  await expect(policy.locator(".poll-record-list")).toContainText("YouGov");
  await expect(policy.locator(".poll-record-list a").first()).toHaveAttribute("href", /^https?:\/\//);
  await page.getByLabel("Poll records for", { exact: true }).selectOption("SD");
  await expect(policy.locator(".poll-record-status.included")).toHaveCount(0);
  await expect(policy.locator(".poll-record-status.excluded")).toHaveCount(1);
  await page.getByLabel("Poll records for", { exact: true }).selectOption("CO");
  await expect(policy).toContainText("That does not establish that no polls were conducted.");
  await page.getByLabel("Poll records for", { exact: true }).selectOption("");
  await expect(policy.locator(".poll-record-status.included")).toHaveCount(49);
  await expect(policy.locator(".poll-record-status.excluded")).toHaveCount(14);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBeTruthy();
});

test("poll source explanation follows the selected model and historical evidence limits", async ({ page }) => {
  await page.goto("/?run=current-fundamentals");
  const policy = page.locator("#poll-selection");
  await expect(policy).toContainText("No polls are included.");
  await expect(policy.locator(".poll-records")).toHaveCount(0);
  await page.locator(".stack-builder > summary").click();
  await page.getByRole("checkbox", { name: "Include polling evidence", exact: true }).check();
  await expect(policy).toContainText("49 poll records across 23 of 35 races");
  await page.getByRole("group", { name: "Election year", exact: true }).getByRole("button", { name: "2018", exact: true }).click();
  await expect(policy).toContainText("cutoff 14 days before the election");
  await expect(policy.locator(".poll-records")).toHaveCount(0);
  const scoreboard = page.getByRole("region", { name: "How this model did", exact: true });
  await expect(scoreboard).toContainText("Recreated in 2026");
  const quality = scoreboard.getByRole("article", { name: "Probability quality", exact: true });
  await expect(quality.locator(".scoreboard-value")).toHaveText(/^0\.\d{3}$/);
  await expect(quality.locator(".scoreboard-raw")).not.toHaveAttribute("open", "");
  await expect(quality).toContainText("0 is perfect on this 0–2 scale");
});
