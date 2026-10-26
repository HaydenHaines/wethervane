import { expect, test } from "@playwright/test";

test("refreshed poll explanations connect counts, date bounds, sample Ns and provider proxies", async ({ page }) => {
  await page.goto("/?run=current-polling&race=2026%3ATX%3Asenate%3Aregular");
  const detail = page.getByRole("complementary", { name: "Selected race detail" });
  await expect(detail.getByText("Poll records included", { exact: true })).toBeVisible();
  await detail.getByRole("link", { name: /Why some polls are included/ }).click();
  const policy = page.locator("#poll-selection");
  await expect(policy).toContainText("131 poll records across 25 of 35 races");
  const disclosure = policy.locator(".poll-records > summary");
  await disclosure.focus();
  await page.keyboard.press("Enter");
  await expect(page.getByLabel("Poll records for", { exact: true })).toHaveValue("TX");
  await expect(policy.locator(".poll-record-status.included")).toHaveCount(17);
  await expect(policy.locator(".poll-record-status.excluded")).toHaveCount(7);
  await expect(policy.locator(".poll-record-list")).toContainText("YouGov");
  await expect(policy.locator(".poll-record-list a").first()).toHaveAttribute("href", /^https?:\/\//);
  await page.getByLabel("Poll records for", { exact: true }).selectOption("SD");
  await expect(policy.locator(".poll-record-status.included")).toHaveCount(4);
  await expect(policy.locator(".poll-record-status.excluded")).toHaveCount(4);
  await page.getByLabel("Poll records for", { exact: true }).selectOption("AK");
  await expect(policy.locator(".poll-record-status.included")).toHaveCount(2);
  await expect(policy.locator(".poll-record-status.excluded")).toHaveCount(6);
  const akQuestion = policy.locator(".poll-record-list li")
    .filter({ hasText: "Data for Progress" })
    .filter({ has: page.locator(".poll-record-status.included") });
  await expect(akQuestion).toContainText("Exact publication date not verified.");
  await expect(akQuestion).toContainText("Source confirmed available by Mon, 26 Oct 2026 15:03:57 GMT (not a publication date).");
  await expect(akQuestion).toContainText("Reported sample size for this question: 671.");
  await expect(akQuestion).toContainText("Reported overall survey sample: 694 · Likely voters.");
  await expect(akQuestion).toContainText("polling-provider grouping uses the named survey organization");
  await expect(akQuestion.getByRole("link", { name: "Publisher or source record ↗" })).toHaveAttribute("href", /^https:\/\//);
  const alternative = policy.locator(".poll-record-list li")
    .filter({ hasText: "Data for Progress" })
    .filter({ hasText: /Ranked-choice first preference/ });
  await expect(alternative).toContainText("final ranked-choice preferences");
  await expect(alternative).toContainText("Preferred question in this same survey: Ranked-choice final preference; Likely voters (included).");

  await policy.locator(".assumptions > summary").click();
  await expect(policy).toContainText("22 included records have no verified exact publication date.");
  await expect(policy).toContainText("that time is not presented as their release date");
  await page.getByLabel("Poll records for", { exact: true }).selectOption("CO");
  await expect(policy).toContainText("That does not establish that no polls were conducted.");
  await page.getByLabel("Poll records for", { exact: true }).selectOption("");
  await expect(policy.locator(".poll-record-status.included")).toHaveCount(131);
  await expect(policy.locator(".poll-record-status.excluded")).toHaveCount(56);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBeTruthy();
});

test("poll source explanation follows the selected model and historical evidence limits", async ({ page }) => {
  await page.goto("/?run=current-fundamentals");
  const policy = page.locator("#poll-selection");
  await expect(policy).toContainText("No polls are included.");
  await expect(policy.locator(".poll-records")).toHaveCount(0);
  await page.locator(".stack-builder > summary").click();
  await page.getByRole("checkbox", { name: "Include polling evidence", exact: true }).check();
  await expect(policy).toContainText("131 poll records across 25 of 35 races");
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
