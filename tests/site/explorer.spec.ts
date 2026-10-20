import { expect, test } from "@playwright/test";

test("2018 archive preserves grouped candidates, all contests, and simultaneous seats", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/?run=history-2018-FP-1&race=2018%3ACA%3Asenate%3Aregular");
  await expect(page.getByRole("group", { name: "Election year" }).getByRole("button", { name: "2018", exact: true })).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByLabel("Forecast date", { exact: true })).toBeDisabled();
  await expect(page.getByLabel("Forecast date", { exact: true })).toHaveValue("h14");
  const ribbon = page.getByRole("region", { name: "Historical race ribbon, ordered by leading outcome win probability", exact: true });
  await expect(ribbon.locator("button[data-race-id]")).toHaveCount(35);
  await expect(page.locator(".race-table tbody tr")).toHaveCount(35);
  await expect(page.locator("svg [role=button]")).toHaveCount(33);
  await expect(page.locator(".threshold-combined")).toHaveCount(0);
  const detail = page.getByRole("complementary", { name: "Selected race detail" });
  await expect(detail.getByRole("heading", { name: "California", exact: true })).toBeVisible();
  await expect(detail.locator(".candidate")).toHaveCount(1);
  await expect(detail.locator(".candidate-name")).toHaveText("Democratic candidates");
  await expect(detail.locator(".candidate-prob")).toHaveText("100.0%");
  await expect(detail.getByText(/Group probability for Dianne Feinstein; Kevin De Leon/)).toBeVisible();
  await expect(detail.getByText(/probability is not allocated between individual members/)).toBeVisible();
  await expect(detail.locator(".observed-result.winner")).toContainText("Observed winner: Dianne Feinstein");
  await expect(detail.getByText(/Candidate vote-share estimates are unavailable/)).toBeVisible();
  await expect(detail.locator(".interval-line")).toHaveCount(0);
  const performance = page.getByRole("region", { name: "Against the observed result", exact: true });
  await expect(performance).toContainText("Observed seats: D 47, R 53, unaligned 0.");
  await expect(performance.locator(".performance-metric").first()).toContainText("0.829");
  await expect(performance.getByRole("img", { name: /^Democratic caucus seat probability distribution\. Observed 47 seats\./ })).toBeVisible();
  await expect(performance.getByRole("img", { name: /^Republican caucus seat probability distribution\. Observed 53 seats\./ })).toBeVisible();
  await expect(performance.locator("polyline")).toHaveCount(0);

  const regular = ribbon.locator('[data-race-id="2018:MN:senate:regular"]');
  await regular.focus();
  await page.keyboard.press("Enter");
  await expect(regular).toHaveAttribute("aria-pressed", "true");
  const seat = page.getByLabel("Choose election in Minnesota", { exact: true });
  await expect(seat.locator("option")).toHaveCount(2);
  await seat.selectOption("2018:MN:senate:special");
  await expect(detail.getByText("U.S. Senate · Special election", { exact: true })).toBeVisible();
  await page.getByRole("group", { name: "Map view", exact: true }).getByRole("button", { name: "Actual winners", exact: true }).click();
  await expect(page.getByRole("button", { name: /^Minnesota, 2 elections, observed winner/ })).toHaveAttribute("aria-label", /Smith/);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy();
  expect(errors).toEqual([]);
});

test("2022 performance and comparisons follow the selected stack without changing observed truth", async ({ page }) => {
  await page.goto("/?run=history-2022-F-1&race=2022%3AAK%3Asenate%3Aregular");
  const performance = page.getByRole("region", { name: "Against the observed result", exact: true });
  const chamber = page.getByRole("region", { name: "Who could control the chamber?", exact: true });
  await expect(page.getByLabel("Polling component", { exact: true })).toHaveValue("none");
  await expect(performance.locator(".performance-metric").first()).toContainText("0.684");
  await expect(chamber).toContainText("D needs 50; R needs 51");
  await expect(performance).toContainText("Observed seats: D 51, R 49, unaligned 0.");
  await expect(performance.getByRole("img", { name: /^Democratic caucus seat probability distribution\. Observed 51 seats\./ })).toBeVisible();
  await expect(performance.getByRole("img", { name: /^Republican caucus seat probability distribution\. Observed 49 seats\./ })).toBeVisible();
  await expect(performance.locator("polyline")).toHaveCount(0);
  const detail = page.getByRole("complementary", { name: "Selected race detail" });
  await expect(detail.getByText(/Group probability for.*Murkowski/)).toBeVisible();
  await expect(detail.locator(".observed-result.winner")).toContainText("Murkowski");
  await page.getByRole("checkbox", { name: "Compare default", exact: true }).check();
  await expect(page.getByText("Comparing with the 2022 default", { exact: true })).toBeVisible();
  await expect(performance.locator(".performance-metric").first()).toContainText("+0.126 vs default");
  await expect(performance.locator("polyline")).toHaveCount(2);
  await expect(performance).toContainText("the dashed outline shows the default stack");
  await page.getByLabel("Caucus context", { exact: true }).selectOption("unresolved_other_D");
  await expect(performance).toContainText("Observed control: D caucus. Observed seats: D 51, R 49, unaligned 0.");
  await page.getByLabel("Polling component", { exact: true }).selectOption("persistent");
  await expect(page).toHaveURL(/run=history-2022-FP_persistent-1/);
  await expect(performance.locator(".performance-metric").first()).toContainText("0.721");
  await expect(performance).toContainText("Observed seats: D 51, R 49, unaligned 0.");
  await page.getByLabel("Base model", { exact: true }).selectOption("winner");
  await expect(page.getByLabel("Base model", { exact: true })).toHaveValue("winner");
  await expect(page.getByLabel("Polling component", { exact: true })).toHaveValue("none");
  await expect(page.getByRole("status").filter({ hasText: "Adjusted" })).toBeVisible();
  await expect(page).toHaveURL(/run=history-2022-W-1/);
  await page.getByRole("checkbox", { name: "Compare default", exact: true }).uncheck();
  await expect(performance.locator("polyline")).toHaveCount(0);
});

test("deep links, share links, reload, Back, and reset preserve a coherent selection", async ({ page }) => {
  await page.goto("/?run=history-2022-F-1&scenario=asof_alignment_other_U&race=2022%3AOK%3Asenate%3Aspecial&compare=1&view=actual");
  await expect(page.getByLabel("Choose election in Oklahoma", { exact: true })).toHaveValue("2022:OK:senate:special");
  await expect(page.getByRole("button", { name: "Actual winners", exact: true })).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByRole("checkbox", { name: "Compare default", exact: true })).toBeChecked();
  await page.getByLabel("Uncertainty", { exact: true }).selectOption("loading-2");
  await expect(page).toHaveURL(/run=history-2022-F-2/);
  await page.getByLabel("Caucus context", { exact: true }).selectOption("unresolved_other_R");
  await page.getByRole("button", { name: "Share view ↗", exact: true }).click();
  const shared = page.getByLabel("Shareable view", { exact: true });
  await expect(shared).toHaveValue(page.url());
  expect(new URL(await shared.inputValue()).searchParams.get("scenario")).toBe("unresolved_other_R");
  await page.reload();
  await expect(page.getByLabel("Uncertainty", { exact: true })).toHaveValue("loading-2");
  await expect(page.getByLabel("Caucus context", { exact: true })).toHaveValue("unresolved_other_R");
  await expect(page.getByLabel("Choose election in Oklahoma", { exact: true })).toHaveValue("2022:OK:senate:special");
  await page.getByLabel("Uncertainty", { exact: true }).selectOption("loading-0");
  await expect(page).toHaveURL(/run=history-2022-F-0/);
  await page.goBack();
  await expect(page.getByLabel("Uncertainty", { exact: true })).toHaveValue("loading-2");
  await expect(page).toHaveURL(/run=history-2022-F-2/);
  await page.getByRole("button", { name: "Reset stack", exact: true }).click();
  await expect(page).toHaveURL(/run=history-2022-FP-1/);
  await expect(page.getByLabel("Polling component", { exact: true })).toHaveValue("standard");
  await expect(page.getByLabel("Uncertainty", { exact: true })).toHaveValue("loading-1");
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy();
});

test("current sensitivities remain distinct from historical performance and unsupported stacks", async ({ page }) => {
  await page.goto("/?run=does-not-exist");
  await expect(page).toHaveURL(/run=current-polling/);
  await expect(page.getByRole("status").filter({ hasText: "unavailable" })).toBeVisible();
  await expect(page.getByRole("region", { name: "What has been tested?", exact: true })).toContainText("The 2026 election has not been observed.");
  await expect(page.getByLabel("Polling component", { exact: true }).locator("option")).toHaveCount(2);
  await page.getByLabel("Uncertainty", { exact: true }).selectOption("error-high");
  await expect(page).toHaveURL(/run=current-error-high/);
  await expect(page.getByRole("region", { name: "The path to a majority", exact: true }).locator("button[data-race-id]")).toHaveCount(35);
  await page.getByLabel("Polling component", { exact: true }).selectOption("none");
  await expect(page).toHaveURL(/run=current-fundamentals/);
  await expect(page.getByLabel("Uncertainty", { exact: true })).toHaveValue("standard");
  await expect(page.getByLabel("Uncertainty", { exact: true })).toBeDisabled();
  await expect(page.getByRole("status").filter({ hasText: "Adjusted" })).toBeVisible();
  await page.getByRole("group", { name: "Election year", exact: true }).getByRole("button", { name: "2018", exact: true }).click();
  await expect(page).toHaveURL(/run=history-2018-F-1/);
  await expect(page.getByRole("region", { name: "Against the observed result", exact: true })).toBeVisible();
});

for (const failure of ["missing", "changed fingerprint"] as const) {
  test(`a ${failure} stack leaves the prior complete view available`, async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.route("**/data/explorer/history-2022-FP-1.json", async (route) => {
      if (failure === "missing") {
        await route.fulfill({ status: 404, body: "Unavailable" });
      } else {
        const response = await route.fetch();
        await route.fulfill({ response, body: (await response.text()) + "\n" });
      }
    });
    await page.goto("/");
    await expect(page.getByLabel("Polling component", { exact: true })).toHaveValue("standard");
    await page.getByRole("group", { name: "Election year", exact: true }).getByRole("button", { name: "2022", exact: true }).click();
    const error = page.getByRole("alert").filter({ hasText: "Unable to change stacks" });
    await expect(error).toContainText("Unable to change stacks");
    await expect(error).toContainText("The previous stack remains visible.");
    if (failure === "changed fingerprint") await expect(error).toContainText("fingerprint");
    await expect(page).toHaveURL(/run=current-polling/);
    await expect(page.getByRole("region", { name: "The path to a majority", exact: true })).toBeVisible();
    await expect(page.locator(".race-table tbody tr")).toHaveCount(35);
    await page.getByRole("button", { name: /^Texas, 1 election,/ }).click();
    await expect(page.getByRole("complementary").getByRole("heading", { name: "Texas", exact: true })).toBeVisible();
    await page.unroute("**/data/explorer/history-2022-FP-1.json");
    await page.getByRole("group", { name: "Election year", exact: true }).getByRole("button", { name: "2022", exact: true }).click();
    await expect(page).toHaveURL(/run=history-2022-FP-1/);
    await expect(page.getByRole("region", { name: "Against the observed result", exact: true })).toContainText("Observed seats: D 51, R 49, unaligned 0.");
    await expect(error).toHaveCount(0);
    expect(errors).toEqual([]);
  });
}
