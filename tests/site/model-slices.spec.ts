import { expect, test, type Page } from "@playwright/test";

async function openBuilder(page: Page) {
  const controls = page.locator("details.stack-builder");
  await expect(controls).toBeVisible();
  if (await controls.getAttribute("open") === null) await controls.locator(":scope > summary").click();
  await expect(controls).toHaveAttribute("open", "");
  return controls;
}

const polling = (page: Page) => page.getByRole("checkbox", { name: "Include polling evidence", exact: true });

test("all eight model responsibilities are visible with honest required and unavailable states", async ({ page }) => {
  await page.goto("/?run=current-polling");
  const controls = await openBuilder(page);
  const slices = controls.locator(".model-slice");
  await expect(slices).toHaveCount(8);
  for (const [id, label] of [
    ["population", "Population & geography"],
    ["community", "Community composition"],
    ["fundamentals", "Electoral history"],
    ["environment", "National environment"],
    ["measurement", "Polls & surveys"],
    ["participation", "Turnout & participation"],
    ["choice", "Candidate choice"],
    ["counts_rules", "Vote counts & election rules"],
  ]) {
    const slice = controls.locator(`.model-slice[data-slice="${id}"]`);
    await expect(slice.getByRole("heading", { name: label, exact: true })).toBeVisible();
  }
  await expect(controls.locator(".model-slice.is-unavailable")).toHaveCount(5);
  for (const id of ["population", "community", "participation", "choice", "counts_rules"]) {
    const slice = controls.locator(`.model-slice[data-slice="${id}"]`);
    const disabled = slice.getByRole("checkbox");
    const reason = slice.locator(".slice-unavailable-reason");
    await expect(disabled).toBeDisabled();
    await expect(disabled).not.toBeChecked();
    await expect(slice.getByText("Not connected yet", { exact: true })).toBeVisible();
    await expect(reason).toBeVisible();
    expect((await reason.innerText()).trim().length).toBeGreaterThan(30);
    await expect(disabled).toHaveAttribute("aria-describedby", (await reason.getAttribute("id"))!);
  }
  for (const id of ["fundamentals", "environment"]) {
    const slice = controls.locator(`.model-slice[data-slice="${id}"]`);
    await expect(slice.getByText("Always included", { exact: true })).toBeVisible();
    await expect(slice.getByRole("checkbox")).toHaveCount(0);
    await expect(slice.getByRole("combobox")).toBeEnabled();
    await expect(slice.locator('option[value="none"]')).toHaveCount(0);
  }
  await expect(polling(page)).toBeChecked();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBeTruthy();
});

test("polling evidence toggles by keyboard and changes only to verified saved results", async ({ page }) => {
  await page.goto("/?run=current-polling");
  const controls = page.locator("details.stack-builder");
  await expect(controls).not.toHaveAttribute("open", "");
  await expect(polling(page)).not.toBeVisible();
  await openBuilder(page);
  await expect(polling(page)).toBeChecked();
  await expect(page.getByLabel("Polling component", { exact: true })).toHaveCount(0);
  await polling(page).focus();
  await page.keyboard.press("Space");
  await expect(polling(page)).not.toBeChecked();
  await expect(page).toHaveURL(/run=current-fundamentals(?:&|$)/);
  await expect(page.locator(".composition-state")).toContainText("Ready");
  await polling(page).focus();
  await page.keyboard.press("Space");
  await expect(page).toHaveURL(/run=current-polling(?:&|$)/);
  await expect(polling(page)).toBeChecked();
  await expect(page.locator(".composition-state")).toContainText("Ready");
});

test("polling restores the last valid historical choice and falls back safely for the current year", async ({ page }) => {
  await page.goto("/?run=history-2018-FP_persistent-1");
  await openBuilder(page);
  const choice = page.getByLabel("Polling component", { exact: true });
  await expect(choice).toHaveValue("persistent");
  await expect(choice.locator('option[value="none"]')).toHaveCount(0);
  await polling(page).uncheck();
  await expect(page).toHaveURL(/run=history-2018-F-1(?:&|$)/);
  await expect(choice).toBeDisabled();
  await polling(page).check();
  await expect(choice).toHaveValue("persistent");
  await expect(page).toHaveURL(/run=history-2018-FP_persistent-1(?:&|$)/);
  await page.getByRole("group", { name: "Election year", exact: true }).getByRole("button", { name: "2026", exact: true }).click();
  await expect(page).toHaveURL(/run=current-polling(?:&|$)/);
  await expect(polling(page)).toBeChecked();
  await expect(choice).toHaveCount(0);
  await polling(page).uncheck();
  await expect(page).toHaveURL(/run=current-fundamentals(?:&|$)/);
  await polling(page).check();
  await expect(page).toHaveURL(/run=current-polling(?:&|$)/);
  await expect(page.locator(".composition-state")).toContainText("Ready");
});

test("an unsupported polling toggle keeps the prior result and its numbers explicitly displayed", async ({ page }) => {
  await page.goto("/?run=current-error-high&race=2026%3ANE%3Asenate%3Aregular");
  await openBuilder(page);
  const chamber = page.getByRole("region", { name: "Who could control the chamber?", exact: true });
  const detail = page.getByRole("complementary", { name: "Selected race detail", exact: true });
  await expect(detail.getByRole("heading", { name: "Nebraska", exact: true })).toBeVisible();
  const oldControl = await chamber.innerText();
  const oldProbabilities = await detail.locator(".candidate-prob").allTextContents();
  await polling(page).uncheck();
  await expect(page.locator(".composition-state")).toContainText("Unsupported combination");
  await expect(page.locator(".composition-state")).toContainText("Still displaying");
  await expect(page.locator(".stack-builder-summary")).toContainText("Unsupported combination");
  await expect(page).toHaveURL(/run=current-error-high(?:&|$)/);
  await expect(chamber).toHaveText(oldControl, { useInnerText: true });
  await expect(detail.locator(".candidate-prob")).toHaveText(oldProbabilities);
  await page.getByLabel("Uncertainty", { exact: true }).selectOption("standard");
  await expect(page).toHaveURL(/run=current-fundamentals(?:&|$)/);
  await expect(page.locator(".composition-state")).toContainText("Ready");
});
