import { expect, test, type Page } from "@playwright/test";
import { makeHousePublication } from "../fixtures/house.mjs";

const housePreparation = /The House,\s*in preparation\./;
const houseReady = /The House,\s*in perspective\./;
const districtId = (key: string) => `2022:${key}:house:regular`;

async function stubHouse(page: Page, publication = makeHousePublication()) {
  await page.route("**/api/chambers/house*", (route) =>
    route.fulfill({ json: publication }),
  );
}

async function stubPreparingHouse(page: Page) {
  await page.route("**/api/chambers/house*", (route) =>
    route.fulfill({ json: { schemaVersion: 1, status: "preparing" } }),
  );
}

test("a preparing House fixture keeps the collapsed Senate default and shows no invented results", async ({
  page,
}) => {
  await stubPreparingHouse(page);
  await page.goto("/");
  await expect(
    page.getByRole("tab", { name: "Senate", exact: true }),
  ).toHaveAttribute("aria-selected", "true");
  await expect(page.locator("details.stack-builder")).toBeVisible();
  await expect(page.locator("details.stack-builder")).not.toHaveAttribute(
    "open",
    "",
  );
  await expect(page).toHaveURL(/run=current-polling/);
  await page.getByRole("tab", { name: "House", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: housePreparation }),
  ).toBeVisible();
  await expect(
    page.getByRole("tab", { name: "House", exact: true }),
  ).toHaveAttribute("aria-selected", "true");
  await expect(
    page.getByRole("region", { name: "House race ribbon", exact: true }),
  ).toHaveCount(0);
  await expect(page.locator(".house-race-list button")).toHaveCount(0);
  await expect(page.locator(".prob-number")).toHaveCount(0);
  expect(new URL(page.url()).searchParams.get("chamber")).toBe("house");
});

test("House deep links and keyboard chamber tabs preserve their selected panel", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await stubPreparingHouse(page);
  await page.goto("/?chamber=house");
  const house = page.getByRole("tab", { name: "House", exact: true });
  const senate = page.getByRole("tab", { name: "Senate", exact: true });
  await expect(
    page.getByRole("heading", { name: housePreparation }),
  ).toBeVisible();
  await expect(house).toHaveAttribute("tabindex", "0");
  await expect(senate).toHaveAttribute("tabindex", "-1");
  await house.focus();
  await page.keyboard.press("ArrowLeft");
  await expect(senate).toBeFocused();
  await expect(senate).toHaveAttribute("aria-selected", "true");
  await expect(page.locator("details.stack-builder")).toBeVisible();
  await expect(page.locator("details.stack-builder")).not.toHaveAttribute(
    "open",
    "",
  );
  await senate.press("End");
  await expect(house).toBeFocused();
  await expect(
    page.getByRole("heading", { name: housePreparation }),
  ).toBeVisible();
  await house.press("Home");
  await expect(senate).toBeFocused();
  await expect(
    page.getByRole("tabpanel", { name: "Senate", exact: true }),
  ).toBeVisible();
  expect(errors).toEqual([]);
});

test("switching chambers, Back and Forward restore the exact prior Senate selection", async ({
  page,
}) => {
  await stubPreparingHouse(page);
  await page.goto(
    "/?run=history-2022-F-1&scenario=unresolved_other_R&race=2022%3AOK%3Asenate%3Aspecial&compare=1&view=actual",
  );
  const detail = page.getByRole("complementary", {
    name: "Selected race detail",
    exact: true,
  });
  await expect(
    detail.getByRole("heading", { name: "Oklahoma", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Actual winners", exact: true }),
  ).toHaveAttribute("aria-pressed", "true");
  const senateUrl = page.url();
  await page.getByRole("tab", { name: "House", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: housePreparation }),
  ).toBeVisible();
  await page.getByRole("tab", { name: "Senate", exact: true }).click();
  await expect(page).toHaveURL(senateUrl);
  await expect(
    detail.getByRole("heading", { name: "Oklahoma", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByLabel("Choose election in Oklahoma", { exact: true }),
  ).toHaveValue("2022:OK:senate:special");
  await expect(
    page.getByRole("button", { name: "Actual winners", exact: true }),
  ).toHaveAttribute("aria-pressed", "true");
  await page.goBack();
  await expect(
    page.getByRole("tab", { name: "House", exact: true }),
  ).toHaveAttribute("aria-selected", "true");
  await expect(
    page.getByRole("heading", { name: housePreparation }),
  ).toBeVisible();
  await page.goForward();
  await expect(page).toHaveURL(senateUrl);
  await expect(
    detail.getByRole("heading", { name: "Oklahoma", exact: true }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBeTruthy();
});

for (const eligibleCount of [16, 435]) {
  test(`${eligibleCount} eligible districts use one data-driven ribbon without fabricating chamber control`, async ({
    page,
  }) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    const publication = makeHousePublication({ eligibleCount });
    await stubHouse(page, publication);
    await page.goto("/?chamber=house");
    await expect(page.getByRole("heading", { name: houseReady })).toBeVisible();
    const ribbon = page.getByRole("region", {
      name: "House race ribbon",
      exact: true,
    });
    await expect(ribbon).toHaveCount(1);
    await expect(ribbon.locator("button[data-race-id]")).toHaveCount(
      eligibleCount,
    );
    await expect(page.locator(".house-race-list button")).toHaveCount(10);
    await page
      .getByRole("button", {
        name: `See all ${eligibleCount} races`,
        exact: true,
      })
      .click();
    await expect(page.locator(".house-race-list button")).toHaveCount(
      eligibleCount,
    );
    await page
      .getByRole("button", { name: "Show fewer races", exact: true })
      .click();
    await expect(page.locator(".house-race-list button")).toHaveCount(10);
    await expect(page.locator(".chamber-name")).toHaveAttribute(
      "data-chamber-outlook",
      "unavailable",
    );
    await expect(
      page.locator(".house-ribbon-cell.majority-position"),
    ).toHaveCount(0);
    await expect(page.locator(".house-tipping-summary")).toHaveCount(0);
    await expect(page.locator(".threshold-combined,.tile-markers")).toHaveCount(
      0,
    );
    await expect(
      page.getByRole("region", {
        name: "House control availability",
        exact: true,
      }),
    ).toContainText("Not available for this result");
    await expect(page.locator(".prob-number")).toHaveCount(0);
    await expect(ribbon).toContainText("Chamber tipping point unavailable");
    const scoreboard = page.getByRole("region", {
      name: "How this model did",
      exact: true,
    });
    const races = publication.result.races;
    const correct = races.filter(
      (race) =>
        race.outcomes.reduce((favorite, outcome) =>
          outcome.winProbability > favorite.winProbability ? outcome : favorite,
        ).id === race.observed!.outcomeId,
    ).length;
    const brier =
      races.reduce(
        (sum, race) =>
          sum +
          race.outcomes.reduce(
            (score, outcome) =>
              score +
              (outcome.winProbability -
                Number(outcome.id === race.observed!.outcomeId)) **
                2,
            0,
          ),
        0,
      ) / races.length;
    expect(brier).toBeGreaterThanOrEqual(0);
    expect(brier).toBeLessThanOrEqual(2);
    await expect(
      scoreboard
        .getByRole("article")
        .filter({
          has: page.getByRole("heading", {
            name: "Favorites that won",
            exact: true,
          }),
        })
        .locator(".scoreboard-value"),
    ).toHaveText(`${correct} / ${eligibleCount}`);
    await expect(
      scoreboard
        .getByRole("article")
        .filter({
          has: page.getByRole("heading", {
            name: "Probability error",
            exact: true,
          }),
        })
        .locator(".scoreboard-value"),
    ).toHaveText(brier.toFixed(3));
    await expect(
      scoreboard
        .getByRole("article")
        .filter({
          has: page.getByRole("heading", {
            name: "Tested coverage",
            exact: true,
          }),
        })
        .locator(".scoreboard-value"),
    ).toHaveText(`${eligibleCount} / 435`);
    const ranks = await ribbon
      .locator("button[data-race-id]")
      .evaluateAll((buttons) =>
        buttons.map((button) => Number(button.getAttribute("data-rank"))),
      );
    expect(ranks).toEqual(
      Array.from({ length: eligibleCount }, (_, index) => index + 1),
    );
    if (eligibleCount === 435) {
      const directions = await ribbon
        .locator("button[data-race-id]")
        .evaluateAll((buttons) => [
          ...new Set(
            buttons.map((button) => button.getAttribute("data-direction")),
          ),
        ]);
      expect(directions).toContain("left-to-right");
      expect(directions).toContain("right-to-left");
    }
    if (eligibleCount === 435) {
      const density = ribbon.getByRole("checkbox", {
        name: "Show district labels and larger tiles",
        exact: true,
      });
      await expect(density).not.toBeChecked();
      await expect(ribbon).toHaveClass(/is-compact/);
      const compactBox = await ribbon.boundingBox();
      expect(compactBox!.height).toBeLessThan(1000);
      const compactGrid = await ribbon
        .locator(".house-ribbon-grid")
        .boundingBox();
      const compactOrder = await ribbon
        .locator("button[data-race-id]")
        .evaluateAll((buttons) =>
          buttons.map((button) => [
            button.getAttribute("data-race-id"),
            button.getAttribute("data-rank"),
          ]),
        );
      await density.check();
      await expect(density).toBeChecked();
      await expect(ribbon).not.toHaveClass(/is-compact/);
      await expect
        .poll(
          async () =>
            (await ribbon.locator(".house-ribbon-grid").boundingBox())!.height,
        )
        .toBeGreaterThan(compactGrid!.height);
      await expect(ribbon.locator("button[data-race-id]")).toHaveCount(435);
      expect(
        await ribbon
          .locator("button[data-race-id]")
          .evaluateAll((buttons) =>
            buttons.map((button) => [
              button.getAttribute("data-race-id"),
              button.getAttribute("data-rank"),
            ]),
          ),
      ).toEqual(compactOrder);
      await density.uncheck();
      await expect(ribbon).toHaveClass(/is-compact/);
    }
    const first = ribbon.locator("button[data-race-id]").first();
    await first.focus();
    await page.keyboard.press("ArrowRight");
    const next = ribbon.locator("button[data-race-id]").nth(1);
    await expect(next).toBeFocused();
    await page.keyboard.press("Enter");
    const selectedId = await next.getAttribute("data-race-id");
    await expect(next).toHaveAttribute("aria-pressed", "true");
    const key = selectedId!.split(":")[1];
    await expect(
      page
        .getByRole("complementary", {
          name: "Selected House race detail",
          exact: true,
        })
        .getByRole("heading", { name: key, exact: true }),
    ).toBeVisible();
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBeTruthy();
    expect(errors).toEqual([]);
  });
}

test("district state focus, map keyboard selection and House history update the same detail", async ({
  page,
}) => {
  await stubHouse(page);
  await page.goto("/?chamber=house");
  await expect(page.getByRole("heading", { name: houseReady })).toBeVisible();
  const map = page.getByRole("region", { name: "House map", exact: true });
  const detail = page.getByRole("complementary", {
    name: "Selected House race detail",
    exact: true,
  });
  await page.getByLabel("Focus on a state", { exact: true }).selectOption("CA");
  await expect(map.locator(".district-map-svg [role=button]")).toHaveCount(2);
  const first = map.getByRole("button", { name: /^California district 01,/ });
  await first.click();
  await expect(
    detail.getByRole("heading", { name: "CA-01", exact: true }),
  ).toBeVisible();
  await expect(first).toHaveAttribute("aria-pressed", "true");
  await first.focus();
  await page.keyboard.press("ArrowRight");
  const second = map.getByRole("button", { name: /^California district 02,/ });
  await expect(second).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(
    detail.getByRole("heading", { name: "CA-02", exact: true }),
  ).toBeVisible();
  expect(new URL(page.url()).searchParams.get("race")).toBe(
    districtId("CA-02"),
  );
  await page.goBack();
  await expect(
    detail.getByRole("heading", { name: "CA-01", exact: true }),
  ).toBeVisible();
  await page.goForward();
  await expect(
    detail.getByRole("heading", { name: "CA-02", exact: true }),
  ).toBeVisible();
  await page
    .getByRole("group", { name: "Map view", exact: true })
    .getByRole("button", { name: "Actual winners", exact: true })
    .click();
  await expect(
    map.getByRole("button", {
      name: /^California district 02,.*observed winner/,
    }),
  ).toBeVisible();
  await page
    .getByRole("region", { name: "House race ribbon", exact: true })
    .locator(`[data-race-id="${districtId("TX-01")}"]`)
    .click();
  await expect(
    page.getByLabel("Focus on a state", { exact: true }),
  ).toHaveValue("TX");
  await expect(
    detail.getByRole("heading", { name: "TX-01", exact: true }),
  ).toBeVisible();
  await page.getByLabel("Focus on a state", { exact: true }).selectOption("AK");
  await expect(
    map.getByRole("button", { name: /^Alaska at-large district,/ }),
  ).toBeVisible();
  await expect(map.locator(".district-map-svg [role=button]")).toHaveCount(1);
});

test("a mismatched map vintage never colors districts and leaves race exploration available", async ({
  page,
}) => {
  const publication = makeHousePublication();
  publication.geography!.metadata.mapVintage = "house_election_map_2024";
  await stubHouse(page, publication);
  await page.goto("/?chamber=house");
  await expect(page.getByRole("heading", { name: houseReady })).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: "District map unavailable",
      exact: true,
    }),
  ).toBeVisible();
  await expect(page.locator(".district-map-svg")).toHaveCount(0);
  await expect(page.locator(".house-race-list button")).toHaveCount(10);
  await page
    .getByLabel("Search districts or candidates", { exact: true })
    .fill("CA-02");
  await expect(page.locator(".house-race-list button")).toHaveCount(1);
  await page
    .locator(".house-race-list")
    .getByRole("button", { name: /^CA-02/ })
    .click();
  await expect(
    page
      .getByRole("complementary", {
        name: "Selected House race detail",
        exact: true,
      })
      .getByRole("heading", { name: "CA-02", exact: true }),
  ).toBeVisible();
});

for (const failure of ["server error", "invalid result"] as const) {
  test(`a House ${failure} offers a working retry`, async ({ page }) => {
    let attempts = 0;
    await page.route("**/api/chambers/house", async (route) => {
      attempts++;
      if (attempts === 1) {
        if (failure === "server error")
          await route.fulfill({
            status: 503,
            json: { error: "Synthetic service failure" },
          });
        else {
          const invalid = makeHousePublication();
          invalid.result.coverage = [];
          await route.fulfill({ json: invalid });
        }
      } else
        await route.fulfill({
          json: { schemaVersion: 1, status: "preparing" },
        });
    });
    await page.goto("/?chamber=house");
    await expect(
      page.getByRole("heading", {
        name: "House view unavailable",
        exact: true,
      }),
    ).toBeVisible();
    await expect(page.locator(".house-race-list button")).toHaveCount(0);
    await page.getByRole("button", { name: "Try again", exact: true }).click();
    await expect(
      page.getByRole("heading", { name: housePreparation }),
    ).toBeVisible();
    expect(attempts).toBe(2);
    await page
      .getByRole("button", { name: "Explore the Senate", exact: false })
      .click();
    await expect(
      page.getByRole("tab", { name: "Senate", exact: true }),
    ).toHaveAttribute("aria-selected", "true");
    await expect(page.locator("details.stack-builder")).toBeVisible();
  });
}
