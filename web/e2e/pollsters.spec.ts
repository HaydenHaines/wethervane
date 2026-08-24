import { expect, test } from "@playwright/test";

const COVERAGE_REPORT = {
  metadata: {
    total_polls: 204,
    polls_with_xt_data: 24,
    polls_analyzed: 24,
    active_xt_columns: [
      "xt_education_college",
      "xt_race_white",
      "xt_race_black",
      "xt_race_hispanic",
    ],
    mappable_xt_columns: [
      "xt_education_college",
      "xt_race_white",
      "xt_race_black",
      "xt_race_hispanic",
    ],
    oversample_threshold: 1.2,
    undersample_threshold: 0.8,
  },
  summary: {
    by_group: {
      xt_race_hispanic: {
        label: "Hispanic",
        n_undersampled: 10,
        n_oversampled: 1,
        n_representative: 13,
        n_total_polls: 24,
        top_affected_types: [
          { type_label: "South Texas Hispanic Belt", n_races_affected: 4 },
          { type_label: "Southwest Metro Mix", n_races_affected: 3 },
        ],
      },
      xt_race_asian: {
        label: "Asian",
        n_undersampled: 7,
        n_oversampled: 0,
        n_representative: 17,
        n_total_polls: 24,
        top_affected_types: [{ type_label: "Pacific Tech Suburbs", n_races_affected: 3 }],
      },
    },
    undersampled_ranking: [
      { group: "xt_race_hispanic", label: "Hispanic", n_polls_undersampled: 10 },
      { group: "xt_race_asian", label: "Asian", n_polls_undersampled: 7 },
    ],
  },
  per_poll_results: [
    {
      race: "AZ-SEN",
      state: "Arizona",
      pollster: "Desert Research",
      date: "2026-04-18",
      n_sample: 1100,
      n_groups_analyzed: 4,
      n_undersampled: 2,
      n_oversampled: 1,
      gaps: [
        {
          demographic_group: "xt_race_hispanic",
          label: "Hispanic",
          poll_share: 0.17,
          population_share: 0.29,
          ratio: 0.59,
          status: "undersampled",
          affected_types: [
            {
              type_id: 18,
              display_name: "Southwest Metro Mix",
              group_share: 0.42,
              state_weight: 0.31,
              exposure: 0.13,
            },
          ],
        },
        {
          demographic_group: "xt_race_asian",
          label: "Asian",
          poll_share: 0.02,
          population_share: 0.05,
          ratio: 0.4,
          status: "undersampled",
          affected_types: [
            {
              type_id: 27,
              display_name: "Pacific Tech Suburbs",
              group_share: 0.18,
              state_weight: 0.12,
              exposure: 0.02,
            },
          ],
        },
      ],
    },
  ],
};

test.describe("Pollsters page coverage diagnostics", () => {
  test("renders populated diagnostics alongside the accuracy table", async ({ page }) => {
    let sawCoverageRequest = false;

    await page.route("**/api/v1/pollsters/coverage", async (route) => {
      sawCoverageRequest = true;
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(COVERAGE_REPORT),
      });
    });

    await page.goto("/pollsters");

    await expect(page.locator("h1")).toContainText("Pollster Accuracy");
    await expect(page.locator('[data-testid="pollster-accuracy-table"]')).toBeVisible({
      timeout: 15_000,
    });
    await expect(page.locator('[data-testid="coverage-section"]')).toBeVisible({
      timeout: 15_000,
    });
    await expect(page.getByText("Poll Coverage Diagnostics")).toBeVisible();
    await expect(
      page.locator('[data-testid="coverage-top-groups"]').getByText("Hispanic", {
        exact: true,
      }),
    ).toBeVisible();
    await expect(page.getByText("South Texas Hispanic Belt (4)")).toBeVisible();
    await expect(page.getByText("Desert Research • AZ-SEN")).toBeVisible();
    await expect(page.getByText("Poll share 17.0% vs electorate 29.0%")).toBeVisible();
    expect(sawCoverageRequest).toBeTruthy();
  });

  test("renders an explicit unavailable state for a 503 report", async ({ page }) => {
    let sawCoverageRequest = false;

    await page.route("**/api/v1/pollsters/coverage", async (route) => {
      sawCoverageRequest = true;
      await route.fulfill({
        status: 503,
        contentType: "application/json",
        body: JSON.stringify({
          detail:
            "Poll coverage diagnostics report not yet generated. Run: uv run python scripts/analyze_poll_coverage.py",
        }),
      });
    });

    await page.goto("/pollsters");

    await expect(page.locator('[data-testid="pollster-accuracy-table"]')).toBeVisible({
      timeout: 15_000,
    });
    await expect(page.locator('[data-testid="coverage-unavailable"]')).toBeVisible({
      timeout: 15_000,
    });
    await expect(page.getByText("Coverage diagnostics unavailable")).toBeVisible();
    await expect(page.getByText("The current report has not been generated yet.")).toBeVisible();
    await expect(
      page.getByText(
        "Poll coverage diagnostics report not yet generated. Run: uv run python scripts/analyze_poll_coverage.py",
      ),
    ).toBeVisible();
    await expect(
      page.getByText("Could not load pollster accuracy data. Please try again later."),
    ).toHaveCount(0);
    expect(sawCoverageRequest).toBeTruthy();
  });
});
