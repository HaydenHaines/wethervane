"""Tests for scripts/build_county_type_assignments.py.

Tests the tract-to-county aggregation logic using synthetic data.
Does NOT depend on real data files.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add scripts/ to path so we can import the build script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from build_county_type_assignments import (  # noqa: E402
    aggregate_tract_to_county,
    derive_county_fips,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_tract_scores(
    n_tracts: int = 6,
    j: int = 3,
    county_fips_list: list[str] | None = None,
) -> pd.DataFrame:
    """Build synthetic tract-level type scores for testing.

    Creates n_tracts tracts split across given counties.
    Scores are deterministic for reproducibility.
    """
    if county_fips_list is None:
        county_fips_list = ["01001", "01003"]

    rng = np.random.RandomState(42)
    # Distribute tracts evenly across counties.
    tracts_per_county = n_tracts // len(county_fips_list)
    geoids = []
    for fips in county_fips_list:
        for i in range(tracts_per_county):
            # Tract GEOID = 5-digit county FIPS + 6-digit tract suffix.
            geoids.append(f"{fips}{str(i + 1).zfill(6)}")

    # Add remaining tracts to the last county if n_tracts doesn't divide evenly.
    while len(geoids) < n_tracts:
        geoids.append(f"{county_fips_list[-1]}{str(len(geoids) + 1).zfill(6)}")

    # Generate random scores and row-normalize.
    raw_scores = rng.rand(len(geoids), j)
    row_sums = raw_scores.sum(axis=1, keepdims=True)
    scores = raw_scores / row_sums

    data = {"tract_geoid": geoids}
    for i in range(j):
        data[f"type_{i}_score"] = scores[:, i]

    return pd.DataFrame(data)


def _make_weights(tract_scores: pd.DataFrame, values: list[float] | None = None) -> pd.Series:
    """Build synthetic population weights for testing."""
    geoids = tract_scores["tract_geoid"].tolist()
    if values is None:
        values = list(range(100, 100 + len(geoids)))
    return pd.Series(values[:len(geoids)], index=geoids[:len(values)])


# ---------------------------------------------------------------------------
# Tests: derive_county_fips
# ---------------------------------------------------------------------------

class TestDeriveCountyFips:
    def test_extracts_first_5_digits(self):
        geoids = pd.Series(["01001020100", "12345678901"])
        result = derive_county_fips(geoids)
        assert result.tolist() == ["01001", "12345"]

    def test_zero_pads_short_geoids(self):
        # Edge case: numeric GEOID stored without leading zeros.
        geoids = pd.Series(["1001020100"])  # 10 digits — missing leading zero.
        result = derive_county_fips(geoids)
        # zfill(11) pads to 11 digits, then first 5 = "01001".
        assert result.tolist() == ["01001"]


# ---------------------------------------------------------------------------
# Tests: aggregate_tract_to_county
# ---------------------------------------------------------------------------

class TestAggregateTractToCounty:
    def test_basic_aggregation_produces_correct_counties(self):
        tract_scores = _make_tract_scores(n_tracts=6, j=3, county_fips_list=["01001", "01003"])
        weights = _make_weights(tract_scores)
        result = aggregate_tract_to_county(tract_scores, weights)

        assert "county_fips" in result.columns
        assert set(result["county_fips"]) == {"01001", "01003"}

    def test_output_has_correct_score_columns(self):
        tract_scores = _make_tract_scores(n_tracts=4, j=5, county_fips_list=["01001"])
        weights = _make_weights(tract_scores)
        result = aggregate_tract_to_county(tract_scores, weights)

        score_cols = [c for c in result.columns if c.endswith("_score")]
        assert len(score_cols) == 5
        expected = [f"type_{i}_score" for i in range(5)]
        assert score_cols == expected

    def test_scores_sum_to_one(self):
        tract_scores = _make_tract_scores(n_tracts=10, j=4, county_fips_list=["01001", "01003", "01005"])
        weights = _make_weights(tract_scores)
        result = aggregate_tract_to_county(tract_scores, weights)

        score_cols = [c for c in result.columns if c.endswith("_score")]
        row_sums = result[score_cols].sum(axis=1)
        np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-10)

    def test_scores_are_nonnegative(self):
        tract_scores = _make_tract_scores(n_tracts=6, j=3)
        weights = _make_weights(tract_scores)
        result = aggregate_tract_to_county(tract_scores, weights)

        score_cols = [c for c in result.columns if c.endswith("_score")]
        assert (result[score_cols] >= 0).all().all()

    def test_equal_weights_give_simple_mean(self):
        """With equal weights, county scores should be the mean of tract scores."""
        tract_scores = _make_tract_scores(n_tracts=4, j=2, county_fips_list=["01001"])
        # Use empty Series to trigger equal-weight fallback.
        empty_weights = pd.Series(dtype=float)
        result = aggregate_tract_to_county(tract_scores, empty_weights)

        score_cols = [c for c in result.columns if c.endswith("_score")]
        tract_mean = tract_scores[score_cols].mean()
        # Row-normalized, but mean of row-normalized scores is also row-normalized
        # if all tracts in same county.
        county_scores = result[score_cols].iloc[0]
        expected = tract_mean / tract_mean.sum()
        np.testing.assert_allclose(county_scores.values, expected.values, atol=1e-10)

    def test_population_weighting_affects_result(self):
        """Tracts with higher weight should dominate the county score."""
        # Create 2 tracts in one county with very different scores.
        data = {
            "tract_geoid": ["01001000001", "01001000002"],
            "type_0_score": [1.0, 0.0],
            "type_1_score": [0.0, 1.0],
        }
        tract_scores = pd.DataFrame(data)

        # Give the first tract 99x the weight of the second.
        heavy_weights = pd.Series([990.0, 10.0], index=["01001000001", "01001000002"])
        result = aggregate_tract_to_county(tract_scores, heavy_weights)

        # County should be dominated by tract 1 (type_0).
        assert result["type_0_score"].iloc[0] > 0.95
        assert result["type_1_score"].iloc[0] < 0.05

    def test_county_fips_is_zero_padded_string(self):
        tract_scores = _make_tract_scores(n_tracts=2, j=2, county_fips_list=["01001"])
        weights = _make_weights(tract_scores)
        result = aggregate_tract_to_county(tract_scores, weights)

        fips = result["county_fips"].iloc[0]
        assert isinstance(fips, str)
        assert len(fips) == 5
        assert fips == "01001"

    def test_no_duplicate_county_fips(self):
        tract_scores = _make_tract_scores(n_tracts=6, j=3)
        weights = _make_weights(tract_scores)
        result = aggregate_tract_to_county(tract_scores, weights)

        assert result["county_fips"].is_unique

    def test_columns_in_natural_numeric_order(self):
        """Score columns should be type_0, type_1, ..., NOT alphabetical."""
        tract_scores = _make_tract_scores(n_tracts=4, j=12, county_fips_list=["01001"])
        weights = _make_weights(tract_scores)
        result = aggregate_tract_to_county(tract_scores, weights)

        score_cols = [c for c in result.columns if c.endswith("_score")]
        expected = [f"type_{i}_score" for i in range(12)]
        assert score_cols == expected, (
            f"Columns should be in natural numeric order, got: {score_cols}"
        )

    def test_missing_weights_fallback_to_one(self):
        """Tracts without population weights should get weight=1.0, not be dropped."""
        tract_scores = _make_tract_scores(n_tracts=4, j=2, county_fips_list=["01001"])
        # Only provide weights for half the tracts.
        partial_weights = pd.Series(
            [500.0, 500.0],
            index=tract_scores["tract_geoid"].iloc[:2].tolist(),
        )
        result = aggregate_tract_to_county(tract_scores, partial_weights)

        # Should still produce a result (all 4 tracts contribute).
        assert len(result) == 1
        score_cols = [c for c in result.columns if c.endswith("_score")]
        assert not result[score_cols].isna().any().any()
