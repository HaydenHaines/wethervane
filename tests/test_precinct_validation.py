"""Tests for scripts/experiments/validate_precinct_types.py

Uses synthetic data with known structure to verify each analysis function.
No real data files are needed — all tests are pure unit tests.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Allow importing from scripts/experiments via sys.path adjustment
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.validate_precinct_types import (
    compute_county_precinct_stats,
    compute_precinct_variance_by_type,
    compute_super_type_summary,
    compute_type_prediction_correlation,
    compute_within_type_variance,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def simple_precinct_df():
    """10 precincts across 2 counties, 5 each with clean vote data.

    County A (FIPS 01001): 5 precincts, all ~60% Dem
    County B (FIPS 01003): 5 precincts, all ~30% Dem
    """
    rows = []
    for i in range(5):
        rows.append({
            "county_fips": "01001",
            "precinct_geoid": f"01001-p{i}",
            "votes_total": 200,
            "votes_dem": 120 + i,  # ~60% Dem
            "dem_share": (120 + i) / 200,
        })
    for i in range(5):
        rows.append({
            "county_fips": "01003",
            "precinct_geoid": f"01003-p{i}",
            "votes_total": 200,
            "votes_dem": 60 + i,  # ~30% Dem
            "dem_share": (60 + i) / 200,
        })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def county_stats_simple():
    """County-level stats for 4 counties with known Dem shares and precinct variance."""
    return pd.DataFrame({
        "county_fips": ["01001", "01003", "01005", "01007"],
        "n_precincts": [5, 5, 5, 5],
        "total_votes": [1000, 1000, 1000, 1000],
        "vote_weighted_dem_share": [0.60, 0.30, 0.65, 0.25],
        "precinct_variance": [0.001, 0.001, 0.002, 0.002],
        "precinct_std": [0.032, 0.032, 0.045, 0.045],
    })


@pytest.fixture(scope="module")
def type_assignments_simple():
    """4 counties: 01001+01003 in type 0, 01005+01007 in type 1.

    Type 0: high Dem (0.60) + low Dem (0.30) — mixed type
    Type 1: high Dem (0.65) + low Dem (0.25) — also mixed
    """
    return pd.DataFrame({
        "county_fips": ["01001", "01003", "01005", "01007"],
        "dominant_type": [0, 0, 1, 1],
        "super_type": [0, 0, 1, 1],
    })


@pytest.fixture(scope="module")
def type_assignments_well_separated():
    """4 counties assigned to types where both counties in each type are similar.

    Type 0: 01001 (0.60) + 01003 (0.30) — wide spread within type
    This fixture tests that within-type variance is high when types mix Dem/Rep.
    """
    return pd.DataFrame({
        "county_fips": ["01001", "01003", "01005", "01007"],
        "dominant_type": [0, 1, 0, 1],  # Now 01001+01005 are type 0 (both high Dem)
        "super_type": [0, 1, 0, 1],
    })


@pytest.fixture(scope="module")
def type_priors_simple():
    """Type-level predicted Dem share for basic structure tests.

    Types have different priors so pearsonr is computable.
    """
    return pd.DataFrame({
        "type_id": [0, 1],
        "prior_dem_share": [0.55, 0.30],
    })


@pytest.fixture(scope="module")
def type_priors_predictive():
    """Type-level predicted Dem share that actually correlates with county dem_share."""
    return pd.DataFrame({
        "type_id": [0, 1],
        "prior_dem_share": [0.60, 0.25],  # Type 0 ~ high Dem, Type 1 ~ low Dem
    })


@pytest.fixture(scope="module")
def large_county_stats():
    """20 counties with known Dem shares for correlation tests.

    We need >= 10 counties to satisfy the pearsonr guard in
    compute_type_prediction_correlation.

    Types alternate: even FIPS -> type 0 (high Dem), odd FIPS -> type 1 (low Dem).
    """
    rng = np.random.default_rng(42)
    fips = [f"{i:05d}" for i in range(1, 21)]
    dem_shares = []
    for i, f in enumerate(fips):
        # Type 0 counties (even index): 0.55-0.75; Type 1 (odd): 0.20-0.35
        if i % 2 == 0:
            dem_shares.append(0.65 + rng.uniform(-0.05, 0.05))
        else:
            dem_shares.append(0.25 + rng.uniform(-0.05, 0.05))
    return pd.DataFrame({
        "county_fips": fips,
        "n_precincts": [5] * 20,
        "total_votes": [1000] * 20,
        "vote_weighted_dem_share": dem_shares,
        "precinct_variance": [0.001] * 20,
        "precinct_std": [0.032] * 20,
    })


@pytest.fixture(scope="module")
def large_type_assignments():
    """Type assignments for large_county_stats: even index -> type 0, odd -> type 1."""
    fips = [f"{i:05d}" for i in range(1, 21)]
    dominant = [0 if i % 2 == 0 else 1 for i in range(20)]
    return pd.DataFrame({
        "county_fips": fips,
        "dominant_type": dominant,
        "super_type": dominant,
    })


@pytest.fixture(scope="module")
def large_type_priors_predictive():
    """Priors that match the large_county_stats partisan structure."""
    return pd.DataFrame({
        "type_id": [0, 1],
        "prior_dem_share": [0.65, 0.25],
    })


# ── Tests: compute_county_precinct_stats ─────────────────────────────────────


class TestComputeCountyPrecinctStats:
    def test_returns_one_row_per_county(self, simple_precinct_df):
        """Each county in input should produce one output row."""
        result = compute_county_precinct_stats(simple_precinct_df, min_precinct_votes=0, min_precincts_per_county=2)
        assert len(result) == 2
        assert set(result["county_fips"]) == {"01001", "01003"}

    def test_vote_weighted_dem_share_correct(self, simple_precinct_df):
        """Vote-weighted Dem share should match total_dem / total_votes."""
        result = compute_county_precinct_stats(simple_precinct_df, min_precinct_votes=0, min_precincts_per_county=2)
        county_a = result[result["county_fips"] == "01001"].iloc[0]
        # 5 precincts: 120, 121, 122, 123, 124 Dem votes out of 200 each
        expected = (120 + 121 + 122 + 123 + 124) / (5 * 200)
        assert abs(county_a["vote_weighted_dem_share"] - expected) < 1e-9

    def test_min_vote_threshold_filters_small_precincts(self):
        """Precincts below min_precinct_votes are excluded."""
        df = pd.DataFrame([
            {"county_fips": "01001", "precinct_geoid": "p1", "votes_total": 10, "votes_dem": 6, "dem_share": 0.6},
            {"county_fips": "01001", "precinct_geoid": "p2", "votes_total": 10, "votes_dem": 6, "dem_share": 0.6},
            {"county_fips": "01003", "precinct_geoid": "p3", "votes_total": 200, "votes_dem": 120, "dem_share": 0.6},
            {"county_fips": "01003", "precinct_geoid": "p4", "votes_total": 200, "votes_dem": 60, "dem_share": 0.3},
        ])
        # With threshold 50, county 01001's precincts (10 votes) get filtered out
        result = compute_county_precinct_stats(df, min_precinct_votes=50, min_precincts_per_county=2)
        # 01001 has no precincts passing threshold, so only 01003 survives
        assert "01001" not in result["county_fips"].values
        assert "01003" in result["county_fips"].values

    def test_min_precincts_threshold_filters_counties(self, simple_precinct_df):
        """Counties with fewer than min_precincts_per_county precincts are excluded."""
        # Require 10 precincts — neither county qualifies
        result = compute_county_precinct_stats(simple_precinct_df, min_precinct_votes=0, min_precincts_per_county=10)
        assert len(result) == 0

    def test_precinct_variance_nonnegative(self, simple_precinct_df):
        """Precinct variance must be non-negative."""
        result = compute_county_precinct_stats(simple_precinct_df, min_precinct_votes=0, min_precincts_per_county=2)
        assert (result["precinct_variance"] >= 0).all()

    def test_precinct_std_nonnegative(self, simple_precinct_df):
        """Precinct std must be non-negative."""
        result = compute_county_precinct_stats(simple_precinct_df, min_precinct_votes=0, min_precincts_per_county=2)
        assert (result["precinct_std"] >= 0).all()

    def test_n_precincts_count_correct(self, simple_precinct_df):
        """n_precincts should equal the number of precincts per county."""
        result = compute_county_precinct_stats(simple_precinct_df, min_precinct_votes=0, min_precincts_per_county=2)
        for _, row in result.iterrows():
            assert row["n_precincts"] == 5


# ── Tests: compute_within_type_variance ──────────────────────────────────────


class TestComputeWithinTypeVariance:
    def test_returns_dict_with_required_keys(self, county_stats_simple, type_assignments_simple):
        """Result must contain all expected keys."""
        result = compute_within_type_variance(
            county_stats_simple, type_assignments_simple, min_counties_per_type=2
        )
        for key in ("within_type_variance", "between_type_variance", "f_ratio", "n_types_included", "type_stats"):
            assert key in result

    def test_n_types_included(self, county_stats_simple, type_assignments_simple):
        """n_types_included should equal number of types with enough counties."""
        result = compute_within_type_variance(
            county_stats_simple, type_assignments_simple, min_counties_per_type=2
        )
        assert result["n_types_included"] == 2

    def test_f_ratio_nonnegative(self, county_stats_simple, type_assignments_simple):
        """F-ratio must be non-negative."""
        result = compute_within_type_variance(
            county_stats_simple, type_assignments_simple, min_counties_per_type=2
        )
        assert result["f_ratio"] >= 0.0

    def test_f_ratio_high_for_well_separated_types(self):
        """When types are perfectly partisan-separated, F-ratio should be >> 1."""
        # County 01001 (Dem 0.70), 01003 (Dem 0.72) -- tight cluster type 0
        # County 01005 (Dem 0.20), 01007 (Dem 0.22) -- tight cluster type 1
        county_stats = pd.DataFrame({
            "county_fips": ["01001", "01003", "01005", "01007"],
            "n_precincts": [5, 5, 5, 5],
            "total_votes": [1000, 1000, 1000, 1000],
            "vote_weighted_dem_share": [0.70, 0.72, 0.20, 0.22],
            "precinct_variance": [0.001, 0.001, 0.001, 0.001],
            "precinct_std": [0.032, 0.032, 0.032, 0.032],
        })
        type_assignments = pd.DataFrame({
            "county_fips": ["01001", "01003", "01005", "01007"],
            "dominant_type": [0, 0, 1, 1],
            "super_type": [0, 1, 0, 1],
        })
        result = compute_within_type_variance(county_stats, type_assignments, min_counties_per_type=2)
        # Within-type variance ~0 (both counties in each type are similar)
        # Between-type variance ~0.25 (types are 0.71 vs 0.21)
        # F >> 1
        assert result["f_ratio"] > 10.0, f"Expected high F-ratio for separated types, got {result['f_ratio']:.2f}"

    def test_type_stats_list_length(self, county_stats_simple, type_assignments_simple):
        """type_stats should have one entry per included type."""
        result = compute_within_type_variance(
            county_stats_simple, type_assignments_simple, min_counties_per_type=2
        )
        assert len(result["type_stats"]) == result["n_types_included"]

    def test_empty_result_on_no_qualified_types(self, county_stats_simple, type_assignments_simple):
        """When min_counties_per_type is too high, result should gracefully handle empty."""
        result = compute_within_type_variance(
            county_stats_simple, type_assignments_simple, min_counties_per_type=100
        )
        assert result["n_types_included"] == 0
        assert result["type_stats"] == []


# ── Tests: compute_type_prediction_correlation ────────────────────────────────


class TestComputeTypePredictionCorrelation:
    def test_returns_dict_with_required_keys(
        self, county_stats_simple, type_assignments_simple, type_priors_simple
    ):
        """Result must contain all expected keys."""
        result = compute_type_prediction_correlation(
            county_stats_simple, type_assignments_simple, type_priors_simple
        )
        for key in ("pearson_r", "p_value", "n_counties", "rmse", "median_abs_error"):
            assert key in result

    def test_pearson_r_bounded(
        self, large_county_stats, large_type_assignments, large_type_priors_predictive
    ):
        """Pearson r must be in [-1, 1].

        Uses the large fixture (20 counties) to clear the 10-county minimum threshold.
        """
        result = compute_type_prediction_correlation(
            large_county_stats, large_type_assignments, large_type_priors_predictive
        )
        assert -1.0 <= result["pearson_r"] <= 1.0

    def test_rmse_nonnegative(
        self, large_county_stats, large_type_assignments, large_type_priors_predictive
    ):
        """RMSE must be non-negative."""
        result = compute_type_prediction_correlation(
            large_county_stats, large_type_assignments, large_type_priors_predictive
        )
        assert result["rmse"] >= 0.0

    def test_high_r_for_predictive_priors(
        self, large_county_stats, large_type_assignments, large_type_priors_predictive
    ):
        """When priors track the partisan split, r should be significantly positive.

        large_county_stats: type 0 counties ~0.65 Dem, type 1 counties ~0.25 Dem
        large_type_priors_predictive: type 0 -> 0.65, type 1 -> 0.25

        The priors capture the direction of the partisan split, so r should be high.
        """
        result = compute_type_prediction_correlation(
            large_county_stats, large_type_assignments, large_type_priors_predictive
        )
        assert result["pearson_r"] > 0.7, (
            f"Expected high r for predictive priors, got {result['pearson_r']:.3f}"
        )

    def test_n_counties_matches_joined_rows(
        self, county_stats_simple, type_assignments_simple, type_priors_simple
    ):
        """n_counties should equal the number of rows after joining all three datasets."""
        result = compute_type_prediction_correlation(
            county_stats_simple, type_assignments_simple, type_priors_simple
        )
        assert result["n_counties"] == 4  # all 4 counties in simple fixture


# ── Tests: compute_precinct_variance_by_type ─────────────────────────────────


class TestComputePrecinctVarianceByType:
    def test_returns_dict_with_required_keys(self, county_stats_simple, type_assignments_simple):
        """Result must contain expected keys."""
        result = compute_precinct_variance_by_type(
            county_stats_simple, type_assignments_simple, min_counties_per_type=2
        )
        for key in (
            "overall_mean_precinct_std",
            "overall_median_precinct_std",
            "n_counties_analyzed",
            "most_heterogeneous_types",
            "most_homogeneous_types",
        ):
            assert key in result

    def test_overall_mean_precinct_std_nonnegative(
        self, county_stats_simple, type_assignments_simple
    ):
        """Mean precinct std must be non-negative."""
        result = compute_precinct_variance_by_type(county_stats_simple, type_assignments_simple)
        assert result["overall_mean_precinct_std"] >= 0.0

    def test_n_counties_analyzed_correct(
        self, county_stats_simple, type_assignments_simple
    ):
        """n_counties_analyzed should equal counties matched to type assignments."""
        result = compute_precinct_variance_by_type(county_stats_simple, type_assignments_simple)
        assert result["n_counties_analyzed"] == 4


# ── Tests: compute_super_type_summary ────────────────────────────────────────


class TestComputeSuperTypeSummary:
    def test_returns_list(self, county_stats_simple, type_assignments_simple):
        """Result should be a list of dicts."""
        result = compute_super_type_summary(county_stats_simple, type_assignments_simple)
        assert isinstance(result, list)

    def test_one_record_per_super_type(self, county_stats_simple, type_assignments_simple):
        """There should be one entry per unique super_type."""
        result = compute_super_type_summary(county_stats_simple, type_assignments_simple)
        assert len(result) == 2  # super_types 0 and 1 in fixture

    def test_required_keys_present(self, county_stats_simple, type_assignments_simple):
        """Each record must have expected keys."""
        result = compute_super_type_summary(county_stats_simple, type_assignments_simple)
        for record in result:
            for key in ("super_type", "n_counties", "mean_dem_share", "std_dem_share", "mean_precinct_std"):
                assert key in record, f"Missing key: {key}"

    def test_sorted_by_super_type(self, county_stats_simple, type_assignments_simple):
        """Output should be sorted ascending by super_type."""
        result = compute_super_type_summary(county_stats_simple, type_assignments_simple)
        super_types = [r["super_type"] for r in result]
        assert super_types == sorted(super_types)
