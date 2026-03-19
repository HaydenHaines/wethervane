"""Tests for community description (census overlay)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.description.describe_communities import (
    build_community_profiles,
    DEMOGRAPHIC_COLS,
)


@pytest.fixture
def assignments():
    return pd.DataFrame({
        "tract_geoid": ["12001000100", "12001000200", "01001000100"],
        "community_id": [0, 0, 1],
    })


@pytest.fixture
def features():
    return pd.DataFrame({
        "tract_geoid": ["12001000100", "12001000200", "01001000100"],
        "pop_total": [5000, 3000, 2000],  # unequal for weighted mean testing
        "pct_white_nh": [0.60, 0.65, 0.80],
        "pct_black": [0.25, 0.20, 0.10],
        "pct_asian": [0.05, 0.05, 0.02],
        "pct_hispanic": [0.10, 0.10, 0.08],
        "log_median_income": [10.5, 10.8, 10.0],
        "pct_mgmt_occ": [0.30, 0.35, 0.20],
        "pct_owner_occ": [0.55, 0.60, 0.70],
        "pct_car_commute": [0.70, 0.65, 0.85],
        "pct_transit_commute": [0.15, 0.20, 0.02],
        "pct_wfh_commute": [0.10, 0.10, 0.08],
        "pct_college_plus": [0.40, 0.45, 0.20],
        "median_age": [35.0, 38.0, 42.0],
    })


@pytest.fixture
def shifts():
    return pd.DataFrame({
        "tract_geoid": ["12001000100", "12001000200", "01001000100"],
        "pres_d_shift_16_20": [0.02, 0.03, -0.02],
        "pres_d_shift_20_24": [-0.01, -0.02, -0.05],
    })


class TestBuildCommunityProfiles:
    def test_one_row_per_community(self, assignments, features, shifts):
        result = build_community_profiles(assignments, features, shifts)
        assert len(result) == 2  # community 0 and 1

    def test_has_demographic_columns(self, assignments, features, shifts):
        result = build_community_profiles(assignments, features, shifts)
        for col in DEMOGRAPHIC_COLS:
            assert col in result.columns

    def test_has_community_id(self, assignments, features, shifts):
        result = build_community_profiles(assignments, features, shifts)
        assert "community_id" in result.columns

    def test_has_tract_count(self, assignments, features, shifts):
        result = build_community_profiles(assignments, features, shifts)
        assert "n_tracts" in result.columns
        assert result.loc[result.community_id == 0, "n_tracts"].iloc[0] == 2

    def test_demographics_are_population_weighted(self, assignments, features, shifts):
        result = build_community_profiles(assignments, features, shifts)
        # Community 0 = tracts 12001000100 (pop 5000) + 12001000200 (pop 3000)
        # Population-weighted pct_white_nh = (0.60*5000 + 0.65*3000) / 8000 = 0.61875
        comm0 = result.loc[result.community_id == 0]
        expected = (0.60 * 5000 + 0.65 * 3000) / 8000
        assert abs(comm0["pct_white_nh"].iloc[0] - expected) < 1e-9


from src.description.compare_to_nmf import (
    nmf_hard_assignment,
    within_community_variance,
    random_spatial_variance,
)


class TestNmfHardAssignment:
    def test_assigns_to_max_weight(self):
        weights = pd.DataFrame({
            "tract_geoid": ["12001000100", "12001000200"],
            "c0": [0.7, 0.1],
            "c1": [0.2, 0.8],
            "c2": [0.1, 0.1],
        })
        result = nmf_hard_assignment(weights, component_cols=["c0", "c1", "c2"])
        assert result.loc[result.tract_geoid == "12001000100", "nmf_community"].iloc[0] == 0
        assert result.loc[result.tract_geoid == "12001000200", "nmf_community"].iloc[0] == 1


class TestWithinCommunityVariance:
    def test_identical_tracts_have_zero_variance(self):
        shifts = np.ones((10, 9))
        labels = np.array([0]*5 + [1]*5)
        var = within_community_variance(shifts, labels)
        assert var == 0.0

    def test_different_tracts_have_positive_variance(self):
        rng = np.random.default_rng(42)
        shifts = rng.normal(size=(100, 9))
        labels = np.array([0]*50 + [1]*50)
        var = within_community_variance(shifts, labels)
        assert var > 0.0

    def test_known_variance(self):
        """Two clusters of 2 points each, hand-calculable variance."""
        shifts = np.array([
            [1.0, 0.0],  # cluster 0
            [3.0, 0.0],  # cluster 0  -> centroid [2,0], distances 1,1, var=1.0
            [0.0, 1.0],  # cluster 1
            [0.0, 3.0],  # cluster 1  -> centroid [0,2], distances 1,1, var=1.0
        ])
        labels = np.array([0, 0, 1, 1])
        var = within_community_variance(shifts, labels)
        # Both clusters have var=1.0, equal size -> weighted mean = 1.0
        assert abs(var - 1.0) < 1e-9


class TestRandomSpatialVariance:
    def test_positive_for_nonuniform_shifts(self):
        from scipy.sparse import csr_matrix
        rng = np.random.default_rng(42)
        shifts = rng.normal(size=(16, 9))
        # Simple chain adjacency for 16 nodes
        rows = list(range(15)) + list(range(1, 16))
        cols = list(range(1, 16)) + list(range(15))
        W = csr_matrix(([1]*30, (rows, cols)), shape=(16, 16))
        var = random_spatial_variance(shifts, W, n_communities=4, n_trials=10)
        assert var > 0.0
