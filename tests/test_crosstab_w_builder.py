"""Tests for crosstab-adjusted W vector construction.

Covers:
  - CROSSTAB_DIMENSION_MAP completeness vs type_profiles columns
  - build_affinity_index correctness (keys, inversion, population weighting)
  - compute_state_baseline_w normalisation and population weighting
  - construct_w_row: valid simplex output, crosstab signal direction,
    fallback, numerical stability at J=100
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.propagation.crosstab_w_builder import (
    ADJUSTMENT_STRENGTH_DEFAULT,
    CROSSTAB_DIMENSION_MAP,
    build_affinity_index,
    compute_state_baseline_w,
    construct_w_row,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_type_profiles(J: int = 3, rng_seed: int = 0) -> pd.DataFrame:
    """Minimal synthetic type_profiles with the columns referenced by CROSSTAB_DIMENSION_MAP."""
    rng = np.random.default_rng(rng_seed)
    return pd.DataFrame({
        "type_id":           np.arange(J),
        "pop_total":         rng.integers(50_000, 500_000, size=J).astype(float),
        "pct_bachelors_plus": rng.uniform(0.15, 0.65, size=J),
        "pct_white_nh":      rng.uniform(0.30, 0.90, size=J),
        "pct_black":         rng.uniform(0.02, 0.40, size=J),
        "pct_hispanic":      rng.uniform(0.02, 0.45, size=J),
        "pct_asian":         rng.uniform(0.01, 0.20, size=J),
        "log_pop_density":   rng.uniform(1.5, 4.5, size=J),
        "median_age":        rng.uniform(30.0, 55.0, size=J),
        "evangelical_share": rng.uniform(0.05, 0.80, size=J),
    })


def _make_county_demographics(N: int = 20, rng_seed: int = 1) -> pd.DataFrame:
    """Minimal synthetic county demographics with pop_total and all needed features."""
    rng = np.random.default_rng(rng_seed)
    return pd.DataFrame({
        "county_fips":       [f"{i:05d}" for i in range(N)],
        "pop_total":         rng.integers(5_000, 800_000, size=N).astype(float),
        "pct_bachelors_plus": rng.uniform(0.10, 0.60, size=N),
        "pct_white_nh":      rng.uniform(0.20, 0.95, size=N),
        "pct_black":         rng.uniform(0.01, 0.50, size=N),
        "pct_hispanic":      rng.uniform(0.01, 0.50, size=N),
        "pct_asian":         rng.uniform(0.01, 0.20, size=N),
        "log_pop_density":   rng.uniform(1.0, 5.5, size=N),
        "median_age":        rng.uniform(28.0, 60.0, size=N),
        "evangelical_share": rng.uniform(0.03, 0.85, size=N),
    })


# ---------------------------------------------------------------------------
# 1. CROSSTAB_DIMENSION_MAP vs type_profiles columns
# ---------------------------------------------------------------------------

class TestCrosstabDimensionMap:
    """Verify the dimension map is consistent with real type_profiles columns."""

    def test_direct_columns_in_type_profiles(self):
        """All non-None mapped features must exist in actual type_profiles."""
        # Load the real parquet to guard against column renames in the pipeline.
        # If the file is not on this machine this test is skipped gracefully.
        try:
            tp = pd.read_parquet("data/communities/type_profiles.parquet")
        except FileNotFoundError:
            pytest.skip("type_profiles.parquet not present on this machine")

        missing = [
            col for col in CROSSTAB_DIMENSION_MAP.values()
            if col is not None and col not in tp.columns
        ]
        assert missing == [], (
            f"Columns in CROSSTAB_DIMENSION_MAP missing from type_profiles: {missing}"
        )

    def test_all_keys_unique(self):
        """Dimension keys must be unique (no silent overwrites)."""
        keys = list(CROSSTAB_DIMENSION_MAP.keys())
        assert len(keys) == len(set(keys)), "Duplicate keys in CROSSTAB_DIMENSION_MAP"

    def test_inverted_dimensions_have_none_value(self):
        """Inverted dims (education_noncollege, urbanicity_rural) must map to None."""
        assert CROSSTAB_DIMENSION_MAP["education_noncollege"] is None
        assert CROSSTAB_DIMENSION_MAP["urbanicity_rural"] is None

    def test_synthetic_type_profiles_has_required_columns(self):
        """Synthetic helper for other tests must have all non-None features."""
        tp = _make_type_profiles(J=5)
        for col in CROSSTAB_DIMENSION_MAP.values():
            if col is not None:
                assert col in tp.columns, f"Synthetic type_profiles missing '{col}'"


# ---------------------------------------------------------------------------
# 2. build_affinity_index
# ---------------------------------------------------------------------------

class TestBuildAffinityIndex:
    """Unit tests for affinity index construction."""

    def _build(self, J: int = 3) -> dict[str, np.ndarray]:
        return build_affinity_index(
            _make_type_profiles(J=J),
            _make_county_demographics(N=20),
        )

    def test_output_keys_match_dimension_map(self):
        """Every key in CROSSTAB_DIMENSION_MAP should appear in the output."""
        affinity = self._build()
        assert set(affinity.keys()) == set(CROSSTAB_DIMENSION_MAP.keys())

    def test_output_shapes(self):
        """Each affinity vector must have shape (J,)."""
        J = 5
        affinity = build_affinity_index(
            _make_type_profiles(J=J),
            _make_county_demographics(N=30),
        )
        for key, vec in affinity.items():
            assert vec.shape == (J,), f"Affinity for '{key}' has wrong shape {vec.shape}"

    def test_urbanicity_rural_is_negation_of_urban(self):
        """urbanicity_rural must be the exact negation of urbanicity_urban."""
        affinity = self._build()
        np.testing.assert_array_equal(
            affinity["urbanicity_rural"],
            -affinity["urbanicity_urban"],
        )

    def test_education_noncollege_is_negation_of_college(self):
        """education_noncollege must be the exact negation of education_college."""
        affinity = self._build()
        np.testing.assert_array_equal(
            affinity["education_noncollege"],
            -affinity["education_college"],
        )

    def test_affinity_is_zero_when_all_types_at_national_mean(self):
        """If every type has the national mean value, affinity should be ~0."""
        J = 4
        # Construct county_demographics with a known national mean for pct_bachelors_plus
        # then set all type_profiles values to that same national mean.
        county_demo = _make_county_demographics(N=10)
        national_mean_educ = float(
            np.average(
                county_demo["pct_bachelors_plus"],
                weights=county_demo["pop_total"],
            )
        )

        tp = _make_type_profiles(J=J)
        tp["pct_bachelors_plus"] = national_mean_educ  # all types equal national mean

        affinity = build_affinity_index(tp, county_demo)
        np.testing.assert_allclose(
            affinity["education_college"],
            np.zeros(J),
            atol=1e-10,
        )

    def test_population_weighting_matters(self):
        """National mean must be pop-weighted, not a simple average."""
        # Two counties: one tiny, one huge; different feature values
        county_demo = pd.DataFrame({
            "pop_total":         [100.0, 1_000_000.0],
            "pct_bachelors_plus": [0.80,  0.20],
            "pct_white_nh":       [0.5,   0.5],
            "pct_black":          [0.1,   0.1],
            "pct_hispanic":       [0.1,   0.1],
            "pct_asian":          [0.1,   0.1],
            "log_pop_density":    [2.0,   2.0],
            "median_age":         [40.0,  40.0],
            "evangelical_share":  [0.3,   0.3],
        })
        tp = _make_type_profiles(J=2)
        # Set type values so we can predict direction of affinity
        tp["pct_bachelors_plus"] = [0.5, 0.5]

        affinity = build_affinity_index(tp, county_demo)

        # Pop-weighted mean ≈ 0.20 (large county dominates)
        # So type value 0.5 > 0.20 → positive college affinity
        assert (affinity["education_college"] > 0).all(), (
            "Expected positive college affinity when types sit above pop-weighted mean"
        )

    def test_raises_on_missing_pop_total(self):
        """county_demographics without pop_total should raise ValueError."""
        tp = _make_type_profiles(J=3)
        county_demo = _make_county_demographics(N=5).drop(columns=["pop_total"])
        with pytest.raises(ValueError, match="pop_total"):
            build_affinity_index(tp, county_demo)

    def test_raises_on_missing_feature_column(self):
        """Missing feature in county_demographics should raise KeyError."""
        tp = _make_type_profiles(J=3)
        county_demo = _make_county_demographics(N=5).drop(columns=["evangelical_share"])
        with pytest.raises(KeyError):
            build_affinity_index(tp, county_demo)


# ---------------------------------------------------------------------------
# 3. compute_state_baseline_w
# ---------------------------------------------------------------------------

class TestComputeStateBaselineW:
    """Unit tests for state baseline W construction."""

    def _make_inputs(
        self,
        N: int = 10,
        J: int = 4,
        rng_seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(rng_seed)
        type_scores = rng.uniform(0.0, 1.0, size=(N, J))
        county_populations = rng.integers(10_000, 500_000, size=N).astype(float)
        state_mask = np.zeros(N, dtype=bool)
        state_mask[:N // 2] = True  # first half belongs to the state
        return type_scores, county_populations, state_mask

    def test_output_sums_to_one(self):
        """W must be a probability vector (sum = 1)."""
        ts, pops, mask = self._make_inputs()
        w = compute_state_baseline_w(ts, pops, mask)
        assert abs(w.sum() - 1.0) < 1e-9, f"W sum is {w.sum()}, expected 1.0"

    def test_output_shape(self):
        """Output shape must match J."""
        J = 7
        N = 15
        rng = np.random.default_rng(0)
        ts = rng.uniform(0.0, 1.0, size=(N, J))
        pops = rng.integers(1_000, 100_000, size=N).astype(float)
        mask = np.ones(N, dtype=bool)
        w = compute_state_baseline_w(ts, pops, mask)
        assert w.shape == (J,)

    def test_population_weighting_direction(self):
        """A county with 10× the population should dominate the W vector."""
        J = 2
        # County 0: tiny, all weight on type 0
        # County 1: huge, all weight on type 1
        type_scores = np.array([[1.0, 0.0], [0.0, 1.0]])
        county_populations = np.array([100.0, 100_000.0])
        state_mask = np.array([True, True])
        w = compute_state_baseline_w(type_scores, county_populations, state_mask)
        # Large county dominates → type 1 should have much more weight
        assert w[1] > w[0], "Large county should dominate the type weight"

    def test_uniform_scores_give_uniform_w(self):
        """If all counties have identical type scores, W equals normalised score."""
        J = 3
        N = 5
        # All counties have same type scores
        score_row = np.array([2.0, 1.0, 3.0])
        type_scores = np.tile(score_row, (N, 1))
        county_populations = np.ones(N) * 1000.0
        state_mask = np.ones(N, dtype=bool)
        w = compute_state_baseline_w(type_scores, county_populations, state_mask)
        expected = score_row / score_row.sum()
        np.testing.assert_allclose(w, expected, atol=1e-9)

    def test_raises_on_zero_state_population(self):
        """Zero population for state counties should raise ValueError."""
        ts = np.ones((5, 3))
        pops = np.zeros(5)
        mask = np.ones(5, dtype=bool)
        with pytest.raises(ValueError, match="zero total population"):
            compute_state_baseline_w(ts, pops, mask)


# ---------------------------------------------------------------------------
# 4. construct_w_row
# ---------------------------------------------------------------------------

class TestConstructWRow:
    """Unit tests for crosstab-adjusted W vector construction."""

    def _setup(self, J: int = 10) -> tuple[np.ndarray, dict, dict]:
        """Return (state_baseline_w, affinity_index, state_demographic_means)."""
        rng = np.random.default_rng(7)

        # Synthetic baseline W (uniform over J types for simplicity)
        w_base = np.ones(J) / J

        # Synthetic affinity: education_college varies from −1 to +1 across types
        college_affinity = np.linspace(-1.0, 1.0, J)
        affinity_index = {
            "education_college":    college_affinity,
            "education_noncollege": -college_affinity,
            "race_white":           rng.uniform(-0.2, 0.2, size=J),
            "race_black":           rng.uniform(-0.2, 0.2, size=J),
            "race_hispanic":        rng.uniform(-0.2, 0.2, size=J),
            "race_asian":           rng.uniform(-0.2, 0.2, size=J),
            "urbanicity_urban":     rng.uniform(-0.5, 0.5, size=J),
            "urbanicity_rural":     rng.uniform(-0.5, 0.5, size=J),
            "age_senior":           rng.uniform(-0.3, 0.3, size=J),
            "religion_evangelical": rng.uniform(-0.3, 0.3, size=J),
        }

        # State means: 35% college (realistic US state average)
        state_demo_means = {
            "education_college":    0.35,
            "education_noncollege": 0.65,
            "race_white":           0.70,
            "race_black":           0.12,
            "race_hispanic":        0.10,
            "race_asian":           0.04,
            "urbanicity_urban":     3.0,
            "urbanicity_rural":     0.30,
            "age_senior":           38.5,
            "religion_evangelical": 0.35,
        }

        return w_base, affinity_index, state_demo_means

    def test_output_sums_to_one(self):
        """W must be a probability vector regardless of crosstab contents."""
        w_base, affinity, means = self._setup(J=10)
        crosstabs = [
            {"demographic_group": "education", "group_value": "college", "pct_of_sample": 0.55},
        ]
        w = construct_w_row(crosstabs, w_base, affinity, means)
        assert abs(w.sum() - 1.0) < 1e-9

    def test_no_negative_entries(self):
        """W must have no negative entries after adjustment."""
        w_base, affinity, means = self._setup(J=20)
        crosstabs = [
            {"demographic_group": "education", "group_value": "college", "pct_of_sample": 0.90},
            {"demographic_group": "race", "group_value": "white", "pct_of_sample": 0.95},
        ]
        w = construct_w_row(crosstabs, w_base, affinity, means)
        assert (w >= 0.0).all(), "W contains negative entries"

    def test_college_oversample_increases_college_heavy_types(self):
        """Poll with high college % should shift weight toward high-college types.

        Construction: affinity[education_college] = linspace(-1, +1, J).
        High-J types are college-heavy.  pct_of_sample=0.55 > state_mean=0.35
        → positive delta → W[high-J] should increase relative to baseline.
        """
        J = 10
        w_base, affinity, means = self._setup(J=J)
        crosstabs = [
            {"demographic_group": "education", "group_value": "college", "pct_of_sample": 0.55},
        ]
        w = construct_w_row(crosstabs, w_base, affinity, means)

        # High-college types are the top-half (indices J//2 .. J-1 since affinity
        # was linspace(-1, +1)).
        low_half = slice(0, J // 2)
        high_half = slice(J // 2, J)

        # With oversample of college, high-half types should gain weight
        assert w[high_half].sum() > w_base[high_half].sum(), (
            "College-oversampled poll should increase weight on college-heavy types"
        )
        assert w[low_half].sum() < w_base[low_half].sum(), (
            "College-oversampled poll should decrease weight on non-college types"
        )

    def test_empty_crosstabs_returns_baseline(self):
        """Empty crosstabs list must return a copy of state_baseline_w."""
        w_base, affinity, means = self._setup(J=8)
        w = construct_w_row([], w_base, affinity, means)
        np.testing.assert_array_equal(w, w_base)
        # Must be a copy, not the same object (caller may modify it)
        assert w is not w_base

    def test_crosstabs_with_all_nulls_returns_baseline(self):
        """Crosstabs where all pct_of_sample are None must fall back to baseline."""
        w_base, affinity, means = self._setup(J=6)
        crosstabs = [
            {"demographic_group": "education", "group_value": "college", "pct_of_sample": None},
            {"demographic_group": "race",      "group_value": "white",   "pct_of_sample": None},
        ]
        w = construct_w_row(crosstabs, w_base, affinity, means)
        np.testing.assert_array_equal(w, w_base)

    def test_zero_delta_returns_baseline(self):
        """When poll % equals state mean, delta=0 and W should equal baseline."""
        w_base, affinity, means = self._setup(J=10)
        # pct_of_sample == state_demographic_mean → delta = 0
        crosstabs = [
            {
                "demographic_group": "education",
                "group_value": "college",
                "pct_of_sample": means["education_college"],  # exactly 0.35
            }
        ]
        w = construct_w_row(crosstabs, w_base, affinity, means)
        np.testing.assert_allclose(w, w_base, atol=1e-12)

    def test_unknown_dimension_key_is_skipped(self):
        """Crosstab rows with dimension_key not in affinity_index must be ignored."""
        w_base, affinity, means = self._setup(J=10)
        crosstabs = [
            {
                "demographic_group": "totally_unknown",
                "group_value": "dimension",
                "pct_of_sample": 0.50,
            }
        ]
        # Should not raise; should return baseline (nothing usable)
        w = construct_w_row(crosstabs, w_base, affinity, means)
        np.testing.assert_array_equal(w, w_base)

    def test_extreme_delta_produces_valid_w(self):
        """Extreme pct_of_sample=1.0 must still yield a valid W (no negatives, sums to 1)."""
        w_base, affinity, means = self._setup(J=20)
        crosstabs = [
            {"demographic_group": "education", "group_value": "college", "pct_of_sample": 1.0},
        ]
        w = construct_w_row(crosstabs, w_base, affinity, means)
        assert abs(w.sum() - 1.0) < 1e-9
        assert (w >= 0.0).all()

    def test_j100_numerical_stability(self):
        """W must be valid at production scale J=100."""
        J = 100
        rng = np.random.default_rng(123)

        w_base = rng.dirichlet(alpha=np.ones(J))
        college_affinity = rng.uniform(-0.3, 0.3, size=J)
        affinity_index = {"education_college": college_affinity}
        state_means = {"education_college": 0.38}

        crosstabs = [
            {"demographic_group": "education", "group_value": "college", "pct_of_sample": 0.52},
        ]

        w = construct_w_row(
            crosstabs,
            w_base,
            affinity_index,
            state_means,
            adjustment_strength=ADJUSTMENT_STRENGTH_DEFAULT,
        )

        assert w.shape == (J,)
        assert abs(w.sum() - 1.0) < 1e-9
        assert (w >= 0.0).all()
        assert not np.any(np.isnan(w)), "NaN values in W"
        assert not np.any(np.isinf(w)), "Inf values in W"

    def test_adjustment_strength_zero_gives_baseline(self):
        """With adjustment_strength=0, crosstab signal is completely suppressed."""
        w_base, affinity, means = self._setup(J=10)
        crosstabs = [
            {"demographic_group": "education", "group_value": "college", "pct_of_sample": 0.70},
        ]
        w = construct_w_row(crosstabs, w_base, affinity, means, adjustment_strength=0.0)
        np.testing.assert_allclose(w, w_base, atol=1e-12)

    def test_missing_dimension_in_state_means_skips_row(self):
        """Crosstab row with dim_key absent from state_demographic_means must be skipped."""
        w_base, affinity, means = self._setup(J=10)
        # Remove education_college from state means
        means_partial = {k: v for k, v in means.items() if k != "education_college"}
        crosstabs = [
            {"demographic_group": "education", "group_value": "college", "pct_of_sample": 0.55},
        ]
        # Only usable crosstab is skipped → should fall back to baseline
        w = construct_w_row(crosstabs, w_base, affinity, means_partial)
        np.testing.assert_array_equal(w, w_base)
