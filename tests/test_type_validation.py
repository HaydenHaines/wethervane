"""Tests for src/validation/validate_types.py

Uses synthetic data with known structure to verify each validation function.
No real data files are needed.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.validation.validate_types import (
    holdout_accuracy,
    holdout_accuracy_county_prior,
    holdout_accuracy_ridge_augmented,
    type_coherence,
    type_stability,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def perfect_type_scores_and_shifts():
    """3 perfectly separated clusters across 6 shift dimensions.

    Counties 0-19 belong to type 0, 20-39 to type 1, 40-59 to type 2.
    Each cluster has a distinct shift profile (mean far from other types).
    """
    rng = np.random.default_rng(0)
    n, d = 60, 9
    # Very distinct cluster centers
    centers = np.array([
        [2.0, 2.0, 2.0, 0.0, 0.0, 0.0, -2.0, -2.0, -2.0],
        [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0],
        [-2.0, -2.0, -2.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
    ])
    labels = np.repeat([0, 1, 2], 20)
    # Small noise so clusters remain clean
    shifts = centers[labels] + rng.standard_normal((n, d)) * 0.05

    # Scores: strong signal for each county's true type, near-zero for others
    scores = np.zeros((n, 3))
    for i, lbl in enumerate(labels):
        scores[i, lbl] = 5.0 + rng.standard_normal() * 0.01

    dominant_types = labels.copy()
    return scores, shifts, dominant_types


@pytest.fixture(scope="module")
def random_type_scores_and_shifts(rng):
    """Random scores and shifts with no coherent structure."""
    n, d, j = 60, 9, 3
    shifts = rng.standard_normal((n, d))
    scores = rng.standard_normal((n, j))
    dominant_types = np.argmax(np.abs(scores), axis=1)
    return scores, shifts, dominant_types


@pytest.fixture(scope="module")
def stable_shift_matrix():
    """300 × 9 shift matrix with strong type structure for stability tests."""
    rng = np.random.default_rng(7)
    n, d = 300, 9
    centers = rng.standard_normal((4, d)) * 3.0
    labels = np.repeat([0, 1, 2, 3], 75)
    shifts = centers[labels] + rng.standard_normal((n, d)) * 0.1
    return shifts


# ── Tests: type_coherence ─────────────────────────────────────────────────────


class TestTypeCoherence:
    def test_coherence_perfect_types(self, perfect_type_scores_and_shifts):
        """Perfectly separated types should produce high coherence ratio (> 0.7)."""
        scores, shifts, dominant_types = perfect_type_scores_and_shifts
        holdout_cols = [6, 7, 8]
        result = type_coherence(scores, shifts, holdout_cols)
        assert result["mean_ratio"] > 0.7, (
            f"Expected high coherence for clear clusters, got {result['mean_ratio']:.3f}"
        )

    def test_coherence_random_types(self, random_type_scores_and_shifts):
        """Random type assignments should produce low coherence ratio (< 0.3)."""
        scores, shifts, dominant_types = random_type_scores_and_shifts
        holdout_cols = [6, 7, 8]
        result = type_coherence(scores, shifts, holdout_cols)
        assert result["mean_ratio"] < 0.3, (
            f"Expected low coherence for random clusters, got {result['mean_ratio']:.3f}"
        )

    def test_coherence_output_fields(self, perfect_type_scores_and_shifts):
        """Result must have mean_ratio and per_dim_ratios."""
        scores, shifts, _ = perfect_type_scores_and_shifts
        holdout_cols = [0, 1, 2]
        result = type_coherence(scores, shifts, holdout_cols)
        assert "mean_ratio" in result
        assert "per_dim_ratios" in result

    def test_coherence_per_dim_count(self, perfect_type_scores_and_shifts):
        """per_dim_ratios must have one entry per holdout column."""
        scores, shifts, _ = perfect_type_scores_and_shifts
        holdout_cols = [0, 1, 2]
        result = type_coherence(scores, shifts, holdout_cols)
        assert len(result["per_dim_ratios"]) == len(holdout_cols)

    def test_coherence_ratio_bounded(self, perfect_type_scores_and_shifts):
        """All ratios must be in [0, 1]."""
        scores, shifts, _ = perfect_type_scores_and_shifts
        holdout_cols = [0, 1, 2]
        result = type_coherence(scores, shifts, holdout_cols)
        assert 0.0 <= result["mean_ratio"] <= 1.0
        for r in result["per_dim_ratios"]:
            assert 0.0 <= r <= 1.0

    def test_coherence_mean_is_mean_of_per_dim(self, perfect_type_scores_and_shifts):
        """mean_ratio must equal the mean of per_dim_ratios."""
        scores, shifts, _ = perfect_type_scores_and_shifts
        holdout_cols = [0, 1, 2]
        result = type_coherence(scores, shifts, holdout_cols)
        expected_mean = float(np.mean(result["per_dim_ratios"]))
        assert abs(result["mean_ratio"] - expected_mean) < 1e-9


# ── Tests: type_stability ─────────────────────────────────────────────────────


class TestTypeStability:
    def test_stability_same_data(self, stable_shift_matrix):
        """Same data for both windows should produce near-zero subspace angle."""
        window_a = [0, 1, 2, 3]
        window_b = [0, 1, 2, 3]
        result = type_stability(stable_shift_matrix, window_a, window_b, j=4)
        assert result["max_angle_degrees"] < 10.0, (
            f"Expected near-zero angle for same data, got {result['max_angle_degrees']:.2f}"
        )

    def test_stability_output_fields(self, stable_shift_matrix):
        """Result must contain max_angle_degrees, mean_angle_degrees, and stable."""
        window_a = [0, 1, 2, 3]
        window_b = [4, 5, 6, 7]
        result = type_stability(stable_shift_matrix, window_a, window_b, j=3)
        assert "max_angle_degrees" in result
        assert "mean_angle_degrees" in result
        assert "stable" in result

    def test_stability_stable_flag_true_when_low_angle(self, stable_shift_matrix):
        """stable should be True when max_angle < 30 degrees."""
        # Same window produces angle ~0, which is < 30
        window_a = [0, 1, 2, 3]
        window_b = [0, 1, 2, 3]
        result = type_stability(stable_shift_matrix, window_a, window_b, j=3)
        assert result["stable"] is True

    def test_stability_stable_definition(self, stable_shift_matrix):
        """stable == (max_angle_degrees < 30)."""
        window_a = [0, 1, 2, 3]
        window_b = [4, 5, 6, 7]
        result = type_stability(stable_shift_matrix, window_a, window_b, j=3)
        expected_stable = result["max_angle_degrees"] < 30.0
        assert result["stable"] == expected_stable

    def test_stability_angles_nonnegative(self, stable_shift_matrix):
        """Subspace angles must be non-negative."""
        window_a = [0, 1, 2, 3]
        window_b = [4, 5, 6, 7]
        result = type_stability(stable_shift_matrix, window_a, window_b, j=3)
        assert result["max_angle_degrees"] >= 0.0
        assert result["mean_angle_degrees"] >= 0.0

    def test_stability_max_geq_mean(self, stable_shift_matrix):
        """max_angle_degrees must be >= mean_angle_degrees."""
        window_a = [0, 1, 2, 3]
        window_b = [4, 5, 6, 7]
        result = type_stability(stable_shift_matrix, window_a, window_b, j=3)
        assert result["max_angle_degrees"] >= result["mean_angle_degrees"] - 1e-9


# ── Tests: holdout_accuracy ───────────────────────────────────────────────────


class TestHoldoutAccuracy:
    def test_holdout_accuracy_perfect(self, perfect_type_scores_and_shifts):
        """Types that perfectly predict holdout shifts should produce r ≈ 1.0."""
        scores, shifts, dominant_types = perfect_type_scores_and_shifts
        holdout_cols = [6, 7, 8]
        result = holdout_accuracy(scores, shifts, holdout_cols, dominant_types)
        assert result["mean_r"] > 0.9, (
            f"Expected high r for perfect types, got {result['mean_r']:.3f}"
        )

    def test_holdout_accuracy_output_fields(self, perfect_type_scores_and_shifts):
        """Result must have mean_r and per_dim_r."""
        scores, shifts, dominant_types = perfect_type_scores_and_shifts
        holdout_cols = [0, 1, 2]
        result = holdout_accuracy(scores, shifts, holdout_cols, dominant_types)
        assert "mean_r" in result
        assert "per_dim_r" in result

    def test_holdout_accuracy_r_bounded(self, random_type_scores_and_shifts):
        """All r values must be in [-1, 1]."""
        scores, shifts, dominant_types = random_type_scores_and_shifts
        holdout_cols = [6, 7, 8]
        result = holdout_accuracy(scores, shifts, holdout_cols, dominant_types)
        assert -1.0 <= result["mean_r"] <= 1.0
        for r in result["per_dim_r"]:
            assert -1.0 <= r <= 1.0

    def test_holdout_accuracy_per_dim_count(self, perfect_type_scores_and_shifts):
        """per_dim_r must have one entry per holdout column."""
        scores, shifts, dominant_types = perfect_type_scores_and_shifts
        holdout_cols = [0, 1, 2]
        result = holdout_accuracy(scores, shifts, holdout_cols, dominant_types)
        assert len(result["per_dim_r"]) == len(holdout_cols)

    def test_holdout_accuracy_mean_is_mean_of_per_dim(
        self, perfect_type_scores_and_shifts
    ):
        """mean_r must equal the mean of per_dim_r."""
        scores, shifts, dominant_types = perfect_type_scores_and_shifts
        holdout_cols = [0, 1, 2]
        result = holdout_accuracy(scores, shifts, holdout_cols, dominant_types)
        expected_mean = float(np.mean(result["per_dim_r"]))
        assert abs(result["mean_r"] - expected_mean) < 1e-9


# ── Tests: holdout_accuracy_county_prior ─────────────────────────────────────


class TestHoldoutAccuracyCountyPrior:
    def test_county_prior_output_fields(self, perfect_type_scores_and_shifts):
        """Result must have mean_r, per_dim_r, mean_rmse, per_dim_rmse."""
        scores, shifts, _ = perfect_type_scores_and_shifts
        training_cols = [0, 1, 2, 3, 4, 5]
        holdout_cols = [6, 7, 8]
        result = holdout_accuracy_county_prior(scores, shifts, training_cols, holdout_cols)
        assert "mean_r" in result
        assert "per_dim_r" in result
        assert "mean_rmse" in result
        assert "per_dim_rmse" in result

    def test_county_prior_per_dim_count(self, perfect_type_scores_and_shifts):
        """per_dim_r and per_dim_rmse must have one entry per holdout column."""
        scores, shifts, _ = perfect_type_scores_and_shifts
        training_cols = [0, 1, 2, 3, 4, 5]
        holdout_cols = [6, 7, 8]
        result = holdout_accuracy_county_prior(scores, shifts, training_cols, holdout_cols)
        assert len(result["per_dim_r"]) == len(holdout_cols)
        assert len(result["per_dim_rmse"]) == len(holdout_cols)

    def test_county_prior_r_bounded(self, random_type_scores_and_shifts):
        """All r values must be in [-1, 1]."""
        scores, shifts, _ = random_type_scores_and_shifts
        training_cols = [0, 1, 2, 3, 4, 5]
        holdout_cols = [6, 7, 8]
        result = holdout_accuracy_county_prior(scores, shifts, training_cols, holdout_cols)
        assert -1.0 <= result["mean_r"] <= 1.0
        for r in result["per_dim_r"]:
            assert -1.0 <= r <= 1.0

    def test_county_prior_rmse_nonnegative(self, random_type_scores_and_shifts):
        """All RMSE values must be non-negative."""
        scores, shifts, _ = random_type_scores_and_shifts
        training_cols = [0, 1, 2, 3, 4, 5]
        holdout_cols = [6, 7, 8]
        result = holdout_accuracy_county_prior(scores, shifts, training_cols, holdout_cols)
        assert result["mean_rmse"] >= 0.0
        for rmse in result["per_dim_rmse"]:
            assert rmse >= 0.0

    def test_county_prior_perfect_types(self, perfect_type_scores_and_shifts):
        """With perfect type structure, county prior should produce high r."""
        scores, shifts, _ = perfect_type_scores_and_shifts
        training_cols = [0, 1, 2, 3, 4, 5]
        holdout_cols = [6, 7, 8]
        result = holdout_accuracy_county_prior(scores, shifts, training_cols, holdout_cols)
        assert result["mean_r"] > 0.9, (
            f"Expected high r for perfect types with county prior, got {result['mean_r']:.3f}"
        )

    def test_county_prior_beats_or_matches_type_mean(self, perfect_type_scores_and_shifts):
        """County prior should produce >= r compared to type-mean approach.

        County priors retain individual county baselines, so they should
        capture more variance than the type-mean-only approach (or equal
        when types are perfectly homogeneous).
        """
        scores, shifts, dominant_types = perfect_type_scores_and_shifts
        training_cols = [0, 1, 2, 3, 4, 5]
        holdout_cols = [6, 7, 8]

        type_mean_result = holdout_accuracy(scores, shifts, holdout_cols, dominant_types)
        county_prior_result = holdout_accuracy_county_prior(
            scores, shifts, training_cols, holdout_cols
        )

        # County prior should be at least as good (within numerical tolerance)
        assert county_prior_result["mean_r"] >= type_mean_result["mean_r"] - 0.05, (
            f"County prior r={county_prior_result['mean_r']:.3f} should be >= "
            f"type mean r={type_mean_result['mean_r']:.3f}"
        )

    def test_county_prior_mean_is_mean_of_per_dim(self, perfect_type_scores_and_shifts):
        """mean_r must equal the mean of per_dim_r."""
        scores, shifts, _ = perfect_type_scores_and_shifts
        training_cols = [0, 1, 2, 3, 4, 5]
        holdout_cols = [6, 7, 8]
        result = holdout_accuracy_county_prior(scores, shifts, training_cols, holdout_cols)
        expected_mean = float(np.mean(result["per_dim_r"]))
        assert abs(result["mean_r"] - expected_mean) < 1e-9


# ── Tests: holdout_accuracy_ridge_augmented ───────────────────────────────────


def _make_demo_parquet(tmp_path, county_fips_list, n_demo_cols=5, seed=0):
    """Write a small synthetic demographics parquet for testing."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    data = {"county_fips": county_fips_list}
    for i in range(n_demo_cols):
        data[f"demo_{i}"] = rng.standard_normal(len(county_fips_list))
    df = pd.DataFrame(data)
    path = tmp_path / "county_features_national.parquet"
    df.to_parquet(path, index=False)
    return str(path)


class TestHoldoutAccuracyRidgeAugmented:
    """Tests for holdout_accuracy_ridge_augmented using synthetic data."""

    @pytest.fixture(scope="class")
    def synthetic_data(self):
        """60 counties, 9 dims, 3 types — same structure as perfect_type fixture."""
        rng = np.random.default_rng(42)
        n, d, j = 60, 9, 3
        centers = np.array([
            [2.0, 2.0, 2.0, 0.0, 0.0, 0.0, -2.0, -2.0, -2.0],
            [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0],
            [-2.0, -2.0, -2.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
        ])
        labels = np.repeat([0, 1, 2], 20)
        shifts = centers[labels] + rng.standard_normal((n, d)) * 0.05

        scores = np.zeros((n, j))
        for i, lbl in enumerate(labels):
            scores[i, lbl] = 5.0 + rng.standard_normal() * 0.01

        # Synthetic FIPS strings: "00001" .. "00060"
        county_fips = np.array([f"{i:05d}" for i in range(1, n + 1)])
        return scores, shifts, county_fips

    def test_runs_without_error(self, synthetic_data, tmp_path):
        """Function completes and returns a dict when demographics file exists."""
        scores, shifts, county_fips = synthetic_data
        demo_path = _make_demo_parquet(tmp_path, county_fips.tolist())
        training_cols = [0, 1, 2, 3, 4, 5]
        holdout_cols = [6, 7, 8]
        result = holdout_accuracy_ridge_augmented(
            scores, shifts, training_cols, holdout_cols,
            county_fips=county_fips,
            demographics_path=demo_path,
        )
        assert result is not None

    def test_output_keys(self, synthetic_data, tmp_path):
        """All expected keys present in result dict."""
        scores, shifts, county_fips = synthetic_data
        demo_path = _make_demo_parquet(tmp_path, county_fips.tolist())
        result = holdout_accuracy_ridge_augmented(
            scores, shifts, [0, 1, 2, 3, 4, 5], [6, 7, 8],
            county_fips=county_fips,
            demographics_path=demo_path,
        )
        for key in ("mean_r", "per_dim_r", "mean_rmse", "per_dim_rmse",
                    "best_alphas", "n_matched_counties", "n_demo_features"):
            assert key in result, f"Missing key: {key}"

    def test_n_matched_and_demo_features(self, synthetic_data, tmp_path):
        """n_matched_counties and n_demo_features reflect the join results."""
        scores, shifts, county_fips = synthetic_data
        n_demo = 7
        demo_path = _make_demo_parquet(tmp_path, county_fips.tolist(), n_demo_cols=n_demo)
        result = holdout_accuracy_ridge_augmented(
            scores, shifts, [0, 1, 2, 3, 4, 5], [6, 7, 8],
            county_fips=county_fips,
            demographics_path=demo_path,
        )
        assert result["n_matched_counties"] == len(county_fips)
        assert result["n_demo_features"] == n_demo

    def test_r_bounded(self, synthetic_data, tmp_path):
        """LOO r values must be in [-1, 1]."""
        scores, shifts, county_fips = synthetic_data
        demo_path = _make_demo_parquet(tmp_path, county_fips.tolist())
        result = holdout_accuracy_ridge_augmented(
            scores, shifts, [0, 1, 2, 3, 4, 5], [6, 7, 8],
            county_fips=county_fips,
            demographics_path=demo_path,
        )
        assert -1.0 <= result["mean_r"] <= 1.0
        for r in result["per_dim_r"]:
            assert -1.0 <= r <= 1.0

    def test_rmse_nonnegative(self, synthetic_data, tmp_path):
        """LOO RMSE must be non-negative."""
        scores, shifts, county_fips = synthetic_data
        demo_path = _make_demo_parquet(tmp_path, county_fips.tolist())
        result = holdout_accuracy_ridge_augmented(
            scores, shifts, [0, 1, 2, 3, 4, 5], [6, 7, 8],
            county_fips=county_fips,
            demographics_path=demo_path,
        )
        assert result["mean_rmse"] >= 0.0
        for rmse in result["per_dim_rmse"]:
            assert rmse >= 0.0

    def test_returns_none_when_no_county_fips(self, synthetic_data, tmp_path):
        """Returns None gracefully when county_fips is None."""
        scores, shifts, county_fips = synthetic_data
        demo_path = _make_demo_parquet(tmp_path, county_fips.tolist())
        result = holdout_accuracy_ridge_augmented(
            scores, shifts, [0, 1, 2, 3, 4, 5], [6, 7, 8],
            county_fips=None,
            demographics_path=demo_path,
        )
        assert result is None

    def test_returns_none_when_file_missing(self, synthetic_data, tmp_path):
        """Returns None gracefully when demographics file does not exist."""
        scores, shifts, county_fips = synthetic_data
        result = holdout_accuracy_ridge_augmented(
            scores, shifts, [0, 1, 2, 3, 4, 5], [6, 7, 8],
            county_fips=county_fips,
            demographics_path=str(tmp_path / "nonexistent.parquet"),
        )
        assert result is None

    def test_partial_fips_overlap(self, synthetic_data, tmp_path):
        """Inner join with partial overlap uses only matched rows."""
        scores, shifts, county_fips = synthetic_data
        # Only provide demographics for first 40 counties
        demo_path = _make_demo_parquet(tmp_path, county_fips[:40].tolist())
        result = holdout_accuracy_ridge_augmented(
            scores, shifts, [0, 1, 2, 3, 4, 5], [6, 7, 8],
            county_fips=county_fips,
            demographics_path=demo_path,
        )
        assert result is not None
        assert result["n_matched_counties"] == 40

    def test_mean_r_is_mean_of_per_dim(self, synthetic_data, tmp_path):
        """mean_r equals the mean of per_dim_r."""
        scores, shifts, county_fips = synthetic_data
        demo_path = _make_demo_parquet(tmp_path, county_fips.tolist())
        result = holdout_accuracy_ridge_augmented(
            scores, shifts, [0, 1, 2, 3, 4, 5], [6, 7, 8],
            county_fips=county_fips,
            demographics_path=demo_path,
        )
        expected = float(np.mean(result["per_dim_r"]))
        assert abs(result["mean_r"] - expected) < 1e-9
