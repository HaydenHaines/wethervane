"""Tests for scripts/experiment_j_sweep.py.

Covers:
  - temperature_soft_membership: numerics, row-normalization
  - group_columns_by_pair: column grouping
  - make_cv_folds: fold generation, year filtering
  - compute_centroid_distances: shape and correctness
  - predict_holdout_columns: weighted-mean reconstruction
  - compute_holdout_r: Pearson r across columns
  - compute_holdout_mae: MAE correctness
  - compute_type_coherence: cosine similarity coherence
  - run_j_sweep_fold: single-fold integration
  - run_j_sweep: full sweep shape and ordering

All tests use synthetic data only — no filesystem access.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Load the experiment module (lives in scripts/, not a package)
# ---------------------------------------------------------------------------

_MODULE_PATH = Path(__file__).parents[1] / "scripts" / "experiments" / "experiment_j_sweep.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("experiment_j_sweep", _MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load_module()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_columns(pairs=None):
    """Return synthetic column names matching real shift name patterns."""
    if pairs is None:
        pairs = ["08_12", "12_16", "16_20", "20_24"]
    cols = []
    for p in pairs:
        cols += [f"pres_d_shift_{p}", f"pres_r_shift_{p}", f"pres_turnout_shift_{p}"]
    return cols


# ---------------------------------------------------------------------------
# temperature_soft_membership
# ---------------------------------------------------------------------------


class TestTemperatureSoftMembership:
    def test_rows_sum_to_one(self, mod):
        rng = np.random.RandomState(0)
        dists = np.abs(rng.randn(50, 15)) + 0.1
        for T in [1.0, 5.0, 10.0, 50.0, 999.0]:
            scores = mod.temperature_soft_membership(dists, T)
            np.testing.assert_allclose(
                scores.sum(axis=1), 1.0, atol=1e-9,
                err_msg=f"Row sums not 1 at T={T}"
            )

    def test_scores_non_negative(self, mod):
        rng = np.random.RandomState(1)
        dists = np.abs(rng.randn(30, 10)) + 0.01
        scores = mod.temperature_soft_membership(dists, T=10.0)
        assert (scores >= 0).all()

    def test_t999_gives_hard_assignment(self, mod):
        """At T=999, should give near-hard assignment (max weight > 0.99)."""
        rng = np.random.RandomState(3)
        dists = np.abs(rng.randn(40, 12)) + 0.1
        scores = mod.temperature_soft_membership(dists, T=999.0)
        assert (scores.max(axis=1) > 0.99).all()

    def test_uniform_dists_uniform_scores(self, mod):
        """Equal distances → equal membership weights."""
        dists = np.ones((5, 6))
        scores = mod.temperature_soft_membership(dists, T=10.0)
        np.testing.assert_allclose(scores, np.full((5, 6), 1.0 / 6), atol=1e-9)

    def test_output_shape_preserved(self, mod):
        dists = np.abs(np.random.randn(20, 8)) + 0.1
        scores = mod.temperature_soft_membership(dists, T=5.0)
        assert scores.shape == (20, 8)


# ---------------------------------------------------------------------------
# group_columns_by_pair
# ---------------------------------------------------------------------------


class TestGroupColumnsByPair:
    def test_groups_by_suffix(self, mod):
        cols = _make_synthetic_columns(["08_12", "12_16"])
        groups = mod.group_columns_by_pair(cols)
        assert "08_12" in groups
        assert "12_16" in groups
        assert len(groups["08_12"]) == 3
        assert len(groups["12_16"]) == 3

    def test_ignores_unparseable_columns(self, mod):
        cols = ["county_fips", "pres_d_shift_08_12", "pres_r_shift_08_12", "pres_turnout_shift_08_12"]
        groups = mod.group_columns_by_pair(cols)
        # county_fips has no pair suffix — should not appear
        assert "county_fips" not in groups
        assert "08_12" in groups

    def test_column_indices_correct(self, mod):
        cols = _make_synthetic_columns(["08_12"])
        groups = mod.group_columns_by_pair(cols)
        assert sorted(groups["08_12"]) == [0, 1, 2]

    def test_multiple_race_types_in_same_pair(self, mod):
        cols = [
            "pres_d_shift_12_16",
            "pres_r_shift_12_16",
            "pres_turnout_shift_12_16",
            "gov_d_shift_12_16",
            "gov_r_shift_12_16",
            "gov_turnout_shift_12_16",
        ]
        groups = mod.group_columns_by_pair(cols)
        assert len(groups["12_16"]) == 6


# ---------------------------------------------------------------------------
# make_cv_folds
# ---------------------------------------------------------------------------


class TestMakeCvFolds:
    def test_fold_count_matches_pairs_after_min_year(self, mod):
        cols = _make_synthetic_columns(["08_12", "12_16", "16_20"])
        folds = mod.make_cv_folds(cols, min_year=2008)
        assert len(folds) == 3

    def test_min_year_filters_old_pairs(self, mod):
        # Include a pre-2008 pair that should be filtered out
        cols = _make_synthetic_columns(["04_08", "08_12", "12_16"])
        folds = mod.make_cv_folds(cols, min_year=2008)
        fold_keys = [f[0] for f in folds]
        assert "04_08" not in fold_keys
        assert "08_12" in fold_keys
        assert "12_16" in fold_keys

    def test_train_holdout_partition_covers_all(self, mod):
        cols = _make_synthetic_columns(["08_12", "12_16", "16_20"])
        folds = mod.make_cv_folds(cols, min_year=2008)
        n_total = len(cols)
        for pair_key, train_cols, holdout_cols in folds:
            assert len(train_cols) + len(holdout_cols) == n_total
            assert set(train_cols) & set(holdout_cols) == set()

    def test_fold_tuple_structure(self, mod):
        cols = _make_synthetic_columns(["08_12", "12_16"])
        folds = mod.make_cv_folds(cols, min_year=2008)
        for fold in folds:
            assert len(fold) == 3
            pair_key, train_cols, holdout_cols = fold
            assert isinstance(pair_key, str)
            assert isinstance(train_cols, list)
            assert isinstance(holdout_cols, list)


# ---------------------------------------------------------------------------
# compute_centroid_distances
# ---------------------------------------------------------------------------


class TestComputeCentroidDistances:
    def test_shape(self, mod):
        N, D, J = 50, 10, 5
        X = np.random.randn(N, D)
        centroids = np.random.randn(J, D)
        dists = mod.compute_centroid_distances(X, centroids)
        assert dists.shape == (N, J)

    def test_zero_distance_on_centroid(self, mod):
        """A county exactly at a centroid should have distance 0 to that centroid."""
        centroids = np.array([[1.0, 0.0], [0.0, 1.0]])
        X = centroids.copy()  # 2 counties exactly on the centroids
        dists = mod.compute_centroid_distances(X, centroids)
        assert dists[0, 0] == pytest.approx(0.0, abs=1e-12)
        assert dists[1, 1] == pytest.approx(0.0, abs=1e-12)

    def test_distances_non_negative(self, mod):
        X = np.random.randn(30, 8)
        centroids = np.random.randn(4, 8)
        dists = mod.compute_centroid_distances(X, centroids)
        assert (dists >= 0).all()


# ---------------------------------------------------------------------------
# predict_holdout_columns
# ---------------------------------------------------------------------------


class TestPredictHoldoutColumns:
    def test_output_shape(self, mod):
        N, J, D_hold = 40, 6, 3
        scores = np.random.dirichlet(np.ones(J), size=N)
        X_holdout = np.random.randn(N, D_hold)
        pred = mod.predict_holdout_columns(scores, X_holdout)
        assert pred.shape == (N, D_hold)

    def test_hard_assignment_recovers_type_mean(self, mod):
        """If all counties are hard-assigned (score=1 on one type), prediction
        should equal the type mean exactly."""
        N, J, D = 9, 3, 2
        # 3 types, 3 counties each
        scores = np.zeros((N, J))
        for j in range(J):
            scores[j * 3:(j + 1) * 3, j] = 1.0

        X_holdout = np.arange(N * D, dtype=float).reshape(N, D)
        pred = mod.predict_holdout_columns(scores, X_holdout)

        # For type 0 (counties 0,1,2), predicted value = mean of rows 0,1,2
        expected_type0 = X_holdout[:3].mean(axis=0)
        np.testing.assert_allclose(pred[:3], np.tile(expected_type0, (3, 1)), atol=1e-10)

    def test_equal_scores_gives_global_mean(self, mod):
        """If all counties have equal membership across all types, prediction = grand mean."""
        N, J, D = 20, 4, 3
        scores = np.full((N, J), 1.0 / J)
        X_holdout = np.random.randn(N, D)
        pred = mod.predict_holdout_columns(scores, X_holdout)
        # Each type mean = weighted mean with equal weights = grand mean
        grand_mean = X_holdout.mean(axis=0)
        np.testing.assert_allclose(pred, np.tile(grand_mean, (N, 1)), atol=1e-10)


# ---------------------------------------------------------------------------
# compute_holdout_r
# ---------------------------------------------------------------------------


class TestComputeHoldoutR:
    def test_perfect_prediction(self, mod):
        X = np.random.randn(50, 3)
        r = mod.compute_holdout_r(X, X.copy())
        assert r == pytest.approx(1.0, abs=1e-9)

    def test_inverted_prediction(self, mod):
        X = np.random.randn(50, 3) + np.arange(50)[:, None]
        r = mod.compute_holdout_r(X, -X)
        assert r == pytest.approx(-1.0, abs=1e-9)

    def test_zero_variance_column_returns_zero(self, mod):
        """Constant column should contribute 0.0 to mean r (no crash)."""
        actual = np.ones((30, 1))
        predicted = np.random.randn(30, 1)
        r = mod.compute_holdout_r(actual, predicted)
        assert r == pytest.approx(0.0, abs=1e-9)

    def test_mean_across_columns(self, mod):
        """r should be the mean Pearson r across all holdout columns."""
        rng = np.random.RandomState(42)
        N = 80
        col1 = rng.randn(N)
        col2 = rng.randn(N)
        pred1 = col1.copy()  # r = 1.0
        pred2 = -col2.copy()  # r = -1.0
        actual = np.stack([col1, col2], axis=1)
        predicted = np.stack([pred1, pred2], axis=1)
        r = mod.compute_holdout_r(actual, predicted)
        assert r == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# compute_holdout_mae
# ---------------------------------------------------------------------------


class TestComputeHoldoutMAE:
    def test_zero_error(self, mod):
        X = np.random.randn(40, 3)
        mae = mod.compute_holdout_mae(X, X.copy())
        assert mae == pytest.approx(0.0, abs=1e-12)

    def test_known_mae(self, mod):
        actual = np.zeros((4, 2))
        predicted = np.ones((4, 2))
        mae = mod.compute_holdout_mae(actual, predicted)
        assert mae == pytest.approx(1.0, abs=1e-12)

    def test_scalar_output(self, mod):
        X = np.random.randn(20, 5)
        result = mod.compute_holdout_mae(X, X + 0.1)
        assert isinstance(result, float)
        assert result == pytest.approx(0.1, abs=1e-9)


# ---------------------------------------------------------------------------
# compute_type_coherence
# ---------------------------------------------------------------------------


class TestComputeTypeCoherence:
    def test_returns_float(self, mod):
        rng = np.random.RandomState(7)
        X = rng.randn(30, 10)
        labels = np.repeat([0, 1, 2], 10)
        coh = mod.compute_type_coherence(X, labels, j=3)
        assert isinstance(coh, float)

    def test_perfect_coherence_identical_vectors(self, mod):
        """All vectors identical within each type → coherence = 1.0."""
        X = np.array([[1.0, 0.0]] * 5 + [[0.0, 1.0]] * 5)
        labels = np.array([0] * 5 + [1] * 5)
        coh = mod.compute_type_coherence(X, labels, j=2)
        assert coh == pytest.approx(1.0, abs=1e-9)

    def test_coherence_range(self, mod):
        """Coherence should be in [-1, 1] for any input."""
        rng = np.random.RandomState(99)
        X = rng.randn(60, 15)
        labels = rng.randint(0, 6, size=60)
        coh = mod.compute_type_coherence(X, labels, j=6)
        assert -1.0 - 1e-9 <= coh <= 1.0 + 1e-9

    def test_singleton_types_skipped(self, mod):
        """Types with only 1 member should not crash or be counted."""
        X = np.random.randn(5, 4)
        labels = np.array([0, 1, 2, 3, 4])  # every type is a singleton
        coh = mod.compute_type_coherence(X, labels, j=5)
        # All singletons → nan (no pairs to average)
        assert np.isnan(coh)


# ---------------------------------------------------------------------------
# run_j_sweep_fold — single fold integration
# ---------------------------------------------------------------------------


class TestRunJSweepFold:
    def test_returns_expected_keys(self, mod):
        rng = np.random.RandomState(42)
        N, D = 60, 12  # 4 pairs × 3 cols each
        X = rng.randn(N, D)
        cols = _make_synthetic_columns(["08_12", "12_16", "16_20", "20_24"])
        folds = mod.make_cv_folds(cols, min_year=2008)
        fold = folds[0]
        result = mod.run_j_sweep_fold(X, cols, j=4, fold=fold)
        assert "pair_key" in result
        assert "holdout_r" in result
        assert "holdout_mae" in result

    def test_holdout_r_in_valid_range(self, mod):
        rng = np.random.RandomState(10)
        N, D = 80, 9  # 3 pairs × 3 cols
        X = rng.randn(N, D)
        cols = _make_synthetic_columns(["08_12", "12_16", "16_20"])
        folds = mod.make_cv_folds(cols, min_year=2008)
        for fold in folds:
            result = mod.run_j_sweep_fold(X, cols, j=3, fold=fold)
            assert -1.0 - 1e-6 <= result["holdout_r"] <= 1.0 + 1e-6

    def test_holdout_mae_non_negative(self, mod):
        rng = np.random.RandomState(11)
        N, D = 60, 9
        X = rng.randn(N, D)
        cols = _make_synthetic_columns(["08_12", "12_16", "16_20"])
        folds = mod.make_cv_folds(cols, min_year=2008)
        for fold in folds:
            result = mod.run_j_sweep_fold(X, cols, j=3, fold=fold)
            assert result["holdout_mae"] >= 0.0


# ---------------------------------------------------------------------------
# run_j_sweep — full sweep output shape and ordering
# ---------------------------------------------------------------------------


class TestRunJSweep:
    @pytest.fixture(scope="class")
    def sweep_results(self, mod):
        """Run a minimal J sweep on synthetic data injected via monkeypatching."""
        import pandas as pd

        # Synthetic data: 80 counties, 4 election pairs × 3 cols = 12 dims
        rng = np.random.RandomState(0)
        N, D = 80, 12
        X = rng.randn(N, D)
        shift_cols = _make_synthetic_columns(["08_12", "12_16", "16_20", "20_24"])
        county_fips = np.array([f"{i:05d}" for i in range(N)])
        weights_vec = np.ones(D)

        # Monkeypatch load_shift_matrix to return synthetic data
        original = mod.load_shift_matrix

        def fake_load(min_year=2008):
            return X, shift_cols, county_fips, weights_vec

        mod.load_shift_matrix = fake_load
        try:
            results = mod.run_j_sweep(
                j_range=range(3, 6),  # small range for speed
                min_year=2008,
                temperature=10.0,
                verbose=False,
            )
        finally:
            mod.load_shift_matrix = original

        return results

    def test_result_is_dataframe(self, sweep_results, mod):
        import pandas as pd
        assert isinstance(sweep_results, pd.DataFrame)

    def test_row_count_matches_j_range(self, sweep_results, mod):
        assert len(sweep_results) == 3  # J=3,4,5

    def test_expected_columns_present(self, sweep_results, mod):
        expected = {"j", "mean_r", "std_r", "mean_mae", "std_mae", "coherence", "n_folds"}
        assert expected.issubset(set(sweep_results.columns))

    def test_j_values_correct(self, sweep_results, mod):
        assert list(sweep_results["j"]) == [3, 4, 5]

    def test_metrics_numeric(self, sweep_results, mod):
        for col in ["mean_r", "std_r", "mean_mae", "std_mae", "coherence"]:
            assert sweep_results[col].dtype in [np.float64, float]
            assert sweep_results[col].notna().all(), f"NaN found in {col}"

    def test_mae_non_negative(self, sweep_results, mod):
        assert (sweep_results["mean_mae"] >= 0).all()

    def test_std_non_negative(self, sweep_results, mod):
        assert (sweep_results["std_r"] >= 0).all()
        assert (sweep_results["std_mae"] >= 0).all()

    def test_n_folds_positive(self, sweep_results, mod):
        assert (sweep_results["n_folds"] > 0).all()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_j_equals_1_runs(self, mod):
        """J=1 is a degenerate case but should not crash."""
        rng = np.random.RandomState(5)
        N, D = 30, 9
        X = rng.randn(N, D)
        cols = _make_synthetic_columns(["08_12", "12_16", "16_20"])
        folds = mod.make_cv_folds(cols, min_year=2008)
        fold = folds[0]
        result = mod.run_j_sweep_fold(X, cols, j=1, fold=fold)
        # J=1: all counties in one type → prediction = constant (type mean)
        assert isinstance(result["holdout_r"], float)

    def test_j_larger_than_n_fails_gracefully(self, mod):
        """J > N counties: KMeans will raise. run_j_sweep handles it via try/except."""
        rng = np.random.RandomState(6)
        N = 5  # tiny
        D = 9
        X = rng.randn(N, D)
        cols = _make_synthetic_columns(["08_12", "12_16", "16_20"])
        folds = mod.make_cv_folds(cols, min_year=2008)
        fold = folds[0]
        # Direct call will raise from KMeans — that's expected
        with pytest.raises(Exception):
            mod.run_j_sweep_fold(X, cols, j=N + 10, fold=fold)
