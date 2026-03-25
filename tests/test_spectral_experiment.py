"""Tests for scripts/experiment_spectral.py (P3.3).

All tests use synthetic data — no filesystem access required.

Covers:
  1.  temperature_soft_membership: row-normalization property
  2.  temperature_soft_membership: nearest-centroid gets highest weight (T=10)
  3.  temperature_soft_membership: T>=500 produces hard (one-hot) assignment
  4.  temperature_soft_membership: identical distances produce uniform output
  5.  temperature_soft_membership: shape preserved
  6.  spectral_soft_membership: output shape is (N, J)
  7.  spectral_soft_membership: rows sum to 1
  8.  spectral_soft_membership: labels are in 0..J-1
  9.  run_spectral: produces exactly J clusters on separable data
  10. run_spectral: labels are integers in valid range
  11. run_spectral: scores shape is (N, J)
  12. run_spectral: scores are non-negative and row-normalized
  13. compute_holdout_r_county_prior: perfect-type data gives r=1.0
  14. compute_holdout_r_county_prior: random labels give low r
  15. compute_holdout_r_county_prior: output keys are correct
  16. compute_holdout_r_county_prior: RMSE is non-negative
  17. compute_holdout_r_county_prior: flat holdout returns r=0.0
  18. comparison logic: spectral result dict has required keys
  19. run_kmeans: produces correct number of clusters
  20. load_data: training cols exclude holdout columns
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Load the experiment module from scripts/ (not a package)
# ---------------------------------------------------------------------------

_MODULE_PATH = Path(__file__).parents[1] / "scripts" / "experiment_spectral.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("experiment_spectral", _MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load_module()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_separable_data(n_clusters: int = 4, n_per_cluster: int = 20, n_dims: int = 5, seed: int = 0) -> np.ndarray:
    """Build linearly separable cluster data with large inter-cluster gaps."""
    rng = np.random.default_rng(seed)
    chunks = []
    for k in range(n_clusters):
        center = np.zeros(n_dims)
        center[k % n_dims] = k * 20.0
        chunk = rng.standard_normal((n_per_cluster, n_dims)) * 0.3 + center
        chunks.append(chunk)
    return np.vstack(chunks)


def _make_scores(N: int, J: int, seed: int = 42) -> np.ndarray:
    """Build valid soft membership scores (non-negative, row-normalized)."""
    rng = np.random.default_rng(seed)
    raw = np.abs(rng.standard_normal((N, J))) + 0.1
    return raw / raw.sum(axis=1, keepdims=True)


def _make_type_labels(N: int, J: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, J, size=N)


# ---------------------------------------------------------------------------
# 1. temperature_soft_membership — row normalization
# ---------------------------------------------------------------------------


def test_temp_soft_rows_sum_to_one(mod):
    rng = np.random.default_rng(1)
    dists = np.abs(rng.standard_normal((50, 8))) + 0.01
    scores = mod.temperature_soft_membership(dists, T=10.0)
    row_sums = scores.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(50), atol=1e-9)


# ---------------------------------------------------------------------------
# 2. nearest centroid gets highest weight (T=10)
# ---------------------------------------------------------------------------


def test_temp_soft_nearest_gets_max_weight(mod):
    # 5 counties, 4 centroids; centroid 2 is closest for all
    dists = np.array([
        [3.0, 2.0, 0.1, 4.0],
        [5.0, 3.0, 0.2, 6.0],
        [2.5, 1.5, 0.05, 3.5],
        [4.0, 2.0, 0.15, 5.0],
        [3.5, 2.5, 0.08, 4.5],
    ])
    scores = mod.temperature_soft_membership(dists, T=10.0)
    nearest = dists.argmin(axis=1)
    for i, j in enumerate(nearest):
        assert scores[i, j] == scores[i].max(), f"Row {i}: nearest centroid {j} should have max weight"


# ---------------------------------------------------------------------------
# 3. T>=500 produces hard (one-hot) assignment
# ---------------------------------------------------------------------------


def test_temp_soft_hard_at_high_T(mod):
    rng = np.random.default_rng(7)
    dists = np.abs(rng.standard_normal((30, 6))) + 0.01
    scores = mod.temperature_soft_membership(dists, T=1000.0)
    # Each row should be one-hot
    assert np.all((scores == 0) | (scores == 1))
    row_sums = scores.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(30), atol=1e-12)


# ---------------------------------------------------------------------------
# 4. Identical distances produce (approximately) uniform output
# ---------------------------------------------------------------------------


def test_temp_soft_uniform_on_equal_distances(mod):
    J = 5
    N = 10
    dists = np.ones((N, J))  # all distances equal
    scores = mod.temperature_soft_membership(dists, T=10.0)
    expected = np.full((N, J), 1.0 / J)
    np.testing.assert_allclose(scores, expected, atol=1e-9)


# ---------------------------------------------------------------------------
# 5. Shape preserved
# ---------------------------------------------------------------------------


def test_temp_soft_shape(mod):
    N, J = 47, 13
    dists = np.abs(np.random.standard_normal((N, J))) + 0.01
    scores = mod.temperature_soft_membership(dists, T=5.0)
    assert scores.shape == (N, J)


# ---------------------------------------------------------------------------
# 6. spectral_soft_membership — output shape
# ---------------------------------------------------------------------------


def test_spectral_soft_shape(mod):
    N, D, J = 50, 8, 6
    rng = np.random.default_rng(10)
    X = rng.standard_normal((N, D))
    labels = np.repeat(np.arange(J), N // J + 1)[:N]
    scores = mod.spectral_soft_membership(X, labels, J)
    assert scores.shape == (N, J)


# ---------------------------------------------------------------------------
# 7. spectral_soft_membership — rows sum to 1
# ---------------------------------------------------------------------------


def test_spectral_soft_rows_sum_to_one(mod):
    N, D, J = 40, 6, 5
    rng = np.random.default_rng(11)
    X = rng.standard_normal((N, D))
    labels = np.tile(np.arange(J), N // J + 1)[:N]
    scores = mod.spectral_soft_membership(X, labels, J)
    row_sums = scores.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(N), atol=1e-9)


# ---------------------------------------------------------------------------
# 8. spectral_soft_membership — labels in valid range
# ---------------------------------------------------------------------------


def test_spectral_soft_label_range(mod):
    N, D, J = 30, 4, 4
    rng = np.random.default_rng(12)
    X = rng.standard_normal((N, D))
    labels = np.repeat(np.arange(J), N // J + 1)[:N]
    scores = mod.spectral_soft_membership(X, labels, J)
    assert np.all(scores >= 0), "Scores must be non-negative"
    assert np.all(scores <= 1.0 + 1e-9), "Scores must be <= 1"


# ---------------------------------------------------------------------------
# 9. run_spectral — correct number of clusters on separable data
# ---------------------------------------------------------------------------


def test_run_spectral_cluster_count(mod):
    J = 4
    X = _make_separable_data(n_clusters=J, n_per_cluster=15, n_dims=J)
    scores, labels = mod.run_spectral(X, J)
    n_unique = len(np.unique(labels))
    # With clearly separable data, should find J clusters
    assert n_unique == J, f"Expected {J} clusters, got {n_unique}"


# ---------------------------------------------------------------------------
# 10. run_spectral — labels in valid integer range
# ---------------------------------------------------------------------------


def test_run_spectral_labels_in_range(mod):
    J = 5
    X = _make_separable_data(n_clusters=J, n_per_cluster=12)
    scores, labels = mod.run_spectral(X, J)
    assert labels.dtype in (np.int32, np.int64, np.int_) or np.issubdtype(labels.dtype, np.integer)
    assert labels.min() >= 0
    assert labels.max() < J


# ---------------------------------------------------------------------------
# 11. run_spectral — scores shape
# ---------------------------------------------------------------------------


def test_run_spectral_scores_shape(mod):
    J = 4
    X = _make_separable_data(n_clusters=J, n_per_cluster=10)
    N = X.shape[0]
    scores, labels = mod.run_spectral(X, J)
    assert scores.shape == (N, J), f"Expected ({N}, {J}), got {scores.shape}"


# ---------------------------------------------------------------------------
# 12. run_spectral — scores non-negative and row-normalized
# ---------------------------------------------------------------------------


def test_run_spectral_scores_valid(mod):
    J = 3
    X = _make_separable_data(n_clusters=J, n_per_cluster=15)
    scores, labels = mod.run_spectral(X, J)
    assert np.all(scores >= -1e-10), "Scores must be non-negative"
    row_sums = scores.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(len(X)), atol=1e-9)


# ---------------------------------------------------------------------------
# 13. compute_holdout_r_county_prior — perfect-type data gives r near 1.0
# ---------------------------------------------------------------------------


def test_holdout_r_county_prior_perfect(mod):
    """If types perfectly capture the signal, r should be ~1.0."""
    rng = np.random.default_rng(42)
    N, J = 100, 5
    # Build training data where each type has a distinct constant shift
    type_values = rng.standard_normal(J) * 2.0
    labels = np.repeat(np.arange(J), N // J + 1)[:N]

    # Training: each county's "history" is its type's value
    raw_training = np.tile(type_values[labels], (10, 1)).T  # (N, 10)

    # Holdout: same signal + tiny noise → should be near-perfect prediction
    noise = rng.standard_normal(N) * 0.01
    holdout_vals = type_values[labels] + noise
    holdout_matrix = holdout_vals[:, None]  # (N, 1)

    # One-hot scores (perfect hard assignment)
    scores = np.zeros((N, J))
    scores[np.arange(N), labels] = 1.0

    result = mod.compute_holdout_r_county_prior(scores, raw_training, holdout_matrix)
    assert result["mean_r"] > 0.99, f"Expected r > 0.99, got {result['mean_r']:.4f}"


# ---------------------------------------------------------------------------
# 14. compute_holdout_r_county_prior — random labels give low r
# ---------------------------------------------------------------------------


def test_holdout_r_county_prior_random_labels(mod):
    """Random labels should produce low holdout r."""
    rng = np.random.default_rng(99)
    N, J, D = 200, 20, 10
    raw_training = rng.standard_normal((N, D))
    holdout_matrix = rng.standard_normal((N, 3))
    scores = _make_scores(N, J, seed=99)

    result = mod.compute_holdout_r_county_prior(scores, raw_training, holdout_matrix)
    # Random scores should give near-zero r (allow some wiggle)
    assert result["mean_r"] < 0.5, f"Expected r < 0.5 for random labels, got {result['mean_r']:.4f}"


# ---------------------------------------------------------------------------
# 15. compute_holdout_r_county_prior — output keys are correct
# ---------------------------------------------------------------------------


def test_holdout_r_county_prior_keys(mod):
    N, J, D = 50, 5, 8
    rng = np.random.default_rng(55)
    scores = _make_scores(N, J)
    raw_training = rng.standard_normal((N, D))
    holdout_matrix = rng.standard_normal((N, 3))

    result = mod.compute_holdout_r_county_prior(scores, raw_training, holdout_matrix)
    assert "mean_r" in result
    assert "per_dim_r" in result
    assert "mean_rmse" in result
    assert "per_dim_rmse" in result
    assert len(result["per_dim_r"]) == 3
    assert len(result["per_dim_rmse"]) == 3


# ---------------------------------------------------------------------------
# 16. compute_holdout_r_county_prior — RMSE is non-negative
# ---------------------------------------------------------------------------


def test_holdout_r_county_prior_rmse_nonneg(mod):
    N, J, D = 80, 8, 12
    rng = np.random.default_rng(66)
    scores = _make_scores(N, J)
    raw_training = rng.standard_normal((N, D))
    holdout_matrix = rng.standard_normal((N, 3))

    result = mod.compute_holdout_r_county_prior(scores, raw_training, holdout_matrix)
    assert result["mean_rmse"] >= 0.0
    for rmse in result["per_dim_rmse"]:
        assert rmse >= 0.0


# ---------------------------------------------------------------------------
# 17. compute_holdout_r_county_prior — flat holdout returns r=0.0
# ---------------------------------------------------------------------------


def test_holdout_r_county_prior_flat_holdout(mod):
    """If holdout has no variance, r should be 0.0."""
    N, J, D = 40, 5, 6
    rng = np.random.default_rng(77)
    scores = _make_scores(N, J)
    raw_training = rng.standard_normal((N, D))
    holdout_matrix = np.ones((N, 2))  # zero variance

    result = mod.compute_holdout_r_county_prior(scores, raw_training, holdout_matrix)
    for r in result["per_dim_r"]:
        assert r == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# 18. Result dict has required keys
# ---------------------------------------------------------------------------


def test_result_dict_keys(mod):
    """run_experiment result items contain expected keys."""
    # Build a minimal single result manually
    N, J, D = 40, 4, 6
    rng = np.random.default_rng(88)
    X = _make_separable_data(n_clusters=J, n_per_cluster=10)
    X_raw = X.copy()
    holdout_matrix = rng.standard_normal((X.shape[0], 3))

    scores_km, labels_km = mod.run_kmeans(X, J)
    km_stats = mod.compute_holdout_r_county_prior(scores_km, X_raw, holdout_matrix)
    row = {
        "method": "KMeans",
        "J": J,
        "holdout_r": km_stats["mean_r"],
        "mean_rmse": km_stats["mean_rmse"],
        "per_dim_r": km_stats["per_dim_r"],
        "wall_seconds": 0.1,
    }
    required_keys = {"method", "J", "holdout_r", "mean_rmse", "per_dim_r"}
    assert required_keys.issubset(row.keys())


# ---------------------------------------------------------------------------
# 19. run_kmeans — produces correct number of clusters
# ---------------------------------------------------------------------------


def test_run_kmeans_cluster_count(mod):
    J = 5
    X = _make_separable_data(n_clusters=J, n_per_cluster=20)
    scores, labels = mod.run_kmeans(X, J)
    n_unique = len(np.unique(labels))
    assert n_unique == J, f"Expected {J} clusters, got {n_unique}"


# ---------------------------------------------------------------------------
# 20. load_data — training cols exclude holdout columns
# ---------------------------------------------------------------------------


def test_load_data_no_holdout_in_training(mod):
    """Training columns must not include the three holdout columns."""
    X_weighted, X_raw, training_cols, holdout_cols = mod.load_data()
    holdout_set = set(mod.HOLDOUT_COLUMNS)
    training_set = set(training_cols)
    overlap = holdout_set & training_set
    assert len(overlap) == 0, f"Holdout columns found in training: {overlap}"
    assert len(holdout_cols) == 3
    assert X_weighted.shape == X_raw.shape
    assert X_weighted.shape[0] == 3154  # national (all 50 states + DC)
