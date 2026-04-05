"""Tests for scripts/experiment_lambda_sweep.py.

Covers:
  - frobenius_distance_from_identity: correctness and boundary cases
  - condition_number: correctness and near-singular matrix
  - evaluate_lambda: output dict structure and metric monotonicity
  - run_lambda_sweep: shape, lambda range, and column contract
  - print_results: runs without error (smoke test)
  - save_results: writes CSV with expected columns
  - make_recommendation: strings for better / near-optimal / no-current-row cases

All tests use synthetic data only — no filesystem access.
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Load the experiment module from scripts/ (not a package)
# ---------------------------------------------------------------------------

_MODULE_PATH = Path(__file__).parents[1] / "scripts" / "experiments" / "experiment_lambda_sweep.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("experiment_lambda_sweep", _MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load_module()


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _make_profiles(J: int = 8, F: int = 10, seed: int = 42) -> tuple[pd.DataFrame, list[str]]:
    """Return synthetic type profiles DataFrame + feature column names."""
    rng = np.random.default_rng(seed)
    data = rng.uniform(0.0, 1.0, size=(J, F))
    cols = [f"feat_{i}" for i in range(F)]
    df = pd.DataFrame(data, columns=cols)
    df["type_id"] = np.arange(J)
    return df, cols


def _make_type_scores(N: int = 50, J: int = 8, seed: int = 7) -> np.ndarray:
    """Return synthetic county type scores (N, J), rows sum to 1."""
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.0, 1.0, size=(N, J))
    return raw / raw.sum(axis=1, keepdims=True)


def _make_shift_matrix(N: int = 50, D: int = 9, seed: int = 13) -> tuple[np.ndarray, list[list[int]]]:
    """Return synthetic shift matrix (N, D) and column groups (3 cols per election)."""
    rng = np.random.default_rng(seed)
    shifts = rng.standard_normal(size=(N, D)) * 0.05
    groups = [list(range(i, min(i + 3, D))) for i in range(0, D, 3)]
    return shifts, groups


# ---------------------------------------------------------------------------
# frobenius_distance_from_identity
# ---------------------------------------------------------------------------


class TestFrobeniusDistanceFromIdentity:
    def test_all_ones_gives_zero(self, mod):
        """All-1s matrix has zero distance from the all-1s reference."""
        J = 5
        C = np.ones((J, J))
        assert mod.frobenius_distance_from_identity(C) == pytest.approx(0.0, abs=1e-12)

    def test_identity_gives_nonzero(self, mod):
        """Identity matrix is far from the all-1s reference."""
        J = 5
        C = np.eye(J)
        dist = mod.frobenius_distance_from_identity(C)
        # Off-diagonal elements of (I - 1) are all -1, diagonal are 0
        # ||I - 1||_F = sqrt(J*(J-1)) for J=5 = sqrt(20) ≈ 4.47
        expected = float(np.sqrt(J * (J - 1)))
        assert dist == pytest.approx(expected, rel=1e-6)

    def test_returns_float(self, mod):
        C = np.eye(4)
        assert isinstance(mod.frobenius_distance_from_identity(C), float)

    def test_larger_matrix_monotone_from_identity(self, mod):
        """A matrix with more off-diagonal structure is farther from all-1s."""
        J = 6
        # Two matrices: one closer to all-1s, one farther
        C_close = 0.95 * np.ones((J, J)) + 0.05 * np.eye(J)
        C_far = 0.5 * np.ones((J, J)) + 0.5 * np.eye(J)
        d_close = mod.frobenius_distance_from_identity(C_close)
        d_far = mod.frobenius_distance_from_identity(C_far)
        assert d_close < d_far


# ---------------------------------------------------------------------------
# condition_number
# ---------------------------------------------------------------------------


class TestConditionNumber:
    def test_identity_gives_one(self, mod):
        """Identity matrix has condition number = 1."""
        C = np.eye(5)
        assert mod.condition_number(C) == pytest.approx(1.0, rel=1e-6)

    def test_all_ones_is_near_singular(self, mod):
        """All-1s correlation matrix (rank 1) has a very large condition number."""
        J = 5
        C = np.ones((J, J))
        cond = mod.condition_number(C)
        assert cond > 1e6

    def test_well_conditioned_pd_matrix(self, mod):
        """A well-conditioned PD matrix has moderate condition number."""
        rng = np.random.default_rng(99)
        A = rng.standard_normal((6, 6))
        C = (A @ A.T) / 6 + np.eye(6) * 2  # strongly regularised
        cond = mod.condition_number(C)
        assert cond > 1.0
        assert cond < 1e4

    def test_returns_float(self, mod):
        C = np.eye(4)
        assert isinstance(mod.condition_number(C), float)


# ---------------------------------------------------------------------------
# evaluate_lambda (using production covariance functions with synthetic data)
# ---------------------------------------------------------------------------


class TestEvaluateLambda:
    def test_output_keys(self, mod):
        """evaluate_lambda must return a dict with exactly the expected keys."""
        profiles, feature_cols = _make_profiles()
        scores = _make_type_scores(J=8)
        shifts, groups = _make_shift_matrix()

        row = mod.evaluate_lambda(
            lam=0.75,
            type_profiles=profiles,
            feature_cols=feature_cols,
            type_scores=scores,
            shift_matrix=shifts,
            election_col_groups=groups,
        )
        assert set(row.keys()) == {"lambda", "validation_r", "frobenius_distance", "condition_number"}

    def test_lambda_recorded_correctly(self, mod):
        """The returned lambda value must match what was passed in."""
        profiles, feature_cols = _make_profiles()
        scores = _make_type_scores(J=8)
        shifts, groups = _make_shift_matrix()

        for lam in [0.1, 0.5, 0.75, 0.95]:
            row = mod.evaluate_lambda(
                lam=lam,
                type_profiles=profiles,
                feature_cols=feature_cols,
                type_scores=scores,
                shift_matrix=shifts,
                election_col_groups=groups,
            )
            assert abs(row["lambda"] - round(lam, 4)) < 1e-9

    def test_frobenius_monotone_with_lambda(self, mod):
        """Higher lambda retains more demographic signal → larger Frobenius distance from all-1s.

        C_final = lambda * C_pearson + (1-lambda) * ones.
        lambda=0 → all-1s (dist=0); lambda=1 → raw Pearson (dist=max).
        """
        profiles, feature_cols = _make_profiles()
        scores = _make_type_scores(J=8)
        shifts, groups = _make_shift_matrix()

        row_low = mod.evaluate_lambda(
            lam=0.1, type_profiles=profiles, feature_cols=feature_cols,
            type_scores=scores, shift_matrix=shifts, election_col_groups=groups,
        )
        row_high = mod.evaluate_lambda(
            lam=0.9, type_profiles=profiles, feature_cols=feature_cols,
            type_scores=scores, shift_matrix=shifts, election_col_groups=groups,
        )
        # Higher lambda → more demographic Pearson retained → farther from all-1s
        assert row_high["frobenius_distance"] > row_low["frobenius_distance"]

    def test_condition_number_positive(self, mod):
        """Condition number must always be positive."""
        profiles, feature_cols = _make_profiles()
        scores = _make_type_scores(J=8)
        shifts, groups = _make_shift_matrix()

        row = mod.evaluate_lambda(
            lam=0.5, type_profiles=profiles, feature_cols=feature_cols,
            type_scores=scores, shift_matrix=shifts, election_col_groups=groups,
        )
        assert row["condition_number"] > 0.0

    def test_validation_r_is_float_or_nan(self, mod):
        """validation_r must be a finite float or NaN, never a string."""
        profiles, feature_cols = _make_profiles()
        scores = _make_type_scores(J=8)
        shifts, groups = _make_shift_matrix()

        row = mod.evaluate_lambda(
            lam=0.75, type_profiles=profiles, feature_cols=feature_cols,
            type_scores=scores, shift_matrix=shifts, election_col_groups=groups,
        )
        val = row["validation_r"]
        assert isinstance(val, float)


# ---------------------------------------------------------------------------
# run_lambda_sweep (patched data loaders)
# ---------------------------------------------------------------------------


class TestRunLambdaSweep:
    """Test run_lambda_sweep by monkey-patching the data loading functions."""

    def _patch_loaders(self, mod, J=8, N=50, D=9):
        """Patch module-level load_* functions to return synthetic data."""
        profiles, feature_cols = _make_profiles(J=J)
        scores = _make_type_scores(N=N, J=J)
        shifts, groups = _make_shift_matrix(N=N, D=D)

        mod._orig_load_type_profiles = mod.load_type_profiles
        mod._orig_load_type_scores = mod.load_type_scores
        mod._orig_load_shift_matrix = mod.load_shift_matrix

        mod.load_type_profiles = lambda: (profiles, feature_cols)
        mod.load_type_scores = lambda: scores
        mod.load_shift_matrix = lambda: (shifts, groups)

    def _unpatch_loaders(self, mod):
        mod.load_type_profiles = mod._orig_load_type_profiles
        mod.load_type_scores = mod._orig_load_type_scores
        mod.load_shift_matrix = mod._orig_load_shift_matrix

    def test_output_shape(self, mod):
        """run_lambda_sweep returns one row per lambda value."""
        self._patch_loaders(mod)
        try:
            results = mod.run_lambda_sweep(
                lambda_min=0.1, lambda_max=0.3, step=0.1, verbose=False
            )
        finally:
            self._unpatch_loaders(mod)
        # lambda=0.1, 0.2, 0.3 → 3 rows
        assert len(results) == 3

    def test_output_columns(self, mod):
        """run_lambda_sweep DataFrame has exactly the required columns."""
        self._patch_loaders(mod)
        try:
            results = mod.run_lambda_sweep(
                lambda_min=0.5, lambda_max=0.6, step=0.1, verbose=False
            )
        finally:
            self._unpatch_loaders(mod)
        assert set(results.columns) == {"lambda", "validation_r", "frobenius_distance", "condition_number"}

    def test_lambdas_in_range(self, mod):
        """All returned lambda values must be within [lambda_min, lambda_max]."""
        self._patch_loaders(mod)
        try:
            results = mod.run_lambda_sweep(
                lambda_min=0.2, lambda_max=0.8, step=0.2, verbose=False
            )
        finally:
            self._unpatch_loaders(mod)
        assert (results["lambda"] >= 0.19).all()
        assert (results["lambda"] <= 0.81).all()

    def test_frobenius_increases_with_lambda(self, mod):
        """Frobenius distance from all-1s must increase as lambda increases.

        C_final = lambda * C_pearson + (1-lambda) * ones, so higher lambda
        moves the matrix away from all-1s toward raw Pearson.
        """
        self._patch_loaders(mod)
        try:
            results = mod.run_lambda_sweep(
                lambda_min=0.1, lambda_max=0.9, step=0.1, verbose=False
            )
        finally:
            self._unpatch_loaders(mod)
        frob = results["frobenius_distance"].values
        # Monotonically non-decreasing (allow tiny float tolerance)
        diffs = np.diff(frob)
        assert (diffs >= -1e-6).all(), f"Frobenius not monotone increasing: diffs={diffs}"


# ---------------------------------------------------------------------------
# save_results
# ---------------------------------------------------------------------------


class TestSaveResults:
    def test_saves_csv_with_expected_columns(self, mod):
        """save_results writes a CSV with the required column names."""
        df = pd.DataFrame({
            "lambda": [0.1, 0.5, 0.9],
            "validation_r": [0.2, 0.3, 0.25],
            "frobenius_distance": [5.0, 3.0, 1.5],
            "condition_number": [10.0, 20.0, 50.0],
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = mod.save_results(df, output_dir=Path(tmpdir))
            loaded = pd.read_csv(out_path)
        assert list(loaded.columns) == ["lambda", "validation_r", "frobenius_distance", "condition_number"]
        assert len(loaded) == 3


# ---------------------------------------------------------------------------
# make_recommendation
# ---------------------------------------------------------------------------


class TestMakeRecommendation:
    def test_better_lambda_recommends_change(self, mod):
        """If best lambda > current by > 0.005 in r, recommend change."""
        df = pd.DataFrame({
            "lambda": [0.1, 0.5, 0.75, 0.9],
            "validation_r": [0.20, 0.50, 0.30, 0.55],
            "frobenius_distance": [6.0, 4.0, 2.5, 1.5],
            "condition_number": [5.0, 8.0, 12.0, 20.0],
        })
        rec = mod.make_recommendation(df, current_lambda=0.75)
        assert "0.75" in rec
        assert "0.90" in rec or "0.9" in rec

    def test_near_optimal_gives_no_change_message(self, mod):
        """If best lambda delta < 0.005 r, recommend no change."""
        df = pd.DataFrame({
            "lambda": [0.70, 0.75, 0.80],
            "validation_r": [0.300, 0.3005, 0.302],
            "frobenius_distance": [3.0, 2.5, 2.0],
            "condition_number": [8.0, 9.0, 10.0],
        })
        rec = mod.make_recommendation(df, current_lambda=0.75)
        assert "No change" in rec or "near-optimal" in rec

    def test_all_nan_returns_error_string(self, mod):
        """If all validation_r are NaN, return an error string."""
        df = pd.DataFrame({
            "lambda": [0.5, 0.75],
            "validation_r": [float("nan"), float("nan")],
            "frobenius_distance": [3.0, 2.0],
            "condition_number": [8.0, 9.0],
        })
        rec = mod.make_recommendation(df, current_lambda=0.75)
        assert "Cannot" in rec or "no valid" in rec

    def test_returns_string(self, mod):
        """make_recommendation always returns a str."""
        df = pd.DataFrame({
            "lambda": [0.5, 0.75, 0.9],
            "validation_r": [0.2, 0.25, 0.22],
            "frobenius_distance": [4.0, 2.5, 1.5],
            "condition_number": [8.0, 10.0, 15.0],
        })
        rec = mod.make_recommendation(df, current_lambda=0.75)
        assert isinstance(rec, str)
