"""Tests for rmse_by_super_type() in src/validation/validate_types.py.

All tests use synthetic data so no real data files are required.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from src.validation.validate_types import rmse_by_super_type, RMSE_FLAG_THRESHOLD


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestRmseBySuperTypeNormalOperation:
    """Normal multi-super-type case with named and unnamed groups."""

    def test_returns_one_key_per_super_type(self):
        """Result contains exactly one entry per unique super-type label."""
        actual = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        predicted = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        labels = np.array([0, 0, 1, 1, 2, 2])
        result = rmse_by_super_type(actual, predicted, labels)
        assert set(result.keys()) == {"super_type_0", "super_type_1", "super_type_2"}

    def test_uses_display_names_when_provided(self):
        """Keys use the supplied display names rather than fallback IDs."""
        actual = np.array([0.0, 0.1, 0.0, 0.1])
        predicted = np.zeros(4)
        labels = np.array([0, 0, 1, 1])
        names = {0: "Rural Evangelical", 1: "Urban Professional"}
        result = rmse_by_super_type(actual, predicted, labels, super_type_names=names)
        assert "Rural Evangelical" in result
        assert "Urban Professional" in result
        assert len(result) == 2

    def test_rmse_values_are_correct(self):
        """RMSE equals sqrt(mean((actual-predicted)^2)) per group."""
        # Group 0: errors [0, 0.2] -> rmse = sqrt((0 + 0.04)/2) = sqrt(0.02) ≈ 0.1414
        # Group 1: errors [0.3]    -> rmse = 0.3
        actual = np.array([1.0, 1.2, 2.0])
        predicted = np.array([1.0, 1.0, 1.7])
        labels = np.array([0, 0, 1])
        result = rmse_by_super_type(actual, predicted, labels)
        np.testing.assert_allclose(result["super_type_0"], np.sqrt(0.02), rtol=1e-6)
        np.testing.assert_allclose(result["super_type_1"], 0.3, rtol=1e-6)


class TestRmseBySuperTypeSingleSuperType:
    """Edge case: all counties belong to one super-type."""

    def test_single_super_type_returns_one_entry(self):
        actual = np.array([0.1, 0.2, 0.3])
        predicted = np.array([0.1, 0.2, 0.3])
        labels = np.zeros(3, dtype=int)
        result = rmse_by_super_type(actual, predicted, labels)
        assert len(result) == 1
        assert "super_type_0" in result

    def test_single_super_type_perfect_prediction_gives_zero_rmse(self):
        actual = np.array([0.5, 0.6, 0.7, 0.8])
        predicted = actual.copy()
        labels = np.ones(4, dtype=int)
        result = rmse_by_super_type(actual, predicted, labels)
        np.testing.assert_allclose(result["super_type_1"], 0.0, atol=1e-12)


class TestRmseBySuperTypeEmptyInput:
    """Empty arrays should return an empty dict, not raise."""

    def test_empty_arrays_return_empty_dict(self):
        actual = np.array([], dtype=float)
        predicted = np.array([], dtype=float)
        labels = np.array([], dtype=int)
        result = rmse_by_super_type(actual, predicted, labels)
        assert result == {}


class TestRmseBySuperTypeThresholdFlagging:
    """Groups with RMSE > 0.10 should emit a WARNING log; others should not."""

    def test_high_rmse_triggers_warning(self, caplog):
        # Constant error of 0.15 -> RMSE = 0.15 > 0.10
        actual = np.array([0.0, 0.0, 0.0])
        predicted = np.full(3, 0.15)
        labels = np.zeros(3, dtype=int)
        with caplog.at_level(logging.WARNING, logger="src.validation.validate_types"):
            rmse_by_super_type(actual, predicted, labels)
        assert any("High RMSE" in r.message for r in caplog.records)

    def test_low_rmse_does_not_trigger_warning(self, caplog):
        # Constant error of 0.05 -> RMSE = 0.05 < 0.10
        actual = np.array([0.0, 0.0, 0.0])
        predicted = np.full(3, 0.05)
        labels = np.zeros(3, dtype=int)
        with caplog.at_level(logging.WARNING, logger="src.validation.validate_types"):
            rmse_by_super_type(actual, predicted, labels)
        assert not any("High RMSE" in r.message for r in caplog.records)

    def test_threshold_boundary_exactly_at_limit_is_not_flagged(self, caplog):
        """RMSE == threshold is NOT flagged (strict > comparison)."""
        actual = np.array([0.0])
        predicted = np.array([RMSE_FLAG_THRESHOLD])
        labels = np.array([0])
        with caplog.at_level(logging.WARNING, logger="src.validation.validate_types"):
            rmse_by_super_type(actual, predicted, labels)
        assert not any("High RMSE" in r.message for r in caplog.records)


class TestRmseBySuperTypeAllSamePredictionError:
    """When all predictions are wrong by the same amount, RMSE == that amount."""

    def test_uniform_error_all_groups(self):
        error = 0.07
        actual = np.zeros(6)
        predicted = np.full(6, error)
        labels = np.array([0, 0, 1, 1, 2, 2])
        result = rmse_by_super_type(actual, predicted, labels)
        for rmse in result.values():
            np.testing.assert_allclose(rmse, error, rtol=1e-6)


class TestRmseBySuperTypeMissingAssignments:
    """Shape mismatch between arrays should raise ValueError."""

    def test_mismatched_shapes_raise_value_error(self):
        actual = np.array([0.1, 0.2, 0.3])
        predicted = np.array([0.1, 0.2])  # wrong length
        labels = np.array([0, 0, 1])
        with pytest.raises(ValueError, match="same shape"):
            rmse_by_super_type(actual, predicted, labels)

    def test_labels_shorter_than_actual_raises(self):
        actual = np.array([0.1, 0.2, 0.3])
        predicted = np.array([0.1, 0.2, 0.3])
        labels = np.array([0, 0])  # wrong length
        with pytest.raises(ValueError, match="same shape"):
            rmse_by_super_type(actual, predicted, labels)


class TestRmseBySuperTypeLargeVsSmallGroup:
    """Larger groups are not systematically penalised; RMSE is scale-independent."""

    def test_large_group_same_error_as_small_group(self):
        # Large group (100 counties): constant error 0.08
        # Small group (3 counties): same constant error 0.08
        error = 0.08
        n_large = 100
        actual_large = np.zeros(n_large)
        pred_large = np.full(n_large, error)
        labels_large = np.zeros(n_large, dtype=int)

        actual_small = np.zeros(3)
        pred_small = np.full(3, error)
        labels_small = np.ones(3, dtype=int)

        actual = np.concatenate([actual_large, actual_small])
        predicted = np.concatenate([pred_large, pred_small])
        labels = np.concatenate([labels_large, labels_small])

        result = rmse_by_super_type(actual, predicted, labels)
        np.testing.assert_allclose(result["super_type_0"], error, rtol=1e-6)
        np.testing.assert_allclose(result["super_type_1"], error, rtol=1e-6)

    def test_large_group_can_have_lower_rmse_than_small(self, rng):
        """Larger group with tight errors vs small group with large errors."""
        actual_large = rng.standard_normal(200) * 0.01  # near-zero actual
        pred_large = np.zeros(200)                       # near-perfect preds
        labels_large = np.zeros(200, dtype=int)

        actual_small = np.array([0.0, 0.0, 0.0])
        pred_small = np.array([0.5, 0.5, 0.5])          # large errors
        labels_small = np.ones(3, dtype=int)

        actual = np.concatenate([actual_large, actual_small])
        predicted = np.concatenate([pred_large, pred_small])
        labels = np.concatenate([labels_large, labels_small])

        result = rmse_by_super_type(actual, predicted, labels)
        assert result["super_type_0"] < result["super_type_1"]
