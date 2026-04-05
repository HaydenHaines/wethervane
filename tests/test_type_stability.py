"""Tests for the type stability seed experiment (S307).

Covers:
- Temperature soft membership computation
- Hungarian alignment (align_labels_to_reference)
- LOO r computation
- ARI/NMI on known inputs (sklearn)
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from scripts.experiment_type_stability import (
    align_labels_to_reference,
    compute_loo_r,
    temperature_soft_membership,
)


# ---------------------------------------------------------------------------
# Temperature soft membership tests
# ---------------------------------------------------------------------------


class TestTemperatureSoftMembership:
    def test_rows_sum_to_one(self):
        dists = np.array([[1.0, 2.0, 3.0], [0.5, 0.5, 5.0]])
        scores = temperature_soft_membership(dists, T=10.0)
        np.testing.assert_allclose(scores.sum(axis=1), [1.0, 1.0], atol=1e-6)

    def test_closer_centroid_gets_higher_weight(self):
        dists = np.array([[0.1, 1.0, 5.0]])
        scores = temperature_soft_membership(dists, T=10.0)
        assert scores[0, 0] > scores[0, 1] > scores[0, 2]

    def test_equal_distances_give_equal_weights(self):
        dists = np.array([[1.0, 1.0, 1.0]])
        scores = temperature_soft_membership(dists, T=10.0)
        np.testing.assert_allclose(scores[0], [1 / 3, 1 / 3, 1 / 3], atol=1e-6)

    def test_high_temperature_approaches_hard_assignment(self):
        dists = np.array([[0.5, 1.0, 2.0]])
        scores = temperature_soft_membership(dists, T=500.0)
        assert scores[0, 0] == pytest.approx(1.0)
        assert scores[0, 1] == pytest.approx(0.0)

    def test_output_shape(self):
        dists = np.random.default_rng(0).uniform(0.1, 5, size=(20, 8))
        scores = temperature_soft_membership(dists, T=5.0)
        assert scores.shape == (20, 8)

    def test_all_nonnegative(self):
        dists = np.random.default_rng(1).uniform(0.01, 10, size=(50, 10))
        scores = temperature_soft_membership(dists, T=10.0)
        assert np.all(scores >= 0)


# ---------------------------------------------------------------------------
# Hungarian alignment tests
# ---------------------------------------------------------------------------


class TestAlignLabelsToReference:
    def test_identical_labels_perfect_agreement(self):
        labels = np.array([0, 1, 2, 0, 1, 2])
        remapped = align_labels_to_reference(labels, labels.copy(), j=3)
        np.testing.assert_array_equal(remapped, labels)

    def test_permuted_labels_recovers_original(self):
        ref = np.array([0, 0, 1, 1, 2, 2])
        other = np.array([1, 1, 0, 0, 2, 2])  # 0↔1 swapped
        remapped = align_labels_to_reference(ref, other, j=3)
        np.testing.assert_array_equal(remapped, ref)

    def test_agreement_after_align(self):
        ref = np.array([0, 0, 1, 1, 2, 2])
        other = np.array([1, 1, 0, 0, 2, 2])
        remapped = align_labels_to_reference(ref, other, j=3)
        agreement = np.mean(remapped == ref)
        assert agreement == pytest.approx(1.0)

    def test_random_labels_low_agreement(self):
        rng = np.random.default_rng(0)
        ref = rng.integers(0, 10, size=200)
        other = rng.integers(0, 10, size=200)
        remapped = align_labels_to_reference(ref, other, j=10)
        agreement = np.mean(remapped == ref)
        assert agreement < 0.3  # Random = ~10%


# ---------------------------------------------------------------------------
# ARI / NMI on known inputs (sklearn baseline)
# ---------------------------------------------------------------------------


class TestAriNmiKnownInputs:
    def test_ari_identical_is_one(self):
        labels = np.array([0, 1, 2, 0, 1, 2, 0])
        assert adjusted_rand_score(labels, labels) == pytest.approx(1.0)

    def test_ari_random_near_zero(self):
        rng = np.random.default_rng(1)
        a = rng.integers(0, 20, 300)
        b = rng.integers(0, 20, 300)
        assert abs(adjusted_rand_score(a, b)) < 0.1

    def test_nmi_identical_is_one(self):
        labels = np.array([0, 1, 2, 0, 1, 2])
        assert normalized_mutual_info_score(labels, labels) == pytest.approx(1.0)

    def test_ari_symmetric(self):
        rng = np.random.default_rng(7)
        a = rng.integers(0, 5, 50)
        b = rng.integers(0, 5, 50)
        assert adjusted_rand_score(a, b) == pytest.approx(adjusted_rand_score(b, a))


# ---------------------------------------------------------------------------
# LOO r computation
# ---------------------------------------------------------------------------


class TestComputeLooR:
    def test_perfect_types_give_high_r(self):
        """When type structure perfectly matches holdout, LOO r should be high."""
        rng = np.random.default_rng(42)
        N, J = 100, 5

        # Create holdout that's a function of type membership
        true_type = rng.integers(0, J, size=N)
        type_effects = rng.normal(0, 1, size=J)
        holdout = np.zeros((N, 1))
        for i in range(N):
            holdout[i, 0] = type_effects[true_type[i]] + rng.normal(0, 0.1)

        # Weights: one-hot encoding of true type
        weights = np.zeros((N, J))
        weights[np.arange(N), true_type] = 1.0

        # Raw shift matrix (used for county priors)
        shift_raw = rng.normal(size=(N, 10))

        r = compute_loo_r(weights, holdout, shift_raw)
        assert r > 0.8

    def test_random_weights_give_low_r(self):
        rng = np.random.default_rng(0)
        N, J = 200, 10
        weights = rng.dirichlet(np.ones(J), size=N)
        holdout = rng.normal(size=(N, 3))
        shift_raw = rng.normal(size=(N, 10))
        r = compute_loo_r(weights, holdout, shift_raw)
        assert abs(r) < 0.3

    def test_output_is_scalar(self):
        rng = np.random.default_rng(5)
        weights = rng.dirichlet(np.ones(3), size=20)
        holdout = rng.normal(size=(20, 2))
        shift_raw = rng.normal(size=(20, 5))
        r = compute_loo_r(weights, holdout, shift_raw)
        assert isinstance(r, float)
