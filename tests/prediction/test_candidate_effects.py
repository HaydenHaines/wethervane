"""Tests for δ_race estimation — candidate effects from residuals."""

import numpy as np
import pytest

from src.prediction.candidate_effects import estimate_delta_race


def test_no_polls_returns_zeros():
    """With zero polls, δ_race should be all zeros."""
    J = 5
    delta = estimate_delta_race(
        W_polls=np.empty((0, J)),
        residuals=np.empty(0),
        sigma_polls=np.empty(0),
        J=J,
        mu=1.0,
    )
    np.testing.assert_allclose(delta, np.zeros(J))


def test_single_poll_small_delta():
    """One poll with strong regularization should produce small δ."""
    J = 3
    W = np.array([[0.5, 0.3, 0.2]])
    residuals = np.array([0.05])  # Race is 5pp above national
    sigma = np.array([0.03])
    delta = estimate_delta_race(W, residuals, sigma, J, mu=1000.0)
    assert np.all(np.abs(delta) < 0.05)  # Strong regularization keeps δ small


def test_many_polls_larger_delta():
    """Many polls with weak regularization should produce δ close to residual."""
    J = 2
    n = 50
    W = np.tile([0.6, 0.4], (n, 1))
    residuals = np.full(n, 0.03)  # Consistent 3pp above national
    sigma = np.full(n, 0.02)
    delta = estimate_delta_race(W, residuals, sigma, J, mu=0.01)
    pred_residual = W[0] @ delta
    assert abs(pred_residual - 0.03) < 0.01


def test_mu_controls_shrinkage():
    """Higher μ should shrink δ toward zero."""
    J = 3
    W = np.array([[0.5, 0.3, 0.2]] * 5)
    residuals = np.full(5, 0.05)
    sigma = np.full(5, 0.02)

    delta_low_mu = estimate_delta_race(W, residuals, sigma, J, mu=0.01)
    delta_high_mu = estimate_delta_race(W, residuals, sigma, J, mu=100.0)

    assert np.linalg.norm(delta_high_mu) < np.linalg.norm(delta_low_mu)


def test_output_shape():
    J = 10
    W = np.random.rand(3, J)
    W = W / W.sum(axis=1, keepdims=True)
    residuals = np.random.randn(3) * 0.02
    sigma = np.full(3, 0.03)
    delta = estimate_delta_race(W, residuals, sigma, J, mu=1.0)
    assert delta.shape == (J,)
