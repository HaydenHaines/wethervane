"""Tests for θ_national estimation — citizen sentiment from pooled polls."""

import numpy as np
import pytest

from src.prediction.national_environment import estimate_theta_national


def test_no_polls_returns_prior():
    """With zero polls, θ_national should equal θ_prior."""
    J = 5
    theta_prior = np.array([0.5, 0.4, 0.6, 0.45, 0.55])
    theta = estimate_theta_national(
        W_polls=np.empty((0, J)),
        y_polls=np.empty(0),
        sigma_polls=np.empty(0),
        theta_prior=theta_prior,
        lam=1.0,
    )
    np.testing.assert_allclose(theta, theta_prior)


def test_single_poll_moves_toward_observation():
    """One poll should pull θ_national toward the observed value."""
    J = 2
    theta_prior = np.array([0.5, 0.5])
    W = np.array([[0.7, 0.3]])  # Poll mostly observes type 0
    y = np.array([0.6])  # Poll says 60% Dem
    sigma = np.array([0.02])
    theta = estimate_theta_national(W, y, sigma, theta_prior, lam=1.0)
    # Type 0 should move toward 0.6 more than type 1
    assert theta[0] > theta_prior[0]


def test_many_polls_dominate_prior():
    """With many precise polls, θ_national should be data-driven."""
    J = 2
    theta_prior = np.array([0.5, 0.5])
    n_polls = 100
    W = np.tile([0.6, 0.4], (n_polls, 1))
    y = np.full(n_polls, 0.55)
    sigma = np.full(n_polls, 0.01)  # Very precise
    theta = estimate_theta_national(W, y, sigma, theta_prior, lam=0.1)
    # W · θ should be close to 0.55
    pred = W[0] @ theta
    assert abs(pred - 0.55) < 0.02


def test_lambda_controls_prior_weight():
    """Higher λ should keep θ_national closer to θ_prior."""
    J = 3
    theta_prior = np.array([0.5, 0.5, 0.5])
    W = np.array([[1.0, 0.0, 0.0]])
    y = np.array([0.7])
    sigma = np.array([0.02])

    theta_low_lam = estimate_theta_national(W, y, sigma, theta_prior, lam=0.01)
    theta_high_lam = estimate_theta_national(W, y, sigma, theta_prior, lam=100.0)

    # Low λ: θ moves more toward data
    assert abs(theta_low_lam[0] - 0.7) < abs(theta_high_lam[0] - 0.7)
    # High λ: θ stays closer to prior
    assert abs(theta_high_lam[0] - 0.5) < abs(theta_low_lam[0] - 0.5)


def test_output_shape():
    J = 10
    theta_prior = np.random.rand(J)
    W = np.random.rand(5, J)
    W = W / W.sum(axis=1, keepdims=True)
    y = np.random.rand(5) * 0.3 + 0.35
    sigma = np.full(5, 0.03)
    theta = estimate_theta_national(W, y, sigma, theta_prior, lam=1.0)
    assert theta.shape == (J,)
