"""Estimate θ_national — citizen sentiment per type from all polls.

Solves: minimize Σ_p (1/σ_p²)(y_p - W_p·θ)² + λ·||θ - θ_prior||²
This is weighted Ridge regression with prior centering.
Closed-form: θ = (WᵀΣ⁻¹W + λI)⁻¹(WᵀΣ⁻¹y + λ·θ_prior)
"""

import numpy as np


def estimate_theta_national(
    W_polls: np.ndarray,      # (n_polls, J) — type composition per poll
    y_polls: np.ndarray,      # (n_polls,) — observed Dem shares
    sigma_polls: np.ndarray,  # (n_polls,) — poll noise (std dev)
    theta_prior: np.ndarray,  # (J,) — model prior
    lam: float = 1.0,         # regularization strength (prior trust)
) -> np.ndarray:
    """Estimate national citizen sentiment θ_national from pooled polls.

    With zero polls, returns θ_prior exactly.
    With many precise polls, converges to data-driven estimate.
    λ controls the trade-off.
    """
    J = len(theta_prior)

    if len(y_polls) == 0:
        return theta_prior.copy()

    # Inverse-variance weights: (n_polls,)
    inv_var = 1.0 / (sigma_polls ** 2)

    # Weighted normal equations
    # A = WᵀΣ⁻¹W + λI
    # b = WᵀΣ⁻¹y + λ·θ_prior
    W_weighted = W_polls * inv_var[:, np.newaxis]  # (n_polls, J) weighted
    A = W_weighted.T @ W_polls + lam * np.eye(J)
    b = W_weighted.T @ y_polls + lam * theta_prior

    theta = np.linalg.solve(A, b)
    return theta
