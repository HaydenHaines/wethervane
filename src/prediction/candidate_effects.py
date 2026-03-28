"""Estimate δ_race — candidate effects per type for a specific race.

Solves: minimize Σ_p (1/σ_p²)(r_p - W_p·δ)² + μ·||δ||²
where r_p = y_p - W_p·θ_national (residual from national sentiment).

δ represents how this specific candidate matchup deviates from the
generic national environment for each type.
"""

import numpy as np


def estimate_delta_race(
    W_polls: np.ndarray,      # (n_polls, J) — type composition per poll
    residuals: np.ndarray,    # (n_polls,) — y_p - W_p · θ_national
    sigma_polls: np.ndarray,  # (n_polls,) — poll noise
    J: int,                   # number of types
    mu: float = 1.0,          # regularization toward zero
) -> np.ndarray:
    """Estimate candidate effect δ for a single race.

    With zero polls, returns zeros (no candidate effect detected).
    With many polls, captures the race-specific deviation from national sentiment.
    μ controls shrinkage toward zero.
    """
    if len(residuals) == 0:
        return np.zeros(J)

    inv_var = 1.0 / (sigma_polls ** 2)

    # Weighted normal equations (same structure as θ_national but centered at 0)
    W_weighted = W_polls * inv_var[:, np.newaxis]
    A = W_weighted.T @ W_polls + mu * np.eye(J)
    b = W_weighted.T @ residuals  # No prior term — centered at zero

    delta = np.linalg.solve(A, b)
    return delta
