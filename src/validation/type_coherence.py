"""Within-type vs between-type variance analysis for electoral type validation.

Provides the type_coherence() function, which measures how well discovered
types separate holdout shift dimensions — the core falsifiability test for
the KMeans type structure.
"""
from __future__ import annotations

import numpy as np


def type_coherence(
    scores: np.ndarray,
    shift_matrix: np.ndarray,
    holdout_cols: list[int],
) -> dict:
    """Within-type vs between-type variance on holdout shifts.

    For each holdout dimension:
    1. Assign each county to its dominant type (highest absolute score).
    2. Compute within-type variance (mean of per-type variances).
    3. Compute between-type variance (variance of type means).
    4. Compute ratio = between / (within + between).

    A higher ratio means types explain more holdout variance.

    Parameters
    ----------
    scores : ndarray of shape (N, J)
        Rotated county type scores (soft membership).
    shift_matrix : ndarray of shape (N, D)
        Full shift matrix (all dimensions including holdout).
    holdout_cols : list[int]
        Column indices of holdout dimensions in shift_matrix.

    Returns
    -------
    dict with keys:
        "mean_ratio"     -- float, mean coherence ratio across holdout dims
        "per_dim_ratios" -- list[float], one ratio per holdout dim
    """
    dominant_types = np.argmax(np.abs(scores), axis=1)
    unique_types = np.unique(dominant_types)

    per_dim_ratios: list[float] = []

    for col in holdout_cols:
        values = shift_matrix[:, col]

        # Per-type variances and means
        type_variances = []
        type_means = []
        for t in unique_types:
            mask = dominant_types == t
            if mask.sum() < 2:
                # Single-county type has zero variance; skip from within-var
                type_variances.append(0.0)
            else:
                type_variances.append(float(np.var(values[mask], ddof=0)))
            type_means.append(float(np.mean(values[mask])))

        type_means_arr = np.array(type_means)

        within_var = float(np.mean(type_variances))
        # Between-type variance: variance of type means (unweighted)
        between_var = float(np.var(type_means_arr, ddof=0))

        total = within_var + between_var
        if total < 1e-12:
            ratio = 0.0
        else:
            ratio = between_var / total

        # Clamp to [0, 1] for numerical safety
        ratio = float(np.clip(ratio, 0.0, 1.0))
        per_dim_ratios.append(ratio)

    mean_ratio = float(np.mean(per_dim_ratios)) if per_dim_ratios else 0.0

    return {"mean_ratio": mean_ratio, "per_dim_ratios": per_dim_ratios}
