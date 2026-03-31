"""Type stability analysis: subspace angle between types from two time windows.

Measures whether the type structure discovered from early elections
(window A) and late elections (window B) spans the same subspace.
A small principal angle (< 30 degrees) indicates the types are stable
across time — they are not an artifact of the specific years used for
discovery.
"""
from __future__ import annotations

import numpy as np


def type_stability(
    shift_matrix: np.ndarray,
    window_a_cols: list[int],
    window_b_cols: list[int],
    j: int,
) -> dict:
    """Subspace angle between types discovered from two time windows.

    1. Fit SVD+varimax on window A -> scores_a (N x J).
    2. Fit SVD+varimax on window B -> scores_b (N x J).
    3. Compute principal angles between column spaces of scores_a and scores_b.
    4. Report max angle (degrees).

    Parameters
    ----------
    shift_matrix : ndarray of shape (N, D)
        Full shift matrix. Windows are column subsets.
    window_a_cols : list[int]
        Column indices for window A (e.g., 2000-2012 pairs).
    window_b_cols : list[int]
        Column indices for window B (e.g., 2012-2024 pairs).
    j : int
        Number of types to discover in each window.

    Returns
    -------
    dict with keys:
        "max_angle_degrees"  -- float
        "mean_angle_degrees" -- float
        "stable"             -- bool, True when max_angle < 30 degrees
    """
    from src.discovery.run_type_discovery import discover_types

    matrix_a = shift_matrix[:, window_a_cols]
    matrix_b = shift_matrix[:, window_b_cols]

    result_a = discover_types(matrix_a, j=j, random_state=42)
    result_b = discover_types(matrix_b, j=j, random_state=42)

    scores_a = result_a.scores  # N x J
    scores_b = result_b.scores  # N x J

    # Compute principal angles between column spaces via SVD of Q_a^T Q_b
    # QR-decompose each score matrix to get orthonormal bases
    Q_a, _ = np.linalg.qr(scores_a)
    Q_b, _ = np.linalg.qr(scores_b)

    # Only keep J columns (qr returns min(N,J) columns with full mode)
    Q_a = Q_a[:, :j]
    Q_b = Q_b[:, :j]

    # Singular values of Q_a^T @ Q_b are cosines of principal angles
    M = Q_a.T @ Q_b
    sigma = np.linalg.svd(M, compute_uv=False)

    # Clamp to [-1, 1] to guard against numerical drift
    sigma = np.clip(sigma, -1.0, 1.0)
    angles_rad = np.arccos(np.abs(sigma))
    angles_deg = np.degrees(angles_rad)

    max_angle = float(np.max(angles_deg))
    mean_angle = float(np.mean(angles_deg))

    return {
        "max_angle_degrees": max_angle,
        "mean_angle_degrees": mean_angle,
        "stable": max_angle < 30.0,
    }
