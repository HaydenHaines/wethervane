"""Holdout accuracy metrics for electoral type validation.

Provides four accuracy functions at increasing levels of sophistication:
- holdout_accuracy: basic type-mean prior
- holdout_accuracy_county_prior: county historical mean + type adjustment
- holdout_accuracy_county_prior_loo: same as above but leave-one-out
- rmse_by_super_type: per-group RMSE diagnostics

Ridge-based functions (holdout_accuracy_ridge, holdout_accuracy_ridge_augmented)
live in holdout_accuracy_ridge.py and are re-exported here for backwards
compatibility.

These functions implement the LOO honesty rule discovered in S196: the
standard holdout metric inflates by ~0.22 due to type self-prediction in
small types. The LOO variants produce honest generalization estimates.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.stats import pearsonr

from src.validation.holdout_accuracy_ridge import (  # noqa: F401
    holdout_accuracy_ridge,
    holdout_accuracy_ridge_augmented,
)

log = logging.getLogger(__name__)

# RMSE threshold above which a super-type is flagged as poorly predicted
RMSE_FLAG_THRESHOLD = 0.10


def holdout_accuracy(
    scores: np.ndarray,
    shift_matrix: np.ndarray,
    holdout_cols: list[int],
    dominant_types: np.ndarray,
) -> dict:
    """Holdout Pearson r: predict holdout shifts from type means.

    For each holdout column:
    1. Compute type-level mean of that column (weighted by absolute scores).
    2. Predict each county's value = weighted sum of type means.
    3. Compute Pearson r between predicted and actual.

    Parameters
    ----------
    scores : ndarray of shape (N, J)
        Rotated county type scores (soft membership).
    shift_matrix : ndarray of shape (N, D)
        Full shift matrix.
    holdout_cols : list[int]
        Column indices of holdout dimensions in shift_matrix.
    dominant_types : ndarray of shape (N,)
        Dominant type index per county (argmax of abs scores).

    Returns
    -------
    dict with keys:
        "mean_r"    -- float, mean Pearson r across holdout dims
        "per_dim_r" -- list[float], one r per holdout dim
    """
    n, j = scores.shape

    # Normalize absolute scores to weights summing to 1 per county
    abs_scores = np.abs(scores)
    row_sums = abs_scores.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    weights = abs_scores / row_sums  # N x J

    # Type-level weighted means: type_means[t, d] = weighted mean over counties
    weight_sums = weights.sum(axis=0)  # J
    weight_sums = np.where(weight_sums == 0, 1.0, weight_sums)

    per_dim_r: list[float] = []

    for col in holdout_cols:
        actual = shift_matrix[:, col]

        # Weighted type means for this column
        type_means = (weights.T @ actual) / weight_sums  # J

        # Predicted: weighted sum of type means per county
        predicted = weights @ type_means  # N

        if np.std(actual) < 1e-10 or np.std(predicted) < 1e-10:
            per_dim_r.append(0.0)
        else:
            r, _ = pearsonr(actual, predicted)
            per_dim_r.append(float(np.clip(r, -1.0, 1.0)))

    mean_r = float(np.mean(per_dim_r)) if per_dim_r else 0.0

    return {"mean_r": mean_r, "per_dim_r": per_dim_r}


def holdout_accuracy_county_prior(
    scores: np.ndarray,
    shift_matrix: np.ndarray,
    training_cols: list[int],
    holdout_cols: list[int],
) -> dict:
    """Holdout accuracy using county-level priors + type covariance adjustment.

    This mirrors the production prediction pipeline where:
    - Each county's prior = its own historical mean shift (from training cols)
    - Types determine only the comovement adjustment

    For each holdout column:
    1. Compute each county's historical mean shift from training columns.
    2. Compute type-level mean shift for training columns.
    3. Compute type-level mean shift for holdout column.
    4. Type adjustment = holdout type mean - training type mean (per type).
    5. County prediction = county training mean + score-weighted type adjustment.
    6. Compute Pearson r and RMSE between predicted and actual.

    Parameters
    ----------
    scores : ndarray of shape (N, J)
        County type scores (soft membership).
    shift_matrix : ndarray of shape (N, D)
        Full shift matrix (training + holdout columns).
    training_cols : list[int]
        Column indices of training dimensions.
    holdout_cols : list[int]
        Column indices of holdout dimensions.

    Returns
    -------
    dict with keys:
        "mean_r"      -- float, mean Pearson r across holdout dims
        "per_dim_r"   -- list[float], one r per holdout dim
        "mean_rmse"   -- float, mean RMSE across holdout dims
        "per_dim_rmse" -- list[float], one RMSE per holdout dim
    """
    n, j = scores.shape

    # Normalize absolute scores to weights summing to 1 per county
    abs_scores = np.abs(scores)
    row_sums = abs_scores.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    weights = abs_scores / row_sums  # N x J

    weight_sums_per_type = weights.sum(axis=0)  # J
    weight_sums_per_type = np.where(weight_sums_per_type == 0, 1.0, weight_sums_per_type)

    # County-level training mean (each county's own historical baseline)
    training_data = shift_matrix[:, training_cols]
    county_training_means = training_data.mean(axis=1)  # N

    # Type-level training mean
    type_training_means = (weights.T @ county_training_means) / weight_sums_per_type  # J

    per_dim_r: list[float] = []
    per_dim_rmse: list[float] = []

    for col in holdout_cols:
        actual = shift_matrix[:, col]

        # Type-level holdout mean
        type_holdout_means = (weights.T @ actual) / weight_sums_per_type  # J

        # Type adjustment: how much each type shifted from training to holdout
        type_adjustment = type_holdout_means - type_training_means  # J

        # County prediction = own baseline + score-weighted type adjustment
        county_adjustment = (weights * type_adjustment[None, :]).sum(axis=1)  # N
        predicted = county_training_means + county_adjustment

        if np.std(actual) < 1e-10 or np.std(predicted) < 1e-10:
            per_dim_r.append(0.0)
        else:
            r, _ = pearsonr(actual, predicted)
            per_dim_r.append(float(np.clip(r, -1.0, 1.0)))

        rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
        per_dim_rmse.append(rmse)

    mean_r = float(np.mean(per_dim_r)) if per_dim_r else 0.0
    mean_rmse = float(np.mean(per_dim_rmse)) if per_dim_rmse else 0.0

    return {
        "mean_r": mean_r,
        "per_dim_r": per_dim_r,
        "mean_rmse": mean_rmse,
        "per_dim_rmse": per_dim_rmse,
    }


def holdout_accuracy_county_prior_loo(
    scores: np.ndarray,
    shift_matrix: np.ndarray,
    training_cols: list[int],
    holdout_cols: list[int],
) -> dict:
    """Leave-one-out holdout accuracy using county-level priors.

    Same as holdout_accuracy_county_prior but removes each county from the
    type mean computation before predicting it. This eliminates inflation
    from small types where a county dominates its own type mean.

    Returns
    -------
    dict with keys:
        "mean_r"       -- float, mean LOO Pearson r across holdout dims
        "per_dim_r"    -- list[float], one r per holdout dim
        "mean_rmse"    -- float, mean LOO RMSE across holdout dims
        "per_dim_rmse" -- list[float], one RMSE per holdout dim
    """
    n, j = scores.shape

    abs_scores = np.abs(scores)
    row_sums = abs_scores.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    weights = abs_scores / row_sums

    training_data = shift_matrix[:, training_cols]
    county_training_means = training_data.mean(axis=1)

    # Precompute global weighted sums for efficient LOO
    global_weight_sums = weights.sum(axis=0)  # J
    global_weighted_train = weights.T @ county_training_means  # J

    per_dim_r: list[float] = []
    per_dim_rmse: list[float] = []

    for col in holdout_cols:
        actual = shift_matrix[:, col]
        global_weighted_hold = weights.T @ actual  # J

        predicted = np.zeros(n)
        for i in range(n):
            # LOO: subtract county i's contribution from type sums
            loo_ws = global_weight_sums - weights[i]
            loo_ws = np.where(loo_ws < 1e-12, 1e-12, loo_ws)
            loo_train = (global_weighted_train - weights[i] * county_training_means[i]) / loo_ws
            loo_hold = (global_weighted_hold - weights[i] * actual[i]) / loo_ws
            type_adj = loo_hold - loo_train
            predicted[i] = county_training_means[i] + (weights[i] * type_adj).sum()

        if np.std(actual) < 1e-10 or np.std(predicted) < 1e-10:
            per_dim_r.append(0.0)
        else:
            r, _ = pearsonr(actual, predicted)
            per_dim_r.append(float(np.clip(r, -1.0, 1.0)))

        rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
        per_dim_rmse.append(rmse)

    mean_r = float(np.mean(per_dim_r)) if per_dim_r else 0.0
    mean_rmse = float(np.mean(per_dim_rmse)) if per_dim_rmse else 0.0

    return {
        "mean_r": mean_r,
        "per_dim_r": per_dim_r,
        "mean_rmse": mean_rmse,
        "per_dim_rmse": per_dim_rmse,
    }


def rmse_by_super_type(
    actual: np.ndarray,
    predicted: np.ndarray,
    super_type_labels: np.ndarray,
    super_type_names: dict[int, str] | None = None,
    flag_threshold: float = RMSE_FLAG_THRESHOLD,
) -> dict[str, float]:
    """Compute RMSE broken down by super-type and flag high-error groups.

    Groups counties by their assigned super-type and computes root-mean-squared
    error for each group. Groups with RMSE above *flag_threshold* are flagged
    with a warning log message, which aids in identifying poorly predicted
    segments of the electorate.

    Parameters
    ----------
    actual : ndarray of shape (N,)
        True holdout shift values for each county.
    predicted : ndarray of shape (N,)
        Model-predicted shift values aligned with *actual*.
    super_type_labels : ndarray of shape (N,) of int
        Super-type assignment for each county (integer IDs).
    super_type_names : dict[int, str] | None
        Optional mapping from super-type ID to a display name.  If None,
        keys in the returned dict are formatted as ``"super_type_{id}"``.
    flag_threshold : float
        RMSE value above which a super-type is flagged as poorly predicted.
        Default 0.10 (10 pp log-odds shift).

    Returns
    -------
    dict[str, float]
        Keys are super-type names (or ``"super_type_{id}"`` fallbacks), values
        are RMSE floats.  An empty dict is returned when *actual* is empty.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    super_type_labels = np.asarray(super_type_labels)

    if actual.size == 0:
        return {}

    if actual.shape != predicted.shape or actual.shape != super_type_labels.shape:
        raise ValueError(
            "actual, predicted, and super_type_labels must have the same shape; "
            f"got {actual.shape}, {predicted.shape}, {super_type_labels.shape}"
        )

    unique_ids = np.unique(super_type_labels)
    result: dict[str, float] = {}

    for st_id in unique_ids:
        mask = super_type_labels == st_id
        if mask.sum() == 0:
            continue

        squared_errors = (actual[mask] - predicted[mask]) ** 2
        rmse = float(np.sqrt(np.mean(squared_errors)))

        # Resolve display name
        if super_type_names is not None and st_id in super_type_names:
            name = super_type_names[int(st_id)]
        else:
            name = f"super_type_{int(st_id)}"

        result[name] = rmse

        if rmse > flag_threshold:
            log.warning(
                "High RMSE for super-type '%s' (id=%d, n=%d): %.4f > %.4f threshold",
                name,
                int(st_id),
                int(mask.sum()),
                rmse,
                flag_threshold,
            )

    return result
