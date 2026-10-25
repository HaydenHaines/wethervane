"""Scores for finite weighted predictive share distributions, without resampling.

These functions accept outcome values only at evaluation time. They do not load
data, fit models or decide whether a forecast is eligible for endpoint scoring.
"""

import numpy as np


def _distribution(values, weights):
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    if values.ndim != 1 or not len(values) or weights.shape != values.shape:
        raise ValueError("One-dimensional shares with a weight per point required")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Finite shares in [0,1] required")
    if not np.isfinite(weights).all() or (weights < 0).any() or not (weights > 0).any():
        raise ValueError("Finite nonnegative weights with positive mass required")
    weights = weights / weights.max()
    weights /= weights.sum()
    positive = weights > 0
    order = np.argsort(values[positive], kind="stable")
    return values[positive][order], weights[positive][order]


def _quantiles(values, weights, probabilities):
    probabilities = np.asarray(probabilities, float)
    if probabilities.ndim != 1 or not np.isfinite(probabilities).all():
        raise ValueError("Finite one-dimensional quantile probabilities required")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Quantile probabilities must be in [0,1]")
    indices = np.searchsorted(np.cumsum(weights), probabilities, side="left").clip(max=len(values) - 1)
    # Tiny positive masses can disappear when a cumulative sum rounds to one.
    # Endpoint quantiles still denote the extremes of positive support.
    indices[probabilities == 0] = 0
    indices[probabilities == 1] = len(values) - 1
    return values[indices]


def weighted_quantiles(values, weights, probabilities):
    """Left inverse of the weighted empirical CDF; zero-mass points are omitted."""
    values, weights = _distribution(values, weights)
    return _quantiles(values, weights, probabilities)


def _outcome(value):
    if not np.isscalar(value) or not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("A finite observed share in [0,1] is required")
    return float(value)


def _crps(values, weights, outcome):
    # Half E|X-X'| uses sorted prefix sums, avoiding a quadratic pair matrix.
    prior_mass = np.cumsum(weights) - weights
    prior_sum = np.cumsum(weights * values) - weights * values
    dispersion = np.sum(weights * (values * prior_mass - prior_sum))
    score = np.sum(weights * np.abs(values - outcome)) - dispersion
    if score < -1e-12:
        raise FloatingPointError("Weighted CRPS is negative beyond rounding tolerance")
    return float(max(0, score))


def weighted_crps(values, weights, outcome):
    """Exact finite-distribution CRPS in share units, in O(N log N) time."""
    values, weights = _distribution(values, weights)
    return _crps(values, weights, _outcome(outcome))


def score_weighted_prediction(values, weights, outcome, *, alpha=0.1):
    """Point, distribution and central-interval scores of one forecast.

    This is the empirical-distribution CRPS, without a purported iid finite-draw
    correction: HMC samples and importance points are not interchangeable iid
    posterior observations. Interval endpoints use the exact weighted CDF.
    """
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("Interval alpha must lie strictly between zero and one")
    values, weights = _distribution(values, weights)
    outcome = _outcome(outcome)
    mean = weights @ values
    variance = weights @ (values - mean) ** 2
    lower, upper = _quantiles(values, weights, [alpha / 2, 1 - alpha / 2])
    width = upper - lower
    interval_score = width + (2 / alpha) * (max(lower - outcome, 0) + max(outcome - upper, 0))
    return {
        "mean": float(mean),
        "sd": float(np.sqrt(variance)),
        "lower": float(lower),
        "upper": float(upper),
        "absolute_error": float(abs(mean - outcome)),
        "squared_error": float((mean - outcome) ** 2),
        "crps": _crps(values, weights, outcome),
        "coverage": bool(lower <= outcome <= upper),
        "interval_width": float(width),
        "interval_score": float(interval_score),
    }
