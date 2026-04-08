"""Apply voter behavior parameters to tract-level Dem share priors.

Integration helper between the behavior layer (τ, δ) and the prediction pipeline.
For presidential elections this is a no-op; for off-cycle elections it adjusts
priors to reflect both differential turnout (via τ) and residual preference
differences (via δ).

Adjustment logic (off-cycle only):
  1. Reweight the type composition of each tract by τ.
     Types with lower off-cycle turnout (τ < 1) shrink in the effective electorate.
  2. Compute each tract's δ-shift as the τ-reweighted mean of type δ values.
  3. Add the shift to the prior, clip to [0, 1].

This preserves the spirit of the model: types are nouns, τ and δ are verbs that
tell us how each type *expresses itself differently* in non-presidential elections.
"""
from __future__ import annotations

import numpy as np


def adjust_priors_for_cycle(
    priors: np.ndarray,
    tau: np.ndarray,
    delta: np.ndarray,
    is_presidential: bool = False,
) -> np.ndarray:
    """Adjust tract-level Dem share priors for election cycle type.

    Args:
        priors: Dem share priors, shape (n_tracts,) OR (n_tracts, J) type-score × prior.
                If 1-D, treated as a flat prior per tract.
                If 2-D, the second dimension is the type score matrix; the adjustment
                is applied to the first-axis values using the second-axis weights.
                Most callers pass 1-D priors; the 2-D path supports future use.
        tau: Turnout ratios per type, shape (J,).  Must match the type count used
             when generating the priors.
        delta: Choice shifts per type, shape (J,).  Same shape requirement.
        is_presidential: If True, priors are returned unchanged (presidential elections
                         use the priors directly; τ and δ were estimated relative to
                         the presidential baseline, so no correction is needed).

    Returns:
        Adjusted priors, same shape as input, clipped to [0, 1].

    Raises:
        ValueError: If tau and delta have different lengths.
    """
    if len(tau) != len(delta):
        raise ValueError(
            f"tau and delta must have the same length, got {len(tau)} and {len(delta)}."
        )

    if is_presidential:
        # Presidential baseline — no adjustment.  τ and δ are defined relative to
        # this baseline, so applying them here would be circular.
        return priors.copy()

    priors_1d = np.atleast_1d(priors)

    if priors_1d.ndim == 1:
        # Flat prior: no type-score information available, apply mean-τ-weighted
        # mean-δ as a scalar shift.  This is the common prediction pipeline path.
        return _adjust_flat_priors(priors_1d, tau, delta)

    if priors_1d.ndim == 2:
        # Structured prior: second dimension holds per-type scores.  Reweight by τ.
        return _adjust_scored_priors(priors_1d, tau, delta)

    raise ValueError(f"priors must be 1-D or 2-D, got shape {priors.shape}.")


def _adjust_flat_priors(
    priors: np.ndarray,
    tau: np.ndarray,
    delta: np.ndarray,
) -> np.ndarray:
    """Apply a uniform δ shift to flat 1-D priors.

    When no type-score breakdown per tract is available, we use the
    unweighted mean of δ as a universal correction.  This is the simplest
    defensible adjustment and keeps the API usable before type scores are
    available for the prediction year.

    Args:
        priors: 1-D array of Dem share priors, shape (n_tracts,).
        tau: Turnout ratios per type, shape (J,).  Not used for flat priors
             because without type scores we cannot compute τ-reweighted means.
        delta: Choice shifts per type, shape (J,).

    Returns:
        Clipped adjusted priors, shape (n_tracts,).
    """
    # τ is intentionally unused here — without per-tract type scores we cannot
    # apply the τ reweighting step.  A future improvement would take a national
    # type-score distribution as input to estimate the aggregate τ effect.
    _ = tau

    mean_delta = delta.mean()
    return np.clip(priors + mean_delta, 0.0, 1.0)


def _adjust_scored_priors(
    priors: np.ndarray,
    tau: np.ndarray,
    delta: np.ndarray,
) -> np.ndarray:
    """Apply τ-reweighted δ adjustment to structured 2-D priors.

    Args:
        priors: 2-D array, shape (n_tracts, J), where each row is the type-score
                distribution for one tract multiplied by the tract's Dem share prior.
                The caller is responsible for encoding the prior in this format.
        tau: Turnout ratios per type, shape (J,).
        delta: Choice shifts per type, shape (J,).

    Returns:
        Clipped adjusted priors, shape (n_tracts, J).
    """
    n_tracts, n_types = priors.shape
    if n_types != len(tau):
        raise ValueError(
            f"priors second dimension ({n_types}) must match len(tau) ({len(tau)})."
        )

    # Reweight type scores by τ — types with lower off-cycle turnout shrink in the
    # effective electorate composition.
    offcycle_weights = priors * tau[np.newaxis, :]
    row_sums = offcycle_weights.sum(axis=1, keepdims=True)
    # Avoid division by zero for tracts with all-zero scores (shouldn't happen in
    # practice but defensive coding here costs nothing).
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    offcycle_norm = offcycle_weights / row_sums

    # Each tract's δ-adjustment = weighted sum of δ_j by its τ-adjusted type composition.
    adjustment = offcycle_norm @ delta  # shape (n_tracts,)

    # Broadcast the scalar adjustment across the J columns.
    return np.clip(priors + adjustment[:, np.newaxis], 0.0, 1.0)
