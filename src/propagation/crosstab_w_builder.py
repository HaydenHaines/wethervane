"""Crosstab-adjusted W vector construction for poll propagation.

The W vector tells the Bayesian update how to distribute a poll's signal
across J electoral types.  Without crosstab data every poll for a state
gets the same W — population-weighted type scores across the state's
counties.  This is informationally wasteful: a poll that oversampled
college-educated voters should weight college-heavy types more.

This module constructs poll-specific W vectors by:
  1.  Building a *type-demographic affinity index*: for each crosstab
      dimension (education, race, urbanicity, …) compute how much each
      type deviates from the national mean on that feature.
  2.  Using a state baseline W (population-weighted type scores inside the
      state) as the starting point.
  3.  Adjusting W for each crosstab dimension where the poll sample deviates
      from the state demographic mean: ``W_adjusted = W_base + α·Σ(δ·affinity)``.
  4.  Clipping to ≥ 0 and renormalising to sum to 1.

Why this design:
  - The adjustment is additive in W-space (not log-space) because W is a
    convex-combination weight, not a probability.  Clipping + renormalise
    preserves the simplex constraint without introducing a softmax distortion.
  - ``adjustment_strength`` (α) is a single scalar that dials between the
    baseline W (α=0) and full crosstab adjustment.  It is intentionally
    conservative at 0.3 because crosstab percentages are noisy and the
    feature→type mapping is approximate.
  - Normalising each affinity vector by its max absolute value puts all
    dimensions on a unit scale regardless of the raw demographic range.

Usage::

    from src.propagation.crosstab_w_builder import (
        build_affinity_index,
        compute_state_baseline_w,
        construct_w_row,
    )

    affinity = build_affinity_index(type_profiles, county_demographics)
    W_state  = compute_state_baseline_w(type_scores, county_pops, state_mask)
    W_poll   = construct_w_row(poll_crosstabs, W_state, affinity, state_demo_means)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default multiplicative strength of crosstab adjustments.
# α=0  → always use state baseline W (no crosstab signal used)
# α=1  → full affinity adjustment (can over-correct for noisy crosstabs)
# 0.3 is conservative: crosstab %s are single-digit-precision and approximate.
ADJUSTMENT_STRENGTH_DEFAULT: float = 0.3

# After normalising affinity vectors to unit scale, values smaller than this
# are treated as zero to avoid dividing by near-zero max during normalisation.
_AFFINITY_NORM_MIN: float = 1e-9


# ---------------------------------------------------------------------------
# Dimension map: crosstab dimension key → type_profiles column name
# ---------------------------------------------------------------------------

# Maps each crosstab demographic dimension key (as stored in poll_crosstabs
# demographic_group + "_" + group_value) to the corresponding column in
# type_profiles.  None means the column is derived from another entry.
#
# Special handling for direction-inverted dimensions:
#   "education_noncollege" → same feature as "education_college" but negated
#   "urbanicity_rural"     → same feature as "urbanicity_urban" but negated
# These are resolved in build_affinity_index rather than here so the map
# stays a plain dict (no extra metadata).
CROSSTAB_DIMENSION_MAP: dict[str, str | None] = {
    "education_college":    "pct_bachelors_plus",
    "education_noncollege": None,           # derived: −(college affinity)
    "race_white":           "pct_white_nh",
    "race_black":           "pct_black",
    "race_hispanic":        "pct_hispanic",
    "race_asian":           "pct_asian",
    "urbanicity_urban":     "log_pop_density",
    "urbanicity_rural":     None,           # derived: −(urban affinity)
    "age_senior":           "median_age",
    "religion_evangelical": "evangelical_share",
}

# Which dimensions are inversions of a "parent" dimension.
# Maps inverted_key → parent_key.  We negate the parent's affinity.
_INVERTED_DIMENSIONS: dict[str, str] = {
    "education_noncollege": "education_college",
    "urbanicity_rural":     "urbanicity_urban",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_affinity_index(
    type_profiles: pd.DataFrame,
    county_demographics: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Compute per-type demographic affinity relative to national population mean.

    For each dimension in CROSSTAB_DIMENSION_MAP the affinity of type j is:
        affinity[j] = type_profiles[feature][j] - national_mean(feature)

    where the national mean is population-weighted over all counties in
    ``county_demographics``.

    "education_noncollege" and "urbanicity_rural" are inversions of their
    parent dimensions (negated affinity), so that a higher raw value (more
    college / denser) corresponds to *lower* affinity for those keys.

    Parameters
    ----------
    type_profiles:
        DataFrame with one row per type (J rows).  Must contain all feature
        columns referenced in CROSSTAB_DIMENSION_MAP.
    county_demographics:
        County-level DataFrame.  Must contain ``pop_total`` (population weight)
        and all feature columns in CROSSTAB_DIMENSION_MAP.

    Returns
    -------
    dict mapping dimension_key → np.ndarray of shape (J,)
    """
    if "pop_total" not in county_demographics.columns:
        raise ValueError("county_demographics must contain a 'pop_total' column")

    pop_weights = county_demographics["pop_total"].to_numpy(dtype=float)
    total_pop = pop_weights.sum()
    if total_pop <= 0:
        raise ValueError("county_demographics has zero total population")

    # Pre-compute national population-weighted mean for each real (non-None) feature.
    # This avoids recomputing for inverted dimensions that share the same feature.
    unique_features: set[str] = {
        col for col in CROSSTAB_DIMENSION_MAP.values() if col is not None
    }
    national_means: dict[str, float] = {}
    for feature in unique_features:
        if feature not in county_demographics.columns:
            raise KeyError(
                f"Feature '{feature}' from CROSSTAB_DIMENSION_MAP is missing "
                f"from county_demographics columns: {list(county_demographics.columns)}"
            )
        vals = county_demographics[feature].to_numpy(dtype=float)
        national_means[feature] = float(np.average(vals, weights=pop_weights))

    # Build affinity for direct (non-inverted) dimensions first.
    affinity_index: dict[str, np.ndarray] = {}
    for dim_key, feature in CROSSTAB_DIMENSION_MAP.items():
        if dim_key in _INVERTED_DIMENSIONS:
            continue  # defer; handled below
        if feature is None:
            # Safeguard: should only happen for inverted dims handled above
            continue
        if feature not in type_profiles.columns:
            raise KeyError(
                f"Feature '{feature}' not found in type_profiles columns. "
                f"Available: {list(type_profiles.columns)}"
            )
        type_vals = type_profiles[feature].to_numpy(dtype=float)
        affinity_index[dim_key] = type_vals - national_means[feature]

    # Build inverted dimensions by negating their parent's affinity.
    for inv_key, parent_key in _INVERTED_DIMENSIONS.items():
        if parent_key not in affinity_index:
            raise RuntimeError(
                f"Parent dimension '{parent_key}' for inverted key '{inv_key}' "
                f"was not computed — check CROSSTAB_DIMENSION_MAP ordering."
            )
        # Negation: a type that is high-college has NEGATIVE noncollege affinity
        affinity_index[inv_key] = -affinity_index[parent_key]

    return affinity_index


def compute_state_baseline_w(
    type_scores: np.ndarray,
    county_populations: np.ndarray,
    state_mask: np.ndarray,
) -> np.ndarray:
    """Population-weighted mean type scores for counties in a single state.

    This is the baseline W to use for any poll in the state when no crosstab
    data is available, or as the starting point for crosstab adjustment.

    Parameters
    ----------
    type_scores:
        Array of shape (N_counties, J) — soft type membership scores.  Rows
        need not sum to 1 (raw KMeans inverse-distance scores are fine).
    county_populations:
        Array of shape (N_counties,) — population of each county.
    state_mask:
        Boolean array of shape (N_counties,) — True for counties in the target
        state.

    Returns
    -------
    np.ndarray of shape (J,), normalised to sum to 1.
    """
    state_pops = county_populations[state_mask]
    state_scores = type_scores[state_mask]  # (n_state_counties, J)

    total_pop = state_pops.sum()
    if total_pop <= 0:
        raise ValueError("State has zero total population in county_populations")

    # Population-weighted mean across the state's counties.
    # Equivalent to (pops @ scores) / total_pop — matrix multiply is faster for
    # large J.
    w = state_pops @ state_scores / total_pop  # (J,)

    w_sum = w.sum()
    if w_sum <= 0:
        raise ValueError("State baseline W sums to zero — check type_scores for the state")
    return w / w_sum


def construct_w_row(
    poll_crosstabs: list[dict[str, Any]],
    state_baseline_w: np.ndarray,
    affinity_index: dict[str, np.ndarray],
    state_demographic_means: dict[str, float],
    adjustment_strength: float = ADJUSTMENT_STRENGTH_DEFAULT,
) -> np.ndarray:
    """Construct a poll-specific W vector using crosstab demographic composition.

    Algorithm:
      For each usable crosstab row:
        1.  Build dimension_key = f"{demographic_group}_{group_value}".
        2.  Look up the affinity vector for that dimension (shape J).
        3.  Normalise affinity by its max absolute value → unit-scale adjustment.
        4.  delta = pct_of_sample - state_demographic_means[dimension_key]
        5.  Accumulate adjustment += delta * normalised_affinity.
      W_adjusted = W_base + α * adjustment
      Clip to ≥ 0, renormalise to sum to 1.

    If no usable crosstabs are found (missing keys, null pct_of_sample), returns
    state_baseline_w unchanged.

    Parameters
    ----------
    poll_crosstabs:
        List of dicts, each with keys:
          ``demographic_group`` (str): e.g. "education"
          ``group_value``       (str): e.g. "college"
          ``pct_of_sample``     (float | None): fraction of poll sample in group
    state_baseline_w:
        Baseline W vector for the state, shape (J,), sums to 1.
    affinity_index:
        Output of build_affinity_index — maps dimension_key → (J,) array.
    state_demographic_means:
        Maps dimension_key → float (state population-weighted demographic mean
        for that dimension).  Missing keys cause that crosstab row to be skipped.
    adjustment_strength:
        α in [0, 1].  Dials between full baseline (0) and full crosstab
        adjustment (1).  Default is ADJUSTMENT_STRENGTH_DEFAULT = 0.3.

    Returns
    -------
    np.ndarray of shape (J,), sums to 1.0, all entries ≥ 0.
    """
    J = state_baseline_w.shape[0]
    cumulative_adjustment = np.zeros(J, dtype=float)
    n_usable = 0

    for row in poll_crosstabs:
        pct = row.get("pct_of_sample")
        if pct is None:
            continue

        dim_key = f"{row.get('demographic_group', '')}_{row.get('group_value', '')}"
        if dim_key not in affinity_index:
            log.debug("construct_w_row: dimension %r not in affinity_index, skipping", dim_key)
            continue
        if dim_key not in state_demographic_means:
            log.debug(
                "construct_w_row: dimension %r not in state_demographic_means, skipping",
                dim_key,
            )
            continue

        affinity = affinity_index[dim_key]
        max_abs = np.abs(affinity).max()
        if max_abs < _AFFINITY_NORM_MIN:
            # All types identical on this feature — no signal to use.
            log.debug(
                "construct_w_row: dimension %r has near-zero affinity range, skipping",
                dim_key,
            )
            continue

        # Normalise affinity to unit scale so all dimensions contribute equally
        # regardless of their raw demographic range (e.g. log density vs fraction).
        normalised_affinity = affinity / max_abs

        delta = float(pct) - state_demographic_means[dim_key]
        cumulative_adjustment += delta * normalised_affinity
        n_usable += 1

    if n_usable == 0:
        # No crosstab signal — fall back to state baseline unchanged.
        return state_baseline_w.copy()

    w_adjusted = state_baseline_w + adjustment_strength * cumulative_adjustment

    # Clip to simplex: all weights must be non-negative.
    # This can push mass away from types that were heavily down-weighted,
    # but that is intentional — the crosstab data says the poll under-represents
    # those types.
    np.clip(w_adjusted, 0.0, None, out=w_adjusted)

    w_sum = w_adjusted.sum()
    if w_sum <= 0:
        # Extreme delta caused all weights to clip to zero — extremely unlikely
        # but handled gracefully by falling back to baseline.
        log.warning(
            "construct_w_row: adjustment zeroed out all weights; falling back to baseline"
        )
        return state_baseline_w.copy()

    return w_adjusted / w_sum
