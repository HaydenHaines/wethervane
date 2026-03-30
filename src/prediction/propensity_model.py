"""Voter propensity model: estimates how likely each type is to be represented in LV/RV screens.

This is a stub implementation. Task 1 (config + propensity model) will replace the body of
compute_propensity_scores with a fitted model. The interface contract is frozen:

  load_config() -> dict
  compute_propensity_scores(type_profiles: pd.DataFrame, config: dict) -> np.ndarray

propensity scores are in [0, 1], normalized so their mean equals 1.0. A score of 0.5
means this type is half as likely to appear in a likely-voter screen as the average type.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "poll_enrichment.yaml"

# Default config used when no config file exists (stub / pre-Task-1 state).
_DEFAULT_CONFIG: dict = {
    "lv_downweight_factor": 0.5,
    "rv_downweight_factor": 0.8,
    "propensity_features": [
        "pct_owner_occupied",
        "pct_bachelors_plus",
        "median_age",
    ],
    "propensity_feature_weights": {
        "pct_owner_occupied": 1.0,
        "pct_bachelors_plus": 1.0,
        "median_age": 0.5,
    },
    "w_vector_dimensions": {
        "core": [],
        "full": [],
    },
    "method_reach_profiles": {},
}


def load_config() -> dict:
    """Load poll enrichment config from disk, falling back to defaults."""
    if _CONFIG_PATH.exists():
        with _CONFIG_PATH.open() as fh:
            return yaml.safe_load(fh) or _DEFAULT_CONFIG
    return _DEFAULT_CONFIG


def compute_propensity_scores(
    type_profiles: pd.DataFrame,
    config: dict,
) -> np.ndarray:
    """Estimate relative voter propensity for each type.

    Returns an array of shape (J,) with values in [0,1]. Normalized so the
    population-weighted mean equals 1.0 (i.e., relative, not absolute propensity).
    Types with higher homeownership, education, and age are more likely to appear
    in LV screens; types with lower values on these dimensions are downweighted.

    This stub uses a simple weighted-feature composite. Task 1 replaces this with
    a fitted model, but the interface is stable.
    """
    features = config.get("propensity_features", _DEFAULT_CONFIG["propensity_features"])
    weights = config.get(
        "propensity_feature_weights",
        _DEFAULT_CONFIG["propensity_feature_weights"],
    )

    J = len(type_profiles)
    scores = np.zeros(J)
    total_weight = 0.0

    for feat in features:
        if feat not in type_profiles.columns:
            continue
        w = weights.get(feat, 1.0)
        col = type_profiles[feat].values.astype(float)
        col_min, col_max = col.min(), col.max()
        col_range = col_max - col_min
        if col_range > 0:
            # Normalize to [0, 1] within this feature
            scores += w * (col - col_min) / col_range
            total_weight += w

    if total_weight > 0:
        scores /= total_weight
    else:
        # No usable features: uniform propensity
        return np.ones(J)

    # Shift to [0, 1] range with mean=1.0 relative scaling
    # scores is already in [0,1]; rescale so mean = 1.0 for use as a multiplicative factor
    mean_score = scores.mean()
    if mean_score > 0:
        scores = scores / mean_score
    else:
        scores = np.ones(J)

    return scores
