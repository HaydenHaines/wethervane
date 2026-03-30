"""LV/RV type-screen propensity scoring.

Computes a propensity score per electoral type based on demographic proxies
for voter turnout. Used by poll_enrichment.py to model which types an LV
screen systematically includes/excludes.

The model is a config-driven linear combination — not a trained model.
Coefficients are from political science literature on voter turnout correlates.
All tunable parameters live in data/config/poll_method_adjustments.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "data" / "config" / "poll_method_adjustments.json"


def load_config(path: Path | str | None = None) -> dict:
    """Load poll method adjustment configuration."""
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    with p.open() as f:
        return json.load(f)


def compute_propensity_scores(
    type_profiles: pd.DataFrame,
    config: dict | None = None,
) -> np.ndarray:
    """Compute turnout propensity score per type.

    Returns array of shape (n_types,) with values in [0, 1].
    Higher = more likely to pass an LV screen.
    """
    if config is None:
        config = load_config()

    coefficients = config["lv_propensity_coefficients"]
    fields = [k for k in coefficients if not k.startswith("_")]
    weights = np.array([coefficients[k] for k in fields])

    values = np.zeros((len(type_profiles), len(fields)))
    for i, field in enumerate(fields):
        col = type_profiles[field].values.astype(float)
        col_min, col_max = col.min(), col.max()
        if col_max > col_min:
            values[:, i] = (col - col_min) / (col_max - col_min)
        else:
            values[:, i] = 0.5

    raw_scores = values @ weights
    s_min, s_max = raw_scores.min(), raw_scores.max()
    if s_max > s_min:
        return (raw_scores - s_min) / (s_max - s_min)
    return np.full(len(type_profiles), 0.5)
