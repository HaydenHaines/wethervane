"""Tests for LV/RV propensity scoring."""

import numpy as np
import pandas as pd
import pytest

from src.prediction.propensity_model import compute_propensity_scores, load_config


def _make_type_profiles(n_types: int = 5) -> pd.DataFrame:
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "type_id": range(n_types),
        "median_age": rng.uniform(25, 65, n_types),
        "pct_owner_occupied": rng.uniform(0.3, 0.8, n_types),
        "pct_bachelors_plus": rng.uniform(0.1, 0.6, n_types),
        "evangelical_share": rng.uniform(0.05, 0.5, n_types),
    })


class TestLoadConfig:
    def test_loads_from_default_path(self):
        cfg = load_config()
        assert "lv_propensity_coefficients" in cfg
        assert "lv_downweight_factor" in cfg
        assert "w_vector_dimensions" in cfg

    def test_coefficients_sum_to_one(self):
        cfg = load_config()
        coeffs = cfg["lv_propensity_coefficients"]
        total = sum(v for k, v in coeffs.items() if not k.startswith("_"))
        assert abs(total - 1.0) < 1e-6


class TestComputePropensity:
    def test_output_shape(self):
        tp = _make_type_profiles(10)
        scores = compute_propensity_scores(tp)
        assert scores.shape == (10,)

    def test_scores_in_unit_interval(self):
        tp = _make_type_profiles(20)
        scores = compute_propensity_scores(tp)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_older_homeowner_educated_scores_higher(self):
        tp = pd.DataFrame({
            "type_id": [0, 1],
            "median_age": [60.0, 25.0],
            "pct_owner_occupied": [0.8, 0.3],
            "pct_bachelors_plus": [0.5, 0.1],
        })
        scores = compute_propensity_scores(tp)
        assert scores[0] > scores[1]
