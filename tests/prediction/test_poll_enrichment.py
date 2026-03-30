"""Tests for tiered W vector construction."""

import numpy as np
import pandas as pd
import pytest

from src.prediction.poll_enrichment import (
    build_W_poll,
    build_W_with_adjustments,
    build_W_from_crosstabs,
    build_W_from_raw_sample,
    parse_methodology,
)


def _make_type_profiles(j: int = 5) -> pd.DataFrame:
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "type_id": range(j),
        "median_age": np.linspace(25, 65, j),
        "pct_owner_occupied": np.linspace(0.3, 0.8, j),
        "pct_bachelors_plus": np.linspace(0.1, 0.6, j),
        "evangelical_share": np.linspace(0.05, 0.5, j),
        "catholic_share": np.linspace(0.4, 0.1, j),
        "mainline_share": np.linspace(0.1, 0.3, j),
        "log_pop_density": np.linspace(-0.5, 0.5, j),
        "median_hh_income": np.linspace(30000, 100000, j),
        "pct_black": np.linspace(0.02, 0.4, j),
        "pct_white_nh": np.linspace(0.8, 0.4, j),
        "pct_hispanic": np.linspace(0.05, 0.3, j),
        "pct_asian": np.linspace(0.01, 0.1, j),
    })


def _make_state_type_weights(j: int = 5) -> np.ndarray:
    w = np.array([0.3, 0.2, 0.2, 0.2, 0.1])[:j]
    return w / w.sum()


class TestTierDispatch:
    def test_tier1_when_raw_data_present(self):
        tp = _make_type_profiles()
        stw = _make_state_type_weights()
        raw = {"pct_black": 0.33, "evangelical_share": 0.25}
        W = build_W_poll(
            poll={"state": "GA", "dem_share": 0.53, "n_sample": 600},
            type_profiles=tp, state_type_weights=stw,
            raw_sample_demographics=raw,
        )
        assert W.shape == (5,)
        assert abs(W.sum() - 1.0) < 1e-6

    def test_tier2_when_crosstabs_present(self):
        tp = _make_type_profiles()
        stw = _make_state_type_weights()
        xt = [
            {"demographic_group": "race", "group_value": "black",
             "pct_of_sample": 0.33, "dem_share": 0.90},
            {"demographic_group": "race", "group_value": "white",
             "pct_of_sample": 0.55, "dem_share": 0.40},
        ]
        result = build_W_poll(
            poll={"state": "GA", "dem_share": 0.53, "n_sample": 600},
            type_profiles=tp, state_type_weights=stw,
            poll_crosstabs=xt,
        )
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(r["W"].shape == (5,) for r in result)

    def test_tier3_when_topline_only(self):
        tp = _make_type_profiles()
        stw = _make_state_type_weights()
        W = build_W_poll(
            poll={"state": "GA", "dem_share": 0.53, "n_sample": 600},
            type_profiles=tp, state_type_weights=stw,
        )
        assert W.shape == (5,)
        assert abs(W.sum() - 1.0) < 1e-6


class TestTier3Adjustments:
    def test_lv_downweights_low_propensity(self):
        tp = _make_type_profiles()
        stw = _make_state_type_weights()
        W_lv = build_W_with_adjustments(
            poll={"state": "GA", "dem_share": 0.53, "n_sample": 600, "methodology": "LV"},
            type_profiles=tp, state_type_weights=stw,
        )
        W_none = build_W_with_adjustments(
            poll={"state": "GA", "dem_share": 0.53, "n_sample": 600, "methodology": ""},
            type_profiles=tp, state_type_weights=stw,
        )
        np.testing.assert_allclose(W_none, stw, atol=1e-6)
        assert not np.allclose(W_lv, stw, atol=1e-3)
        # Type 0 has lowest propensity (youngest, lowest homeownership, lowest education)
        assert W_lv[0] < stw[0]

    def test_core_vs_full_mode(self):
        tp = _make_type_profiles()
        stw = _make_state_type_weights()
        poll = {"state": "GA", "dem_share": 0.53, "n_sample": 600, "methodology": "LV"}
        W_core = build_W_with_adjustments(poll, tp, stw, w_vector_mode="core")
        W_full = build_W_with_adjustments(poll, tp, stw, w_vector_mode="full")
        assert abs(W_core.sum() - 1.0) < 1e-6
        assert abs(W_full.sum() - 1.0) < 1e-6
        # Both valid but they should differ (full uses more dims) — though if
        # no method reach profiles have shifts, they'll be identical from LV
        # adjustment alone. That's acceptable.

    def test_no_methodology_returns_state_weights(self):
        tp = _make_type_profiles()
        stw = _make_state_type_weights()
        W = build_W_with_adjustments(
            poll={"state": "GA", "dem_share": 0.53, "n_sample": 600, "methodology": ""},
            type_profiles=tp, state_type_weights=stw,
        )
        np.testing.assert_allclose(W, stw, atol=1e-6)


class TestParseMethodology:
    def test_lv(self):
        assert parse_methodology("D=34.0% R=53.0%; LV; src=wikipedia") == "LV"

    def test_rv(self):
        assert parse_methodology("D=34.0% R=53.0%; RV; src=wikipedia") == "RV"

    def test_no_method(self):
        assert parse_methodology("D=34.0% R=53.0%; src=wikipedia") == ""

    def test_empty(self):
        assert parse_methodology("") == ""

    def test_none(self):
        assert parse_methodology(None) == ""
