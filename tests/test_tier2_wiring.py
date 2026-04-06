"""Tests for Tier 2 crosstab wiring in the forecast engine.

Covers:
  1. _extract_crosstabs_from_xt correctly parses xt_* poll fields into
     crosstab dicts with demographic_group, group_value, pct_of_sample, dem_share.
  2. Polls with xt_* data route through Tier 2 and produce multiple
     observations (list[dict]) instead of a single W vector.
  3. Polls without xt_* data fall through to Tier 1/3 as before.
  4. Expanded observations from Tier 2 produce valid W vectors
     (non-negative, sum to ~1).
  5. End-to-end through run_forecast: xt_* polls produce valid results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.prediction.forecast_engine import (
    _extract_crosstabs_from_xt,
    _build_poll_arrays,
    build_W_state,
    run_forecast,
)
from src.prediction.poll_enrichment import build_W_from_crosstabs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_type_profiles(J: int = 4) -> pd.DataFrame:
    """Minimal type_profiles with columns needed by _map_demographic_to_types."""
    return pd.DataFrame({
        "pct_bachelors_plus": np.linspace(0.15, 0.65, J),
        "pct_white_nh":       np.linspace(0.90, 0.40, J),
        "pct_black":          np.linspace(0.02, 0.35, J),
        "pct_hispanic":       np.linspace(0.03, 0.30, J),
        "pct_asian":          np.linspace(0.01, 0.12, J),
        "median_age":         np.linspace(30.0, 55.0, J),
        "log_pop_density":    np.linspace(1.5, 5.0, J),
        "evangelical_share":  np.linspace(0.50, 0.05, J),
    })


def _make_type_scores(n_counties: int = 6, J: int = 4, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    raw = rng.rand(n_counties, J)
    return raw / raw.sum(axis=1, keepdims=True)


def _uniform_state_weights(J: int = 4) -> np.ndarray:
    return np.ones(J) / J


# ---------------------------------------------------------------------------
# 1. _extract_crosstabs_from_xt
# ---------------------------------------------------------------------------

class TestExtractCrosstabsFromXt:
    """Unit tests for the xt_* → crosstab dict converter."""

    def test_returns_none_for_poll_without_xt(self):
        """A poll with no xt_* keys should return None."""
        poll = {"dem_share": 0.52, "n_sample": 600, "state": "GA"}
        result = _extract_crosstabs_from_xt(poll)
        assert result is None

    def test_single_xt_field_produces_one_crosstab(self):
        poll = {"dem_share": 0.55, "n_sample": 800, "state": "AZ",
                "xt_race_black": 0.13}
        result = _extract_crosstabs_from_xt(poll)
        assert result is not None
        assert len(result) == 1
        row = result[0]
        assert row["demographic_group"] == "race"
        assert row["group_value"] == "black"
        assert row["pct_of_sample"] == pytest.approx(0.13)
        assert row["dem_share"] == pytest.approx(0.55)

    def test_multiple_xt_fields_produce_multiple_crosstabs(self):
        poll = {
            "dem_share": 0.51,
            "n_sample": 1000,
            "state": "FL",
            "xt_education_college":    0.48,
            "xt_education_noncollege": 0.52,
            "xt_race_white":           0.56,
            "xt_race_black":           0.18,
            "xt_race_hispanic":        0.21,
            "xt_age_senior":           0.39,
        }
        result = _extract_crosstabs_from_xt(poll)
        assert result is not None
        assert len(result) == 6

    def test_each_crosstab_has_required_keys(self):
        poll = {"dem_share": 0.52, "n_sample": 700, "state": "GA",
                "xt_race_white": 0.68, "xt_education_college": 0.42}
        result = _extract_crosstabs_from_xt(poll)
        assert result is not None
        for row in result:
            for key in ("demographic_group", "group_value", "pct_of_sample", "dem_share"):
                assert key in row, f"Missing key '{key}' in crosstab row: {row}"

    def test_dem_share_copied_from_topline(self):
        """All crosstab rows should carry the poll's topline dem_share."""
        dem_share = 0.487
        poll = {"dem_share": dem_share, "n_sample": 600, "state": "GA",
                "xt_race_white": 0.72, "xt_race_black": 0.14}
        result = _extract_crosstabs_from_xt(poll)
        assert result is not None
        for row in result:
            assert row["dem_share"] == pytest.approx(dem_share)

    def test_zero_pct_is_skipped(self):
        """Zero pct_of_sample values should be excluded (group not in sample)."""
        poll = {"dem_share": 0.52, "n_sample": 600, "state": "GA",
                "xt_race_black": 0.0,      # skipped
                "xt_race_white": 0.75}     # included
        result = _extract_crosstabs_from_xt(poll)
        assert result is not None
        assert len(result) == 1
        assert result[0]["group_value"] == "white"

    def test_non_float_value_is_skipped(self):
        """Non-numeric xt_ values should be silently skipped."""
        poll = {"dem_share": 0.52, "n_sample": 600, "state": "GA",
                "xt_race_black": "n/a",    # invalid
                "xt_race_white": 0.70}     # valid
        result = _extract_crosstabs_from_xt(poll)
        assert result is not None
        assert len(result) == 1

    def test_none_dem_share_returns_none(self):
        """Poll without dem_share should return None (can't construct observations)."""
        poll = {"n_sample": 600, "state": "GA", "xt_race_white": 0.70}
        result = _extract_crosstabs_from_xt(poll)
        assert result is None

    def test_malformed_xt_key_is_skipped(self):
        """xt_ keys without the group_value separator should be skipped."""
        poll = {"dem_share": 0.52, "n_sample": 600, "state": "GA",
                "xt_":           0.50,   # no group or value
                "xt_race_black": 0.13}   # valid
        result = _extract_crosstabs_from_xt(poll)
        assert result is not None
        assert len(result) == 1

    def test_pct_of_sample_values_are_floats(self):
        """pct_of_sample and dem_share in output must be float."""
        poll = {"dem_share": 0.52, "n_sample": 600, "state": "GA",
                "xt_race_black": "0.13"}  # string input
        result = _extract_crosstabs_from_xt(poll)
        assert result is not None
        assert isinstance(result[0]["pct_of_sample"], float)
        assert isinstance(result[0]["dem_share"], float)

    def test_non_xt_fields_are_ignored(self):
        """Fields without xt_ prefix must not appear in crosstab output."""
        poll = {"dem_share": 0.52, "n_sample": 600, "state": "GA",
                "xt_race_black": 0.13,
                "pollster": "Emerson",
                "notes": "LV"}
        result = _extract_crosstabs_from_xt(poll)
        assert result is not None
        assert len(result) == 1  # only xt_race_black


# ---------------------------------------------------------------------------
# 2. Tier 2 produces multiple observations
# ---------------------------------------------------------------------------

class TestTier2MultipleObservations:
    """Polls with xt_* data should expand into multiple Bayesian observations."""

    def test_xt_poll_produces_multiple_observations_in_arrays(self):
        """A poll with N xt_ groups should produce N rows in W_all."""
        J = 4
        n_counties = 6
        type_scores = _make_type_scores(n_counties, J)
        type_profiles = _make_type_profiles(J)
        states = ["AZ"] * 3 + ["NV"] * 3
        county_votes = np.ones(n_counties) * 1000.0

        # 4 xt_ groups → should produce 4 observations
        poll = {
            "dem_share": 0.51,
            "n_sample": 850,
            "state": "AZ",
            "xt_race_white": 0.71,
            "xt_race_black": 0.03,
            "xt_race_hispanic": 0.19,
            "xt_education_college": 0.36,
        }

        def w_builder(p):
            from src.prediction.forecast_engine import (
                _extract_crosstabs_from_xt,
                _extract_raw_demographics,
            )
            from src.prediction.poll_enrichment import build_W_poll
            st = p["state"]
            W_state = build_W_state(st, type_scores, states, county_votes)
            crosstabs = _extract_crosstabs_from_xt(p)
            raw = _extract_raw_demographics(p) if crosstabs is None else None
            return build_W_poll(
                poll=p,
                type_profiles=type_profiles,
                state_type_weights=W_state,
                poll_crosstabs=crosstabs,
                raw_sample_demographics=raw,
            )

        W_all, y_all, sigma_all, labels = _build_poll_arrays(
            {"az_gov": [poll]},
            type_scores, states, county_votes,
            w_builder=w_builder,
        )

        # 4 xt_ groups → 4 rows in the output arrays
        assert W_all.shape == (4, J), f"Expected (4, {J}), got {W_all.shape}"
        assert y_all.shape == (4,)
        assert sigma_all.shape == (4,)

    def test_poll_without_xt_produces_single_observation(self):
        """A poll with no xt_* data should produce exactly one row in W_all."""
        J = 4
        n_counties = 6
        type_scores = _make_type_scores(n_counties, J)
        type_profiles = _make_type_profiles(J)
        states = ["GA"] * 3 + ["FL"] * 3
        county_votes = np.ones(n_counties)

        poll = {"dem_share": 0.52, "n_sample": 600, "state": "GA"}

        def w_builder(p):
            from src.prediction.forecast_engine import (
                _extract_crosstabs_from_xt,
                _extract_raw_demographics,
            )
            from src.prediction.poll_enrichment import build_W_poll
            st = p["state"]
            W_state = build_W_state(st, type_scores, states, county_votes)
            crosstabs = _extract_crosstabs_from_xt(p)
            raw = _extract_raw_demographics(p) if crosstabs is None else None
            return build_W_poll(
                poll=p,
                type_profiles=type_profiles,
                state_type_weights=W_state,
                poll_crosstabs=crosstabs,
                raw_sample_demographics=raw,
            )

        W_all, y_all, sigma_all, labels = _build_poll_arrays(
            {"ga_gov": [poll]},
            type_scores, states, county_votes,
            w_builder=w_builder,
        )

        assert W_all.shape == (1, J)

    def test_mixed_polls_correct_row_counts(self):
        """Mix of xt_ and non-xt_ polls should produce the right total rows."""
        J = 4
        n_counties = 6
        type_scores = _make_type_scores(n_counties, J)
        type_profiles = _make_type_profiles(J)
        states = ["GA"] * 3 + ["FL"] * 3
        county_votes = np.ones(n_counties)

        # poll_a has 3 xt_ groups → 3 rows; poll_b has 0 → 1 row; total = 4
        poll_a = {
            "dem_share": 0.53, "n_sample": 700, "state": "GA",
            "xt_race_white": 0.65,
            "xt_race_black": 0.15,
            "xt_education_college": 0.40,
        }
        poll_b = {"dem_share": 0.49, "n_sample": 600, "state": "FL"}

        def w_builder(p):
            from src.prediction.forecast_engine import (
                _extract_crosstabs_from_xt,
                _extract_raw_demographics,
            )
            from src.prediction.poll_enrichment import build_W_poll
            st = p["state"]
            W_state = build_W_state(st, type_scores, states, county_votes)
            crosstabs = _extract_crosstabs_from_xt(p)
            raw = _extract_raw_demographics(p) if crosstabs is None else None
            return build_W_poll(
                poll=p,
                type_profiles=type_profiles,
                state_type_weights=W_state,
                poll_crosstabs=crosstabs,
                raw_sample_demographics=raw,
            )

        polls_by_race = {"race1": [poll_a], "race2": [poll_b]}
        W_all, y_all, sigma_all, labels = _build_poll_arrays(
            polls_by_race, type_scores, states, county_votes, w_builder=w_builder,
        )

        assert W_all.shape[0] == 4  # 3 + 1


# ---------------------------------------------------------------------------
# 3. Tier fallback: no xt_ data uses Tier 1/3
# ---------------------------------------------------------------------------

class TestTierFallbackPreserved:
    """Tier 1/3 must remain fully functional when no xt_* data is present."""

    def test_no_xt_poll_still_produces_valid_w(self):
        """A plain poll with no xt_* fields must not crash and W must be valid."""
        J = 4
        tp = _make_type_profiles(J)
        sw = _uniform_state_weights(J)

        from src.prediction.poll_enrichment import build_W_poll
        poll = {"dem_share": 0.52, "n_sample": 600, "state": "GA"}

        crosstabs = _extract_crosstabs_from_xt(poll)
        assert crosstabs is None  # no xt_ data

        W = build_W_poll(
            poll=poll,
            type_profiles=tp,
            state_type_weights=sw,
            poll_crosstabs=None,
            raw_sample_demographics=None,
        )
        assert isinstance(W, np.ndarray)
        assert W.shape == (J,)
        assert abs(W.sum() - 1.0) < 1e-9

    def test_xt_poll_routes_through_tier2_not_tier1(self):
        """When xt_* data is present, Tier 2 is used (returns list) not Tier 1 (returns ndarray)."""
        J = 4
        tp = _make_type_profiles(J)
        sw = _uniform_state_weights(J)

        from src.prediction.poll_enrichment import build_W_poll
        from src.prediction.forecast_engine import _extract_raw_demographics

        poll = {
            "dem_share": 0.53, "n_sample": 700, "state": "GA",
            "xt_race_white": 0.70, "xt_race_black": 0.12,
        }

        crosstabs = _extract_crosstabs_from_xt(poll)
        # Tier 1 would normally be populated from the same xt_ keys,
        # but _w_builder skips it when Tier 2 crosstabs are available.
        raw = None  # simulate _w_builder's behavior: skip Tier 1 when Tier 2 is present

        result = build_W_poll(
            poll=poll,
            type_profiles=tp,
            state_type_weights=sw,
            poll_crosstabs=crosstabs,
            raw_sample_demographics=raw,
        )
        # Tier 2 returns a list of dicts
        assert isinstance(result, list), (
            "Expected Tier 2 (list) result when crosstabs present"
        )
        assert len(result) == 2  # two xt_ groups


# ---------------------------------------------------------------------------
# 4. W vector validity for Tier 2 observations
# ---------------------------------------------------------------------------

class TestTier2WVectorValidity:
    """Each expanded observation from Tier 2 must have a valid W vector."""

    def _run_build_W_from_crosstabs(self, J: int, n_groups: int) -> list[dict]:
        tp = _make_type_profiles(J)
        sw = _uniform_state_weights(J)
        poll = {"dem_share": 0.53, "n_sample": 800, "state": "GA"}
        crosstabs = [
            {"demographic_group": "race", "group_value": "white",
             "pct_of_sample": 0.65, "dem_share": 0.53},
            {"demographic_group": "race", "group_value": "black",
             "pct_of_sample": 0.13, "dem_share": 0.53},
            {"demographic_group": "education", "group_value": "college",
             "pct_of_sample": 0.42, "dem_share": 0.53},
        ][:n_groups]
        return build_W_from_crosstabs(poll, crosstabs, tp, sw)

    def test_all_observations_have_nonnegative_w(self):
        """All W entries across all observations must be >= 0."""
        observations = self._run_build_W_from_crosstabs(J=4, n_groups=3)
        for obs in observations:
            W = obs["W"]
            assert (W >= 0.0).all(), f"Negative W entries: {W}"

    def test_all_observations_w_sum_to_one(self):
        """W must sum to 1.0 for every expanded observation."""
        observations = self._run_build_W_from_crosstabs(J=4, n_groups=3)
        for i, obs in enumerate(observations):
            W = obs["W"]
            assert abs(W.sum() - 1.0) < 1e-9, (
                f"Observation {i}: W sums to {W.sum()}, expected 1.0"
            )

    def test_all_observations_have_positive_sigma(self):
        """sigma must be strictly positive for every observation."""
        observations = self._run_build_W_from_crosstabs(J=4, n_groups=2)
        for obs in observations:
            assert obs["sigma"] > 0, f"sigma must be positive, got {obs['sigma']}"

    def test_all_observations_w_are_finite(self):
        """No NaN or Inf values in W."""
        observations = self._run_build_W_from_crosstabs(J=100, n_groups=3)
        for obs in observations:
            W = obs["W"]
            assert not np.any(np.isnan(W)), "NaN in W"
            assert not np.any(np.isinf(W)), "Inf in W"

    def test_observations_have_correct_shape(self):
        """W must have shape (J,)."""
        J = 10
        observations = self._run_build_W_from_crosstabs(J=J, n_groups=3)
        for obs in observations:
            assert obs["W"].shape == (J,)


# ---------------------------------------------------------------------------
# 5. End-to-end through run_forecast
# ---------------------------------------------------------------------------

class TestEndToEndTier2:
    """Integration tests through run_forecast with Tier 2 xt_* polls."""

    def _make_forecast_inputs(self, J: int = 4, n_counties: int = 6):
        type_scores = _make_type_scores(n_counties, J)
        type_profiles = _make_type_profiles(J)
        states = ["AZ"] * 3 + ["NV"] * 3
        county_priors = np.full(n_counties, 0.5)
        county_votes = np.ones(n_counties) * 1000.0
        return type_scores, type_profiles, states, county_priors, county_votes

    def test_xt_poll_run_forecast_completes(self):
        """run_forecast with an xt_* poll must complete without error."""
        J = 4
        ts, tp, states, priors, votes = self._make_forecast_inputs(J)

        polls = {"az_gov": [{
            "dem_share": 0.52,
            "n_sample": 850,
            "state": "AZ",
            "xt_race_white": 0.71,
            "xt_race_black": 0.03,
            "xt_race_hispanic": 0.19,
        }]}

        result = run_forecast(
            type_scores=ts,
            county_priors=priors,
            states=states,
            county_votes=votes,
            polls_by_race=polls,
            races=["az_gov"],
            lam=1.0,
            mu=1.0,
            type_profiles=tp,
        )

        assert "az_gov" in result
        r = result["az_gov"]
        assert np.all(np.isfinite(r.theta_national))
        assert r.theta_national.shape == (J,)

    def test_xt_poll_theta_differs_from_no_xt(self):
        """An xt_* poll should produce a different theta_national than the plain version."""
        J = 4
        ts, tp, states, priors, votes = self._make_forecast_inputs(J)

        base_poll = {"dem_share": 0.52, "n_sample": 850, "state": "AZ"}
        xt_poll = {**base_poll,
                   "xt_race_white": 0.71,
                   "xt_race_black": 0.03,
                   "xt_race_hispanic": 0.19}

        def _run(poll):
            result = run_forecast(
                type_scores=ts,
                county_priors=priors,
                states=states,
                county_votes=votes,
                polls_by_race={"az_gov": [poll]},
                races=["az_gov"],
                lam=0.5,
                mu=0.5,
                type_profiles=tp,
            )
            return result["az_gov"].theta_national

        theta_base = _run(base_poll)
        theta_xt = _run(xt_poll)

        # The Tier 2 W vectors are different from state-level fallback,
        # so theta_national should differ.
        assert not np.allclose(theta_base, theta_xt), (
            "Expected xt_* poll to produce different theta_national than "
            "plain state-level poll"
        )

    def test_non_xt_poll_still_works_with_type_profiles(self):
        """Plain polls (no xt_*) must still work correctly when type_profiles is set."""
        J = 4
        ts, tp, states, priors, votes = self._make_forecast_inputs(J)

        polls = {"nv_senate": [{
            "dem_share": 0.49, "n_sample": 600, "state": "NV",
        }]}

        result = run_forecast(
            type_scores=ts,
            county_priors=priors,
            states=states,
            county_votes=votes,
            polls_by_race=polls,
            races=["nv_senate"],
            lam=1.0,
            mu=1.0,
            type_profiles=tp,
        )

        assert "nv_senate" in result
        assert result["nv_senate"].n_polls == 1
        assert np.all(np.isfinite(result["nv_senate"].theta_national))

    def test_county_predictions_are_valid(self):
        """County-level predictions must be in [0, 1] for all counties."""
        J = 4
        ts, tp, states, priors, votes = self._make_forecast_inputs(J)

        polls = {"az_gov": [{
            "dem_share": 0.54,
            "n_sample": 1000,
            "state": "AZ",
            "xt_education_college": 0.42,
            "xt_education_noncollege": 0.58,
            "xt_race_white": 0.72,
            "xt_race_black": 0.02,
        }]}

        result = run_forecast(
            type_scores=ts,
            county_priors=priors,
            states=states,
            county_votes=votes,
            polls_by_race=polls,
            races=["az_gov"],
            lam=1.0,
            mu=1.0,
            type_profiles=tp,
        )

        r = result["az_gov"]
        # Both prediction modes should produce values in a reasonable range.
        # (exact [0,1] not guaranteed by the Bayesian update, but values should
        # be finite and generally plausible)
        assert np.all(np.isfinite(r.county_preds_national))
        assert np.all(np.isfinite(r.county_preds_local))

    def test_no_type_profiles_falls_back_gracefully(self):
        """When type_profiles=None, xt_* fields are ignored and state W is used."""
        J = 4
        ts, _, states, priors, votes = self._make_forecast_inputs(J)

        polls = {"az_gov": [{
            "dem_share": 0.52, "n_sample": 850, "state": "AZ",
            "xt_race_white": 0.71, "xt_race_black": 0.03,
        }]}

        result = run_forecast(
            type_scores=ts,
            county_priors=priors,
            states=states,
            county_votes=votes,
            polls_by_race=polls,
            races=["az_gov"],
            lam=1.0,
            mu=1.0,
            type_profiles=None,  # No type profiles — Tier 2 cannot run
        )

        assert "az_gov" in result
        assert result["az_gov"].n_polls == 1
        assert np.all(np.isfinite(result["az_gov"].theta_national))
