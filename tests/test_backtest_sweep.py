"""Tests for src/validation/backtest_sweep.py.

Tests cover:
  - Year-adaptive prior loading (correct year selection, fallback, coverage)
  - Parameterized backtest runner (params injection, output shape)
  - Parameter sweep (grid generation, output DataFrame shape)
  - Prior comparison (Ridge vs adaptive, output structure)
  - Edge cases (missing data, unknown years)
"""
from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.validation.backtest_sweep import (
    _ALL_RACE_CONFIGS,
    _FALLBACK_DEM_SHARE,
    _PRESIDENTIAL_ACTUALS_YEARS,
    _find_prior_presidential_year,
    _format_params,
    _quick_grid,
    build_year_adaptive_priors,
    compare_priors,
    run_backtest_with_params,
    sweep_parameters,
)


# ---------------------------------------------------------------------------
# Year selection logic
# ---------------------------------------------------------------------------

class TestFindPriorPresidentialYear:
    def test_2008_uses_2004(self):
        assert _find_prior_presidential_year(2008) == 2004

    def test_2012_uses_2008(self):
        assert _find_prior_presidential_year(2012) == 2008

    def test_2016_uses_2012(self):
        assert _find_prior_presidential_year(2016) == 2012

    def test_2020_uses_2016(self):
        assert _find_prior_presidential_year(2020) == 2016

    def test_2010_senate_uses_2008(self):
        """Senate 2010 should use the most recent pres election: 2008."""
        assert _find_prior_presidential_year(2010) == 2008

    def test_2018_governor_uses_2016(self):
        """Governor 2018 should use pres 2016."""
        assert _find_prior_presidential_year(2018) == 2016

    def test_2022_uses_2020(self):
        assert _find_prior_presidential_year(2022) == 2020

    def test_year_before_all_data_returns_none(self):
        """If target_year is before all available data, return None."""
        assert _find_prior_presidential_year(1999) is None

    def test_year_2000_returns_none(self):
        """2000 is in our data but there's nothing before it."""
        assert _find_prior_presidential_year(2000) is None


# ---------------------------------------------------------------------------
# Year-adaptive prior loading
# ---------------------------------------------------------------------------

class TestBuildYearAdaptivePriors:
    """Tests that actually load data from disk (integration tests)."""

    def test_2008_priors_from_2004_returns_correct_shape(self):
        """Loading 2004 actuals for 2008 backtest should produce a float array."""
        # Use a small subset of FIPS codes we know exist in 2004 data.
        test_fips = ["01001", "06037", "17031", "36061", "48201"]
        priors = build_year_adaptive_priors(test_fips, 2008)
        assert priors.shape == (5,)
        assert priors.dtype == np.float64 or priors.dtype == np.float32

    def test_priors_are_valid_dem_shares(self):
        """All loaded priors should be in (0, 1) for real counties."""
        test_fips = ["01001", "06037", "17031", "36061", "48201"]
        priors = build_year_adaptive_priors(test_fips, 2012)
        # All should be valid probabilities (or fallback, which is 0.45).
        for i, p in enumerate(priors):
            assert 0.0 < p < 1.0, f"Prior {p} for {test_fips[i]} out of range"

    def test_missing_fips_gets_fallback(self):
        """A fake FIPS code should get the fallback prior."""
        priors = build_year_adaptive_priors(["99999"], 2016)
        assert abs(priors[0] - _FALLBACK_DEM_SHARE) < 1e-9

    def test_very_early_year_gets_all_fallback(self):
        """Year before all available data should produce all-fallback array."""
        test_fips = ["01001", "06037"]
        priors = build_year_adaptive_priors(test_fips, 1999)
        np.testing.assert_allclose(priors, _FALLBACK_DEM_SHARE)

    def test_prior_year_matches_expected(self):
        """Verify that 2020 backtest loads 2016 data (not 2020 itself)."""
        # Load both the 2016 actuals directly and the year-adaptive priors for 2020.
        # They should be identical for the same FIPS.
        test_fips = ["06037"]  # LA County
        assembled_dir = Path(__file__).resolve().parents[1] / "data" / "assembled"

        priors = build_year_adaptive_priors(test_fips, 2020, assembled_dir=assembled_dir)

        df_2016 = pd.read_parquet(assembled_dir / "medsl_county_presidential_2016.parquet")
        df_2016["county_fips"] = df_2016["county_fips"].astype(str).str.zfill(5)
        la_row = df_2016[df_2016["county_fips"] == "06037"]
        expected = la_row["pres_dem_share_2016"].values[0]

        assert abs(priors[0] - expected) < 1e-9, (
            f"Year-adaptive prior {priors[0]} != 2016 actual {expected}"
        )


# ---------------------------------------------------------------------------
# Parameterized backtest runner
# ---------------------------------------------------------------------------

class TestRunBacktestWithParams:
    """Integration tests that run actual backtests with different params."""

    def test_default_params_produce_valid_metrics(self):
        """Run with defaults and check output structure."""
        result = run_backtest_with_params(2020, "president", {})
        assert "r" in result
        assert "rmse" in result
        assert "bias" in result
        assert "direction_accuracy" in result
        assert "n_counties" in result
        assert result["n_counties"] > 0
        assert 0.5 < result["r"] < 1.0, f"Unexpected r={result['r']}"

    def test_year_adaptive_priors_flag(self):
        """Year-adaptive priors should produce different (generally better) results."""
        ridge = run_backtest_with_params(2008, "president", {"use_year_adaptive_priors": False})
        adaptive = run_backtest_with_params(2008, "president", {"use_year_adaptive_priors": True})

        # Both should produce valid metrics.
        assert not math.isnan(ridge["r"])
        assert not math.isnan(adaptive["r"])
        # They should be different (different priors → different forecasts).
        assert ridge["r"] != adaptive["r"], "Ridge and adaptive priors gave identical r"

    def test_custom_lam_changes_output(self):
        """Different lam values should produce different results."""
        low_lam = run_backtest_with_params(2020, "president", {"lam": 0.1})
        high_lam = run_backtest_with_params(2020, "president", {"lam": 10.0})
        assert low_lam["r"] != high_lam["r"], "Different lam values gave identical r"

    def test_poll_blend_scale_changes_output(self):
        """Different poll_blend_scale should affect county predictions."""
        low_k = run_backtest_with_params(2020, "president", {"poll_blend_scale": 1.0})
        high_k = run_backtest_with_params(2020, "president", {"poll_blend_scale": 50.0})
        assert low_k["rmse"] != high_k["rmse"], "Different poll_blend_scale gave identical RMSE"

    def test_senate_backtest_works(self):
        """Senate race backtest should produce valid output."""
        result = run_backtest_with_params(2022, "senate", {"use_year_adaptive_priors": True})
        assert result["n_counties"] > 0
        assert not math.isnan(result["r"])

    def test_governor_backtest_works(self):
        """Governor race backtest should produce valid output."""
        result = run_backtest_with_params(2018, "governor", {})
        assert result["n_counties"] > 0
        assert not math.isnan(result["r"])

    def test_invalid_race_raises(self):
        """Invalid race type should raise."""
        with pytest.raises(ValueError):
            run_backtest_with_params(2020, "dogcatcher", {})

    def test_params_are_recorded_in_output(self):
        """The result dict should include the params that were used."""
        params = {"lam": 2.5, "mu": 0.5, "poll_blend_scale": 7.0}
        result = run_backtest_with_params(2020, "president", params)
        assert result["params"]["lam"] == 2.5
        assert result["params"]["mu"] == 0.5
        assert result["params"]["poll_blend_scale"] == 7.0


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------

class TestSweepParameters:
    def test_small_sweep_output_shape(self):
        """A 2x2 grid over 2 elections should produce 2*2*2 = 8 rows."""
        grid = {"lam": [0.5, 2.0], "mu": [0.5, 2.0]}
        configs = [(2020, "president"), (2022, "senate")]
        df = sweep_parameters(grid, configs)

        assert len(df) == 4 * 2  # 4 param combos × 2 elections
        assert "lam" in df.columns
        assert "mu" in df.columns
        assert "year" in df.columns
        assert "r" in df.columns
        assert "rmse" in df.columns

    def test_sweep_includes_all_metric_columns(self):
        """Output should have r, rmse, bias, direction_accuracy."""
        grid = {"lam": [1.0]}
        configs = [(2020, "president")]
        df = sweep_parameters(grid, configs)

        for col in ["r", "rmse", "bias", "direction_accuracy"]:
            assert col in df.columns, f"Missing column: {col}"
            assert not df[col].isna().all(), f"Column {col} is all NaN"


# ---------------------------------------------------------------------------
# Prior comparison
# ---------------------------------------------------------------------------

class TestComparePriors:
    def test_comparison_output_structure(self):
        """compare_priors should return a DataFrame with both Ridge and adaptive columns."""
        configs = [(2020, "president")]
        df = compare_priors(configs)

        assert len(df) == 1
        for col in ["ridge_r", "adaptive_r", "r_delta", "ridge_rmse", "adaptive_rmse", "rmse_delta"]:
            assert col in df.columns, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

class TestFormatParams:
    def test_float_params(self):
        s = _format_params({"lam": 1.5, "mu": 0.3})
        assert "lam=1.50" in s
        assert "mu=0.30" in s

    def test_bool_params(self):
        s = _format_params({"use_year_adaptive_priors": True})
        assert "Y" in s

    def test_empty_params(self):
        assert _format_params({}) == ""


class TestQuickGrid:
    def test_quick_grid_has_expected_params(self):
        grid, configs = _quick_grid()
        assert "lam" in grid
        assert "mu" in grid
        assert "poll_blend_scale" in grid
        assert "use_year_adaptive_priors" in grid
        assert len(configs) >= 3, "Quick grid should have at least 3 elections"


class TestAllRaceConfigs:
    def test_all_configs_count(self):
        """Should have 4 pres + 7 senate + 3 governor = 14 elections."""
        assert len(_ALL_RACE_CONFIGS) == 14

    def test_all_configs_contain_expected_types(self):
        types = {rt for _, rt in _ALL_RACE_CONFIGS}
        assert types == {"president", "senate", "governor"}
