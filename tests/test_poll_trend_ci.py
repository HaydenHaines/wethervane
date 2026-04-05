"""Tests for poll trend CI band computation in race_detail router.

Covers:
  - CI half-width is 2 × backtest RMSE by race type
  - Senate races use narrower band than Governor races
  - CI arrays are same length as trend arrays
  - CI values are clamped to [0, 1]
  - CI fields absent when trend is None (< 2 polls)
  - race_type routing via slug parsing
"""
from __future__ import annotations

import pandas as pd
import pytest

from api.routers.forecast._helpers import _SENATE_STD_FLOOR, _GOVERNOR_STD_FLOOR
from api.routers.forecast.race_detail import _compute_poll_trend


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_polls_df(
    n: int = 4,
    base_dem: float = 0.50,
    start_date: str = "2026-01-01",
) -> pd.DataFrame:
    """Construct a minimal polls DataFrame for _compute_poll_trend."""
    dates = pd.date_range(start=start_date, periods=n, freq="14D")
    dem_shares = [base_dem + i * 0.005 for i in range(n)]
    n_samples = [600] * n
    pollsters = [f"Pollster{i}" for i in range(n)]
    return pd.DataFrame({
        "date": [str(d.date()) for d in dates],
        "pollster": pollsters,
        "dem_share": dem_shares,
        "n_sample": n_samples,
    })


# ── CI width tests ────────────────────────────────────────────────────────────

class TestCIHalfWidth:
    """CI half-width should be 2 × race-type floor."""

    def test_senate_ci_half_width(self):
        """Senate: CI half = 2 × 0.037 = 0.074."""
        df = _make_polls_df()
        trend = _compute_poll_trend(df, race_type="senate")
        assert trend is not None
        expected_half = 2.0 * _SENATE_STD_FLOOR  # 0.074
        for dem, lo, hi in zip(trend.dem_trend, trend.dem_ci_lower, trend.dem_ci_upper):
            # Allow for floating-point rounding to 4dp
            assert abs((dem - lo) - expected_half) < 1e-3, (
                f"Dem lower: expected {dem - expected_half:.4f}, got {lo}"
            )
            assert abs((hi - dem) - expected_half) < 1e-3, (
                f"Dem upper: expected {dem + expected_half:.4f}, got {hi}"
            )

    def test_governor_ci_half_width(self):
        """Governor: CI half = 2 × 0.055 = 0.110 — wider than Senate."""
        df = _make_polls_df()
        trend_senate = _compute_poll_trend(df, race_type="senate")
        trend_gov = _compute_poll_trend(df, race_type="governor")
        assert trend_senate is not None
        assert trend_gov is not None
        # Governor band width must exceed Senate band width
        senate_width = trend_senate.dem_ci_upper[0] - trend_senate.dem_ci_lower[0]
        gov_width = trend_gov.dem_ci_upper[0] - trend_gov.dem_ci_lower[0]
        assert gov_width > senate_width, (
            f"Governor width {gov_width:.4f} should exceed Senate width {senate_width:.4f}"
        )

    def test_governor_specific_width(self):
        """Governor: CI half = 2 × 0.055 = 0.110."""
        df = _make_polls_df()
        trend = _compute_poll_trend(df, race_type="governor")
        assert trend is not None
        expected_half = 2.0 * _GOVERNOR_STD_FLOOR  # 0.110
        for dem, lo, hi in zip(trend.dem_trend, trend.dem_ci_lower, trend.dem_ci_upper):
            assert abs((dem - lo) - expected_half) < 1e-3
            assert abs((hi - dem) - expected_half) < 1e-3

    def test_unknown_race_type_uses_generic_floor(self):
        """Unrecognized race types fall back to _GENERIC_STD_FLOOR (3.5pp)."""
        from api.routers.forecast._helpers import _GENERIC_STD_FLOOR
        df = _make_polls_df()
        trend = _compute_poll_trend(df, race_type="unknown_office")
        assert trend is not None
        expected_half = 2.0 * _GENERIC_STD_FLOOR
        for dem, lo, hi in zip(trend.dem_trend, trend.dem_ci_lower, trend.dem_ci_upper):
            assert abs((dem - lo) - expected_half) < 1e-3


# ── Array length tests ────────────────────────────────────────────────────────

class TestCIArrayLengths:
    """CI arrays must be parallel to the trend date/value arrays."""

    def test_ci_arrays_same_length_as_trend(self):
        df = _make_polls_df(n=6)
        trend = _compute_poll_trend(df, race_type="senate")
        assert trend is not None
        n = len(trend.dates)
        assert len(trend.dem_ci_lower) == n
        assert len(trend.dem_ci_upper) == n
        assert len(trend.rep_ci_lower) == n
        assert len(trend.rep_ci_upper) == n

    def test_dem_rep_ci_are_symmetric(self):
        """Rep CI mirrors dem CI since rep_share = 1 - dem_share."""
        df = _make_polls_df()
        trend = _compute_poll_trend(df, race_type="senate")
        assert trend is not None
        for dem_lo, rep_hi in zip(trend.dem_ci_lower, trend.rep_ci_upper):
            # dem_lo + rep_hi ≈ 1.0 (sum of two-party shares = 1)
            assert abs(dem_lo + rep_hi - 1.0) < 1e-3, (
                f"dem_ci_lower {dem_lo:.4f} + rep_ci_upper {rep_hi:.4f} != 1.0"
            )
        for dem_hi, rep_lo in zip(trend.dem_ci_upper, trend.rep_ci_lower):
            assert abs(dem_hi + rep_lo - 1.0) < 1e-3


# ── Clamping tests ────────────────────────────────────────────────────────────

class TestCIClamping:
    """CI bounds must stay within [0, 1]."""

    def test_ci_lower_clamped_to_zero(self):
        """When dem_share is very low, ci_lower must not go below 0."""
        df = _make_polls_df(base_dem=0.02)
        trend = _compute_poll_trend(df, race_type="governor")
        assert trend is not None
        for lo in trend.dem_ci_lower:
            assert lo >= 0.0, f"CI lower {lo} below 0"

    def test_ci_upper_clamped_to_one(self):
        """When dem_share is very high, ci_upper must not exceed 1."""
        df = _make_polls_df(base_dem=0.98)
        trend = _compute_poll_trend(df, race_type="governor")
        assert trend is not None
        for hi in trend.dem_ci_upper:
            assert hi <= 1.0, f"CI upper {hi} above 1"

    def test_ci_all_values_in_unit_interval(self):
        df = _make_polls_df(base_dem=0.50)
        trend = _compute_poll_trend(df, race_type="governor")
        assert trend is not None
        for arr in (trend.dem_ci_lower, trend.dem_ci_upper,
                    trend.rep_ci_lower, trend.rep_ci_upper):
            for v in arr:
                assert 0.0 <= v <= 1.0, f"CI value {v} outside [0, 1]"


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestCIEdgeCases:
    """Verify graceful handling of degenerate inputs."""

    def test_returns_none_for_single_poll(self):
        """Trend requires at least 2 polls — single poll returns None."""
        df = _make_polls_df(n=1)
        assert _compute_poll_trend(df, race_type="senate") is None

    def test_returns_none_for_empty_dataframe(self):
        """Empty DataFrame returns None."""
        df = pd.DataFrame(columns=["date", "pollster", "dem_share", "n_sample"])
        assert _compute_poll_trend(df, race_type="senate") is None

    def test_ci_bounds_ordered(self):
        """ci_lower must always be less than or equal to ci_upper."""
        df = _make_polls_df(n=4, base_dem=0.50)
        trend = _compute_poll_trend(df, race_type="governor")
        assert trend is not None
        for lo, hi in zip(trend.dem_ci_lower, trend.dem_ci_upper):
            assert lo <= hi, f"ci_lower {lo} > ci_upper {hi}"
        for lo, hi in zip(trend.rep_ci_lower, trend.rep_ci_upper):
            assert lo <= hi

    def test_two_polls_minimum_works(self):
        """Exactly 2 polls should produce a valid trend with CI."""
        df = _make_polls_df(n=2)
        trend = _compute_poll_trend(df, race_type="senate")
        assert trend is not None
        assert len(trend.dem_ci_lower) == len(trend.dates)
        assert len(trend.dem_ci_upper) == len(trend.dates)
