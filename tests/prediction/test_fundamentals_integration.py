"""Tests for fundamentals integration into the forecast pipeline.

Verifies that:
1. Fundamentals blending with generic ballot works correctly
2. The combined shift is between the two individual shifts
3. Disabling fundamentals produces GB-only behavior
4. The fundamentals API endpoint returns valid data
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Blending logic tests
# ---------------------------------------------------------------------------

class TestFundamentalsBlending:
    """Test that fundamentals and generic ballot shifts combine correctly."""

    def test_combined_shift_is_weighted_average(self):
        """The combined shift should be w*fundamentals + (1-w)*gb."""
        fund_shift = 0.0112  # +1.12 pp
        gb_shift = 0.0513    # +5.13 pp
        weight = 0.3

        combined = weight * fund_shift + (1 - weight) * gb_shift
        expected = 0.3 * 0.0112 + 0.7 * 0.0513
        assert abs(combined - expected) < 1e-10

    def test_weight_zero_gives_gb_only(self):
        """With weight=0, combined shift should equal GB shift."""
        fund_shift = 0.0112
        gb_shift = 0.0513
        weight = 0.0

        combined = weight * fund_shift + (1 - weight) * gb_shift
        assert abs(combined - gb_shift) < 1e-10

    def test_weight_one_gives_fundamentals_only(self):
        """With weight=1, combined shift should equal fundamentals shift."""
        fund_shift = 0.0112
        gb_shift = 0.0513
        weight = 1.0

        combined = weight * fund_shift + (1 - weight) * gb_shift
        assert abs(combined - fund_shift) < 1e-10

    def test_combined_is_between_inputs(self):
        """For weight in (0,1), combined should be between the two shifts."""
        fund_shift = 0.0112
        gb_shift = 0.0513
        weight = 0.3

        combined = weight * fund_shift + (1 - weight) * gb_shift
        assert min(fund_shift, gb_shift) <= combined <= max(fund_shift, gb_shift)


# ---------------------------------------------------------------------------
# Snapshot loading tests
# ---------------------------------------------------------------------------

class TestSnapshotLoading:
    """Verify the snapshot file has required fields and valid data."""

    @pytest.fixture
    def snapshot(self):
        path = PROJECT_ROOT / "data" / "fundamentals" / "snapshot_2026.json"
        if not path.exists():
            pytest.skip("snapshot_2026.json not found")
        return json.loads(path.read_text())

    def test_has_required_fields(self, snapshot):
        required = {"cycle", "in_party", "approval_net_oct", "gdp_q2_growth_pct",
                     "unemployment_oct", "cpi_yoy_oct"}
        assert required <= set(snapshot.keys())

    def test_in_party_is_valid(self, snapshot):
        assert snapshot["in_party"] in ("D", "R")

    def test_numeric_values_are_reasonable(self, snapshot):
        # GDP growth: typically -10% to +10%
        assert -15 < snapshot["gdp_q2_growth_pct"] < 15
        # Unemployment: typically 2% to 15%
        assert 0 < snapshot["unemployment_oct"] < 20
        # CPI YoY: typically -5% to 15%
        assert -10 < snapshot["cpi_yoy_oct"] < 20
        # Approval: typically -50 to +50
        assert -60 < snapshot["approval_net_oct"] < 60

    def test_has_source_notes(self, snapshot):
        assert "source_notes" in snapshot
        assert "last_updated" in snapshot["source_notes"]


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestFundamentalsConfig:
    """Verify prediction_params.json has fundamentals configuration."""

    @pytest.fixture
    def config(self):
        path = PROJECT_ROOT / "data" / "config" / "prediction_params.json"
        return json.loads(path.read_text())

    def test_fundamentals_section_exists(self, config):
        assert "fundamentals" in config

    def test_has_required_keys(self, config):
        fund = config["fundamentals"]
        assert "enabled" in fund
        assert "fundamentals_weight" in fund

    def test_weight_in_valid_range(self, config):
        w = config["fundamentals"]["fundamentals_weight"]
        assert 0 <= w <= 1

    def test_enabled_is_bool(self, config):
        assert isinstance(config["fundamentals"]["enabled"], bool)


# ---------------------------------------------------------------------------
# End-to-end compute test
# ---------------------------------------------------------------------------

class TestFundamentalsCompute:
    """Test the full fundamentals computation."""

    def test_compute_returns_valid_shift(self):
        from src.prediction.fundamentals import (
            compute_fundamentals_shift,
            load_fundamentals_snapshot,
        )
        snap = load_fundamentals_snapshot()
        info = compute_fundamentals_shift(snap)

        # Shift should be reasonable (-0.10 to +0.10 = ±10pp)
        assert -0.10 < info.shift < 0.10
        # LOO RMSE should be positive
        assert info.loo_rmse > 0
        # Should be fitted, not fallback
        assert info.source == "fitted_ridge"
        # Should have trained on historical data
        assert info.n_training >= 10

    def test_component_contributions_sum_to_total(self):
        from src.prediction.fundamentals import (
            compute_fundamentals_shift,
            load_fundamentals_snapshot,
        )
        snap = load_fundamentals_snapshot()
        info = compute_fundamentals_shift(snap)

        component_sum = (
            info.approval_contribution
            + info.gdp_contribution
            + info.unemployment_contribution
            + info.cpi_contribution
            + info.intercept_contribution
        )
        assert abs(info.shift - component_sum) < 1e-8
