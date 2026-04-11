"""Tests for src.prediction.state_economics — state-level QCEW economic signals.

Tests cover:
  - Feature computation from QCEW data
  - County-to-state mapping
  - State-varying fundamentals adjustment
  - Edge cases (missing data, sensitivity=0)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.prediction.state_economics import (
    _DEFAULT_ECON_SENSITIVITY,
    ECON_FEATURE_COLS,
    build_state_econ_features,
    compute_state_econ_adjustment,
    map_county_econ_features,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_qcew_parquet(tmp_path: Path) -> Path:
    """Create a minimal QCEW parquet with 3 states and 2 years."""
    rows = []
    # State 01 (AL) — 2 counties
    for county, emp_20, emp_23, wage_20, wage_23 in [
        ("01001", 1000, 1100, 40_000_000, 48_000_000),
        ("01003", 2000, 2100, 80_000_000, 92_000_000),
    ]:
        for year, emp, wage in [(2021, emp_20, wage_20), (2023, emp_23, wage_23)]:
            rows.append({
                "county_fips": county, "own_code": "0",
                "industry_code": "10", "year": year,
                "annual_avg_estabs": 100, "annual_avg_emplvl": emp,
                "total_annual_wages": wage,
            })
            # Manufacturing
            rows.append({
                "county_fips": county, "own_code": "0",
                "industry_code": "31-33", "year": year,
                "annual_avg_estabs": 20, "annual_avg_emplvl": int(emp * 0.15),
                "total_annual_wages": int(wage * 0.15),
            })

    # State 06 (CA) — 1 county, slower growth
    for year, emp, wage in [(2021, 5000, 250_000_000), (2023, 5050, 270_000_000)]:
        rows.append({
            "county_fips": "06037", "own_code": "0",
            "industry_code": "10", "year": year,
            "annual_avg_estabs": 500, "annual_avg_emplvl": emp,
            "total_annual_wages": wage,
        })
        rows.append({
            "county_fips": "06037", "own_code": "0",
            "industry_code": "31-33", "year": year,
            "annual_avg_estabs": 50, "annual_avg_emplvl": int(emp * 0.05),
            "total_annual_wages": int(wage * 0.05),
        })

    # State 48 (TX) — 1 county, fastest growth
    for year, emp, wage in [(2021, 3000, 120_000_000), (2023, 3300, 145_000_000)]:
        rows.append({
            "county_fips": "48201", "own_code": "0",
            "industry_code": "10", "year": year,
            "annual_avg_estabs": 300, "annual_avg_emplvl": emp,
            "total_annual_wages": wage,
        })
        rows.append({
            "county_fips": "48201", "own_code": "0",
            "industry_code": "31-33", "year": year,
            "annual_avg_estabs": 40, "annual_avg_emplvl": int(emp * 0.20),
            "total_annual_wages": int(wage * 0.20),
        })

    df = pd.DataFrame(rows)
    path = tmp_path / "qcew_county.parquet"
    df.to_parquet(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Tests: build_state_econ_features
# ---------------------------------------------------------------------------


class TestBuildStateEconFeatures:
    def test_produces_expected_columns(self, sample_qcew_parquet: Path):
        result = build_state_econ_features(qcew_path=sample_qcew_parquet)
        assert "state_fips" in result.columns
        for col in ECON_FEATURE_COLS:
            assert col in result.columns

    def test_produces_one_row_per_state(self, sample_qcew_parquet: Path):
        result = build_state_econ_features(qcew_path=sample_qcew_parquet)
        assert len(result) == 3  # AL, CA, TX

    def test_relative_growth_sums_near_zero(self, sample_qcew_parquet: Path):
        """Relative growth (deviation from national) should roughly center on zero."""
        result = build_state_econ_features(qcew_path=sample_qcew_parquet)
        # Not exactly zero because employment-weighted mean != simple mean,
        # but should be close.
        assert abs(result["qcew_emp_growth_rel"].mean()) < 0.05

    def test_manufacturing_share_reasonable(self, sample_qcew_parquet: Path):
        result = build_state_econ_features(qcew_path=sample_qcew_parquet)
        mfg = result.set_index("state_fips")["qcew_mfg_emp_share"]
        # TX has 20% mfg, CA has 5%
        assert mfg.loc["48"] > mfg.loc["06"]
        # All shares between 0 and 1
        assert (mfg >= 0).all()
        assert (mfg <= 1).all()

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="QCEW"):
            build_state_econ_features(qcew_path=tmp_path / "nonexistent.parquet")

    def test_invalid_year_raises(self, sample_qcew_parquet: Path):
        with pytest.raises(ValueError, match="year_start"):
            build_state_econ_features(qcew_path=sample_qcew_parquet, year_start=2019)


# ---------------------------------------------------------------------------
# Tests: map_county_econ_features
# ---------------------------------------------------------------------------


class TestMapCountyEconFeatures:
    def test_maps_counties_to_state_signal(self, sample_qcew_parquet: Path):
        state_econ = build_state_econ_features(qcew_path=sample_qcew_parquet)
        counties = ["01001", "01003", "06037", "48201"]
        mapped = map_county_econ_features(counties, state_econ=state_econ)

        assert len(mapped) == 4
        # Both AL counties should have the same value
        al_vals = mapped[mapped["county_fips"].str.startswith("01")]
        assert al_vals["qcew_emp_growth_rel"].nunique() == 1

    def test_unknown_state_gets_zero(self, sample_qcew_parquet: Path):
        state_econ = build_state_econ_features(qcew_path=sample_qcew_parquet)
        counties = ["99001"]  # Fictitious state
        mapped = map_county_econ_features(counties, state_econ=state_econ)
        for col in ECON_FEATURE_COLS:
            assert mapped[col].iloc[0] == 0.0

    def test_preserves_input_order(self, sample_qcew_parquet: Path):
        state_econ = build_state_econ_features(qcew_path=sample_qcew_parquet)
        counties = ["48201", "01001", "06037"]
        mapped = map_county_econ_features(counties, state_econ=state_econ)
        assert mapped["county_fips"].tolist() == counties


# ---------------------------------------------------------------------------
# Tests: compute_state_econ_adjustment
# ---------------------------------------------------------------------------


class TestComputeStateEconAdjustment:
    def test_zero_sensitivity_returns_uniform(self, sample_qcew_parquet: Path):
        state_econ = build_state_econ_features(qcew_path=sample_qcew_parquet)
        counties = ["01001", "06037", "48201"]
        states = ["AL", "CA", "TX"]
        result = compute_state_econ_adjustment(
            county_fips=counties, states=states,
            national_shift=0.02, econ_sensitivity=0.0,
            state_econ=state_econ,
        )
        np.testing.assert_allclose(result, 0.02, atol=1e-10)

    def test_nonzero_sensitivity_creates_variation(self, sample_qcew_parquet: Path):
        state_econ = build_state_econ_features(qcew_path=sample_qcew_parquet)
        counties = ["01001", "06037", "48201"]
        states = ["AL", "CA", "TX"]
        result = compute_state_econ_adjustment(
            county_fips=counties, states=states,
            national_shift=0.0, econ_sensitivity=1.0,
            state_econ=state_econ,
        )
        # Not all the same
        assert result.max() - result.min() > 0.01

    def test_output_shape_matches_input(self, sample_qcew_parquet: Path):
        state_econ = build_state_econ_features(qcew_path=sample_qcew_parquet)
        counties = ["01001", "01003", "06037", "48201"]
        states = ["AL", "AL", "CA", "TX"]
        result = compute_state_econ_adjustment(
            county_fips=counties, states=states,
            national_shift=0.01, econ_sensitivity=0.5,
            state_econ=state_econ,
        )
        assert result.shape == (4,)

    def test_missing_qcew_falls_back_to_national(self):
        """When QCEW data file doesn't exist, should return uniform national shift."""
        counties = ["01001", "06037"]
        states = ["AL", "CA"]
        result = compute_state_econ_adjustment(
            county_fips=counties, states=states,
            national_shift=0.03, econ_sensitivity=0.5,
            qcew_path=Path("/nonexistent/qcew.parquet"),
        )
        np.testing.assert_allclose(result, 0.03, atol=1e-10)

    def test_default_sensitivity_is_positive(self):
        assert _DEFAULT_ECON_SENSITIVITY > 0
        assert _DEFAULT_ECON_SENSITIVITY <= 2.0


# ---------------------------------------------------------------------------
# Tests: integration with forecast_engine
# ---------------------------------------------------------------------------


class TestForecastEngineIntegration:
    """Verify the forecast engine accepts array-typed generic_ballot_shift."""

    def test_forecast_engine_accepts_array_shift(self):
        """Run forecast_engine.run_forecast with an array shift (smoke test)."""
        from src.prediction.forecast_engine import run_forecast

        n_counties = 10
        J = 3
        type_scores = np.random.dirichlet(np.ones(J), size=n_counties)
        county_priors = np.random.uniform(0.3, 0.7, n_counties)
        states = ["AL"] * 5 + ["CA"] * 5
        county_votes = np.ones(n_counties)

        # Array-typed shift (per-county)
        shift_array = np.random.uniform(-0.02, 0.02, n_counties)

        results = run_forecast(
            type_scores=type_scores,
            county_priors=county_priors,
            states=states,
            county_votes=county_votes,
            polls_by_race={},
            races=["test_race"],
            generic_ballot_shift=shift_array,
        )

        assert "test_race" in results
        fr = results["test_race"]
        assert fr.county_preds_national.shape == (n_counties,)
        assert fr.county_preds_local.shape == (n_counties,)
