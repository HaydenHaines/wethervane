"""Tests for BEA state income composition pipeline.

Covers:
  1. fetch_bea_income_composition.py -- API parsing, share computation, error handling
  2. build_bea_income_composition_features.py -- state-to-county mapping
  3. Integration with build_county_features_national.py

All tests use synthetic in-memory DataFrames or temporary parquet files.
No network access, no BEA_API_KEY, and no real disk parquet files required.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.assembly.build_bea_income_composition_features import (
    COL_INVESTMENT,
    COL_TRANSFER,
    COL_WAGES,
    FEATURE_COLS,
    build_county_bea_income_composition,
    load_state_shares,
)
from src.assembly.fetch_bea_income_composition import (
    LINE_DIVIDENDS_INTEREST,
    LINE_PERSONAL_INCOME,
    LINE_TRANSFERS,
    LINE_WAGES_SALARIES,
    SHARE_COLS,
    _parse_data_value,
    compute_state_income_shares,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_raw_sainc4(states: list[dict]) -> pd.DataFrame:
    """Build a minimal SAINC4-style raw DataFrame for testing."""
    rows = []
    for s in states:
        fips = str(s["GeoFips"])
        rows.extend([
            {"GeoFips": fips, "GeoName": fips, "LineCode": str(LINE_PERSONAL_INCOME),
             "TimePeriod": "2023", "DataValue": str(s["personal_income"])},
            {"GeoFips": fips, "GeoName": fips, "LineCode": str(LINE_WAGES_SALARIES),
             "TimePeriod": "2023", "DataValue": str(s["wages"])},
            {"GeoFips": fips, "GeoName": fips, "LineCode": str(LINE_DIVIDENDS_INTEREST),
             "TimePeriod": "2023", "DataValue": str(s["investment"])},
            {"GeoFips": fips, "GeoName": fips, "LineCode": str(LINE_TRANSFERS),
             "TimePeriod": "2023", "DataValue": str(s["transfers"])},
        ])
    return pd.DataFrame(rows)


def _minimal_state_shares(tmp_path: Path) -> Path:
    """Three-state shares parquet for tests."""
    p = tmp_path / "bea_income_composition.parquet"
    pd.DataFrame([
        {"state_fips_prefix": "12", "state_abbr": "FL",
         "wages_share": 0.60, "transfer_share": 0.20, "investment_share": 0.12},
        {"state_fips_prefix": "13", "state_abbr": "GA",
         "wages_share": 0.62, "transfer_share": 0.19, "investment_share": 0.11},
        {"state_fips_prefix": "01", "state_abbr": "AL",
         "wages_share": 0.55, "transfer_share": 0.25, "investment_share": 0.10},
    ]).to_parquet(p, index=False)
    return p


# ---------------------------------------------------------------------------
# 1. _parse_data_value() -- BEA special codes
# ---------------------------------------------------------------------------


class TestParseDataValue:
    def test_numeric_string(self):
        assert _parse_data_value("123,456") == pytest.approx(123456.0)

    def test_float_passthrough(self):
        assert _parse_data_value(99.5) == pytest.approx(99.5)

    def test_suppressed_d(self):
        assert pd.isna(_parse_data_value("(D)"))

    def test_na_code(self):
        assert pd.isna(_parse_data_value("(NA)"))

    def test_empty_string(self):
        assert pd.isna(_parse_data_value(""))

    def test_double_dash(self):
        assert pd.isna(_parse_data_value("--"))

    def test_nan_input(self):
        assert pd.isna(_parse_data_value(float("nan")))

    def test_comma_large_number(self):
        assert _parse_data_value("1,234,567") == pytest.approx(1234567.0)


# ---------------------------------------------------------------------------
# 2. compute_state_income_shares()
# ---------------------------------------------------------------------------


class TestComputeStateIncomeShares:
    def test_basic_shares(self):
        raw = _make_raw_sainc4([{
            "GeoFips": "12", "personal_income": 1_000_000,
            "wages": 600_000, "transfers": 200_000, "investment": 100_000,
        }])
        result = compute_state_income_shares(raw)
        assert len(result) == 1
        assert result["wages_share"].iloc[0] == pytest.approx(0.6)
        assert result["transfer_share"].iloc[0] == pytest.approx(0.2)
        assert result["investment_share"].iloc[0] == pytest.approx(0.1)

    def test_output_columns(self):
        raw = _make_raw_sainc4([{
            "GeoFips": "12", "personal_income": 1_000_000,
            "wages": 600_000, "transfers": 200_000, "investment": 100_000,
        }])
        result = compute_state_income_shares(raw)
        expected = {"state_fips_prefix", "state_abbr", "wages_share", "transfer_share", "investment_share"}
        assert set(result.columns) == expected

    def test_national_aggregate_excluded(self):
        raw = _make_raw_sainc4([
            {"GeoFips": "00", "personal_income": 100_000_000,
             "wages": 60_000_000, "transfers": 20_000_000, "investment": 10_000_000},
            {"GeoFips": "12", "personal_income": 1_000_000,
             "wages": 600_000, "transfers": 200_000, "investment": 100_000},
        ])
        result = compute_state_income_shares(raw)
        assert "00" not in result["state_fips_prefix"].values

    def test_zero_income_excluded(self):
        raw = _make_raw_sainc4([
            {"GeoFips": "12", "personal_income": 0, "wages": 0, "transfers": 0, "investment": 0},
        ])
        assert len(compute_state_income_shares(raw)) == 0

    def test_state_abbr_mapped(self):
        raw = _make_raw_sainc4([{
            "GeoFips": "12", "personal_income": 1_000_000,
            "wages": 600_000, "transfers": 200_000, "investment": 100_000,
        }])
        assert compute_state_income_shares(raw)["state_abbr"].iloc[0] == "FL"

    def test_shares_range(self):
        raw = _make_raw_sainc4([
            {"GeoFips": "12", "personal_income": 1_000_000,
             "wages": 600_000, "transfers": 200_000, "investment": 100_000},
            {"GeoFips": "13", "personal_income": 800_000,
             "wages": 500_000, "transfers": 150_000, "investment": 90_000},
        ])
        result = compute_state_income_shares(raw)
        for col in SHARE_COLS:
            assert result[col].between(0, 1).all()

    def test_multiple_states(self):
        raw = _make_raw_sainc4([
            {"GeoFips": "12", "personal_income": 1_000_000,
             "wages": 600_000, "transfers": 200_000, "investment": 100_000},
            {"GeoFips": "13", "personal_income": 800_000,
             "wages": 500_000, "transfers": 150_000, "investment": 90_000},
            {"GeoFips": "01", "personal_income": 500_000,
             "wages": 280_000, "transfers": 140_000, "investment": 50_000},
        ])
        assert len(compute_state_income_shares(raw)) == 3


# ---------------------------------------------------------------------------
# 3. load_state_shares()
# ---------------------------------------------------------------------------


class TestLoadStateShares:
    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="fetch_bea_income_composition"):
            load_state_shares(tmp_path / "nonexistent.parquet")

    def test_missing_columns(self, tmp_path):
        p = tmp_path / "bad.parquet"
        pd.DataFrame({"state_fips_prefix": ["12"], "wrong_col": [0.5]}).to_parquet(p)
        with pytest.raises(ValueError, match="wages_share"):
            load_state_shares(p)

    def test_indexed_by_fips(self, tmp_path):
        result = load_state_shares(_minimal_state_shares(tmp_path))
        assert result.index.name == "state_fips_prefix"
        assert "12" in result.index

    def test_columns(self, tmp_path):
        result = load_state_shares(_minimal_state_shares(tmp_path))
        assert set(result.columns) == {"wages_share", "transfer_share", "investment_share"}


# ---------------------------------------------------------------------------
# 4. build_county_bea_income_composition()
# ---------------------------------------------------------------------------


class TestBuildCountyBeaIncomeComposition:
    def test_florida_wages(self, tmp_path):
        result = build_county_bea_income_composition(["12001", "12003", "12005"], _minimal_state_shares(tmp_path))
        for val in result[COL_WAGES]:
            assert val == pytest.approx(0.60)

    def test_georgia_transfer(self, tmp_path):
        result = build_county_bea_income_composition(["13001"], _minimal_state_shares(tmp_path))
        assert result[COL_TRANSFER].iloc[0] == pytest.approx(0.19)

    def test_different_states(self, tmp_path):
        result = build_county_bea_income_composition(["12001", "13001", "01001"], _minimal_state_shares(tmp_path))
        assert result[COL_WAGES].nunique() == 3

    def test_output_columns(self, tmp_path):
        result = build_county_bea_income_composition(["12001"], _minimal_state_shares(tmp_path))
        assert set(result.columns) == {"county_fips"} | set(FEATURE_COLS)

    def test_fips_preserved(self, tmp_path):
        fips = ["12001", "13001", "01001"]
        result = build_county_bea_income_composition(fips, _minimal_state_shares(tmp_path))
        assert list(result["county_fips"]) == fips

    def test_row_count(self, tmp_path):
        result = build_county_bea_income_composition(["12001", "12003", "13001", "01001"], _minimal_state_shares(tmp_path))
        assert len(result) == 4

    def test_no_nan_with_unknown_state(self, tmp_path):
        result = build_county_bea_income_composition(["12001", "06001"], _minimal_state_shares(tmp_path))
        assert not result[FEATURE_COLS].isna().any().any()

    def test_unknown_state_gets_median(self, tmp_path):
        result = build_county_bea_income_composition(["12001", "13001", "01001", "06001"], _minimal_state_shares(tmp_path))
        ca = result[result["county_fips"] == "06001"]
        assert not ca[COL_WAGES].isna().any()

    def test_fips_zero_padded(self, tmp_path):
        result = build_county_bea_income_composition(["01001"], _minimal_state_shares(tmp_path))
        assert result["county_fips"].str.len().eq(5).all()

    def test_empty_input(self, tmp_path):
        result = build_county_bea_income_composition([], _minimal_state_shares(tmp_path))
        assert len(result) == 0
        assert set(result.columns) == {"county_fips"} | set(FEATURE_COLS)

    def test_shares_range(self, tmp_path):
        result = build_county_bea_income_composition(["12001", "13001", "01001"], _minimal_state_shares(tmp_path))
        for col in FEATURE_COLS:
            assert result[col].between(0, 1).all()

    def test_feature_cols_constant(self):
        assert len(FEATURE_COLS) == 3
        assert COL_WAGES in FEATURE_COLS
        assert COL_TRANSFER in FEATURE_COLS
        assert COL_INVESTMENT in FEATURE_COLS


# ---------------------------------------------------------------------------
# 5. Integration -- national pipeline
# ---------------------------------------------------------------------------


class TestNationalPipelineIntegration:
    def _make_acs_stub(self, fips):
        return pd.DataFrame({
            "county_fips": [str(f).zfill(5) for f in fips],
            "pop_total": [10_000] * len(fips),
            "pct_white_nh": [0.7] * len(fips),
        })

    def _make_rcms_stub(self, fips):
        return pd.DataFrame({
            "county_fips": [str(f).zfill(5) for f in fips],
            "evangelical_share": [0.2] * len(fips),
            "mainline_share": [0.1] * len(fips),
            "catholic_share": [0.15] * len(fips),
            "black_protestant_share": [0.05] * len(fips),
            "congregations_per_1000": [3.0] * len(fips),
            "religious_adherence_rate": [400.0] * len(fips),
        })

    def _make_bea_composition_stub(self, fips):
        return pd.DataFrame({
            "county_fips": [str(f).zfill(5) for f in fips],
            "bea_wages_share": [0.60] * len(fips),
            "bea_transfer_share": [0.20] * len(fips),
            "bea_investment_share": [0.12] * len(fips),
        })

    def test_columns_present(self):
        from src.assembly.build_county_features_national import build_national_features
        fips = ["12001", "13001", "01001"]
        result = build_national_features(self._make_acs_stub(fips), self._make_rcms_stub(fips), bea_income_composition=self._make_bea_composition_stub(fips))
        assert "bea_wages_share" in result.columns
        assert "bea_transfer_share" in result.columns
        assert "bea_investment_share" in result.columns

    def test_values_pass_through(self):
        from src.assembly.build_county_features_national import build_national_features
        fips = ["12001"]
        result = build_national_features(self._make_acs_stub(fips), self._make_rcms_stub(fips), bea_income_composition=self._make_bea_composition_stub(fips))
        assert result["bea_wages_share"].iloc[0] == pytest.approx(0.60)
        assert result["bea_transfer_share"].iloc[0] == pytest.approx(0.20)
        assert result["bea_investment_share"].iloc[0] == pytest.approx(0.12)

    def test_works_without_bea_composition(self):
        from src.assembly.build_county_features_national import build_national_features
        fips = ["12001", "13001"]
        result = build_national_features(self._make_acs_stub(fips), self._make_rcms_stub(fips))
        assert "bea_wages_share" not in result.columns
        assert len(result) == 2
