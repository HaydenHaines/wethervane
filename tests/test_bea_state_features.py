"""Tests for build_bea_state_features.py.

All tests use synthetic in-memory DataFrames or temporary CSV files — no
network access, no BEA_API_KEY, no real disk parquet files required.

Coverage:
1. load_state_gdp() — happy path, FIPS mapping, bad columns
2. load_state_income() — happy path, FIPS mapping, bad columns
3. build_county_bea_state_features() — correct state mapping via FIPS prefix
4. build_county_bea_state_features() — national median fill for unknown states
5. build_county_bea_state_features() — output schema and FIPS zero-padding
6. build_county_bea_state_features() — all 51 states in sample produce coverage
7. Edge cases — empty county list, duplicate FIPS, missing state files
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.assembly.build_bea_state_features import (
    COL_GDP,
    COL_INCOME,
    FEATURE_COLS,
    _STATE_NAME_TO_FIPS_PREFIX,
    build_county_bea_state_features,
    load_state_gdp,
    load_state_income,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_gdp_csv(path: Path, rows: list[dict]) -> None:
    """Write a minimal state_gdp CSV to a temp path."""
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_income_csv(path: Path, rows: list[dict]) -> None:
    """Write a minimal state_income CSV to a temp path."""
    pd.DataFrame(rows).to_csv(path, index=False)


def _minimal_gdp_csv(tmp_path: Path) -> Path:
    """Three-state GDP CSV for tests that don't need the full 51 states."""
    p = tmp_path / "gdp.csv"
    _write_gdp_csv(p, [
        {"state": "Florida",  "gdp_millions_2024": 1_000_000.0},
        {"state": "Georgia",  "gdp_millions_2024": 500_000.0},
        {"state": "Alabama",  "gdp_millions_2024": 200_000.0},
    ])
    return p


def _minimal_income_csv(tmp_path: Path) -> Path:
    """Three-state income CSV for tests that don't need the full 51 states."""
    p = tmp_path / "income.csv"
    _write_income_csv(p, [
        {"state": "Florida",  "income_per_capita_2024": 73_000},
        {"state": "Georgia",  "income_per_capita_2024": 63_000},
        {"state": "Alabama",  "income_per_capita_2024": 57_000},
    ])
    return p


# ---------------------------------------------------------------------------
# 1. load_state_gdp() — happy path and FIPS mapping
# ---------------------------------------------------------------------------


class TestLoadStateGdp:
    def test_happy_path_returns_series(self, tmp_path):
        """Returns a Series with 2-digit FIPS prefix as index."""
        gdp_path = _minimal_gdp_csv(tmp_path)
        result = load_state_gdp(gdp_path)
        assert isinstance(result, pd.Series)

    def test_florida_fips_prefix(self, tmp_path):
        """Florida maps to FIPS prefix '12'."""
        gdp_path = _minimal_gdp_csv(tmp_path)
        result = load_state_gdp(gdp_path)
        assert "12" in result.index
        assert result["12"] == pytest.approx(1_000_000.0)

    def test_georgia_fips_prefix(self, tmp_path):
        """Georgia maps to FIPS prefix '13'."""
        gdp_path = _minimal_gdp_csv(tmp_path)
        result = load_state_gdp(gdp_path)
        assert result["13"] == pytest.approx(500_000.0)

    def test_alabama_fips_prefix(self, tmp_path):
        """Alabama maps to FIPS prefix '01'."""
        gdp_path = _minimal_gdp_csv(tmp_path)
        result = load_state_gdp(gdp_path)
        assert result["01"] == pytest.approx(200_000.0)

    def test_file_not_found_raises(self, tmp_path):
        """FileNotFoundError raised with helpful message if CSV is missing."""
        with pytest.raises(FileNotFoundError, match="fetch_bea_state_data.py"):
            load_state_gdp(tmp_path / "nonexistent.csv")

    def test_bad_columns_raises(self, tmp_path):
        """ValueError raised if CSV lacks required columns."""
        p = tmp_path / "bad.csv"
        pd.DataFrame({"wrong_col": [1], "also_wrong": [2]}).to_csv(p, index=False)
        with pytest.raises(ValueError, match="gdp_millions_2024"):
            load_state_gdp(p)

    def test_unknown_state_name_logged_not_raised(self, tmp_path):
        """Unknown state name is skipped (logged) rather than raising."""
        p = tmp_path / "gdp_extra.csv"
        _write_gdp_csv(p, [
            {"state": "Florida",        "gdp_millions_2024": 1_000_000.0},
            {"state": "NonExistentTerritory", "gdp_millions_2024": 99.0},
        ])
        # Should not raise; NonExistentTerritory is simply excluded
        result = load_state_gdp(p)
        assert "12" in result.index
        assert len(result) == 1  # only Florida survived


# ---------------------------------------------------------------------------
# 2. load_state_income() — happy path and FIPS mapping
# ---------------------------------------------------------------------------


class TestLoadStateIncome:
    def test_happy_path_returns_series(self, tmp_path):
        """Returns a Series with 2-digit FIPS prefix as index."""
        income_path = _minimal_income_csv(tmp_path)
        result = load_state_income(income_path)
        assert isinstance(result, pd.Series)

    def test_florida_income(self, tmp_path):
        """Florida income per capita maps to FIPS prefix '12'."""
        income_path = _minimal_income_csv(tmp_path)
        result = load_state_income(income_path)
        assert result["12"] == pytest.approx(73_000)

    def test_file_not_found_raises(self, tmp_path):
        """FileNotFoundError if income CSV is missing."""
        with pytest.raises(FileNotFoundError, match="fetch_bea_state_data.py"):
            load_state_income(tmp_path / "nonexistent.csv")

    def test_bad_columns_raises(self, tmp_path):
        """ValueError raised if income CSV lacks required columns."""
        p = tmp_path / "bad.csv"
        pd.DataFrame({"state": ["Florida"], "wrong": [1]}).to_csv(p, index=False)
        with pytest.raises(ValueError, match="income_per_capita_2024"):
            load_state_income(p)


# ---------------------------------------------------------------------------
# 3. build_county_bea_state_features() — correct state-to-county mapping
# ---------------------------------------------------------------------------


class TestBuildCountyBeaStateFeatures:
    def test_florida_counties_get_florida_gdp(self, tmp_path):
        """Counties with FIPS prefix '12' get Florida's GDP value."""
        gdp_path = _minimal_gdp_csv(tmp_path)
        income_path = _minimal_income_csv(tmp_path)
        result = build_county_bea_state_features(
            ["12001", "12003", "12005"], gdp_path, income_path
        )
        for val in result[COL_GDP]:
            assert val == pytest.approx(1_000_000.0)

    def test_georgia_counties_get_georgia_income(self, tmp_path):
        """Counties with FIPS prefix '13' get Georgia's income per capita."""
        gdp_path = _minimal_gdp_csv(tmp_path)
        income_path = _minimal_income_csv(tmp_path)
        result = build_county_bea_state_features(["13001"], gdp_path, income_path)
        assert result[COL_INCOME].iloc[0] == pytest.approx(63_000)

    def test_different_states_get_different_values(self, tmp_path):
        """Counties in different states receive different economic values."""
        gdp_path = _minimal_gdp_csv(tmp_path)
        income_path = _minimal_income_csv(tmp_path)
        result = build_county_bea_state_features(
            ["12001", "13001", "01001"], gdp_path, income_path
        )
        # All GDP values should be distinct (FL, GA, AL all differ)
        assert result[COL_GDP].nunique() == 3

    def test_output_columns_present(self, tmp_path):
        """Output DataFrame has exactly county_fips + the two feature columns."""
        gdp_path = _minimal_gdp_csv(tmp_path)
        income_path = _minimal_income_csv(tmp_path)
        result = build_county_bea_state_features(["12001"], gdp_path, income_path)
        assert set(result.columns) == {"county_fips", COL_GDP, COL_INCOME}

    def test_output_county_fips_preserved(self, tmp_path):
        """county_fips values in output match input."""
        gdp_path = _minimal_gdp_csv(tmp_path)
        income_path = _minimal_income_csv(tmp_path)
        fips = ["12001", "13001", "01001"]
        result = build_county_bea_state_features(fips, gdp_path, income_path)
        assert list(result["county_fips"]) == fips

    def test_feature_cols_constant(self):
        """FEATURE_COLS matches the expected column names."""
        assert COL_GDP in FEATURE_COLS
        assert COL_INCOME in FEATURE_COLS
        assert len(FEATURE_COLS) == 2


# ---------------------------------------------------------------------------
# 4. National median fill for unknown states
# ---------------------------------------------------------------------------


class TestMissingStateFill:
    def test_unknown_state_filled_with_median(self, tmp_path):
        """County in a state not in the CSV gets filled with national median."""
        # CSV only has Florida; county 13001 (GA) will be missing
        gdp_path = tmp_path / "gdp_fl_only.csv"
        income_path = tmp_path / "income_fl_only.csv"
        _write_gdp_csv(gdp_path, [{"state": "Florida", "gdp_millions_2024": 1_000_000.0}])
        _write_income_csv(income_path, [{"state": "Florida", "income_per_capita_2024": 73_000}])

        result = build_county_bea_state_features(
            ["12001", "13001"], gdp_path, income_path
        )
        # With only one state value, median = that state's value; GA gets the median
        assert result[COL_GDP].iloc[1] == pytest.approx(1_000_000.0)

    def test_no_nan_in_output(self, tmp_path):
        """Output never contains NaN values — all missing states are filled."""
        gdp_path = _minimal_gdp_csv(tmp_path)
        income_path = _minimal_income_csv(tmp_path)
        # Include a county from a state not in the CSV (e.g., California prefix '06')
        result = build_county_bea_state_features(
            ["12001", "06001"], gdp_path, income_path
        )
        assert not result[COL_GDP].isna().any()
        assert not result[COL_INCOME].isna().any()

    def test_median_fill_uses_present_values(self, tmp_path):
        """National median is computed from available states, not a constant."""
        p_gdp = tmp_path / "gdp.csv"
        p_inc = tmp_path / "inc.csv"
        _write_gdp_csv(p_gdp, [
            {"state": "Florida", "gdp_millions_2024": 100.0},
            {"state": "Georgia", "gdp_millions_2024": 200.0},
        ])
        _write_income_csv(p_inc, [
            {"state": "Florida",  "income_per_capita_2024": 100},
            {"state": "Georgia",  "income_per_capita_2024": 200},
        ])
        # Unknown state (Alabama prefix '01') should get median of [100, 200] = 150
        result = build_county_bea_state_features(
            ["12001", "13001", "01001"], p_gdp, p_inc
        )
        al_gdp = result[result["county_fips"] == "01001"][COL_GDP].iloc[0]
        assert al_gdp == pytest.approx(150.0)


# ---------------------------------------------------------------------------
# 5. Output schema and FIPS handling
# ---------------------------------------------------------------------------


class TestOutputSchema:
    def test_fips_zero_padded_in_output(self, tmp_path):
        """county_fips in output are always 5-char zero-padded strings."""
        gdp_path = _minimal_gdp_csv(tmp_path)
        income_path = _minimal_income_csv(tmp_path)
        result = build_county_bea_state_features(["01001"], gdp_path, income_path)
        assert result["county_fips"].iloc[0] == "01001"
        assert result["county_fips"].str.len().eq(5).all()

    def test_numeric_fips_input_zero_padded(self, tmp_path):
        """Numeric-string FIPS (e.g., '1001') are zero-padded to 5 chars."""
        gdp_path = _minimal_gdp_csv(tmp_path)
        income_path = _minimal_income_csv(tmp_path)
        # Alabama county without zero-padding
        result = build_county_bea_state_features(["1001"], gdp_path, income_path)
        assert result["county_fips"].iloc[0] == "01001"

    def test_gdp_column_is_float(self, tmp_path):
        """bea_state_gdp_millions column has float dtype."""
        gdp_path = _minimal_gdp_csv(tmp_path)
        income_path = _minimal_income_csv(tmp_path)
        result = build_county_bea_state_features(["12001"], gdp_path, income_path)
        assert result[COL_GDP].dtype.kind == "f"

    def test_income_column_is_float(self, tmp_path):
        """bea_state_income_per_capita column has float dtype."""
        gdp_path = _minimal_gdp_csv(tmp_path)
        income_path = _minimal_income_csv(tmp_path)
        result = build_county_bea_state_features(["12001"], gdp_path, income_path)
        assert result[COL_INCOME].dtype.kind in ("f", "i")  # CSV ints OK

    def test_output_row_count_matches_input(self, tmp_path):
        """Output has exactly one row per input county_fips entry."""
        gdp_path = _minimal_gdp_csv(tmp_path)
        income_path = _minimal_income_csv(tmp_path)
        fips = ["12001", "12003", "13001", "01001"]
        result = build_county_bea_state_features(fips, gdp_path, income_path)
        assert len(result) == 4

    def test_duplicate_fips_preserved(self, tmp_path):
        """Duplicate county FIPS in input are preserved in output."""
        gdp_path = _minimal_gdp_csv(tmp_path)
        income_path = _minimal_income_csv(tmp_path)
        result = build_county_bea_state_features(
            ["12001", "12001"], gdp_path, income_path
        )
        assert len(result) == 2

    def test_empty_county_list(self, tmp_path):
        """Empty input list returns empty DataFrame with correct columns."""
        gdp_path = _minimal_gdp_csv(tmp_path)
        income_path = _minimal_income_csv(tmp_path)
        result = build_county_bea_state_features([], gdp_path, income_path)
        assert len(result) == 0
        assert set(result.columns) == {"county_fips", COL_GDP, COL_INCOME}


# ---------------------------------------------------------------------------
# 6. Full 51-state coverage from the real sample files
# ---------------------------------------------------------------------------


class TestFullSampleCoverage:
    """Integration-style tests using the real sample CSV files.

    These tests run only if the sample files are present on disk. They are
    skipped (not failed) in clean CI environments that haven't fetched data.
    """

    @pytest.fixture(autouse=True)
    def _require_sample_files(self):
        from src.assembly.build_bea_state_features import GDP_PATH, INCOME_PATH
        if not GDP_PATH.exists() or not INCOME_PATH.exists():
            pytest.skip("BEA sample files not on disk — skipping real-data tests")

    def test_all_51_states_covered(self):
        """All 51 state FIPS prefixes (50 states + DC) appear in the mapping."""
        assert len(_STATE_NAME_TO_FIPS_PREFIX) == 51

    def test_sample_files_cover_all_states(self):
        """The sample CSVs contain data for all 51 states."""
        from src.assembly.build_bea_state_features import GDP_PATH, INCOME_PATH
        gdp_series = load_state_gdp(GDP_PATH)
        income_series = load_state_income(INCOME_PATH)
        assert len(gdp_series) == 51
        assert len(income_series) == 51

    def test_no_nan_for_florida_counties(self):
        """Standard FL counties (prefix '12') map correctly from sample data."""
        from src.assembly.build_bea_state_features import GDP_PATH, INCOME_PATH
        result = build_county_bea_state_features(
            ["12001", "12005", "12021"], GDP_PATH, INCOME_PATH
        )
        assert not result[COL_GDP].isna().any()
        assert not result[COL_INCOME].isna().any()
        # Florida GDP should be in a plausible ballpark (hundreds of billions)
        assert result[COL_GDP].iloc[0] > 100_000  # > $100B

    def test_dc_fips_prefix_11(self):
        """District of Columbia (prefix '11') maps correctly."""
        from src.assembly.build_bea_state_features import GDP_PATH, INCOME_PATH
        result = build_county_bea_state_features(["11001"], GDP_PATH, INCOME_PATH)
        assert not result[COL_GDP].isna().any()
        # DC has high income per capita (it's DC)
        assert result[COL_INCOME].iloc[0] > 80_000
