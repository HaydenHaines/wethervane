"""Tests for IRS SOI migration feature builder.

Tests exercise build_irs_migration_features.py — feature computation from
the county-to-county migration edge list.

These tests use synthetic DataFrames and do not touch the filesystem.
They verify:
  - Basic feature computation from a small synthetic edge list
  - Correct handling of counties that appear only as origins (pure outflow)
  - Correct handling of counties that appear only as destinations (pure inflow)
  - Averaging across multiple year_pairs
  - Division-by-zero protection for counties with no inflow
  - Suppression sentinel (n_returns == -1) rows are excluded
  - Output schema and column presence
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.assembly.build_irs_migration_features import (
    TARGET_PREFIXES,
    _INFLOW_FLOOR,
    _is_target,
    _suppress_sentinel,
    build_features,
    build_features_for_year,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_edges(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal migration edge DataFrame from a list of row dicts.

    Fills defaults so individual tests can be concise.
    """
    default = {
        "origin_fips": "13001",  # GA county (origin)
        "dest_fips": "12001",    # FL county (dest)
        "n_returns": 100,
        "n_exemptions": 250,
        "agi": 5000,
        "year_pair": "2021-2022",
    }
    records = [{**default, **r} for r in rows]
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# _is_target
# ---------------------------------------------------------------------------


class TestIsTarget:
    """Tests for the _is_target() helper."""

    def test_fl_is_target(self):
        """FL FIPS prefix 12 must be recognized as a target state."""
        s = pd.Series(["12001", "12086"])
        assert _is_target(s).all()

    def test_ga_is_target(self):
        """GA FIPS prefix 13 must be recognized as a target state."""
        s = pd.Series(["13001", "13121"])
        assert _is_target(s).all()

    def test_al_is_target(self):
        """AL FIPS prefix 01 must be recognized as a target state."""
        s = pd.Series(["01001", "01073"])
        assert _is_target(s).all()

    def test_non_target_state_rejected(self):
        """TX (48) and CA (06) FIPS must not be flagged as target."""
        s = pd.Series(["48001", "06001"])
        assert not _is_target(s).any()

    def test_mixed_series(self):
        """Mixed series must return True only for FL/GA/AL rows."""
        s = pd.Series(["12001", "48001", "13121", "06001"])
        expected = [True, False, True, False]
        assert _is_target(s).tolist() == expected


# ---------------------------------------------------------------------------
# _suppress_sentinel
# ---------------------------------------------------------------------------


class TestSuppressSentinel:
    """Tests for the _suppress_sentinel() helper."""

    def test_drops_minus_one_rows(self):
        """Rows with n_returns == -1 must be dropped."""
        df = _make_edges([
            {"n_returns": -1},
            {"n_returns": 50},
        ])
        result = _suppress_sentinel(df)
        assert len(result) == 1
        assert result.iloc[0]["n_returns"] == 50

    def test_keeps_all_when_no_suppressed(self):
        """No rows must be dropped when n_returns > 0 for all rows."""
        df = _make_edges([
            {"n_returns": 25},
            {"n_returns": 100},
        ])
        result = _suppress_sentinel(df)
        assert len(result) == 2

    def test_empty_input_unchanged(self):
        """Empty DataFrame must pass through unchanged."""
        df = pd.DataFrame(columns=["origin_fips", "dest_fips", "n_returns", "agi", "year_pair"])
        result = _suppress_sentinel(df)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# build_features_for_year — basic computation
# ---------------------------------------------------------------------------


class TestBuildFeaturesForYear:
    """Tests for build_features_for_year() — single year_pair."""

    @pytest.fixture()
    def simple_df(self):
        """Two flows into FL county 12001, one flow out of 12001."""
        return _make_edges([
            # Flow into 12001 from GA (origin = non-target)
            {"origin_fips": "13001", "dest_fips": "12001", "n_returns": 200, "agi": 12000},
            # Flow into 12001 from TX (origin = non-target)
            {"origin_fips": "48001", "dest_fips": "12001", "n_returns": 100, "agi": 8000},
            # Flow out of 12001 to GA
            {"origin_fips": "12001", "dest_fips": "13001", "n_returns": 80, "agi": 4000},
        ])

    def test_output_columns(self, simple_df):
        """Output must have exactly the four expected feature columns."""
        result = build_features_for_year(simple_df)
        expected = {"net_migration_rate", "avg_inflow_income", "migration_diversity",
                    "inflow_outflow_ratio"}
        assert expected == set(result.columns)

    def test_net_migration_rate_positive_for_inflow_county(self, simple_df):
        """County 12001 has more inflow (300) than outflow (80) — rate must be positive."""
        result = build_features_for_year(simple_df)
        assert result.loc["12001", "net_migration_rate"] > 0

    def test_net_migration_rate_formula(self, simple_df):
        """net_migration_rate == (inflow - outflow) / inflow for county 12001."""
        result = build_features_for_year(simple_df)
        inflow = 300
        outflow = 80
        expected = (inflow - outflow) / inflow
        assert abs(result.loc["12001", "net_migration_rate"] - expected) < 1e-9

    def test_avg_inflow_income(self, simple_df):
        """avg_inflow_income == total_inflow_agi / total_inflow_returns."""
        result = build_features_for_year(simple_df)
        expected = (12000 + 8000) / 300
        assert abs(result.loc["12001", "avg_inflow_income"] - expected) < 1e-9

    def test_migration_diversity(self, simple_df):
        """migration_diversity must count unique origin counties (2: GA + TX)."""
        result = build_features_for_year(simple_df)
        assert result.loc["12001", "migration_diversity"] == 2

    def test_inflow_outflow_ratio(self, simple_df):
        """inflow_outflow_ratio == inflow / (inflow + outflow) for county 12001."""
        result = build_features_for_year(simple_df)
        expected = 300 / (300 + 80)
        assert abs(result.loc["12001", "inflow_outflow_ratio"] - expected) < 1e-9

    def test_inflow_outflow_ratio_bounds(self, simple_df):
        """inflow_outflow_ratio must be in [0, 1]."""
        result = build_features_for_year(simple_df)
        assert (result["inflow_outflow_ratio"] >= 0).all()
        assert (result["inflow_outflow_ratio"] <= 1).all()

    def test_only_target_counties_in_output(self, simple_df):
        """Output index must contain only FL/GA/AL FIPS codes."""
        result = build_features_for_year(simple_df)
        for fips in result.index:
            assert fips[:2] in TARGET_PREFIXES, f"Non-target county {fips} found in output"


# ---------------------------------------------------------------------------
# build_features_for_year — pure inflow county
# ---------------------------------------------------------------------------


class TestPureInflowCounty:
    """County that appears only as destination (no outflows)."""

    @pytest.fixture()
    def pure_inflow_df(self):
        """County 12001 receives migrants but none leave."""
        return _make_edges([
            {"origin_fips": "13001", "dest_fips": "12001", "n_returns": 150, "agi": 9000},
            {"origin_fips": "48001", "dest_fips": "12001", "n_returns": 50,  "agi": 3000},
        ])

    def test_outflow_treated_as_zero(self, pure_inflow_df):
        """County with no outflow must have outflow treated as 0."""
        result = build_features_for_year(pure_inflow_df)
        # net_migration_rate = (200 - 0) / 200 = 1.0
        assert abs(result.loc["12001", "net_migration_rate"] - 1.0) < 1e-9

    def test_inflow_outflow_ratio_is_one(self, pure_inflow_df):
        """Pure-inflow county must have inflow_outflow_ratio == 1.0."""
        result = build_features_for_year(pure_inflow_df)
        assert abs(result.loc["12001", "inflow_outflow_ratio"] - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# build_features_for_year — pure outflow county
# ---------------------------------------------------------------------------


class TestPureOutflowCounty:
    """County that appears only as origin (no inflows)."""

    @pytest.fixture()
    def pure_outflow_df(self):
        """County 12001 sends migrants away but receives none."""
        return _make_edges([
            # Outflows from 12001 to a non-FL/GA/AL destination
            {"origin_fips": "12001", "dest_fips": "48001", "n_returns": 300, "agi": 15000},
        ])

    def test_county_present_in_output(self, pure_outflow_df):
        """Pure-outflow county must still appear in the output."""
        result = build_features_for_year(pure_outflow_df)
        assert "12001" in result.index

    def test_no_division_by_zero(self, pure_outflow_df):
        """net_migration_rate must be finite (not NaN or inf) for pure-outflow county."""
        result = build_features_for_year(pure_outflow_df)
        val = result.loc["12001", "net_migration_rate"]
        assert not pd.isna(val), "net_migration_rate is NaN for pure-outflow county"
        assert not (val == float("inf") or val == float("-inf")), "net_migration_rate is infinite"

    def test_net_migration_rate_negative(self, pure_outflow_df):
        """Pure-outflow county must have net_migration_rate < 0."""
        result = build_features_for_year(pure_outflow_df)
        # inflow = 0 (floored to _INFLOW_FLOOR), outflow = 300
        # rate = (floor - 300) / floor, strongly negative
        assert result.loc["12001", "net_migration_rate"] < 0

    def test_inflow_outflow_ratio_near_zero(self, pure_outflow_df):
        """Pure-outflow county must have inflow_outflow_ratio close to 0."""
        result = build_features_for_year(pure_outflow_df)
        # inflow = 0 (floored to floor), total = floor + 300 ≈ 300
        ratio = result.loc["12001", "inflow_outflow_ratio"]
        assert ratio < 0.05, f"Expected near-0 ratio for pure outflow county, got {ratio}"


# ---------------------------------------------------------------------------
# build_features — averaging across multiple year_pairs
# ---------------------------------------------------------------------------


class TestMultiYearAveraging:
    """Tests that build_features() averages correctly across year_pairs."""

    @pytest.fixture()
    def two_year_df(self):
        """County 12001: year 1 has large inflow, year 2 has smaller inflow."""
        return _make_edges([
            # Year 1: large inflow
            {"origin_fips": "13001", "dest_fips": "12001",
             "n_returns": 400, "agi": 20000, "year_pair": "2020-2021"},
            {"origin_fips": "12001", "dest_fips": "13001",
             "n_returns": 100, "agi": 5000, "year_pair": "2020-2021"},
            # Year 2: small inflow
            {"origin_fips": "13001", "dest_fips": "12001",
             "n_returns": 200, "agi": 10000, "year_pair": "2021-2022"},
            {"origin_fips": "12001", "dest_fips": "13001",
             "n_returns": 100, "agi": 5000, "year_pair": "2021-2022"},
        ])

    def test_output_has_one_row_per_county(self, two_year_df):
        """Averaging must collapse two year_pairs into a single row per county."""
        result = build_features(two_year_df)
        assert result["county_fips"].nunique() == len(result)

    def test_net_migration_rate_is_average(self, two_year_df):
        """net_migration_rate must be the average of the two year values."""
        result = build_features(two_year_df)
        row = result[result["county_fips"] == "12001"].iloc[0]
        # Year 1: (400 - 100) / 400 = 0.75
        # Year 2: (200 - 100) / 200 = 0.50
        # Average: 0.625
        expected = (0.75 + 0.50) / 2
        assert abs(row["net_migration_rate"] - expected) < 1e-9

    def test_output_schema(self, two_year_df):
        """build_features() output must have county_fips plus the four feature columns."""
        result = build_features(two_year_df)
        expected_cols = {"county_fips", "net_migration_rate", "avg_inflow_income",
                         "migration_diversity", "inflow_outflow_ratio"}
        assert expected_cols == set(result.columns)

    def test_county_fips_is_string(self, two_year_df):
        """county_fips column must be string dtype."""
        result = build_features(two_year_df)
        assert result["county_fips"].dtype == object

    def test_three_year_pairs_produces_correct_average(self):
        """Average over 3 year_pairs must match manual calculation."""
        # County 12001: inflow=100, outflow=50 each year => rate = 0.5 each year
        rows = []
        for yp in ["2019-2020", "2020-2021", "2021-2022"]:
            rows.append({"origin_fips": "13001", "dest_fips": "12001",
                         "n_returns": 100, "agi": 5000, "year_pair": yp})
            rows.append({"origin_fips": "12001", "dest_fips": "13001",
                         "n_returns": 50, "agi": 2500, "year_pair": yp})
        df = _make_edges(rows)
        # Clear the default row injected by _make_edges by rebuilding cleanly
        df = pd.DataFrame(rows)
        result = build_features(df)
        row = result[result["county_fips"] == "12001"].iloc[0]
        assert abs(row["net_migration_rate"] - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# build_features — division by zero protection
# ---------------------------------------------------------------------------


class TestDivisionByZeroProtection:
    """Ensures no NaN or inf outputs even in degenerate edge cases."""

    def test_pure_outflow_no_nan(self):
        """County with zero inflow must produce finite feature values."""
        df = pd.DataFrame([
            {"origin_fips": "12001", "dest_fips": "13001",
             "n_returns": 200, "agi": 10000, "year_pair": "2021-2022"},
        ])
        result = build_features(df)
        row = result[result["county_fips"] == "12001"].iloc[0]
        for col in ["net_migration_rate", "avg_inflow_income",
                    "migration_diversity", "inflow_outflow_ratio"]:
            val = row[col]
            assert not pd.isna(val), f"{col} is NaN for pure-outflow county"
            assert not (val == float("inf") or val == float("-inf")), f"{col} is infinite"

    def test_single_edge_no_crash(self):
        """Single edge must not raise any exception."""
        df = pd.DataFrame([
            {"origin_fips": "13001", "dest_fips": "12001",
             "n_returns": 50, "agi": 2000, "year_pair": "2021-2022"},
        ])
        result = build_features(df)
        assert len(result) >= 1

    def test_inflow_floor_constant_is_positive(self):
        """_INFLOW_FLOOR must be a positive number."""
        assert _INFLOW_FLOOR > 0


# ---------------------------------------------------------------------------
# build_features — suppressed rows excluded
# ---------------------------------------------------------------------------


class TestSuppressionInPipeline:
    """Verify that suppressed rows (n_returns == -1) are excluded end-to-end."""

    def test_suppressed_rows_not_counted(self):
        """Suppressed flow (n_returns == -1) must not inflate inflow counts."""
        df = pd.DataFrame([
            # Real flow: 100 returns
            {"origin_fips": "13001", "dest_fips": "12001",
             "n_returns": 100, "agi": 5000, "year_pair": "2021-2022"},
            # Suppressed flow: must be ignored
            {"origin_fips": "48001", "dest_fips": "12001",
             "n_returns": -1, "agi": 999, "year_pair": "2021-2022"},
        ])
        result = build_features(df)
        row = result[result["county_fips"] == "12001"].iloc[0]
        # avg_inflow_income must be based on 100 returns, not 99+1 = 100 (suppressed has bad agi)
        # If suppressed row were included: agi = 5000 + 999 = 5999, returns = 99
        # Since suppressed is excluded: agi = 5000, returns = 100 => 50.0
        assert abs(row["avg_inflow_income"] - 50.0) < 1e-9
        # migration_diversity must be 1 (only GA, TX suppressed row excluded)
        assert row["migration_diversity"] == 1


# ---------------------------------------------------------------------------
# Integration test (skipped if parquet not present)
# ---------------------------------------------------------------------------


class TestMigrationFeaturesIntegration:
    """Integration test against the actual saved irs_migration.parquet."""

    @pytest.fixture(scope="class")
    def features(self):
        from pathlib import Path
        raw_path = Path(__file__).parents[1] / "data" / "raw" / "irs_migration.parquet"
        if not raw_path.exists():
            pytest.skip("irs_migration.parquet not found — run fetch_irs_migration.py first")
        raw = pd.read_parquet(raw_path)
        return build_features(raw)

    def test_output_columns(self, features):
        """Output must have county_fips and the four feature columns."""
        expected = {"county_fips", "net_migration_rate", "avg_inflow_income",
                    "migration_diversity", "inflow_outflow_ratio"}
        assert expected == set(features.columns)

    def test_all_counties_target_states(self, features):
        """All county_fips in output must belong to FL, GA, or AL."""
        for fips in features["county_fips"]:
            assert fips[:2] in TARGET_PREFIXES, f"Non-target county {fips} in output"

    def test_no_null_values(self, features):
        """Feature columns must have no null values."""
        assert not features.isnull().any(axis=None), \
            f"Null values found:\n{features.isnull().sum()}"

    def test_no_infinite_values(self, features):
        """Feature columns must have no infinite values."""
        numeric_cols = ["net_migration_rate", "avg_inflow_income",
                        "migration_diversity", "inflow_outflow_ratio"]
        for col in numeric_cols:
            assert not features[col].isin([float("inf"), float("-inf")]).any(), \
                f"Infinite values found in {col}"

    def test_inflow_outflow_ratio_in_bounds(self, features):
        """inflow_outflow_ratio must be in [0, 1] for all counties."""
        assert (features["inflow_outflow_ratio"] >= 0).all()
        assert (features["inflow_outflow_ratio"] <= 1).all()

    def test_migration_diversity_non_negative(self, features):
        """migration_diversity must be >= 0 for all counties."""
        assert (features["migration_diversity"] >= 0).all()

    def test_county_fips_unique(self, features):
        """Each county_fips must appear exactly once (averaged across year_pairs)."""
        assert features["county_fips"].nunique() == len(features)

    def test_reasonable_county_count(self, features):
        """FL+GA+AL have 293 counties total — output should be close to that."""
        n = len(features)
        assert 50 <= n <= 293, f"Unexpected county count: {n} (expected 50–293)"
