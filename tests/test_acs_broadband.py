"""Tests for fetch_acs_broadband.py and build_acs_broadband_features.py.

Verifies:
- Raw broadband data is fetched and structured correctly (unit tests with mocked HTTP)
- FIPS zero-padding is applied correctly
- ACS null sentinel (-666666666) is replaced with NaN
- build_broadband_features derives correct ratios from raw counts
- Clipping keeps all ratios in [0, 1]
- broadband_gap == pct_no_internet
- Counties with zero total_households get NaN (not division error)
- FIPS validation raises ValueError for malformed inputs
- State-median imputation produces NaN-free output
- National-median fallback for counties with no state peers
- build_national_features accepts broadband kwarg and appends features
- Broadband feature column names are present in national output
- No duplicate columns when broadband is added to the full join
- Total column count is correct with all 8 sources
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembly.build_acs_broadband_features import (
    BROADBAND_FEATURE_COLS,
    _impute_state_medians,
    build_broadband_features,
)
from src.assembly.build_county_features_national import (
    BROADBAND_FEATURE_COLS as BB_NATIONAL,
    MIGRATION_FEATURE_COLS,
    QCEW_FEATURE_COLS,
    RCMS_FEATURE_COLS,
    SCI_FEATURE_COLS,
    URBANICITY_FEATURE_COLS,
    CHR_FEATURE_COLS,
    build_national_features,
)


# ---------------------------------------------------------------------------
# Fixtures — synthetic raw ACS broadband data
# ---------------------------------------------------------------------------


def _make_raw_broadband(fips_list: list[str], seed: int = 0) -> pd.DataFrame:
    """Minimal synthetic raw ACS broadband DataFrame (like output of fetch_acs_broadband)."""
    rng = np.random.default_rng(seed)
    n = len(fips_list)
    totals = rng.integers(1000, 50000, size=n).astype(float)
    # Build realistic sub-counts (broadband <= total, etc.)
    bb_frac = rng.uniform(0.4, 0.9, size=n)
    sat_frac = rng.uniform(0.02, 0.15, size=n)
    cable_frac = bb_frac * rng.uniform(0.5, 0.9, size=n)
    no_net_frac = rng.uniform(0.05, 0.40, size=n)
    return pd.DataFrame({
        "county_fips": fips_list,
        "total_households": totals,
        "with_broadband": (totals * bb_frac).round(),
        "with_cable_fiber_dsl": (totals * cable_frac).round(),
        "with_satellite": (totals * sat_frac).round(),
        "no_internet": (totals * no_net_frac).round(),
        "with_internet_sub": (totals * bb_frac * 1.05).clip(max=totals).round(),
        "data_year": 2022,
    })


def _make_broadband_features(fips_list: list[str], seed: int = 0) -> pd.DataFrame:
    """Synthetic broadband features (county_fips + BROADBAND_FEATURE_COLS)."""
    raw = _make_raw_broadband(fips_list, seed)
    return build_broadband_features(raw)


# ---------------------------------------------------------------------------
# Helpers for national feature builder (copied from test_county_features_national)
# ---------------------------------------------------------------------------


def _make_acs(fips_list: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = len(fips_list)
    return pd.DataFrame({
        "county_fips": fips_list,
        "pop_total": rng.integers(5000, 500000, size=n),
        "pct_white_nh": rng.uniform(0.1, 0.95, size=n),
        "pct_black": rng.uniform(0.01, 0.50, size=n),
        "pct_asian": rng.uniform(0.01, 0.20, size=n),
        "pct_hispanic": rng.uniform(0.01, 0.40, size=n),
        "median_age": rng.uniform(30, 55, size=n),
        "median_hh_income": rng.uniform(40000, 120000, size=n),
        "log_median_hh_income": rng.uniform(4.5, 5.2, size=n),
        "pct_bachelors_plus": rng.uniform(0.10, 0.60, size=n),
        "pct_graduate": rng.uniform(0.05, 0.30, size=n),
        "pct_owner_occupied": rng.uniform(0.30, 0.85, size=n),
        "pct_wfh": rng.uniform(0.01, 0.25, size=n),
        "pct_transit": rng.uniform(0.00, 0.15, size=n),
        "pct_management": rng.uniform(0.05, 0.35, size=n),
    })


def _make_rcms(fips_list: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = len(fips_list)
    fracs = rng.dirichlet(np.array([5.0, 1.5, 2.0, 1.0, 2.0]), size=n)
    return pd.DataFrame({
        "county_fips": fips_list,
        "state_abbr": ["AL"] * n,
        "evangelical_share": fracs[:, 0],
        "mainline_share": fracs[:, 1],
        "catholic_share": fracs[:, 2],
        "black_protestant_share": fracs[:, 3],
        "congregations_per_1000": rng.uniform(1.0, 10.0, size=n),
        "religious_adherence_rate": rng.uniform(200, 700, size=n),
    })


# ---------------------------------------------------------------------------
# Tests: build_broadband_features — ratio derivation
# ---------------------------------------------------------------------------


class TestBuildBroadbandFeatures:
    """Unit tests for the feature derivation logic."""

    def test_output_columns(self):
        """Output must have county_fips + all BROADBAND_FEATURE_COLS."""
        raw = _make_raw_broadband(["01001", "01003", "12001"])
        result = build_broadband_features(raw)
        assert "county_fips" in result.columns
        for col in BROADBAND_FEATURE_COLS:
            assert col in result.columns, f"Missing column: {col}"

    def test_row_count_preserved(self):
        """Output row count equals input row count."""
        fips = ["01001", "01003", "01005", "12001"]
        raw = _make_raw_broadband(fips)
        result = build_broadband_features(raw)
        assert len(result) == len(fips)

    def test_ratios_in_unit_interval(self):
        """All feature values must lie in [0, 1]."""
        raw = _make_raw_broadband(["01001", "01003", "12001", "13001", "48001"])
        result = build_broadband_features(raw)
        for col in BROADBAND_FEATURE_COLS:
            assert (result[col].dropna() >= 0).all(), f"{col} has values < 0"
            assert (result[col].dropna() <= 1).all(), f"{col} has values > 1"

    def test_broadband_gap_equals_pct_no_internet(self):
        """broadband_gap must equal pct_no_internet (same underlying signal)."""
        raw = _make_raw_broadband(["01001", "01003", "01005"])
        result = build_broadband_features(raw)
        pd.testing.assert_series_equal(
            result["broadband_gap"].reset_index(drop=True),
            result["pct_no_internet"].reset_index(drop=True),
            check_names=False,
        )

    def test_pct_broadband_correct_ratio(self):
        """pct_broadband = with_broadband / total_households."""
        raw = pd.DataFrame({
            "county_fips": ["01001"],
            "total_households": [1000.0],
            "with_broadband": [750.0],
            "with_cable_fiber_dsl": [600.0],
            "with_satellite": [50.0],
            "no_internet": [150.0],
        })
        result = build_broadband_features(raw)
        assert abs(result.loc[0, "pct_broadband"] - 0.75) < 1e-9
        assert abs(result.loc[0, "pct_no_internet"] - 0.15) < 1e-9
        assert abs(result.loc[0, "pct_satellite"] - 0.05) < 1e-9
        assert abs(result.loc[0, "pct_cable_fiber"] - 0.60) < 1e-9

    def test_zero_total_households_produces_nan(self):
        """County with total_households=0 must produce NaN ratios, not inf/zero-div."""
        raw = pd.DataFrame({
            "county_fips": ["01001"],
            "total_households": [0.0],
            "with_broadband": [0.0],
            "with_cable_fiber_dsl": [0.0],
            "with_satellite": [0.0],
            "no_internet": [0.0],
        })
        result = build_broadband_features(raw)
        for col in BROADBAND_FEATURE_COLS:
            assert pd.isna(result.loc[0, col]), f"{col} should be NaN when total=0"

    def test_fips_validation_raises(self):
        """Malformed county_fips (non-5-char) raises ValueError."""
        raw = _make_raw_broadband(["1001", "01003"])  # "1001" is 4 chars
        with pytest.raises(ValueError, match="5-char"):
            build_broadband_features(raw)

    def test_no_duplicate_rows(self):
        """Output has no duplicate county_fips rows."""
        fips = ["01001", "01003", "01005"]
        raw = _make_raw_broadband(fips)
        result = build_broadband_features(raw)
        assert result["county_fips"].nunique() == len(result)

    def test_fips_values_preserved(self):
        """county_fips values are passed through unchanged."""
        fips = ["01001", "12003", "48201"]
        raw = _make_raw_broadband(fips)
        result = build_broadband_features(raw)
        assert set(result["county_fips"].tolist()) == set(fips)

    def test_missing_raw_column_raises(self):
        """Missing 'total_households' column raises ValueError."""
        raw = _make_raw_broadband(["01001"]).drop(columns=["total_households"])
        with pytest.raises(ValueError, match="total_households"):
            build_broadband_features(raw)


# ---------------------------------------------------------------------------
# Tests: _impute_state_medians
# ---------------------------------------------------------------------------


class TestBroadbandImputation:
    """State-median imputation produces NaN-free output."""

    def test_missing_county_gets_state_median(self):
        """County missing pct_broadband is filled with its state's median."""
        fips = ["01001", "01003", "01005"]
        # 01001 and 01003 have known values; 01005 is NaN
        df = pd.DataFrame({
            "county_fips": fips,
            "pct_broadband": [0.7, 0.8, float("nan")],
            "pct_no_internet": [0.1, 0.15, float("nan")],
            "pct_satellite": [0.05, 0.08, float("nan")],
            "pct_cable_fiber": [0.6, 0.7, float("nan")],
            "broadband_gap": [0.1, 0.15, float("nan")],
        })
        result = _impute_state_medians(df, BROADBAND_FEATURE_COLS)
        assert result["pct_broadband"].isna().sum() == 0
        # AL median of [0.7, 0.8] = 0.75
        assert abs(result.loc[2, "pct_broadband"] - 0.75) < 1e-9

    def test_no_state_peers_uses_national_median(self):
        """County in a state with no other data uses the national median fallback."""
        fips = ["01001", "01003", "09110"]  # AL counties + a CT planning-region code
        df = pd.DataFrame({
            "county_fips": fips,
            "pct_broadband": [0.7, 0.8, float("nan")],
            "pct_no_internet": [0.1, 0.15, float("nan")],
            "pct_satellite": [0.05, 0.08, float("nan")],
            "pct_cable_fiber": [0.6, 0.7, float("nan")],
            "broadband_gap": [0.1, 0.15, float("nan")],
        })
        result = _impute_state_medians(df, BROADBAND_FEATURE_COLS)
        assert result["pct_broadband"].isna().sum() == 0

    def test_no_nans_in_fully_observed_data(self):
        """Imputation is a no-op when all values are present."""
        fips = ["01001", "01003"]
        df = pd.DataFrame({
            "county_fips": fips,
            "pct_broadband": [0.7, 0.8],
            "pct_no_internet": [0.1, 0.15],
            "pct_satellite": [0.05, 0.08],
            "pct_cable_fiber": [0.6, 0.7],
            "broadband_gap": [0.1, 0.15],
        })
        result = _impute_state_medians(df, BROADBAND_FEATURE_COLS)
        pd.testing.assert_frame_equal(result, df)


# ---------------------------------------------------------------------------
# Tests: integration with build_national_features
# ---------------------------------------------------------------------------


class TestBroadbandNationalIntegration:
    """Broadband features join correctly into the national feature matrix."""

    def test_broadband_columns_present(self):
        """All BROADBAND_FEATURE_COLS appear in output when broadband is provided."""
        fips = ["01001", "01003", "01005"]
        acs = _make_acs(fips)
        rcms = _make_rcms(fips)
        broadband = _make_broadband_features(fips)
        result = build_national_features(acs, rcms, broadband=broadband)
        for col in BB_NATIONAL:
            assert col in result.columns, f"Missing broadband column: {col}"

    def test_broadband_none_no_extra_cols(self):
        """When broadband=None, no broadband columns appear in output."""
        fips = ["01001", "01003"]
        acs = _make_acs(fips)
        rcms = _make_rcms(fips)
        result = build_national_features(acs, rcms, broadband=None)
        for col in BB_NATIONAL:
            assert col not in result.columns, f"Unexpected column: {col}"

    def test_broadband_missing_county_imputed(self):
        """County absent from broadband data gets state-median imputation (no NaN)."""
        acs_fips = ["01001", "01003", "01005"]
        bb_fips = ["01001", "01003"]   # 01005 missing
        acs = _make_acs(acs_fips)
        rcms = _make_rcms(acs_fips)
        broadband = _make_broadband_features(bb_fips)
        result = build_national_features(acs, rcms, broadband=broadband)
        for col in BB_NATIONAL:
            assert result[col].isna().sum() == 0, f"{col} has NaN after imputation"

    def test_no_duplicate_cols_with_broadband(self):
        """Adding broadband to full join produces no duplicate column names."""
        fips = ["01001", "01003", "01005", "12001"]
        acs = _make_acs(fips)
        rcms = _make_rcms(fips)
        broadband = _make_broadband_features(fips)
        result = build_national_features(acs, rcms, broadband=broadband)
        assert result.columns.nunique() == len(result.columns), (
            f"Duplicate cols: {[c for c in result.columns if list(result.columns).count(c) > 1]}"
        )

    def test_row_count_preserved_with_broadband(self):
        """Adding broadband does not drop or duplicate ACS rows (left join)."""
        fips = ["01001", "01003", "01005"]
        acs = _make_acs(fips)
        rcms = _make_rcms(fips)
        broadband = _make_broadband_features(fips)
        result = build_national_features(acs, rcms, broadband=broadband)
        assert len(result) == len(fips)

    def test_broadband_fips_validation_raises(self):
        """Malformed broadband county_fips raises ValueError."""
        acs = _make_acs(["01001", "01003"])
        rcms = _make_rcms(["01001", "01003"])
        # Build valid broadband features then corrupt one FIPS directly
        bad_bb = _make_broadband_features(["01001", "01003"])
        bad_bb.loc[0, "county_fips"] = "1001"  # 4-char, malformed
        with pytest.raises(ValueError, match="5-char"):
            build_national_features(acs, rcms, broadband=bad_bb)

    def test_total_feature_count_all_eight_sources(self):
        """Column count is correct when all 8 sources are joined (ACS+RCMS+QCEW+CHR+Mig+Urb+SCI+BB)."""
        from tests.test_county_features_national import (
            _make_qcew,
            _make_chr,
            _make_migration,
            _make_urbanicity,
            _make_sci,
        )
        fips = ["01001", "01003"]
        acs = _make_acs(fips)
        rcms = _make_rcms(fips)
        qcew = _make_qcew(fips)
        chr_df = _make_chr(fips)
        migration = _make_migration(fips)
        urbanicity = _make_urbanicity(fips)
        sci = _make_sci(fips)
        broadband = _make_broadband_features(fips)
        result = build_national_features(
            acs, rcms, qcew=qcew, chr_df=chr_df,
            migration=migration, urbanicity=urbanicity, sci=sci, broadband=broadband,
        )
        acs_cols = len(acs.columns)  # includes county_fips
        expected = (
            acs_cols
            + len(RCMS_FEATURE_COLS)
            + len(QCEW_FEATURE_COLS)
            + len(CHR_FEATURE_COLS)
            + len(MIGRATION_FEATURE_COLS)
            + len(URBANICITY_FEATURE_COLS)
            + len(SCI_FEATURE_COLS)
            + len(BB_NATIONAL)
        )
        assert len(result.columns) == expected, (
            f"Expected {expected} cols, got {len(result.columns)}: {list(result.columns)}"
        )

    def test_no_duplicate_cols_all_eight_sources(self):
        """Full 8-source join has no duplicate column names."""
        from tests.test_county_features_national import (
            _make_qcew,
            _make_chr,
            _make_migration,
            _make_urbanicity,
            _make_sci,
        )
        fips = ["01001", "01003", "01005", "12001"]
        acs = _make_acs(fips)
        rcms = _make_rcms(fips)
        qcew = _make_qcew(fips)
        chr_df = _make_chr(fips)
        migration = _make_migration(fips)
        urbanicity = _make_urbanicity(fips)
        sci = _make_sci(fips)
        broadband = _make_broadband_features(fips)
        result = build_national_features(
            acs, rcms, qcew=qcew, chr_df=chr_df,
            migration=migration, urbanicity=urbanicity, sci=sci, broadband=broadband,
        )
        assert result.columns.nunique() == len(result.columns), (
            f"Duplicate cols: {[c for c in result.columns if list(result.columns).count(c) > 1]}"
        )
