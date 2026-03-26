"""Tests for build_county_features_national.py.

Verifies:
- ACS + RCMS join produces correct row count and columns
- FIPS format validated (5-char zero-padded strings)
- Missing RCMS counties imputed with state-level medians (fallback to national)
- Connecticut 2022 planning region FIPS (09xxx) handled via national median fallback
- All RCMS features bounded in reasonable ranges after imputation
- Merge does not lose ACS counties (left join preserves all)
- No spurious duplicate rows
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembly.build_county_features_national import (
    RCMS_FEATURE_COLS,
    build_national_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_acs(fips_list: list[str]) -> pd.DataFrame:
    """Minimal synthetic ACS county features DataFrame."""
    rng = np.random.default_rng(0)
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
    """Minimal synthetic RCMS county features DataFrame."""
    rng = np.random.default_rng(1)
    n = len(fips_list)
    fracs = rng.dirichlet(np.array([5.0, 1.5, 2.0, 1.0, 2.0]), size=n)
    return pd.DataFrame({
        "county_fips": fips_list,
        "state_abbr": ["AL"] * n,  # simplified for testing
        "evangelical_share": fracs[:, 0],
        "mainline_share": fracs[:, 1],
        "catholic_share": fracs[:, 2],
        "black_protestant_share": fracs[:, 3],
        "congregations_per_1000": rng.uniform(1.0, 10.0, size=n),
        "religious_adherence_rate": rng.uniform(200, 700, size=n),
    })


# ---------------------------------------------------------------------------
# Tests: basic join
# ---------------------------------------------------------------------------


class TestBuildNationalFeatures:
    def test_output_row_count_equals_acs(self):
        """Left join preserves all ACS counties."""
        acs_fips = ["01001", "01003", "01005", "12001", "12003"]
        rcms_fips = ["01001", "01003", "01005", "12001", "12003"]
        acs = _make_acs(acs_fips)
        rcms = _make_rcms(rcms_fips)
        result = build_national_features(acs, rcms)
        assert len(result) == len(acs_fips)

    def test_output_columns_include_acs_and_rcms(self):
        """Output must have both ACS and RCMS feature columns."""
        acs_fips = ["01001", "01003"]
        rcms_fips = ["01001", "01003"]
        acs = _make_acs(acs_fips)
        rcms = _make_rcms(rcms_fips)
        result = build_national_features(acs, rcms)
        for col in RCMS_FEATURE_COLS:
            assert col in result.columns, f"Missing RCMS column: {col}"
        assert "pct_white_nh" in result.columns
        assert "pct_bachelors_plus" in result.columns
        assert "county_fips" in result.columns

    def test_fips_preserved(self):
        """county_fips values from ACS are preserved unchanged."""
        acs_fips = ["01001", "01003", "12001"]
        rcms_fips = ["01001", "01003", "12001"]
        acs = _make_acs(acs_fips)
        rcms = _make_rcms(rcms_fips)
        result = build_national_features(acs, rcms)
        assert set(result["county_fips"].tolist()) == set(acs_fips)

    def test_no_duplicates(self):
        """No duplicate FIPS rows in output."""
        acs_fips = ["01001", "01003", "12001"]
        rcms_fips = ["01001", "01003", "12001"]
        acs = _make_acs(acs_fips)
        rcms = _make_rcms(rcms_fips)
        result = build_national_features(acs, rcms)
        assert result["county_fips"].nunique() == len(result)


# ---------------------------------------------------------------------------
# Tests: missing RCMS handling
# ---------------------------------------------------------------------------


class TestMissingRcmsImputation:
    def test_missing_rcms_county_imputed(self):
        """ACS county with no RCMS match gets imputed RCMS features (not NaN)."""
        acs_fips = ["01001", "01003", "01005"]  # 3 ACS counties
        rcms_fips = ["01001", "01003"]           # RCMS missing 01005
        acs = _make_acs(acs_fips)
        rcms = _make_rcms(rcms_fips)
        result = build_national_features(acs, rcms)
        for col in RCMS_FEATURE_COLS:
            assert result[col].isna().sum() == 0, f"{col} has NaN after imputation"

    def test_all_state_missing_uses_national_median(self):
        """Counties whose state has no RCMS data at all get national median fallback."""
        # AL counties have RCMS; CT new planning regions (09xxx) do not
        acs_fips = ["01001", "01003", "09110", "09120"]  # AL has RCMS; CT does not
        rcms_fips = ["01001", "01003"]                    # No CT entries
        acs = _make_acs(acs_fips)
        rcms = _make_rcms(rcms_fips)
        result = build_national_features(acs, rcms)
        for col in RCMS_FEATURE_COLS:
            assert result[col].isna().sum() == 0, f"{col} still NaN after national fallback"

    def test_imputed_value_equals_state_median(self):
        """Missing RCMS county gets the state-level median of the other counties."""
        # Two AL counties with known evangelical_share: 0.3 and 0.5 → median = 0.4
        acs_fips = ["01001", "01003", "01005"]
        rcms_fips = ["01001", "01003"]
        acs = _make_acs(acs_fips)

        rcms = pd.DataFrame({
            "county_fips": rcms_fips,
            "state_abbr": ["AL", "AL"],
            "evangelical_share": [0.3, 0.5],
            "mainline_share": [0.1, 0.2],
            "catholic_share": [0.2, 0.3],
            "black_protestant_share": [0.05, 0.10],
            "congregations_per_1000": [3.0, 5.0],
            "religious_adherence_rate": [400.0, 600.0],
        })

        result = build_national_features(acs, rcms)
        missing_row = result[result["county_fips"] == "01005"]
        assert len(missing_row) == 1
        # State median of [0.3, 0.5] = 0.4
        assert abs(missing_row["evangelical_share"].iloc[0] - 0.4) < 1e-6

    def test_rcms_features_nonnegative(self):
        """All RCMS feature values are non-negative after imputation."""
        acs_fips = ["01001", "01003", "01005", "12001"]
        rcms_fips = ["01001", "01003"]
        acs = _make_acs(acs_fips)
        rcms = _make_rcms(rcms_fips)
        result = build_national_features(acs, rcms)
        for col in RCMS_FEATURE_COLS:
            assert (result[col] >= 0).all(), f"{col} has negative values"


# ---------------------------------------------------------------------------
# Tests: FIPS validation
# ---------------------------------------------------------------------------


class TestFipsValidation:
    def test_bad_acs_fips_raises(self):
        """Malformed ACS county_fips raises ValueError."""
        bad_acs = _make_acs(["1001", "01003"])  # '1001' is 4 chars, not 5
        rcms = _make_rcms(["01001", "01003"])
        with pytest.raises(ValueError, match="5-char"):
            build_national_features(bad_acs, rcms)

    def test_bad_rcms_fips_raises(self):
        """Malformed RCMS county_fips raises ValueError."""
        acs = _make_acs(["01001", "01003"])
        bad_rcms = _make_rcms(["1001", "01003"])  # '1001' is 4 chars
        with pytest.raises(ValueError, match="5-char"):
            build_national_features(acs, bad_rcms)
