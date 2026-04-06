"""Tests for RCMS denomination feature pipeline.

Covers:
- CSV parsing and FIPS zero-padding
- Rate computation from adherent counts and pre-computed rates
- Missing data / NaN handling
- Hindu + Sikh combination logic
- Output schema validation
- Integration with build_rcms_denomination_features.py
- Edge cases (zero population, all-NaN inputs, single-county)
"""

from __future__ import annotations

import io
import numpy as np
import pandas as pd
import pytest

from src.assembly.fetch_rcms_denominations import (
    EXPECTED_HINDU_ADH_PER_CNG,
    _compute_hindu_sikh_rate,
    _compute_jewish_rate,
    _compute_lds_rate,
    _compute_muslim_rate,
    _compute_rate_from_adh_pop,
    _pad_fips,
    parse_denominations,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DENOM_COLS = ["lds_rate", "muslim_rate", "jewish_rate", "hindu_sikh_rate"]

REQUIRED_RAW_COLS = [
    "FIPS",
    "POP2020",
    "LDSADH_2020",
    "LDSRATE_2020",
    "MSLMADH_2020",
    "MSLMRATE_2020",
    "JWADH_2020",
    "JWRATE_2020",
    "HINTADH_2020",
    "HINYMADH_2020",
    "HINTCNG_2020",
    "SIKHCNG_2020",
]


def _make_raw_df(**overrides) -> pd.DataFrame:
    """Create a minimal single-row raw DataFrame for testing.

    Defaults provide a county with known values for all fields.
    Pass column=value kwargs to override specific fields.
    """
    defaults = {
        "FIPS": 1001,
        "POP2020": 10000,
        "LDSADH_2020": 500.0,
        "LDSRATE_2020": 50.0,
        "MSLMADH_2020": 200.0,
        "MSLMRATE_2020": 20.0,
        "JWADH_2020": 100.0,
        "JWRATE_2020": 10.0,
        "HINTADH_2020": 1000.0,
        "HINYMADH_2020": 500.0,
        "HINTCNG_2020": 1.0,
        "SIKHCNG_2020": 2.0,
    }
    defaults.update(overrides)
    return pd.DataFrame([defaults])


# ---------------------------------------------------------------------------
# FIPS padding
# ---------------------------------------------------------------------------


def test_fips_padding_strips_leading_zeros_issue():
    """FIPS 1001 (Autauga, AL) should become '01001', not '1001'."""
    s = pd.Series([1001, 48001, 6037])
    result = _pad_fips(s)
    assert result.tolist() == ["01001", "48001", "06037"]


def test_fips_padding_five_digit_fips_unchanged():
    """Already-5-digit integer FIPS (e.g. 48001) should be padded to string."""
    s = pd.Series([48001])
    result = _pad_fips(s)
    assert result.iloc[0] == "48001"
    assert len(result.iloc[0]) == 5


def test_fips_padding_output_is_string():
    """FIPS output should be string dtype."""
    s = pd.Series([1001])
    result = _pad_fips(s)
    assert result.dtype == object  # string series in pandas = object dtype


# ---------------------------------------------------------------------------
# Rate computation from adherents / population
# ---------------------------------------------------------------------------


def test_rate_from_adh_pop_basic():
    """1000 adherents in population of 10000 → rate of 100.0 per 1000."""
    adh = pd.Series([1000.0])
    pop = pd.Series([10000])
    rate = _compute_rate_from_adh_pop(adh, pop)
    assert abs(rate.iloc[0] - 100.0) < 1e-9


def test_rate_from_adh_pop_zero_population_returns_nan():
    """Division by zero population should return NaN, not infinity."""
    adh = pd.Series([100.0])
    pop = pd.Series([0])
    rate = _compute_rate_from_adh_pop(adh, pop)
    assert np.isnan(rate.iloc[0])


def test_rate_from_adh_pop_nan_adherents_returns_nan():
    """NaN adherents with valid population should propagate NaN."""
    adh = pd.Series([np.nan])
    pop = pd.Series([10000])
    rate = _compute_rate_from_adh_pop(adh, pop)
    assert np.isnan(rate.iloc[0])


# ---------------------------------------------------------------------------
# LDS rate computation
# ---------------------------------------------------------------------------


def test_lds_rate_uses_precomputed_rate_column():
    """When LDSRATE_2020 is available, it should be used directly."""
    df = _make_raw_df(LDSRATE_2020=42.5)
    rate = _compute_lds_rate(df)
    assert abs(rate.iloc[0] - 42.5) < 1e-9


def test_lds_rate_fallback_from_adh_pop():
    """When LDSRATE_2020 is NaN but LDSADH_2020 is present, derive from ADH/POP."""
    df = _make_raw_df(LDSRATE_2020=np.nan, LDSADH_2020=500.0, POP2020=10000)
    rate = _compute_lds_rate(df)
    expected = 500.0 / 10000 * 1000.0
    assert abs(rate.iloc[0] - expected) < 1e-9


def test_lds_rate_all_nan_returns_nan():
    """When both LDSRATE and LDSADH are NaN, result should be NaN."""
    df = _make_raw_df(LDSRATE_2020=np.nan, LDSADH_2020=np.nan)
    rate = _compute_lds_rate(df)
    assert np.isnan(rate.iloc[0])


# ---------------------------------------------------------------------------
# Hindu + Sikh combination
# ---------------------------------------------------------------------------


def test_hindu_sikh_combines_both_sub_bodies():
    """Hindu rate should reflect HINTADH + HINYMADH summed."""
    df = _make_raw_df(
        HINTADH_2020=1000.0,
        HINYMADH_2020=500.0,
        HINTCNG_2020=1.0,
        SIKHCNG_2020=0.0,  # No Sikh congregations
        POP2020=10000,
    )
    rate = _compute_hindu_sikh_rate(df)
    # Only Hindu: (1000 + 500) / 10000 * 1000 = 150.0
    assert abs(rate.iloc[0] - 150.0) < 1e-6


def test_hindu_sikh_includes_sikh_proxy():
    """Sikh congregations should be converted to estimated adherents."""
    df = _make_raw_df(
        HINTADH_2020=np.nan,
        HINYMADH_2020=np.nan,
        HINTCNG_2020=np.nan,
        SIKHCNG_2020=1.0,   # 1 Sikh congregation
        POP2020=10000,
    )
    rate = _compute_hindu_sikh_rate(df)
    # Expected: EXPECTED_HINDU_ADH_PER_CNG * 1 / 10000 * 1000
    expected = EXPECTED_HINDU_ADH_PER_CNG * 1.0 / 10000 * 1000.0
    assert abs(rate.iloc[0] - expected) < 1.0  # Allow 1 pp tolerance for rounding


def test_hindu_sikh_all_nan_returns_nan():
    """No Hindu or Sikh data at all should produce NaN (later filled to 0)."""
    df = _make_raw_df(
        HINTADH_2020=np.nan,
        HINYMADH_2020=np.nan,
        HINTCNG_2020=np.nan,
        SIKHCNG_2020=np.nan,
    )
    rate = _compute_hindu_sikh_rate(df)
    assert np.isnan(rate.iloc[0])


# ---------------------------------------------------------------------------
# parse_denominations — full pipeline
# ---------------------------------------------------------------------------


def test_parse_denominations_output_schema():
    """Output should have county_fips and exactly the 4 denomination rate columns."""
    df = _make_raw_df()
    result = parse_denominations(df)
    assert list(result.columns) == ["county_fips"] + _DENOM_COLS


def test_parse_denominations_fips_zero_padded():
    """FIPS 1001 should become '01001' in output."""
    df = _make_raw_df(FIPS=1001)
    result = parse_denominations(df)
    assert result["county_fips"].iloc[0] == "01001"


def test_parse_denominations_no_nan_in_output():
    """All NaN values should be filled with 0.0 — output has no NaN."""
    df = _make_raw_df(
        LDSADH_2020=np.nan,
        LDSRATE_2020=np.nan,
        MSLMADH_2020=np.nan,
        MSLMRATE_2020=np.nan,
        JWADH_2020=np.nan,
        JWRATE_2020=np.nan,
        HINTADH_2020=np.nan,
        HINYMADH_2020=np.nan,
        HINTCNG_2020=np.nan,
        SIKHCNG_2020=np.nan,
    )
    result = parse_denominations(df)
    assert not result[_DENOM_COLS].isna().any().any()


def test_parse_denominations_absent_denomination_fills_zero():
    """County with no denomination data should have all rates = 0.0."""
    df = _make_raw_df(
        LDSADH_2020=np.nan,
        LDSRATE_2020=np.nan,
        MSLMADH_2020=np.nan,
        MSLMRATE_2020=np.nan,
        JWADH_2020=np.nan,
        JWRATE_2020=np.nan,
        HINTADH_2020=np.nan,
        HINYMADH_2020=np.nan,
        HINTCNG_2020=np.nan,
        SIKHCNG_2020=np.nan,
    )
    result = parse_denominations(df)
    for col in _DENOM_COLS:
        assert result[col].iloc[0] == 0.0, f"{col} should be 0.0 for absent denomination"


def test_parse_denominations_rates_are_nonnegative():
    """All output rates should be >= 0.0."""
    df = _make_raw_df()
    result = parse_denominations(df)
    for col in _DENOM_COLS:
        assert (result[col] >= 0.0).all(), f"{col} has negative values"


def test_parse_denominations_multi_county():
    """Test with multiple counties — output count matches input count."""
    rows = [
        {"FIPS": 1001, "POP2020": 50000, "LDSADH_2020": 200.0, "LDSRATE_2020": 4.0,
         "MSLMADH_2020": 100.0, "MSLMRATE_2020": 2.0, "JWADH_2020": 50.0, "JWRATE_2020": 1.0,
         "HINTADH_2020": 500.0, "HINYMADH_2020": 100.0, "HINTCNG_2020": 1.0, "SIKHCNG_2020": 1.0},
        {"FIPS": 6037, "POP2020": 10000000, "LDSADH_2020": 50000.0, "LDSRATE_2020": 5.0,
         "MSLMADH_2020": 200000.0, "MSLMRATE_2020": 20.0, "JWADH_2020": 500000.0, "JWRATE_2020": 50.0,
         "HINTADH_2020": np.nan, "HINYMADH_2020": np.nan, "HINTCNG_2020": np.nan, "SIKHCNG_2020": 10.0},
        {"FIPS": 48001, "POP2020": 1000, "LDSADH_2020": np.nan, "LDSRATE_2020": np.nan,
         "MSLMADH_2020": np.nan, "MSLMRATE_2020": np.nan, "JWADH_2020": np.nan, "JWRATE_2020": np.nan,
         "HINTADH_2020": np.nan, "HINYMADH_2020": np.nan, "HINTCNG_2020": np.nan, "SIKHCNG_2020": np.nan},
    ]
    df = pd.DataFrame(rows)
    result = parse_denominations(df)
    assert len(result) == 3
    # Third county (all NaN) should have all zeros
    assert result.iloc[2]["lds_rate"] == 0.0
    assert result.iloc[2]["muslim_rate"] == 0.0
    # LA county should have high jewish rate (~50/1000)
    assert result.iloc[1]["jewish_rate"] == pytest.approx(50.0, abs=0.1)


# ---------------------------------------------------------------------------
# Integration with build_rcms_denomination_features
# ---------------------------------------------------------------------------


def test_load_denomination_features_from_disk(tmp_path, monkeypatch):
    """load_denomination_features() should read parquet and validate schema."""
    from src.assembly.build_rcms_denomination_features import (
        DENOMINATION_FEATURE_COLS,
        load_denomination_features,
    )

    # Create a minimal valid parquet
    test_df = pd.DataFrame({
        "county_fips": ["01001", "06037"],
        "lds_rate": [5.0, 1.0],
        "muslim_rate": [0.5, 10.0],
        "jewish_rate": [0.2, 20.0],
        "hindu_sikh_rate": [0.1, 5.0],
    })
    parquet_path = tmp_path / "rcms_denominations.parquet"
    test_df.to_parquet(parquet_path, index=False)

    # Monkeypatch INPUT_PATH in the module
    import src.assembly.build_rcms_denomination_features as mod
    original = mod.INPUT_PATH
    mod.INPUT_PATH = parquet_path
    try:
        result = load_denomination_features()
        assert list(result.columns) == ["county_fips"] + DENOMINATION_FEATURE_COLS
        assert len(result) == 2
    finally:
        mod.INPUT_PATH = original
