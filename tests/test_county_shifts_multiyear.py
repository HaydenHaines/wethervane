"""Tests for multi-year county shift vector builder."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from src.assembly.build_county_shifts_multiyear import (
    compute_pres_shift,
    compute_gov_shift,
    build_multiyear_shifts,
    TRAINING_SHIFT_COLS,
    HOLDOUT_SHIFT_COLS,
    AL_FIPS_PREFIX,
)


@pytest.fixture
def early_pres():
    return pd.DataFrame({
        "county_fips": ["12001", "13001", "01001"],
        "pres_dem_share_2016": [0.60, 0.55, 0.35],
        "pres_total_2016": [100000, 80000, 40000],
    })


@pytest.fixture
def late_pres():
    return pd.DataFrame({
        "county_fips": ["12001", "13001", "01001"],
        "pres_dem_share_2020": [0.65, 0.52, 0.33],
        "pres_total_2020": [110000, 82000, 41000],
    })


@pytest.fixture
def early_gov():
    return pd.DataFrame({
        "county_fips": ["12001", "13001", "01001"],
        "gov_dem_share_2014": [0.50, 0.45, 0.30],
        "gov_total_2014": [90000, 70000, 35000],
    })


@pytest.fixture
def late_gov():
    return pd.DataFrame({
        "county_fips": ["12001", "13001", "01001"],
        "gov_dem_share_2018": [0.49, 0.50, 0.37],
        "gov_total_2018": [95000, 72000, 36000],
    })


def test_pres_d_shift_math(early_pres, late_pres):
    result = compute_pres_shift(early_pres, late_pres, "16", "20")
    fl_row = result[result["county_fips"] == "12001"].iloc[0]
    assert abs(fl_row["pres_d_shift_16_20"] - 0.05) < 1e-6


def test_pres_r_shift_is_negative_d(early_pres, late_pres):
    result = compute_pres_shift(early_pres, late_pres, "16", "20")
    fl_row = result[result["county_fips"] == "12001"].iloc[0]
    assert abs(fl_row["pres_r_shift_16_20"] + fl_row["pres_d_shift_16_20"]) < 1e-10


def test_pres_turnout_shift(early_pres, late_pres):
    result = compute_pres_shift(early_pres, late_pres, "16", "20")
    fl_row = result[result["county_fips"] == "12001"].iloc[0]
    expected = (110000 - 100000) / 100000
    assert abs(fl_row["pres_turnout_shift_16_20"] - expected) < 1e-9


def test_gov_shift_math(early_gov, late_gov):
    """Gov shift math should work like pres shift."""
    result = compute_gov_shift(early_gov, late_gov, "14", "18")
    fl_row = result[result["county_fips"] == "12001"].iloc[0]
    assert abs(fl_row["gov_d_shift_14_18"] - (-0.01)) < 1e-6


def test_output_column_count():
    assert len(TRAINING_SHIFT_COLS) == 30
    assert len(HOLDOUT_SHIFT_COLS) == 3


def test_build_multiyear_spine(early_pres, late_pres, early_gov, late_gov, tmp_path):
    """build_multiyear_shifts returns all counties on the spine."""
    spine = pd.DataFrame({"county_fips": ["12001", "13001", "01001"]})
    pres_pairs = [("16", "20", early_pres, late_pres)]
    gov_pairs = [("14", "18", early_gov, late_gov)]
    result = build_multiyear_shifts(spine, pres_pairs, gov_pairs)
    assert len(result) == 3
    assert "county_fips" in result.columns
    all_shift_cols = TRAINING_SHIFT_COLS + HOLDOUT_SHIFT_COLS
    for col in all_shift_cols:
        assert col in result.columns, f"Missing column: {col}"
