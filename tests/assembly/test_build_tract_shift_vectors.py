"""
Tests for src/assembly/build_tract_shift_vectors.py

Covers:
  - Presidential shift calculation (correct formula, pair years)
  - Population filtering (< MIN_VOTES excluded)
  - Off-cycle shift calculation (GOV and SEN)
  - Senate type consolidation (SEN_SPEC, SEN_ROFF, SEN_SPECROFF → SEN)
  - State-centering correctness (state means ≈ 0 after centering)
  - NaN handling for states without a given race
  - Edge cases: single-year elections, zero-vote tracts, tracts in only one year
  - Full matrix assembly
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembly.build_tract_shift_vectors import (
    MIN_VOTES,
    SENATE_RACE_TYPES,
    _consolidate_senate,
    _find_consecutive_pairs,
    build_tract_shift_matrix,
    compute_offcycle_shifts,
    compute_presidential_shifts,
    year_label,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────


def _make_elections(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal tract elections DataFrame from a list of dicts."""
    df = pd.DataFrame(rows)
    for col in ["tract_geoid", "year", "race_type"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    if "total_votes" not in df.columns:
        df["total_votes"] = 1000.0
    if "dem_votes" not in df.columns:
        df["dem_votes"] = df["total_votes"] * 0.5
    if "rep_votes" not in df.columns:
        df["rep_votes"] = df["total_votes"] * 0.5
    if "dem_share" not in df.columns:
        df["dem_share"] = df["dem_votes"] / (df["dem_votes"] + df["rep_votes"])
    return df


def _two_state_pres_elections(
    early_year: int = 2016,
    late_year: int = 2020,
    state_a: str = "01",
    state_b: str = "12",
    n_tracts_a: int = 5,
    n_tracts_b: int = 5,
    early_dem_a: float = 0.40,
    late_dem_a: float = 0.45,
    early_dem_b: float = 0.55,
    late_dem_b: float = 0.50,
    total_votes: float = 1000.0,
) -> pd.DataFrame:
    """Helper: build two-state presidential election data for shift testing."""
    rows = []
    for state, n, early, late in [
        (state_a, n_tracts_a, early_dem_a, late_dem_a),
        (state_b, n_tracts_b, early_dem_b, late_dem_b),
    ]:
        for i in range(1, n + 1):
            geoid = f"{state}{str(i).zfill(9)}"
            rows.append({
                "tract_geoid": geoid,
                "year": early_year,
                "race_type": "PRES",
                "total_votes": total_votes,
                "dem_share": early,
            })
            rows.append({
                "tract_geoid": geoid,
                "year": late_year,
                "race_type": "PRES",
                "total_votes": total_votes,
                "dem_share": late,
            })
    df = pd.DataFrame(rows)
    df["dem_votes"] = df["dem_share"] * df["total_votes"]
    df["rep_votes"] = (1 - df["dem_share"]) * df["total_votes"]
    return df


# ── year_label ─────────────────────────────────────────────────────────────────


def test_year_label_basic():
    assert year_label(2018, 2022) == "18_22"


def test_year_label_early_2000s():
    assert year_label(2008, 2012) == "08_12"


def test_year_label_single_digit_mod():
    assert year_label(2000, 2004) == "00_04"


# ── _find_consecutive_pairs ────────────────────────────────────────────────────


def test_find_consecutive_pairs_basic():
    pairs = _find_consecutive_pairs([2018, 2022])
    assert pairs == [(2018, 2022)]


def test_find_consecutive_pairs_too_far():
    """10-year gap exceeds MAX_PAIR_GAP_YEARS; should not be paired."""
    pairs = _find_consecutive_pairs([2010, 2020])
    assert pairs == []


def test_find_consecutive_pairs_multiple():
    pairs = _find_consecutive_pairs([2014, 2018, 2022])
    assert (2014, 2018) in pairs
    assert (2018, 2022) in pairs
    # 2014→2022 is 8 years, exceeds limit
    assert (2014, 2022) not in pairs


def test_find_consecutive_pairs_unsorted_input():
    """Input order should not matter."""
    pairs = _find_consecutive_pairs([2022, 2018])
    assert pairs == [(2018, 2022)]


def test_find_consecutive_pairs_single_year():
    assert _find_consecutive_pairs([2018]) == []


# ── compute_presidential_shifts ────────────────────────────────────────────────


def test_presidential_shift_formula():
    """Shift = late_dem_share - early_dem_share."""
    elections = _two_state_pres_elections(
        early_year=2016, late_year=2020,
        early_dem_a=0.40, late_dem_a=0.45,
        n_tracts_a=3, n_tracts_b=0,
    )
    result = compute_presidential_shifts(elections)
    col = "pres_shift_16_20"
    assert col in result.columns
    shifts = result.dropna(subset=[col])[col]
    assert len(shifts) == 3
    np.testing.assert_allclose(shifts.values, 0.05, atol=1e-6)


def test_presidential_shift_returns_four_columns():
    """Output always has the 4 standard presidential shift columns."""
    elections = _two_state_pres_elections(early_year=2016, late_year=2020)
    result = compute_presidential_shifts(elections)
    for col in ["pres_shift_08_12", "pres_shift_12_16", "pres_shift_16_20", "pres_shift_20_24"]:
        assert col in result.columns, f"Missing column: {col}"


def test_presidential_shift_not_state_centered():
    """Presidential shifts must NOT be state-centered (cross-state signal preserved)."""
    elections = _two_state_pres_elections(
        early_year=2016, late_year=2020,
        early_dem_a=0.40, late_dem_a=0.45,  # state A shifts +0.05
        early_dem_b=0.55, late_dem_b=0.50,  # state B shifts -0.05
        n_tracts_a=5, n_tracts_b=5,
    )
    result = compute_presidential_shifts(elections)
    col = "pres_shift_16_20"
    shifts_a = result[result["tract_geoid"].str.startswith("01")][col].dropna()
    shifts_b = result[result["tract_geoid"].str.startswith("12")][col].dropna()
    # Raw: A should be ~+0.05, B should be ~-0.05 (not both centered to 0)
    np.testing.assert_allclose(shifts_a.values, 0.05, atol=1e-6)
    np.testing.assert_allclose(shifts_b.values, -0.05, atol=1e-6)


def test_presidential_shift_population_filter():
    """Tracts with < MIN_VOTES in either year are excluded (NaN, not 0)."""
    rows = [
        # Sufficient votes in both years
        {"tract_geoid": "01000000001", "year": 2016, "race_type": "PRES",
         "total_votes": MIN_VOTES + 1, "dem_share": 0.40},
        {"tract_geoid": "01000000001", "year": 2020, "race_type": "PRES",
         "total_votes": MIN_VOTES + 1, "dem_share": 0.50},
        # Below threshold in early year
        {"tract_geoid": "01000000002", "year": 2016, "race_type": "PRES",
         "total_votes": MIN_VOTES - 1, "dem_share": 0.40},
        {"tract_geoid": "01000000002", "year": 2020, "race_type": "PRES",
         "total_votes": MIN_VOTES + 1, "dem_share": 0.50},
        # Below threshold in late year
        {"tract_geoid": "01000000003", "year": 2016, "race_type": "PRES",
         "total_votes": MIN_VOTES + 1, "dem_share": 0.40},
        {"tract_geoid": "01000000003", "year": 2020, "race_type": "PRES",
         "total_votes": MIN_VOTES - 1, "dem_share": 0.50},
    ]
    elections = _make_elections(rows)
    result = compute_presidential_shifts(elections)
    col = "pres_shift_16_20"
    valid = result.dropna(subset=[col])
    assert len(valid) == 1
    assert valid.iloc[0]["tract_geoid"] == "01000000001"


def test_presidential_shift_missing_year_is_nan():
    """Tracts present in only one of the two years get NaN, not 0."""
    rows = [
        {"tract_geoid": "01000000001", "year": 2016, "race_type": "PRES",
         "total_votes": 1000, "dem_share": 0.45},
        # No 2020 row for this tract
    ]
    elections = _make_elections(rows)
    result = compute_presidential_shifts(elections)
    col = "pres_shift_16_20"
    row = result[result["tract_geoid"] == "01000000001"]
    # If tract_geoid doesn't appear in result or has NaN, that's correct.
    if len(row) > 0:
        assert pd.isna(row.iloc[0][col])


def test_presidential_shift_zero_vote_tract():
    """Zero-vote tracts (edge case) are excluded by population filter."""
    rows = [
        {"tract_geoid": "01000000001", "year": 2016, "race_type": "PRES",
         "total_votes": 0, "dem_share": 0.0},
        {"tract_geoid": "01000000001", "year": 2020, "race_type": "PRES",
         "total_votes": 0, "dem_share": 0.0},
    ]
    elections = _make_elections(rows)
    result = compute_presidential_shifts(elections)
    col = "pres_shift_16_20"
    row = result[result["tract_geoid"] == "01000000001"]
    if len(row) > 0:
        assert pd.isna(row.iloc[0][col])


# ── _consolidate_senate ────────────────────────────────────────────────────────


def test_consolidate_senate_relabels_spec_types():
    """SEN_SPEC, SEN_ROFF, SEN_SPECROFF should be treated as SEN."""
    rows = [
        {"tract_geoid": "13000000001", "year": 2021, "race_type": "SEN_SPEC",
         "total_votes": 2000, "dem_share": 0.55},
        {"tract_geoid": "13000000002", "year": 2021, "race_type": "SEN_SPEC",
         "total_votes": 1500, "dem_share": 0.45},
    ]
    elections = _make_elections(rows)
    result = _consolidate_senate(elections)
    assert set(result["race_type"].unique()) == {"SEN"}


def test_consolidate_senate_keeps_best_when_multiple_types_same_year():
    """When a state has both SEN and SEN_SPEC in the same year, keep the one
    with more aggregate votes."""
    state_fips = "01"
    # SEN_SPEC: smaller election (1000 votes each, 2 tracts → 2000 total)
    # SEN: bigger election (3000 votes each, 2 tracts → 6000 total)
    rows = [
        {"tract_geoid": f"{state_fips}000000001", "year": 2017,
         "race_type": "SEN_SPEC", "total_votes": 1000, "dem_share": 0.40},
        {"tract_geoid": f"{state_fips}000000002", "year": 2017,
         "race_type": "SEN_SPEC", "total_votes": 1000, "dem_share": 0.40},
        {"tract_geoid": f"{state_fips}000000001", "year": 2017,
         "race_type": "SEN", "total_votes": 3000, "dem_share": 0.55},
        {"tract_geoid": f"{state_fips}000000002", "year": 2017,
         "race_type": "SEN", "total_votes": 3000, "dem_share": 0.55},
    ]
    elections = _make_elections(rows)
    result = _consolidate_senate(elections)
    # All surviving rows should have dem_share=0.55 (from the bigger SEN race)
    assert len(result) == 2
    np.testing.assert_allclose(result["dem_share"].values, 0.55, atol=1e-6)


def test_consolidate_senate_single_type_unchanged():
    """When only one senate type exists, consolidation should preserve all rows."""
    rows = [
        {"tract_geoid": "12000000001", "year": 2018, "race_type": "SEN",
         "total_votes": 1500, "dem_share": 0.48},
        {"tract_geoid": "12000000002", "year": 2018, "race_type": "SEN",
         "total_votes": 2000, "dem_share": 0.52},
    ]
    elections = _make_elections(rows)
    result = _consolidate_senate(elections)
    assert len(result) == 2
    assert set(result["race_type"].unique()) == {"SEN"}


# ── compute_offcycle_shifts ────────────────────────────────────────────────────


def test_offcycle_shifts_state_centering():
    """State mean of each centered off-cycle shift column must be ≈ 0."""
    # Build GOV elections for two states with different absolute shifts
    # State 01: all tracts shift +0.10 (dem_share: 0.40 → 0.50)
    # State 12: all tracts shift -0.10 (dem_share: 0.60 → 0.50)
    rows = []
    for state, early_share, late_share in [("01", 0.40, 0.50), ("12", 0.60, 0.50)]:
        for i in range(1, 6):
            geoid = f"{state}{str(i).zfill(9)}"
            rows.append({"tract_geoid": geoid, "year": 2018, "race_type": "GOV",
                          "total_votes": 1000, "dem_share": early_share})
            rows.append({"tract_geoid": geoid, "year": 2022, "race_type": "GOV",
                          "total_votes": 1000, "dem_share": late_share})
    elections = _make_elections(rows)
    # Need to add dem_votes/rep_votes for _make_elections
    elections["dem_votes"] = elections["dem_share"] * elections["total_votes"]
    elections["rep_votes"] = (1 - elections["dem_share"]) * elections["total_votes"]

    result = compute_offcycle_shifts(elections)
    col = "gov_shift_18_22_centered"
    assert col in result.columns

    # Check state means are ≈ 0
    result["state_fips"] = result["tract_geoid"].str[:2]
    for state in ["01", "12"]:
        state_shifts = result[result["state_fips"] == state][col].dropna()
        assert len(state_shifts) > 0, f"No shifts for state {state}"
        np.testing.assert_allclose(
            state_shifts.mean(), 0.0, atol=1e-6,
            err_msg=f"State {state} mean not centered to 0"
        )


def test_offcycle_shifts_within_state_variation_preserved():
    """Within a state, tracts that shifted more should still rank higher after centering."""
    rows = []
    # State 01: 3 tracts with different shifts (+0.02, +0.05, +0.10)
    # All get centered but relative ordering must be preserved
    for i, (early, late) in enumerate([(0.40, 0.42), (0.40, 0.45), (0.40, 0.50)], 1):
        geoid = f"01{str(i).zfill(9)}"
        rows.append({"tract_geoid": geoid, "year": 2018, "race_type": "GOV",
                      "total_votes": 1000, "dem_share": early})
        rows.append({"tract_geoid": geoid, "year": 2022, "race_type": "GOV",
                      "total_votes": 1000, "dem_share": late})
    elections = _make_elections(rows)

    result = compute_offcycle_shifts(elections)
    col = "gov_shift_18_22_centered"
    shifts = (
        result[result["tract_geoid"].str.startswith("01")]
        .sort_values("tract_geoid")[col]
        .values
    )
    # Ordering: tract 1 < tract 2 < tract 3 (after centering, relative order preserved)
    assert shifts[0] < shifts[1] < shifts[2]


def test_offcycle_shifts_nan_for_missing_state():
    """Tracts in a state without a given race should get NaN, not 0."""
    # Build GOV 2018→2022 for state 01 only. State 12 should have NaN.
    rows = []
    for i in range(1, 4):
        geoid = f"01{str(i).zfill(9)}"
        rows.append({"tract_geoid": geoid, "year": 2018, "race_type": "GOV",
                      "total_votes": 1000, "dem_share": 0.45})
        rows.append({"tract_geoid": geoid, "year": 2022, "race_type": "GOV",
                      "total_votes": 1000, "dem_share": 0.50})
    # State 12 only has one year → no valid pair → should get NaN
    rows.append({"tract_geoid": "12000000001", "year": 2018, "race_type": "GOV",
                  "total_votes": 1000, "dem_share": 0.55})

    elections = _make_elections(rows)
    result = compute_offcycle_shifts(elections)
    col = "gov_shift_18_22_centered"

    if col in result.columns:
        state_12_rows = result[result["tract_geoid"].str.startswith("12")]
        if len(state_12_rows) > 0:
            assert state_12_rows[col].isna().all(), \
                "State 12 (no valid pair) should have NaN, not 0"


def test_offcycle_shifts_population_filter():
    """Tracts with < MIN_VOTES in either year are excluded from off-cycle shifts."""
    rows = [
        # Sufficient votes → should appear in output
        {"tract_geoid": "13000000001", "year": 2018, "race_type": "GOV",
         "total_votes": MIN_VOTES + 1, "dem_share": 0.45},
        {"tract_geoid": "13000000001", "year": 2022, "race_type": "GOV",
         "total_votes": MIN_VOTES + 1, "dem_share": 0.50},
        # Below threshold → should be NaN or absent
        {"tract_geoid": "13000000002", "year": 2018, "race_type": "GOV",
         "total_votes": MIN_VOTES - 1, "dem_share": 0.45},
        {"tract_geoid": "13000000002", "year": 2022, "race_type": "GOV",
         "total_votes": MIN_VOTES + 1, "dem_share": 0.50},
    ]
    elections = _make_elections(rows)
    result = compute_offcycle_shifts(elections)
    col = "gov_shift_18_22_centered"

    if col in result.columns:
        # Tract 2 (below threshold) should be NaN
        tract2 = result[result["tract_geoid"] == "13000000002"]
        if len(tract2) > 0:
            assert pd.isna(tract2.iloc[0][col]), \
                "Below-threshold tract should have NaN, not a real value"


def test_offcycle_shifts_senate_spec_treated_as_senate():
    """SEN_SPEC elections should be pairhable with SEN elections (both treated as senate)."""
    # State 13: SEN_SPEC in 2021, SEN in 2022 — gap=1 year → valid pair
    rows = []
    for i in range(1, 4):
        geoid = f"13{str(i).zfill(9)}"
        rows.append({"tract_geoid": geoid, "year": 2021, "race_type": "SEN_SPEC",
                      "total_votes": 1000, "dem_share": 0.48})
        rows.append({"tract_geoid": geoid, "year": 2022, "race_type": "SEN",
                      "total_votes": 1500, "dem_share": 0.52})
    elections = _make_elections(rows)
    result = compute_offcycle_shifts(elections)
    # Should produce sen_shift_21_22_centered (not empty)
    col = "sen_shift_21_22_centered"
    assert col in result.columns, f"Expected column {col} in {list(result.columns)}"
    valid_rows = result[result["tract_geoid"].str.startswith("13")][col].dropna()
    assert len(valid_rows) > 0, "Expected non-NaN shifts for GA (FIPS 13) sen_21_22"


# ── build_tract_shift_matrix ───────────────────────────────────────────────────


def test_build_tract_shift_matrix_has_four_pres_columns():
    """Matrix must always have the 4 standard presidential shift columns."""
    elections = _two_state_pres_elections(
        early_year=2016, late_year=2020,
        n_tracts_a=3, n_tracts_b=3,
    )
    result = build_tract_shift_matrix(elections, output_path=None)
    for col in ["pres_shift_08_12", "pres_shift_12_16", "pres_shift_16_20", "pres_shift_20_24"]:
        assert col in result.columns


def test_build_tract_shift_matrix_index_is_tract_geoid():
    """The index should be tract_geoid."""
    elections = _two_state_pres_elections(
        early_year=2016, late_year=2020,
        n_tracts_a=3, n_tracts_b=3,
    )
    result = build_tract_shift_matrix(elections, output_path=None)
    assert result.index.name == "tract_geoid"


def test_build_tract_shift_matrix_saves_parquet(tmp_path):
    """Output parquet is written to the specified path."""
    elections = _two_state_pres_elections(
        early_year=2016, late_year=2020,
        n_tracts_a=3, n_tracts_b=3,
    )
    out_path = tmp_path / "shifts" / "test_shifts.parquet"
    build_tract_shift_matrix(elections, output_path=out_path)
    assert out_path.exists()

    loaded = pd.read_parquet(out_path)
    assert "tract_geoid" in loaded.columns
    assert "pres_shift_16_20" in loaded.columns


def test_build_tract_shift_matrix_no_save_when_output_path_is_none():
    """When output_path=None, no file write occurs and result is still returned."""
    elections = _two_state_pres_elections(
        early_year=2016, late_year=2020,
        n_tracts_a=3, n_tracts_b=3,
    )
    result = build_tract_shift_matrix(elections, output_path=None)
    assert result is not None
    assert len(result) > 0
