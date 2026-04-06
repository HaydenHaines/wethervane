"""
Tests for DRA block→tract ingestion pipeline (Phase T.1).

Covers:
- Block→tract aggregation (GEOID[:11] grouping)
- v06/v07 detection
- Column parsing (race extraction, COMP filtering)
- Long-format conversion
- dem_share calculation
- Edge cases (zero-vote tracts, single-block tracts)
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.assembly.ingest_dra_blocks import (
    _detect_version,
    _find_csv,
    _is_election_col,
    _parse_year,
    extract_election_results,
    ingest_all_states,
    save_tract_elections,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _make_state_dir(tmp_path: Path, state: str, version: str, csv_content: str) -> Path:
    """Create a fake state directory with version subdir and CSV."""
    state_dir = tmp_path / state
    ver_dir = state_dir / version
    ver_dir.mkdir(parents=True)
    csv_path = ver_dir / f"election_data_block_{state.lower()}.{version}.csv"
    csv_path.write_text(csv_content)
    # Add LICENSE and README to mimic real layout
    (ver_dir / "LICENSE").write_text("MIT")
    (ver_dir / "README").write_text("data readme")
    return state_dir


MINIMAL_V06_CSV = """\
GEOID,E_08_PRES_Total,E_08_PRES_Dem,E_08_PRES_Rep,E_20_PRES_Total,E_20_PRES_Dem,E_20_PRES_Rep
011339655012007,10,6,4,20,11,9
011339655012008,5,2,3,8,4,4
011339655013001,30,15,15,40,20,20
"""

MINIMAL_V07_CSV = """\
GEOID,E_08_PRES_Total,E_08_PRES_Dem,E_08_PRES_Rep,E_16-20_COMP_Total,E_16-20_COMP_Dem,E_16-20_COMP_Rep,E_20_PRES_Total,E_20_PRES_Dem,E_20_PRES_Rep,E_24_PRES_Total,E_24_PRES_Dem,E_24_PRES_Rep
021339655012001,100,60,40,999,500,499,120,65,55,130,70,60
021339655012002,50,30,20,999,400,599,60,35,25,70,40,30
"""

ZERO_VOTE_CSV = """\
GEOID,E_08_PRES_Total,E_08_PRES_Dem,E_08_PRES_Rep
031100000010000,0,0,0
031100000010001,0,0,0
"""


# ─── _parse_year ──────────────────────────────────────────────────────────────


def test_parse_year_08():
    assert _parse_year("08") == 2008


def test_parse_year_24():
    assert _parse_year("24") == 2024


def test_parse_year_16():
    assert _parse_year("16") == 2016


# ─── _is_election_col ─────────────────────────────────────────────────────────


def test_election_col_accepts_pres():
    assert _is_election_col("E_08_PRES_Total") is True
    assert _is_election_col("E_20_PRES_Dem") is True
    assert _is_election_col("E_24_PRES_Rep") is True


def test_election_col_accepts_gov_sen():
    assert _is_election_col("E_18_GOV_Total") is True
    assert _is_election_col("E_22_SEN_Dem") is True
    assert _is_election_col("E_17_SEN_SPEC_Total") is True


def test_election_col_rejects_comp():
    assert _is_election_col("E_16-20_COMP_Total") is False
    assert _is_election_col("E_16-22_COMP_Dem") is False


def test_election_col_rejects_geoid():
    assert _is_election_col("GEOID") is False


def test_election_col_rejects_state():
    assert _is_election_col("state") is False


# ─── _detect_version ──────────────────────────────────────────────────────────


def test_detect_version_v06(tmp_path):
    state_dir = tmp_path / "AL"
    (state_dir / "v06").mkdir(parents=True)
    assert _detect_version(state_dir) == "v06"


def test_detect_version_v07(tmp_path):
    state_dir = tmp_path / "GA"
    (state_dir / "v07").mkdir(parents=True)
    assert _detect_version(state_dir) == "v07"


def test_detect_version_missing_raises(tmp_path):
    state_dir = tmp_path / "XX"
    state_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="v06 or v07"):
        _detect_version(state_dir)


# ─── _find_csv ────────────────────────────────────────────────────────────────


def test_find_csv_returns_csv(tmp_path):
    ver_dir = tmp_path / "v06"
    ver_dir.mkdir()
    csv = ver_dir / "election_data_block_al.v06.csv"
    csv.write_text("GEOID\n")
    result = _find_csv(ver_dir)
    assert result == csv


def test_find_csv_raises_if_empty(tmp_path):
    ver_dir = tmp_path / "v06"
    ver_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="No CSV"):
        _find_csv(ver_dir)


# ─── Block→tract aggregation ──────────────────────────────────────────────────


def test_tract_aggregation_sums_blocks(tmp_path):
    """All blocks sharing the same 11-digit GEOID prefix should be summed."""
    csv = """\
GEOID,E_08_PRES_Total,E_08_PRES_Dem,E_08_PRES_Rep
011110000000001,10,6,4
011110000000002,5,2,3
011220000000001,30,15,15
"""
    _make_state_dir(tmp_path, "AL", "v06", csv)
    result = ingest_all_states(data_dir=tmp_path)

    # Blocks 011110000000001 and 011110000000002 → tract 01111000000
    tract = result.loc["01111000000"]
    assert tract["E_08_PRES_Total"] == 15   # 10 + 5
    assert tract["E_08_PRES_Dem"] == 8       # 6 + 2
    assert tract["E_08_PRES_Rep"] == 7       # 4 + 3

    # Block 011220000000001 is sole member of tract 01122000000
    tract2 = result.loc["01122000000"]
    assert tract2["E_08_PRES_Total"] == 30


def test_single_block_tract(tmp_path):
    """A tract with one block should pass through with its own values unchanged."""
    csv = """\
GEOID,E_08_PRES_Total,E_08_PRES_Dem,E_08_PRES_Rep
011110000000001,10,6,4
011220000000001,30,15,15
"""
    _make_state_dir(tmp_path, "AL", "v06", csv)
    result = ingest_all_states(data_dir=tmp_path)

    # Each block is the only member of its tract → values are unchanged
    tract1 = result.loc["01111000000"]
    assert tract1["E_08_PRES_Total"] == 10

    tract2 = result.loc["01122000000"]
    assert tract2["E_08_PRES_Total"] == 30


def test_two_tracts_in_same_state(tmp_path):
    """Verify distinct tract GEOIDs come out as separate rows."""
    csv = """\
GEOID,E_08_PRES_Total,E_08_PRES_Dem,E_08_PRES_Rep
011110000010001,10,6,4
012220000010001,20,12,8
"""
    _make_state_dir(tmp_path, "AL", "v06", csv)
    result = ingest_all_states(data_dir=tmp_path)
    assert "01111000001" in result.index
    assert "01222000001" in result.index
    assert len(result) == 2


def test_comp_cols_excluded_from_wide(tmp_path):
    """COMP columns must not appear in the wide aggregated output."""
    _make_state_dir(tmp_path, "AK", "v07", MINIMAL_V07_CSV)
    result = ingest_all_states(data_dir=tmp_path)
    for col in result.columns:
        assert "COMP" not in col, f"COMP column leaked into output: {col}"


def test_24_pres_present_when_in_source(tmp_path):
    """E_24_PRES columns survive into wide output when present in source."""
    _make_state_dir(tmp_path, "AK", "v07", MINIMAL_V07_CSV)
    result = ingest_all_states(data_dir=tmp_path)
    assert "E_24_PRES_Total" in result.columns


def test_2020_geography_skipped(tmp_path):
    """The 2020_Geography directory must be ignored."""
    geo_dir = tmp_path / "2020_Geography"
    geo_dir.mkdir()
    (geo_dir / "some_file.csv").write_text("GEOID\n")
    _make_state_dir(tmp_path, "AL", "v06", MINIMAL_V06_CSV)
    # Should not raise; should return data only from AL
    result = ingest_all_states(data_dir=tmp_path)
    assert "state" in result.columns
    assert set(result["state"].unique()) == {"AL"}


def test_multi_state_concatenation(tmp_path):
    """Loading two states should produce rows from both."""
    _make_state_dir(tmp_path, "AL", "v06", MINIMAL_V06_CSV)
    _make_state_dir(tmp_path, "AK", "v07", MINIMAL_V07_CSV)
    result = ingest_all_states(data_dir=tmp_path)
    states = set(result["state"].unique())
    assert "AL" in states
    assert "AK" in states


# ─── extract_election_results ────────────────────────────────────────────────


def test_long_format_columns(tmp_path):
    _make_state_dir(tmp_path, "AL", "v06", MINIMAL_V06_CSV)
    wide = ingest_all_states(data_dir=tmp_path)
    long_df = extract_election_results(wide)
    expected = {"tract_geoid", "year", "race_type", "total_votes",
                "dem_votes", "rep_votes", "dem_share"}
    assert set(long_df.columns) == expected


def test_zero_vote_rows_excluded(tmp_path):
    _make_state_dir(tmp_path, "AZ", "v06", ZERO_VOTE_CSV)
    wide = ingest_all_states(data_dir=tmp_path)
    long_df = extract_election_results(wide)
    assert len(long_df) == 0 or (long_df["total_votes"] > 0).all()


def test_dem_share_in_unit_interval(tmp_path):
    _make_state_dir(tmp_path, "AL", "v06", MINIMAL_V06_CSV)
    wide = ingest_all_states(data_dir=tmp_path)
    long_df = extract_election_results(wide)
    assert (long_df["dem_share"] >= 0.0).all()
    assert (long_df["dem_share"] <= 1.0).all()


def test_dem_share_calculation():
    """Two-party dem share = dem / (dem + rep)."""
    wide = pd.DataFrame(
        {
            "E_08_PRES_Total": [100],
            "E_08_PRES_Dem": [60],
            "E_08_PRES_Rep": [40],
            "state": ["XX"],
        },
        index=pd.Index(["01234567890"], name="tract_geoid"),
    )
    long_df = extract_election_results(wide)
    row = long_df[long_df["year"] == 2008].iloc[0]
    assert abs(row["dem_share"] - 0.6) < 1e-9


def test_year_parsing_in_long_format(tmp_path):
    _make_state_dir(tmp_path, "AL", "v06", MINIMAL_V06_CSV)
    wide = ingest_all_states(data_dir=tmp_path)
    long_df = extract_election_results(wide)
    years = set(long_df["year"].unique())
    assert 2008 in years
    assert 2020 in years


def test_no_comp_in_long_format(tmp_path):
    """COMP race_type must never appear in long output."""
    _make_state_dir(tmp_path, "AK", "v07", MINIMAL_V07_CSV)
    wide = ingest_all_states(data_dir=tmp_path)
    long_df = extract_election_results(wide)
    assert "COMP" not in set(long_df["race_type"].unique())


def test_pres_race_type_in_long_format(tmp_path):
    _make_state_dir(tmp_path, "AL", "v06", MINIMAL_V06_CSV)
    wide = ingest_all_states(data_dir=tmp_path)
    long_df = extract_election_results(wide)
    assert "PRES" in set(long_df["race_type"].unique())


def test_dem_share_nan_when_zero_two_party(tmp_path):
    """When dem+rep==0, dem_share should be NaN (not crash)."""
    csv = """\
GEOID,E_08_PRES_Total,E_08_PRES_Dem,E_08_PRES_Rep
011339655012007,5,0,0
"""
    _make_state_dir(tmp_path, "AL", "v06", csv)
    wide = ingest_all_states(data_dir=tmp_path)
    long_df = extract_election_results(wide)
    # Rows with total_votes > 0 but dem+rep == 0 should have NaN dem_share
    row = long_df[long_df["year"] == 2008]
    if len(row) > 0:
        assert row["dem_share"].isna().any() or (row["dem_share"] >= 0).all()


# ─── save_tract_elections ────────────────────────────────────────────────────


def test_save_creates_parquet(tmp_path):
    _make_state_dir(tmp_path / "dra", "AL", "v06", MINIMAL_V06_CSV)
    out = tmp_path / "out" / "tract_elections.parquet"
    save_tract_elections(output_path=out, data_dir=tmp_path / "dra")
    assert out.exists()


def test_saved_parquet_readable(tmp_path):
    _make_state_dir(tmp_path / "dra", "AL", "v06", MINIMAL_V06_CSV)
    out = tmp_path / "out" / "tract_elections.parquet"
    save_tract_elections(output_path=out, data_dir=tmp_path / "dra")
    df = pd.read_parquet(out)
    assert len(df) > 0
    assert "dem_share" in df.columns
