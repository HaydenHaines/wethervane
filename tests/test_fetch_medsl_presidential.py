"""Tests for MEDSL county presidential fetcher.

Uses synthetic DataFrames to verify filtering, two-party share,
per-year output shape, and county_fips zero-padding.
"""
from __future__ import annotations
import pandas as pd
import pytest
from src.assembly.fetch_medsl_county_presidential import (
    filter_presidential_rows,
    aggregate_county_year,
    STATES,
)


@pytest.fixture
def raw_df():
    return pd.DataFrame({
        "year":            [2020, 2020, 2020, 2020, 2020],
        "state_po":        ["FL",  "FL",  "FL",  "TX",  "FL"],
        "county_fips":     ["12001","12001","12001","48001","12003"],
        "office":          ["PRESIDENT","PRESIDENT","PRESIDENT","PRESIDENT","PRESIDENT"],
        "party_simplified":["DEMOCRAT","REPUBLICAN","LIBERTARIAN","DEMOCRAT","DEMOCRAT"],
        "candidatevotes":  [100, 80, 5, 200, 50],
        "totalvotes":      [185, 185, 185, 200, 50],
    })


def test_filter_drops_non_presidential(raw_df):
    raw_df.loc[0, "office"] = "SENATE"
    result = filter_presidential_rows(raw_df)
    assert all(result["office"].str.upper().str.contains("PRESIDENT"))


def test_filter_drops_third_party(raw_df):
    result = filter_presidential_rows(raw_df)
    assert set(result["party_simplified"]) == {"DEMOCRAT", "REPUBLICAN"}


def test_filter_drops_other_states(raw_df):
    result = filter_presidential_rows(raw_df)
    assert set(result["state_po"]).issubset(set(STATES.values()))


def test_aggregate_county_year_dem_share(raw_df):
    filtered = filter_presidential_rows(raw_df)
    result = aggregate_county_year(filtered, 2020)
    fl_row = result[result["county_fips"] == "12001"].iloc[0]
    # dem_share = 100 / (100+80)
    assert abs(fl_row["pres_dem_share_2020"] - 100/180) < 1e-6


def test_aggregate_county_year_columns(raw_df):
    filtered = filter_presidential_rows(raw_df)
    result = aggregate_county_year(filtered, 2020)
    expected = {"county_fips","state_abbr",
                "pres_dem_2020","pres_rep_2020",
                "pres_total_2020","pres_dem_share_2020"}
    assert set(result.columns) == expected


def test_county_fips_zero_padded(raw_df):
    # Simulate numeric county_fips
    raw_df["county_fips"] = raw_df["county_fips"].astype(int)
    filtered = filter_presidential_rows(raw_df)
    result = aggregate_county_year(filtered, 2020)
    assert all(result["county_fips"].str.len() == 5)
