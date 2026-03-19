"""Tests for FL voter registration fetching and feature computation.

Tests exercise:
1. fetch_voter_registration.py — URL construction, Excel parsing, FIPS mapping,
   header detection, party column identification, multi-year download logic
2. build_voter_registration_features.py — share computation, change computation,
   edge cases (single year, NaN propagation), output schema

These tests use synthetic data and mock HTTP responses so they run without
network access or the real voter registration parquet files. Tests verify:
  - URL construction uses correct FL DOS base and media ID
  - Header row detection handles both 2016-style and 2018+-style Excel layouts
  - Party columns correctly mapped from varying column names across years
  - FIPS mapping covers all 67 FL counties
  - Statewide total row is dropped (not treated as a county)
  - Non-county rows (metadata) are filtered out
  - Share features sum to approximately 1 per county
  - rep_minus_dem = rep_share - dem_share
  - Change features computed as latest - earliest across all snapshots
  - Single-snapshot counties get NaN change features
  - Empty input handled gracefully
  - Output schema has required columns and types
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import numpy as np
import openpyxl
import pandas as pd
import pytest

from src.assembly.fetch_voter_registration import (
    ELECTION_FILES,
    FL_COUNTY_FIPS,
    OUTPUT_COLUMNS,
    _find_header_row,
    _parse_registration_excel,
    build_url,
    download_excel,
    fetch_election_cycle,
)
from src.assembly.build_voter_registration_features import (
    CHANGE_FEATURE_COLS,
    SHARE_FEATURE_COLS,
    VOTER_REG_FEATURE_COLS,
    _safe_diff,
    build_county_features,
    compute_change_features,
    compute_share_features,
)


# ---------------------------------------------------------------------------
# Helper: build synthetic Excel files as bytes
# ---------------------------------------------------------------------------


def _make_excel_2024_style(counties: list[dict]) -> bytes:
    """Build a synthetic 2024-style voter registration Excel (header in row 8, 0-indexed).

    Column order mirrors the real 2024 FL DOS file:
      County Name | Republican Party of Florida | Florida Democratic Party |
      ... minor parties ... | No Party Affiliation | TOTAL | Precincts
    """
    wb = openpyxl.Workbook()
    ws = wb.active

    # Metadata rows (rows 0-7 in 0-indexed = Excel rows 1-8)
    ws.append(["FLORIDA DEPARTMENT OF STATE"])
    ws.append(["DIVISION OF ELECTIONS"])
    ws.append(["2024 General Election"])
    ws.append(["Active Registered Voters by Party"])
    ws.append(["Book Closing: October 7, 2024"])
    ws.append(["Statistics Generated: October 17, 2024"])
    ws.append([])  # blank row
    ws.append([])  # blank row (header is next)

    # Header row (Excel row 9 = 0-indexed row 8)
    ws.append([
        "County Name",
        "Republican Party of Florida",
        "Florida Democratic Party ",  # trailing space is real in the source data
        "American Solidarity Party of Florida",
        "No Party Affiliation",
        "TOTAL",
        "Precincts",
    ])

    # Data rows
    for c in counties:
        ws.append([
            c.get("county_name", "Unknown"),
            c.get("rep", 0),
            c.get("dem", 0),
            c.get("other", 0),
            c.get("npa", 0),
            c.get("total", 0),
            c.get("precincts", 10),
        ])

    # Statewide total row
    ws.append(["Total", 999999, 999999, 0, 100000, 2099998, None])

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _make_excel_2016_style(counties: list[dict]) -> bytes:
    """Build a synthetic 2016-style voter registration Excel (header in row 7, 0-indexed).

    2016 format has ElectionDate, BookClosing, JurisType prefix columns before CountyName.
    """
    wb = openpyxl.Workbook()
    ws = wb.active

    # Metadata rows (0-indexed rows 0-5)
    ws.append(["FLORIDA DEPARTMENT OF STATE"])
    ws.append(["DIVISION OF ELECTIONS"])
    ws.append(["2016 General Election"])
    ws.append(["Active Registered Voters by Party"])
    ws.append(["Bookclosing: October 18, 2016"])
    ws.append(["Generated: October 28, 2016"])
    ws.append([])  # blank row

    # Header row (0-indexed row 7)
    ws.append([
        "ElectionDate",
        "BookClosing",
        "JurisType",
        "CountyName",
        "Republican",
        "Democrat",
        "Constitution Party",
        "Green Party",
        "Libertarian Party",
        "No Party Affiliation",
        "TOTAL",
        "Precincts",
    ])

    # Data rows
    for c in counties:
        ws.append([
            "2016-11-08 00:00:00",
            "2016-10-18 00:00:00",
            "ALL",
            c.get("county_name", "Unknown"),
            c.get("rep", 0),
            c.get("dem", 0),
            c.get("constitution", 0),
            c.get("green", 0),
            c.get("libertarian", 0),
            c.get("npa", 0),
            c.get("total", 0),
            c.get("precincts", 10),
        ])

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sample_counties_2024() -> list[dict]:
    """Synthetic county data for 3 FL counties (2024 style)."""
    return [
        {
            "county_name": "Alachua",
            "rep": 46584,
            "dem": 75430,
            "npa": 36858,
            "other": 4489,
            "total": 163361,
        },
        {
            "county_name": "Baker",
            "rep": 12706,
            "dem": 3613,
            "npa": 2371,
            "other": 407,
            "total": 19097,
        },
        {
            "county_name": "Bay",
            "rep": 74349,
            "dem": 26313,
            "npa": 27755,
            "other": 4474,
            "total": 132891,
        },
    ]


@pytest.fixture(scope="module")
def excel_2024_bytes(sample_counties_2024) -> bytes:
    """Synthetic 2024-style Excel bytes."""
    return _make_excel_2024_style(sample_counties_2024)


@pytest.fixture(scope="module")
def excel_2016_bytes() -> bytes:
    """Synthetic 2016-style Excel bytes with 2 counties."""
    counties = [
        {
            "county_name": "Alachua",
            "rep": 50097,
            "dem": 85143,
            "npa": 38564,
            "constitution": 20,
            "green": 216,
            "libertarian": 725,
            "total": 177947,
        },
        {
            "county_name": "Baker",
            "rep": 7640,
            "dem": 5922,
            "npa": 1402,
            "constitution": 3,
            "green": 3,
            "libertarian": 25,
            "total": 15147,
        },
    ]
    return _make_excel_2016_style(counties)


@pytest.fixture(scope="module")
def parsed_2024(excel_2024_bytes) -> pd.DataFrame:
    """Parsed DataFrame from synthetic 2024-style Excel."""
    return _parse_registration_excel(excel_2024_bytes, 2024, "2024-10-07")


@pytest.fixture(scope="module")
def parsed_2016(excel_2016_bytes) -> pd.DataFrame:
    """Parsed DataFrame from synthetic 2016-style Excel."""
    return _parse_registration_excel(excel_2016_bytes, 2016, "2016-10-18")


@pytest.fixture(scope="module")
def multi_year_raw() -> pd.DataFrame:
    """Synthetic multi-year raw registration DataFrame (2 years, 3 counties)."""
    return pd.DataFrame({
        "county_fips": ["12001", "12003", "12005", "12001", "12003", "12005"],
        "county_name": ["Alachua", "Baker", "Bay", "Alachua", "Baker", "Bay"],
        "election_year": [2016, 2016, 2016, 2024, 2024, 2024],
        "book_closing_date": pd.to_datetime([
            "2016-10-18", "2016-10-18", "2016-10-18",
            "2024-10-07", "2024-10-07", "2024-10-07",
        ]),
        "rep": [50097, 7640, 60998, 46584, 12706, 74349],
        "dem": [85143, 5922, 31698, 75430, 3613, 26313],
        "npa": [38564, 1402, 22702, 36858, 2371, 27755],
        "other": [3943, 183, 2618, 4489, 407, 4474],
        "total": [177947, 15147, 118016, 163361, 19097, 132891],
    })


@pytest.fixture(scope="module")
def shares_df(multi_year_raw) -> pd.DataFrame:
    """Shares computed from multi-year raw data."""
    return compute_share_features(multi_year_raw)


@pytest.fixture(scope="module")
def county_features(multi_year_raw) -> pd.DataFrame:
    """Full county feature table from multi-year raw data."""
    return build_county_features(multi_year_raw)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Tests for module-level constants."""

    def test_fl_county_fips_has_67_counties(self):
        """FL_COUNTY_FIPS must map exactly 67 unique FL county FIPS codes."""
        unique_fips = set(FL_COUNTY_FIPS.values())
        assert len(unique_fips) == 67

    def test_all_fips_start_with_12(self):
        """All FL county FIPS codes must start with '12' (FL state FIPS prefix)."""
        for county, fips in FL_COUNTY_FIPS.items():
            assert fips.startswith("12"), f"{county} has non-FL FIPS: {fips}"

    def test_all_fips_are_5_digits(self):
        """All FIPS codes must be exactly 5-character digit strings."""
        for county, fips in FL_COUNTY_FIPS.items():
            assert len(fips) == 5 and fips.isdigit(), f"{county}: invalid FIPS '{fips}'"

    def test_miami_dade_fips(self):
        """Miami-Dade must have FIPS 12086."""
        assert FL_COUNTY_FIPS["Miami-Dade"] == "12086"

    def test_alachua_fips(self):
        """Alachua (alphabetically first FL county) must have FIPS 12001."""
        assert FL_COUNTY_FIPS["Alachua"] == "12001"

    def test_washington_fips(self):
        """Washington (alphabetically last FL county) must have FIPS 12133."""
        assert FL_COUNTY_FIPS["Washington"] == "12133"

    def test_output_columns_count(self):
        """OUTPUT_COLUMNS must have exactly 9 columns."""
        assert len(OUTPUT_COLUMNS) == 9

    def test_output_columns_include_required(self):
        """OUTPUT_COLUMNS must include county_fips, election_year, and all party cols."""
        required = {"county_fips", "county_name", "election_year", "book_closing_date",
                    "rep", "dem", "npa", "other", "total"}
        assert required == set(OUTPUT_COLUMNS)

    def test_election_files_has_5_cycles(self):
        """ELECTION_FILES must cover 5 election cycles (2016, 2018, 2020, 2022, 2024)."""
        years = [y for y, _, _ in ELECTION_FILES]
        assert set(years) == {2016, 2018, 2020, 2022, 2024}

    def test_voter_reg_feature_cols_count(self):
        """VOTER_REG_FEATURE_COLS must have exactly 9 features."""
        assert len(VOTER_REG_FEATURE_COLS) == 9

    def test_share_feature_cols_count(self):
        """SHARE_FEATURE_COLS must have exactly 5 features."""
        assert len(SHARE_FEATURE_COLS) == 5

    def test_change_feature_cols_count(self):
        """CHANGE_FEATURE_COLS must have exactly 4 features."""
        assert len(CHANGE_FEATURE_COLS) == 4


# ---------------------------------------------------------------------------
# URL construction
# ---------------------------------------------------------------------------


class TestBuildUrl:
    """Tests for build_url()."""

    def test_url_contains_media_id(self):
        """URL must contain the media ID."""
        url = build_url("708493")
        assert "708493" in url

    def test_url_starts_with_floridados(self):
        """URL must be rooted at files.floridados.gov."""
        url = build_url("708493")
        assert "floridados.gov" in url

    def test_different_ids_produce_different_urls(self):
        """Different media IDs must produce different URLs."""
        url1 = build_url("708493")
        url2 = build_url("697211")
        assert url1 != url2

    def test_url_contains_media_path(self):
        """URL must include /media/ path component."""
        url = build_url("12345")
        assert "/media/" in url


# ---------------------------------------------------------------------------
# download_excel — mocked HTTP
# ---------------------------------------------------------------------------


class TestDownloadExcel:
    """Tests for download_excel() with mocked HTTP responses."""

    def test_returns_bytes_on_success(self, excel_2024_bytes):
        """download_excel() must return bytes on HTTP 200."""
        mock_response = MagicMock()
        mock_response.content = excel_2024_bytes
        mock_response.raise_for_status.return_value = None

        with patch("src.assembly.fetch_voter_registration.requests.get",
                   return_value=mock_response):
            result = download_excel("708493")

        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_returns_none_on_http_error(self):
        """download_excel() must return None when the HTTP request fails."""
        import requests as _requests

        with patch(
            "src.assembly.fetch_voter_registration.requests.get",
            side_effect=_requests.RequestException("timeout"),
        ):
            result = download_excel("708493")

        assert result is None

    def test_returns_correct_content(self, excel_2024_bytes):
        """download_excel() must return the response content bytes unchanged."""
        mock_response = MagicMock()
        mock_response.content = excel_2024_bytes
        mock_response.raise_for_status.return_value = None

        with patch("src.assembly.fetch_voter_registration.requests.get",
                   return_value=mock_response):
            result = download_excel("708493")

        assert result == excel_2024_bytes


# ---------------------------------------------------------------------------
# _find_header_row — header detection
# ---------------------------------------------------------------------------


class TestFindHeaderRow:
    """Tests for _find_header_row()."""

    def test_finds_header_in_2024_style(self, excel_2024_bytes):
        """Should detect the header row in 2024-style Excel files."""
        df_raw = pd.read_excel(io.BytesIO(excel_2024_bytes), header=None)
        header_row = _find_header_row(df_raw)
        # The header row should contain "County"
        row_values = df_raw.iloc[header_row].tolist()
        row_str = " ".join(str(v) for v in row_values if pd.notna(v)).lower()
        assert "county" in row_str

    def test_finds_header_in_2016_style(self, excel_2016_bytes):
        """Should detect the header row in 2016-style Excel files (CountyName column)."""
        df_raw = pd.read_excel(io.BytesIO(excel_2016_bytes), header=None)
        header_row = _find_header_row(df_raw)
        row_values = df_raw.iloc[header_row].tolist()
        row_str = " ".join(str(v) for v in row_values if pd.notna(v)).lower()
        assert "county" in row_str

    def test_raises_on_no_county_header(self):
        """Should raise ValueError when no header row contains 'County'."""
        df_no_county = pd.DataFrame({
            0: ["Title Row", "Subtitle", "Data here"],
            1: [None, None, 12345],
        })
        with pytest.raises(ValueError, match="No header row"):
            _find_header_row(df_no_county)


# ---------------------------------------------------------------------------
# _parse_registration_excel — Excel parsing
# ---------------------------------------------------------------------------


class TestParseRegistrationExcel:
    """Tests for _parse_registration_excel()."""

    def test_2024_output_has_required_columns(self, parsed_2024):
        """Parsed 2024 output must have all OUTPUT_COLUMNS."""
        assert set(OUTPUT_COLUMNS).issubset(set(parsed_2024.columns))

    def test_2016_output_has_required_columns(self, parsed_2016):
        """Parsed 2016 output must have all OUTPUT_COLUMNS."""
        assert set(OUTPUT_COLUMNS).issubset(set(parsed_2016.columns))

    def test_2024_drops_total_row(self, parsed_2024):
        """The statewide 'Total' row must not appear in parsed output."""
        county_names = parsed_2024["county_name"].str.lower().tolist()
        assert "total" not in county_names

    def test_2016_drops_non_county_rows(self, parsed_2016):
        """Metadata and blank rows must be filtered out."""
        # Should only have real county rows
        for name in parsed_2016["county_name"]:
            assert len(name) > 1
            assert name.lower() not in {"nan", "", "total"}

    def test_2024_county_count(self, parsed_2024, sample_counties_2024):
        """Parsed 2024 must have exactly as many rows as the synthetic county list."""
        assert len(parsed_2024) == len(sample_counties_2024)

    def test_2024_fips_all_fl(self, parsed_2024):
        """All FIPS codes in parsed 2024 output must start with '12'."""
        assert parsed_2024["county_fips"].str.startswith("12").all()

    def test_2024_fips_are_5_digits(self, parsed_2024):
        """FIPS codes must be 5-character digit strings."""
        assert (parsed_2024["county_fips"].str.len() == 5).all()
        assert parsed_2024["county_fips"].str.isdigit().all()

    def test_2024_election_year_column(self, parsed_2024):
        """election_year column must equal 2024."""
        assert (parsed_2024["election_year"] == 2024).all()

    def test_2016_election_year_column(self, parsed_2016):
        """election_year column must equal 2016."""
        assert (parsed_2016["election_year"] == 2016).all()

    def test_book_closing_date_is_datetime(self, parsed_2024):
        """book_closing_date column must be datetime dtype."""
        assert pd.api.types.is_datetime64_any_dtype(parsed_2024["book_closing_date"])

    def test_rep_dem_npa_are_numeric(self, parsed_2024):
        """rep, dem, npa, total columns must be numeric."""
        for col in ["rep", "dem", "npa", "total"]:
            assert pd.api.types.is_numeric_dtype(parsed_2024[col]), \
                f"Column '{col}' is not numeric"

    def test_alachua_rep_count_2024(self, parsed_2024):
        """Alachua republican count must match synthetic input (46584)."""
        alachua = parsed_2024[parsed_2024["county_name"] == "Alachua"]
        assert len(alachua) == 1
        assert alachua.iloc[0]["rep"] == pytest.approx(46584)

    def test_alachua_dem_count_2024(self, parsed_2024):
        """Alachua democrat count must match synthetic input (75430)."""
        alachua = parsed_2024[parsed_2024["county_name"] == "Alachua"]
        assert alachua.iloc[0]["dem"] == pytest.approx(75430)

    def test_alachua_npa_count_2024(self, parsed_2024):
        """Alachua NPA count must match synthetic input (36858)."""
        alachua = parsed_2024[parsed_2024["county_name"] == "Alachua"]
        assert alachua.iloc[0]["npa"] == pytest.approx(36858)

    def test_other_nonnegative(self, parsed_2024):
        """'other' column must be non-negative (minor parties can't be negative)."""
        assert (parsed_2024["other"] >= 0).all()

    def test_empty_bytes_returns_empty(self):
        """Invalid/empty content must return empty DataFrame with correct schema."""
        result = _parse_registration_excel(b"not an excel file", 2024, "2024-10-07")
        assert len(result) == 0
        assert set(OUTPUT_COLUMNS).issubset(set(result.columns))

    def test_2016_alachua_fips(self, parsed_2016):
        """Alachua in 2016 file must map to FIPS 12001."""
        alachua = parsed_2016[parsed_2016["county_name"] == "Alachua"]
        assert len(alachua) == 1
        assert alachua.iloc[0]["county_fips"] == "12001"


# ---------------------------------------------------------------------------
# fetch_election_cycle — download + parse integration (mocked)
# ---------------------------------------------------------------------------


class TestFetchElectionCycle:
    """Tests for fetch_election_cycle() with mocked HTTP."""

    def test_returns_dataframe_on_success(self, excel_2024_bytes):
        """fetch_election_cycle() must return a non-empty DataFrame on success."""
        with patch("src.assembly.fetch_voter_registration.download_excel",
                   return_value=excel_2024_bytes):
            result = fetch_election_cycle(2024, "2024-10-07", "708493")

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_returns_empty_on_download_failure(self):
        """fetch_election_cycle() must return empty DataFrame when download fails."""
        with patch("src.assembly.fetch_voter_registration.download_excel",
                   return_value=None):
            result = fetch_election_cycle(2024, "2024-10-07", "708493")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_election_year_set_correctly(self, excel_2024_bytes):
        """election_year column must match the requested year."""
        with patch("src.assembly.fetch_voter_registration.download_excel",
                   return_value=excel_2024_bytes):
            result = fetch_election_cycle(2024, "2024-10-07", "708493")

        assert (result["election_year"] == 2024).all()


# ---------------------------------------------------------------------------
# FIPS mapping completeness
# ---------------------------------------------------------------------------


class TestFipsMappingCompleteness:
    """Tests for FL_COUNTY_FIPS coverage."""

    def test_no_duplicate_fips_values(self):
        """No two county entries should share the same FIPS code."""
        fips_values = list(FL_COUNTY_FIPS.values())
        unique_fips = set(fips_values)
        duplicates = [fips for fips in unique_fips if fips_values.count(fips) > 1]
        assert len(duplicates) == 0, f"Duplicate FIPS codes found: {duplicates}"

    def test_known_large_counties_present(self):
        """Major FL counties must have FIPS entries."""
        major_counties = [
            "Miami-Dade", "Broward", "Palm Beach", "Hillsborough",
            "Orange", "Pinellas", "Duval", "Lee", "Sarasota", "Collier",
        ]
        for county in major_counties:
            assert county in FL_COUNTY_FIPS, f"{county} missing from FL_COUNTY_FIPS"

    def test_fips_values_are_mostly_odd_increments(self):
        """FL county FIPS codes must mostly end in odd numbers.

        All FL counties use odd-numbered county codes except Miami-Dade, which
        has FIPS 12086 (even) — a historical exception due to the county being
        renamed from Dade County. Verify all except Miami-Dade are odd.
        """
        known_even_exceptions = {"Miami-Dade"}
        for county, fips in FL_COUNTY_FIPS.items():
            if county in known_even_exceptions:
                continue
            county_code = int(fips[2:])  # last 3 digits
            assert county_code % 2 == 1, \
                f"{county} FIPS {fips} has even county code {county_code}"

    def test_fips_range_valid(self):
        """FL county FIPS must be in range [12001, 12133]."""
        for county, fips in FL_COUNTY_FIPS.items():
            fips_int = int(fips)
            assert 12001 <= fips_int <= 12133, \
                f"{county} FIPS {fips_int} out of valid FL range"


# ---------------------------------------------------------------------------
# compute_share_features
# ---------------------------------------------------------------------------


class TestComputeShareFeatures:
    """Tests for compute_share_features()."""

    def test_output_has_share_columns(self, shares_df):
        """Output must contain all SHARE_FEATURE_COLS."""
        for col in SHARE_FEATURE_COLS:
            assert col in shares_df.columns, f"Missing column: {col}"

    def test_shares_between_zero_and_one(self, shares_df):
        """All share features must be in [0, 1]."""
        for col in ["dem_share", "rep_share", "npa_share", "other_share"]:
            valid = shares_df[col].dropna()
            assert (valid >= 0).all(), f"{col} has values below 0"
            assert (valid <= 1).all(), f"{col} has values above 1"

    def test_shares_sum_to_approximately_one(self, shares_df):
        """dem + rep + npa + other shares must sum to approximately 1 per row.

        Tolerance is 0.5% (0.995–1.005) to accommodate synthetic test data
        where party counts don't exactly sum to the total due to rounding in
        test fixtures. Real FL DOS data sums exactly; the 0.5% window handles
        both.
        """
        share_sum = (
            shares_df["dem_share"]
            + shares_df["rep_share"]
            + shares_df["npa_share"]
            + shares_df["other_share"]
        )
        valid_sum = share_sum.dropna()
        assert (valid_sum.between(0.995, 1.005)).all(), \
            f"Share sums out of [0.995, 1.005]: {valid_sum[~valid_sum.between(0.995, 1.005)].tolist()}"

    def test_rep_minus_dem_correctness(self, shares_df):
        """rep_minus_dem must equal rep_share - dem_share."""
        diff = shares_df["rep_share"] - shares_df["dem_share"]
        pd.testing.assert_series_equal(
            shares_df["rep_minus_dem"].reset_index(drop=True),
            diff.reset_index(drop=True),
            check_names=False,
            rtol=1e-5,
        )

    def test_zero_total_produces_nan(self):
        """Counties with total == 0 must produce NaN shares."""
        df_zero = pd.DataFrame({
            "county_fips": ["12001"],
            "county_name": ["Alachua"],
            "election_year": [2024],
            "book_closing_date": [pd.Timestamp("2024-10-07")],
            "rep": [0], "dem": [0], "npa": [0], "other": [0], "total": [0],
        })
        result = compute_share_features(df_zero)
        assert pd.isna(result.iloc[0]["dem_share"])
        assert pd.isna(result.iloc[0]["rep_share"])

    def test_input_unchanged(self, multi_year_raw):
        """compute_share_features must not modify the input DataFrame in place."""
        original_cols = set(multi_year_raw.columns)
        compute_share_features(multi_year_raw)
        assert set(multi_year_raw.columns) == original_cols

    def test_row_count_preserved(self, multi_year_raw, shares_df):
        """Output row count must match input row count."""
        assert len(shares_df) == len(multi_year_raw)


# ---------------------------------------------------------------------------
# compute_change_features
# ---------------------------------------------------------------------------


class TestComputeChangeFeatures:
    """Tests for compute_change_features()."""

    def test_one_row_per_county(self, shares_df):
        """Output must have exactly one row per unique county_fips."""
        change = compute_change_features(shares_df)
        assert change["county_fips"].nunique() == len(change)

    def test_change_equals_latest_minus_earliest(self, shares_df):
        """dem_share_change must equal latest dem_share - earliest dem_share."""
        change = compute_change_features(shares_df)
        alachua_change = change[change["county_fips"] == "12001"]
        assert len(alachua_change) == 1

        alachua_shares = shares_df[shares_df["county_fips"] == "12001"].sort_values("election_year")
        expected_dem_change = (
            alachua_shares.iloc[-1]["dem_share"] - alachua_shares.iloc[0]["dem_share"]
        )
        assert alachua_change.iloc[0]["dem_share_change"] == pytest.approx(expected_dem_change, rel=1e-4)

    def test_single_year_produces_nan_change(self):
        """Counties with only one snapshot must get NaN change features."""
        df_one_year = pd.DataFrame({
            "county_fips": ["12001"],
            "county_name": ["Alachua"],
            "election_year": [2024],
            "book_closing_date": [pd.Timestamp("2024-10-07")],
            "rep": [46584], "dem": [75430], "npa": [36858], "other": [4489], "total": [163361],
        })
        df_shares = compute_share_features(df_one_year)
        change = compute_change_features(df_shares)
        assert len(change) == 1
        assert pd.isna(change.iloc[0]["dem_share_change"])
        assert pd.isna(change.iloc[0]["rep_share_change"])
        assert pd.isna(change.iloc[0]["npa_share_change"])
        assert pd.isna(change.iloc[0]["registration_growth"])

    def test_registration_growth_direction(self, shares_df):
        """registration_growth must be positive if total grew and negative if it shrank."""
        change = compute_change_features(shares_df)
        # Baker county: 2016 total=15147, 2024 total=19097 → should be positive
        baker = change[change["county_fips"] == "12003"]
        assert len(baker) == 1
        assert baker.iloc[0]["registration_growth"] > 0

    def test_empty_input_returns_empty(self):
        """Empty input must return empty output with correct schema."""
        df_empty = pd.DataFrame(
            columns=["county_fips", "county_name", "election_year",
                     "dem_share", "rep_share", "npa_share", "total"]
        )
        result = compute_change_features(df_empty)
        assert len(result) == 0
        for col in CHANGE_FEATURE_COLS:
            assert col in result.columns

    def test_change_features_in_output(self, shares_df):
        """Output must contain all CHANGE_FEATURE_COLS."""
        change = compute_change_features(shares_df)
        for col in CHANGE_FEATURE_COLS:
            assert col in change.columns


# ---------------------------------------------------------------------------
# _safe_diff
# ---------------------------------------------------------------------------


class TestSafeDiff:
    """Tests for _safe_diff()."""

    def test_basic_subtraction(self):
        """_safe_diff(a, b) must return a - b for non-NaN inputs."""
        assert _safe_diff(0.5, 0.3) == pytest.approx(0.2)

    def test_nan_a_returns_nan(self):
        """_safe_diff(NaN, b) must return NaN."""
        assert pd.isna(_safe_diff(float("nan"), 0.3))

    def test_nan_b_returns_nan(self):
        """_safe_diff(a, NaN) must return NaN."""
        assert pd.isna(_safe_diff(0.5, float("nan")))

    def test_both_nan_returns_nan(self):
        """_safe_diff(NaN, NaN) must return NaN."""
        assert pd.isna(_safe_diff(float("nan"), float("nan")))

    def test_negative_result(self):
        """_safe_diff can return negative values."""
        result = _safe_diff(0.2, 0.5)
        assert result == pytest.approx(-0.3)


# ---------------------------------------------------------------------------
# build_county_features — integrated pipeline
# ---------------------------------------------------------------------------


class TestBuildCountyFeatures:
    """Tests for build_county_features()."""

    def test_one_row_per_county(self, county_features):
        """Output must have exactly one row per county_fips."""
        assert county_features["county_fips"].nunique() == len(county_features)

    def test_output_has_all_feature_columns(self, county_features):
        """Output must contain all VOTER_REG_FEATURE_COLS."""
        for col in VOTER_REG_FEATURE_COLS:
            assert col in county_features.columns, f"Missing column: {col}"

    def test_output_has_county_name(self, county_features):
        """Output must include county_name column."""
        assert "county_name" in county_features.columns

    def test_share_features_from_latest_year(self, county_features, multi_year_raw):
        """Share features must reflect the latest (2024) snapshot, not 2016."""
        # Alachua 2024: rep=46584, total=163361 → rep_share ≈ 0.285
        # Alachua 2016: rep=50097, total=177947 → rep_share ≈ 0.281
        alachua = county_features[county_features["county_fips"] == "12001"]
        assert len(alachua) == 1
        expected_rep_share = 46584 / 163361
        assert alachua.iloc[0]["rep_share"] == pytest.approx(expected_rep_share, rel=1e-3)

    def test_change_features_present_with_multiple_years(self, county_features):
        """With 2 years of data, change features must be non-NaN."""
        for col in CHANGE_FEATURE_COLS:
            valid_count = county_features[col].notna().sum()
            assert valid_count > 0, f"All change features for '{col}' are NaN"

    def test_empty_input_returns_empty(self):
        """Empty input must return empty output with correct schema."""
        result = build_county_features(pd.DataFrame())
        assert len(result) == 0
        for col in VOTER_REG_FEATURE_COLS:
            assert col in result.columns

    def test_all_fips_are_fl(self, county_features):
        """All county_fips in feature output must be FL (prefix 12)."""
        assert county_features["county_fips"].str.startswith("12").all()

    def test_dem_share_in_valid_range(self, county_features):
        """dem_share must be in [0, 1]."""
        valid = county_features["dem_share"].dropna()
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_baker_is_rep_leaning(self, county_features):
        """Baker county (majority Republican) must have rep_share > dem_share."""
        baker = county_features[county_features["county_fips"] == "12003"]
        assert len(baker) == 1
        assert baker.iloc[0]["rep_share"] > baker.iloc[0]["dem_share"]

    def test_alachua_is_dem_leaning(self, county_features):
        """Alachua county (majority Democrat) must have dem_share > rep_share."""
        alachua = county_features[county_features["county_fips"] == "12001"]
        assert len(alachua) == 1
        assert alachua.iloc[0]["dem_share"] > alachua.iloc[0]["rep_share"]

    def test_rep_minus_dem_sign_matches_majority(self, county_features):
        """rep_minus_dem must be negative for Alachua (Dem county) and positive for Baker (Rep)."""
        alachua = county_features[county_features["county_fips"] == "12001"].iloc[0]
        baker = county_features[county_features["county_fips"] == "12003"].iloc[0]
        assert alachua["rep_minus_dem"] < 0  # Dem-majority
        assert baker["rep_minus_dem"] > 0     # Rep-majority


# ---------------------------------------------------------------------------
# Integration tests (skip if data not present)
# ---------------------------------------------------------------------------


class TestVoterRegistrationIntegration:
    """Integration tests that verify the actual saved parquet files."""

    @pytest.fixture(scope="class")
    def raw_parquet(self):
        """Load data/raw/fl_voter_registration.parquet if it exists."""
        from pathlib import Path
        path = Path(__file__).parents[1] / "data" / "raw" / "fl_voter_registration.parquet"
        if not path.exists():
            pytest.skip("fl_voter_registration.parquet not found — run fetch_voter_registration.py first")
        return pd.read_parquet(path)

    @pytest.fixture(scope="class")
    def assembled_parquet(self):
        """Load data/assembled/county_voter_registration_features.parquet if it exists."""
        from pathlib import Path
        path = Path(__file__).parents[1] / "data" / "assembled" / "county_voter_registration_features.parquet"
        if not path.exists():
            pytest.skip(
                "county_voter_registration_features.parquet not found — "
                "run build_voter_registration_features.py first"
            )
        return pd.read_parquet(path)

    def test_raw_has_required_columns(self, raw_parquet):
        """Raw parquet must have all OUTPUT_COLUMNS."""
        assert set(OUTPUT_COLUMNS).issubset(set(raw_parquet.columns))

    def test_raw_has_5_election_cycles(self, raw_parquet):
        """Raw parquet must contain data for 5 election cycles."""
        assert raw_parquet["election_year"].nunique() == 5

    def test_raw_has_67_fl_counties(self, raw_parquet):
        """Each election cycle should cover all 67 FL counties."""
        for year, grp in raw_parquet.groupby("election_year"):
            assert len(grp) == 67, f"{year}: expected 67 counties, got {len(grp)}"

    def test_raw_fips_are_fl_only(self, raw_parquet):
        """All FIPS codes in raw file must start with '12'."""
        assert raw_parquet["county_fips"].str.startswith("12").all()

    def test_raw_totals_positive(self, raw_parquet):
        """All total registration counts must be positive."""
        assert (raw_parquet["total"] > 0).all()

    def test_assembled_has_feature_columns(self, assembled_parquet):
        """Assembled parquet must have all VOTER_REG_FEATURE_COLS."""
        for col in VOTER_REG_FEATURE_COLS:
            assert col in assembled_parquet.columns

    def test_assembled_has_67_counties(self, assembled_parquet):
        """Assembled features must have exactly 67 FL counties."""
        assert len(assembled_parquet) == 67

    def test_assembled_shares_sum_to_one(self, assembled_parquet):
        """Share features must sum to approximately 1 per county."""
        share_sum = (
            assembled_parquet["dem_share"]
            + assembled_parquet["rep_share"]
            + assembled_parquet["npa_share"]
            + assembled_parquet["other_share"]
        )
        assert share_sum.between(0.999, 1.001).all()

    def test_assembled_change_features_nonnull(self, assembled_parquet):
        """All change features must be non-null (5 years of data ensures this)."""
        for col in CHANGE_FEATURE_COLS:
            assert assembled_parquet[col].notna().all(), \
                f"Unexpected NaN values in '{col}'"
