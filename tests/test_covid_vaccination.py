"""Tests for CDC COVID-19 vaccination data fetching and feature computation.

Tests exercise:
1. fetch_covid_vaccination.py — URL construction, response parsing, FIPS filtering,
   pagination logic, latest-snapshot deduplication
2. build_covid_features.py — feature computation, NaN handling, imputation,
   output schema

These tests use synthetic data and mock HTTP responses so they run without
network access or the real COVID vaccination parquet file. Tests verify:
  - SODA URL construction includes correct date filter and pagination params
  - Raw JSON response is correctly parsed and typed
  - FIPS validation rejects non-5-digit codes
  - All valid 5-digit county FIPS codes are retained (national scope)
  - Latest-snapshot logic selects the row with maximum date per county
  - Feature computation renames CDC columns and clips to [0, 100]
  - NaN counties handled gracefully (not filled, not dropped)
  - State-median imputation fills NaN with correct state peer median
  - Output schema has required columns and types
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.assembly.fetch_covid_vaccination import (
    DATE_CUTOFF,
    OUTPUT_COLUMNS,
    SODA_BASE_URL,
    build_soda_url,
    fetch_page,
    get_latest_snapshot,
    parse_raw_df,
)
from src.assembly.build_covid_features import (
    COVID_FEATURE_COLS,
    compute_covid_features,
    impute_covid_state_medians,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_raw_json() -> list[dict]:
    """Synthetic CDC SODA JSON rows for 6 FL, 4 GA, 3 AL counties.

    Designed to test:
    - Multiple dates per county (latest-snapshot logic)
    - One county with all-NaN vaccination data
    - One county where booster_doses_vax_pct is missing
    """
    rows = [
        # FL counties — two snapshots each for first two
        {
            "fips": "12001",
            "recip_state": "FL",
            "recip_county": "Alachua",
            "date": "2022-06-01T00:00:00.000",
            "series_complete_pop_pct": "55.0",
            "booster_doses_vax_pct": "35.0",
            "administered_dose1_pop_pct": "62.0",
        },
        {
            "fips": "12001",
            "recip_state": "FL",
            "recip_county": "Alachua",
            "date": "2023-01-15T00:00:00.000",
            "series_complete_pop_pct": "60.0",
            "booster_doses_vax_pct": "40.0",
            "administered_dose1_pop_pct": "67.0",
        },
        {
            "fips": "12003",
            "recip_state": "FL",
            "recip_county": "Baker",
            "date": "2022-06-01T00:00:00.000",
            "series_complete_pop_pct": "30.0",
            "booster_doses_vax_pct": "18.0",
            "administered_dose1_pop_pct": "35.0",
        },
        {
            "fips": "12003",
            "recip_state": "FL",
            "recip_county": "Baker",
            "date": "2023-02-01T00:00:00.000",
            "series_complete_pop_pct": "32.0",
            "booster_doses_vax_pct": "20.0",
            "administered_dose1_pop_pct": "37.0",
        },
        {
            "fips": "12005",
            "recip_state": "FL",
            "recip_county": "Bay",
            "date": "2023-01-15T00:00:00.000",
            "series_complete_pop_pct": "42.0",
            "booster_doses_vax_pct": "25.0",
            "administered_dose1_pop_pct": "48.0",
        },
        # FL county with missing booster data
        {
            "fips": "12007",
            "recip_state": "FL",
            "recip_county": "Bradford",
            "date": "2023-01-15T00:00:00.000",
            "series_complete_pop_pct": "38.0",
            "booster_doses_vax_pct": "",       # suppressed/missing
            "administered_dose1_pop_pct": "44.0",
        },
        # GA counties
        {
            "fips": "13001",
            "recip_state": "GA",
            "recip_county": "Appling",
            "date": "2023-01-01T00:00:00.000",
            "series_complete_pop_pct": "28.0",
            "booster_doses_vax_pct": "15.0",
            "administered_dose1_pop_pct": "32.0",
        },
        {
            "fips": "13003",
            "recip_state": "GA",
            "recip_county": "Atkinson",
            "date": "2023-01-01T00:00:00.000",
            "series_complete_pop_pct": "31.0",
            "booster_doses_vax_pct": "17.0",
            "administered_dose1_pop_pct": "36.0",
        },
        # GA county with all NaN vaccination data
        {
            "fips": "13005",
            "recip_state": "GA",
            "recip_county": "Bacon",
            "date": "2023-01-01T00:00:00.000",
            "series_complete_pop_pct": "",
            "booster_doses_vax_pct": "",
            "administered_dose1_pop_pct": "",
        },
        {
            "fips": "13007",
            "recip_state": "GA",
            "recip_county": "Baker",
            "date": "2023-01-01T00:00:00.000",
            "series_complete_pop_pct": "45.0",
            "booster_doses_vax_pct": "28.0",
            "administered_dose1_pop_pct": "50.0",
        },
        # AL counties
        {
            "fips": "01001",
            "recip_state": "AL",
            "recip_county": "Autauga",
            "date": "2023-01-01T00:00:00.000",
            "series_complete_pop_pct": "40.0",
            "booster_doses_vax_pct": "22.0",
            "administered_dose1_pop_pct": "46.0",
        },
        {
            "fips": "01003",
            "recip_state": "AL",
            "recip_county": "Baldwin",
            "date": "2023-01-01T00:00:00.000",
            "series_complete_pop_pct": "43.0",
            "booster_doses_vax_pct": "25.0",
            "administered_dose1_pop_pct": "49.0",
        },
        {
            "fips": "01005",
            "recip_state": "AL",
            "recip_county": "Barbour",
            "date": "2023-01-01T00:00:00.000",
            "series_complete_pop_pct": "33.0",
            "booster_doses_vax_pct": "18.0",
            "administered_dose1_pop_pct": "38.0",
        },
    ]
    return rows


@pytest.fixture(scope="module")
def parsed_df(synthetic_raw_json) -> pd.DataFrame:
    """Parsed DataFrame from synthetic JSON (before latest-snapshot reduction)."""
    raw = pd.DataFrame(synthetic_raw_json)
    return parse_raw_df(raw)


@pytest.fixture(scope="module")
def latest_df(parsed_df) -> pd.DataFrame:
    """Latest-snapshot DataFrame: one row per county."""
    return get_latest_snapshot(parsed_df)


@pytest.fixture(scope="module")
def features_df(latest_df) -> pd.DataFrame:
    """Computed COVID features from latest snapshots."""
    return compute_covid_features(latest_df)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Tests for module-level constants."""

    def test_date_cutoff_defined(self):
        """DATE_CUTOFF must be a non-empty string in YYYY-MM-DD format."""
        assert isinstance(DATE_CUTOFF, str)
        assert len(DATE_CUTOFF) == 10
        assert DATE_CUTOFF[:4].isdigit()

    def test_output_columns_includes_required_fields(self):
        """OUTPUT_COLUMNS must include county_fips, state_abbr, date, and all 3 vax pct cols."""
        required = {
            "county_fips",
            "state_abbr",
            "date",
            "series_complete_pop_pct",
            "booster_doses_vax_pct",
            "administered_dose1_pop_pct",
        }
        assert required.issubset(set(OUTPUT_COLUMNS))

    def test_covid_feature_cols_count(self):
        """COVID_FEATURE_COLS must have exactly 3 features."""
        assert len(COVID_FEATURE_COLS) == 3

    def test_covid_feature_cols_names(self):
        """COVID_FEATURE_COLS must include the canonical feature names."""
        assert "vax_complete_pct" in COVID_FEATURE_COLS
        assert "vax_booster_pct" in COVID_FEATURE_COLS
        assert "vax_dose1_pct" in COVID_FEATURE_COLS


# ---------------------------------------------------------------------------
# URL construction
# ---------------------------------------------------------------------------


class TestBuildSodaUrl:
    """Tests for build_soda_url()."""

    def test_url_starts_with_soda_base(self):
        """URL must be rooted at the CDC SODA base endpoint."""
        url = build_soda_url()
        assert url.startswith(SODA_BASE_URL)

    def test_url_contains_date_filter(self):
        """URL must include a date cutoff filter (not a FIPS state filter)."""
        url = build_soda_url()
        # National scope: filter is by date, not by state FIPS
        assert "date" in url
        assert DATE_CUTOFF[:4] in url  # Year from cutoff must appear in URL

    def test_url_contains_limit(self):
        """URL must include $limit parameter."""
        url = build_soda_url(limit=500)
        assert "$limit" in url
        assert "500" in url

    def test_url_contains_offset(self):
        """URL must include $offset parameter."""
        url = build_soda_url(offset=5000)
        assert "$offset" in url
        assert "5000" in url

    def test_offset_zero_by_default(self):
        """Default offset must be 0."""
        url = build_soda_url()
        assert "$offset=0" in url

    def test_different_offsets_produce_different_urls(self):
        """Different offset values must produce different URLs."""
        url1 = build_soda_url(offset=0)
        url2 = build_soda_url(offset=10000)
        assert url1 != url2

    def test_url_contains_select_columns(self):
        """URL must include $select with the CDC column names."""
        url = build_soda_url()
        assert "$select" in url
        assert "fips" in url


# ---------------------------------------------------------------------------
# SODA fetch_page — mocked HTTP
# ---------------------------------------------------------------------------


class TestFetchPage:
    """Tests for fetch_page() with mocked HTTP responses."""

    def test_returns_list_on_success(self):
        """fetch_page() must return a list of dicts on HTTP 200."""
        mock_response = MagicMock()
        mock_response.json.return_value = [{"fips": "12001", "date": "2023-01-01T00:00:00.000"}]
        mock_response.raise_for_status.return_value = None

        with patch("src.assembly.fetch_covid_vaccination.requests.get", return_value=mock_response):
            result = fetch_page(offset=0)

        assert isinstance(result, list)
        assert len(result) == 1

    def test_returns_none_on_http_error(self):
        """fetch_page() must return None when the HTTP request fails."""
        import requests as _requests

        with patch(
            "src.assembly.fetch_covid_vaccination.requests.get",
            side_effect=_requests.RequestException("timeout"),
        ):
            result = fetch_page(offset=0)

        assert result is None

    def test_returns_empty_list_for_empty_response(self):
        """fetch_page() must return empty list when API returns no rows."""
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None

        with patch("src.assembly.fetch_covid_vaccination.requests.get", return_value=mock_response):
            result = fetch_page(offset=0)

        assert result == []


# ---------------------------------------------------------------------------
# parse_raw_df — response parsing
# ---------------------------------------------------------------------------


class TestParseRawDf:
    """Tests for parse_raw_df() — raw JSON DataFrame → clean output."""

    def test_output_has_required_columns(self, parsed_df):
        """Output must have all OUTPUT_COLUMNS."""
        assert set(OUTPUT_COLUMNS).issubset(set(parsed_df.columns))

    def test_county_fips_are_5_digits(self, parsed_df):
        """county_fips must be 5-character digit strings."""
        assert (parsed_df["county_fips"].str.len() == 5).all()
        assert parsed_df["county_fips"].str.isdigit().all()

    def test_date_column_is_datetime(self, parsed_df):
        """date column must be parsed as datetime."""
        assert pd.api.types.is_datetime64_any_dtype(parsed_df["date"])

    def test_vax_pct_columns_are_numeric(self, parsed_df):
        """All three vaccination percentage columns must be numeric."""
        for col in ["series_complete_pop_pct", "booster_doses_vax_pct", "administered_dose1_pop_pct"]:
            assert pd.api.types.is_float_dtype(parsed_df[col]) or pd.api.types.is_integer_dtype(parsed_df[col]), \
                f"Column '{col}' is not numeric: dtype={parsed_df[col].dtype}"

    def test_empty_string_pct_coerced_to_nan(self, parsed_df):
        """Empty string vaccination percentages must be coerced to NaN."""
        # The synthetic data has empty strings for some booster and GA county values
        # After parsing, those rows should have NaN not empty string
        booster_nulls = parsed_df["booster_doses_vax_pct"].isna().sum()
        assert booster_nulls >= 1, "Expected at least one NaN booster_doses_vax_pct from empty string input"

    def test_national_scope_keeps_all_valid_fips(self):
        """parse_raw_df must keep all valid 5-digit county FIPS (not just FL/GA/AL)."""
        raw = pd.DataFrame([
            {
                "fips": "06001",  # California — should be KEPT (national scope)
                "recip_state": "CA",
                "recip_county": "Alameda",
                "date": "2023-01-01T00:00:00.000",
                "series_complete_pop_pct": "70.0",
                "booster_doses_vax_pct": "45.0",
                "administered_dose1_pop_pct": "75.0",
            },
            {
                "fips": "12001",  # FL — should be kept
                "recip_state": "FL",
                "recip_county": "Alachua",
                "date": "2023-01-01T00:00:00.000",
                "series_complete_pop_pct": "60.0",
                "booster_doses_vax_pct": "38.0",
                "administered_dose1_pop_pct": "66.0",
            },
        ])
        result = parse_raw_df(raw)
        assert len(result) == 2  # Both rows kept — national scope

    def test_bad_fips_dropped(self):
        """Rows with non-5-digit FIPS must be dropped."""
        raw = pd.DataFrame([
            {
                "fips": "1234",  # Only 4 digits — invalid
                "recip_state": "FL",
                "recip_county": "Unknown",
                "date": "2023-01-01T00:00:00.000",
                "series_complete_pop_pct": "50.0",
                "booster_doses_vax_pct": "30.0",
                "administered_dose1_pop_pct": "55.0",
            },
            {
                "fips": "12001",  # Valid
                "recip_state": "FL",
                "recip_county": "Alachua",
                "date": "2023-01-01T00:00:00.000",
                "series_complete_pop_pct": "60.0",
                "booster_doses_vax_pct": "38.0",
                "administered_dose1_pop_pct": "66.0",
            },
        ])
        result = parse_raw_df(raw)
        assert len(result) == 1
        assert result.iloc[0]["county_fips"] == "12001"

    def test_multiple_states_retained(self, parsed_df):
        """Counties from multiple states must all be present in parsed output."""
        state_prefixes = parsed_df["county_fips"].str[:2].unique()
        # Synthetic data includes FL (12), GA (13), AL (01) — all should be retained
        assert "12" in state_prefixes  # FL
        assert "13" in state_prefixes  # GA
        assert "01" in state_prefixes  # AL

    def test_empty_input_returns_empty(self):
        """Empty DataFrame input must return empty output with correct schema."""
        result = parse_raw_df(pd.DataFrame())
        assert len(result) == 0
        assert set(OUTPUT_COLUMNS).issubset(set(result.columns))


# ---------------------------------------------------------------------------
# get_latest_snapshot — deduplication logic
# ---------------------------------------------------------------------------


class TestGetLatestSnapshot:
    """Tests for get_latest_snapshot() — one row per county (latest date)."""

    def test_one_row_per_county(self, latest_df):
        """Output must have exactly one row per unique county_fips."""
        n_unique = latest_df["county_fips"].nunique()
        assert len(latest_df) == n_unique

    def test_selects_maximum_date(self, parsed_df, latest_df):
        """For counties with multiple dates, must select the latest date."""
        # Check county 12001 (Alachua FL) — has 2022-06-01 and 2023-01-15
        alachua = latest_df[latest_df["county_fips"] == "12001"]
        assert len(alachua) == 1
        assert alachua.iloc[0]["date"].year == 2023

    def test_selects_later_values_for_fl_county(self, latest_df):
        """Values for FL county 12001 must be from the 2023 snapshot (60.0), not 2022 (55.0)."""
        alachua = latest_df[latest_df["county_fips"] == "12001"]
        assert alachua.iloc[0]["series_complete_pop_pct"] == pytest.approx(60.0)

    def test_drops_all_nan_rows(self, parsed_df, latest_df):
        """County 13005 has all-empty vaccination data and must be excluded from output."""
        bacon_in_parsed = (parsed_df["county_fips"] == "13005").any()
        bacon_in_latest = (latest_df["county_fips"] == "13005").any()

        # The row exists in parsed (with NaN values) but is excluded from latest
        # (get_latest_snapshot drops rows where all vax cols are NaN)
        if bacon_in_parsed:
            # If it survived parsing, it should be excluded from latest
            assert not bacon_in_latest, (
                "County 13005 (all-NaN vaccination data) should not appear in latest snapshot"
            )

    def test_county_with_partial_nan_is_retained(self, latest_df):
        """County 12007 has NaN booster data but valid other columns — must be retained."""
        bradford = latest_df[latest_df["county_fips"] == "12007"]
        assert len(bradford) == 1

    def test_empty_input_returns_empty(self):
        """Empty input must return empty output."""
        result = get_latest_snapshot(pd.DataFrame(columns=OUTPUT_COLUMNS))
        assert len(result) == 0

    def test_output_has_all_output_columns(self, latest_df):
        """Latest snapshot output must have all OUTPUT_COLUMNS."""
        assert set(OUTPUT_COLUMNS).issubset(set(latest_df.columns))


# ---------------------------------------------------------------------------
# FIPS filtering — target states
# ---------------------------------------------------------------------------


class TestFipsFiltering:
    """Tests for FIPS-based state scope filtering."""

    def test_fl_fips_prefix_12(self):
        """FL county FIPS codes must start with '12'."""
        raw = pd.DataFrame([{
            "fips": "12086",
            "recip_state": "FL",
            "recip_county": "Miami-Dade",
            "date": "2023-01-01T00:00:00.000",
            "series_complete_pop_pct": "65.0",
            "booster_doses_vax_pct": "42.0",
            "administered_dose1_pop_pct": "70.0",
        }])
        result = parse_raw_df(raw)
        assert len(result) == 1
        assert result.iloc[0]["county_fips"][:2] == "12"

    def test_ga_fips_prefix_13(self):
        """GA county FIPS codes must start with '13'."""
        raw = pd.DataFrame([{
            "fips": "13121",
            "recip_state": "GA",
            "recip_county": "Fulton",
            "date": "2023-01-01T00:00:00.000",
            "series_complete_pop_pct": "55.0",
            "booster_doses_vax_pct": "35.0",
            "administered_dose1_pop_pct": "62.0",
        }])
        result = parse_raw_df(raw)
        assert len(result) == 1
        assert result.iloc[0]["county_fips"][:2] == "13"

    def test_al_fips_prefix_01(self):
        """AL county FIPS codes must start with '01'."""
        raw = pd.DataFrame([{
            "fips": "01073",
            "recip_state": "AL",
            "recip_county": "Jefferson",
            "date": "2023-01-01T00:00:00.000",
            "series_complete_pop_pct": "44.0",
            "booster_doses_vax_pct": "26.0",
            "administered_dose1_pop_pct": "50.0",
        }])
        result = parse_raw_df(raw)
        assert len(result) == 1
        assert result.iloc[0]["county_fips"][:2] == "01"

    def test_non_fl_ga_al_state_retained(self):
        """Texas (FIPS prefix 48) must be retained — national scope."""
        raw = pd.DataFrame([{
            "fips": "48201",  # Harris County TX — kept (national scope)
            "recip_state": "TX",
            "recip_county": "Harris",
            "date": "2023-01-01T00:00:00.000",
            "series_complete_pop_pct": "58.0",
            "booster_doses_vax_pct": "36.0",
            "administered_dose1_pop_pct": "64.0",
        }])
        result = parse_raw_df(raw)
        assert len(result) == 1
        assert result.iloc[0]["county_fips"] == "48201"


# ---------------------------------------------------------------------------
# compute_covid_features — feature computation
# ---------------------------------------------------------------------------


class TestComputeCovidFeatures:
    """Tests for compute_covid_features() — CDC column → feature renaming."""

    def test_output_has_required_columns(self, features_df):
        """Output must have county_fips, state_abbr, and all COVID_FEATURE_COLS."""
        required = {"county_fips", "state_abbr"} | set(COVID_FEATURE_COLS)
        assert required.issubset(set(features_df.columns))

    def test_row_count_matches_input(self, latest_df, features_df):
        """Output must have the same number of rows as input."""
        assert len(features_df) == len(latest_df)

    def test_vax_complete_pct_renamed(self, features_df):
        """series_complete_pop_pct must be renamed to vax_complete_pct."""
        assert "vax_complete_pct" in features_df.columns
        assert "series_complete_pop_pct" not in features_df.columns

    def test_vax_booster_pct_renamed(self, features_df):
        """booster_doses_vax_pct must be renamed to vax_booster_pct."""
        assert "vax_booster_pct" in features_df.columns
        assert "booster_doses_vax_pct" not in features_df.columns

    def test_vax_dose1_pct_renamed(self, features_df):
        """administered_dose1_pop_pct must be renamed to vax_dose1_pct."""
        assert "vax_dose1_pct" in features_df.columns
        assert "administered_dose1_pop_pct" not in features_df.columns

    def test_values_clipped_to_100(self):
        """Values exceeding 100 must be clipped down to 100."""
        df = pd.DataFrame({
            "county_fips": ["12001"],
            "state_abbr": ["FL"],
            "county_name": ["Alachua"],
            "date": [pd.Timestamp("2023-01-01")],
            "series_complete_pop_pct": [102.5],  # Slightly over 100 (CDC artifact)
            "booster_doses_vax_pct": [50.0],
            "administered_dose1_pop_pct": [99.0],
        })
        result = compute_covid_features(df)
        assert result.iloc[0]["vax_complete_pct"] == pytest.approx(100.0)

    def test_values_clipped_to_zero(self):
        """Negative values must be clipped up to 0."""
        df = pd.DataFrame({
            "county_fips": ["12001"],
            "state_abbr": ["FL"],
            "county_name": ["Alachua"],
            "date": [pd.Timestamp("2023-01-01")],
            "series_complete_pop_pct": [-1.0],  # Impossible but defensive
            "booster_doses_vax_pct": [50.0],
            "administered_dose1_pop_pct": [55.0],
        })
        result = compute_covid_features(df)
        assert result.iloc[0]["vax_complete_pct"] == pytest.approx(0.0)

    def test_nan_preserved_for_missing_data(self, features_df):
        """NaN values from suppressed CDC data must remain NaN in features."""
        # County 12007 had empty string booster → NaN after parsing
        bradford = features_df[features_df["county_fips"] == "12007"]
        if len(bradford) == 1:
            assert pd.isna(bradford.iloc[0]["vax_booster_pct"])
            # But other features should be present
            assert not pd.isna(bradford.iloc[0]["vax_complete_pct"])

    def test_values_are_numeric(self, features_df):
        """All COVID_FEATURE_COLS must be numeric dtype."""
        for col in COVID_FEATURE_COLS:
            assert pd.api.types.is_float_dtype(features_df[col]) or \
                   pd.api.types.is_integer_dtype(features_df[col]), \
                f"Column '{col}' is not numeric"

    def test_county_fips_passed_through(self, latest_df, features_df):
        """county_fips must be passed through unchanged."""
        pd.testing.assert_series_equal(
            features_df["county_fips"].reset_index(drop=True),
            latest_df["county_fips"].reset_index(drop=True),
        )

    def test_empty_input_returns_correct_schema(self):
        """Empty input must return empty output with correct columns."""
        result = compute_covid_features(pd.DataFrame())
        assert set(["county_fips", "state_abbr"] + COVID_FEATURE_COLS).issubset(
            set(result.columns)
        )
        assert len(result) == 0


# ---------------------------------------------------------------------------
# impute_covid_state_medians — imputation logic
# ---------------------------------------------------------------------------


class TestImpute:
    """Tests for impute_covid_state_medians()."""

    @pytest.fixture(scope="class")
    def partial_nan_features(self) -> pd.DataFrame:
        """Synthetic features with one county missing vax_booster_pct."""
        return pd.DataFrame({
            "county_fips": ["12001", "12003", "12005", "13001", "13003"],
            "state_abbr": ["FL", "FL", "FL", "GA", "GA"],
            "vax_complete_pct": [60.0, 32.0, 42.0, 28.0, 31.0],
            "vax_booster_pct": [40.0, np.nan, 25.0, 15.0, 17.0],  # 12003 is NaN
            "vax_dose1_pct": [67.0, 37.0, 48.0, 32.0, 36.0],
        })

    def test_imputation_fills_nan(self, partial_nan_features):
        """NaN features should be filled after state-median imputation."""
        imputed = impute_covid_state_medians(partial_nan_features)
        assert not pd.isna(imputed.loc[1, "vax_booster_pct"])

    def test_imputation_uses_state_median(self, partial_nan_features):
        """Imputed value must match the FL state median of non-NaN booster values."""
        imputed = impute_covid_state_medians(partial_nan_features)

        # FL non-NaN booster values: 40.0 (12001), 25.0 (12005) → median = 32.5
        fl_booster_values = [40.0, 25.0]
        expected_fl_median = float(np.median(fl_booster_values))
        assert imputed.loc[1, "vax_booster_pct"] == pytest.approx(expected_fl_median, rel=1e-3)

    def test_imputation_uses_state_not_global(self, partial_nan_features):
        """Imputed value for FL county must use FL median, not GA median."""
        imputed = impute_covid_state_medians(partial_nan_features)
        # FL median ≠ GA median, so verify it matches FL
        fl_mask = partial_nan_features["state_abbr"] == "FL"
        non_nan_mask = partial_nan_features["vax_booster_pct"].notna()
        fl_median = partial_nan_features.loc[fl_mask & non_nan_mask, "vax_booster_pct"].median()
        assert imputed.loc[1, "vax_booster_pct"] == pytest.approx(fl_median, rel=1e-3)

    def test_imputation_preserves_non_nan_values(self, partial_nan_features):
        """Non-NaN values must not be modified by imputation."""
        original = partial_nan_features.copy()
        imputed = impute_covid_state_medians(partial_nan_features)

        non_nan_mask = partial_nan_features["vax_booster_pct"].notna()
        pd.testing.assert_series_equal(
            imputed.loc[non_nan_mask, "vax_booster_pct"],
            original.loc[non_nan_mask, "vax_booster_pct"],
        )

    def test_no_nan_after_full_imputation(self):
        """After imputation, no NaN should remain when all states have non-NaN peers.

        Each state must have at least one non-NaN value for each feature so
        that the state median is defined and can be used to fill the NaN county.
        """
        # Each state has 2 counties: one with NaN, one with a valid value.
        # This guarantees a non-NaN peer exists in every state for every feature.
        df = pd.DataFrame({
            "county_fips": ["12001", "12003", "13001", "13003", "01001", "01003"],
            "state_abbr": ["FL", "FL", "GA", "GA", "AL", "AL"],
            "vax_complete_pct": [60.0, np.nan, 28.0, 30.0, 40.0, 42.0],
            "vax_booster_pct": [40.0, 35.0, np.nan, 17.0, 22.0, 24.0],
            "vax_dose1_pct": [67.0, 38.0, 32.0, 36.0, np.nan, 49.0],
        })
        imputed = impute_covid_state_medians(df)
        remaining_nan = imputed[COVID_FEATURE_COLS].isna().sum().sum()
        assert remaining_nan == 0, f"{remaining_nan} NaN values remain after imputation"


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


class TestOutputSchema:
    """Tests for the output DataFrame schema from fetch and feature pipelines."""

    def test_parsed_df_fips_match_state_abbr(self, parsed_df):
        """FIPS prefix in county_fips must match the state_abbr column."""
        fips_to_abbr = {"01": "AL", "12": "FL", "13": "GA"}
        derived_state = parsed_df["county_fips"].str[:2].map(fips_to_abbr)
        # Rows with state_abbr should match derived state
        rows_with_abbr = parsed_df["state_abbr"].notna()
        mismatches = parsed_df.loc[rows_with_abbr, "state_abbr"] != derived_state[rows_with_abbr]
        assert not mismatches.any(), (
            f"{mismatches.sum()} rows have FIPS/state_abbr mismatch"
        )

    def test_features_df_has_only_three_states(self, features_df):
        """Features output must contain only FL, GA, and AL counties."""
        state_prefixes = features_df["county_fips"].str[:2].unique()
        assert set(state_prefixes) <= {"01", "12", "13"}

    def test_vax_values_in_plausible_range(self, features_df):
        """Vaccination percentages must be in [0, 100]."""
        for col in COVID_FEATURE_COLS:
            valid = features_df[col].dropna()
            assert (valid >= 0).all(), f"{col} has values below 0"
            assert (valid <= 100).all(), f"{col} has values above 100"

    def test_county_fips_are_five_digit_strings(self, features_df):
        """county_fips must be 5-digit strings throughout the pipeline."""
        assert (features_df["county_fips"].str.len() == 5).all()
        assert features_df["county_fips"].str.isdigit().all()


# ---------------------------------------------------------------------------
# Integration tests (skip if data not present)
# ---------------------------------------------------------------------------


class TestCovidVaccinationIntegration:
    """Integration tests that verify the actual saved parquet files (skipped if absent)."""

    @pytest.fixture(scope="class")
    def raw_parquet(self):
        """Load data/raw/covid_vaccination.parquet if it exists."""
        from pathlib import Path
        path = Path(__file__).parents[1] / "data" / "raw" / "covid_vaccination.parquet"
        if not path.exists():
            pytest.skip("covid_vaccination.parquet not found — run fetch_covid_vaccination.py first")
        return pd.read_parquet(path)

    @pytest.fixture(scope="class")
    def assembled_parquet(self):
        """Load data/assembled/county_covid_features.parquet if it exists."""
        from pathlib import Path
        path = Path(__file__).parents[1] / "data" / "assembled" / "county_covid_features.parquet"
        if not path.exists():
            pytest.skip("county_covid_features.parquet not found — run build_covid_features.py first")
        return pd.read_parquet(path)

    def test_raw_has_required_columns(self, raw_parquet):
        """Raw parquet must have all OUTPUT_COLUMNS."""
        assert set(OUTPUT_COLUMNS).issubset(set(raw_parquet.columns))

    def test_raw_one_row_per_county(self, raw_parquet):
        """Raw parquet (latest snapshots) must have one row per county_fips."""
        assert raw_parquet["county_fips"].nunique() == len(raw_parquet)

    def test_raw_fips_are_five_digits(self, raw_parquet):
        """All FIPS codes in raw file must be 5-digit strings."""
        assert (raw_parquet["county_fips"].str.len() == 5).all()
        assert raw_parquet["county_fips"].str.isdigit().all()

    def test_raw_covers_all_three_states(self, raw_parquet):
        """Raw parquet must include counties from FL, GA, and AL."""
        prefixes = raw_parquet["county_fips"].str[:2].unique()
        assert "12" in prefixes, "No FL counties found"
        assert "13" in prefixes, "No GA counties found"
        assert "01" in prefixes, "No AL counties found"

    def test_assembled_has_required_columns(self, assembled_parquet):
        """Assembled parquet must have county_fips, state_abbr, and COVID_FEATURE_COLS."""
        required = {"county_fips", "state_abbr"} | set(COVID_FEATURE_COLS)
        assert required.issubset(set(assembled_parquet.columns))

    def test_assembled_covers_all_fifty_states(self, assembled_parquet):
        """Assembled features must include counties from all US states (national scope)."""
        states = set(assembled_parquet["state_abbr"].unique())
        # Spot-check a few states from across the country
        assert "FL" in states, "FL missing from national assembled data"
        assert "GA" in states, "GA missing from national assembled data"
        assert "AL" in states, "AL missing from national assembled data"
        assert "TX" in states, "TX missing from national assembled data"
        assert "CA" in states, "CA missing from national assembled data"
        # Should have substantially more than 3 states
        assert len(states) >= 50, f"Expected 50+ states, got: {len(states)}"

    def test_assembled_vax_values_in_range(self, assembled_parquet):
        """All vaccination features must be in [0, 100] after feature computation."""
        for col in COVID_FEATURE_COLS:
            valid = assembled_parquet[col].dropna()
            assert (valid >= 0).all(), f"{col} has values below 0"
            assert (valid <= 100).all(), f"{col} has values above 100"

    def test_assembled_fips_format_valid(self, assembled_parquet):
        """All county_fips values must be 5-digit strings."""
        all_5_digit = assembled_parquet["county_fips"].str.match(r"^\d{5}$").all()
        assert all_5_digit, "Some county_fips values are not 5-digit strings"
