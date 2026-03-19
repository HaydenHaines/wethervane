"""
Stage 1 data assembly: fetch CDC COVID-19 county-level vaccination data.

Source: CDC COVID-19 Vaccinations in the United States, County
Dataset ID: 8xkx-amqh (data.cdc.gov)
SODA API: https://data.cdc.gov/resource/8xkx-amqh.json
Scope: FL (FIPS 12), GA (FIPS 13), AL (FIPS 01) — 293 counties total

The CDC publishes daily county-level snapshots of COVID-19 vaccination coverage.
Each row represents one county on one date, with cumulative counts and percentages.

This fetcher retrieves the LATEST snapshot per county — the most recent date
for which vaccination percentages are recorded. For most counties this is late
2023 or early 2024, when CDC stopped updating the data.

Variables fetched per county (all are cumulative percentages):
  Series_Complete_Pop_Pct   : Percent of population fully vaccinated (2-dose or J&J)
  Booster_Doses_Vax_Pct     : Percent of fully vaccinated who received a booster
  Administered_Dose1_Pop_Pct: Percent of population who received at least 1 dose

SODA API pagination:
  Default: 1000 rows per request. This fetcher paginates using $limit/$offset
  until the full county set is retrieved.

Derived features (computed in build_covid_features.py):
  vax_complete_pct  : Series_Complete_Pop_Pct (fully vaccinated %)
  vax_booster_pct   : Booster_Doses_Vax_Pct
  vax_dose1_pct     : Administered_Dose1_Pop_Pct (at-least-1-dose %)

NaN handling:
  - Counties where the CDC suppressed or never reported vaccination data
    are retained with NaN values. Downstream imputation uses county-level
    state medians.
  - Suppressed small-count cells appear as empty strings in the JSON and
    are coerced to NaN.

Output: data/raw/covid_vaccination.parquet
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[2]
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "covid_vaccination.parquet"

# Target states: abbreviation → FIPS prefix
STATES = {"AL": "01", "FL": "12", "GA": "13"}

# Set of state FIPS prefixes we care about (for filtering)
TARGET_STATE_FIPS = frozenset(STATES.values())

# CDC SODA API endpoint for county-level COVID vaccination data
SODA_BASE_URL = "https://data.cdc.gov/resource/8xkx-amqh.json"

# SODA API pagination chunk size
SODA_PAGE_SIZE = 10_000

# Polite delay between pagination requests (seconds)
REQUEST_DELAY = 0.5

# CDC column names in the raw JSON response
CDC_COLUMNS = {
    "fips": "fips",
    "date": "date",
    "series_complete_pop_pct": "series_complete_pop_pct",
    "booster_doses_vax_pct": "booster_doses_vax_pct",
    "administered_dose1_pop_pct": "administered_dose1_pop_pct",
    "recip_county": "recip_county",
    "recip_state": "recip_state",
}

# Output column names (after renaming from CDC format)
OUTPUT_COLUMNS = [
    "county_fips",
    "state_abbr",
    "county_name",
    "date",
    "series_complete_pop_pct",
    "booster_doses_vax_pct",
    "administered_dose1_pop_pct",
]


def build_soda_url(offset: int = 0, limit: int = SODA_PAGE_SIZE) -> str:
    """Construct a CDC SODA API URL for one page of county vaccination data.

    Filters to FL (12xxx), GA (13xxx), and AL (01xxx) county FIPS codes using
    SODA's $where clause with LIKE patterns. Requests only the columns we need.

    Args:
        offset: Row offset for pagination.
        limit: Number of rows to request per page.

    Returns:
        Full SODA API URL string with query parameters.
    """
    # Filter FIPS codes to our three target states using SODA $where
    # FIPS is stored as a 5-character string in the CDC dataset
    where_clause = (
        "fips like '12%' OR fips like '13%' OR fips like '01%'"
    )
    select_cols = ",".join(CDC_COLUMNS.keys())

    params = {
        "$where": where_clause,
        "$select": select_cols,
        "$limit": limit,
        "$offset": offset,
        "$order": "fips,date",
    }

    # Build query string manually to preserve $-prefixed param names
    query_parts = [f"{k}={requests.utils.quote(str(v))}" for k, v in params.items()]
    return f"{SODA_BASE_URL}?{'&'.join(query_parts)}"


def fetch_page(offset: int = 0, limit: int = SODA_PAGE_SIZE) -> list[dict] | None:
    """Fetch one page of CDC SODA data.

    Args:
        offset: Row offset for pagination.
        limit: Number of rows per page.

    Returns:
        List of row dicts from the SODA JSON response, or None on failure.
    """
    url = build_soda_url(offset=offset, limit=limit)
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        log.warning("  CDC SODA request failed (offset=%d): %s", offset, exc)
        return None


def fetch_all_pages() -> pd.DataFrame:
    """Fetch all CDC vaccination data for FL/GA/AL counties via SODA pagination.

    Loops through SODA pages until a page returns fewer rows than requested
    (indicating we've reached the end of the result set).

    Returns:
        Combined DataFrame with all rows for FL/GA/AL counties.
        Empty DataFrame on failure.
    """
    all_rows: list[dict] = []
    offset = 0

    while True:
        log.info("  Fetching rows %d–%d...", offset, offset + SODA_PAGE_SIZE - 1)
        page = fetch_page(offset=offset, limit=SODA_PAGE_SIZE)

        if page is None:
            log.warning("  Page fetch failed at offset %d; stopping pagination", offset)
            break

        all_rows.extend(page)
        log.info("  Got %d rows (total so far: %d)", len(page), len(all_rows))

        if len(page) < SODA_PAGE_SIZE:
            # Last page — we've exhausted the result set
            break

        offset += SODA_PAGE_SIZE
        time.sleep(REQUEST_DELAY)

    if not all_rows:
        log.error("No data retrieved from CDC SODA API")
        return pd.DataFrame()

    return pd.DataFrame(all_rows)


def parse_raw_df(df: pd.DataFrame) -> pd.DataFrame:
    """Parse and clean the raw CDC SODA response DataFrame.

    Performs:
    1. Rename CDC column names to our output names
    2. Coerce vaccination percentage columns to float (empty strings → NaN)
    3. Parse date column
    4. Validate FIPS format (must be 5 digits)
    5. Filter to only our target states

    Args:
        df: Raw DataFrame from fetch_all_pages().

    Returns:
        Cleaned DataFrame with OUTPUT_COLUMNS.
    """
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = df.copy()

    # Rename columns to our canonical names
    rename_map = {
        "fips": "county_fips",
        "recip_state": "state_abbr",
        "recip_county": "county_name",
    }
    df = df.rename(columns=rename_map)

    # Ensure required columns exist (fill missing with NaN)
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            log.warning("  Expected column '%s' not found in CDC response; filling NaN", col)
            df[col] = None

    # Coerce vaccination percentage columns to float
    vax_pct_cols = [
        "series_complete_pop_pct",
        "booster_doses_vax_pct",
        "administered_dose1_pop_pct",
    ]
    for col in vax_pct_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse date column
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Validate FIPS: must be 5-digit strings
    fips_ok = df["county_fips"].str.match(r"^\d{5}$", na=False)
    n_bad = (~fips_ok).sum()
    if n_bad > 0:
        log.warning("  Dropping %d rows with non-5-digit FIPS", n_bad)
        df = df[fips_ok]

    # Filter to target state FIPS prefixes
    state_ok = df["county_fips"].str[:2].isin(TARGET_STATE_FIPS)
    n_filtered = (~state_ok).sum()
    if n_filtered > 0:
        log.warning("  Dropping %d rows outside FL/GA/AL FIPS range", n_filtered)
        df = df[state_ok]

    return df[OUTPUT_COLUMNS].reset_index(drop=True)


def get_latest_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce to the single latest snapshot per county.

    The CDC dataset contains daily snapshots. For our use (correlation with
    partisan lean), we want the final cumulative vaccination rate for each
    county — the record with the latest date that has non-null vaccination data.

    Strategy:
    1. Drop rows where all three vaccination pct columns are NaN
    2. Per county_fips, keep the row with the maximum date
    3. Rows with tied latest dates are deduplicated by taking the first

    Args:
        df: Parsed DataFrame from parse_raw_df().

    Returns:
        DataFrame with one row per county_fips (latest snapshot with data).
    """
    if df.empty:
        return df

    vax_cols = [
        "series_complete_pop_pct",
        "booster_doses_vax_pct",
        "administered_dose1_pop_pct",
    ]

    # Keep rows where at least one vaccination column has data
    has_data = df[vax_cols].notna().any(axis=1)
    n_dropped = (~has_data).sum()
    if n_dropped > 0:
        log.info("  Dropped %d rows with no vaccination data", n_dropped)
    df = df[has_data].copy()

    if df.empty:
        return df

    # Per county, keep the row with the maximum date
    idx = df.groupby("county_fips")["date"].idxmax()
    latest = df.loc[idx].reset_index(drop=True)

    log.info("  Reduced %d rows → %d counties (latest snapshot per county)", len(df), len(latest))
    return latest


def main() -> None:
    """Fetch CDC COVID-19 county-level vaccination data and save to parquet.

    Fetches all available snapshots for FL, GA, and AL counties from the
    CDC SODA API, reduces to the latest snapshot per county, and saves
    to data/raw/covid_vaccination.parquet.
    """
    log.info("Fetching CDC COVID-19 county vaccination data (FL, GA, AL)")
    log.info("Source: data.cdc.gov/resource/8xkx-amqh.json")
    log.info("Target states: %s", list(STATES.keys()))

    # Fetch all pages
    raw = fetch_all_pages()
    if raw.empty:
        log.error("No raw data retrieved. Aborting.")
        return

    log.info("Fetched %d raw rows across all pages", len(raw))

    # Parse and clean
    parsed = parse_raw_df(raw)
    log.info("After parsing: %d rows", len(parsed))

    if parsed.empty:
        log.error("No data after parsing. Aborting.")
        return

    # Reduce to latest snapshot per county
    latest = get_latest_snapshot(parsed)
    log.info("Latest snapshots: %d counties", len(latest))

    if latest.empty:
        log.error("No latest snapshots found. Aborting.")
        return

    # Summary statistics
    n_counties = len(latest)
    n_states = latest["state_abbr"].nunique()
    state_counts = latest.groupby("state_abbr").size().to_dict()
    date_range = f"{latest['date'].min().date()} to {latest['date'].max().date()}"

    log.info(
        "\nSummary: %d counties across %d states | date range: %s",
        n_counties, n_states, date_range,
    )
    for state, count in sorted(state_counts.items()):
        log.info("  %s: %d counties", state, count)

    # NaN audit
    vax_cols = [
        "series_complete_pop_pct",
        "booster_doses_vax_pct",
        "administered_dose1_pop_pct",
    ]
    nan_counts = latest[vax_cols].isna().sum()
    for col, n in nan_counts.items():
        if n > 0:
            log.info("  NaN in %s: %d counties (%.1f%%)", col, n, 100 * n / n_counties)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    latest.to_parquet(OUTPUT_PATH, index=False)
    log.info(
        "\nSaved → %s  (%d rows × %d cols)",
        OUTPUT_PATH, len(latest), len(latest.columns),
    )


if __name__ == "__main__":
    main()
