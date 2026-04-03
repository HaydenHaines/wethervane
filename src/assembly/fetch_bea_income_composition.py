"""Fetch BEA state-level personal income composition data from the Regional API.

Source: BEA Regional Data API
  https://apps.bea.gov/api/data/

Dataset: SAINC4 -- Personal Income and Employment by Major Component (state-level)

Line codes used (SAINC4):
  LineCode 10: Personal income (total)
  LineCode 50: Wages and salaries
  LineCode 46: Dividends, interest, and rent
  LineCode 47: Personal current transfer receipts

From these we compute three composition shares:
  wages_share      = wages_and_salaries / personal_income
  transfer_share   = personal_current_transfer_receipts / personal_income
  investment_share = dividends_interest_rent / personal_income

Why state-level rather than county-level for this use case?
  The existing county_bea_features.parquet (from fetch_bea_income.py / CAINC4) provides
  county-level income shares (earnings/transfers/investment). SAINC4 provides the same
  decomposition at the state level with official BEA aggregates. The state-level shares
  serve as a complementary coarser signal: counties in high-transfer-share states are
  exposed to different economic incentive structures than those in wage-dominated states,
  independent of the county's own composition. This state-level context correlates with
  political behavior in ways the county-level signal does not fully capture.

API key:
  Set environment variable BEA_API_KEY.
  Free key: https://apps.bea.gov/API/signup/

Output:
  data/assembled/bea_income_composition.parquet
    Columns: state_abbr, state_fips_prefix, wages_share, transfer_share, investment_share

Cache:
  data/raw/bea/sainc4_national_{year}.parquet
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from src.core import config as _cfg

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "bea"
ASSEMBLED_DIR = PROJECT_ROOT / "data" / "assembled"

OUTPUT_PATH = ASSEMBLED_DIR / "bea_income_composition.parquet"

BEA_BASE = "https://apps.bea.gov/api/data/"

# SAINC4 = "Personal Income and Employment by Major Component" -- state level
BEA_TABLE = "SAINC4"

# Line codes for the components we need.
LINE_PERSONAL_INCOME = 10      # Personal income (total)
LINE_WAGES_SALARIES = 50       # Wages and salaries
LINE_DIVIDENDS_INTEREST = 46   # Dividends, interest, and rent
LINE_TRANSFERS = 47            # Personal current transfer receipts

PRIMARY_YEAR = 2023
FALLBACK_YEAR = 2022

# Maps 2-digit FIPS prefix -> state abbreviation (e.g. "12" -> "FL")
STATE_ABBR: dict[str, str] = _cfg.STATE_ABBR

# Output column names
COL_WAGES = "wages_share"
COL_TRANSFER = "transfer_share"
COL_INVESTMENT = "investment_share"
SHARE_COLS = [COL_WAGES, COL_TRANSFER, COL_INVESTMENT]

FEATURE_COLS = SHARE_COLS  # alias for pipeline conventions


def _get_api_key() -> str:
    """Return the BEA API key from environment, or raise with a helpful message."""
    key = os.environ.get("BEA_API_KEY", "").strip()
    if not key:
        raise EnvironmentError(
            "BEA_API_KEY environment variable is not set.\n"
            "Get a free key at https://apps.bea.gov/API/signup/\n"
            "Then export it: export BEA_API_KEY=your_key_here"
        )
    return key


def _fetch_sainc4_one_linecode(line_code: int, year: int, api_key: str) -> pd.DataFrame:
    """Fetch SAINC4 data for ALL US states for one LineCode and one year.

    The BEA Regional API does not support multiple LineCode values in a single
    request (APIErrorCode 41). Each component must be fetched in a separate call.
    GeoFips=STATE fetches all 50 states + DC in one request.

    Returns a DataFrame with columns: GeoFips, GeoName, LineCode, TimePeriod, DataValue.
    """
    params = {
        "UserID": api_key,
        "method": "GetData",
        "DataSetName": "Regional",
        "TableName": BEA_TABLE,
        "LineCode": str(line_code),
        "GeoFips": "STATE",
        "Year": str(year),
        "ResultFormat": "JSON",
    }

    log.info("  Fetching BEA %s LineCode=%d year=%d ...", BEA_TABLE, line_code, year)
    resp = requests.get(BEA_BASE, params=params, timeout=120)
    resp.raise_for_status()

    payload = resp.json()

    if "BEAAPI" not in payload:
        raise ValueError(f"Unexpected BEA response structure: {list(payload.keys())}")

    beaapi = payload["BEAAPI"]
    if "Results" not in beaapi:
        error = beaapi.get("Error", {})
        raise ValueError(f"BEA API error: {error}")

    results = beaapi["Results"]
    if "Error" in results and "Data" not in results:
        raise ValueError(f"BEA API error: {results['Error']}")

    data = results.get("Data", [])
    if not data:
        log.warning("BEA returned no data for %s LineCode=%d year=%d", BEA_TABLE, line_code, year)
        return pd.DataFrame(columns=["GeoFips", "GeoName", "LineCode", "TimePeriod", "DataValue"])

    df = pd.DataFrame(data)
    df["LineCode"] = str(line_code)
    return df


def _fetch_sainc4_national(year: int, api_key: str) -> pd.DataFrame:
    """Fetch SAINC4 for all US states, all required LineCodes, and cache the result.

    Makes exactly 4 API calls (one per LineCode) and caches the combined result.
    """
    national_cache = RAW_DIR / f"sainc4_national_{year}.parquet"
    if national_cache.exists():
        log.info("Using cached SAINC4 data: %s", national_cache)
        return pd.read_parquet(national_cache)

    line_codes = [LINE_PERSONAL_INCOME, LINE_WAGES_SALARIES, LINE_DIVIDENDS_INTEREST, LINE_TRANSFERS]

    frames: list[pd.DataFrame] = []
    for lc in line_codes:
        lc_df = _fetch_sainc4_one_linecode(lc, year, api_key)
        frames.append(lc_df)

    if not any(len(f) > 0 for f in frames):
        log.warning("BEA returned no SAINC4 data for year=%d", year)
        return pd.DataFrame(columns=["GeoFips", "GeoName", "LineCode", "TimePeriod", "DataValue"])

    national = pd.concat(frames, ignore_index=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    national.to_parquet(national_cache, index=False)
    n_states = national["GeoFips"].nunique() if len(national) > 0 else 0
    log.info("Saved SAINC4 national cache -> %s (%d rows, %d states)", national_cache, len(national), n_states)
    return national


def _parse_data_value(val: str | float) -> float:
    """Convert a BEA DataValue string to float. Returns NaN for suppressed/missing values.

    BEA uses special codes: (D)=withheld, (NA)=not available, (L)=less than $50K,
    (S)=suppressed. These are treated as missing (NaN).
    """
    if pd.isna(val):
        return float("nan")
    s = str(val).strip().replace(",", "")
    if s in ("(D)", "(NA)", "(L)", "(S)", "(X)", "--", ""):
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def compute_state_income_shares(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Compute income composition shares from raw SAINC4 data.

    From total personal income and three components, computes:
      wages_share      = wages_and_salaries / personal_income
      transfer_share   = transfer_receipts / personal_income
      investment_share = dividends_interest_rent / personal_income

    States with zero or missing personal_income are excluded.

    Args:
        raw_df: DataFrame with columns GeoFips, LineCode, DataValue.

    Returns:
        DataFrame with columns: state_fips_prefix, state_abbr, wages_share,
        transfer_share, investment_share.
    """
    df = raw_df.copy()
    df["LineCode"] = pd.to_numeric(df["LineCode"], errors="coerce").astype("Int64")
    df["value"] = df["DataValue"].apply(_parse_data_value)
    # SAINC4 GeoFips is 5-digit (e.g., "01000" for Alabama, "00000" for US total).
    df["state_fips_prefix"] = df["GeoFips"].astype(str).str[:2]

    # Exclude the US national aggregate (prefix "00")
    df = df[df["state_fips_prefix"] != "00"].copy()

    pivot = df.pivot_table(index="state_fips_prefix", columns="LineCode", values="value", aggfunc="first")
    pivot.columns.name = None
    pivot = pivot.reset_index()

    line_map = {
        LINE_PERSONAL_INCOME: "personal_income",
        LINE_WAGES_SALARIES: "wages_and_salaries",
        LINE_DIVIDENDS_INTEREST: "dividends_interest_rent",
        LINE_TRANSFERS: "transfer_receipts",
    }
    pivot = pivot.rename(columns=line_map)

    for col in line_map.values():
        if col not in pivot.columns:
            pivot[col] = float("nan")

    def safe_share(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        result = pd.Series(float("nan"), index=denominator.index, dtype=float)
        valid = denominator.notna() & (denominator > 0) & numerator.notna()
        result[valid] = numerator[valid] / denominator[valid]
        return result

    pivot[COL_WAGES] = safe_share(pivot["wages_and_salaries"], pivot["personal_income"])
    pivot[COL_TRANSFER] = safe_share(pivot["transfer_receipts"], pivot["personal_income"])
    pivot[COL_INVESTMENT] = safe_share(pivot["dividends_interest_rent"], pivot["personal_income"])

    pivot = pivot[pivot["personal_income"].notna() & (pivot["personal_income"] > 0)].copy()
    pivot["state_abbr"] = pivot["state_fips_prefix"].map(STATE_ABBR)
    pivot = pivot[pivot["state_fips_prefix"].isin(STATE_ABBR)].copy()

    n_missing_abbr = pivot["state_abbr"].isna().sum()
    if n_missing_abbr > 0:
        unknown_fips = pivot[pivot["state_abbr"].isna()]["state_fips_prefix"].tolist()
        log.warning("%d state FIPS prefixes have no configured abbreviation: %s", n_missing_abbr, unknown_fips)

    result_cols = ["state_fips_prefix", "state_abbr", COL_WAGES, COL_TRANSFER, COL_INVESTMENT]
    return pivot[result_cols].reset_index(drop=True)


def fetch_bea_income_composition(year: int | None = None, force_refresh: bool = False) -> pd.DataFrame:
    """Fetch and compute BEA state-level income composition shares.

    Falls back to FALLBACK_YEAR if the primary year is unavailable or empty.
    """
    if year is None:
        year = PRIMARY_YEAR

    if force_refresh:
        for yr in (year, FALLBACK_YEAR):
            cache = RAW_DIR / f"sainc4_national_{yr}.parquet"
            if cache.exists():
                log.info("Removing cache for forced refresh: %s", cache)
                cache.unlink()

    api_key = _get_api_key()
    raw = _fetch_sainc4_national(year, api_key)

    if raw.empty:
        log.warning("No SAINC4 data for year=%d, falling back to year=%d", year, FALLBACK_YEAR)
        raw = _fetch_sainc4_national(FALLBACK_YEAR, api_key)

    result = compute_state_income_shares(raw)

    if len(result) == 0:
        log.warning("No states with valid income shares after processing year=%d", year)
    else:
        log.info(
            "BEA state income composition: %d states  wages_share mean=%.3f  transfer_share mean=%.3f  investment_share mean=%.3f",
            len(result), result[COL_WAGES].mean(), result[COL_TRANSFER].mean(), result[COL_INVESTMENT].mean(),
        )

    return result


def main() -> None:
    """Fetch BEA state income composition and save to assembled parquet."""
    df = fetch_bea_income_composition()
    ASSEMBLED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    log.info("Saved -> %s (%d states)", OUTPUT_PATH, len(df))
    for col in SHARE_COLS:
        if col in df.columns:
            log.info("  %s: mean=%.3f  min=%.3f  max=%.3f  n_valid=%d", col, df[col].mean(), df[col].min(), df[col].max(), df[col].notna().sum())


if __name__ == "__main__":
    main()
