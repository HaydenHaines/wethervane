"""Fetch county-level ACS broadband / internet access data (B28002 series).

Source: Census ACS 5-year 2022 estimates, table B28002
        "Presence and Types of Internet Subscriptions in Household"
URL:    https://api.census.gov/data/2022/acs/acs5

This captures the *information environment* of each county: how connected
households are to high-speed internet, which is structurally correlated with
political information consumption but largely orthogonal to income or
education after conditioning on those variables.

**Variables fetched (estimates only):**

    B28002_001E  total_households        — universe denominator
    B28002_002E  with_internet_sub       — any internet subscription
    B28002_004E  with_broadband          — broadband of any type
    B28002_007E  with_cable_fiber_dsl    — cable / fiber optic / DSL
    B28002_009E  with_satellite          — satellite internet
    B28002_013E  no_internet             — no internet access at all

**Derived features (computed in build_acs_broadband_features.py):**

    pct_broadband      = with_broadband / total_households
    pct_no_internet    = no_internet    / total_households
    pct_satellite      = with_satellite / total_households
    pct_cable_fiber    = with_cable_fiber_dsl / total_households
    broadband_gap      = pct_no_internet  (alias; high = underserved)

**NaN handling:**
  - Census returns -666666666 as the null sentinel → replaced with NaN.
  - State-median imputation applied in build_acs_broadband_features.py.

**CLI:**
  --force  Re-download even if the output file already exists.

Output: data/raw/acs_broadband/acs_broadband_2022.parquet
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from src.core import config as _cfg

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "acs_broadband"
OUTPUT_PATH = OUTPUT_DIR / "acs_broadband_2022.parquet"

ACS_BASE_URL = "https://api.census.gov/data/2022/acs/acs5"
ACS_YEAR = 2022

# Census ACS null sentinel (returned as string in JSON rows)
_ACS_NULL_SENTINEL = -666666666

# State list from central config (abbreviation → 2-digit FIPS)
STATES: dict[str, str] = _cfg.STATES  # abbr → fips prefix

# ACS B28002 broadband variables (estimate codes only)
BROADBAND_VARS: dict[str, str] = {
    "B28002_001E": "total_households",
    "B28002_002E": "with_internet_sub",
    "B28002_004E": "with_broadband",
    "B28002_007E": "with_cable_fiber_dsl",
    "B28002_009E": "with_satellite",
    "B28002_013E": "no_internet",
}

# Polite inter-request delay (seconds)
_REQUEST_DELAY = 0.3
_REQUEST_TIMEOUT = 60


def _fetch_state(state_fips: str, api_key: str | None) -> pd.DataFrame:
    """Fetch B28002 variables for all counties in one state.

    Returns a raw DataFrame with API code columns + 'state'/'county' geo cols.
    """
    api_vars = list(BROADBAND_VARS.keys())
    params: dict[str, str] = {
        "get": ",".join(api_vars),
        "for": "county:*",
        "in": f"state:{state_fips}",
    }
    if api_key:
        params["key"] = api_key

    resp = requests.get(ACS_BASE_URL, params=params, timeout=_REQUEST_TIMEOUT)
    resp.raise_for_status()

    data = resp.json()
    headers, rows = data[0], data[1:]
    return pd.DataFrame(rows, columns=headers)


def _cast_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce ACS value columns to float, replacing null sentinel with NaN."""
    for col in BROADBAND_VARS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").replace(
                _ACS_NULL_SENTINEL, float("nan")
            )
    return df


def fetch_national(api_key: str | None = None) -> pd.DataFrame:
    """Download B28002 for all 50 states + DC and return a combined DataFrame.

    Columns returned: county_fips + raw ACS estimate columns (renamed to
    friendly names from BROADBAND_VARS) + data_year.
    """
    frames: list[pd.DataFrame] = []
    state_items = list(STATES.items())
    log.info(
        "Fetching ACS B28002 broadband data for %d states (key=%s)",
        len(state_items),
        "set" if api_key else "unset (anonymous)",
    )

    for i, (abbr, fips) in enumerate(state_items, start=1):
        log.info("  [%d/%d] %s (FIPS %s)…", i, len(state_items), abbr, fips)
        try:
            raw = _fetch_state(fips, api_key)
        except requests.RequestException as exc:
            log.warning("    State %s failed: %s — skipping", abbr, exc)
            continue

        raw = _cast_and_clean(raw)

        # Build 5-digit FIPS
        raw["county_fips"] = raw["state"].str.zfill(2) + raw["county"].str.zfill(3)

        # Rename API codes to friendly column names
        raw = raw.rename(columns=BROADBAND_VARS)

        # Keep only needed columns
        keep = ["county_fips"] + list(BROADBAND_VARS.values())
        raw = raw[[c for c in keep if c in raw.columns]]

        frames.append(raw)
        if i < len(state_items):
            time.sleep(_REQUEST_DELAY)

    if not frames:
        raise RuntimeError("No state broadband data could be fetched.")

    combined = pd.concat(frames, ignore_index=True)
    combined["data_year"] = ACS_YEAR

    # Validate FIPS format
    bad = ~combined["county_fips"].str.match(r"^\d{5}$", na=False)
    if bad.any():
        log.warning("Dropping %d rows with malformed county_fips", bad.sum())
        combined = combined[~bad]

    log.info(
        "Fetched %d counties across %d states",
        len(combined),
        combined["county_fips"].str[:2].nunique(),
    )
    return combined.reset_index(drop=True)


def main(force: bool = False) -> None:
    """Download ACS B28002 broadband data and save to parquet.

    Idempotent: skips download when output exists unless --force.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists() and not force:
        log.info(
            "ACS broadband already at %s — skipping (pass --force to re-download)",
            OUTPUT_PATH,
        )
        return

    api_key = os.getenv("CENSUS_API_KEY")
    df = fetch_national(api_key)

    log.info("\nBroadband data summary (%d counties):", len(df))
    for col in BROADBAND_VARS.values():
        if col in df.columns:
            n_nan = df[col].isna().sum()
            pct_nan = 100 * n_nan / len(df)
            log.info("  %-25s  NaN: %d (%.1f%%)", col, n_nan, pct_nan)

    df.to_parquet(OUTPUT_PATH, index=False)
    log.info("Saved → %s  (%d rows × %d cols)", OUTPUT_PATH, len(df), len(df.columns))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch ACS B28002 broadband/internet access data (national county-level)."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-download even if output file already exists",
    )
    args = parser.parse_args()
    main(force=args.force)
