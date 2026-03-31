"""Fetch county-level decennial census data (2000, 2010, 2020) for FL, GA, AL.

Two API calls per year:
  - Decennial (SF1/DHC) for race, age, housing
  - Supplemental (SF3/ACS5) for income, education, commute

Output:
    data/assembled/census_{year}.parquet

Reference: docs/references/data-sources/census-decennial.md
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv

from src.core import config as _cfg

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

API_KEY = os.getenv("CENSUS_API_KEY")

# State list comes from config/model.yaml (all 50+DC by default).
# fetch_year() accepts an optional states= override for targeted runs.
STATES: dict[str, str] = _cfg.STATES  # abbr → fips prefix

PROJECT_ROOT = Path(__file__).parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "assembled"
_CENSUS_API_CONFIG_PATH = PROJECT_ROOT / "data" / "config" / "census_api_config.yaml"

MAX_RETRIES = 3
RETRY_BACKOFF_S = 5

# ── Variable crosswalk per year ───────────────────────────────────────────────
# Each year has two endpoints.  For each endpoint we define:
#   base_url — full Census API base URL
#   vars     — dict mapping Census variable codes to friendly names
#              OR to None if the variable is used in a derived column
#
# Config lives in data/config/census_api_config.yaml so Census URLs and
# variable codes can be updated without touching Python source.


def _load_year_config() -> dict:
    """Load Census API config from data/config/census_api_config.yaml.

    Returns the ``years`` subtree as a plain dict keyed by integer year.
    Raises FileNotFoundError if the config file is missing — the file is
    checked into source control and should always be present.
    """
    if not _CENSUS_API_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Census API config not found: {_CENSUS_API_CONFIG_PATH}. "
            "Run git checkout or restore the file from source control."
        )
    with open(_CENSUS_API_CONFIG_PATH) as f:
        raw = yaml.safe_load(f)
    # YAML parses integer keys as ints when they are bare numbers, but be
    # explicit: ensure all year keys are ints so YEAR_CONFIG[2020] works.
    return {int(year): cfg for year, cfg in raw["years"].items()}


YEAR_CONFIG = _load_year_config()

# ── Fetch ─────────────────────────────────────────────────────────────────────


def _fetch_endpoint(
    base_url: str,
    api_vars: list[str],
    state_fips: str,
    retries: int = MAX_RETRIES,
) -> pd.DataFrame:
    """Fetch one Census API endpoint for all counties in one state.

    Returns a DataFrame with API variable columns plus 'state' and 'county'.
    Retries on failure with exponential backoff.
    """
    params: dict = {
        "get": ",".join(api_vars),
        "for": "county:*",
        "in": f"state:{state_fips}",
    }
    if API_KEY:
        params["key"] = API_KEY

    for attempt in range(retries):
        try:
            resp = requests.get(base_url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            headers, rows = data[0], data[1:]
            return pd.DataFrame(rows, columns=headers)
        except (requests.RequestException, ValueError) as exc:
            if attempt < retries - 1:
                wait = RETRY_BACKOFF_S * (attempt + 1)
                log.warning("  Retry %d/%d after %ds: %s", attempt + 1, retries, wait, exc)
                time.sleep(wait)
            else:
                raise


def _cast_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Cast API columns to float; replace Census null sentinel with NaN."""
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").replace(-666666666, float("nan"))
    return df


def _build_county_fips(df: pd.DataFrame) -> pd.DataFrame:
    """Construct 5-digit county FIPS from the state/county geo columns."""
    df["county_fips"] = df["state"].str.zfill(2) + df["county"].str.zfill(3)
    return df.drop(columns=["state", "county"])


# ── Transform ─────────────────────────────────────────────────────────────────


def _derive_housing_owner(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Compute housing_owner from year-specific columns."""
    if year in (2000, 2010):
        df["housing_owner"] = df["_housing_owner_mortgage"] + df["_housing_owner_free"]
        df = df.drop(columns=["_housing_owner_mortgage", "_housing_owner_free"])
    # 2020: housing_owner already directly mapped
    return df


def _derive_educ_bachelors_plus(df: pd.DataFrame) -> pd.DataFrame:
    """Sum 8 education cells (4 male + 4 female degree levels) into one column."""
    educ_cols = [
        "_educ_male_ba", "_educ_male_ma", "_educ_male_prof", "_educ_male_doc",
        "_educ_female_ba", "_educ_female_ma", "_educ_female_prof", "_educ_female_doc",
    ]
    df["educ_bachelors_plus"] = sum(df[c] for c in educ_cols)
    df = df.drop(columns=educ_cols)
    return df


# ── Public API ────────────────────────────────────────────────────────────────


def fetch_year(
    year: int,
    states: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Fetch decennial census data for one year across all states.

    Parameters
    ----------
    year : int
        Census year: 2000, 2010, or 2020.
    states : dict, optional
        Mapping of state abbreviation to FIPS code.  Defaults to FL+GA+AL.

    Returns
    -------
    pd.DataFrame
        Standardized DataFrame with columns defined in EXPECTED_STANDARDIZED_COLS.
    """
    if year not in YEAR_CONFIG:
        raise ValueError(f"Unsupported census year: {year}. Must be 2000, 2010, or 2020.")

    if states is None:
        states = STATES

    config = YEAR_CONFIG[year]
    primary_cfg = config["primary"]
    suppl_cfg = config["supplemental"]

    primary_vars = list(primary_cfg["vars"].keys())
    suppl_vars = list(suppl_cfg["vars"].keys())

    frames: list[pd.DataFrame] = []

    for state_abbr, state_fips in sorted(states.items()):
        log.info("  %s (FIPS %s) — year %d ...", state_abbr, state_fips, year)

        # Primary endpoint (SF1 / DHC)
        df_primary = _fetch_endpoint(primary_cfg["base_url"], primary_vars, state_fips)
        df_primary = _cast_numeric(df_primary, primary_vars)
        df_primary = df_primary.rename(columns=primary_cfg["vars"])
        df_primary = _build_county_fips(df_primary)

        # Supplemental endpoint (SF3 / ACS5)
        df_suppl = _fetch_endpoint(suppl_cfg["base_url"], suppl_vars, state_fips)
        df_suppl = _cast_numeric(df_suppl, suppl_vars)
        df_suppl = df_suppl.rename(columns=suppl_cfg["vars"])
        df_suppl = _build_county_fips(df_suppl)

        # Merge on county_fips
        df = df_primary.merge(df_suppl, on="county_fips", how="outer")
        frames.append(df)
        time.sleep(0.5)  # polite rate limiting

    combined = pd.concat(frames, ignore_index=True)

    # Derive composite columns
    combined = _derive_housing_owner(combined, year)
    combined = _derive_educ_bachelors_plus(combined)

    # Add year column
    combined["year"] = year

    log.info("Year %d: %d counties x %d columns", year, len(combined), len(combined.columns))
    return combined


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    """Fetch all three decennial census years and save to parquet."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for year in (2000, 2010, 2020):
        log.info("Fetching decennial census %d ...", year)
        df = fetch_year(year)
        out_path = OUTPUT_DIR / f"census_{year}.parquet"
        df.to_parquet(out_path, index=False)
        log.info("Saved -> %s (%d rows)", out_path, len(df))


if __name__ == "__main__":
    main()
