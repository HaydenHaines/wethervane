"""Build county-level VA disability features from raw VA compensation data.

VA disability compensation rates capture military/veteran community presence and
economic dependency on federal transfers -- both strong correlates of partisan lean
in rural and exurban counties.

Features computed (3 total):
  va_disability_per_1000     : Total disability compensation recipients per 1,000 population
  va_disability_pct_100rated : % of recipients with 100% SCD rating (severe disability)
  va_disability_pct_young    : % of recipients aged 17-44 (post-9/11 veteran cohort)

Input:  data/raw/va_disability_county.csv
Output: data/assembled/va_disability_features.parquet

Population data sourced from county_acs_features.parquet (pop_total column).
Counties without ACS population data are excluded.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "va_disability_county.csv"
ACS_PATH = PROJECT_ROOT / "data" / "assembled" / "county_acs_features.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "assembled" / "va_disability_features.parquet"

VA_FEATURE_COLS = [
    "va_disability_per_1000",
    "va_disability_pct_100rated",
    "va_disability_pct_young",
]


def build_va_features(raw: pd.DataFrame, acs: pd.DataFrame) -> pd.DataFrame:
    """Compute VA disability features from raw compensation data.

    Parameters
    ----------
    raw:
        Raw VA disability CSV with columns: FIPS code, Total: Disability
        Compensation Recipients, SCD rating: 100%, Age: 17-44, etc.
    acs:
        ACS features with county_fips and pop_total columns for normalization.

    Returns
    -------
    DataFrame with county_fips + VA_FEATURE_COLS. One row per county.
    """
    df = raw.copy()
    df = df.rename(columns={"FIPS code": "county_fips"})
    df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)

    # Filter to valid 5-digit county FIPS
    df = df[df["county_fips"].str.match(r"^\d{5}$", na=False)].copy()
    df = df[df["county_fips"].str[2:] != "000"].copy()

    # Parse numeric columns
    total_col = "Total: Disability Compensation Recipients"
    rated_100_col = "SCD rating: 100%"
    young_col = "Age: 17-44"

    df["total_recipients"] = pd.to_numeric(df[total_col], errors="coerce")
    df["rated_100"] = pd.to_numeric(df[rated_100_col], errors="coerce")
    df["young"] = pd.to_numeric(df[young_col], errors="coerce")

    # Join population from ACS
    pop = acs[["county_fips", "pop_total"]].copy()
    df = df.merge(pop, on="county_fips", how="inner")

    # Compute features
    df["va_disability_per_1000"] = (
        df["total_recipients"] / df["pop_total"].replace(0, float("nan")) * 1000
    )
    df["va_disability_pct_100rated"] = (
        df["rated_100"] / df["total_recipients"].replace(0, float("nan")) * 100
    )
    df["va_disability_pct_young"] = (
        df["young"] / df["total_recipients"].replace(0, float("nan")) * 100
    )

    result = df[["county_fips"] + VA_FEATURE_COLS].reset_index(drop=True)
    log.info("VA features: %d counties × %d features", len(result), len(VA_FEATURE_COLS))
    return result


def main() -> None:
    if not INPUT_PATH.exists():
        log.error("VA disability data not found at %s", INPUT_PATH)
        return
    if not ACS_PATH.exists():
        log.error("ACS features not found at %s — needed for population normalization", ACS_PATH)
        return

    log.info("Loading VA disability data from %s", INPUT_PATH)
    raw = pd.read_csv(INPUT_PATH)
    log.info("  %d rows × %d cols", len(raw), len(raw.columns))

    log.info("Loading ACS features from %s", ACS_PATH)
    acs = pd.read_parquet(ACS_PATH)

    features = build_va_features(raw, acs)

    # Summary
    for col in VA_FEATURE_COLS:
        vals = features[col].dropna()
        if len(vals) > 0:
            q1, med, q3 = vals.quantile([0.25, 0.5, 0.75])
            log.info("  %-30s  Q1=%.2f  median=%.2f  Q3=%.2f", col, q1, med, q3)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUTPUT_PATH, index=False)
    log.info("Saved → %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
