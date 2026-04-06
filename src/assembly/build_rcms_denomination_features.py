"""Build RCMS denomination features for county-level integration.

Thin wrapper around fetch_rcms_denominations.py that loads the denomination
parquet and returns a clean DataFrame ready to merge into the national feature
matrix via build_county_features_national.py.

Denomination features (per 1,000 residents):
  lds_rate        — LDS adherents per 1,000 residents
  muslim_rate     — Muslim adherents per 1,000 residents
  jewish_rate     — Jewish adherents per 1,000 residents
  hindu_sikh_rate — (Hindu + Sikh-estimated) adherents per 1,000 residents

Input:  data/raw/rcms_denominations.parquet (from fetch_rcms_denominations.py)
Output: data/assembled/rcms_denomination_features.parquet
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "rcms_denominations.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "assembled" / "rcms_denomination_features.parquet"

# Feature columns produced by this builder
DENOMINATION_FEATURE_COLS = [
    "lds_rate",
    "muslim_rate",
    "jewish_rate",
    "hindu_sikh_rate",
]


def load_denomination_features() -> pd.DataFrame:
    """Load and validate denomination features from the raw parquet.

    Returns
    -------
    pd.DataFrame
        Columns: county_fips (5-char string), plus DENOMINATION_FEATURE_COLS.
        All rate columns are non-negative floats, 0.0 where no denomination data.
    """
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Denomination parquet not found at {INPUT_PATH}. "
            "Run: uv run python -m src.assembly.fetch_rcms_denominations"
        )

    df = pd.read_parquet(INPUT_PATH)
    log.info("Loaded denomination features: %d rows × %d cols", len(df), len(df.columns))

    # Validate FIPS format
    if not df["county_fips"].str.len().eq(5).all():
        raise ValueError("county_fips must be 5-char zero-padded strings in denomination features")

    # Validate expected columns
    missing_cols = [c for c in DENOMINATION_FEATURE_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Denomination parquet missing expected columns: {missing_cols}")

    return df[["county_fips"] + DENOMINATION_FEATURE_COLS].copy()


def main() -> None:
    """Load denomination features and write assembled parquet."""
    df = load_denomination_features()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    log.info("Saved → %s  (%d rows × %d cols)", OUTPUT_PATH, len(df), len(df.columns))

    # Summary
    log.info("\nFeature summary:")
    for col in DENOMINATION_FEATURE_COLS:
        nonzero = df[df[col] > 0][col]
        log.info(
            "  %-20s  %d non-zero counties | median(nonzero)=%.1f",
            col, len(nonzero), nonzero.median() if len(nonzero) > 0 else 0.0
        )


if __name__ == "__main__":
    main()
