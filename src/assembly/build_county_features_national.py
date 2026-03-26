"""Build a consolidated national county features file joining ACS + RCMS.

Reads:
    data/assembled/county_acs_features.parquet   (3,144 counties × 15 cols)
    data/assembled/county_rcms_features.parquet  (3,141 counties × 8 cols)

Outputs:
    data/assembled/county_features_national.parquet  (3,100+ counties × ~20 cols)

Derived ACS features (from build_county_acs_features.py):
    pct_white_nh, pct_black, pct_asian, pct_hispanic
    median_age, median_hh_income, log_median_hh_income
    pct_bachelors_plus, pct_graduate
    pct_owner_occupied, pct_wfh, pct_transit, pct_management
    pop_total

RCMS features (joined on county_fips):
    evangelical_share, mainline_share, catholic_share, black_protestant_share
    congregations_per_1000, religious_adherence_rate

Counties missing RCMS data (~11) are imputed with state-level medians for
each RCMS feature, consistent with the strategy in build_features.py.

FIPS format: string "01001" throughout (zero-padded, 5 characters).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[2]
ACS_PATH = PROJECT_ROOT / "data" / "assembled" / "county_acs_features.parquet"
RCMS_PATH = PROJECT_ROOT / "data" / "assembled" / "county_rcms_features.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "assembled" / "county_features_national.parquet"

RCMS_FEATURE_COLS = [
    "evangelical_share",
    "mainline_share",
    "catholic_share",
    "black_protestant_share",
    "congregations_per_1000",
    "religious_adherence_rate",
]


def _impute_state_medians(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Impute NaN values in cols using state-level medians (state derived from county_fips).

    Falls back to national median for counties whose entire state has NaN values
    (e.g., Connecticut 2022 planning regions which have no RCMS match because
    RCMS still uses the pre-2022 county structure).
    """
    df = df.copy()
    state_fips = df["county_fips"].str[:2]

    for col in cols:
        n_missing = df[col].isna().sum()
        if n_missing == 0:
            continue

        state_medians = df.groupby(state_fips)[col].median()
        mask = df[col].isna()
        df.loc[mask, col] = state_fips[mask].map(state_medians)

        # Fall back to national median for counties where state median is also NaN
        still_missing = df[col].isna()
        if still_missing.any():
            national_median = df[col].median()
            df.loc[still_missing, col] = national_median
            log.info(
                "  %-35s  %d NaN → state imputed, then %d → national median (%.3f)",
                col,
                n_missing,
                still_missing.sum(),
                national_median,
            )
        else:
            n_remaining = df[col].isna().sum()
            log.info(
                "  %-35s  %d NaN → imputed %d, remaining %d",
                col,
                n_missing,
                n_missing - n_remaining,
                n_remaining,
            )

    return df


def build_national_features(
    acs: pd.DataFrame,
    rcms: pd.DataFrame,
) -> pd.DataFrame:
    """Join ACS and RCMS county features into a single national DataFrame.

    Parameters
    ----------
    acs:
        Output of build_county_acs_features.py — county_fips + 14 derived ACS cols.
    rcms:
        Output of build_features.py RCMS section — county_fips + state_abbr + 6 RCMS cols.

    Returns
    -------
    DataFrame with county_fips, pop_total, 13 ACS ratio features, and 6 RCMS features.
    Missing RCMS values are imputed with state-level medians.
    """
    # Validate FIPS format
    if not acs["county_fips"].str.len().eq(5).all():
        raise ValueError("ACS county_fips must be 5-char zero-padded strings")
    if not rcms["county_fips"].str.len().eq(5).all():
        raise ValueError("RCMS county_fips must be 5-char zero-padded strings")

    # Left join: keep all ACS counties, bring in RCMS where available
    rcms_cols = ["county_fips"] + RCMS_FEATURE_COLS
    merged = acs.merge(rcms[rcms_cols], on="county_fips", how="left")

    n_missing_rcms = merged[RCMS_FEATURE_COLS[0]].isna().sum()
    if n_missing_rcms > 0:
        log.info(
            "%d counties lack RCMS data — imputing with state-level medians",
            n_missing_rcms,
        )
        merged = _impute_state_medians(merged, RCMS_FEATURE_COLS)

    return merged.reset_index(drop=True)


def main() -> None:
    log.info("Loading ACS county features from %s", ACS_PATH)
    acs = pd.read_parquet(ACS_PATH)
    log.info("  %d counties × %d cols", len(acs), len(acs.columns))

    log.info("Loading RCMS county features from %s", RCMS_PATH)
    rcms = pd.read_parquet(RCMS_PATH)
    log.info("  %d counties × %d cols", len(rcms), len(rcms.columns))

    features = build_national_features(acs, rcms)

    n_na_any = features.isnull().any(axis=1).sum()
    log.info(
        "Built %d county feature rows × %d columns | %d counties with ≥1 NaN",
        len(features),
        len(features.columns),
        n_na_any,
    )

    if n_na_any > 0:
        log.warning("Columns with remaining NaN:")
        for col in features.columns:
            n = features[col].isna().sum()
            if n > 0:
                log.warning("  %-35s  %d NaN", col, n)

    # Sanity check: FIPS format
    bad_fips = features[~features["county_fips"].str.match(r"^\d{5}$")]
    if len(bad_fips):
        log.error("%d rows with malformed county_fips: %s", len(bad_fips), bad_fips["county_fips"].tolist()[:5])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUTPUT_PATH, index=False)
    log.info("Saved → %s", OUTPUT_PATH)

    # Summary stats for key features
    log.info("\nFeature summary (all counties):")
    summary_cols = ["pct_white_nh", "pct_black", "pct_hispanic", "pct_bachelors_plus",
                    "median_hh_income", "evangelical_share", "catholic_share"]
    for col in summary_cols:
        if col not in features.columns:
            continue
        q1, med, q3 = features[col].quantile([0.25, 0.5, 0.75])
        log.info("  %-30s  Q1=%.3f  median=%.3f  Q3=%.3f", col, q1, med, q3)


if __name__ == "__main__":
    main()
