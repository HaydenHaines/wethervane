"""
Stage 1 feature engineering: compute county-level COVID vaccination features.

Takes the latest-snapshot vaccination data from fetch_covid_vaccination.py
and produces a clean feature DataFrame suitable for community characterization
and correlation with partisan lean.

COVID vaccination rates are one of the strongest county-level correlates of
partisan behavior (r > 0.8 with 2020 Trump vote share), making them a high-
value feature for community description overlays.

Scope: ALL US counties (3,000+ counties across all 50 states + DC).
Imputation uses state-level medians derived from county_fips[:2].

Features computed (3 total):
  vax_complete_pct  : Series_Complete_Pop_Pct — fully vaccinated (2-dose or J&J)
  vax_booster_pct   : Booster_Doses_Vax_Pct — boosted as % of fully vaccinated
  vax_dose1_pct     : Administered_Dose1_Pop_Pct — at-least-one-dose %

NaN handling:
  - Counties with CDC-suppressed or missing vaccination data retain NaN values.
  - Downstream imputation uses state-level medians (same strategy as RCMS and ACS).

Input:  data/raw/covid_vaccination.parquet
Output: data/assembled/county_covid_features.parquet
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "covid_vaccination.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "assembled" / "county_covid_features.parquet"

# COVID feature column names in the output
COVID_FEATURE_COLS = [
    "vax_complete_pct",
    "vax_booster_pct",
    "vax_dose1_pct",
]

# Mapping from raw CDC column names to our feature names
_CDC_TO_FEATURE = {
    "series_complete_pop_pct": "vax_complete_pct",
    "booster_doses_vax_pct": "vax_booster_pct",
    "administered_dose1_pop_pct": "vax_dose1_pct",
}


def compute_covid_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute COVID vaccination features from the raw CDC snapshot DataFrame.

    Renames CDC columns to canonical feature names and clips percentages to
    [0, 100] (CDC data occasionally contains fractional values just above 100
    due to population denominator updates).

    Args:
        df: DataFrame from data/raw/covid_vaccination.parquet with columns:
            county_fips, state_abbr, series_complete_pop_pct,
            booster_doses_vax_pct, administered_dose1_pop_pct

    Returns:
        DataFrame with columns: county_fips, state_abbr, + COVID_FEATURE_COLS
        One row per county.
    """
    if df.empty:
        return pd.DataFrame(columns=["county_fips", "state_abbr"] + COVID_FEATURE_COLS)

    result = pd.DataFrame({"county_fips": df["county_fips"], "state_abbr": df["state_abbr"]})

    for cdc_col, feat_col in _CDC_TO_FEATURE.items():
        if cdc_col in df.columns:
            values = pd.to_numeric(df[cdc_col], errors="coerce")
            # Clip to [0, 100] — CDC can have slight over-100 values from pop denominator updates
            result[feat_col] = values.clip(lower=0, upper=100)
        else:
            log.warning("  Column '%s' not found in input; setting '%s' to NaN", cdc_col, feat_col)
            result[feat_col] = float("nan")

    return result.reset_index(drop=True)


def impute_covid_state_medians(df: pd.DataFrame) -> pd.DataFrame:
    """Impute NaN COVID feature values with state-level medians.

    State is derived from the first 2 digits of county_fips. Matches the
    imputation strategy used for RCMS and ACS features for consistency.

    Args:
        df: DataFrame from compute_covid_features() with county_fips, state_abbr,
            and COVID_FEATURE_COLS.

    Returns:
        DataFrame with NaN values filled by state-level medians.
    """
    df = df.copy()
    df["state_fips"] = df["county_fips"].str[:2]

    for col in COVID_FEATURE_COLS:
        n_missing = df[col].isna().sum()
        if n_missing == 0:
            continue

        state_medians = df.groupby("state_fips")[col].median()
        mask = df[col].isna()
        df.loc[mask, col] = df.loc[mask, "state_fips"].map(state_medians)

        n_remaining = df[col].isna().sum()
        log.info(
            "  COVID %-22s  %d NaN → imputed %d, remaining %d",
            col, n_missing, n_missing - n_remaining, n_remaining,
        )

    df = df.drop(columns=["state_fips"])
    return df


def main() -> None:
    """Compute COVID vaccination features from raw CDC data and save to parquet.

    Reads data/raw/covid_vaccination.parquet (produced by fetch_covid_vaccination.py),
    computes 3 vaccination features, imputes missing values with state medians,
    and saves to data/assembled/county_covid_features.parquet.
    """
    if not INPUT_PATH.exists():
        log.error(
            "COVID vaccination data not found at %s.\n"
            "Run: uv run python src/assembly/fetch_covid_vaccination.py",
            INPUT_PATH,
        )
        return

    log.info("Loading CDC COVID vaccination data from %s...", INPUT_PATH)
    raw = pd.read_parquet(INPUT_PATH)
    log.info("Loaded: %d counties × %d cols", len(raw), len(raw.columns))

    # Compute features
    features = compute_covid_features(raw)

    # Report NaN before imputation
    nan_counts = features[COVID_FEATURE_COLS].isna().sum()
    log.info("\nNaN counts before imputation:")
    for col, n in nan_counts[nan_counts > 0].items():
        pct = 100 * n / len(features)
        log.info("  %-25s  %d (%.1f%%)", col, n, pct)

    # Impute with state-level medians
    log.info("\nImputing with state-level medians...")
    features = impute_covid_state_medians(features)

    # Final NaN audit
    remaining_nan = features[COVID_FEATURE_COLS].isna().sum().sum()
    if remaining_nan > 0:
        log.warning(
            "%d NaN values remain after imputation (possibly entire state missing data)",
            remaining_nan,
        )

    # Summary
    n_counties = len(features)
    n_states = features["state_abbr"].nunique()
    state_counts = features.groupby("state_abbr").size().to_dict()
    log.info(
        "\nSummary: %d counties across %d states",
        n_counties, n_states,
    )
    for state, count in sorted(state_counts.items()):
        log.info("  %s: %d counties", state, count)

    # Feature distribution summary
    log.info("\nFeature ranges (all counties):")
    for col in COVID_FEATURE_COLS:
        q1, med, q3 = features[col].quantile([0.25, 0.5, 0.75])
        log.info("  %-25s  Q1=%.1f%%  median=%.1f%%  Q3=%.1f%%", col, q1, med, q3)

    # Final column order
    out_cols = ["county_fips", "state_abbr"] + COVID_FEATURE_COLS
    output = features[out_cols]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(OUTPUT_PATH, index=False)
    log.info(
        "\nSaved → %s  (%d rows × %d cols)",
        OUTPUT_PATH, len(output), len(output.columns),
    )


if __name__ == "__main__":
    main()
