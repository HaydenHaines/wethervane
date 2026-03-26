"""Build county-level broadband connectivity features from ACS B28002 data.

Reads:
    data/raw/acs_broadband/acs_broadband_2022.parquet
        (raw ACS estimates: total_households, with_broadband, no_internet, …)

Outputs:
    data/assembled/county_broadband_features.parquet

**Derived features:**

    pct_broadband       % of households with broadband of any type.
                        Captures general connectivity level.

    pct_no_internet     % of households with no internet access at all.
                        The "digital desert" measure; highest in rural
                        and poor counties. Strongly orthogonal to income
                        after controlling for it (captures infrastructure
                        access constraints, not just ability to pay).

    pct_satellite       % of households with satellite internet.
                        Satellite = geographic isolation proxy; high in
                        counties where cable/fiber can't reach.

    pct_cable_fiber     % of households with cable, fiber optic, or DSL.
                        High-quality fixed broadband; correlated with
                        urban/suburban infrastructure investment.

    broadband_gap       Alias for pct_no_internet (1 − pct_broadband
                        captures partial subscriptions too; kept separate
                        for interpretability). Higher = more underserved.

NaN handling:
    - Counties with missing raw values retain NaN derived features.
    - State-median imputation is applied here so the output is NaN-free
      before joining into county_features_national.parquet.
    - National median fallback for counties whose state is entirely NaN
      (e.g., Connecticut 2022 planning regions that don't map to
       pre-2022 Census county geography).

FIPS format: 5-char zero-padded string throughout.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "acs_broadband" / "acs_broadband_2022.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "assembled" / "county_broadband_features.parquet"

# Raw columns needed to derive features
_RAW_COLS = [
    "county_fips",
    "total_households",
    "with_broadband",
    "with_cable_fiber_dsl",
    "with_satellite",
    "no_internet",
]

# Output feature columns (in output order)
BROADBAND_FEATURE_COLS = [
    "pct_broadband",
    "pct_no_internet",
    "pct_satellite",
    "pct_cable_fiber",
    "broadband_gap",
]


def build_broadband_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Derive broadband features from raw ACS B28002 estimates.

    Parameters
    ----------
    raw:
        DataFrame from fetch_acs_broadband.py, with columns:
        county_fips, total_households, with_broadband,
        with_cable_fiber_dsl, with_satellite, no_internet.

    Returns
    -------
    DataFrame with county_fips + BROADBAND_FEATURE_COLS.
    Missing values in raw imply NaN in derived features; callers should
    apply imputation as needed.
    """
    if not raw["county_fips"].str.len().eq(5).all():
        raise ValueError("county_fips must be 5-char zero-padded strings")

    if "total_households" not in raw.columns:
        raise ValueError("Raw data missing 'total_households' column")

    df = raw[_RAW_COLS].copy()

    # Denominator: total households in universe; protect against divide-by-zero
    denom = df["total_households"].replace(0, float("nan"))

    df["pct_broadband"] = df["with_broadband"] / denom
    df["pct_no_internet"] = df["no_internet"] / denom
    df["pct_satellite"] = df["with_satellite"] / denom
    df["pct_cable_fiber"] = df["with_cable_fiber_dsl"] / denom

    # broadband_gap = fraction without any internet; same as pct_no_internet
    # but kept as a named signal for ensemble feature importance
    df["broadband_gap"] = df["pct_no_internet"]

    # Clip ratios to [0, 1] — rounding in ACS can produce tiny negatives
    for col in BROADBAND_FEATURE_COLS:
        df[col] = df[col].clip(lower=0.0, upper=1.0)

    return df[["county_fips"] + BROADBAND_FEATURE_COLS].reset_index(drop=True)


def _impute_state_medians(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Impute NaN values with state-level medians, falling back to national median.

    Mirrors the same strategy used in build_county_features_national.py so
    the output is always NaN-free before joining into the national feature table.
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

        still_missing = df[col].isna()
        if still_missing.any():
            national_median = df[col].median()
            df.loc[still_missing, col] = national_median
            log.info(
                "  %-25s  %d NaN → state, then %d → national median (%.3f)",
                col,
                n_missing,
                still_missing.sum(),
                national_median,
            )
        else:
            n_remaining = df[col].isna().sum()
            log.info(
                "  %-25s  %d NaN → imputed %d, remaining %d",
                col,
                n_missing,
                n_missing - n_remaining,
                n_remaining,
            )

    return df


def main() -> None:
    """Build broadband features and write to parquet."""
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw ACS broadband data not found at {RAW_PATH}.\n"
            "Run: uv run python -m src.assembly.fetch_acs_broadband"
        )

    log.info("Loading raw ACS broadband data from %s", RAW_PATH)
    raw = pd.read_parquet(RAW_PATH)
    log.info("  %d counties × %d cols", len(raw), len(raw.columns))

    features = build_broadband_features(raw)

    # NaN audit before imputation
    n_nan = features[BROADBAND_FEATURE_COLS].isna().any(axis=1).sum()
    if n_nan > 0:
        log.info("%d counties have ≥1 NaN feature — imputing with state medians", n_nan)
        features = _impute_state_medians(features, BROADBAND_FEATURE_COLS)

    n_still_nan = features[BROADBAND_FEATURE_COLS].isna().any(axis=1).sum()
    if n_still_nan > 0:
        log.error("%d counties still have NaN after imputation — check raw data", n_still_nan)

    # Summary statistics
    log.info("\nBroadband feature summary (%d counties):", len(features))
    for col in BROADBAND_FEATURE_COLS:
        q1, med, q3 = features[col].quantile([0.25, 0.5, 0.75])
        log.info("  %-25s  Q1=%.3f  median=%.3f  Q3=%.3f", col, q1, med, q3)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUTPUT_PATH, index=False)
    log.info("Saved → %s  (%d rows × %d cols)", OUTPUT_PATH, len(features), len(features.columns))


if __name__ == "__main__":
    main()
