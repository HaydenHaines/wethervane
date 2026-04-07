"""Build county-level migration features from IRS SOI county-to-county migration edge data.

Reads data/raw/irs_migration.parquet (edge list with columns: origin_fips, dest_fips,
n_returns, n_exemptions, agi, year_pair) and outputs
data/assembled/county_migration_features.parquet with:

    net_migration_rate      (inflow - outflow) / inflow; positive = growing county
    avg_inflow_income       total_inflow_agi / total_inflow_returns (in $1,000s, IRS units)
    migration_diversity     number of unique origin counties sending migrants here
    inflow_outflow_ratio    inflow / (inflow + outflow); in [0, 1], >0.5 = net inflow

Scope: All 50 states + DC (national coverage, ~3,100+ counties).
Origin/destination counties may be anywhere in the US.
If multiple year_pairs are present the per-year features are averaged for stability.

IRS note: n_returns == -1 is a suppression sentinel (fewer than 10 returns).  These rows
are excluded from all aggregations because they carry no reliable volume signal.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.core import config as _cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "irs_migration.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "assembled" / "county_migration_features.parquet"

# FIPS prefixes for all target states (national: all 50 states + DC)
TARGET_PREFIXES: tuple[str, ...] = tuple(_cfg.STATES.values())

# Small positive floor to prevent division by zero for pure-outflow counties
_INFLOW_FLOOR = 1.0


def _is_target(fips_series: pd.Series) -> pd.Series:
    """Return boolean mask: True where the 5-digit FIPS is in FL/GA/AL."""
    return fips_series.str[:2].isin(TARGET_PREFIXES)


def _suppress_sentinel(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where n_returns == -1 (IRS suppression sentinel)."""
    mask = df["n_returns"] == -1
    n_dropped = mask.sum()
    if n_dropped:
        log.info("  Dropping %d suppressed rows (n_returns == -1)", n_dropped)
    return df[~mask].copy()


def build_features_for_year(df_year: pd.DataFrame) -> pd.DataFrame:
    """Compute migration features for a single year_pair slice.

    Parameters
    ----------
    df_year:
        Edge list for one year_pair, with suppressed rows already removed.
        Columns: origin_fips, dest_fips, n_returns, agi.

    Returns
    -------
    DataFrame indexed by county_fips (FL/GA/AL only) with four feature columns.
    """
    # -- Inflows: rows where dest_fips is a target county -------------------
    inflow_mask = _is_target(df_year["dest_fips"])
    inflows = df_year[inflow_mask].copy()

    inflow_agg = (
        inflows.groupby("dest_fips")
        .agg(
            inflow_returns=("n_returns", "sum"),
            inflow_agi=("agi", "sum"),
            inflow_unique_origins=("origin_fips", "nunique"),
        )
        .rename_axis("county_fips")
    )

    # -- Outflows: rows where origin_fips is a target county ----------------
    outflow_mask = _is_target(df_year["origin_fips"])
    outflows = df_year[outflow_mask].copy()

    outflow_agg = (
        outflows.groupby("origin_fips")
        .agg(outflow_returns=("n_returns", "sum"))
        .rename_axis("county_fips")
    )

    # -- Merge inflow and outflow on county_fips ----------------------------
    # outer join so counties that appear only as origin or only as destination are retained
    combined = inflow_agg.join(outflow_agg, how="outer")

    # Fill counties with no observed inflow or outflow with 0
    combined["inflow_returns"] = combined["inflow_returns"].fillna(0.0)
    combined["outflow_returns"] = combined["outflow_returns"].fillna(0.0)
    combined["inflow_agi"] = combined["inflow_agi"].fillna(0.0)
    combined["inflow_unique_origins"] = combined["inflow_unique_origins"].fillna(0.0)

    # Limit to target-state counties (the outer join may include non-target counties
    # if an edge appears with a non-target origin that also appears as a non-target dest)
    target_mask = pd.Series(combined.index).str[:2].isin(TARGET_PREFIXES).values
    combined = combined[target_mask].copy()

    # -- Compute features ---------------------------------------------------
    # Apply inflow floor so pure-outflow counties don't divide by zero
    safe_inflow = combined["inflow_returns"].clip(lower=_INFLOW_FLOOR)
    total_volume = combined["inflow_returns"] + combined["outflow_returns"]

    combined["net_migration_rate"] = (
        (combined["inflow_returns"] - combined["outflow_returns"]) / safe_inflow
    )
    combined["avg_inflow_income"] = combined["inflow_agi"] / safe_inflow
    combined["migration_diversity"] = combined["inflow_unique_origins"]
    combined["inflow_outflow_ratio"] = combined["inflow_returns"] / total_volume.clip(lower=_INFLOW_FLOOR)

    return combined[
        ["net_migration_rate", "avg_inflow_income", "migration_diversity", "inflow_outflow_ratio"]
    ]


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Compute migration features averaged across all available year_pairs.

    Parameters
    ----------
    raw:
        Full IRS migration edge list (all year_pairs).

    Returns
    -------
    DataFrame with county_fips and four feature columns.
    """
    raw = _suppress_sentinel(raw)

    year_pairs = sorted(raw["year_pair"].unique())
    log.info("  year_pairs found: %s", year_pairs)

    per_year: list[pd.DataFrame] = []
    for yp in year_pairs:
        df_year = raw[raw["year_pair"] == yp].copy()
        log.info("  Processing year_pair %s (%d edges)", yp, len(df_year))
        features_yp = build_features_for_year(df_year)
        per_year.append(features_yp)

    if not per_year:
        raise ValueError("No year_pair data found after filtering — cannot build features.")

    # Stack and average across year_pairs
    stacked = pd.concat(per_year)
    averaged = stacked.groupby(stacked.index).mean()
    averaged.index.name = "county_fips"
    return averaged.reset_index()


def main() -> None:
    log.info("Loading IRS migration edge list from %s", INPUT_PATH)
    raw = pd.read_parquet(INPUT_PATH)
    log.info("  %d edges, %d columns, %d year_pairs",
             len(raw), len(raw.columns), raw["year_pair"].nunique())

    features = build_features(raw)

    n_counties = len(features)
    n_na = features.isnull().any(axis=1).sum()
    log.info(
        "Built %d county feature rows x %d feature columns | %d counties with at least one NaN",
        n_counties,
        len(features.columns) - 1,  # exclude county_fips
        n_na,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUTPUT_PATH, index=False)
    log.info("Saved -> %s", OUTPUT_PATH)

    # Sanity check: print top/bottom 5 by net_migration_rate
    sorted_df = features.sort_values("net_migration_rate", ascending=False)
    print("\n--- Top 5 net_migration_rate counties (fastest growing) ---")
    print(sorted_df.head(5)[["county_fips", "net_migration_rate", "avg_inflow_income",
                               "migration_diversity", "inflow_outflow_ratio"]].to_string(index=False))
    print("\n--- Bottom 5 net_migration_rate counties (fastest shrinking) ---")
    print(sorted_df.tail(5)[["county_fips", "net_migration_rate", "avg_inflow_income",
                               "migration_diversity", "inflow_outflow_ratio"]].to_string(index=False))


if __name__ == "__main__":
    main()
