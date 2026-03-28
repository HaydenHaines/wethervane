"""Build county-level transportation features from DOT transportation typology data.

DOT transportation data captures the built environment and mobility infrastructure
of communities. Road network structure, commuting patterns, and transit access
correlate with urban/rural character and partisan lean.

Raw data is at census tract level (GEOID = 11-digit tract FIPS). Aggregated to
county level using population-weighted means (pop_density as weight proxy).

Features computed (7 total):
  transport_pop_density          : Population density (persons/sq km)
  transport_job_density          : Job density (jobs/sq km)
  transport_intersection_density : Road intersection density per sq km
  transport_pct_local_roads      : % of road network that is local roads
  transport_broadband            : Broadband availability score (1-5)
  transport_dead_end_proportion  : Proportion of road segments that are dead ends
  transport_circuity_avg         : Average circuity of road network (1.0 = perfectly direct)

Input:  data/raw/dot_transportation_typology.csv
Output: data/assembled/transportation_features.parquet
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "dot_transportation_typology.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "assembled" / "transportation_features.parquet"

# Mapping from raw column names to feature names
_RAW_TO_FEATURE = {
    "pop_density": "transport_pop_density",
    "job_density": "transport_job_density",
    "intersection_density_km": "transport_intersection_density",
    "pct_local_roads": "transport_pct_local_roads",
    "broadband": "transport_broadband",
    "dead_end_proportion": "transport_dead_end_proportion",
    "circuity_avg": "transport_circuity_avg",
}

TRANSPORT_FEATURE_COLS = list(_RAW_TO_FEATURE.values())


def build_transportation_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate tract-level DOT data to county-level transportation features.

    Uses population-density-weighted means for continuous features.
    Counties are derived from the first 5 digits of the 11-digit tract GEOID.

    Parameters
    ----------
    raw:
        Tract-level DOT data with columns: GEOID, pop_density, job_density,
        intersection_density_km, pct_local_roads, broadband,
        dead_end_proportion, circuity_avg, etc.

    Returns
    -------
    DataFrame with county_fips + TRANSPORT_FEATURE_COLS.
    """
    df = raw.copy()

    # Extract county FIPS from 11-digit tract GEOID
    df["GEOID"] = df["GEOID"].astype(str).str.zfill(11)
    df["county_fips"] = df["GEOID"].str[:5]

    # Filter to valid county FIPS
    df = df[df["county_fips"].str.match(r"^\d{5}$", na=False)].copy()
    df = df[df["county_fips"].str[2:] != "000"].copy()

    # Parse numeric columns
    raw_cols = list(_RAW_TO_FEATURE.keys())
    for col in raw_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Population-weighted aggregation to county level
    # Use pop_density as the weight (higher-pop tracts matter more)
    weight_col = "pop_density"
    df["_weight"] = df[weight_col].clip(lower=0.01)  # Avoid zero weights

    results = []
    for county_fips, group in df.groupby("county_fips"):
        row = {"county_fips": county_fips}
        w = group["_weight"].values
        w_sum = w.sum()
        if w_sum == 0:
            w = np.ones(len(group))
            w_sum = len(group)

        for raw_col, feat_col in _RAW_TO_FEATURE.items():
            if raw_col in group.columns:
                vals = group[raw_col].values
                mask = ~np.isnan(vals)
                if mask.any():
                    row[feat_col] = np.average(vals[mask], weights=w[mask])
                else:
                    row[feat_col] = float("nan")
            else:
                row[feat_col] = float("nan")
        results.append(row)

    result = pd.DataFrame(results)
    log.info("Transportation features: %d counties × %d features", len(result), len(TRANSPORT_FEATURE_COLS))
    return result


def main() -> None:
    if not INPUT_PATH.exists():
        log.error("DOT transportation data not found at %s", INPUT_PATH)
        return

    log.info("Loading DOT transportation typology from %s", INPUT_PATH)
    raw = pd.read_csv(INPUT_PATH)
    log.info("  %d tracts × %d cols", len(raw), len(raw.columns))

    features = build_transportation_features(raw)

    # Summary
    for col in TRANSPORT_FEATURE_COLS:
        vals = features[col].dropna()
        if len(vals) > 0:
            q1, med, q3 = vals.quantile([0.25, 0.5, 0.75])
            log.info("  %-35s  Q1=%.3f  median=%.3f  Q3=%.3f", col, q1, med, q3)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUTPUT_PATH, index=False)
    log.info("Saved → %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
