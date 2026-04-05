"""Precinct-level validation of the county type structure.

Tests whether the KMeans county types generalize to precinct granularity.
Specifically answers:
  1. Do precincts within a county-type show consistent Dem vote share?
  2. Is between-type variance significantly larger than within-type variance?
  3. Does the county-level type prediction correlate with precinct-level outcomes?

This is a READ-ONLY validation — no model code is modified.

Data sources:
  - data/raw/nyt_precinct/precincts_2020_national.geojson.gz
    NYTimes 2020 precinct-level presidential results.
    GEOID format: "{county_fips}-{precinct_name}" — county FIPS is first 5 chars.
  - data/communities/county_type_assignments_full.parquet
    KMeans type assignments per county (dominant_type, super_type, 100 type scores).
  - data/communities/type_priors.parquet
    Type-level predicted Dem share (prior_dem_share per type_id).

Output:
  data/experiments/precinct_validation_results.json

Usage:
    uv run python scripts/experiments/validate_precinct_types.py
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Allow running from project root or from this file's directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# ── Constants ─────────────────────────────────────────────────────────────────

# Minimum total votes to include a precinct in the analysis.
# Precincts with fewer votes are statistically unreliable and often reflect
# uncontested or split-precinct reporting artifacts.
MIN_PRECINCT_VOTES = 50

# Minimum number of precincts for a county to be included in the type analysis.
# Counties with only 1 precinct provide no within-county variance signal.
MIN_PRECINCTS_PER_COUNTY = 2

# Minimum number of counties per type to compute meaningful type statistics.
# Types with too few counties produce noisy variance estimates.
MIN_COUNTIES_PER_TYPE = 3

DATA_DIR = PROJECT_ROOT / "data"
PRECINCT_PATH = DATA_DIR / "raw" / "nyt_precinct" / "precincts_2020_national.geojson.gz"
TYPE_ASSIGNMENTS_PATH = DATA_DIR / "communities" / "county_type_assignments_full.parquet"
TYPE_PRIORS_PATH = DATA_DIR / "communities" / "type_priors.parquet"
OUTPUT_PATH = DATA_DIR / "experiments" / "precinct_validation_results.json"


# ── Data loading ──────────────────────────────────────────────────────────────


def load_precinct_data(path: Path) -> pd.DataFrame:
    """Load NYTimes 2020 precinct GeoJSON and return a flat DataFrame.

    The GeoJSON stores votes_dem, votes_rep, votes_total, and pct_dem_lead
    per precinct. We derive dem_share = votes_dem / votes_total.

    County FIPS is the first 5 characters of the GEOID field, following the
    NYTimes convention of "{county_fips}-{precinct_description}".

    Precincts with missing or zero votes_total are dropped — these are
    reporting gaps, not genuine zero-turnout precincts.
    """
    with gzip.open(path, "rt") as f:
        geojson = json.load(f)

    rows = []
    for feature in geojson["features"]:
        props = feature["properties"]
        geoid = props.get("GEOID", "")
        votes_total = props.get("votes_total")
        votes_dem = props.get("votes_dem")

        # Skip precincts with missing or zero totals — reporting artifacts
        if not votes_total or votes_total <= 0:
            continue
        if votes_dem is None:
            continue

        county_fips = geoid[:5]
        # Sanity-check: county FIPS must be exactly 5 digits
        if not county_fips.isdigit():
            continue

        dem_share = votes_dem / votes_total
        rows.append({
            "county_fips": county_fips,
            "precinct_geoid": geoid,
            "votes_total": votes_total,
            "votes_dem": votes_dem,
            "dem_share": dem_share,
        })

    df = pd.DataFrame(rows)
    # Clip dem_share to [0, 1] — a handful of precincts have reporting
    # artifacts that push them marginally outside this range
    df["dem_share"] = df["dem_share"].clip(0.0, 1.0)
    return df


def load_county_type_assignments(path: Path) -> pd.DataFrame:
    """Load county-to-type mapping.

    Returns a DataFrame with county_fips, dominant_type, and super_type.
    """
    df = pd.read_parquet(path, columns=["county_fips", "dominant_type", "super_type"])
    return df


def load_type_priors(path: Path) -> pd.DataFrame:
    """Load per-type predicted Dem share (prior_dem_share).

    This is the model's predicted Dem share for each type, computed from
    the Ridge+demographics ensemble trained on 2020 presidential results.
    """
    return pd.read_parquet(path)


# ── County-level aggregation ──────────────────────────────────────────────────


def compute_county_precinct_stats(
    precincts: pd.DataFrame,
    min_precinct_votes: int = MIN_PRECINCT_VOTES,
    min_precincts_per_county: int = MIN_PRECINCTS_PER_COUNTY,
) -> pd.DataFrame:
    """Aggregate precinct data to county level.

    For each county, computes:
      - n_precincts: number of valid precincts
      - vote_weighted_dem_share: county Dem share (total votes weighted)
      - precinct_variance: variance of precinct dem_share values
      - precinct_std: standard deviation of precinct dem_share
      - total_votes: total votes across all precincts

    The vote-weighted Dem share matches what official county totals report.
    Precinct variance measures within-county spread.

    Precincts below min_precinct_votes are excluded as statistically unreliable.
    Counties with fewer than min_precincts_per_county are excluded because
    a single precinct provides no within-county variance signal.
    """
    # Apply minimum precinct vote threshold
    valid = precincts[precincts["votes_total"] >= min_precinct_votes].copy()

    # Build county-level stats
    records = []
    for county_fips, group in valid.groupby("county_fips"):
        if len(group) < min_precincts_per_county:
            continue
        total_votes = group["votes_total"].sum()
        # Vote-weighted Dem share = total Dem votes / total votes
        vote_weighted_dem_share = group["votes_dem"].sum() / total_votes
        precinct_variance = group["dem_share"].var()
        precinct_std = group["dem_share"].std()
        records.append({
            "county_fips": county_fips,
            "n_precincts": len(group),
            "total_votes": total_votes,
            "vote_weighted_dem_share": vote_weighted_dem_share,
            "precinct_variance": precinct_variance,
            "precinct_std": precinct_std,
        })

    return pd.DataFrame(records)


# ── Variance partitioning ─────────────────────────────────────────────────────


def compute_within_type_variance(
    county_stats: pd.DataFrame,
    type_assignments: pd.DataFrame,
    min_counties_per_type: int = MIN_COUNTIES_PER_TYPE,
) -> dict:
    """Partition precinct variance into within-type and between-type components.

    The core question: does the type structure explain county-level Dem share?
    If types are meaningful, between-type variance should dwarf within-type variance.

    The F-ratio (between / within) is a simple signal strength metric:
      - F >> 1: types explain most of the variance; type structure is meaningful
      - F ~= 1: types explain no more variance than random partitioning

    Within-type variance = mean of (per-type variance of county Dem share).
    Between-type variance = variance of type-mean Dem shares.

    We also compute the precinct-level within-county variance, which is a
    different signal — it tells us how homogeneous precincts are within a county,
    not whether counties cluster by type.

    Returns:
        Dict with within_type_variance, between_type_variance, f_ratio,
        n_types_included, type_stats (list of per-type summaries).
    """
    merged = county_stats.merge(type_assignments, on="county_fips", how="inner")

    type_records = []
    for type_id, group in merged.groupby("dominant_type"):
        if len(group) < min_counties_per_type:
            continue
        type_records.append({
            "type_id": int(type_id),
            "n_counties": len(group),
            "mean_dem_share": float(group["vote_weighted_dem_share"].mean()),
            "county_variance": float(group["vote_weighted_dem_share"].var()),
            "mean_precinct_variance": float(group["precinct_variance"].mean()),
        })

    type_df = pd.DataFrame(type_records)
    if type_df.empty:
        return {
            "within_type_variance": float("nan"),
            "between_type_variance": float("nan"),
            "f_ratio": float("nan"),
            "n_types_included": 0,
            "type_stats": [],
        }

    # Within-type variance: mean variance of county Dem share within each type
    within_type_variance = float(type_df["county_variance"].mean())

    # Between-type variance: variance of the type-mean Dem shares
    between_type_variance = float(type_df["mean_dem_share"].var())

    # F-ratio: ratio of between to within variance
    # High F means types capture real structure; F~1 means they don't
    if within_type_variance > 0:
        f_ratio = between_type_variance / within_type_variance
    else:
        f_ratio = float("inf")

    return {
        "within_type_variance": within_type_variance,
        "between_type_variance": between_type_variance,
        "f_ratio": f_ratio,
        "n_types_included": len(type_df),
        "type_stats": type_df.sort_values("type_id").to_dict(orient="records"),
    }


# ── Prediction correlation ────────────────────────────────────────────────────


def compute_type_prediction_correlation(
    county_stats: pd.DataFrame,
    type_assignments: pd.DataFrame,
    type_priors: pd.DataFrame,
) -> dict:
    """Correlate type-predicted Dem share with precinct-level county Dem share.

    This is the key validity check: if the county-level type prior (Ridge+demo
    ensemble trained on 2020 presidential results) predicts the vote-weighted
    county Dem share from precincts, the type structure is internally consistent.

    We use the county vote-weighted Dem share (from precinct rollup) as the
    ground truth, not the official county totals — this tests whether the
    precinct data is consistent with the county-level model inputs.

    Returns:
        Dict with pearson_r, p_value, n_counties, rmse, and median_abs_error.
    """
    # Map county -> dominant type -> type prior
    merged = county_stats.merge(type_assignments[["county_fips", "dominant_type"]], on="county_fips", how="inner")
    merged = merged.merge(
        type_priors.rename(columns={"type_id": "dominant_type", "prior_dem_share": "predicted_dem_share"}),
        on="dominant_type",
        how="inner",
    )

    if len(merged) < 10:
        return {
            "pearson_r": float("nan"),
            "p_value": float("nan"),
            "n_counties": len(merged),
            "rmse": float("nan"),
            "median_abs_error": float("nan"),
        }

    actual = merged["vote_weighted_dem_share"].values
    predicted = merged["predicted_dem_share"].values

    r, p = pearsonr(actual, predicted)
    residuals = actual - predicted
    rmse = float(np.sqrt(np.mean(residuals**2)))
    median_abs_error = float(np.median(np.abs(residuals)))

    return {
        "pearson_r": float(r),
        "p_value": float(p),
        "n_counties": int(len(merged)),
        "rmse": rmse,
        "median_abs_error": median_abs_error,
    }


# ── Within-type precinct variance analysis ───────────────────────────────────


def compute_precinct_variance_by_type(
    county_stats: pd.DataFrame,
    type_assignments: pd.DataFrame,
    min_counties_per_type: int = MIN_COUNTIES_PER_TYPE,
) -> dict:
    """Analyze how much within-county precinct variance exists by type.

    Within-county precinct variance measures how uniform precincts are within
    a county. High within-county precinct variance indicates a heterogeneous
    county (mixed urban/rural, contested, etc.).

    Types with high mean within-county precinct variance may be inherently
    more heterogeneous geographically — this is useful diagnostic information
    for the tract-primary migration (types with high within-county variance
    benefit most from sub-county granularity).

    Returns dict with overall stats and per-type breakdown.
    """
    merged = county_stats.merge(type_assignments[["county_fips", "dominant_type"]], on="county_fips", how="inner")

    overall_mean_precinct_std = float(merged["precinct_std"].mean())
    overall_median_precinct_std = float(merged["precinct_std"].median())

    type_variance_records = []
    for type_id, group in merged.groupby("dominant_type"):
        if len(group) < min_counties_per_type:
            continue
        type_variance_records.append({
            "type_id": int(type_id),
            "n_counties": int(len(group)),
            "mean_precinct_std": float(group["precinct_std"].mean()),
            "median_precinct_std": float(group["precinct_std"].median()),
            "mean_county_dem_share": float(group["vote_weighted_dem_share"].mean()),
        })

    type_variance_df = pd.DataFrame(type_variance_records)

    # Find types with highest and lowest within-county precinct variance
    if not type_variance_df.empty:
        most_heterogeneous = type_variance_df.nlargest(5, "mean_precinct_std")[
            ["type_id", "mean_precinct_std", "mean_county_dem_share", "n_counties"]
        ].to_dict(orient="records")
        most_homogeneous = type_variance_df.nsmallest(5, "mean_precinct_std")[
            ["type_id", "mean_precinct_std", "mean_county_dem_share", "n_counties"]
        ].to_dict(orient="records")
    else:
        most_heterogeneous = []
        most_homogeneous = []

    return {
        "overall_mean_precinct_std": overall_mean_precinct_std,
        "overall_median_precinct_std": overall_median_precinct_std,
        "n_counties_analyzed": int(len(merged)),
        "most_heterogeneous_types": most_heterogeneous,
        "most_homogeneous_types": most_homogeneous,
    }


# ── Super-type analysis ───────────────────────────────────────────────────────


def compute_super_type_summary(
    county_stats: pd.DataFrame,
    type_assignments: pd.DataFrame,
) -> list[dict]:
    """Compute Dem share summary per super-type (8 high-level groupings).

    Super-types are the 8 broad political communities derived from the 100
    fine-grained types via hierarchical nesting. They provide a human-readable
    lens for the validation results.

    Returns a list of dicts with super_type, n_counties, mean_dem_share,
    std_dem_share, and mean_precinct_std.
    """
    merged = county_stats.merge(type_assignments[["county_fips", "super_type"]], on="county_fips", how="inner")

    records = []
    for super_type_id, group in merged.groupby("super_type"):
        records.append({
            "super_type": int(super_type_id),
            "n_counties": int(len(group)),
            "mean_dem_share": float(group["vote_weighted_dem_share"].mean()),
            "std_dem_share": float(group["vote_weighted_dem_share"].std()),
            "mean_precinct_std": float(group["precinct_std"].mean()),
        })

    return sorted(records, key=lambda r: r["super_type"])


# ── Main pipeline ─────────────────────────────────────────────────────────────


def run_validation(
    precinct_path: Path = PRECINCT_PATH,
    type_assignments_path: Path = TYPE_ASSIGNMENTS_PATH,
    type_priors_path: Path = TYPE_PRIORS_PATH,
    output_path: Path = OUTPUT_PATH,
) -> dict:
    """Run the full precinct-level type validation pipeline.

    Steps:
      1. Load precinct GeoJSON and flatten to DataFrame
      2. Aggregate to county level (vote-weighted dem_share, precinct variance)
      3. Merge with type assignments
      4. Compute within-type vs between-type variance (F-ratio)
      5. Compute correlation between type prediction and precinct-level outcomes
      6. Compute precinct variance by type (heterogeneity diagnostic)
      7. Compute super-type summary
      8. Write results JSON

    Returns the results dict.
    """
    print("Loading precinct data...")
    precincts = load_precinct_data(precinct_path)
    print(f"  Loaded {len(precincts):,} valid precincts from {precincts['county_fips'].nunique():,} counties")

    print("Aggregating to county level...")
    county_stats = compute_county_precinct_stats(precincts)
    print(f"  {len(county_stats):,} counties with >= {MIN_PRECINCTS_PER_COUNTY} precincts and >= {MIN_PRECINCT_VOTES} votes per precinct")

    print("Loading type assignments and priors...")
    type_assignments = load_county_type_assignments(type_assignments_path)
    type_priors = load_type_priors(type_priors_path)

    n_matched = county_stats["county_fips"].isin(type_assignments["county_fips"]).sum()
    print(f"  {n_matched:,} counties matched to type assignments (of {len(county_stats):,})")

    print("Computing variance partitioning...")
    variance_partition = compute_within_type_variance(county_stats, type_assignments)
    print(f"  F-ratio (between/within type variance): {variance_partition['f_ratio']:.2f}")
    print(f"  Types included: {variance_partition['n_types_included']}")

    print("Computing type prediction correlation...")
    prediction_corr = compute_type_prediction_correlation(
        county_stats, type_assignments, type_priors
    )
    print(f"  Pearson r (type prior vs precinct-derived Dem share): {prediction_corr['pearson_r']:.3f}")
    print(f"  RMSE: {prediction_corr['rmse']:.4f}")
    print(f"  N counties: {prediction_corr['n_counties']:,}")

    print("Computing precinct variance by type...")
    precinct_variance_by_type = compute_precinct_variance_by_type(county_stats, type_assignments)
    print(f"  Mean within-county precinct std: {precinct_variance_by_type['overall_mean_precinct_std']:.4f}")

    print("Computing super-type summary...")
    super_type_summary = compute_super_type_summary(county_stats, type_assignments)

    # Build complete results dict
    results = {
        "metadata": {
            "analysis_date": "2026-04-05",
            "data_source": "NYTimes 2020 precinct-level presidential results",
            "type_model": "KMeans J=100, county-primary, Ridge+demo ensemble priors",
            "min_precinct_votes": MIN_PRECINCT_VOTES,
            "min_precincts_per_county": MIN_PRECINCTS_PER_COUNTY,
            "min_counties_per_type": MIN_COUNTIES_PER_TYPE,
        },
        "data_coverage": {
            "total_precincts_loaded": int(len(precincts)),
            "unique_counties_in_precinct_data": int(precincts["county_fips"].nunique()),
            "counties_with_enough_precincts": int(len(county_stats)),
            "counties_matched_to_type_model": int(n_matched),
        },
        "variance_partition": {
            "within_type_variance": variance_partition["within_type_variance"],
            "between_type_variance": variance_partition["between_type_variance"],
            "f_ratio": variance_partition["f_ratio"],
            "n_types_included": variance_partition["n_types_included"],
            "interpretation": (
                "F-ratio >> 1 means between-type variance dominates, "
                "confirming types capture real partisan structure"
            ),
        },
        "type_prediction_correlation": prediction_corr,
        "precinct_heterogeneity": precinct_variance_by_type,
        "super_type_summary": super_type_summary,
    }

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {output_path}")

    return results


# ── Entry point ───────────────────────────────────────────────────────────────


if __name__ == "__main__":
    results = run_validation()

    print("\n" + "=" * 60)
    print("PRECINCT-LEVEL TYPE VALIDATION SUMMARY")
    print("=" * 60)

    vp = results["variance_partition"]
    print(f"\nVariance Partition:")
    print(f"  Between-type variance : {vp['between_type_variance']:.5f}")
    print(f"  Within-type variance  : {vp['within_type_variance']:.5f}")
    print(f"  F-ratio               : {vp['f_ratio']:.2f}x")
    print(f"  Types analyzed        : {vp['n_types_included']}")

    pc = results["type_prediction_correlation"]
    print(f"\nType Prior vs Precinct-Derived County Dem Share:")
    print(f"  Pearson r   : {pc['pearson_r']:.4f}")
    print(f"  RMSE        : {pc['rmse']:.4f}")
    print(f"  p-value     : {pc['p_value']:.2e}")
    print(f"  N counties  : {pc['n_counties']:,}")

    ph = results["precinct_heterogeneity"]
    print(f"\nWithin-County Precinct Heterogeneity:")
    print(f"  Mean precinct std  : {ph['overall_mean_precinct_std']:.4f}")
    print(f"  Median precinct std: {ph['overall_median_precinct_std']:.4f}")
    print(f"  Counties analyzed  : {ph['n_counties_analyzed']:,}")

    print("\nSuper-Type Summary (mean county Dem share by super-type):")
    for st in results["super_type_summary"]:
        print(f"  Super-type {st['super_type']}: {st['mean_dem_share']:.3f} ± {st['std_dem_share']:.3f} ({st['n_counties']} counties)")
