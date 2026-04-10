"""Build county-level type assignments from tract-level type scores.

Aggregates tract-level soft type membership (J=100) to county level using
population-weighted averaging. Population weights come from 2024 presidential
vote totals in DRA tract data. Tracts without vote data get equal weight (1.0).

The tract→county crosswalk is derived from the Census GEOID structure:
tract GEOID is 11 digits (SSCCCTTTTTT), where the first 5 digits are the
county FIPS code.

Output schema (matches backtest_harness._load_type_data_for_backtest):
    county_fips (str, zero-padded 5-digit)
    type_0_score ... type_{J-1}_score (float, sum to 1.0 per row)

Writes to: data/communities/type_assignments.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_tract_type_scores(project_root: Path) -> pd.DataFrame:
    """Load tract-level type assignment scores.

    Returns DataFrame with tract_geoid and type_*_score columns.
    Drops duplicate GEOIDs (known DRA data issue) keeping first occurrence.
    """
    path = project_root / "data" / "communities" / "tract_type_assignments.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Tract type assignments not found: {path}\n"
            "Run the tract type discovery pipeline first: "
            "uv run python -m src.discovery.run_tract_type_discovery"
        )

    df = pd.read_parquet(path)
    n_before = len(df)
    df = df.drop_duplicates(subset="tract_geoid")
    n_after = len(df)
    if n_before != n_after:
        print(f"Dropped {n_before - n_after} duplicate tract GEOIDs ({n_before} → {n_after})")

    return df


def load_tract_population_weights(project_root: Path) -> pd.Series:
    """Load 2024 presidential vote totals as population proxy for weighting.

    Returns Series indexed by tract_geoid with total votes as values.
    Tracts not in the vote file get NaN (caller should fill with fallback).
    """
    path = project_root / "data" / "tracts" / "tract_votes_dra.parquet"
    if not path.exists():
        print(f"WARNING: Tract vote data not found: {path}")
        print("Falling back to equal-weighted aggregation.")
        return pd.Series(dtype=float)

    votes = pd.read_parquet(path)

    # Use 2024 presidential votes as the best proxy for current population.
    # Fall back to 2020 if 2024 isn't available.
    pres_2024 = votes[(votes["year"] == 2024) & (votes["race"] == "president")]
    if len(pres_2024) == 0:
        pres_2024 = votes[(votes["year"] == 2020) & (votes["race"] == "president")]
        print("Using 2020 presidential votes as population proxy (2024 not available).")

    weights = pres_2024.drop_duplicates(subset="tract_geoid").set_index("tract_geoid")[
        "votes_total"
    ]
    print(f"Loaded population weights for {len(weights)} tracts")
    return weights


def derive_county_fips(tract_geoid: pd.Series) -> pd.Series:
    """Extract 5-digit county FIPS from 11-digit tract GEOID.

    Census GEOID structure: SS-CCC-TTTTTT (state-county-tract).
    The first 5 characters are the county FIPS code.
    """
    return tract_geoid.astype(str).str.zfill(11).str[:5]


def aggregate_tract_to_county(
    tract_scores: pd.DataFrame,
    weights: pd.Series,
) -> pd.DataFrame:
    """Aggregate tract-level type scores to county level via population-weighted mean.

    Parameters
    ----------
    tract_scores : DataFrame
        Must contain 'tract_geoid' and type_*_score columns.
    weights : Series
        Indexed by tract_geoid, values are population weights.
        Tracts not in this series get weight=1.0 (equal weight fallback).

    Returns
    -------
    DataFrame with county_fips and type_*_score columns, row-normalized to sum to 1.
    """
    # Sort score columns in natural numeric order (type_0, type_1, ..., type_99)
    # NOT alphabetical (which would give type_0, type_10, type_11, ..., type_1, type_2).
    # Natural order matches the covariance matrix indexing and the discovery pipeline output.
    score_cols = [c for c in tract_scores.columns if c.endswith("_score")]
    score_cols.sort(key=lambda c: int(c.split("_")[1]))
    if not score_cols:
        raise ValueError("No type score columns found in tract data")

    j = len(score_cols)
    print(f"Aggregating {len(tract_scores)} tracts → counties using J={j} type scores")

    df = tract_scores[["tract_geoid"] + score_cols].copy()
    df["county_fips"] = derive_county_fips(df["tract_geoid"])

    # Assign population weights; fallback to 1.0 for tracts without vote data.
    if len(weights) > 0:
        df["weight"] = df["tract_geoid"].map(weights).fillna(1.0)
        # Clamp zero-vote tracts to 1.0 so they still contribute minimally.
        df["weight"] = df["weight"].clip(lower=1.0)
        n_weighted = (df["tract_geoid"].isin(weights.index)).sum()
        print(f"  {n_weighted}/{len(df)} tracts have population weights")
    else:
        df["weight"] = 1.0
        print("  Using equal weights for all tracts (no population data)")

    # Weighted aggregation: for each county, compute
    #   county_score_j = sum(tract_weight_i * tract_score_ij) / sum(tract_weight_i)
    weighted_scores = df[score_cols].multiply(df["weight"], axis=0)
    weighted_scores["county_fips"] = df["county_fips"]
    weighted_scores["weight"] = df["weight"]

    county_sums = weighted_scores.groupby("county_fips")[score_cols].sum()
    county_weight_totals = weighted_scores.groupby("county_fips")["weight"].sum()

    county_scores = county_sums.div(county_weight_totals, axis=0)

    # Row-normalize to ensure scores sum to 1.0 (correct for any floating-point drift).
    row_totals = county_scores.sum(axis=1)
    county_scores = county_scores.div(row_totals, axis=0)

    result = county_scores.reset_index()
    result["county_fips"] = result["county_fips"].astype(str).str.zfill(5)

    print(f"  Result: {len(result)} counties")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build county-level type assignments from tract-level scores."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root directory (default: auto-detected)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: data/communities/type_assignments.parquet)",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    output_path = args.output or (project_root / "data" / "communities" / "type_assignments.parquet")

    print(f"Project root: {project_root}")
    print(f"Output: {output_path}")
    print()

    # Load tract-level type scores.
    tract_scores = load_tract_type_scores(project_root)

    # Load population weights for weighted aggregation.
    weights = load_tract_population_weights(project_root)

    # Aggregate to county level.
    county_df = aggregate_tract_to_county(tract_scores, weights)

    # Validate output schema.
    score_cols = [c for c in county_df.columns if c.endswith("_score")]
    score_cols.sort(key=lambda c: int(c.split("_")[1]))
    assert "county_fips" in county_df.columns, "Missing county_fips column"
    assert len(score_cols) > 0, "No score columns in output"
    assert county_df["county_fips"].is_unique, "Duplicate county FIPS in output"

    # Sanity checks.
    row_sums = county_df[score_cols].sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), (
        f"Row sums not close to 1.0: min={row_sums.min()}, max={row_sums.max()}"
    )

    print()
    print(f"Schema: county_fips + {len(score_cols)} score columns")
    print(f"Counties: {len(county_df)}")
    print(f"FIPS examples: {county_df['county_fips'].head(3).tolist()}")
    print(f"Score range: [{county_df[score_cols].min().min():.6f}, {county_df[score_cols].max().max():.6f}]")

    # Write output.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    county_df.to_parquet(output_path, index=False)
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
