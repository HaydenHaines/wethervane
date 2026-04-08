"""Compute per-type voter behavior parameters τ (turnout ratio) and δ (choice shift).

These parameters make the tract-primary model cycle-type-aware: without them the model
treats every election as a presidential-shaped electorate, systematically overestimating
Republican performance in midterms because high-propensity-Democrat groups (young, urban,
low-income) drop off at higher rates than the presidential electorate would imply.

**τ (turnout ratio):**
  τ_j = mean(off_cycle_votes / presidential_votes) across tracts, weighted by type-j
  membership.  τ < 1 means type j turns out at less than its presidential rate in
  off-cycle elections.  Expected range: 0.5–1.0 for most types.

**δ (residual choice shift):**
  δ_j = mean(off_cycle_dem_share - nearest_presidential_dem_share), weighted by type-j
  membership.  Captures genuine preference differences beyond what τ-driven composition
  change already explains.  Expected |δ| < 0.10 for most types.

Data pipeline position:
  Inputs:  T.1  data/assembled/tract_elections.parquet
           T.3  data/communities/tract_type_assignments.parquet
  Outputs: data/behavior/tau.npy, delta.npy, summary.json

Run as a module:
  uv run python -m src.behavior.compute_behavior_params
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.behavior.voter_behavior import (
    OFFCYCLE_RACES,
    PRESIDENTIAL_RACES,
    TAU_CLIP_MAX,
    TAU_CLIP_MIN,
    TAU_FALLBACK,
    _map_race_type_to_race,
)

# Default paths relative to project root.
DEFAULT_ELECTIONS_PATH = "data/assembled/tract_elections.parquet"
DEFAULT_ASSIGNMENTS_PATH = "data/communities/tract_type_assignments.parquet"
DEFAULT_OUTPUT_DIR = "data/behavior"

# Minimum total presidential votes for a tract to contribute to τ and δ estimates.
# Tracts below this threshold are noise (uncontested races, partial data, tiny populations).
MIN_PRESIDENTIAL_VOTES = 50


def _extract_score_matrix(
    assignments_df: pd.DataFrame,
    n_types: int,
    common_geoids: pd.Index,
) -> np.ndarray:
    """Return a (n_tracts, J) matrix of type membership scores for the given GEOIDs.

    Args:
        assignments_df: DataFrame indexed by tract_geoid with type_j_score columns.
        n_types: Number of types J (determines which score columns to pull).
        common_geoids: Index of GEOIDs to extract (already intersection of data sources).

    Returns:
        float64 array of shape (len(common_geoids), n_types).
    """
    score_cols = [f"type_{j}_score" for j in range(n_types)]
    return assignments_df.loc[common_geoids, score_cols].values.astype(np.float64)


def compute_tau(elections_df: pd.DataFrame, assignments_df: pd.DataFrame) -> np.ndarray:
    """Compute per-type turnout ratio τ from tract election and assignment data.

    τ_j = membership-weighted mean of (mean_off_cycle_votes / mean_presidential_votes)
    across all tracts with coverage in both election types.

    Args:
        elections_df: tract_elections.parquet in T.1 format.
                      Required columns: tract_geoid, race_type, total_votes, dem_share.
                      race_type values used: PRES, GOV, SEN (and variants).
        assignments_df: tract_type_assignments.parquet in T.3 format.
                        Required columns: tract_geoid, type_0_score … type_J-1_score.

    Returns:
        np.ndarray of shape (J,), values clipped to [TAU_CLIP_MIN, TAU_CLIP_MAX].
        Types with no data coverage receive TAU_FALLBACK.
    """
    # Map T.1 race_type codes to canonical {president, governor, senate} labels and
    # rename total_votes → votes_total.  Non-electoral races (CONG, AG, etc.) are dropped.
    tract_votes = _map_race_type_to_race(elections_df)

    # Prepare assignments: deduplicate GEOIDs (known DRA gotcha), set as index.
    assignments = _load_and_prepare_assignments_if_needed(assignments_df)

    pres_rows = tract_votes[tract_votes["race"].isin(PRESIDENTIAL_RACES)]
    off_rows = tract_votes[tract_votes["race"].isin(OFFCYCLE_RACES)]

    # Average across all available years per tract — smooths individual-cycle noise.
    pres_mean = pres_rows.groupby("tract_geoid")["votes_total"].mean()
    off_mean = off_rows.groupby("tract_geoid")["votes_total"].mean()

    # Require the tract to have meaningful presidential turnout (avoids division noise)
    # and to appear in both datasets and the type assignments.
    pres_qualified = pres_mean[pres_mean >= MIN_PRESIDENTIAL_VOTES]
    common = (
        pres_qualified.index
        .intersection(off_mean.index)
        .intersection(assignments.index)
    )

    pres_vals = pres_mean.loc[common].values
    off_vals = off_mean.loc[common].values

    # Per-tract τ = off_cycle / presidential votes.
    # Zero presidential turnout (already filtered above) cannot produce NaN here.
    tract_tau = off_vals / pres_vals

    # Determine J from score column count.
    n_types = sum(1 for c in assignments.columns if c.endswith("_score"))
    W = _extract_score_matrix(assignments, n_types, common)

    tau = np.empty(n_types)
    for j in range(n_types):
        weights = W[:, j]
        total_w = weights.sum()
        if total_w > 0:
            tau[j] = np.average(tract_tau, weights=weights)
        else:
            # No tracts have meaningful membership in this type — use fallback.
            tau[j] = TAU_FALLBACK

    return np.clip(tau, TAU_CLIP_MIN, TAU_CLIP_MAX)


def compute_delta(
    elections_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
    tau: np.ndarray,
) -> np.ndarray:
    """Compute per-type residual choice shift δ using τ-reweighted presidential baseline.

    δ_j captures genuine preference differences in off-cycle elections BEYOND what
    turnout composition change alone would explain.

    The naive approach (off_dem_share - pres_dem_share) conflates two effects:
      1. Composition change: low-τ types drop off in midterms, shifting the effective
         electorate toward high-turnout types (and their Dem share).
      2. Genuine preference shift: the same voters actually vote differently.

    This implementation separates them using the τ-reweighted expected off-cycle
    dem share as the baseline. For each tract, the expected off-cycle dem share is
    computed as: sum_j(τ_j * W_ij * θ_j) / sum_j(τ_j * W_ij), where θ_j is the
    type-level presidential Dem share. The residual after subtracting this expected
    value is the pure preference signal δ.

    For temporal precision, each off-cycle observation is matched to its nearest
    presidential year before computing the residuals. This removes secular trend
    artifacts that would arise from always comparing against the same presidential cycle.

    Args:
        elections_df: Same T.1 format as compute_tau.
        assignments_df: Same T.3 format as compute_tau.
        tau: Turnout ratios from compute_tau — shape (J,). Used to reweight the
             presidential baseline by off-cycle type composition.

    Returns:
        np.ndarray of shape (J,). Positive = type votes more Dem in off-cycle years
        beyond what τ-driven composition change predicts.
    """
    tract_votes = _map_race_type_to_race(elections_df)
    assignments = _load_and_prepare_assignments_if_needed(assignments_df)

    pres_rows = tract_votes[tract_votes["race"].isin(PRESIDENTIAL_RACES)][
        ["tract_geoid", "year", "dem_share"]
    ].dropna(subset=["dem_share"])

    off_rows = tract_votes[tract_votes["race"].isin(OFFCYCLE_RACES)][
        ["tract_geoid", "year", "dem_share"]
    ].dropna(subset=["dem_share"])

    if off_rows.empty or pres_rows.empty:
        n_types = sum(1 for c in assignments.columns if c.endswith("_score"))
        return np.zeros(n_types)

    # Build a lookup: for each (tract, year) pair in off-cycle data, find the
    # presidential year closest in time.
    pres_years = sorted(pres_rows["year"].unique())

    def nearest_presidential_year(off_year: int) -> int:
        """Return the presidential year closest to off_year."""
        return min(pres_years, key=lambda y: abs(y - off_year))

    # Map each off-cycle observation to its nearest presidential baseline.
    off_rows = off_rows.copy()
    off_rows["pres_year"] = off_rows["year"].map(nearest_presidential_year)

    # Join off-cycle rows to their nearest presidential dem_share per tract.
    pres_lookup = (
        pres_rows.rename(columns={"dem_share": "pres_dem_share", "year": "pres_year"})
        .groupby(["tract_geoid", "pres_year"])["pres_dem_share"]
        .mean()
        .reset_index()
    )

    merged = off_rows.merge(pres_lookup, on=["tract_geoid", "pres_year"], how="inner")

    # Average per-tract pres and off-cycle dem shares across all paired elections.
    tract_pres_share = merged.groupby("tract_geoid")["pres_dem_share"].mean()
    tract_off_share = merged.groupby("tract_geoid")["dem_share"].mean()

    # Restrict to tracts present in both elections and assignments.
    common = (
        tract_pres_share.index
        .intersection(tract_off_share.index)
        .intersection(assignments.index)
    )
    pres_vals = tract_pres_share.loc[common].values
    off_vals = tract_off_share.loc[common].values

    n_types = sum(1 for c in assignments.columns if c.endswith("_score"))
    W = _extract_score_matrix(assignments, n_types, common)  # (n_tracts, J)

    # --- τ-reweighted baseline ---
    #
    # Step 1: type-level presidential Dem share θ_j = membership-weighted mean
    # of pres_dem_share across tracts. Anchors what each type votes presidentially.
    pres_valid = ~np.isnan(pres_vals)
    theta = np.zeros(n_types)
    overall_pres_mean = float(np.nanmean(pres_vals))
    for j in range(n_types):
        weights = W[pres_valid, j]
        total_w = weights.sum()
        if total_w > 0:
            theta[j] = np.average(pres_vals[pres_valid], weights=weights)
        else:
            # Fall back to overall mean for types with no data coverage.
            theta[j] = overall_pres_mean

    # Step 2: expected off-cycle dem share per tract, assuming only turnout
    # composition changes and not preferences. Types with low τ shrink in the
    # effective off-cycle electorate; their presidential θ_j is down-weighted.
    # expected_off_i = Σ_j(τ_j * W_ij * θ_j) / Σ_j(τ_j * W_ij)
    tau_weighted = W * tau[np.newaxis, :]  # (n_tracts, J)
    tau_row_sums = tau_weighted.sum(axis=1, keepdims=True)
    tau_row_sums = np.where(tau_row_sums > 0, tau_row_sums, 1.0)
    tau_norm = tau_weighted / tau_row_sums  # row-normalized, (n_tracts, J)

    expected_off = tau_norm @ theta  # (n_tracts,): pure-composition expected share

    # Step 3: residual = actual off-cycle minus τ-reweighted expected.
    # This is the genuine preference signal stripped of composition effects.
    tract_residuals = off_vals - expected_off

    valid = ~np.isnan(tract_residuals)
    delta = np.empty(n_types)
    for j in range(n_types):
        weights = W[valid, j]
        total_w = weights.sum()
        if total_w > 0:
            delta[j] = np.average(tract_residuals[valid], weights=weights)
        else:
            delta[j] = 0.0

    return delta


def _load_and_prepare_assignments_if_needed(
    assignments_df: pd.DataFrame,
) -> pd.DataFrame:
    """Set tract_geoid as index and deduplicate if not already indexed.

    The DataFrame may arrive already indexed (if prepared upstream) or with
    tract_geoid as a column (raw parquet load).  Either case is handled cleanly.

    Args:
        assignments_df: T.3 assignments, either column-indexed or already indexed.

    Returns:
        DataFrame indexed by tract_geoid.
    """
    if assignments_df.index.name == "tract_geoid":
        return assignments_df

    if "tract_geoid" not in assignments_df.columns:
        raise ValueError(
            "assignments_df must have a 'tract_geoid' column or be indexed by it."
        )

    n_before = len(assignments_df)
    df = assignments_df.drop_duplicates(subset="tract_geoid").set_index("tract_geoid")
    n_after = len(df)

    if n_before != n_after:
        import warnings
        warnings.warn(
            f"Dropped {n_before - n_after} duplicate GEOIDs from assignments "
            f"(kept {n_after} unique tracts).",
            UserWarning,
            stacklevel=3,
        )

    return df


def compute_and_save(output_dir: str = DEFAULT_OUTPUT_DIR) -> dict:
    """Load pipeline data, compute τ and δ, write outputs to disk.

    Reads from the standard T.1/T.3 parquet paths.  Creates output_dir if needed.

    Output files:
        tau.npy     — float64 array, shape (J,)
        delta.npy   — float64 array, shape (J,)
        summary.json — metadata: n_types, n_tracts_used, mean_tau, std_tau,
                       mean_delta, std_delta, computation_date

    Args:
        output_dir: Directory to write outputs (default: data/behavior/).

    Returns:
        Summary dict (same content as summary.json).
    """
    elections_df = pd.read_parquet(DEFAULT_ELECTIONS_PATH)
    assignments_df = pd.read_parquet(DEFAULT_ASSIGNMENTS_PATH)

    tau = compute_tau(elections_df, assignments_df)
    delta = compute_delta(elections_df, assignments_df, tau)

    # Count tracts used: intersection of coverage in both election types + assignments.
    tract_votes = _map_race_type_to_race(elections_df)
    assignments = _load_and_prepare_assignments_if_needed(assignments_df)

    pres_rows = tract_votes[tract_votes["race"].isin(PRESIDENTIAL_RACES)]
    off_rows = tract_votes[tract_votes["race"].isin(OFFCYCLE_RACES)]

    pres_mean = pres_rows.groupby("tract_geoid")["votes_total"].mean()
    pres_qualified = pres_mean[pres_mean >= MIN_PRESIDENTIAL_VOTES]
    off_tracts = set(off_rows["tract_geoid"].unique())
    n_tracts_used = len(
        set(pres_qualified.index) & off_tracts & set(assignments.index)
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    np.save(output_path / "tau.npy", tau)
    np.save(output_path / "delta.npy", delta)

    n_types = len(tau)
    summary = {
        "n_types": n_types,
        "n_tracts_used": n_tracts_used,
        "mean_tau": float(tau.mean()),
        "std_tau": float(tau.std()),
        "mean_delta": float(delta.mean()),
        "std_delta": float(delta.std()),
        "computation_date": date.today().isoformat(),
    }

    with open(output_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    _print_summary(tau, delta, n_tracts_used)

    return summary


def _print_summary(tau: np.ndarray, delta: np.ndarray, n_tracts: int) -> None:
    """Print a human-readable summary of τ/δ statistics to stdout."""
    sep = "=" * 60
    print(f"\n{sep}")
    print("Voter Behavior Layer — τ and δ Parameters")
    print(sep)
    print(f"  Tracts used:  {n_tracts:,}")
    print(f"  Types (J):    {len(tau)}")

    print("\n  τ (turnout ratio):")
    print(f"    mean={tau.mean():.4f}  std={tau.std():.4f}  "
          f"min={tau.min():.4f}  max={tau.max():.4f}")

    top5 = np.argsort(tau)[-5:][::-1]
    bot5 = np.argsort(tau)[:5]
    print(f"    Highest: {[(int(j), round(float(tau[j]), 3)) for j in top5]}")
    print(f"    Lowest:  {[(int(j), round(float(tau[j]), 3)) for j in bot5]}")

    pct_below_one = (tau < 1.0).mean() * 100
    status = "PASS" if pct_below_one > 60 else "WARN"
    print(f"    τ < 1.0: {pct_below_one:.0f}% of types ({status}, expect >60%)")

    print("\n  δ (residual choice shift):")
    print(f"    mean={delta.mean():.4f}  std={delta.std():.4f}  "
          f"min={delta.min():.4f}  max={delta.max():.4f}")

    top5d = np.argsort(delta)[-5:][::-1]
    bot5d = np.argsort(delta)[:5]
    print(f"    Highest: {[(int(j), round(float(delta[j]), 4)) for j in top5d]}")
    print(f"    Lowest:  {[(int(j), round(float(delta[j]), 4)) for j in bot5d]}")

    pct_small = (np.abs(delta) < 0.1).mean() * 100
    status = "PASS" if pct_small > 70 else "WARN"
    print(f"    |δ| < 0.1: {pct_small:.0f}% of types ({status}, expect >70%)")
    print(f"{sep}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute voter behavior parameters τ and δ for the tract-primary model."
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    summary = compute_and_save(args.output_dir)
    print(f"\nSummary: {json.dumps(summary, indent=2)}")
