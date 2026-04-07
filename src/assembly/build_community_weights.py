"""
Stage 4 prep: build community-type weight matrices for spectral unmixing.

For each geographic aggregation level (tract, county, state), compute the
fraction of voting population that belongs to each of the 7 community types.

This is the "W" matrix in the poll propagation unmixing equation:
  y_poll ≈ W · θ
where y_poll is the observed poll result and θ is the community vote share vector.

Two weighting schemes:
  - vote-weighted: w_ik ∝ v_i * membership_ik (uses historical vote totals)
  - population-weighted: w_ik ∝ pop_i * membership_ik (uses ACS total population)

Vote-weighted is used for election prediction (weights by actual voters).
Population-weighted is used for MRP (weights by eligible voters).

Key property: for each geographic unit g,
  sum_k(W[g,k]) = 1  (weights sum to 1 across community types)

Inputs:
  data/communities/tract_memberships_k7.parquet
  data/assembled/vest_tracts_2020.parquet  (vote totals per tract)
  data/assembled/acs_features.parquet      (population per tract, if available)

Outputs:
  data/propagation/community_weights_tract.parquet   — (9393, 7) tract × community
  data/propagation/community_weights_county.parquet  — (226, 7) county × community
  data/propagation/community_weights_state.parquet   — (3, 7) state × community
  data/propagation/community_weights_state.csv       — same, for R
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "propagation"

COMP_COLS = [f"c{k}" for k in range(1, 8)]

LABELS = {
    "c1": "White rural homeowner",
    "c2": "Black urban",
    "c3": "Knowledge worker",
    "c4": "Asian",
    "c5": "Working-class homeowner",
    "c6": "Hispanic low-income",
    "c7": "Generic suburban baseline",
}

STATE_FIPS = {"01": "AL", "12": "FL", "13": "GA"}
STATE_NAMES = {"AL": "Alabama", "FL": "Florida", "GA": "Georgia"}


# ── Load ──────────────────────────────────────────────────────────────────────


def load_memberships() -> pd.DataFrame:
    """Load K=7 tract community memberships."""
    return pd.read_parquet(
        PROJECT_ROOT / "data" / "communities" / "tract_memberships_k7.parquet"
    )


def load_vote_totals() -> pd.DataFrame:
    """Load 2020 presidential vote totals per tract."""
    vest = pd.read_parquet(
        PROJECT_ROOT / "data" / "assembled" / "vest_tracts_2020.parquet"
    )
    return vest[["tract_geoid", "pres_total_2020"]].rename(
        columns={"pres_total_2020": "vote_total"}
    )


# ── GEOID parsing ─────────────────────────────────────────────────────────────


def parse_geoids(df: pd.DataFrame) -> pd.DataFrame:
    """Extract state_fips and county_fips from 11-digit tract_geoid."""
    df = df.copy()
    df["state_fips"] = df["tract_geoid"].str[:2]
    df["county_fips"] = df["tract_geoid"].str[:5]
    df["state_abbr"] = df["state_fips"].map(STATE_FIPS)
    return df


# ── Weight computation ────────────────────────────────────────────────────────


def compute_normalized_weights(df: pd.DataFrame, weight_col: str) -> pd.DataFrame:
    """
    Compute vote-weighted community membership fractions per tract.

    For each tract i:
      effective_weight[i, k] = vote_total[i] * membership[i, k]
      normalized_weight[i, k] = effective_weight[i, k] / sum_k(effective_weight[i, k])

    Returns df with COMP_COLS replaced by normalized values.
    """
    df = df.copy()

    # Effective weight = vote total × membership weight
    for comp in COMP_COLS:
        df[f"_eff_{comp}"] = df[weight_col] * df[comp]

    eff_cols = [f"_eff_{c}" for c in COMP_COLS]
    eff_sum = df[eff_cols].sum(axis=1).replace(0, np.nan)

    for comp in COMP_COLS:
        df[f"_norm_{comp}"] = df[f"_eff_{comp}"] / eff_sum

    return df


def aggregate_weights(df: pd.DataFrame, group_col: str, weight_col: str) -> pd.DataFrame:
    """
    Aggregate normalized community weights to a geographic level.

    For each geographic unit g:
      W[g, k] = sum_i(vote_total[i] * membership[i, k]) / sum_i(vote_total[i])
             where i ranges over tracts in g

    This gives the vote-share-weighted community composition of each unit.
    Sum over k is guaranteed to equal 1.0 per unit (up to floating point).
    """
    agg = {}
    for comp in COMP_COLS:
        # Numerator: sum of (vote_total × membership) across tracts in unit
        agg[f"num_{comp}"] = df[weight_col] * df[comp]

    agg_df = df[[group_col, weight_col]].copy()
    for comp in COMP_COLS:
        agg_df[f"num_{comp}"] = df[weight_col] * df[comp]

    grouped = agg_df.groupby(group_col)
    result = grouped[[f"num_{c}" for c in COMP_COLS]].sum()
    total_votes = grouped[weight_col].sum()

    # Normalize: W[g, k] = sum(v_i * w_ik) / sum(v_i)
    for comp in COMP_COLS:
        result[comp] = result[f"num_{comp}"] / total_votes
        result = result.drop(columns=[f"num_{comp}"])

    result = result.reset_index()

    # Verify normalization
    row_sums = result[COMP_COLS].sum(axis=1)
    max_deviation = abs(row_sums - 1.0).max()
    if max_deviation > 1e-6:
        log.warning("Weight rows don't sum to 1. Max deviation: %.2e", max_deviation)
    else:
        log.info("Weights sum to 1.0 ± %.2e for all %s units",
                 max_deviation, group_col)

    return result


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mem = load_memberships()
    votes = load_vote_totals()

    # Join memberships + vote totals
    df = mem.merge(votes, on="tract_geoid", how="inner")
    df = df[~df["is_uninhabited"]].dropna(
        subset=COMP_COLS + ["vote_total"]
    )
    df = parse_geoids(df)

    log.info("Working dataset: %d tracts across %d counties in %s",
             len(df), df["county_fips"].nunique(),
             ", ".join(df["state_abbr"].dropna().unique()))

    # ── Tract-level weights (the memberships themselves, vote-normalized) ──────
    # For tracts, the weight is just the membership normalized across components
    # (the NMF memberships already approximately sum to 1, but we re-normalize
    # using vote-weighting for consistency with county/state aggregations)
    tract_weights = aggregate_weights(df, "tract_geoid", "vote_total")
    tract_weights = tract_weights.merge(
        df[["tract_geoid", "state_fips", "state_abbr", "county_fips"]].drop_duplicates(),
        on="tract_geoid", how="left"
    )
    out_tract = OUTPUT_DIR / "community_weights_tract.parquet"
    tract_weights.to_parquet(out_tract, index=False)
    log.info("Tract weights: %d rows → %s", len(tract_weights), out_tract)

    # ── County-level weights ───────────────────────────────────────────────────
    county_weights = aggregate_weights(df, "county_fips", "vote_total")
    county_weights["state_fips"] = county_weights["county_fips"].str[:2]
    county_weights["state_abbr"] = county_weights["state_fips"].map(STATE_FIPS)
    out_county = OUTPUT_DIR / "community_weights_county.parquet"
    county_weights.to_parquet(out_county, index=False)
    log.info("County weights: %d rows → %s", len(county_weights), out_county)

    # ── State-level weights ────────────────────────────────────────────────────
    state_weights = aggregate_weights(df, "state_abbr", "vote_total")
    out_state = OUTPUT_DIR / "community_weights_state.parquet"
    state_weights.to_parquet(out_state, index=False)
    state_weights.to_csv(OUTPUT_DIR / "community_weights_state.csv", index=False)
    log.info("State weights: %d rows → %s", len(state_weights), out_state)

    # ── Print summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Community-type voting population shares by state")
    print("(vote-weighted, 2020 presidential totals)")
    print("=" * 70)

    print(f"\n{'State':<8}", end="")
    for comp in COMP_COLS:
        print(f"  {comp:>6}", end="")
    print()

    print(f"{'':8}", end="")
    for comp in COMP_COLS:
        short = LABELS[comp].split()[0][:6]
        print(f"  {short:>6}", end="")
    print()

    print("-" * 70)
    for _, row in state_weights.sort_values("state_abbr").iterrows():
        print(f"{row['state_abbr']:<8}", end="")
        for comp in COMP_COLS:
            print(f"  {row[comp]:.1%}", end="")
        print()

    # Compute "FL poll composition" — how much of FL's vote comes from each type
    # This is W_FL used in the unmixing equation: y_FL_poll ≈ W_FL · θ
    print("\nPoll interpretation guide:")
    print("A Florida-statewide poll with N=1000 respondents effectively samples:")
    fl_row = state_weights[state_weights["state_abbr"] == "FL"].iloc[0]
    for comp in COMP_COLS:
        n = fl_row[comp] * 1000
        print(f"  {comp} ({LABELS[comp]:<28}): ~{n:.0f} respondents")

    print("\nFor spectral unmixing: a FL poll result 'y' satisfies approximately:")
    print("  y ≈ " + " + ".join(
        f"{fl_row[c]:.3f}*θ_{c}" for c in COMP_COLS
    ))


if __name__ == "__main__":
    main()
