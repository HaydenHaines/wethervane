"""
Stage 5: 2026 forward predictions.

Loads current polls from data/polls/polls_2026.csv, runs the Bayesian
community update per race, and generates county-level vote share predictions.

Prior chain: 2016+2018+2020 (Stan) → 2022 back-calc → 2024 back-calc
The 2024 community estimates are the most recent available prior.

Each race (FL Senate, GA Senate, etc.) is treated independently:
  - Prior: state-level 2024 community vote shares (most recent year)
  - Update: all polls for that race combined (information accumulates additively)
  - Output: county-level predicted Democrat two-party vote share

Usage:
  python src/prediction/predict_2026.py
  python src/prediction/predict_2026.py --race "2026 FL Senate"
  python src/prediction/predict_2026.py --top-counties 10
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.propagation.propagate_polls import (
    bayesian_poll_update,
    load_prior,
    COMP_COLS,
    LABELS,
)
from src.assembly.ingest_polls import load_polls, list_races
from src.validation.poll_accuracy import predict_from_posterior

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_county_weights() -> pd.DataFrame:
    weights = pd.read_parquet(
        PROJECT_ROOT / "data" / "propagation" / "community_weights_county.parquet"
    )
    names_path = PROJECT_ROOT / "data" / "assembled" / "county_fips_names.parquet"
    if names_path.exists():
        names = pd.read_parquet(names_path)[["county_fips", "county_name"]]
        weights = weights.merge(names, on="county_fips", how="left")
    return weights


def run_race_prediction(
    race_name: str,
    polls: list,
    county_weights_df: pd.DataFrame,
) -> dict:
    """
    Run the full Stage 4+5 pipeline for a single race.

    Returns a dict with:
      race, state, n_polls, mu_prior, mu_post, Sigma_post,
      pred_counties (DataFrame), state_pred, poll_avg
    """
    # All polls for a single race share the same geography (state)
    state = polls[0].geography

    # Load most recent prior (2024 for FL/GA; 2020 for AL which has no 2022/2024)
    _, Sigma_prior = load_prior(state=None)  # covariance is always pooled
    try:
        mu_prior, _ = load_prior(state=state)          # most recent year
        prior_year = "most recent"
    except Exception:
        mu_prior, _ = load_prior(state=state, year=2020)
        prior_year = "2020 (fallback)"

    # Bayesian update: all polls for this race passed at once
    posterior = bayesian_poll_update(mu_prior, Sigma_prior, polls)

    # County-level predictions
    state_counties = county_weights_df[county_weights_df["state_abbr"] == state]
    pred = predict_from_posterior(posterior, state_counties, "county_fips")

    # Vote-weighted state-level prediction
    # (use equal weights since we don't have 2026 turnout — use 2024 as proxy)
    state_pred = float(pred["pred_dem_share"].mean())

    # Poll average (unweighted mean of dem_share inputs)
    poll_avg = float(np.mean([p.dem_share for p in polls]))

    return {
        "race": race_name,
        "state": state,
        "prior_year": prior_year,
        "n_polls": len(polls),
        "poll_avg": poll_avg,
        "mu_prior": mu_prior,
        "posterior": posterior,
        "pred_counties": pred,
        "state_pred": state_pred,
    }


def print_race_results(result: dict, county_weights_df: pd.DataFrame, top_n: int = 5) -> None:
    race = result["race"]
    state = result["state"]
    posterior = result["posterior"]
    mu_prior = result["mu_prior"]
    pred = result["pred_counties"]

    print(f"\n{'─'*68}")
    print(f"  {race}")
    print(f"{'─'*68}")
    print(f"  Polls: {result['n_polls']}  "
          f"Poll avg: {result['poll_avg']:.1%}  "
          f"Model prediction: {result['state_pred']:.1%}  "
          f"(prior: {result['prior_year']})")

    # Community posteriors
    df_post = posterior.to_dataframe()
    print(f"\n  {'Community':<28}  {'Prior':>8}  {'Posterior':>10}  {'Shift':>8}  {'90% CI':>18}")
    print("  " + "─" * 78)
    for k, (_, row) in enumerate(df_post.iterrows()):
        shift = row["mu_post"] - mu_prior[k]
        shift_str = f"{shift:+.1%}" if abs(shift) > 0.001 else "  —"
        ci_str = f"[{row['lo90']:.1%}, {row['hi90']:.1%}]"
        marker = " ◄" if abs(shift) > 0.02 else ""
        print(f"  {row['label']:<28}  {mu_prior[k]:.1%}  {row['mu_post']:.1%}  "
              f"{shift_str:>8}  {ci_str}{marker}")

    # Top/bottom counties
    pred_sorted = pred.sort_values("pred_dem_share", ascending=False)
    county_info = county_weights_df[["county_fips", "county_name"]].drop_duplicates() \
        if "county_name" in county_weights_df.columns else None

    # Merge county names for display
    if "county_name" in county_weights_df.columns:
        name_map = county_weights_df.set_index("county_fips")["county_name"].to_dict()
        pred_sorted = pred_sorted.copy()
        pred_sorted["county_label"] = pred_sorted["county_fips"].map(name_map).fillna(pred_sorted["county_fips"])
    else:
        pred_sorted["county_label"] = pred_sorted["county_fips"]

    print(f"\n  Most Democratic counties ({state}):")
    for _, row in pred_sorted.head(top_n).iterrows():
        print(f"    {row['county_label']:<30}  {row['pred_dem_share']:.1%}")

    print(f"\n  Most Republican counties ({state}):")
    for _, row in pred_sorted.tail(top_n).iterrows():
        print(f"    {row['county_label']:<30}  {row['pred_dem_share']:.1%}")


def main(race_filter: str | None = None, top_n: int = 5) -> None:
    county_weights_df = load_county_weights()

    print("\n" + "=" * 68)
    print("  2026 Election Forecast — Community-Weighted Bayesian Model")
    print(f"  Prior chain: Stan(2016-2020) → 2022 gov → 2024 pres")
    print("=" * 68)

    all_races = list_races("2026")
    if race_filter:
        all_races = [r for r in all_races if race_filter.lower() in r.lower()]
        if not all_races:
            print(f"No races matching '{race_filter}'. Available: {list_races('2026')}")
            return

    # ── State-level summary table ──────────────────────────────────────────────
    print("\n── Summary: state-level predictions ───────────────────────────────")
    print(f"  {'Race':<26}  {'Poll avg':>10}  {'Model pred':>12}  {'Polls':>6}")
    print("  " + "─" * 60)

    all_results = []
    for race_name in all_races:
        polls = load_polls("2026", race=race_name)
        if not polls:
            continue
        result = run_race_prediction(race_name, polls, county_weights_df)
        all_results.append(result)
        lean = "D+" if result["state_pred"] > 0.5 else "R+"
        margin = abs(result["state_pred"] - 0.5) * 2
        print(f"  {race_name:<26}  {result['poll_avg']:.1%}  "
              f"{result['state_pred']:.1%} ({lean}{margin:.1%})  "
              f"{result['n_polls']:>4}")

    # ── Detailed per-race output ───────────────────────────────────────────────
    for result in all_results:
        print_race_results(result, county_weights_df, top_n=top_n)

    # ── Save county predictions ────────────────────────────────────────────────
    output_dir = PROJECT_ROOT / "data" / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for result in all_results:
        df = result["pred_counties"].copy()
        df["race"] = result["race"]
        df["state_pred"] = result["state_pred"]
        df["poll_avg"] = result["poll_avg"]
        frames.append(df)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        out_path = output_dir / "county_predictions_2026.parquet"
        combined.to_parquet(out_path, index=False)
        print(f"\n  Saved {len(combined)} county-race predictions → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2026 election community-weighted forecast")
    parser.add_argument("--race", type=str, default=None,
                        help="Filter to specific race (substring match)")
    parser.add_argument("--top-counties", type=int, default=5,
                        help="Number of top/bottom counties to show per race")
    args = parser.parse_args()
    main(race_filter=args.race, top_n=args.top_counties)
