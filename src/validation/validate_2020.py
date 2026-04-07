"""
Validation test 1: 2020 Presidential election.

Feeds 2020 final state-level polling averages into Stage 4, then uses
Stage 5 to produce county-level predictions, and compares against
actual 2020 VEST results.

Three baselines:
  1. Naive poll:         W · poll_dem_share (no community structure, just scale
                         the state poll down to counties using W)
  2. Model (pooled):     Stage 4 with pooled Stan prior — genuinely predictive,
                         does not know 2020 state results in advance
  3. Model (2020 prior): Stage 4 with state-stratified 2020 prior — in-sample
                         sanity check; prior already encodes 2020 result

Polling averages are approximate RCP final averages for 2020 Presidential.
They represent what a forecaster would have had available ~election day.

Two-party dem_share = Biden / (Biden + Trump)
  FL: Biden 49.2%, Trump 47.9% → 0.507   actual → 0.483  (polls missed R by ~2.5pp)
  GA: Biden 47.9%, Trump 47.5% → 0.502   actual → 0.501  (near-exact)
  AL: Biden 37.0%, Trump 59.0% → 0.385   actual → 0.371  (sparse polling)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.propagation.propagate_polls import (  # noqa: E402
    PollObservation,
    bayesian_poll_update,
    load_prior,
)
from src.validation.poll_accuracy import (  # noqa: E402
    accuracy_report,
    county_actuals_from_vest,
    predict_from_posterior,
    print_accuracy_table,
)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── 2020 polling averages ──────────────────────────────────────────────────────
# Approximate RCP final averages (two-party dem_share)
# Source: RealClearPolitics 2020 Presidential average, final pre-election
# FL polls overestimated Biden by ~2.5pp — a known "shy Trump" effect in FL
POLLS_2020 = [
    PollObservation("FL", dem_share=0.507, n_sample=1200,
                    race="2020 President", date="2020-11-02",
                    pollster="RCP avg", geo_level="state"),
    PollObservation("GA", dem_share=0.502, n_sample=1000,
                    race="2020 President", date="2020-11-02",
                    pollster="RCP avg", geo_level="state"),
    PollObservation("AL", dem_share=0.385, n_sample=400,
                    race="2020 President", date="2020-11-02",
                    pollster="RCP avg (sparse)", geo_level="state"),
]

# 2020 actual state-level results (computed from VEST, for reference)
ACTUAL_2020_STATE = {"FL": 0.483, "GA": 0.501, "AL": 0.371}


def naive_poll_prediction(
    county_weights_df: pd.DataFrame,
    polls: list[PollObservation],
) -> pd.DataFrame:
    """
    Baseline: directly apply the state poll to all counties in that state.

    Each county's prediction = the state poll dem_share for its state.
    This is the 'no model' baseline — what you'd get if you assumed uniform
    swing across all counties.
    """
    poll_map = {p.geography: p.dem_share for p in polls}
    result = county_weights_df[["county_fips", "state_abbr"]].copy()
    result["pred_dem_share"] = result["state_abbr"].map(poll_map)
    result["pred_std"] = 0.0
    result["pred_lo90"] = result["pred_dem_share"]
    result["pred_hi90"] = result["pred_dem_share"]
    return result


def run_model(
    polls: list[PollObservation],
    county_weights_df: pd.DataFrame,
    prior_type: str,           # "pooled" | "FL" | "GA" | "AL" (per-state)
    label: str,
) -> tuple[pd.DataFrame, dict]:
    """
    Run Stage 4 + Stage 5 for a given prior type.

    For per-state priors, we update state by state and concatenate predictions.
    For pooled, we update once with all polls and predict all counties.
    """
    _, Sigma_prior = load_prior(state=None)   # covariance always pooled

    if prior_type == "pooled":
        mu_prior, _ = load_prior(state=None)
        posterior = bayesian_poll_update(mu_prior, Sigma_prior, polls)
        pred_df = predict_from_posterior(posterior, county_weights_df, "county_fips")
    else:
        # Per-state: separate prior and update for each state
        frames = []
        states = county_weights_df["state_abbr"].unique()
        for state in states:
            state_polls = [p for p in polls if p.geography == state]
            if not state_polls:
                continue
            mu_prior, _ = load_prior(state=state)
            posterior = bayesian_poll_update(mu_prior, Sigma_prior, state_polls)
            state_counties = county_weights_df[county_weights_df["state_abbr"] == state]
            pred = predict_from_posterior(posterior, state_counties, "county_fips")
            frames.append(pred)
        pred_df = pd.concat(frames, ignore_index=True)

    return pred_df


def main() -> None:
    print("\n" + "=" * 70)
    print("Validation Test 1: 2020 Presidential Election")
    print("=" * 70)

    # ── Load data ──────────────────────────────────────────────────────────────
    county_weights_df = pd.read_parquet(
        PROJECT_ROOT / "data" / "propagation" / "community_weights_county.parquet"
    )
    county_actual, state_actual = county_actuals_from_vest(2020, "pres")

    print(f"\nLoaded: {len(county_weights_df)} counties, {len(county_actual)} with actuals")

    # ── Print poll inputs vs actuals ───────────────────────────────────────────
    print("\nPoll inputs vs. actual 2020 results:")
    print(f"  {'State':<6}  {'Poll avg':>10}  {'Actual':>10}  {'Poll error':>12}")
    print("  " + "-" * 44)
    for p in POLLS_2020:
        actual = ACTUAL_2020_STATE[p.geography]
        error = p.dem_share - actual
        print(f"  {p.geography:<6}  {p.dem_share:.1%}  {actual:.1%}  {error:+.1%}")

    print("\n(Positive error = polls overestimated Democrat share)")

    # ── Run all models ─────────────────────────────────────────────────────────
    print("\nRunning county-level predictions...")

    # 1. Naive poll baseline
    naive_pred = naive_poll_prediction(county_weights_df, POLLS_2020)
    naive_acc = accuracy_report(naive_pred, county_actual, "county_fips", "Naive poll (uniform swing)")

    # 2. Model with pooled prior
    pooled_pred = run_model(POLLS_2020, county_weights_df, "pooled", "Model (pooled prior)")
    pooled_acc = accuracy_report(pooled_pred, county_actual, "county_fips", "Model (pooled Stan prior)")

    # 3. Model with state-stratified 2020 prior (in-sample)
    strat_pred = run_model(POLLS_2020, county_weights_df, "per-state", "Model (2020 state prior)")
    strat_acc = accuracy_report(strat_pred, county_actual, "county_fips", "Model (2020 state prior, in-sample)")

    # ── Accuracy table ─────────────────────────────────────────────────────────
    print("\n── County-level accuracy (vote-weighted) ──────────────────────────")
    print_accuracy_table([naive_acc, pooled_acc, strat_acc])
    print("\n  wMAE/wRMSE = vote-weighted mean abs/root-mean-sq error across counties")
    print("  Bias > 0 = over-predicts Democrat share")

    # ── Per-state breakdown ────────────────────────────────────────────────────
    print("\n── State-level accuracy ────────────────────────────────────────────")
    print(f"  {'State':<6}  {'Naive poll':>10}  {'Model(pooled)':>14}  {'Model(2020)':>12}  {'Actual':>8}")
    print("  " + "-" * 56)

    for state in ["FL", "GA", "AL"]:
        def state_pred(pred_df: pd.DataFrame) -> float:
            state_counties = pred_df[pred_df["state_abbr"] == state]
            state_actual_row = county_actual[county_actual["state_abbr"] == state]
            # Vote-weighted average
            merged = state_counties.merge(state_actual_row[["county_fips", "actual_total"]], on="county_fips")
            w = merged["actual_total"] / merged["actual_total"].sum()
            return float((merged["pred_dem_share"] * w).sum())

        naive_s  = state_pred(naive_pred)
        pooled_s = state_pred(pooled_pred)
        strat_s  = state_pred(strat_pred)
        actual_s = ACTUAL_2020_STATE[state]

        print(f"  {state:<6}  {naive_s:.1%}  {pooled_s:.1%}  {strat_s:.1%}  {actual_s:.1%}")

    # ── Community posteriors for FL (pooled prior) ─────────────────────────────
    print("\n── FL community posteriors: pooled prior + 2020 polls ──────────────")
    _, Sigma_prior = load_prior(state=None)
    mu_prior, _   = load_prior(state=None)
    fl_poll = [p for p in POLLS_2020 if p.geography == "FL"]
    posterior_fl = bayesian_poll_update(mu_prior, Sigma_prior, fl_poll)
    df_fl = posterior_fl.to_dataframe()

    print(f"  {'Community':<28}  {'Prior':>8}  {'Posterior':>10}  {'Shift':>8}")
    print("  " + "-" * 60)
    for k, (_, row) in enumerate(df_fl.iterrows()):
        shift = row["mu_post"] - mu_prior[k]
        shift_str = f"{shift:+.1%}" if abs(shift) > 0.001 else "  —"
        print(f"  {row['label']:<28}  {mu_prior[k]:.1%}  {row['mu_post']:.1%}  {shift_str:>8}")


if __name__ == "__main__":
    main()
