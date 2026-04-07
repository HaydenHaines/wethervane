"""
Holdout validation: 2020 Presidential election with T=2 prior.

The Stan covariance model was re-run on 2016+2018 only (T=2).
This script uses the resulting posterior as the prior — the model
genuinely has never seen 2020 data before we feed it the 2020 polls.

This is the cleanest out-of-sample test of the core hypothesis:
  "community structure discovered from non-political data,
   calibrated on 2016+2018 elections,
   can predict 2020 county-level vote shares
   when given only state-level poll averages as input"

Prior source: data/covariance/holdout_t2/   (T=2, 2016+2018 only)
Poll input:   RCP 2020 final averages        (same as validate_2020.py)
Ground truth: VEST 2020 county actuals       (held out from Stan training)

Compare against two baselines:
  1. Naive poll (uniform swing — no community structure)
  2. In-sample model (validate_2020.py result, for reference)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.propagation.propagate_polls import (  # noqa: E402
    COMP_COLS,
    LABELS,
    PollObservation,
    bayesian_poll_update,
)
from src.validation.poll_accuracy import (  # noqa: E402
    accuracy_report,
    county_actuals_from_vest,
    predict_from_posterior,
    print_accuracy_table,
)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

HOLDOUT_DIR = PROJECT_ROOT / "data" / "covariance" / "holdout_t2"

# Same 2020 polls as validate_2020.py
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

ACTUAL_2020_STATE = {"FL": 0.483, "GA": 0.501, "AL": 0.371}

# In-sample county correlation from validate_2020.py (pooled prior), for reference
INSAMPLE_CORR = 0.842
INSAMPLE_WMAE = 0.109


def load_holdout_prior() -> tuple[np.ndarray, np.ndarray]:
    """
    Load the T=2 holdout Stan posterior as prior.

    mu:    Stan posterior mean mu (pooled, 2016+2018 only)
    Sigma: Stan posterior mean covariance (2016+2018 only)
    """
    sigma_path   = HOLDOUT_DIR / "community_sigma.parquet"
    summary_path = HOLDOUT_DIR / "covariance_summary.parquet"

    Sigma = pd.read_parquet(sigma_path).values

    summary = pd.read_parquet(summary_path)
    mu_rows = summary[summary.index.str.startswith("mu[")]
    mu = mu_rows["Mean"].values

    return mu, Sigma


def naive_poll_prediction(
    county_weights_df: pd.DataFrame,
    polls: list[PollObservation],
) -> pd.DataFrame:
    poll_map = {p.geography: p.dem_share for p in polls}
    result = county_weights_df[["county_fips", "state_abbr"]].copy()
    result["pred_dem_share"] = result["state_abbr"].map(poll_map)
    result["pred_std"] = 0.0
    return result


def main() -> None:
    print("\n" + "=" * 70)
    print("Holdout Validation: 2020 Presidential — T=2 Prior (2016+2018 only)")
    print("The model has NEVER seen 2020 data before this run.")
    print("=" * 70)

    if not HOLDOUT_DIR.exists():
        print(f"\nERROR: Holdout model not found at {HOLDOUT_DIR}")
        print("Run: python src/covariance/run_covariance_model.py --holdout")
        return

    # ── Load data ──────────────────────────────────────────────────────────────
    mu_prior, Sigma_prior = load_holdout_prior()
    county_weights_df = pd.read_parquet(
        PROJECT_ROOT / "data" / "propagation" / "community_weights_county.parquet"
    )
    county_actual, state_actual = county_actuals_from_vest(2020, "pres")

    print(f"\nT=2 prior loaded from: {HOLDOUT_DIR}")
    print(f"Prior mu (pooled, 2016+2018): {np.round(mu_prior, 3)}")

    # ── Print prior mu vs 2020 actual community estimates ─────────────────────
    state_shares = pd.read_parquet(
        PROJECT_ROOT / "data" / "covariance" / "community_vote_shares_by_state.parquet"
    )
    print("\n── T=2 prior vs 2020 actual community vote shares (pooled) ─────────")
    print(f"  {'Community':<28}  {'T=2 prior':>10}  {'2020 actual':>12}  {'Diff':>6}")
    print("  " + "-" * 62)

    actual_2020_pooled = (
        state_shares[state_shares["year"] == 2020]
        .groupby("component")["theta_direct"]
        .mean()
    )
    for k, comp in enumerate(COMP_COLS):
        act = actual_2020_pooled.get(comp, np.nan)
        diff = mu_prior[k] - act if not np.isnan(act) else np.nan
        diff_str = f"{diff:+.1%}" if not np.isnan(diff) else "  n/a"
        print(f"  {LABELS[comp]:<28}  {mu_prior[k]:.1%}  {act:.1%}  {diff_str}")

    # ── Run models ─────────────────────────────────────────────────────────────
    print("\nRunning county-level predictions (pooled T=2 prior + 2020 state polls)...")

    # Naive baseline
    naive_pred = naive_poll_prediction(county_weights_df, POLLS_2020)
    naive_acc  = accuracy_report(naive_pred, county_actual, "county_fips", "Naive poll (uniform swing)")

    # Holdout model: single pooled update across all 3 states simultaneously
    posterior_holdout = bayesian_poll_update(mu_prior, Sigma_prior, POLLS_2020)
    holdout_pred = predict_from_posterior(posterior_holdout, county_weights_df, "county_fips")
    holdout_acc  = accuracy_report(holdout_pred, county_actual, "county_fips",
                                   "Holdout model (T=2 prior, pooled)")

    # ── Accuracy table ─────────────────────────────────────────────────────────
    print("\n── County-level accuracy (vote-weighted) ──────────────────────────")
    print_accuracy_table([naive_acc, holdout_acc])

    print("\n  For reference — in-sample (T=3 pooled prior from validate_2020.py):")
    print(f"    wMAE={INSAMPLE_WMAE:.3f}  Corr={INSAMPLE_CORR:.3f}")
    print()
    print("  KEY TEST: Does holdout correlation stay close to in-sample?")
    delta_corr = holdout_acc["corr"] - INSAMPLE_CORR
    print(f"  Holdout corr {holdout_acc['corr']:.3f} vs in-sample {INSAMPLE_CORR:.3f}  "
          f"(Δ = {delta_corr:+.3f})")
    if abs(delta_corr) < 0.10:
        print("  → Correlation holds up: community structure generalizes across election cycles.")
    else:
        print("  → Correlation drop ≥ 10pp: in-sample results were inflated by prior contamination.")

    # ── Per-state predictions ──────────────────────────────────────────────────
    print("\n── State-level accuracy ────────────────────────────────────────────")
    print(f"  {'State':<6}  {'Naive poll':>10}  {'Holdout':>10}  {'Actual':>8}  {'Bias':>8}")
    print("  " + "-" * 50)

    for state in ["FL", "GA", "AL"]:
        s_pred  = holdout_pred[holdout_pred["state_abbr"] == state]
        s_naive = naive_pred[naive_pred["state_abbr"] == state]
        s_act   = county_actual[county_actual["state_abbr"] == state]
        merged_h = s_pred.merge(s_act[["county_fips", "actual_total"]], on="county_fips")
        merged_n = s_naive.merge(s_act[["county_fips", "actual_total"]], on="county_fips")
        w = merged_h["actual_total"] / merged_h["actual_total"].sum()
        holdout_s = float((merged_h["pred_dem_share"] * w).sum())
        naive_s   = float((merged_n["pred_dem_share"] * merged_n["actual_total"] /
                           merged_n["actual_total"].sum()).sum())
        actual_s  = ACTUAL_2020_STATE[state]
        print(f"  {state:<6}  {naive_s:.1%}  {holdout_s:.1%}  {actual_s:.1%}  {holdout_s - actual_s:+.1%}")

    # ── Community posteriors after 2020 polls ──────────────────────────────────
    print("\n── Community posteriors: T=2 prior → after 2020 polls → 2020 actual ─")
    df_post = posterior_holdout.to_dataframe()
    print(f"  {'Community':<28}  {'T=2 prior':>10}  {'After poll':>10}  {'2020 actual':>12}")
    print("  " + "-" * 66)
    for k, (_, row) in enumerate(df_post.iterrows()):
        act = actual_2020_pooled.get(COMP_COLS[k], np.nan)
        act_str = f"{act:.1%}" if not np.isnan(act) else "  n/a"
        print(f"  {row['label']:<28}  {mu_prior[k]:.1%}  {row['mu_post']:.1%}  {act_str}")


if __name__ == "__main__":
    main()
