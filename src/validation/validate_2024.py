"""
Validation Test 3: 2024 Presidential election — FL, GA, and AL.

Out-of-sample test: validate 2022 prior → 2024 state polls → actual 2024
county results.

The 2022 prior was built from 2016+2018+2020 historical data. Feeding 2024
final RCP poll averages and comparing against actual 2024 MEDSL county results
gives a 4-year-stale prior forecast. This tests how well community structure
holds up across a full election cycle.

Key contrasts:
  FL: polls overestimated Harris by ~5pp — mirrors the 2022 pattern, suggesting
      a durable rightward shift in FL's Hispanic/working-class communities.
  GA: polls slightly overestimated Harris (~0.6pp) — closer to accurate.
  AL: sparse polling; used approximate two-party share.

Poll inputs (approximate RCP final two-party dem_share):
  FL: 0.474  (Harris ~47.4%, Trump ~52.6% two-party — polls significantly
              overestimated Harris; actual was 42.5%)
  GA: 0.504  (Harris ~50.4%, Trump ~49.6% two-party — polls slightly
              overestimated Harris; actual was 49.1%)
  AL: 0.380  (Harris ~38% — sparse polling; actual was 35.0%)

Actual 2024 state results (two-party dem_share):
  FL: 0.4248   GA: 0.4910   AL: 0.3502

NOTE: county_actuals_from_vest is not used here because no VEST 2024 data
exists. County actuals come directly from the MEDSL 2024 presidential parquet.
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
    predict_from_posterior,
    print_accuracy_table,
)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── 2024 polling averages ──────────────────────────────────────────────────────
# Approximate RCP final averages (two-party dem_share)
# FL: polls had ~47.4% — actual 42.5%; polls missed D by ~5pp (same pattern as 2022)
# GA: polls had ~50.4% — actual 49.1%; small miss
# AL: polls had ~38%   — actual 35.0%; sparse data, larger uncertainty
POLLS_2024 = [
    PollObservation("FL", dem_share=0.474, n_sample=1100,
                    race="2024 FL President", date="2024-11-05",
                    pollster="RCP avg", geo_level="state"),
    PollObservation("GA", dem_share=0.504, n_sample=900,
                    race="2024 GA President", date="2024-11-05",
                    pollster="RCP avg", geo_level="state"),
    PollObservation("AL", dem_share=0.380, n_sample=500,
                    race="2024 AL President", date="2024-11-05",
                    pollster="RCP avg", geo_level="state"),
]

ACTUAL_2024_STATE = {"FL": 0.4248, "GA": 0.4910, "AL": 0.3502}

STATES = ["FL", "GA", "AL"]


def load_2024_county_actuals() -> pd.DataFrame:
    """
    Load 2024 MEDSL county presidential results (FL, GA, AL).

    Renames columns to the standard actual_* format expected by accuracy_report.
    """
    path = PROJECT_ROOT / "data" / "assembled" / "medsl_county_2024_president.parquet"
    df = pd.read_parquet(path)
    df = df.rename(columns={
        "pres_dem_2024":       "actual_dem",
        "pres_rep_2024":       "actual_rep",
        "pres_total_2024":     "actual_total",
        "pres_dem_share_2024": "actual_dem_share",
    })
    return df


def naive_poll_prediction(
    county_weights_df: pd.DataFrame,
    polls: list[PollObservation],
) -> pd.DataFrame:
    """Uniform-swing baseline: assign state poll share to every county."""
    poll_map = {p.geography: p.dem_share for p in polls}
    result = county_weights_df[["county_fips", "state_abbr"]].copy()
    result["pred_dem_share"] = result["state_abbr"].map(poll_map)
    result["pred_std"] = 0.0
    return result


def run_model_per_state(
    polls: list[PollObservation],
    county_weights_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run Stage 4 + Stage 5 with 2022 state-stratified prior.

    Explicitly uses year=2022 so this is a true out-of-sample forecast.
    """
    _, Sigma_prior = load_prior(state=None)
    frames = []
    for poll in polls:
        state = poll.geography
        # Explicitly use year=2022 — genuine out-of-sample test.
        # AL has no 2022 estimates (excluded due to data quality); fall back to 2020.
        try:
            mu_prior, _ = load_prior(state=state, year=2022)
        except ValueError:
            mu_prior, _ = load_prior(state=state, year=2020)
        posterior = bayesian_poll_update(mu_prior, Sigma_prior, [poll])
        state_counties = county_weights_df[county_weights_df["state_abbr"] == state]
        pred = predict_from_posterior(posterior, state_counties, "county_fips")
        frames.append(pred)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    print("\n" + "=" * 70)
    print("Validation Test 3: 2024 Presidential (FL + GA + AL)")
    print("Prior: 2022 state-stratified community estimates (2-year lag)")
    print("=" * 70)

    # ── Load data ──────────────────────────────────────────────────────────────
    county_weights_df = pd.read_parquet(
        PROJECT_ROOT / "data" / "propagation" / "community_weights_county.parquet"
    )
    county_weights_df = county_weights_df[
        county_weights_df["state_abbr"].isin(STATES)
    ]
    county_actual = load_2024_county_actuals()

    print(f"\nLoaded: {len(county_weights_df)} counties (FL+GA+AL), "
          f"{len(county_actual)} with actuals")

    # ── Print poll inputs vs actuals ───────────────────────────────────────────
    print("\nPoll inputs vs. actual 2024 results:")
    print(f"  {'State':<6}  {'Poll avg':>10}  {'Actual':>10}  {'Poll error':>12}")
    print("  " + "-" * 44)
    for p in POLLS_2024:
        actual = ACTUAL_2024_STATE[p.geography]
        error = p.dem_share - actual
        print(f"  {p.geography:<6}  {p.dem_share:.1%}  {actual:.1%}  {error:+.1%}")
    print("\n  (FL polls again dramatically overestimated Democrats — "
          "consistent with 2022 structural shift)")

    # ── Run all models ─────────────────────────────────────────────────────────
    print("\nRunning county-level predictions...")

    naive_pred = naive_poll_prediction(county_weights_df, POLLS_2024)
    naive_acc  = accuracy_report(naive_pred, county_actual, "county_fips",
                                 "Naive poll (uniform swing)")

    model_pred = run_model_per_state(POLLS_2024, county_weights_df)
    model_acc  = accuracy_report(model_pred, county_actual, "county_fips",
                                 "Model (2022 state prior)")

    # ── Overall accuracy table ─────────────────────────────────────────────────
    print("\n── County-level accuracy (vote-weighted) ──────────────────────────")
    print_accuracy_table([naive_acc, model_acc])
    print("\n  Note: model uses 2022 prior — FL structural shift in c6 (Hispanic)")
    print("  and c5 (working-class homeowner) may persist as systematic bias.")

    # ── Per-state state-level predictions ─────────────────────────────────────
    print("\n── State-level accuracy ────────────────────────────────────────────")
    print(f"  {'State':<6}  {'Naive poll':>10}  {'Model':>10}  {'Actual':>8}  {'Bias':>8}")
    print("  " + "-" * 50)

    for state in STATES:
        def state_pred_weighted(pred_df: pd.DataFrame, _state: str = state) -> float:
            s_pred = pred_df[pred_df["state_abbr"] == _state]
            s_act  = county_actual[county_actual["state_abbr"] == _state]
            merged = s_pred.merge(s_act[["county_fips", "actual_total"]], on="county_fips")
            if len(merged) == 0:
                return float("nan")
            w = merged["actual_total"] / merged["actual_total"].sum()
            return float((merged["pred_dem_share"] * w).sum())

        naive_s  = state_pred_weighted(naive_pred)
        model_s  = state_pred_weighted(model_pred)
        actual_s = ACTUAL_2024_STATE[state]
        bias_s   = model_s - actual_s
        print(f"  {state:<6}  {naive_s:.1%}  {model_s:.1%}  {actual_s:.1%}  {bias_s:+.1%}")

    # ── Community posteriors: FL ───────────────────────────────────────────────
    print("\n── FL community posteriors: 2022 prior + 2024 poll ─────────────────")
    print("  (Shift = what the 2024 FL poll implies vs the 2022 baseline)")
    _, Sigma_prior = load_prior(state=None)
    mu_fl_2022, _  = load_prior(state="FL", year=2022)
    fl_poll_2024   = [p for p in POLLS_2024 if p.geography == "FL"]
    posterior_fl   = bayesian_poll_update(mu_fl_2022, Sigma_prior, fl_poll_2024)
    df_fl          = posterior_fl.to_dataframe()

    print(f"  {'Community':<28}  {'2022 prior':>10}  {'2024 posterior':>14}  {'Shift':>8}")
    print("  " + "-" * 66)
    for k, (_, row) in enumerate(df_fl.iterrows()):
        shift = row["mu_post"] - mu_fl_2022[k]
        shift_str = f"{shift:+.1%}" if abs(shift) > 0.001 else "  —"
        print(f"  {row['label']:<28}  {mu_fl_2022[k]:.1%}  {row['mu_post']:.1%}  {shift_str:>8}")

    # ── Community posteriors: GA ───────────────────────────────────────────────
    print("\n── GA community posteriors: 2022 prior + 2024 poll ─────────────────")
    mu_ga_2022, _  = load_prior(state="GA", year=2022)
    ga_poll_2024   = [p for p in POLLS_2024 if p.geography == "GA"]
    posterior_ga   = bayesian_poll_update(mu_ga_2022, Sigma_prior, ga_poll_2024)
    df_ga          = posterior_ga.to_dataframe()

    print(f"  {'Community':<28}  {'2022 prior':>10}  {'2024 posterior':>14}  {'Shift':>8}")
    print("  " + "-" * 66)
    for k, (_, row) in enumerate(df_ga.iterrows()):
        shift = row["mu_post"] - mu_ga_2022[k]
        shift_str = f"{shift:+.1%}" if abs(shift) > 0.001 else "  —"
        print(f"  {row['label']:<28}  {mu_ga_2022[k]:.1%}  {row['mu_post']:.1%}  {shift_str:>8}")


if __name__ == "__main__":
    main()
