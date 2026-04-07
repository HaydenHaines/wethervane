"""
Validation Test 2: 2022 Gubernatorial election — FL and GA.

Out-of-sample test: our Stan covariance model was trained on 2016+2018+2020.
We use the 2020 state-stratified prior (most recent available) and feed
2022 final poll averages, then compare county predictions against actual
2022 MEDSL county results.

This is a genuine forecast simulation: the model does not know the 2022
result in advance. The prior is 2 years stale (2020 → 2022), which tests
how well community structure holds up over one election cycle.

Key contrast:
  FL: polls missed DeSantis by ~5pp — a structural shift in Hispanic/
      working-class communities that a stale 2020 prior may not capture
  GA: polls were accurate — a more stable electorate across cycles

Data sources:
  - County actuals: MEDSL 2022-elections-official GitHub repo
  - Polling averages: approximate RCP final (two-party dem_share)

NOTE — Alabama excluded: MEDSL AL 2022 data includes Will Ainsworth
(Republican primary/runoff candidate) incorrectly labeled as GEN stage,
inflating apparent Republican vote. Data quality is unreliable.

FL 2022 polls: Crist ~44%, DeSantis ~53% → poll dem_share ≈ 0.454
  Actual: Crist 40.2%  →  polls missed D by +5.2pp (large FL polling error)
GA 2022 polls: Abrams ~46%, Kemp ~52% → poll dem_share ≈ 0.469
  Actual: Abrams 46.8% →  polls essentially correct (+0.1pp)
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

# ── 2022 polling averages ──────────────────────────────────────────────────────
# Approximate RCP final averages (two-party dem_share)
# FL: DeSantis won by 19pp; polls had him up ~9pp — largest miss
# GA: Kemp won by ~7pp; polls had him up ~6pp — accurately predicted
POLLS_2022 = [
    PollObservation("FL", dem_share=0.454, n_sample=1100,
                    race="2022 FL Governor", date="2022-11-07",
                    pollster="RCP avg", geo_level="state"),
    PollObservation("GA", dem_share=0.469, n_sample=900,
                    race="2022 GA Governor", date="2022-11-07",
                    pollster="RCP avg", geo_level="state"),
]

ACTUAL_2022_STATE = {"FL": 0.402, "GA": 0.468}


def load_2022_county_actuals() -> pd.DataFrame:
    """Load 2022 MEDSL county governor results (FL + GA only)."""
    path = PROJECT_ROOT / "data" / "assembled" / "medsl_county_2022_governor.parquet"
    df = pd.read_parquet(path)
    # Exclude AL (data quality issue — primary candidate mislabeled as GEN)
    df = df[df["state_abbr"].isin(["FL", "GA"])].copy()
    df = df.rename(columns={
        "gov_dem_2022":       "actual_dem",
        "gov_rep_2022":       "actual_rep",
        "gov_total_2022":     "actual_total",
        "gov_dem_share_2022": "actual_dem_share",
    })
    return df


def naive_poll_prediction(
    county_weights_df: pd.DataFrame,
    polls: list[PollObservation],
) -> pd.DataFrame:
    poll_map = {p.geography: p.dem_share for p in polls}
    result = county_weights_df[["county_fips", "state_abbr"]].copy()
    result["pred_dem_share"] = result["state_abbr"].map(poll_map)
    result["pred_std"] = 0.0
    return result


def run_model_per_state(
    polls: list[PollObservation],
    county_weights_df: pd.DataFrame,
) -> pd.DataFrame:
    """Run Stage 4 + Stage 5 with state-stratified 2020 prior."""
    _, Sigma_prior = load_prior(state=None)
    frames = []
    for poll in polls:
        state = poll.geography
        mu_prior, _ = load_prior(state=state)
        posterior = bayesian_poll_update(mu_prior, Sigma_prior, [poll])
        state_counties = county_weights_df[county_weights_df["state_abbr"] == state]
        pred = predict_from_posterior(posterior, state_counties, "county_fips")
        frames.append(pred)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    print("\n" + "=" * 70)
    print("Validation Test 2: 2022 Gubernatorial (FL + GA)")
    print("Prior: 2020 state-stratified community estimates (2-year lag)")
    print("=" * 70)

    # ── Load data ──────────────────────────────────────────────────────────────
    county_weights_df = pd.read_parquet(
        PROJECT_ROOT / "data" / "propagation" / "community_weights_county.parquet"
    )
    county_weights_df = county_weights_df[
        county_weights_df["state_abbr"].isin(["FL", "GA"])
    ]
    county_actual = load_2022_county_actuals()

    print(f"\nLoaded: {len(county_weights_df)} counties (FL+GA), {len(county_actual)} with actuals")

    # ── Print poll inputs vs actuals ───────────────────────────────────────────
    print("\nPoll inputs vs. actual 2022 results:")
    print(f"  {'State':<6}  {'Poll avg':>10}  {'Actual':>10}  {'Poll error':>12}")
    print("  " + "-" * 44)
    for p in POLLS_2022:
        actual = ACTUAL_2022_STATE[p.geography]
        error = p.dem_share - actual
        print(f"  {p.geography:<6}  {p.dem_share:.1%}  {actual:.1%}  {error:+.1%}")
    print("\n  (FL polls dramatically overestimated Democrats — large structural shift)")

    # ── Run all models ─────────────────────────────────────────────────────────
    print("\nRunning county-level predictions...")

    naive_pred = naive_poll_prediction(county_weights_df, POLLS_2022)
    naive_acc  = accuracy_report(naive_pred, county_actual, "county_fips", "Naive poll (uniform swing)")

    model_pred = run_model_per_state(POLLS_2022, county_weights_df)
    model_acc  = accuracy_report(model_pred, county_actual, "county_fips", "Model (2020 state prior)")

    # ── Accuracy table ─────────────────────────────────────────────────────────
    print("\n── County-level accuracy (vote-weighted) ──────────────────────────")
    print_accuracy_table([naive_acc, model_acc])
    print("\n  Note: both models use the stale 2020 prior — FL structural shift")
    print("  (c6 Hispanic moving R) will appear as systematic positive bias.")

    # ── Per-state state-level predictions ─────────────────────────────────────
    print("\n── State-level accuracy ────────────────────────────────────────────")
    print(f"  {'State':<6}  {'Naive poll':>10}  {'Model':>10}  {'Actual':>8}  {'Bias':>8}")
    print("  " + "-" * 50)

    for state in ["FL", "GA"]:
        def state_pred_weighted(pred_df: pd.DataFrame) -> float:
            s_pred = pred_df[pred_df["state_abbr"] == state]
            s_act  = county_actual[county_actual["state_abbr"] == state]
            merged = s_pred.merge(s_act[["county_fips", "actual_total"]], on="county_fips")
            w = merged["actual_total"] / merged["actual_total"].sum()
            return float((merged["pred_dem_share"] * w).sum())

        naive_s = state_pred_weighted(naive_pred)
        model_s = state_pred_weighted(model_pred)
        actual_s = ACTUAL_2022_STATE[state]
        bias_s   = model_s - actual_s
        print(f"  {state:<6}  {naive_s:.1%}  {model_s:.1%}  {actual_s:.1%}  {bias_s:+.1%}")

    # ── Community posteriors: what did the 2022 polls imply? ──────────────────
    print("\n── FL community posteriors: 2020 prior + 2022 poll ─────────────────")
    print("  (Shift = what the 2022 FL poll implies vs the 2020 baseline)")
    _, Sigma_prior = load_prior(state=None)
    mu_fl_2020, _  = load_prior(state="FL")
    fl_poll_2022   = [p for p in POLLS_2022 if p.geography == "FL"]
    posterior_fl   = bayesian_poll_update(mu_fl_2020, Sigma_prior, fl_poll_2022)
    df_fl          = posterior_fl.to_dataframe()

    print(f"  {'Community':<28}  {'2020 prior':>10}  {'2022 posterior':>14}  {'Shift':>8}")
    print("  " + "-" * 66)
    for k, (_, row) in enumerate(df_fl.iterrows()):
        shift = row["mu_post"] - mu_fl_2020[k]
        shift_str = f"{shift:+.1%}" if abs(shift) > 0.001 else "  —"
        print(f"  {row['label']:<28}  {mu_fl_2020[k]:.1%}  {row['mu_post']:.1%}  {shift_str:>8}")

    print("\n── GA community posteriors: 2020 prior + 2022 poll ─────────────────")
    mu_ga_2020, _  = load_prior(state="GA")
    ga_poll_2022   = [p for p in POLLS_2022 if p.geography == "GA"]
    posterior_ga   = bayesian_poll_update(mu_ga_2020, Sigma_prior, ga_poll_2022)
    df_ga          = posterior_ga.to_dataframe()

    print(f"  {'Community':<28}  {'2020 prior':>10}  {'2022 posterior':>14}  {'Shift':>8}")
    print("  " + "-" * 66)
    for k, (_, row) in enumerate(df_ga.iterrows()):
        shift = row["mu_post"] - mu_ga_2020[k]
        shift_str = f"{shift:+.1%}" if abs(shift) > 0.001 else "  —"
        print(f"  {row['label']:<28}  {mu_ga_2020[k]:.1%}  {row['mu_post']:.1%}  {shift_str:>8}")


if __name__ == "__main__":
    main()
