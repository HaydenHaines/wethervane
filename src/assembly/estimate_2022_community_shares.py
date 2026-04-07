"""
Back-calculate 2022 community vote shares from county-level results.

Uses vote-normalized Tikhonov ridge regression (bounded), which is the
MAP estimate of a Gaussian prior projected onto [0,1]:

    min_{0≤θ≤1}  ||W_w·θ - y_w||² + λ||θ - θ₀||²

where W_w and y_w are vote-weighted (normalized by sqrt(v_c / v_total)).
Vote normalization prevents large counties from overwhelming the prior
the way raw sampling-noise σ_c = sqrt(p(1-p)/n) would — a 700K-vote county
would otherwise contribute 350M× more information than the Stan prior.

The adaptive lambda search finds the minimum regularization that keeps
all estimates physically valid (in [0,1]), maximizing data fidelity.

Why not pure Bayesian (bayesian_poll_update)?
The Stan Sigma prior (diag ≈ 0.002) was fit from 3 state-level elections
and gives prior precision ≈ 500.  With n_votes ≈ 100K per county,
σ_c ≈ 0.002, so data precision ≈ 250,000 per county × 67 counties ≈ 1340×
the prior.  The unconstrained posterior goes wildly outside [0,1].

Alabama: excluded — MEDSL 2022 AL data quality issue.

Inputs:
  data/propagation/community_weights_county.parquet
  data/assembled/medsl_county_2022_governor.parquet
  data/covariance/community_vote_shares_by_state.parquet  (2020 prior)

Outputs:
  Appends year=2022 rows to data/covariance/community_vote_shares_by_state.parquet
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.propagation.propagate_polls import (  # noqa: E402
    COMP_COLS,
    LABELS,
    load_prior,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

STATES = ["FL", "GA"]   # AL excluded — MEDSL data quality issue

K = len(COMP_COLS)

# Lambda search grid: smallest to largest
LAMBDA_GRID = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]


def estimate_via_tikhonov(
    state: str,
    county_weights_df: pd.DataFrame,
    county_actuals_df: pd.DataFrame,
    lambda_reg: float | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Estimate 2022 community vote shares via vote-normalized Tikhonov regression.

    Solves:  min_{0≤θ≤1}  ||W_w·θ - y_w||² + λ||θ - θ₀||²

    where:
      W_w = W * sqrt(v / v.sum())   (vote-normalized county weight matrix)
      y_w = y * sqrt(v / v.sum())   (vote-normalized dem shares)
      θ₀  = 2020 state-stratified prior mean

    If lambda_reg is None, finds the minimum λ from LAMBDA_GRID such that the
    unconstrained Tikhonov solution stays in [0,1].  Uses lsq_linear(bounds)
    for the final solve regardless, ensuring physical estimates.

    Returns (theta_post, diagnostics).
    """
    mu_prior, _ = load_prior(state=state, year=2020)

    state_weights = county_weights_df[county_weights_df["state_abbr"] == state]
    state_actuals = county_actuals_df[county_actuals_df["state_abbr"] == state]

    merged = state_weights.merge(
        state_actuals[["county_fips", "actual_dem_share", "actual_total"]],
        on="county_fips",
    ).dropna(subset=["actual_dem_share"])
    merged = merged[merged["actual_total"] >= 100]

    W = merged[COMP_COLS].values
    y = merged["actual_dem_share"].values
    v = merged["actual_total"].values.astype(float)

    # Vote normalization: cap max county influence at sqrt(v_c/total)
    sqrt_w = np.sqrt(v / v.sum())
    W_w = W * sqrt_w[:, None]
    y_w = y * sqrt_w

    # ── Adaptive lambda search ──────────────────────────────────────────────
    if lambda_reg is None:
        for lam in LAMBDA_GRID:
            W_aug = np.vstack([W_w, np.sqrt(lam) * np.eye(K)])
            y_aug = np.hstack([y_w, np.sqrt(lam) * mu_prior])
            theta_test, _, _, _ = np.linalg.lstsq(W_aug, y_aug, rcond=None)
            if np.all((theta_test >= 0) & (theta_test <= 1)):
                lambda_reg = lam
                log.info("  %s: adaptive lambda = %.4f (unconstrained solution in [0,1])",
                         state, lambda_reg)
                break
        else:
            lambda_reg = LAMBDA_GRID[-1]
            log.warning("  %s: no lambda in grid gives fully in-bounds unconstrained "
                        "solution; using lambda = %.1f", state, lambda_reg)

    # ── Bounded solve (final) ───────────────────────────────────────────────
    W_aug = np.vstack([W_w, np.sqrt(lambda_reg) * np.eye(K)])
    y_aug = np.hstack([y_w, np.sqrt(lambda_reg) * mu_prior])
    result = lsq_linear(W_aug, y_aug, bounds=(0, 1), method="bvls")
    theta = result.x

    # ── Diagnostics ─────────────────────────────────────────────────────────
    y_hat = W @ theta
    residuals = y - y_hat
    wrmse = float(np.sqrt(np.average(residuals**2, weights=v)))
    implied_state = float(np.average(y_hat, weights=v))
    actual_state = float(np.average(y, weights=v))
    n_boundary = int(np.sum((theta < 0.001) | (theta > 0.999)))

    diag = {
        "n_counties": len(merged),
        "wrmse": wrmse,
        "implied_state": implied_state,
        "actual_state": actual_state,
        "lambda_reg": lambda_reg,
        "n_boundary": n_boundary,
    }
    return theta, diag


def main() -> None:
    county_weights_df = pd.read_parquet(
        PROJECT_ROOT / "data" / "propagation" / "community_weights_county.parquet"
    )
    county_actuals_df = pd.read_parquet(
        PROJECT_ROOT / "data" / "assembled" / "medsl_county_2022_governor.parquet"
    ).rename(columns={
        "gov_dem_share_2022": "actual_dem_share",
        "gov_total_2022":     "actual_total",
    })

    shares_path = PROJECT_ROOT / "data" / "covariance" / "community_vote_shares_by_state.parquet"
    existing = pd.read_parquet(shares_path)

    print("\n" + "=" * 70)
    print("2022 Community vote share estimation (Tikhonov ridge back-calculation)")
    print("Prior: 2020 state-stratified estimates")
    print("Data:  2022 county-level governor results (67 FL + 159 GA counties)")
    print("Method: vote-normalized bounded ridge regression")
    print("=" * 70)

    new_rows = []

    for state in STATES:
        log.info("=== %s ===", state)
        theta_post, diag = estimate_via_tikhonov(state, county_weights_df, county_actuals_df)

        prev = existing[(existing["state"] == state) & (existing["year"] == 2020)]
        prev_dict = dict(zip(prev["component"], prev["theta_direct"]))
        mu_prior_vals = np.array([prev_dict.get(c, np.nan) for c in COMP_COLS])

        print(f"\n── {state} 2022 (Tikhonov ridge, λ={diag['lambda_reg']:.4f}) ──────────────")
        print(f"  Counties: {diag['n_counties']}  "
              f"wRMSE: {diag['wrmse']:.4f}  "
              f"Boundary solutions: {diag['n_boundary']}/7")
        print(f"  Implied state dem share: {diag['implied_state']:.1%}  "
              f"(actual: {diag['actual_state']:.1%})")

        print(f"\n  {'Community':<28}  {'2020 prior':>10}  {'2022 est':>10}  {'Δ':>7}")
        print("  " + "-" * 60)
        for k, comp in enumerate(COMP_COLS):
            prior_v = mu_prior_vals[k]
            delta = theta_post[k] - prior_v
            print(f"  {LABELS[comp]:<28}  {prior_v:.1%}  {theta_post[k]:.1%}  {delta:+.1%}")

        for k, comp in enumerate(COMP_COLS):
            new_rows.append({
                "state":        state,
                "component":    comp,
                "year":         2022,
                "theta_direct": float(theta_post[k]),
                "race":         "governor_tikhonov",
            })

    # Append to parquet
    existing_no_2022 = existing[existing["year"] != 2022]
    updated = pd.concat(
        [existing_no_2022, pd.DataFrame(new_rows)], ignore_index=True
    ).sort_values(["state", "component", "year"]).reset_index(drop=True)
    updated.to_parquet(shares_path, index=False)

    log.info("Saved %d new 2022 rows → %s", len(new_rows), shares_path)
    print(f"\nYears now available: {sorted(updated['year'].unique())}")


if __name__ == "__main__":
    main()
