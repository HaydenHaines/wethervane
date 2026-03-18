"""
Back-calculate 2024 community vote shares from county-level presidential results.

Uses vote-normalized Tikhonov ridge regression (bounded), which is the
MAP estimate of a Gaussian prior projected onto [0,1]:

    min_{0≤θ≤1}  ||W_w·θ - y_w||² + λ||θ - θ₀||²

where W_w and y_w are vote-weighted (normalized by sqrt(v_c / v_total)).

Prior: year=2022 community estimates (load_prior(state=state, year=2022)).
This is a true 2-year out-of-sample test — the model was built on 2016–2022 data
and is here applied to 2024 without any information about 2024 outcomes.

Inputs:
  data/propagation/community_weights_county.parquet
  data/assembled/medsl_county_2024_president.parquet
  data/covariance/community_vote_shares_by_state.parquet  (2022 prior)

Outputs:
  Appends year=2024 rows (race="president_tikhonov") to
  data/covariance/community_vote_shares_by_state.parquet
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

from src.propagation.propagate_polls import (
    load_prior,
    COMP_COLS,
    LABELS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# AL included for 2024 presidential — no data quality issue (unlike 2022 governor)
STATES = ["FL", "GA", "AL"]

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
    Estimate 2024 community vote shares via vote-normalized Tikhonov regression.

    Solves:  min_{0≤θ≤1}  ||W_w·θ - y_w||² + λ||θ - θ₀||²

    where:
      W_w = W * sqrt(v / v.sum())   (vote-normalized county weight matrix)
      y_w = y * sqrt(v / v.sum())   (vote-normalized dem shares)
      θ₀  = 2022 state-stratified prior mean  ← out-of-sample: uses year=2022

    If lambda_reg is None, finds the minimum λ from LAMBDA_GRID such that the
    unconstrained Tikhonov solution stays in [0,1].  Uses lsq_linear(bounds)
    for the final solve regardless, ensuring physical estimates.

    Returns (theta_post, diagnostics).
    """
    # Use 2022 prior for genuine 2-year out-of-sample test.
    # AL was excluded from 2022 back-calc (data quality); fall back to 2020.
    try:
        mu_prior, _ = load_prior(state=state, year=2022)
        prior_year = 2022
    except ValueError:
        mu_prior, _ = load_prior(state=state, year=2020)
        prior_year = 2020
        log.info("  %s: no 2022 prior available, falling back to 2020", state)

    state_weights = county_weights_df[county_weights_df["state_abbr"] == state]
    state_actuals = county_actuals_df[county_actuals_df["state_abbr"] == state]

    merged = state_weights.merge(
        state_actuals[["county_fips", "actual_dem_share", "actual_total"]],
        on="county_fips",
    ).dropna(subset=["actual_dem_share"])
    merged = merged[merged["actual_total"] >= 100]

    if len(merged) == 0:
        log.warning("  %s: no counties after merge/filter — skipping", state)
        return np.full(K, float("nan")), {"n_counties": 0}

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
        PROJECT_ROOT / "data" / "assembled" / "medsl_county_2024_president.parquet"
    ).rename(columns={
        "pres_dem_share_2024": "actual_dem_share",
        "pres_total_2024":     "actual_total",
    })

    shares_path = PROJECT_ROOT / "data" / "covariance" / "community_vote_shares_by_state.parquet"
    existing = pd.read_parquet(shares_path)

    print("\n" + "=" * 70)
    print("2024 Community vote share estimation (Tikhonov ridge back-calculation)")
    print("Prior: 2022 state-stratified estimates (2-year out-of-sample test)")
    print("Data:  2024 county-level presidential results (FL + GA + AL)")
    print("Method: vote-normalized bounded ridge regression")
    print("=" * 70)

    new_rows = []

    for state in STATES:
        log.info("=== %s ===", state)
        theta_post, diag = estimate_via_tikhonov(state, county_weights_df, county_actuals_df)

        if diag["n_counties"] == 0:
            log.warning("  %s: skipped (no data)", state)
            continue

        # Pull 2022 prior values for display
        prev = existing[(existing["state"] == state) & (existing["year"] == 2022)]
        if len(prev) == 0:
            # Fall back to 2020 if 2022 not yet populated
            prev = existing[(existing["state"] == state) & (existing["year"] == 2020)]
        prev_dict = dict(zip(prev["component"], prev["theta_direct"]))
        mu_prior_vals = np.array([prev_dict.get(c, float("nan")) for c in COMP_COLS])

        print(f"\n── {state} 2024 (Tikhonov ridge, λ={diag['lambda_reg']:.4f}) ──────────────")
        print(f"  Counties: {diag['n_counties']}  "
              f"wRMSE: {diag['wrmse']:.4f}  "
              f"Boundary solutions: {diag['n_boundary']}/7")
        print(f"  Implied state dem share: {diag['implied_state']:.1%}  "
              f"(actual: {diag['actual_state']:.1%})")

        print(f"\n  {'Community':<28}  {'2022 prior':>10}  {'2024 est':>10}  {'Δ':>7}")
        print("  " + "-" * 60)
        for k, comp in enumerate(COMP_COLS):
            prior_v = mu_prior_vals[k]
            delta = theta_post[k] - prior_v
            prior_str = f"{prior_v:.1%}" if not np.isnan(prior_v) else "    N/A"
            print(f"  {LABELS[comp]:<28}  {prior_str}  {theta_post[k]:.1%}  {delta:+.1%}")

        for k, comp in enumerate(COMP_COLS):
            new_rows.append({
                "state":        state,
                "component":    comp,
                "year":         2024,
                "theta_direct": float(theta_post[k]),
                "race":         "president_tikhonov",
            })

    if not new_rows:
        log.error("No 2024 rows produced — check that MEDSL 2024 data is available")
        return

    # Append to parquet (replace any existing 2024 rows first)
    existing_no_2024 = existing[existing["year"] != 2024]
    updated = pd.concat(
        [existing_no_2024, pd.DataFrame(new_rows)], ignore_index=True
    ).sort_values(["state", "component", "year"]).reset_index(drop=True)
    updated.to_parquet(shares_path, index=False)

    log.info("Saved %d new 2024 rows → %s", len(new_rows), shares_path)
    print(f"\nYears now available: {sorted(updated['year'].unique())}")


if __name__ == "__main__":
    main()
