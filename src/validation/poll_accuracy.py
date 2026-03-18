"""
Stage 5 utilities: translate community posteriors to geographic predictions,
and measure accuracy against known election results.

predict_from_posterior()  → county/state-level predicted dem_share from W·μ_post
county_actuals_from_vest() → county-level actual dem_share from VEST tract data
accuracy_report()         → MAE, RMSE, correlation, bias

Designed to be called from validate_historical.py or any validation script.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Import propagate_polls from the sibling package
import sys
sys.path.insert(0, str(Path(__file__).parents[2]))
from src.propagation.propagate_polls import CommunityPosterior, COMP_COLS

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[2]
STATE_FIPS = {"01": "AL", "12": "FL", "13": "GA"}


# ── Stage 5: predictions ──────────────────────────────────────────────────────


def predict_from_posterior(
    posterior: CommunityPosterior,
    weights_df: pd.DataFrame,
    geo_col: str,
) -> pd.DataFrame:
    """
    Stage 5: predict dem_share for each geographic unit from community posterior.

    y = W · μ_post
    Var(y) = W · Σ_post · Wᵀ  (diagonal elements only — per-unit predictive variance)

    Args:
      posterior:   CommunityPosterior from Stage 4
      weights_df:  DataFrame with geo_col + COMP_COLS (community weight matrix)
      geo_col:     column name for the geographic ID (e.g. "county_fips")

    Returns:
      DataFrame with geo_col, pred_dem_share, pred_std, pred_lo90, pred_hi90
    """
    mu = posterior.mu                   # (K,)
    Sigma = posterior.sigma             # (K, K)
    W = weights_df[COMP_COLS].values    # (n_geo, K)

    pred_mean = W @ mu                  # (n_geo,)

    # Per-unit predictive variance: diag(W Σ Wᵀ)
    # Equivalent to sum over k,j of W[i,k] * Sigma[k,j] * W[i,j]
    pred_var = np.einsum("ik,kl,il->i", W, Sigma, W)  # (n_geo,)
    pred_std = np.sqrt(np.maximum(pred_var, 0))

    keep_cols = [geo_col]
    if "state_abbr" in weights_df.columns:
        keep_cols.append("state_abbr")

    result = weights_df[keep_cols].copy().reset_index(drop=True)
    result["pred_dem_share"] = pred_mean
    result["pred_std"] = pred_std
    result["pred_lo90"] = pred_mean - 1.645 * pred_std
    result["pred_hi90"] = pred_mean + 1.645 * pred_std

    return result


# ── Actual results from VEST ──────────────────────────────────────────────────


def county_actuals_from_vest(year: int, prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate VEST tract-level data to county-level and state-level actual dem_share.

    Returns:
      county_df:  county_fips, state_abbr, actual_dem_share, actual_total
      state_df:   state_abbr, actual_dem_share, actual_total
    """
    vest_path = PROJECT_ROOT / "data" / "assembled" / f"vest_tracts_{year}.parquet"
    df = pd.read_parquet(vest_path)

    dem_col   = f"{prefix}_dem_{year}"
    rep_col   = f"{prefix}_rep_{year}"
    total_col = f"{prefix}_total_{year}"

    df["county_fips"] = df["tract_geoid"].str[:5]
    df["state_abbr"]  = df["tract_geoid"].str[:2].map(STATE_FIPS)

    # County aggregation
    county = (
        df.groupby(["county_fips", "state_abbr"])
        .agg(
            actual_dem   = (dem_col,   "sum"),
            actual_rep   = (rep_col,   "sum"),
            actual_total = (total_col, "sum"),
        )
        .reset_index()
    )
    denom = county["actual_dem"] + county["actual_rep"]
    county["actual_dem_share"] = county["actual_dem"] / denom.replace(0, np.nan)

    # State aggregation
    state = (
        county.groupby("state_abbr")
        .agg(
            actual_dem   = ("actual_dem",   "sum"),
            actual_rep   = ("actual_rep",   "sum"),
            actual_total = ("actual_total", "sum"),
        )
        .reset_index()
    )
    denom_s = state["actual_dem"] + state["actual_rep"]
    state["actual_dem_share"] = state["actual_dem"] / denom_s

    return county, state


# ── Accuracy metrics ──────────────────────────────────────────────────────────


def accuracy_report(
    pred_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    geo_col: str,
    label: str = "",
) -> dict:
    """
    Compute accuracy metrics for predicted vs. actual dem_share.

    Vote-weighted metrics (wMAE, wRMSE) down-weight small precincts.
    Mean bias > 0 means model over-predicts Democratic share.
    """
    merged = pred_df.merge(actual_df[[geo_col, "actual_dem_share", "actual_total"]], on=geo_col)
    merged = merged.dropna(subset=["pred_dem_share", "actual_dem_share"])

    err = merged["pred_dem_share"] - merged["actual_dem_share"]
    w   = merged["actual_total"] / merged["actual_total"].sum()

    metrics = {
        "label":       label,
        "n":           len(merged),
        "mae":         float(abs(err).mean()),
        "rmse":        float(np.sqrt((err**2).mean())),
        "wmae":        float((abs(err) * w).sum()),
        "wrmse":       float(np.sqrt(((err**2) * w).sum())),
        "corr":        float(merged["pred_dem_share"].corr(merged["actual_dem_share"])),
        "mean_bias":   float(err.mean()),   # + = over-predicts D; – = over-predicts R
    }
    return metrics


def print_accuracy_table(metrics_list: list[dict]) -> None:
    """Print a formatted comparison table for multiple model variants."""
    print(f"\n{'Model':<32}  {'wMAE':>6}  {'wRMSE':>6}  {'Corr':>5}  {'Bias':>7}  {'N':>5}")
    print("-" * 70)
    for m in metrics_list:
        print(
            f"  {m['label']:<30}  "
            f"{m['wmae']:.3f}  {m['wrmse']:.3f}  "
            f"{m['corr']:.3f}  {m['mean_bias']:+.3f}  {m['n']:>5}"
        )
