"""
Stage 3 (initial): estimate community-type Democratic vote shares from
2020 presidential returns.

This is the first empirical test of the model's core hypothesis:
communities discovered from non-political data (demographics, commute,
education, occupation, age) predict political behavior.

Two-part approach:

1. NNLS R² — primary validation metric for hypothesis testing.
   Solves d ≈ W @ θ (vote-weighted NNLS). The R² measures the maximum
   predictive power of the community structure. Unrealistic individual
   θ values (e.g., c2=381%) signal that sparse components need
   regularization — exactly what the Stage 3 Bayesian model provides.
   R² is valid even when individual estimates are not.

2. Direct weighted mean — community political profiles for visualization.
   θ_k = Σ_i (v_i · w_ik · d_i) / Σ_i (v_i · w_ik)
   Always in [0,1]; interpretable as "average dem_share in tracts that
   belong to community k." More compressed than NNLS but unambiguous.

The θ_k vector is what the Stage 3 Bayesian model will eventually
estimate hierarchically across multiple election cycles. This script
gives us the 2020 single-election estimate as a validation baseline.

Usage:
  python -m src.covariance.estimate_community_votes          # default K=7
  python -m src.covariance.estimate_community_votes --k 11  # any K

Inputs:
  data/communities/tract_memberships_k{K}.parquet
  data/assembled/vest_tracts_2020.parquet

Outputs:
  data/covariance/community_vote_shares_2020_k{K}.parquet
  data/covariance/community_vote_shares_2020_k{K}.png

Previously: estimate_community_votes_k4.py, _k6.py, _k7_k9.py, _k11.py
All collapsed into this file (2026-03-30). Use --k argument to vary K.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import nnls

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "covariance"

# Default K — matches the legacy NMF K=7 model used when this script was written.
# Override at runtime with --k. The type-primary production model uses types.j
# from config/model.yaml (currently 100), but this script operates on the earlier
# NMF-based community memberships which only exist for small K values.
DEFAULT_K = 7

# Provisional labels keyed by K value, then by component name.
# Labels are interpretive summaries of NMF component profiles; absent labels
# fall back to the component name (e.g. "c8") so any K works without label data.
LABELS_BY_K: dict[int, dict[str, str]] = {
    4: {
        "c1": "White affluent (income+mgmt+owner+car)",
        "c2": "Black urban (transit)",
        "c3": "Asian/Knowledge worker (WFH+college+mgmt)",
        "c4": "Hispanic (low-income)",
    },
    6: {
        "c1": "White affluent (income+mgmt+car)",
        "c2": "Black urban (transit+age)",
        "c3": "Asian/Knowledge worker (mgmt+WFH+college)",
        "c4": "Homeowner (non-professional)",
        "c5": "Retiree (age+slight WFH)",
        "c6": "Hispanic",
    },
    7: {
        "c1": "White rural homeowner (older+WFH)",
        "c2": "Black urban (transit+income)",
        "c3": "Knowledge worker (mgmt+WFH+college)",
        "c4": "Asian",
        "c5": "Working-class homeowner (owner-occ)",
        "c6": "Hispanic low-income",
        "c7": "Generic suburban baseline",
    },
    9: {
        "c1": "White suburban (car-dependent)",
        "c2": "Black urban (car+transit+income)",
        "c3": "Retiree manager (mgmt+age, no commute)",
        "c4": "WFH urban mix (white+Hispanic+transit)",
        "c5": "Educated retiree (college+owner+age)",
        "c6": "Hispanic",
        "c7": "Asian",
        "c8": "Homeowner (non-professional)",
        "c9": "College-educated (no other signal)",
    },
    11: {
        "c1":  "White car-commute (working class)",
        "c2":  "Black urban (high income + transit)",
        "c3":  "Professional (mgmt + college + income)",
        "c4":  "Remote worker (WFH + high income)",
        "c5":  "Retiree (old age + owner-occ)",
        "c6":  "Hispanic low-income",
        "c7":  "Asian",
        "c8":  "High income (no occupation signal)",
        "c9":  "Homeowner (non-professional)",
        "c10": "Generic baseline (car commute only)",
        "c11": "Management class (no college signal)",
    },
}


def load_data(k: int) -> pd.DataFrame:
    """Join memberships + election returns on tract_geoid for the given K."""
    mem = pd.read_parquet(
        PROJECT_ROOT / "data" / "communities" / f"tract_memberships_k{k}.parquet"
    )
    vest = pd.read_parquet(
        PROJECT_ROOT / "data" / "assembled" / "vest_tracts_2020.parquet"
    )

    comp_cols = [f"c{i}" for i in range(1, k + 1)]
    df = mem.merge(vest[["tract_geoid", "pres_dem_share_2020", "pres_total_2020"]],
                   on="tract_geoid", how="inner")

    # Drop uninhabited and tracts with no vote data
    df = df[~df["is_uninhabited"]].dropna(
        subset=comp_cols + ["pres_dem_share_2020", "pres_total_2020"]
    )
    log.info("Joined dataset: %d tracts with memberships + vote data", len(df))
    return df


def estimate_community_vote_shares(
    df: pd.DataFrame, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (theta_nnls, theta_direct):
      theta_nnls   — vote-weighted NNLS solution; used for R² (primary metric)
      theta_direct — vote-weighted mean per community; used for visualization
    """
    comp_cols = [f"c{i}" for i in range(1, k + 1)]
    W = df[comp_cols].values
    d = df["pres_dem_share_2020"].values
    v = df["pres_total_2020"].values

    # NNLS solution (for R²)
    sqrt_v = np.sqrt(v)
    theta_nnls, _ = nnls(W * sqrt_v[:, np.newaxis], d * sqrt_v)

    # Direct weighted mean (for visualization / saved profiles)
    weights = v[:, np.newaxis] * W    # (n_tracts, K)
    theta_direct = (weights * d[:, np.newaxis]).sum(axis=0) / weights.sum(axis=0)

    return theta_nnls, theta_direct


def reconstruction_r2(df: pd.DataFrame, theta: np.ndarray, k: int) -> float:
    """R² of predicted vs. actual dem_share (vote-weighted)."""
    comp_cols = [f"c{i}" for i in range(1, k + 1)]
    W = df[comp_cols].values
    d = df["pres_dem_share_2020"].values
    v = df["pres_total_2020"].values

    d_hat = W @ theta
    ss_res = np.sum(v * (d - d_hat) ** 2)
    ss_tot = np.sum(v * (d - np.average(d, weights=v)) ** 2)
    return 1.0 - ss_res / ss_tot


def plot_community_vote_shares(
    theta: np.ndarray, df: pd.DataFrame, k: int, path: Path
) -> None:
    """Bar chart of estimated Dem vote share per community type."""
    comp_cols = [f"c{i}" for i in range(1, k + 1)]
    # Fall back to component name for any K without configured labels
    k_labels = LABELS_BY_K.get(k, {})
    labels_for_plot = [k_labels.get(c, c) for c in comp_cols]

    # Sort by vote share for readability
    order = np.argsort(theta)
    ordered_labels = [labels_for_plot[i] for i in order]
    values = theta[order]
    colors = ["#ef4444" if v < 0.5 else "#3b82f6" for v in values]

    # Tract counts per dominant community
    dominant = df[comp_cols].idxmax(axis=1)
    counts = dominant.value_counts()

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(range(len(ordered_labels)), values, color=colors, alpha=0.85, edgecolor="white")

    ax.axvline(0.5, color="#94a3b8", linestyle="--", linewidth=1.2, label="50% threshold")

    for i, (bar, comp_idx) in enumerate(zip(bars, order)):
        comp = comp_cols[comp_idx]
        n = counts.get(comp, 0)
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{values[i]:.1%}  (n={n:,})", va="center", fontsize=9, color="#374151")

    ax.set_yticks(range(len(ordered_labels)))
    ax.set_yticklabels(ordered_labels, fontsize=9)
    ax.set_xlabel("Estimated 2020 Democratic presidential vote share", fontsize=11)
    ax.set_title(f"Community-type vote shares — K={k} NMF, FL+GA+AL 2020\n"
                 "(discovered from non-political demographic data only)", fontsize=12)
    ax.set_xlim(0, 0.75)
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    log.info("Saved plot → %s", path)


def run(k: int) -> None:
    """Run community vote share estimation for the given K and save outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(k)
    comp_cols = [f"c{i}" for i in range(1, k + 1)]
    theta_nnls, theta_direct = estimate_community_vote_shares(df, k)

    # R² uses NNLS: measures maximum predictive power of community structure
    r2 = reconstruction_r2(df, theta_nnls, k)
    log.info("Vote share R² (NNLS — community structure → dem share): %.4f", r2)

    # Fall back to component name for any K without configured labels
    k_labels = LABELS_BY_K.get(k, {})

    # Build results table using direct weighted mean (interpretable profiles)
    dominant = df[comp_cols].idxmax(axis=1)
    counts = dominant.value_counts().rename("n_dominant_tracts")

    results = pd.DataFrame({
        "component": comp_cols,
        "label": [k_labels.get(c, c) for c in comp_cols],
        "dem_share_2020": theta_direct,          # direct mean — always in [0,1]
        "dem_share_nnls": theta_nnls,            # NNLS — may exceed [0,1] for sparse
    })
    results = results.join(counts, on="component").fillna({"n_dominant_tracts": 0})
    results["n_dominant_tracts"] = results["n_dominant_tracts"].astype(int)

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"Community-type 2020 Democratic vote shares  K={k}  (R²={r2:.3f}, NNLS)")
    print(f"{'component':<10}{'direct mean':>12}{'NNLS θ':>10}{'n_dominant':>12}  label")
    print("=" * 70)
    for _, row in results.sort_values("dem_share_2020").iterrows():
        lean = "D" if row["dem_share_2020"] > 0.5 else "R"
        margin = abs(row["dem_share_2020"] - 0.5)
        label_clean = row["label"].replace("\n", " ")
        nnls_str = (
            f"{row['dem_share_nnls']:.1%}"
            if row["dem_share_nnls"] <= 1.0
            else f"{row['dem_share_nnls']:.1%}(!)"
        )
        print(f"  {row['component']}  {row['dem_share_2020']:.1%} ({lean}+{margin:.1%})"
              f"  {nnls_str:>10}  n={row['n_dominant_tracts']:>5,}  {label_clean}")

    print(f"\nR² = {r2:.3f}  (NNLS) — community structure explains "
          f"{r2*100:.0f}% of vote-share variance")
    print("Note: NNLS θ > 100% for sparse components = regularization needed (Stage 3 Bayesian)")
    if r2 > 0.65:
        print("  Strong: community types are politically meaningful")
    elif r2 > 0.40:
        print("  Moderate: community types capture major political structure")
    else:
        print("  Weak: community structure does not predict vote share well")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = OUTPUT_DIR / f"community_vote_shares_2020_k{k}.parquet"
    results.to_parquet(out_path, index=False)
    log.info("Saved → %s", out_path)

    # Plot uses direct weighted mean (values in [0,1] — interpretable)
    plot_community_vote_shares(
        theta_direct, df, k, OUTPUT_DIR / f"community_vote_shares_2020_k{k}.png"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate community-type Democratic vote shares from 2020 returns."
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"Number of NMF components (default: {DEFAULT_K}). "
             "Must match an existing tract_memberships_k{{K}}.parquet file.",
    )
    args = parser.parse_args()
    run(args.k)


if __name__ == "__main__":
    main()
