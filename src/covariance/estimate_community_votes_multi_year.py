"""
Stage 3 (multi-election): estimate community-type vote shares across
2016, 2018, and 2020 elections and analyze covariance structure.

This script extends the single-election Stage 3 validation to the full
historical set. It serves two purposes:

1. VALIDATION — does each election individually replicate the strong R²
   result from 2020? Confirms that community structure is consistently
   predictive, not a 2020-specific artifact.

2. COVARIANCE FOUNDATION — produces the community × election vote share
   matrix that will feed the Stan factor model. Also computes a naive
   empirical covariance as a prior check / sanity baseline.

Election set and data notes:
  2016: presidential returns (pres_dem_share_2016)
  2018: gubernatorial returns (gov_dem_share_2018)
        ⚠ Alabama 2018 governor race was uncontested (Ivey, R).
        AL 2018 dem shares are near-zero — excluded from covariance.
  2020: presidential returns (pres_dem_share_2020)

Office-type mixing (pres vs. gov):
  Absolute dem share levels differ between presidential and gubernatorial
  races due to candidate quality / ticket-splitting. This affects only the
  absolute level of θ_2018, not the relative ordering of community types.
  Covariance analysis normalizes each election to its weighted mean before
  computing community deviations (swing), making the office-type mixing
  less problematic for the covariance structure. Documented in A011.

Inputs:
  data/communities/tract_memberships_k7.parquet
  data/assembled/vest_tracts_2016.parquet
  data/assembled/vest_tracts_2018.parquet
  data/assembled/vest_tracts_2020.parquet

Outputs:
  data/covariance/community_vote_shares_multi_year.parquet
  data/covariance/community_vote_shares_multi_year.png
  data/covariance/community_covariance_empirical.parquet
"""

from __future__ import annotations

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

COMP_COLS = [f"c{k}" for k in range(1, 8)]

LABELS = {
    "c1": "White rural homeowner\n(older+WFH)",
    "c2": "Black urban\n(transit+income)",
    "c3": "Knowledge worker\n(mgmt+WFH+college)",
    "c4": "Asian",
    "c5": "Working-class homeowner\n(owner-occ)",
    "c6": "Hispanic\nlow-income",
    "c7": "Generic suburban\nbaseline",
}

# Elections to process: (year, column_prefix, label_for_display)
# Note: 2018 uses gubernatorial (gov_), 2016+2020 use presidential (pres_)
ELECTIONS = [
    (2016, "pres", "2016 Presidential"),
    (2018, "gov",  "2018 Gubernatorial"),
    (2020, "pres", "2020 Presidential"),
]

# AL 2018 is excluded: gubernatorial race was uncontested (Ivey, R).
# dem share values are near-zero and do not represent community partisan lean.
AL_2018_EXCLUDED = True
AL_FIPS_PREFIX = "01"


# ── Data loading ──────────────────────────────────────────────────────────────


def load_memberships() -> pd.DataFrame:
    return pd.read_parquet(
        PROJECT_ROOT / "data" / "communities" / "tract_memberships_k7.parquet"
    )


def load_election(year: int, prefix: str) -> pd.DataFrame:
    """Load VEST tract-level data for one election year."""
    path = PROJECT_ROOT / "data" / "assembled" / f"vest_tracts_{year}.parquet"
    df = pd.read_parquet(path)
    share_col = f"{prefix}_dem_share_{year}"
    total_col = f"{prefix}_total_{year}"
    if share_col not in df.columns:
        raise ValueError(f"Expected column {share_col!r} not found in {path}. "
                         f"Available: {list(df.columns)}")
    return df[["tract_geoid", share_col, total_col]].copy()


def join_election(mem: pd.DataFrame, elec: pd.DataFrame, year: int, prefix: str) -> pd.DataFrame:
    """Inner join memberships + election data; exclude uninhabited tracts."""
    share_col = f"{prefix}_dem_share_{year}"
    total_col = f"{prefix}_total_{year}"
    df = mem.merge(elec, on="tract_geoid", how="inner")
    df = df[~df["is_uninhabited"]].dropna(subset=COMP_COLS + [share_col, total_col])

    if AL_2018_EXCLUDED and year == 2018:
        n_before = len(df)
        df = df[~df["tract_geoid"].str.startswith(AL_FIPS_PREFIX)]
        n_after = len(df)
        log.info(
            "  AL 2018 excluded (uncontested governor): removed %d tracts, %d remain",
            n_before - n_after, n_after,
        )

    log.info("  %d %s %s tracts after join and filtering", len(df), year, prefix)
    return df


# ── Estimation ────────────────────────────────────────────────────────────────


def estimate_direct_weighted_mean(df: pd.DataFrame, share_col: str, total_col: str) -> np.ndarray:
    """θ_k = Σ(v · w_k · d) / Σ(v · w_k). Always in [0,1]."""
    W = df[COMP_COLS].values
    d = df[share_col].values
    v = df[total_col].values
    weights = v[:, np.newaxis] * W
    return (weights * d[:, np.newaxis]).sum(axis=0) / weights.sum(axis=0)


def estimate_nnls(df: pd.DataFrame, share_col: str, total_col: str) -> np.ndarray:
    """Vote-weighted NNLS solution for R² computation."""
    W = df[COMP_COLS].values
    d = df[share_col].values
    v = df[total_col].values
    sqrt_v = np.sqrt(v)
    theta, _ = nnls(W * sqrt_v[:, np.newaxis], d * sqrt_v)
    return theta


def reconstruction_r2(df: pd.DataFrame, theta: np.ndarray, share_col: str, total_col: str) -> float:
    """Vote-weighted R²."""
    W = df[COMP_COLS].values
    d = df[share_col].values
    v = df[total_col].values
    d_hat = W @ theta
    ss_res = np.sum(v * (d - d_hat) ** 2)
    ss_tot = np.sum(v * (d - np.average(d, weights=v)) ** 2)
    return 1.0 - ss_res / ss_tot


# ── Covariance analysis ───────────────────────────────────────────────────────


def compute_election_swings(theta_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Compute community-level swing relative to weighted mean across elections.

    Each election's θ vector is shifted by its cross-community mean, leaving
    only the relative deviation per community per election. This removes
    level differences between presidential and gubernatorial elections.

    Returns a DataFrame of the same shape as theta_matrix.
    """
    swings = theta_matrix.copy()
    for col in swings.columns:
        swings[col] = swings[col] - swings[col].mean()
    return swings


def compute_empirical_covariance(theta_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Naive T×K empirical covariance matrix (K communities × K communities).

    With T=3 elections this is maximally rank-2, so use only for qualitative
    insight. The Stan factor model will provide the proper regularized estimate.
    """
    # theta_matrix: rows=communities (K), cols=elections (T)
    # np.cov treats rows as variables, cols as observations → (K, K) result
    arr = theta_matrix.values  # (K, T) — communities as variables, elections as observations
    cov = np.cov(arr)          # (K, K), uses T-1 denominator
    return pd.DataFrame(cov, index=COMP_COLS, columns=COMP_COLS)


# ── Visualization ─────────────────────────────────────────────────────────────


def plot_multi_year(theta_matrix: pd.DataFrame, r2_by_year: dict[str, float], path: Path) -> None:
    """
    Horizontal bar chart: community × election vote shares.
    Each community shows 3 bars (2016/2018/2020).
    """
    elections = list(theta_matrix.columns)
    n_communities = len(COMP_COLS)
    n_elections = len(elections)

    colors = {
        elections[0]: "#6366f1",  # indigo — 2016
        elections[1]: "#f59e0b",  # amber — 2018 (gubernatorial, different office)
        elections[2]: "#3b82f6",  # blue  — 2020
    }
    width = 0.25
    y = np.arange(n_communities)

    # Sort communities by 2020 vote share for readability
    sort_order = theta_matrix.iloc[:, -1].argsort().values

    fig, ax = plt.subplots(figsize=(13, 8))

    for i, elec_col in enumerate(elections):
        year_label = elec_col
        values = theta_matrix[elec_col].values[sort_order]
        r2 = r2_by_year.get(elec_col, float("nan"))
        bars = ax.barh(
            y + (i - 1) * width,
            values,
            width,
            color=colors[elec_col],
            alpha=0.8,
            label=f"{year_label} (R²={r2:.3f})",
        )
        for j, (bar, val) in enumerate(zip(bars, values)):
            if not np.isnan(val):
                ax.text(
                    bar.get_width() + 0.003,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.0%}",
                    va="center",
                    fontsize=7,
                    color=colors[elec_col],
                    alpha=0.8,
                )

    ax.axvline(0.5, color="#94a3b8", linestyle="--", linewidth=1.2)
    ax.set_yticks(y)
    sorted_labels = [list(LABELS.values())[i] for i in sort_order]
    ax.set_yticklabels(sorted_labels, fontsize=9)
    ax.set_xlabel("Democratic vote share", fontsize=11)
    ax.set_title(
        "Community-type vote shares by election — K=7 NMF, FL+GA+AL\n"
        "(non-political community structure; 2018=gubernatorial, AL excluded from 2018)",
        fontsize=11,
    )
    ax.set_xlim(0, 0.85)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    log.info("Saved plot → %s", path)


def plot_covariance_heatmap(cov: pd.DataFrame, path: Path) -> None:
    """Heatmap of empirical community×community covariance (normalized as correlation)."""
    # Normalize to correlation matrix for interpretability
    std = np.sqrt(np.diag(cov.values))
    std_outer = np.outer(std, std)
    corr = cov.values / std_outer
    np.fill_diagonal(corr, 1.0)

    short_labels = [f"c{k}" for k in range(1, 8)]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(short_labels)))
    ax.set_yticks(range(len(short_labels)))
    ax.set_xticklabels(short_labels, fontsize=9)
    ax.set_yticklabels(short_labels, fontsize=9)

    for i in range(len(short_labels)):
        for j in range(len(short_labels)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black" if abs(corr[i, j]) < 0.7 else "white")

    plt.colorbar(im, ax=ax, label="Pearson correlation (3 elections)")
    ax.set_title(
        "Community-type political covariance (empirical, T=3)\n"
        "Note: only rank-2 with 3 elections — qualitative only; Stan model will regularize",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    log.info("Saved covariance heatmap → %s", path)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mem = load_memberships()

    theta_direct_by_year: dict[str, np.ndarray] = {}
    r2_by_year: dict[str, float] = {}
    rows = []

    print("\n" + "=" * 75)
    print("Community-type vote shares — multi-election (K=7 NMF, FL+GA+AL)")
    print("=" * 75)

    for year, prefix, label in ELECTIONS:
        elec_path = PROJECT_ROOT / "data" / "assembled" / f"vest_tracts_{year}.parquet"
        if not elec_path.exists():
            log.warning("Skipping %s — file not found: %s", label, elec_path)
            continue

        share_col = f"{prefix}_dem_share_{year}"
        total_col = f"{prefix}_total_{year}"

        elec = load_election(year, prefix)
        df = join_election(mem, elec, year, prefix)

        theta_direct = estimate_direct_weighted_mean(df, share_col, total_col)
        theta_nnls = estimate_nnls(df, share_col, total_col)
        r2 = reconstruction_r2(df, theta_nnls, share_col, total_col)

        theta_direct_by_year[label] = theta_direct
        r2_by_year[label] = r2

        log.info("%s R² (NNLS): %.4f", label, r2)

        print(f"\n{label}  (R²={r2:.3f})")
        print(f"  {'component':<8}{'direct mean':>12}  {'NNLS θ':>10}  label")
        print(f"  {'-'*60}")
        for k, comp in enumerate(COMP_COLS):
            direct_str = f"{theta_direct[k]:.1%}"
            nnls_str = f"{theta_nnls[k]:.1%}" + ("(!)" if theta_nnls[k] > 1.0 else "")
            label_clean = LABELS[comp].replace("\n", " ")
            print(f"  {comp:<8}{direct_str:>12}  {nnls_str:>10}  {label_clean}")

        rows.append({
            "year": year,
            "office": prefix,
            "election_label": label,
            "r2_nnls": r2,
            **{f"theta_{comp}": theta_direct[k] for k, comp in enumerate(COMP_COLS)},
        })

    if not rows:
        log.error("No election data found. Run fetch_vest_multi_year.py first.")
        return

    # ── Community × election matrix (for Stan input and visualization) ─────────
    theta_matrix = pd.DataFrame(
        {label: theta for label, theta in theta_direct_by_year.items()},
        index=COMP_COLS,
    )

    print("\n" + "=" * 75)
    print("Community × election vote share matrix (direct weighted mean)")
    print("=" * 75)
    print(theta_matrix.to_string(float_format=lambda x: f"{x:.1%}"))

    print("\nElection R² summary:")
    for lbl, r2 in r2_by_year.items():
        status = "✓ Strong" if r2 > 0.65 else ("~ Moderate" if r2 > 0.40 else "✗ Weak")
        print(f"  {lbl:<30} R²={r2:.3f}  {status}")

    # ── Swing analysis ─────────────────────────────────────────────────────────
    if len(theta_matrix.columns) >= 2:
        swings = compute_election_swings(theta_matrix)
        print("\nCommunity deviations from election mean (swing, normalized):")
        print(swings.to_string(float_format=lambda x: f"{x:+.1%}"))

    # ── Empirical covariance ───────────────────────────────────────────────────
    if len(theta_matrix.columns) >= 2:
        cov = compute_empirical_covariance(theta_matrix)
        print("\nEmpirical community covariance (note: rank ≤ T-1 = rank-2 with 3 elections):")
        print(cov.round(4).to_string())

    # ── Save ───────────────────────────────────────────────────────────────────
    results_df = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "community_vote_shares_multi_year.parquet"
    results_df.to_parquet(out_path, index=False)
    log.info("Saved → %s", out_path)

    if len(theta_matrix.columns) >= 2:
        cov_path = OUTPUT_DIR / "community_covariance_empirical.parquet"
        cov.to_parquet(cov_path)
        log.info("Saved → %s", cov_path)

    # ── Plots ──────────────────────────────────────────────────────────────────
    plot_multi_year(theta_matrix, r2_by_year, OUTPUT_DIR / "community_vote_shares_multi_year.png")

    if len(theta_matrix.columns) >= 2:
        plot_covariance_heatmap(cov, OUTPUT_DIR / "community_covariance_heatmap.png")


if __name__ == "__main__":
    main()
