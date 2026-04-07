"""
Stage 3 Bayesian covariance estimation.

Compiles and runs the Stan factor model
(src/covariance/stan/community_covariance.stan) using the multi-election
community vote share matrix as input. Extracts the posterior community
covariance matrix Sigma for use in Stage 4 poll propagation.

Key design choices:
  F=2 latent factors — captures "overall partisan level" (factor 1) and
  "realignment direction" (factor 2, driven by c6 Hispanic vs c4 Asian
  divergence observed in the empirical data). F=3 is available but
  likely underfits with T=3 elections.

  Observation uncertainty (theta_se) is computed from effective sample
  size per community-election cell: se = sqrt(p*(1-p) / n_eff), where
  n_eff = sum(v_i * w_ik) / max(v_i * w_ik). This characterizes the
  precision of the direct weighted mean estimator.

Inputs:
  data/communities/tract_memberships_k7.parquet
  data/assembled/vest_tracts_2016.parquet
  data/assembled/vest_tracts_2018.parquet
  data/assembled/vest_tracts_2020.parquet

Outputs:
  data/covariance/stan_draws/           — raw CmdStanPy output
  data/covariance/community_sigma.parquet  — posterior mean covariance matrix
  data/covariance/community_rho.parquet    — posterior mean correlation matrix
  data/covariance/covariance_summary.parquet — full posterior summary
  data/covariance/covariance_diagnostics.png — trace plots and diagnostics
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[2]
STAN_MODEL = PROJECT_ROOT / "src" / "covariance" / "stan" / "community_covariance.stan"
OUTPUT_DIR = PROJECT_ROOT / "data" / "covariance"

COMP_COLS = [f"c{k}" for k in range(1, 8)]
K = 7  # community types

ELECTIONS = [
    (2016, "pres", "2016 Presidential"),
    (2018, "gov",  "2018 Gubernatorial"),
    (2020, "pres", "2020 Presidential"),
]

# AL excluded from 2018 (uncontested governor)
AL_FIPS_PREFIX = "01"

LABELS = {
    "c1": "White rural homeowner",
    "c2": "Black urban",
    "c3": "Knowledge worker",
    "c4": "Asian",
    "c5": "Working-class homeowner",
    "c6": "Hispanic low-income",
    "c7": "Generic suburban baseline",
}


# ── Data preparation ──────────────────────────────────────────────────────────


def compute_community_stats(
    mem: pd.DataFrame,
    elec: pd.DataFrame,
    year: int,
    prefix: str,
    exclude_al: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute direct weighted mean and effective-sample-size SE per community.

    Returns:
      theta_direct: (K,) array of community vote shares
      theta_se:     (K,) array of standard errors
    """
    share_col = f"{prefix}_dem_share_{year}"
    total_col = f"{prefix}_total_{year}"

    df = mem.merge(elec[["tract_geoid", share_col, total_col]], on="tract_geoid", how="inner")
    df = df[~df["is_uninhabited"]].dropna(subset=COMP_COLS + [share_col, total_col])

    if exclude_al:
        df = df[~df["tract_geoid"].str.startswith(AL_FIPS_PREFIX)]

    W = df[COMP_COLS].values    # (n, K)
    d = df[share_col].values    # (n,)
    v = df[total_col].values    # (n,)

    # Direct weighted mean: theta_k = sum(v_i * w_ik * d_i) / sum(v_i * w_ik)
    weights = v[:, np.newaxis] * W  # (n, K)
    theta_direct = (weights * d[:, np.newaxis]).sum(axis=0) / weights.sum(axis=0)

    # Standard error from effective sample size per community:
    # n_eff_k = (sum w_k)^2 / sum(w_k^2) where w_k = v_i * w_ik
    # se_k = sqrt(theta_k * (1 - theta_k) / n_eff_k)
    w_k = weights  # (n, K)
    w_sum = w_k.sum(axis=0)              # (K,)
    w_sum_sq = (w_k ** 2).sum(axis=0)   # (K,)
    n_eff = w_sum ** 2 / w_sum_sq       # Kish effective sample size
    theta_se = np.sqrt(theta_direct * (1 - theta_direct) / n_eff)

    return theta_direct, theta_se


def build_stan_data(mem: pd.DataFrame, elections: list | None = None) -> dict:
    """
    Assemble the Stan data dictionary from membership + election files.

    Args:
      elections: list of (year, prefix, label) tuples. Defaults to module-level
                 ELECTIONS. Pass a subset (e.g. ELECTIONS[:2]) for holdout runs.

    Returns:
      stan_data: dict ready for CmdStanPy sampling
      theta_matrix: (K, T) array for diagnostics
      obs_labels: list of election label strings
    """
    if elections is None:
        elections = ELECTIONS

    T = len(elections)
    theta_obs = np.zeros((K, T))
    theta_se_arr = np.zeros((K, T))
    obs_mask = np.zeros((K, T))

    for t, (year, prefix, label) in enumerate(elections):
        elec_path = PROJECT_ROOT / "data" / "assembled" / f"vest_tracts_{year}.parquet"
        if not elec_path.exists():
            log.warning("Missing %s, leaving as unobserved", elec_path)
            continue

        elec = pd.read_parquet(elec_path)
        exclude_al = (year == 2018)
        theta_direct, theta_se = compute_community_stats(mem, elec, year, prefix, exclude_al)

        theta_obs[:, t] = theta_direct
        theta_se_arr[:, t] = theta_se
        obs_mask[:, t] = 1.0

    stan_data = {
        "K": K,
        "T": T,
        "theta_obs": theta_obs,
        "theta_se": theta_se_arr,
        "obs_mask": obs_mask,
    }

    obs_labels = [label for _, _, label in elections]
    return stan_data, theta_obs, obs_labels


# ── Stan model ────────────────────────────────────────────────────────────────


def compile_and_sample(stan_data: dict) -> object:
    """Compile the Stan model and run MCMC sampling."""
    import cmdstanpy

    log.info("Compiling Stan model: %s", STAN_MODEL)
    model = cmdstanpy.CmdStanModel(stan_file=str(STAN_MODEL))

    draws_dir = OUTPUT_DIR / "stan_draws"
    draws_dir.mkdir(parents=True, exist_ok=True)

    log.info("Running MCMC sampling (4 chains × 2000 iterations)...")
    fit = model.sample(
        data=stan_data,
        chains=4,
        iter_warmup=1000,
        iter_sampling=1000,
        seed=42,
        adapt_delta=0.95,          # reduce divergences
        max_treedepth=12,          # allow deeper trees for complex geometry
        output_dir=str(draws_dir),
        show_progress=True,
        show_console=False,
    )

    return fit


# ── Diagnostics ───────────────────────────────────────────────────────────────


def print_diagnostics(fit) -> None:
    """Print MCMC diagnostics summary."""
    log.info("=== MCMC Diagnostics ===")
    print(fit.diagnose())

    # Key scalar parameters
    summary = fit.summary(percentiles=[5, 50, 95])
    mu_rows = summary[summary.index.str.startswith("mu")]
    # cmdstanpy 1.3+ uses ESS_bulk/ESS_tail; older versions used N_Eff
    ess_col = "ESS_bulk" if "ESS_bulk" in summary.columns else "N_Eff"
    diag_cols = ["Mean", "5%", "50%", "95%", "R_hat", ess_col]
    available_cols = [c for c in diag_cols if c in summary.columns]
    print("\nPosterior community baselines (mu):")
    print(mu_rows[available_cols].round(4).to_string())

    print("\nR² factor decomposition per community:")
    r2_rows = summary[summary.index.str.startswith("r2_factor")]
    print(r2_rows[["Mean", "5%", "95%"]].round(3).to_string())


def plot_diagnostics(fit, path: Path) -> None:
    """Trace plots for key parameters."""
    draws = fit.draws_pd()

    mu_cols = [f"mu[{k}]" for k in range(1, K + 1)]
    available = [c for c in mu_cols if c in draws.columns]

    fig, axes = plt.subplots(len(available), 1, figsize=(12, 2.5 * len(available)))
    if len(available) == 1:
        axes = [axes]

    for ax, col in zip(axes, available):
        k_idx = int(col.split("[")[1].rstrip("]")) - 1
        for chain in range(4):
            chain_draws = draws[draws["chain__"] == chain + 1][col]
            ax.plot(chain_draws.values, alpha=0.7, linewidth=0.8)
        ax.set_title(f"mu[{k_idx+1}] — {LABELS[COMP_COLS[k_idx]]}", fontsize=9)
        ax.set_ylabel("Democratic share")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)

    fig.suptitle("MCMC trace plots — community baseline vote shares", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    log.info("Saved diagnostics → %s", path)


# ── Extract and save results ──────────────────────────────────────────────────


def extract_covariance(fit) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract posterior mean Sigma (covariance) and Rho (correlation).
    Returns (sigma_df, rho_df) — both (K, K) DataFrames indexed by comp name.
    """
    draws = fit.draws_pd()

    sigma_mean = np.zeros((K, K))
    rho_mean = np.zeros((K, K))

    for k in range(K):
        for j in range(K):
            sigma_col = f"Sigma[{k+1},{j+1}]"
            rho_col = f"Rho[{k+1},{j+1}]"
            if sigma_col in draws.columns:
                sigma_mean[k, j] = draws[sigma_col].mean()
            if rho_col in draws.columns:
                rho_mean[k, j] = draws[rho_col].mean()

    sigma_df = pd.DataFrame(sigma_mean, index=COMP_COLS, columns=COMP_COLS)
    rho_df = pd.DataFrame(rho_mean, index=COMP_COLS, columns=COMP_COLS)
    return sigma_df, rho_df


def plot_correlation_heatmap(rho_df: pd.DataFrame, path: Path) -> None:
    """Heatmap of posterior mean community correlation matrix."""
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(rho_df.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels([LABELS[c] for c in COMP_COLS], fontsize=8, rotation=45, ha="right")
    ax.set_yticklabels([LABELS[c] for c in COMP_COLS], fontsize=8)

    for i in range(K):
        for j in range(K):
            val = rho_df.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color="black" if abs(val) < 0.7 else "white")

    plt.colorbar(im, ax=ax, label="Posterior mean correlation")
    ax.set_title(
        "Community-type political correlation (Bayesian factor model)\n"
        "Stage 3 covariance — input to Stage 4 poll propagation",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    log.info("Saved correlation heatmap → %s", path)


# ── Main ──────────────────────────────────────────────────────────────────────


def main(n_factors: int = 2, holdout: bool = False) -> None:
    """
    Run the covariance model.

    Args:
      holdout: if True, use only 2016+2018 elections (T=2) and save outputs
               to data/covariance/holdout_t2/.  Used to produce a prior that
               genuinely has not seen 2020 data, for out-of-sample validation.
    """
    elections = ELECTIONS[:2] if holdout else ELECTIONS
    output_dir = (
        PROJECT_ROOT / "data" / "covariance" / "holdout_t2" if holdout else OUTPUT_DIR
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if holdout:
        log.info("=== HOLDOUT MODE: T=2 (2016+2018 only) — 2020 withheld ===")
        log.info("Outputs → %s", output_dir)

    mem = pd.read_parquet(
        PROJECT_ROOT / "data" / "communities" / "tract_memberships_k7.parquet"
    )

    log.info("Building Stan data (F=%d factors, T=%d elections)...", n_factors, len(elections))
    stan_data, theta_matrix, obs_labels = build_stan_data(mem, elections)

    log.info("theta_obs matrix:")
    for k, comp in enumerate(COMP_COLS):
        row = "  ".join(f"{theta_matrix[k, t]:.3f}" for t in range(len(elections)))
        log.info("  %s: %s", comp, row)

    fit = compile_and_sample(stan_data)

    print_diagnostics(fit)

    sigma_df, rho_df = extract_covariance(fit)

    print("\nPosterior mean covariance matrix (Sigma):")
    print(sigma_df.round(5).to_string())

    print("\nPosterior mean correlation matrix (Rho):")
    print(rho_df.round(3).to_string())

    # Save outputs
    sigma_df.to_parquet(output_dir / "community_sigma.parquet")
    rho_df.to_parquet(output_dir / "community_rho.parquet")

    summary = fit.summary(percentiles=[5, 50, 95])
    summary.to_parquet(output_dir / "covariance_summary.parquet")
    log.info("Saved covariance outputs → %s", output_dir)

    # Plots
    plot_diagnostics(fit, output_dir / "covariance_diagnostics.png")
    plot_correlation_heatmap(rho_df, output_dir / "community_rho_heatmap.png")


if __name__ == "__main__":
    import sys
    numeric_args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n_factors = int(numeric_args[0]) if numeric_args else 2
    holdout   = "--holdout" in sys.argv
    main(n_factors=n_factors, holdout=holdout)
