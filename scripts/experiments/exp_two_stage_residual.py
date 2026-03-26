"""Experiment: Two-Stage Residual Prediction

Stage 1: Ridge predicts county dem_share (exact LOO via hat-matrix)
Stage 2: HGB on Stage-1 residuals (10-fold CV as LOO proxy)
Final:    Stage1_LOO + Stage2_residual_LOO

Theory: Ridge residuals may have nonlinear structure that HGB can capture,
potentially outperforming the naive 50/50 blend (LOO r=0.690).

Feature subsets tried for Stage 2:
  a) All 128 features (same as Stage 1)
  b) Demographics only (27 features)
  c) Type scores only (101 features: 100 scores + county_mean)

Also tries different HGB configs for the residual model.

Baseline to beat: Ridge+HGB 50/50 ensemble LOO r = 0.690 (S201)

Usage:
    uv run python scripts/experiments/exp_two_stage_residual.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.discovery.run_type_discovery import discover_types

warnings.filterwarnings("ignore")


# ── Helpers ──────────────────────────────────────────────────────────────────


def parse_start_year(col: str) -> int | None:
    parts = col.split("_")
    try:
        y2 = int(parts[-2])
        return y2 + (1900 if y2 >= 50 else 2000)
    except (ValueError, IndexError):
        return None


def is_holdout_col(col: str) -> bool:
    return "20_24" in col


def classify_columns(
    all_cols: list[str], min_year: int = 2008
) -> tuple[list[str], list[str]]:
    holdout = [c for c in all_cols if is_holdout_col(c)]
    training = []
    for c in all_cols:
        if is_holdout_col(c):
            continue
        start = parse_start_year(c)
        if start is None or start >= min_year:
            training.append(c)
    return training, holdout


def ridge_loo_predictions(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Exact LOO predictions for Ridge via hat-matrix shortcut."""
    N, P = X.shape
    X_aug = np.column_stack([np.ones(N), X])
    pen = alpha * np.eye(P + 1)
    pen[0, 0] = 0.0
    A = X_aug.T @ X_aug + pen
    A_inv = np.linalg.inv(A)
    h = np.einsum("ij,ij->i", X_aug @ A_inv, X_aug)
    beta = A_inv @ X_aug.T @ y
    y_hat = X_aug @ beta
    e = y - y_hat
    denom = np.where(np.abs(1.0 - h) < 1e-10, 1e-10, 1.0 - h)
    return y - e / denom


def ridge_loo(X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """RidgeCV + exact LOO. Returns (r, loo_preds)."""
    alphas = np.logspace(-3, 6, 100)
    rcv = RidgeCV(alphas=alphas, fit_intercept=True, gcv_mode="auto")
    rcv.fit(X, y)
    y_loo = ridge_loo_predictions(X, y, float(rcv.alpha_))
    r, _ = pearsonr(y, y_loo)
    return float(r), y_loo


def hgb_cv_predictions(
    X: np.ndarray,
    y: np.ndarray,
    kf: KFold,
    **hgb_kwargs,
) -> np.ndarray:
    """K-fold CV predictions for HGB."""
    model = HistGradientBoostingRegressor(**hgb_kwargs)
    return cross_val_predict(model, X, y, cv=kf)


def build_acs_base(acs: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame()
    feat["county_fips"] = acs["county_fips"]
    pop = acs["pop_total"].replace(0, np.nan)
    feat["pct_white_nh"] = acs["pop_white_nh"] / pop
    feat["pct_black"] = acs["pop_black"] / pop
    feat["pct_asian"] = acs["pop_asian"] / pop
    feat["pct_hispanic"] = acs["pop_hispanic"] / pop
    educ_total = acs["educ_total"].replace(0, np.nan)
    educ_college = (
        acs["educ_bachelors"] + acs["educ_masters"]
        + acs["educ_professional"] + acs["educ_doctorate"]
    )
    feat["pct_college_plus"] = educ_college / educ_total
    housing = acs["housing_units"].replace(0, np.nan)
    feat["pct_owner_occupied"] = acs["housing_owner"] / housing
    commute = acs["commute_total"].replace(0, np.nan)
    feat["pct_car_commute"] = acs["commute_car"] / commute
    feat["pct_transit"] = acs["commute_transit"] / commute
    feat["pct_wfh"] = acs["commute_wfh"] / commute
    feat["median_hh_income"] = acs["median_hh_income"]
    feat["log_median_income"] = np.log1p(acs["median_hh_income"].clip(lower=1))
    feat["median_age"] = acs["median_age"]
    occ_total = acs["occ_total"].replace(0, np.nan)
    feat["pct_management"] = (acs["occ_mgmt_male"] + acs["occ_mgmt_female"]) / occ_total
    return feat


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    assembled = PROJECT_ROOT / "data" / "assembled"

    print("=" * 70)
    print("EXPERIMENT: Two-Stage Residual Prediction")
    print("Baseline (Ridge+HGB 50/50 ensemble, S201): LOO r = 0.690")
    print("Stage 1: Ridge (exact LOO), Stage 2: HGB on residuals (10-fold CV)")
    print("=" * 70)
    print()

    # ── Load data (same pipeline as ensemble experiments) ─────────────────────
    shifts_path = PROJECT_ROOT / "data" / "shifts" / "county_shifts_multiyear.parquet"
    df = pd.read_parquet(shifts_path)
    all_cols = [c for c in df.columns if c != "county_fips"]
    training_cols, holdout_cols = classify_columns(all_cols, min_year=2008)

    mat = df[training_cols + holdout_cols].values.astype(float)
    n_train = len(training_cols)
    training_raw = mat[:, :n_train]
    holdout_raw = mat[:, n_train:]
    county_fips = df["county_fips"].values

    pres_idx = [i for i, c in enumerate(training_cols) if "pres_" in c]
    scaler = StandardScaler()
    training_scaled = scaler.fit_transform(training_raw)
    training_scaled[:, pres_idx] *= 8.0

    print("Discovering types (J=100, T=10, pw=8.0)...")
    type_result = discover_types(training_scaled, j=100, temperature=10.0, random_state=42)
    scores = type_result.scores  # (N, 100)
    county_mean = training_raw.mean(axis=1)  # (N,)

    acs_raw = pd.read_parquet(assembled / "acs_counties_2022.parquet")
    acs_feat = build_acs_base(acs_raw)
    rcms = pd.read_parquet(assembled / "county_rcms_features.parquet")
    rcms = rcms[["county_fips", "evangelical_share", "mainline_share", "catholic_share",
                 "black_protestant_share", "congregations_per_1000", "religious_adherence_rate"]]
    merged = acs_feat.merge(rcms, on="county_fips", how="left")

    fips_df = pd.DataFrame({"county_fips": county_fips})
    aligned = fips_df.merge(merged, on="county_fips", how="inner")
    demo_fips = aligned["county_fips"].values
    feat_cols = [c for c in aligned.columns if c != "county_fips"]
    demo_mat = aligned[feat_cols].values.astype(float)

    fips_series = pd.Series(county_fips)
    mask = fips_series.isin(set(demo_fips)).values
    fips_in = county_fips[mask]
    demo_idx_map = {f: i for i, f in enumerate(demo_fips)}
    reindex = [demo_idx_map[f] for f in fips_in]
    demo_mat = demo_mat[reindex]

    # Impute missing demographics with column median
    for col_i in range(demo_mat.shape[1]):
        col = demo_mat[:, col_i]
        if np.isnan(col).any():
            demo_mat[np.isnan(col), col_i] = np.nanmedian(col)

    scores_in = scores[mask]          # (N_in, 100)
    county_mean_in = county_mean[mask]  # (N_in,)
    holdout_in = holdout_raw[mask]      # (N_in, H)
    N_in = mask.sum()

    demo_scaler = StandardScaler()
    demo_scaled = demo_scaler.fit_transform(demo_mat)  # (N_in, 20)

    # Full feature matrix: 100 type scores + county_mean + 27 demo = 128
    X_full = np.column_stack([scores_in, county_mean_in, demo_scaled])
    # Type-only: 100 scores + county_mean = 101
    X_types = np.column_stack([scores_in, county_mean_in])
    # Demo-only: 20 ACS + 6 RCMS + 1 county_mean = 27
    X_demo = np.column_stack([county_mean_in, demo_scaled])

    H = holdout_in.shape[1]

    print(f"Working set: {N_in} counties")
    print(f"  X_full  shape: {X_full.shape}  (100 types + county_mean + {demo_scaled.shape[1]} demo)")
    print(f"  X_types shape: {X_types.shape}  (100 types + county_mean)")
    print(f"  X_demo  shape: {X_demo.shape}  (county_mean + {demo_scaled.shape[1]} demo)")
    print()

    # ── HGB configs to try for Stage 2 ────────────────────────────────────────
    hgb_configs = {
        "default": dict(max_iter=300, learning_rate=0.05, max_depth=4,
                        min_samples_leaf=20, l2_regularization=1.0, random_state=42),
        "shallow": dict(max_iter=500, learning_rate=0.02, max_depth=2,
                        min_samples_leaf=50, l2_regularization=5.0, random_state=42),
        "deep":    dict(max_iter=200, learning_rate=0.1,  max_depth=6,
                        min_samples_leaf=10, l2_regularization=0.5, random_state=42),
    }

    feature_sets = {
        "full_128":  X_full,
        "types_101": X_types,
        "demo_27":   X_demo,
    }

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # ── Per-holdout dimension ─────────────────────────────────────────────────
    # Collect all results
    all_results = {}  # key: (feat_name, hgb_name) → list of per-dim r values

    for feat_name in feature_sets:
        for hgb_name in hgb_configs:
            all_results[(feat_name, hgb_name)] = []

    ridge_rs = []
    baseline_50_50_rs = []

    for h in range(H):
        y = holdout_in[:, h]

        # ── Stage 1: Ridge (exact LOO) ────────────────────────────────────────
        r_ridge, y_loo_ridge = ridge_loo(X_full, y)
        ridge_rs.append(r_ridge)
        residuals = y - y_loo_ridge  # Stage 2 targets

        # ── Baseline: naive 50/50 blend for reference ─────────────────────────
        y_hgb_cv = hgb_cv_predictions(X_full, y, kf, **hgb_configs["default"])
        blend_50_50 = 0.5 * y_loo_ridge + 0.5 * y_hgb_cv
        r_5050, _ = pearsonr(y, blend_50_50)
        baseline_50_50_rs.append(float(r_5050))

        # Residual stats for diagnostics
        resid_mean = float(np.mean(residuals))
        resid_std = float(np.std(residuals))
        resid_r2 = float(1.0 - np.var(residuals) / np.var(y))

        print(f"Holdout dim {h}:")
        print(f"  Ridge LOO:        r = {r_ridge:.4f}")
        print(f"  Residuals:        mean = {resid_mean:+.4f}, std = {resid_std:.4f}, "
              f"R²_explained = {resid_r2:.4f}")
        print(f"  50/50 blend (CV): r = {r_5050:.4f}  [baseline]")
        print()

        # ── Stage 2: HGB on residuals, all feature/config combos ─────────────
        for feat_name, X_stage2 in feature_sets.items():
            for hgb_name, hgb_kw in hgb_configs.items():
                # 10-fold CV on residuals
                resid_pred_cv = hgb_cv_predictions(X_stage2, residuals, kf, **hgb_kw)
                # Two-stage final prediction
                y_two_stage = y_loo_ridge + resid_pred_cv
                r_ts, _ = pearsonr(y, y_two_stage)
                all_results[(feat_name, hgb_name)].append(float(r_ts))

                label = f"{feat_name} / {hgb_name}"
                delta = float(r_ts) - r_5050
                print(f"  Two-stage [{label:30s}]: r = {r_ts:.4f}  Δ vs 50/50 = {delta:+.4f}")

        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    mean_ridge = float(np.mean(ridge_rs))
    mean_5050 = float(np.mean(baseline_50_50_rs))

    print("=" * 70)
    print("TWO-STAGE RESIDUAL EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"  Published baseline (Ridge+HGB 50/50, S201): LOO r = 0.690")
    print(f"  Ridge exact LOO (this run):                  r = {mean_ridge:.4f}")
    print(f"  50/50 blend (10-fold CV, this run):          r = {mean_5050:.4f}  [reference]")
    print()
    print("  Two-stage residual results (Stage1=Ridge LOO + Stage2=HGB on residuals):")
    print(f"  {'Feature set / HGB config':<35} {'Mean r':>8}  {'Δ vs 50/50':>10}  {'Δ vs Ridge':>10}")
    print("  " + "-" * 68)

    best_r = mean_5050
    best_label = "50/50 blend"

    rows = []
    for (feat_name, hgb_name), rs in all_results.items():
        mean_r = float(np.mean(rs))
        delta_blend = mean_r - mean_5050
        delta_ridge = mean_r - mean_ridge
        label = f"{feat_name} / {hgb_name}"
        rows.append((mean_r, label, delta_blend, delta_ridge, rs))
        if mean_r > best_r:
            best_r = mean_r
            best_label = f"two-stage [{label}]"

    rows.sort(key=lambda x: -x[0])
    for mean_r, label, delta_blend, delta_ridge, rs in rows:
        per_dim = ", ".join(f"{r:.3f}" for r in rs)
        marker = " <-- BEST" if label == best_label.replace("two-stage [", "").rstrip("]") else ""
        print(f"  {label:<35} {mean_r:>8.4f}  {delta_blend:>+10.4f}  {delta_ridge:>+10.4f}{marker}")

    print()
    print(f"  Best approach: {best_label}")
    print(f"  Best mean r:   {best_r:.4f}")
    print()

    if best_r > 0.690:
        delta = best_r - 0.690
        print(f"  BEATS PUBLISHED BASELINE (0.690): Δ = {delta:+.4f}")
        print("  NOTE: Stage-2 uses 10-fold CV (not exact LOO), so true LOO may differ")
        print("  slightly. Consider running full LOO for the best config to confirm.")
    elif best_r > mean_ridge:
        delta = best_r - mean_ridge
        print(f"  BEATS RIDGE ALONE ({mean_ridge:.4f}): Δ = {delta:+.4f}")
        print("  But does not beat published 50/50 ensemble (0.690).")
        print("  Two-stage residual approach adds no value over 50/50 blending.")
    else:
        print(f"  DOES NOT BEAT RIDGE ALONE ({mean_ridge:.4f}).")
        print("  Ridge residuals contain no useful learnable structure for HGB.")
        print("  Recommendation: stick with Ridge+HGB 50/50 ensemble (r=0.690).")


if __name__ == "__main__":
    main()
