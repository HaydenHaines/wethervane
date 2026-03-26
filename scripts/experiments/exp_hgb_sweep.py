"""HGB Hyperparameter Sweep (coordinate descent)

Sweeps HistGradientBoosting parameters one at a time (coordinate descent),
using 10-fold CV as a cheap proxy for LOO, then validates the best combination
with full LOO (same hat-matrix + N-model approach used in exp1b/exp4).

Baseline: Ridge+HGB ensemble LOO r = 0.690 (S201)
HGB baseline params: max_iter=300, lr=0.05, max_depth=4, min_samples_leaf=20, l2_reg=1.0

Data loading: reuses build_feature_matrix / compute_county_historical_mean / load_target
from src.prediction.train_ridge_model, exactly as train_ensemble_model.py does.

Evaluation:
  - Sweep phase: 10-fold CV for HGB; Ridge uses hat-matrix exact LOO.
    Ensemble metric = 0.5 * ridge_loo + 0.5 * hgb_10cv. Mean across 3 holdout dims.
  - Final validation: full LOO for best HGB config. All 3 holdout dims averaged.

Usage:
    uv run python scripts/experiments/exp_hgb_sweep.py
"""
from __future__ import annotations

import sys
import time
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

from src.prediction.train_ridge_model import (
    _HISTORY_YEARS,
    build_feature_matrix,
    compute_county_historical_mean,
    load_target,
)
from src.discovery.run_type_discovery import discover_types

warnings.filterwarnings("ignore")

# ── Baseline params ─────────────────────────────────────────────────────────
BASELINE_PARAMS = {
    "max_iter": 300,
    "learning_rate": 0.05,
    "max_depth": 4,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
}

SWEEP_GRID = {
    "max_iter":          [100, 200, 300, 500, 800],
    "learning_rate":     [0.01, 0.03, 0.05, 0.1],
    "max_depth":         [3, 4, 5, 6],
    "min_samples_leaf":  [10, 20, 30, 50],
    "l2_regularization": [0.1, 0.5, 1.0, 5.0],
}

BASELINE_LOO_ENSEMBLE = 0.690  # S201 measured


# ── Utility: Ridge hat-matrix exact LOO ─────────────────────────────────────

def ridge_loo_predictions(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
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


def ridge_loo_for_dim(X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    alphas = np.logspace(-3, 6, 100)
    rcv = RidgeCV(alphas=alphas, fit_intercept=True, gcv_mode="auto")
    rcv.fit(X, y)
    y_loo = ridge_loo_predictions(X, y, float(rcv.alpha_))
    r, _ = pearsonr(y, y_loo)
    return float(r), y_loo


# ── Utility: HGB full LOO (expensive — only used for final validation) ───────

def hgb_full_loo(
    X: np.ndarray,
    y: np.ndarray,
    params: dict,
    verbose_every: int = 500,
) -> np.ndarray:
    N = len(y)
    y_loo = np.empty(N)
    for i in range(N):
        if verbose_every > 0 and i % verbose_every == 0:
            print(f"      LOO {i}/{N}...", flush=True)
        mask = np.ones(N, dtype=bool)
        mask[i] = False
        m = HistGradientBoostingRegressor(**params, random_state=42)
        m.fit(X[mask], y[mask])
        y_loo[i] = m.predict(X[i : i + 1])[0]
    return y_loo


# ── Utility: 10-fold CV for HGB (sweep proxy) ────────────────────────────────

def hgb_10fold_cv(X: np.ndarray, y: np.ndarray, params: dict, seed: int = 42) -> np.ndarray:
    kf = KFold(n_splits=10, shuffle=True, random_state=seed)
    model = HistGradientBoostingRegressor(**params, random_state=42)
    return cross_val_predict(model, X, y, cv=kf)


# ── Ensemble metric: mean over holdout dims ───────────────────────────────────

def ensemble_cv_r(
    X: np.ndarray,
    holdout: np.ndarray,
    ridge_loo_preds: list[np.ndarray],
    hgb_params: dict,
) -> float:
    """Compute mean ensemble r over all holdout dims using 10-fold CV for HGB."""
    H = holdout.shape[1]
    rs = []
    for h in range(H):
        y = holdout[:, h]
        hgb_cv = hgb_10fold_cv(X, y, hgb_params)
        blended = 0.5 * ridge_loo_preds[h] + 0.5 * hgb_cv
        r, _ = pearsonr(y, blended)
        rs.append(float(r))
    return float(np.mean(rs))


def ensemble_loo_r(
    X: np.ndarray,
    holdout: np.ndarray,
    ridge_loo_preds: list[np.ndarray],
    hgb_params: dict,
) -> tuple[float, list[float]]:
    """Compute mean ensemble r over all holdout dims using full HGB LOO."""
    H = holdout.shape[1]
    rs = []
    for h in range(H):
        y = holdout[:, h]
        print(f"    Full HGB LOO — holdout dim {h}...", flush=True)
        t0 = time.time()
        hgb_loo_preds = hgb_full_loo(X, y, hgb_params)
        elapsed = time.time() - t0
        blended = 0.5 * ridge_loo_preds[h] + 0.5 * hgb_loo_preds
        r, _ = pearsonr(y, blended)
        hgb_r, _ = pearsonr(y, hgb_loo_preds)
        rs.append(float(r))
        print(f"      dim {h}: HGB LOO r={hgb_r:.4f}, ensemble LOO r={r:.4f}  ({elapsed:.0f}s)")
    return float(np.mean(rs)), rs


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load feature matrix X and holdout matrix using the same pipeline as
    train_ensemble_model.py + exp1b/exp4 for the holdout construction.

    Returns
    -------
    X       : (N, F) feature matrix aligned to N matched counties
    holdout : (N, H) holdout shift dims (2020→2024 pairs)
    ridge_loo_preds : list of H precomputed Ridge LOO prediction arrays
    """
    assembled = PROJECT_ROOT / "data" / "assembled"

    # ── Shifts (for type discovery + holdout) ───────────────────────────────
    shifts_path = PROJECT_ROOT / "data" / "shifts" / "county_shifts_multiyear.parquet"
    df = pd.read_parquet(shifts_path)
    all_cols = [c for c in df.columns if c != "county_fips"]

    def is_holdout_col(col: str) -> bool:
        return "20_24" in col

    def parse_start_year(col: str) -> int | None:
        parts = col.split("_")
        try:
            y2 = int(parts[-2])
            return y2 + (1900 if y2 >= 50 else 2000)
        except (ValueError, IndexError):
            return None

    holdout_cols = [c for c in all_cols if is_holdout_col(c)]
    training_cols = [
        c for c in all_cols
        if not is_holdout_col(c) and (parse_start_year(c) is None or parse_start_year(c) >= 2008)
    ]

    mat = df[training_cols + holdout_cols].values.astype(float)
    n_train = len(training_cols)
    training_raw = mat[:, :n_train]
    holdout_raw = mat[:, n_train:]
    county_fips_shifts = df["county_fips"].astype(str).str.zfill(5).values

    # Type discovery (same as exp1b)
    pres_idx = [i for i, c in enumerate(training_cols) if "pres_" in c]
    scaler = StandardScaler()
    training_scaled = scaler.fit_transform(training_raw)
    training_scaled[:, pres_idx] *= 8.0

    print("  Discovering types (J=100, T=10, pw=8.0)...", flush=True)
    type_result = discover_types(training_scaled, j=100, temperature=10.0, random_state=42)
    scores = type_result.scores
    county_mean_shifts = training_raw.mean(axis=1)  # used in exp1b; we use train_ridge_model below

    # ── Production feature matrix (same as train_ensemble_model.py) ──────────
    ta_path = PROJECT_ROOT / "data" / "communities" / "type_assignments.parquet"
    demo_path = PROJECT_ROOT / "data" / "assembled" / "county_features_national.parquet"

    ta_df = pd.read_parquet(ta_path)
    county_fips_ta = ta_df["county_fips"].astype(str).str.zfill(5).tolist()
    score_cols = sorted([c for c in ta_df.columns if c.endswith("_score")])
    scores_ta = ta_df[score_cols].values.astype(float)

    demo_df = pd.read_parquet(demo_path)
    demo_df["county_fips"] = demo_df["county_fips"].astype(str).str.zfill(5)

    county_mean_ta = compute_county_historical_mean(county_fips_ta, assembled)
    y_full = load_target(county_fips_ta, assembled)

    X_all, feature_names, row_mask = build_feature_matrix(
        scores_ta, np.array(county_fips_ta), demo_df, county_mean_ta
    )
    y_matched = y_full[row_mask]
    valid_mask = ~np.isnan(y_matched)
    X = X_all[row_mask][valid_mask] if False else X_all[valid_mask]  # X_all already matched
    # NOTE: X_all is already (n_matched, F); row_mask selects from original N
    # valid_mask applies to y_matched (also n_matched)
    X = X_all[valid_mask]
    y_target = y_matched[valid_mask]
    matched_fips = np.array(county_fips_ta)[row_mask][valid_mask]

    # ── Align holdout to matched FIPS ────────────────────────────────────────
    shift_fips_to_idx = {f: i for i, f in enumerate(county_fips_shifts)}
    holdout_rows = []
    for fips in matched_fips:
        if fips in shift_fips_to_idx:
            holdout_rows.append(holdout_raw[shift_fips_to_idx[fips]])
        else:
            holdout_rows.append(np.full(holdout_raw.shape[1], np.nan))
    holdout_matched = np.array(holdout_rows)

    # Drop rows where any holdout is NaN (counties with no 2020→2024 shift data)
    holdout_valid = ~np.isnan(holdout_matched).any(axis=1)
    X = X[holdout_valid]
    holdout = holdout_matched[holdout_valid]

    N, F = X.shape
    H = holdout.shape[1]
    print(f"  Final working set: {N} counties, {F} features, {H} holdout dims")

    return X, holdout


# ── Sweep logic ───────────────────────────────────────────────────────────────

def run_sweep(X: np.ndarray, holdout: np.ndarray) -> dict:
    """Coordinate-descent sweep over HGB params. Returns best params dict."""

    H = holdout.shape[1]
    print("\nPrecomputing Ridge exact LOO (needed for ensemble metric)...", flush=True)
    ridge_loo_preds = []
    ridge_rs = []
    for h in range(H):
        y = holdout[:, h]
        r, y_loo = ridge_loo_for_dim(X, y)
        ridge_loo_preds.append(y_loo)
        ridge_rs.append(r)
    print(f"  Ridge LOO r per dim: {[f'{r:.4f}' for r in ridge_rs]}")
    print(f"  Ridge LOO r mean:    {np.mean(ridge_rs):.4f}")

    # Baseline
    print(f"\nBaseline HGB params: {BASELINE_PARAMS}")
    baseline_cv_r = ensemble_cv_r(X, holdout, ridge_loo_preds, BASELINE_PARAMS)
    print(f"Baseline 10-fold ensemble CV r = {baseline_cv_r:.4f}  (reference: LOO {BASELINE_LOO_ENSEMBLE})")

    best_params = dict(BASELINE_PARAMS)
    all_results = []

    for param_name, values in SWEEP_GRID.items():
        print(f"\n{'─'*60}")
        print(f"Sweeping: {param_name}  (values: {values})")
        print(f"{'─'*60}")

        param_results = []
        for val in values:
            test_params = dict(best_params)
            test_params[param_name] = val

            t0 = time.time()
            r = ensemble_cv_r(X, holdout, ridge_loo_preds, test_params)
            elapsed = time.time() - t0

            marker = " *" if val == BASELINE_PARAMS[param_name] else ""
            print(f"  {param_name}={val:<8}  ensemble CV r = {r:.4f}  ({elapsed:.1f}s){marker}")
            param_results.append({"param": param_name, "value": val, "cv_r": r})

        best_val_entry = max(param_results, key=lambda x: x["cv_r"])
        best_val = best_val_entry["value"]
        best_r = best_val_entry["cv_r"]

        print(f"\n  Best {param_name} = {best_val}  (ensemble CV r = {best_r:.4f})")
        if best_val != best_params[param_name]:
            old_val = best_params[param_name]
            print(f"  Updated: {param_name} {old_val} → {best_val}")
        else:
            print(f"  No change from baseline ({param_name} = {best_val})")

        best_params[param_name] = best_val
        all_results.extend(param_results)

    # Final combined run with all best values (10-fold CV)
    print(f"\n{'='*60}")
    print("FINAL COMBINED PARAMS (all best values together):")
    for k, v in best_params.items():
        changed = " ← CHANGED" if v != BASELINE_PARAMS[k] else ""
        print(f"  {k}: {BASELINE_PARAMS[k]} → {v}{changed}")
    print()

    final_cv_r = ensemble_cv_r(X, holdout, ridge_loo_preds, best_params)
    print(f"Final combined ensemble 10-fold CV r = {final_cv_r:.4f}")
    print(f"Baseline 10-fold CV r was:              {baseline_cv_r:.4f}  (Δ={final_cv_r - baseline_cv_r:+.4f})")

    return best_params, ridge_loo_preds, all_results, baseline_cv_r, final_cv_r


def run_full_loo_validation(
    X: np.ndarray,
    holdout: np.ndarray,
    best_params: dict,
    ridge_loo_preds: list[np.ndarray],
) -> tuple[float, list[float]]:
    """Full LOO validation for best params. Also runs baseline params for direct comparison."""
    H = holdout.shape[1]

    print(f"\n{'='*60}")
    print("FULL LOO VALIDATION (honest metric)")
    print(f"{'='*60}")

    # Baseline LOO
    print("\n[A] Baseline HGB params LOO:")
    baseline_loo_r, baseline_per_dim = ensemble_loo_r(X, holdout, ridge_loo_preds, BASELINE_PARAMS)
    print(f"  Baseline ensemble full LOO r = {baseline_loo_r:.4f}")

    # Best params LOO
    if best_params == BASELINE_PARAMS:
        print("\n[B] Best params == baseline — skipping redundant full LOO run.")
        best_loo_r = baseline_loo_r
        best_per_dim = baseline_per_dim
    else:
        print(f"\n[B] Best params: {best_params}")
        best_loo_r, best_per_dim = ensemble_loo_r(X, holdout, ridge_loo_preds, best_params)
        print(f"  Best ensemble full LOO r = {best_loo_r:.4f}")

    return baseline_loo_r, baseline_per_dim, best_loo_r, best_per_dim


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("HGB HYPERPARAMETER SWEEP (coordinate descent)")
    print(f"Baseline ensemble LOO r = {BASELINE_LOO_ENSEMBLE} (S201)")
    print("Sweep uses 10-fold CV proxy; best config validated with full LOO")
    print("=" * 70)

    t_start = time.time()

    print("\n[1] Loading data...", flush=True)
    X, holdout = load_data()

    print("\n[2] Coordinate-descent sweep (10-fold CV)...", flush=True)
    best_params, ridge_loo_preds, all_results, baseline_cv_r, final_cv_r = run_sweep(X, holdout)

    print("\n[3] Full LOO validation for best config...", flush=True)
    baseline_loo_r, baseline_per_dim, best_loo_r, best_per_dim = run_full_loo_validation(
        X, holdout, best_params, ridge_loo_preds
    )

    elapsed_total = time.time() - t_start

    # ── Final summary ─────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("SWEEP SUMMARY")
    print("=" * 70)
    print(f"\nBaseline params: {BASELINE_PARAMS}")
    print(f"Best params:     {best_params}")
    print()
    print("Parameters changed:")
    any_changed = False
    for k in BASELINE_PARAMS:
        if best_params[k] != BASELINE_PARAMS[k]:
            print(f"  {k}: {BASELINE_PARAMS[k]} → {best_params[k]}")
            any_changed = True
    if not any_changed:
        print("  (none — baseline is optimal)")
    print()
    print("Metric summary:")
    print(f"  Published baseline LOO r (S201):       {BASELINE_LOO_ENSEMBLE:.4f}")
    print(f"  Baseline 10-fold CV r (this run):      {baseline_cv_r:.4f}")
    print(f"  Best combined 10-fold CV r:             {final_cv_r:.4f}  (Δ={final_cv_r-baseline_cv_r:+.4f})")
    print(f"  Baseline full LOO r (re-measured):     {baseline_loo_r:.4f}  per-dim={[f'{r:.3f}' for r in baseline_per_dim]}")
    print(f"  Best params full LOO r:                {best_loo_r:.4f}  per-dim={[f'{r:.3f}' for r in best_per_dim]}")
    print(f"  LOO improvement vs baseline:           {best_loo_r - baseline_loo_r:+.4f}")
    print()

    if best_loo_r > BASELINE_LOO_ENSEMBLE:
        print(f"  BEATS PUBLISHED BASELINE: LOO r = {best_loo_r:.4f}  (Δ={best_loo_r - BASELINE_LOO_ENSEMBLE:+.4f})")
    else:
        print(f"  DOES NOT BEAT PUBLISHED BASELINE: LOO r = {best_loo_r:.4f}")
        print("  Interpretation: baseline params are already near-optimal.")

    print(f"\nTotal time: {elapsed_total/60:.1f} minutes")

    # ── Detailed results table ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DETAILED SWEEP RESULTS BY PARAMETER")
    print("=" * 70)
    param_order = list(SWEEP_GRID.keys())
    for param_name in param_order:
        rows = [r for r in all_results if r["param"] == param_name]
        print(f"\n  {param_name}:")
        for row in rows:
            marker = " *" if row["value"] == BASELINE_PARAMS[param_name] else ""
            best_marker = " ← BEST" if row["value"] == best_params[param_name] else ""
            print(f"    {row['value']:<8}  CV r = {row['cv_r']:.4f}{marker}{best_marker}")


if __name__ == "__main__":
    main()
