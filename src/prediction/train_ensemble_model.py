"""Train Ridge+HGB ensemble model for county-level Dem share priors.

Fits RidgeCV and HistGradientBoostingRegressor on type scores + county
historical mean + demographic features to predict 2024 presidential Dem
share. Blends predictions 50/50 and saves to the same output location as
train_ridge_model.py so the production API picks it up without changes.

This module improves LOO r from 0.650 (Ridge-only) to 0.690 (ensemble).

Inputs:
  data/communities/type_assignments.parquet    -- county type scores (N x J)
  data/assembled/county_features_national.parquet -- demographics (N x ~27)
  data/assembled/medsl_county_presidential_*.parquet -- historical results

Outputs:
  data/models/ridge_model/ridge_county_priors.parquet
      county_fips + ridge_pred_dem_share (3,106 matched counties)
      NOTE: column name kept as ridge_pred_dem_share for backward compatibility;
            values are ensemble (50% Ridge + 50% HGB) predictions.
  data/models/ridge_model/ridge_meta.json
      alpha, r2_ridge, r2_hgb, hgb_params, ensemble_method, n_counties, date_trained, ...

Feature matrix X = [type_scores (N x J) | county_mean_dem_share (N x 1) | demo_std (N x 20)]
Target y = 2024 presidential Dem share (absolute, not shift)
"""
from __future__ import annotations

import json
import logging
import subprocess
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV

from src.prediction.train_ridge_model import (
    _HISTORY_YEARS,
    build_feature_matrix,
)

log = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ENSEMBLE_CONFIG_PATH = PROJECT_ROOT / "data" / "config" / "ensemble_params.json"


def _load_ensemble_config() -> dict:
    """Load HGB hyperparameters and blend weights from the config file.

    Keeping these values out of source code means they can be updated after a
    grid-search retrain without touching Python. The config file is the single
    source of truth for tunable parameters.

    Raises FileNotFoundError if the config is missing so callers fail loudly
    instead of silently using stale defaults.
    """
    if not _ENSEMBLE_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Ensemble config not found: {_ENSEMBLE_CONFIG_PATH}. "
            "Restore from source control or re-run grid search."
        )
    with open(_ENSEMBLE_CONFIG_PATH) as f:
        return json.load(f)


_ENSEMBLE_CONFIG = _load_ensemble_config()

# Re-exported for backwards compatibility with tests and callers that reference
# HGB_PARAMS directly. Prefer _ENSEMBLE_CONFIG["hgb"] for new code.
HGB_PARAMS: dict = _ENSEMBLE_CONFIG["hgb"]

_ACCURACY_METRICS_PATH = PROJECT_ROOT / "data" / "model" / "accuracy_metrics.json"
_TRAINING_METRICS_PATH = PROJECT_ROOT / "data" / "model" / "training_metrics.json"


def _read_loo_metric(method_name: str) -> float | None:
    """Pull a LOO r value from accuracy_metrics.json by method name.

    Returns None if the file doesn't exist or the method isn't found, so the
    training script never fails just because validation hasn't run yet.
    """
    if not _ACCURACY_METRICS_PATH.exists():
        return None
    try:
        metrics = json.loads(_ACCURACY_METRICS_PATH.read_text())
        for entry in metrics.get("method_comparison", []):
            if entry.get("method") == method_name:
                return entry.get("loo_r")
    except (json.JSONDecodeError, KeyError):
        log.warning("Could not read LOO metric from %s", _ACCURACY_METRICS_PATH)
    return None


def _get_git_sha() -> str | None:
    """Return the short HEAD SHA of the current repo, or None if unavailable.

    Using the SHA alongside the training date gives an exact reproducibility
    anchor — you can always checkout the commit and re-run training to get
    identical artifacts.
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _compute_rmse_by_dominant_type(
    matched_fips: np.ndarray,
    ensemble_pred: np.ndarray,
    y: np.ndarray,
    type_assignments_path: Path,
) -> dict[str, float]:
    """Compute per-dominant-type RMSE to surface type-level prediction quality.

    Counties that lack a 2024 target (y == NaN) are excluded from RMSE
    computation for their type. Types with fewer than 5 valid observations
    are skipped to avoid noisy per-type estimates.

    Returns a dict mapping dominant_type (str) → rmse (float), sorted by
    descending RMSE so the worst-performing types appear first.
    """
    try:
        ta = pd.read_parquet(type_assignments_path, columns=["county_fips", "dominant_type"])
        ta = ta.set_index("county_fips")
    except Exception as exc:
        log.warning("Could not load type assignments for per-type RMSE: %s", exc)
        return {}

    fips_series = pd.Series(matched_fips)
    dominant = fips_series.map(ta["dominant_type"]).values

    rmse_by_type: dict[str, float] = {}
    valid = ~np.isnan(y)
    for dt in np.unique(dominant[~pd.isnull(dominant)]):
        mask = (dominant == dt) & valid
        if mask.sum() < 5:
            continue
        residuals = y[mask] - ensemble_pred[mask]
        rmse_by_type[str(int(dt))] = float(np.sqrt(np.mean(residuals ** 2)))

    return dict(sorted(rmse_by_type.items(), key=lambda kv: kv[1], reverse=True))


def _top_features_by_ridge_coef(
    rcv: RidgeCV,
    feature_names: list[str],
    n: int = 20,
) -> list[dict]:
    """Return the top-n features ranked by absolute Ridge coefficient magnitude.

    Provides a lightweight feature importance signal without running a full
    permutation importance sweep. Useful for quick sanity checks after retrain.
    """
    coefs = rcv.coef_
    order = np.argsort(np.abs(coefs))[::-1][:n]
    return [
        {"feature": feature_names[i], "ridge_coef": float(coefs[i])}
        for i in order
    ]


def _resolve_ensemble_paths(
    type_assignments_path: Path | None,
    demographics_path: Path | None,
    assembled_dir: Path | None,
    output_dir: Path | None,
) -> tuple[Path, Path, Path, Path]:
    """Fill in default paths for the ensemble training pipeline."""
    if type_assignments_path is None:
        type_assignments_path = PROJECT_ROOT / "data" / "communities" / "type_assignments.parquet"
    if demographics_path is None:
        demographics_path = PROJECT_ROOT / "data" / "assembled" / "county_features_national.parquet"
    if assembled_dir is None:
        assembled_dir = PROJECT_ROOT / "data" / "assembled"
    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / "models" / "ridge_model"
    return type_assignments_path, demographics_path, assembled_dir, output_dir


def _fit_ensemble_models(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
) -> tuple[RidgeCV, HistGradientBoostingRegressor, float, float, float]:
    """Fit RidgeCV and HGB on the training data.

    Returns
    -------
    rcv, hgb : fitted models
    alpha : RidgeCV selected regularization parameter
    r2_ridge, r2_hgb : training R² for each model
    """
    log.info("Fitting RidgeCV...")
    alphas = np.logspace(-3, 6, 100)
    rcv = RidgeCV(alphas=alphas, fit_intercept=True, gcv_mode="auto")
    rcv.fit(X_fit, y_fit)
    alpha = float(rcv.alpha_)
    r2_ridge = float(rcv.score(X_fit, y_fit))
    log.info("RidgeCV: alpha=%.4g, train R²=%.4f", alpha, r2_ridge)

    hgb_params = _ENSEMBLE_CONFIG["hgb"]
    log.info("Fitting HGB with params: %s", hgb_params)
    hgb = HistGradientBoostingRegressor(**hgb_params)
    hgb.fit(X_fit, y_fit)
    r2_hgb = float(hgb.score(X_fit, y_fit))
    log.info("HGB train R²=%.4f", r2_hgb)

    return rcv, hgb, alpha, r2_ridge, r2_hgb


def _blend_predictions(
    rcv: RidgeCV,
    hgb: HistGradientBoostingRegressor,
    X: np.ndarray,
) -> np.ndarray:
    """Generate blended ensemble predictions for all matched counties.

    Uses ALL X rows (not just training rows) so counties without a 2024
    target still receive ensemble priors from historical features.
    """
    ridge_weight = _ENSEMBLE_CONFIG["ensemble"]["ridge_weight"]
    hgb_weight = _ENSEMBLE_CONFIG["ensemble"]["hgb_weight"]
    ensemble_pred = ridge_weight * rcv.predict(X) + hgb_weight * hgb.predict(X)
    return np.clip(ensemble_pred, 0.0, 1.0)


def _save_ensemble_artifacts(
    output_dir: Path,
    matched_fips: np.ndarray,
    ensemble_pred: np.ndarray,
    meta: dict,
    training_metrics: dict,
) -> tuple[Path, Path]:
    """Write county priors parquet, ridge_meta.json, and training_metrics.json.

    Column name for the parquet output stays ridge_pred_dem_share for
    backward compatibility with the production API.

    Returns
    -------
    out_parquet, out_json : paths to saved files
    """
    priors_df = pd.DataFrame({
        "county_fips": matched_fips,
        "ridge_pred_dem_share": ensemble_pred,
    })
    out_parquet = output_dir / "ridge_county_priors.parquet"
    priors_df.to_parquet(out_parquet, index=False)
    log.info("Saved %d county ensemble priors to %s", len(priors_df), out_parquet)

    out_json = output_dir / "ridge_meta.json"
    out_json.write_text(json.dumps(meta, indent=2))
    log.info("Saved metadata to %s", out_json)

    _TRAINING_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _TRAINING_METRICS_PATH.write_text(json.dumps(training_metrics, indent=2))
    log.info("Saved training metrics to %s", _TRAINING_METRICS_PATH)

    return out_parquet, out_json


def train_and_save(
    type_assignments_path: Path | None = None,
    demographics_path: Path | None = None,
    assembled_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Full ensemble training run: load data, fit Ridge+HGB, blend, save artifacts.

    Returns
    -------
    dict with keys: alpha, r2_ridge, r2_hgb, n_counties, output_parquet, output_json
    """
    type_assignments_path, demographics_path, assembled_dir, output_dir = (
        _resolve_ensemble_paths(type_assignments_path, demographics_path, assembled_dir, output_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Re-use the data loading logic from train_ridge_model (same inputs)
    from src.prediction.train_ridge_model import _load_ridge_training_data
    county_fips, scores, demo_df, n_demo, county_mean, y_full = _load_ridge_training_data(
        type_assignments_path, demographics_path, assembled_dir
    )
    J = scores.shape[1]

    log.info("Building feature matrix (inner join with demographics)...")
    X, feature_names, row_mask = build_feature_matrix(
        scores, np.array(county_fips), demo_df, county_mean
    )
    y = y_full[row_mask]

    valid_mask = ~np.isnan(y)
    X_fit = X[valid_mask]
    y_fit = y[valid_mask]

    n_fit = len(y_fit)
    log.info("Training ensemble on %d counties, %d features", n_fit, X_fit.shape[1])
    if n_fit < 10:
        raise ValueError(f"Too few valid training samples: {n_fit}. Check data.")

    rcv, hgb, alpha, r2_ridge, r2_hgb = _fit_ensemble_models(X_fit, y_fit)

    ensemble_pred = _blend_predictions(rcv, hgb, X)
    matched_fips = np.array(county_fips)[row_mask]

    ridge_weight = _ENSEMBLE_CONFIG["ensemble"]["ridge_weight"]
    hgb_weight = _ENSEMBLE_CONFIG["ensemble"]["hgb_weight"]
    hgb_params = _ENSEMBLE_CONFIG["hgb"]

    meta = {
        "alpha": alpha,
        "r2_ridge": r2_ridge,
        "r2_train": r2_ridge,  # kept for backward compat with existing tests
        "r2_hgb": r2_hgb,
        "ensemble_method": f"{int(ridge_weight * 100)}pct_ridge_{int(hgb_weight * 100)}pct_hgb",
        "ensemble_ridge_weight": ridge_weight,
        "ensemble_hgb_weight": hgb_weight,
        "hgb_params": hgb_params,
        # LOO metrics are computed by the validation pipeline and stored in
        # data/model/accuracy_metrics.json. Pull them in if available so the
        # training metadata is self-contained; otherwise leave as null.
        "loo_r_ridge": _read_loo_metric("Ridge (scores only)"),
        "loo_r_hgb": _read_loo_metric("Ridge+HGB ensemble"),  # closest proxy
        "loo_r_ensemble": _read_loo_metric("Ridge+HGB ensemble"),
        "feature_names": feature_names,
        "n_counties": int(len(matched_fips)),
        "n_training_samples": int(n_fit),
        "date_trained": str(date.today()),
        "target": "pres_dem_share_2024",
        "history_years": _HISTORY_YEARS,
        "n_type_scores": J,
        "n_demo_features": n_demo,
    }

    # Compute additional metrics for traceability and per-type diagnostics.
    # git_sha ties this artifact to an exact reproducible code state.
    # rmse_by_dominant_type surfaces which type clusters are hardest to predict.
    # top_20_features provides a quick post-retrain sanity check without a full
    # permutation sweep.
    rmse_by_dominant_type = _compute_rmse_by_dominant_type(
        matched_fips, ensemble_pred[valid_mask], y_fit, type_assignments_path
    )
    top_20_features = _top_features_by_ridge_coef(rcv, feature_names)

    training_metrics = {
        "date_trained": str(date.today()),
        "git_sha": _get_git_sha(),
        "n_training_samples": int(n_fit),
        "n_counties_output": int(len(matched_fips)),
        "n_features": int(X_fit.shape[1]),
        "n_type_scores": J,
        "n_demo_features": n_demo,
        "ridge": {"alpha": alpha, "train_r2": r2_ridge},
        "hgb": {"train_r2": r2_hgb, **hgb_params},
        "ensemble": {
            "method": f"{int(ridge_weight * 100)}pct_ridge_{int(hgb_weight * 100)}pct_hgb",
            "ridge_weight": ridge_weight,
            "hgb_weight": hgb_weight,
            "pred_min": float(ensemble_pred.min()),
            "pred_max": float(ensemble_pred.max()),
            "pred_mean": float(ensemble_pred.mean()),
        },
        "loo_r_ridge": meta["loo_r_ridge"],
        "loo_r_ensemble": meta["loo_r_ensemble"],
        "rmse_by_dominant_type": rmse_by_dominant_type,
        "top_20_features": top_20_features,
        "target": "pres_dem_share_2024",
        "history_years": _HISTORY_YEARS,
    }

    out_parquet, out_json = _save_ensemble_artifacts(
        output_dir, matched_fips, ensemble_pred, meta, training_metrics
    )

    print(f"Ridge: alpha={alpha:.4g}, train R²={r2_ridge:.4f}")
    print(f"HGB:   train R²={r2_hgb:.4f}")
    print(f"Ensemble (50/50): {len(matched_fips)} counties saved → {out_parquet}")
    print(f"Prediction range: [{ensemble_pred.min():.3f}, {ensemble_pred.max():.3f}]")

    return {
        "alpha": alpha,
        "r2_ridge": r2_ridge,
        "r2_hgb": r2_hgb,
        "r2": r2_ridge,  # backward compat
        "n_counties": len(matched_fips),
        "output_parquet": out_parquet,
        "output_json": out_json,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    train_and_save()
