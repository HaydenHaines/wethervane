"""True holdout validation for governor-specific Ridge priors.

Design
------
The production governor Ridge model (train_ridge_model_governor.py) trains on
2022 governor results and evaluates against those same 2022 results, making the
reported R²=0.962 a training metric, not a generalization metric.

This script performs a genuine temporal holdout:

  TRAINING:  2006/2010/2014 county history → predict 2018 governor results
             (simulates what we'd have had if we ran the model before 2022)
  HOLDOUT:   use the 2018-trained model to predict 2022 governor results
             (unseen counties, unseen election, one full cycle ahead)

Three-way comparison at state level (vote-weighted):
  1. Holdout model  — Ridge trained on ≤2014 history, targeting 2018, predicting 2022
  2. Presidential baseline — 2020 presidential state Dem share (naive carry-forward)
  3. Governor mean baseline — straight county mean of 2006/2010/2014/2018 gov share

Outputs
-------
  data/experiments/holdout_governor_2022.json   — metrics + per-state results
  Formatted comparison table printed to stdout

Notes on data sparsity
----------------------
- Algara years all cover the same 35 states (2,150 counties each).
  States on 4-year cycle: TX, FL, CA, NY, etc.
  States on same 4-year cycle as each other → same 35 states appear each cycle.
- 2022 MEDSL covers 34 states (1,901 counties, adds AK, drops NM and TN).
  County overlap between 2018 Algara and 2022 MEDSL: 1,899 counties.
- Vote-weighting uses 2020 presidential total votes (pres_total_2020) as the
  population proxy for within-state aggregation.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

# Bring in build_feature_matrix from the production module — ensures feature
# construction is identical to what the real model uses.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.prediction.train_ridge_model import build_feature_matrix  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSEMBLED = PROJECT_ROOT / "data" / "assembled"
COMMUNITIES = PROJECT_ROOT / "data" / "communities"
OUTPUT_DIR = PROJECT_ROOT / "data" / "experiments"
OUTPUT_PATH = OUTPUT_DIR / "holdout_governor_2022.json"

# Historical cycles available before 2022.
# We train on 2006/2010/2014 to predict 2018 (the model we'd have shipped
# before 2022). Then we test that model on 2022.
HISTORY_YEARS_FOR_MEAN = [2006, 2010, 2014]
TRAINING_TARGET_YEAR = 2018
HOLDOUT_YEAR = 2022


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_type_assignments() -> tuple[list[str], np.ndarray]:
    """Load county FIPS and type score matrix from type_assignments.parquet.

    Returns
    -------
    county_fips : list[str]  — zero-padded 5-digit FIPS for all 3,154 counties
    scores : ndarray (N, J)  — soft type membership scores
    """
    path = COMMUNITIES / "type_assignments.parquet"
    df = pd.read_parquet(path)
    df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
    score_cols = sorted(c for c in df.columns if c.endswith("_score"))
    county_fips = df["county_fips"].tolist()
    scores = df[score_cols].values.astype(float)
    log.info("Type assignments: %d counties, J=%d types", len(county_fips), scores.shape[1])
    return county_fips, scores


def _load_demographics() -> pd.DataFrame:
    """Load county demographics for feature matrix construction."""
    path = ASSEMBLED / "county_features_national.parquet"
    df = pd.read_parquet(path)
    df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
    log.info("Demographics: %d counties, %d features", len(df), len(df.columns) - 1)
    return df


def _compute_gov_mean(
    county_fips: list[str],
    years: list[int],
) -> np.ndarray:
    """Compute per-county mean governor Dem share over the given history years.

    Governor data is sparse: only the ~35 states that had a race in a given
    year appear in the Algara parquet.  Counties with no governor history
    receive a fallback of 0.45 (the same national prior used in production).

    Parameters
    ----------
    county_fips : list[str]  — zero-padded 5-digit FIPS, length N
    years : list[int]        — Algara years to average

    Returns
    -------
    ndarray (N,)  — county governor Dem share mean, fallback 0.45
    """
    fips_index = {f: i for i, f in enumerate(county_fips)}
    accumulator: dict[str, list[float]] = {f: [] for f in county_fips}

    for year in years:
        path = ASSEMBLED / f"algara_county_governor_{year}.parquet"
        if not path.exists():
            log.warning("Missing Algara file for %d, skipping", year)
            continue
        df = pd.read_parquet(path)
        df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
        share_col = f"gov_dem_share_{year}"
        if share_col not in df.columns:
            log.warning("Column %s not found in %s", share_col, path.name)
            continue
        for _, row in df.iterrows():
            fips = row["county_fips"]
            if fips in fips_index and pd.notna(row[share_col]):
                accumulator[fips].append(float(row[share_col]))

    means = np.full(len(county_fips), 0.45)
    n_with_history = 0
    for i, fips in enumerate(county_fips):
        vals = accumulator[fips]
        if vals:
            means[i] = float(np.mean(vals))
            n_with_history += 1

    log.info(
        "Gov mean (years %s): %d/%d counties have history",
        years, n_with_history, len(county_fips),
    )
    return means


def _load_gov_target(county_fips: list[str], year: int) -> np.ndarray:
    """Load county-level governor Dem share for the given year.

    2018 comes from Algara parquets; 2022 from the MEDSL parquet.

    Parameters
    ----------
    county_fips : list[str]
    year : int  — 2018 or 2022

    Returns
    -------
    ndarray (N,) — Dem share, NaN where county has no race that year
    """
    if year == 2022:
        path = ASSEMBLED / "medsl_county_2022_governor.parquet"
        df = pd.read_parquet(path)
        df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
        share_col = "gov_dem_share_2022"
    else:
        path = ASSEMBLED / f"algara_county_governor_{year}.parquet"
        df = pd.read_parquet(path)
        df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
        share_col = f"gov_dem_share_{year}"

    share_map = dict(zip(df["county_fips"], df[share_col]))
    return np.array([share_map.get(f, float("nan")) for f in county_fips])


def _load_pres_2020(county_fips: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Load 2020 presidential Dem share and total votes for each county.

    Total votes are used as the population proxy for vote-weighting.

    Returns
    -------
    dem_share : ndarray (N,)  — NaN where county not in 2020 data
    total_votes : ndarray (N,) — NaN where county not in 2020 data
    """
    path = ASSEMBLED / "medsl_county_presidential_2020.parquet"
    df = pd.read_parquet(path)
    df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
    share_map = dict(zip(df["county_fips"], df["pres_dem_share_2020"]))
    votes_map = dict(zip(df["county_fips"], df["pres_total_2020"]))
    shares = np.array([share_map.get(f, float("nan")) for f in county_fips])
    votes = np.array([votes_map.get(f, float("nan")) for f in county_fips])
    return shares, votes


def _load_state_abbr(county_fips: list[str]) -> list[str | None]:
    """Get state abbreviation for each county using 2022 governor data.

    Falls back to 2020 presidential for counties not in 2022 governor data.
    """
    path22 = ASSEMBLED / "medsl_county_2022_governor.parquet"
    df22 = pd.read_parquet(path22)
    df22["county_fips"] = df22["county_fips"].astype(str).str.zfill(5)
    state_map = dict(zip(df22["county_fips"], df22["state_abbr"]))

    # Fallback to presidential data for counties not in 2022 gov data
    path20 = ASSEMBLED / "medsl_county_presidential_2020.parquet"
    df20 = pd.read_parquet(path20)
    df20["county_fips"] = df20["county_fips"].astype(str).str.zfill(5)
    for _, row in df20.iterrows():
        fips = row["county_fips"]
        if fips not in state_map and row["state_abbr"]:
            state_map[fips] = row["state_abbr"]

    return [state_map.get(f) for f in county_fips]


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_holdout_model(
    county_fips: list[str],
    scores: np.ndarray,
    demo_df: pd.DataFrame,
    history_years: list[int],
    target_year: int,
) -> tuple[RidgeCV, np.ndarray, np.ndarray, np.ndarray]:
    """Train Ridge on governor history up to target_year.

    Uses county mean from history_years as the "partisan memory" feature,
    exactly mirroring the production governor Ridge model but anchored to a
    purely pre-2022 time window.

    Parameters
    ----------
    county_fips  : list[str]      — all N counties from type_assignments
    scores       : ndarray (N, J) — type membership scores
    demo_df      : DataFrame      — demographics (inner-joined in build_feature_matrix)
    history_years : list[int]     — years used to compute county mean feature
    target_year   : int           — year whose governor results are the training target

    Returns
    -------
    model : fitted RidgeCV
    X_full : ndarray (n_matched, n_features)  — features for all matched counties
    matched_fips : ndarray (n_matched,)       — FIPS codes after demo inner join
    county_mean : ndarray (N,)                — mean feature before row_mask
    """
    log.info("Computing county governor mean from years: %s", history_years)
    county_mean = _compute_gov_mean(county_fips, history_years)

    log.info("Building feature matrix (history %s → target %d)", history_years, target_year)
    X_full, feature_names, row_mask = build_feature_matrix(
        scores, np.array(county_fips), demo_df, county_mean
    )
    matched_fips = np.array(county_fips)[row_mask]

    log.info("Loading %d governor targets", target_year)
    y_full = _load_gov_target(county_fips, target_year)
    y = y_full[row_mask]

    # Only train on counties with a governor race that target year
    valid = ~np.isnan(y)
    X_fit = X_full[valid]
    y_fit = y[valid]
    log.info(
        "Training set: %d counties with %d governor data (of %d matched)",
        len(y_fit), target_year, len(y),
    )

    alphas = np.logspace(-3, 6, 100)
    rcv = RidgeCV(alphas=alphas, fit_intercept=True, gcv_mode="auto")
    rcv.fit(X_fit, y_fit)
    r2_train = float(rcv.score(X_fit, y_fit))
    log.info("Training R²=%.4f, alpha=%.4g", r2_train, rcv.alpha_)

    return rcv, X_full, matched_fips, county_mean


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _pearsonr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r between two 1-D arrays, both assumed finite."""
    if len(a) < 2:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _rmse(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - actual) ** 2)))


def _bias(pred: np.ndarray, actual: np.ndarray) -> float:
    """Mean signed error: positive = over-predicts Dem share."""
    return float(np.mean(pred - actual))


def _direction_accuracy(pred: np.ndarray, actual: np.ndarray) -> float:
    """Fraction of state predictions on the correct side of 50%."""
    if len(pred) == 0:
        return float("nan")
    correct = ((pred > 0.5) == (actual > 0.5)).sum()
    return float(correct / len(pred))


def aggregate_to_states(
    county_fips: np.ndarray,
    state_abbr: list[str | None],
    pred_dem_share: np.ndarray,
    actual_dem_share: np.ndarray,
    vote_weights: np.ndarray,
) -> pd.DataFrame:
    """Vote-weight county predictions up to state-level aggregates.

    Only counties where both prediction and actual are finite, and where we
    have a valid vote weight, are included.

    Parameters
    ----------
    county_fips      : county FIPS (n_counties,)
    state_abbr       : state abbreviation per county (may include None)
    pred_dem_share   : model prediction per county (n_counties,)
    actual_dem_share : ground-truth dem share per county (n_counties,)
    vote_weights     : 2020 presidential total votes per county (n_counties,)

    Returns
    -------
    DataFrame with columns: state, pred_dem_share, actual_dem_share, n_counties
    """
    rows = []
    for fips, state, pred, actual, w in zip(
        county_fips, state_abbr, pred_dem_share, actual_dem_share, vote_weights
    ):
        if state is None:
            continue
        if np.isnan(pred) or np.isnan(actual) or np.isnan(w) or w <= 0:
            continue
        rows.append({"state": state, "pred": pred, "actual": actual, "weight": w})

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["state", "pred_dem_share", "actual_dem_share", "n_counties"])

    result = (
        df.groupby("state")
        .apply(
            lambda g: pd.Series({
                "pred_dem_share": float(np.average(g["pred"], weights=g["weight"])),
                "actual_dem_share": float(np.average(g["actual"], weights=g["weight"])),
                "n_counties": len(g),
            })
        )
        .reset_index()
    )
    return result


def compute_metrics(state_df: pd.DataFrame, label: str) -> dict:
    """Compute r, RMSE, bias, direction accuracy from a state-level DataFrame."""
    pred = state_df["pred_dem_share"].values
    actual = state_df["actual_dem_share"].values
    return {
        "label": label,
        "n_states": len(state_df),
        "r": round(_pearsonr(pred, actual), 4),
        "rmse_pp": round(_rmse(pred, actual) * 100, 2),
        "bias_pp": round(_bias(pred, actual) * 100, 2),
        "direction_accuracy": round(_direction_accuracy(pred, actual), 3),
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment() -> dict:
    """Full holdout validation pipeline.

    Returns the complete results dict (also saved to JSON and printed).
    """
    log.info("=== Holdout Governor 2022 Experiment ===")
    log.info("Training window: %s → target %d | Holdout: %d",
             HISTORY_YEARS_FOR_MEAN, TRAINING_TARGET_YEAR, HOLDOUT_YEAR)

    # ── Load shared inputs ──────────────────────────────────────────────────
    county_fips, scores = _load_type_assignments()
    demo_df = _load_demographics()
    state_abbr = _load_state_abbr(county_fips)
    pres_2020_share, pres_2020_votes = _load_pres_2020(county_fips)

    # ── Train holdout model on 2018 results ─────────────────────────────────
    # History: 2006/2010/2014 gov mean → Ridge target: 2018 gov share
    # This is what we'd have had if we trained before 2022.
    model, X_full, matched_fips, county_mean_train = train_holdout_model(
        county_fips=county_fips,
        scores=scores,
        demo_df=demo_df,
        history_years=HISTORY_YEARS_FOR_MEAN,
        target_year=TRAINING_TARGET_YEAR,
    )

    # ── Apply holdout model to predict 2022 ─────────────────────────────────
    # X_full already covers all matched counties.  Predict county-level
    # prior for 2022 using the model trained entirely on ≤2018 data.
    y_holdout_pred_county = np.clip(model.predict(X_full), 0.0, 1.0)

    # ── Load 2022 actuals for matched counties ───────────────────────────────
    y_2022_all = _load_gov_target(list(matched_fips), HOLDOUT_YEAR)

    # ── Build governor mean baseline (2006/2010/2014/2018 mean) ─────────────
    # This is the simplest possible governor signal: just the historical mean.
    # It uses all 4 pre-2022 cycles so it's slightly more favorable than the
    # holdout model's mean (which only used 3 cycles as a feature), but
    # it is genuinely pre-2022 data.
    county_mean_full = _compute_gov_mean(list(matched_fips), [2006, 2010, 2014, 2018])

    # ── Pres baseline: map 2020 presidential shares onto matched counties ────
    fips_to_idx = {f: i for i, f in enumerate(county_fips)}
    matched_idx = [fips_to_idx[f] for f in matched_fips]
    pres_share_matched = pres_2020_share[matched_idx]
    pres_votes_matched = pres_2020_votes[matched_idx]
    state_abbr_matched = [state_abbr[i] for i in matched_idx]

    # ── Aggregate to states ─────────────────────────────────────────────────
    states_holdout = aggregate_to_states(
        matched_fips,
        state_abbr_matched,
        y_holdout_pred_county,
        y_2022_all,
        pres_votes_matched,
    )
    states_gov_mean = aggregate_to_states(
        matched_fips,
        state_abbr_matched,
        county_mean_full,
        y_2022_all,
        pres_votes_matched,
    )
    states_pres = aggregate_to_states(
        matched_fips,
        state_abbr_matched,
        pres_share_matched,
        y_2022_all,
        pres_votes_matched,
    )

    # Restrict all comparisons to the same set of states present in all three
    common_states = (
        set(states_holdout["state"])
        & set(states_gov_mean["state"])
        & set(states_pres["state"])
    )
    log.info("States with all three baselines: %d", len(common_states))

    states_holdout = states_holdout[states_holdout["state"].isin(common_states)].sort_values("state")
    states_gov_mean = states_gov_mean[states_gov_mean["state"].isin(common_states)].sort_values("state")
    states_pres = states_pres[states_pres["state"].isin(common_states)].sort_values("state")

    # ── Compute metrics ──────────────────────────────────────────────────────
    metrics_holdout = compute_metrics(states_holdout, "Holdout Ridge (2006-2014→2018, predict 2022)")
    metrics_gov_mean = compute_metrics(states_gov_mean, "Governor mean baseline (2006-2018 avg)")
    metrics_pres = compute_metrics(states_pres, "Presidential baseline (2020 pres share)")

    # ── Per-state residuals for top outliers ────────────────────────────────
    states_holdout = states_holdout.copy()
    states_holdout["residual_pp"] = (
        (states_holdout["pred_dem_share"] - states_holdout["actual_dem_share"]) * 100
    ).round(2)
    outliers = (
        states_holdout[["state", "pred_dem_share", "actual_dem_share", "residual_pp"]]
        .assign(
            pred_pct=lambda df: (df["pred_dem_share"] * 100).round(1),
            actual_pct=lambda df: (df["actual_dem_share"] * 100).round(1),
        )
        .nlargest(5, "residual_pp")[["state", "pred_pct", "actual_pct", "residual_pp"]]
        .to_dict("records")
    )
    outliers_neg = (
        states_holdout[["state", "pred_dem_share", "actual_dem_share", "residual_pp"]]
        .assign(
            pred_pct=lambda df: (df["pred_dem_share"] * 100).round(1),
            actual_pct=lambda df: (df["actual_dem_share"] * 100).round(1),
        )
        .nsmallest(5, "residual_pp")[["state", "pred_pct", "actual_pct", "residual_pp"]]
        .to_dict("records")
    )

    # ── Data quality notes ───────────────────────────────────────────────────
    n_2022_counties = int((~np.isnan(y_2022_all)).sum())
    n_matched_counties = len(matched_fips)
    states_2022_in_data = sorted(
        pd.read_parquet(ASSEMBLED / "medsl_county_2022_governor.parquet")["state_abbr"].unique().tolist()
    )

    data_notes = {
        "n_matched_counties": n_matched_counties,
        "n_counties_with_2022_data": n_2022_counties,
        "n_counties_missing_2022": n_matched_counties - n_2022_counties,
        "states_in_2022_data": states_2022_in_data,
        "n_states_in_2022_data": len(states_2022_in_data),
        "note_history_same_35_states": (
            "All four Algara years (2006/2010/2014/2018) cover the same 35 states. "
            "2022 MEDSL covers 34 states (adds AK, drops NM and TN). "
            "County overlap 2018↔2022: 1,899 of 1,901 2022 counties."
        ),
        "note_vote_weights": (
            "Vote-weighting uses 2020 presidential total votes as population proxy. "
            "This is available for all counties regardless of whether they had "
            "a governor race in any given year."
        ),
    }

    results = {
        "experiment": "holdout_governor_2022",
        "training_history_years": HISTORY_YEARS_FOR_MEAN,
        "training_target_year": TRAINING_TARGET_YEAR,
        "holdout_year": HOLDOUT_YEAR,
        "ridge_alpha": float(model.alpha_),
        "metrics": {
            "holdout_ridge": metrics_holdout,
            "governor_mean_baseline": metrics_gov_mean,
            "presidential_baseline": metrics_pres,
        },
        "per_state": {
            row["state"]: {
                "holdout_pred_pct": round(row["pred_dem_share"] * 100, 1),
                "actual_pct": round(row["actual_dem_share"] * 100, 1),
                "residual_pp": row["residual_pp"],
            }
            for _, row in states_holdout.iterrows()
        },
        "outliers_over_predicted": outliers,
        "outliers_under_predicted": outliers_neg,
        "data_notes": data_notes,
    }

    # ── Save ─────────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    log.info("Results saved to %s", OUTPUT_PATH)

    # ── Print comparison table ───────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  HOLDOUT VALIDATION: Governor Ridge Priors — 2022")
    print("  Train: 2006/2010/2014 history → target 2018 → predict 2022")
    print("=" * 70)
    print()
    header = f"{'Model':<46}  {'r':>6}  {'RMSE':>7}  {'Bias':>7}  {'Dir%':>6}"
    print(header)
    print("-" * 70)
    for m in [metrics_holdout, metrics_gov_mean, metrics_pres]:
        label = m["label"][:45]
        print(
            f"{label:<46}  {m['r']:>6.3f}  "
            f"{m['rmse_pp']:>6.2f}pp  {m['bias_pp']:>+6.2f}pp  "
            f"{m['direction_accuracy']:>5.1%}"
        )
    print("-" * 70)
    print(f"  N states (all models share same {len(common_states)} states)")
    print()
    print("Top over-predicted states (holdout model predicted too Dem):")
    for o in outliers:
        print(f"  {o['state']}: pred={o['pred_pct']:.1f}% actual={o['actual_pct']:.1f}% ({o['residual_pp']:+.1f}pp)")
    print()
    print("Top under-predicted states (holdout model predicted too Rep):")
    for o in outliers_neg:
        print(f"  {o['state']}: pred={o['pred_pct']:.1f}% actual={o['actual_pct']:.1f}% ({o['residual_pp']:+.1f}pp)")
    print()
    print(f"Results saved → {OUTPUT_PATH}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    run_experiment()
