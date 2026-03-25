"""Turnout Dimension Experiment -- P3.5.

Tests whether turnout shift dimensions carry independent signal beyond D/R vote-share
shifts, and whether re-weighting or isolating them improves KMeans type quality.

The production model includes turnout shifts at equal weight alongside D/R shifts.
The hypothesis: turnout shifts might carry DIFFERENT signal than partisan shifts,
and treating them as a separate dimension (or weighting them differently) could
improve type discovery.

Four variants are compared against the production KMeans J=43 baseline:

  A. BASELINE      -- Production: all shifts (D+R+turnout), pres x2.5, turnout at 1.0
  B. NO_TURNOUT    -- Drop all turnout columns; cluster on D/R only
  C. TURNOUT_X2    -- Keep turnout; up-weight to x2.0 (still pres x2.5)
  D. TURNOUT_X0.5  -- Keep turnout; down-weight to x0.5 (half)
  E. TURNOUT_ONLY  -- Cluster on turnout shifts only (no D/R), pres x2.5

Each variant uses J=43 (production) and the same holdout protocol:
  - Training: 2008+ shift pairs (min_year=2008), excluding 2020->2024
  - Holdout: pres_d/r/turnout_shift_20_24
  - County-prior prediction + temperature=10 soft membership
  - Metric: mean Pearson r across 3 holdout dims

Usage:
    uv run python experiments/turnout_dimension_experiment.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.cluster import KMeans

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── Constants matching production ─────────────────────────────────────────────

HOLDOUT_COLUMNS = [
    "pres_d_shift_20_24",
    "pres_r_shift_20_24",
    "pres_turnout_shift_20_24",
]

KMEANS_J = 43
MIN_YEAR = 2008
TEMPERATURE = 10.0
PRES_WEIGHT = 2.5  # Production presidential x2.5

# Baseline holdout r from HDBSCAN experiment (apple-to-apple comparison)
KMEANS_BASELINE_R = 0.8428


# ── Data loading ──────────────────────────────────────────────────────────────


def load_shifts() -> tuple[pd.DataFrame, list[str]]:
    """Load county shift matrix; return DataFrame and training column names (2008+)."""
    shifts_path = PROJECT_ROOT / "data" / "shifts" / "county_shifts_multiyear.parquet"
    df = pd.read_parquet(shifts_path)

    all_shift_cols = [c for c in df.columns if c != "county_fips" and c not in HOLDOUT_COLUMNS]

    # Filter to min_year=2008 (same as production run_type_discovery)
    train_cols: list[str] = []
    for c in all_shift_cols:
        parts = c.split("_")
        try:
            y1 = int("20" + parts[-2])
        except (ValueError, IndexError):
            continue
        if y1 >= MIN_YEAR:
            train_cols.append(c)

    if not train_cols:
        train_cols = all_shift_cols  # fallback

    return df, train_cols


def classify_columns(cols: list[str]) -> dict[str, list[str]]:
    """Classify shift columns into turnout and non-turnout (D/R) buckets."""
    return {
        "pres_dr": [c for c in cols if "pres_" in c and "turnout" not in c],
        "pres_turnout": [c for c in cols if "pres_" in c and "turnout" in c],
        "gov_dr": [c for c in cols if "gov_" in c and "turnout" not in c],
        "gov_turnout": [c for c in cols if "gov_" in c and "turnout" in c],
        "sen_dr": [c for c in cols if "sen_" in c and "turnout" not in c],
        "sen_turnout": [c for c in cols if "sen_" in c and "turnout" in c],
    }


# ── Matrix builders for each variant ─────────────────────────────────────────


def build_matrix_baseline(df: pd.DataFrame, train_cols: list[str]) -> np.ndarray:
    """Production: all shifts, pres x2.5, turnout at x1.0."""
    matrix = df[train_cols].values.copy().astype(float)
    pres_mask = np.array(["pres_" in c for c in train_cols])
    matrix[:, pres_mask] *= PRES_WEIGHT
    return matrix


def build_matrix_no_turnout(df: pd.DataFrame, train_cols: list[str]) -> np.ndarray:
    """Drop all turnout columns; cluster on D/R partisanship only."""
    dr_cols = [c for c in train_cols if "turnout" not in c]
    matrix = df[dr_cols].values.copy().astype(float)
    pres_mask = np.array(["pres_" in c for c in dr_cols])
    matrix[:, pres_mask] *= PRES_WEIGHT
    return matrix


def build_matrix_turnout_weighted(
    df: pd.DataFrame, train_cols: list[str], turnout_weight: float
) -> np.ndarray:
    """All shifts with custom turnout weight; pres d/r still x2.5."""
    matrix = df[train_cols].values.copy().astype(float)
    for i, c in enumerate(train_cols):
        if "pres_" in c and "turnout" not in c:
            matrix[:, i] *= PRES_WEIGHT
        elif "pres_" in c and "turnout" in c:
            # Pres turnout: combine pres weight and turnout weight
            matrix[:, i] *= turnout_weight
        elif "turnout" in c:
            matrix[:, i] *= turnout_weight
        # gov/sen D/R unchanged (weight = 1.0)
    return matrix


def build_matrix_turnout_only(df: pd.DataFrame, train_cols: list[str]) -> np.ndarray:
    """Cluster on turnout shift columns only; pres turnout x2.5."""
    turnout_cols = [c for c in train_cols if "turnout" in c]
    matrix = df[turnout_cols].values.copy().astype(float)
    pres_mask = np.array(["pres_" in c for c in turnout_cols])
    matrix[:, pres_mask] *= PRES_WEIGHT
    return matrix


# ── Prediction / evaluation ───────────────────────────────────────────────────


def temperature_soft_membership(dists: np.ndarray, T: float) -> np.ndarray:
    """Temperature-sharpened soft membership (identical to production)."""
    eps = 1e-10
    if T >= 500.0:
        scores = np.zeros_like(dists)
        scores[np.arange(len(dists)), np.argmin(dists, axis=1)] = 1.0
        return scores
    log_weights = -T * np.log(dists + eps)
    log_weights -= log_weights.max(axis=1, keepdims=True)
    powered = np.exp(log_weights)
    row_sums = powered.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return powered / row_sums


def compute_county_prior_holdout_r(
    train_matrix: np.ndarray,
    holdout: np.ndarray,
    n_clusters: int = KMEANS_J,
    random_state: int = 42,
) -> tuple[float, np.ndarray]:
    """Fit KMeans on train_matrix; compute county-prior holdout r.

    County-prior protocol (matching validate_types.py):
    1. Fit KMeans on train_matrix.
    2. Compute soft membership (T=10 inverse-distance).
    3. Each county's prediction = type adjustment on top of its own training mean.
    4. Mean Pearson r across holdout dims.

    Returns
    -------
    (mean_r, per_dim_r)
    """
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(train_matrix)
    centroids = km.cluster_centers_

    # Soft membership via T=10
    dists = np.stack(
        [np.linalg.norm(train_matrix - centroids[t], axis=1) for t in range(n_clusters)],
        axis=1,
    )
    soft_scores = temperature_soft_membership(dists, T=TEMPERATURE)  # (N, J)

    weight_sums = soft_scores.sum(axis=0)
    weight_sums = np.where(weight_sums == 0, 1.0, weight_sums)

    per_dim_r: list[float] = []
    for d in range(holdout.shape[1]):
        actual = holdout[:, d]
        type_means = (soft_scores.T @ actual) / weight_sums  # (J,)
        predicted = soft_scores @ type_means  # (N,)
        if np.std(actual) < 1e-10 or np.std(predicted) < 1e-10:
            per_dim_r.append(0.0)
        else:
            r, _ = pearsonr(actual, predicted)
            per_dim_r.append(float(r))

    mean_r = float(np.mean(per_dim_r))
    return mean_r, np.array(per_dim_r)


def compute_coherence(
    train_matrix: np.ndarray,
    holdout: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
) -> float:
    """Between-type / total variance ratio on holdout (type coherence metric)."""
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(train_matrix)

    per_dim_ratios = []
    for d in range(holdout.shape[1]):
        values = holdout[:, d]
        type_variances = []
        type_means = []
        for t in range(n_clusters):
            mask = labels == t
            if mask.sum() >= 2:
                type_variances.append(float(np.var(values[mask], ddof=0)))
            else:
                type_variances.append(0.0)
            if mask.sum() > 0:
                type_means.append(float(np.mean(values[mask])))
        within_var = float(np.mean(type_variances))
        between_var = float(np.var(type_means, ddof=0)) if type_means else 0.0
        total = within_var + between_var
        ratio = between_var / total if total > 1e-12 else 0.0
        per_dim_ratios.append(float(np.clip(ratio, 0.0, 1.0)))

    return float(np.mean(per_dim_ratios)) if per_dim_ratios else 0.0


# ── Variant runner ────────────────────────────────────────────────────────────


def run_variant(
    name: str,
    description: str,
    train_matrix: np.ndarray,
    holdout: np.ndarray,
    n_dims_used: int,
) -> dict:
    """Run one experiment variant; return result dict."""
    print(f"\n  [{name}] {description}")
    print(f"    Training dims: {n_dims_used}, counties: {train_matrix.shape[0]}")

    mean_r, per_dim_r = compute_county_prior_holdout_r(train_matrix, holdout)
    coherence = compute_coherence(train_matrix, holdout, KMEANS_J)
    delta = mean_r - KMEANS_BASELINE_R

    print(f"    holdout r = {mean_r:.4f}  (delta vs baseline = {delta:+.4f})")
    print(f"    coherence = {coherence:.4f}")
    print(f"    per-dim r: {[round(r, 4) for r in per_dim_r]}")

    return {
        "variant": name,
        "description": description,
        "n_dims": n_dims_used,
        "holdout_r": round(mean_r, 4),
        "delta_vs_baseline": round(delta, 4),
        "coherence": round(coherence, 4),
        "pres_d_r": round(float(per_dim_r[0]), 4),
        "pres_r_r": round(float(per_dim_r[1]), 4),
        "pres_turnout_r": round(float(per_dim_r[2]), 4),
    }


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 72)
    print("Turnout Dimension Experiment -- P3.5")
    print("=" * 72)

    # Load data
    df, train_cols = load_shifts()
    holdout = df[HOLDOUT_COLUMNS].values
    classified = classify_columns(train_cols)

    n_counties = len(df)
    n_train_total = len(train_cols)
    n_turnout = len([c for c in train_cols if "turnout" in c])
    n_dr = len([c for c in train_cols if "turnout" not in c])

    print(f"\nDataset: {n_counties} counties, {n_train_total} training dims")
    print(f"  D/R shift dims:      {n_dr}")
    print(f"  Turnout shift dims:  {n_turnout}")
    print(f"  Holdout:             {HOLDOUT_COLUMNS}")
    print(f"\nKMeans J={KMEANS_J}, T={TEMPERATURE}, pres_weight={PRES_WEIGHT}")
    print(f"Baseline holdout r (prior experiments): {KMEANS_BASELINE_R}")

    # Build matrices for each variant
    mat_baseline = build_matrix_baseline(df, train_cols)
    mat_no_turnout = build_matrix_no_turnout(df, train_cols)
    mat_turnout_x2 = build_matrix_turnout_weighted(df, train_cols, turnout_weight=2.0)
    mat_turnout_x05 = build_matrix_turnout_weighted(df, train_cols, turnout_weight=0.5)
    mat_turnout_x0 = build_matrix_turnout_weighted(df, train_cols, turnout_weight=0.0)
    mat_turnout_only = build_matrix_turnout_only(df, train_cols)

    # Count dims per variant
    dr_cols = [c for c in train_cols if "turnout" not in c]
    turnout_cols = [c for c in train_cols if "turnout" in c]

    print("\n" + "-" * 72)
    print("Running variants...")
    print("-" * 72)

    results = []

    results.append(run_variant(
        name="A_BASELINE",
        description="Production: all shifts, pres D/R x2.5, turnout x1.0",
        train_matrix=mat_baseline,
        holdout=holdout,
        n_dims_used=n_train_total,
    ))

    results.append(run_variant(
        name="B_NO_TURNOUT",
        description="D/R shifts only — turnout cols dropped entirely",
        train_matrix=mat_no_turnout,
        holdout=holdout,
        n_dims_used=len(dr_cols),
    ))

    results.append(run_variant(
        name="C_TURNOUT_X2",
        description="All shifts, turnout up-weighted to x2.0 (pres D/R still x2.5)",
        train_matrix=mat_turnout_x2,
        holdout=holdout,
        n_dims_used=n_train_total,
    ))

    results.append(run_variant(
        name="D_TURNOUT_X0.5",
        description="All shifts, turnout down-weighted to x0.5",
        train_matrix=mat_turnout_x05,
        holdout=holdout,
        n_dims_used=n_train_total,
    ))

    results.append(run_variant(
        name="E_TURNOUT_X0",
        description="All shifts except turnout zeroed (equivalent to NO_TURNOUT but same J)",
        train_matrix=mat_turnout_x0,
        holdout=holdout,
        n_dims_used=n_dr,
    ))

    results.append(run_variant(
        name="F_TURNOUT_ONLY",
        description="Turnout shifts only — no D/R; pres turnout x2.5",
        train_matrix=mat_turnout_only,
        holdout=holdout,
        n_dims_used=len(turnout_cols),
    ))

    # ── Summary table ─────────────────────────────────────────────────────────

    print("\n\n" + "=" * 72)
    print("RESULTS SUMMARY -- Turnout Dimension Experiment P3.5")
    print("=" * 72)
    print(f"\nBaseline (production KMeans J={KMEANS_J}): holdout r = {KMEANS_BASELINE_R}")
    print()
    header = f"{'Variant':<18} {'Dims':>5} {'Holdout_r':>10} {'Delta':>8} {'Coherence':>10} {'pres_d_r':>9} {'pres_r_r':>9} {'turn_r':>8}"
    print(header)
    print("-" * 72)
    for r in results:
        winner = " *BEST*" if r["holdout_r"] == max(x["holdout_r"] for x in results) else ""
        baseline_mark = " [baseline]" if r["variant"] == "A_BASELINE" else ""
        print(
            f"{r['variant']:<18} {r['n_dims']:>5} {r['holdout_r']:>10.4f} "
            f"{r['delta_vs_baseline']:>+8.4f} {r['coherence']:>10.4f} "
            f"{r['pres_d_r']:>9.4f} {r['pres_r_r']:>9.4f} {r['pres_turnout_r']:>8.4f}"
            f"{winner}{baseline_mark}"
        )
    print("=" * 72)

    # Best variant
    best = max(results, key=lambda x: x["holdout_r"])
    print(f"\nBest variant: {best['variant']} (holdout r = {best['holdout_r']:.4f})")

    # Interpretation
    baseline_r = next(r["holdout_r"] for r in results if r["variant"] == "A_BASELINE")
    no_turnout_r = next(r["holdout_r"] for r in results if r["variant"] == "B_NO_TURNOUT")
    turnout_only_r = next(r["holdout_r"] for r in results if r["variant"] == "F_TURNOUT_ONLY")

    print()
    print("Key comparisons:")
    print(f"  D/R-only vs baseline:      {no_turnout_r:.4f} vs {baseline_r:.4f}  ({no_turnout_r - baseline_r:+.4f})")
    print(f"  Turnout-only vs baseline:  {turnout_only_r:.4f} vs {baseline_r:.4f}  ({turnout_only_r - baseline_r:+.4f})")

    if no_turnout_r > baseline_r + 0.005:
        print("\nINTERPRETATION: Removing turnout IMPROVES holdout r.")
        print("  -> Turnout dims add noise, not signal. RECOMMENDATION: drop them.")
    elif no_turnout_r < baseline_r - 0.005:
        print("\nINTERPRETATION: Removing turnout HURTS holdout r.")
        print("  -> Turnout dims carry useful signal. Keep them.")
    else:
        print("\nINTERPRETATION: Removing turnout has NEGLIGIBLE effect (<0.005 delta).")
        print("  -> Turnout dims are noise-neutral. Current practice is fine.")

    if turnout_only_r > 0.5:
        print(f"\n  Turnout-only types achieve r={turnout_only_r:.4f} — turnout patterns ARE predictive.")
        print("  Consider: turnout shifts encode structural mobilization, separate from partisanship.")
    else:
        print(f"\n  Turnout-only types achieve only r={turnout_only_r:.4f} — turnout alone is weak.")
        print("  Turnout shifts are not independently useful as primary clustering dimensions.")

    # Save results
    out_dir = PROJECT_ROOT / "data" / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "turnout_dimension_experiment_results.parquet"
    pd.DataFrame(results).to_parquet(out_path, index=False)
    print(f"\nResults saved to {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
