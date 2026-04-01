"""County-level holdout validation.

Clusters counties using training shifts (pres 16→20 + mid 18→22),
then tests whether community-level training D-shift means predict
community-level holdout D-shift means (pres 20→24).

Usage:
    uv run python src/validation/validate_county_holdout.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.stats import pearsonr
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster._agglomerative import _hc_cut
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHIFTS_PATH = PROJECT_ROOT / "data" / "shifts" / "county_shifts.parquet"
ADJ_NPZ = PROJECT_ROOT / "data" / "communities" / "county_adjacency.npz"
ADJ_FIPS = PROJECT_ROOT / "data" / "communities" / "county_adjacency.fips.txt"

SHIFT_COLS = [
    "pres_d_shift_16_20",
    "pres_r_shift_16_20",
    "pres_turnout_shift_16_20",
    "pres_d_shift_20_24",
    "pres_r_shift_20_24",
    "pres_turnout_shift_20_24",
    "mid_d_shift_18_22",
    "mid_r_shift_18_22",
    "mid_turnout_shift_18_22",
]

# Training: pres 16→20 (cols 0-2) + mid 18→22 (cols 6-8)
TRAIN_IDX = [0, 1, 2, 6, 7, 8]
# Holdout: pres 20→24 (cols 3-5); D-shift is col 3
HOLDOUT_IDX = [3, 4, 5]


def within_cluster_variance(shifts: np.ndarray, labels: np.ndarray) -> float:
    total_var = 0.0
    total_weight = 0.0
    for label in np.unique(labels):
        mask = labels == label
        count = int(mask.sum())
        var = float(np.var(shifts[mask], ddof=0)) if count > 1 else 0.0
        total_var += var * count
        total_weight += count
    return total_var / total_weight if total_weight > 0 else 0.0


def run_at_k(k: int, train_norm: np.ndarray, model, train_shifts: np.ndarray,
             holdout_shifts: np.ndarray, fips_list: list[str]) -> dict:
    """Run holdout validation at a given k."""
    n_leaves = len(fips_list)
    labels = _hc_cut(k, model.children_, n_leaves)

    # Community-level means
    communities = np.unique(labels)
    n_comm = len(communities)

    train_d_means = []
    holdout_d_means = []
    community_labels = []
    sizes = []

    for c in communities:
        mask = labels == c
        cnt = int(mask.sum())
        # Training D-shift = col index 0 in TRAIN_IDX (pres_d_shift_16_20)
        train_d_mean = float(train_shifts[mask, 0].mean())
        # Holdout D-shift = col 0 in HOLDOUT_IDX (pres_d_shift_20_24)
        holdout_d_mean = float(holdout_shifts[mask, 0].mean())
        train_d_means.append(train_d_mean)
        holdout_d_means.append(holdout_d_mean)
        community_labels.append(int(c))
        sizes.append(cnt)

    train_d_arr = np.array(train_d_means)
    holdout_d_arr = np.array(holdout_d_means)

    if n_comm >= 2:
        r, p_val = pearsonr(train_d_arr, holdout_d_arr)
    else:
        r, p_val = float("nan"), float("nan")

    mae = float(np.mean(np.abs(train_d_arr - holdout_d_arr)))

    return {
        "k": k,
        "n_communities": n_comm,
        "r": r,
        "p_value": p_val,
        "mae": mae,
        "community_labels": community_labels,
        "sizes": sizes,
        "train_d_means": train_d_means,
        "holdout_d_means": holdout_d_means,
    }


def main() -> None:
    # ── Load data ─────────────────────────────────────────────────────────────
    log.info("Loading county shift vectors...")
    shifts_df = pd.read_parquet(SHIFTS_PATH)
    shifts_df["county_fips"] = shifts_df["county_fips"].astype(str).str.zfill(5)

    log.info("Loading adjacency matrix...")
    W = load_npz(str(ADJ_NPZ))
    fips_list = ADJ_FIPS.read_text().splitlines()

    # ── Align shifts to adjacency ordering ────────────────────────────────────
    shifts_indexed = shifts_df.set_index("county_fips")
    aligned = shifts_indexed.reindex(fips_list)
    n_missing = aligned[SHIFT_COLS[0]].isna().sum()
    if n_missing:
        col_means = aligned[SHIFT_COLS].mean()
        aligned[SHIFT_COLS] = aligned[SHIFT_COLS].fillna(col_means)
        log.warning("Filled %d counties with column-mean shifts", n_missing)

    all_shifts = aligned[SHIFT_COLS].values  # (293, 9)
    train_shifts = all_shifts[:, TRAIN_IDX]   # (293, 6)
    holdout_shifts = all_shifts[:, HOLDOUT_IDX]  # (293, 3)

    # ── Normalize training shifts ─────────────────────────────────────────────
    scaler = StandardScaler()
    train_norm = scaler.fit_transform(train_shifts)

    # ── Fit Ward dendrogram ───────────────────────────────────────────────────
    log.info("Fitting Ward dendrogram...")
    model = AgglomerativeClustering(
        linkage="ward",
        connectivity=W,
        n_clusters=1,
        compute_distances=True,
    )
    model.fit(train_norm)

    # ── Find elbow k ──────────────────────────────────────────────────────────
    n_leaves = len(fips_list)
    k_sweep = list(range(3, 51))
    variances = []
    for k in k_sweep:
        labels_k = _hc_cut(k, model.children_, n_leaves)
        variances.append(within_cluster_variance(train_norm, labels_k))

    try:
        from kneed import KneeLocator
        kl = KneeLocator(k_sweep, variances, curve="convex", direction="decreasing", S=1.0)
        elbow_k = int(kl.knee) if kl.knee is not None else None
    except (ImportError, ValueError):
        elbow_k = None

    log.info("Elbow k: %s", elbow_k)

    # ── Run validation at multiple k values ───────────────────────────────────
    k_to_test = sorted(set(filter(None, [elbow_k, 5, 7, 10, 15, 20])))

    print("\n" + "=" * 65)
    print("COUNTY-LEVEL HOLDOUT VALIDATION")
    print("Training: pres 16→20 D-shift + mid 18→22 D-shift")
    print("Holdout:  pres 20→24 D-shift (unseen)")
    print("=" * 65)
    print(f"\n{'k':>4}  {'Pearson r':>10}  {'p-value':>10}  {'MAE':>10}  {'Note'}")
    print("-" * 55)

    best_result = None
    for k in k_to_test:
        if k >= n_leaves:
            continue
        res = run_at_k(k, train_norm, model, train_shifts, holdout_shifts, fips_list)
        note = ""
        if k == elbow_k:
            note = "<-- elbow"
        elif k == 10:
            note = "(sanity check)"
        print(f"{res['k']:>4}  {res['r']:>10.4f}  {res['p_value']:>10.4f}  {res['mae']:>10.6f}  {note}")
        if k == (elbow_k or 10):
            best_result = res

    # ── Print detail table for elbow k ────────────────────────────────────────
    if best_result is not None:
        k_show = best_result["k"]
        print(f"\n--- Community detail for k={k_show} ---")
        print(f"{'Comm':>6}  {'N':>5}  {'Train D-mean':>14}  {'Holdout D-mean':>14}  {'Delta':>10}")
        print("-" * 60)
        for i in range(best_result["n_communities"]):
            c = best_result["community_labels"][i]
            n = best_result["sizes"][i]
            td = best_result["train_d_means"][i]
            hd = best_result["holdout_d_means"][i]
            delta = hd - td
            print(f"{c:>6}  {n:>5}  {td:>14.6f}  {hd:>14.6f}  {delta:>10.6f}")

        print(f"\nPearson r = {best_result['r']:.4f}  (p={best_result['p_value']:.4f})")
        print(f"MAE       = {best_result['mae']:.6f}")
        verdict = "PASS" if best_result["r"] > 0.5 else "MARGINAL" if best_result["r"] > 0.3 else "FAIL"
        print(f"Verdict   : {verdict} (threshold: r > 0.5)")

    print()


if __name__ == "__main__":
    main()
