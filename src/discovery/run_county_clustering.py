"""Run county-level agglomerative clustering on electoral shift vectors.

Loads county_shifts.parquet and county_adjacency.npz, normalizes training
shifts (pres 16->20 + mid 18->22 = 6 dims), and sweeps k=3..50 to find elbow
using Ward linkage with spatial connectivity constraint.

Outputs (Layer 1 — geographic community assignment):
    data/communities/county_community_assignments.parquet — county_fips, community_id
      (also written with legacy column name 'community' for backward compatibility)

Outputs (Layer 2 — electoral type stub):
    data/communities/county_type_assignments_stub.parquet — community_id, type_weight_*, dominant_type_id

Usage:
    uv run python src/discovery/run_county_clustering.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster._agglomerative import _hc_cut
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHIFTS_PATH = PROJECT_ROOT / "data" / "shifts" / "county_shifts.parquet"
ADJ_NPZ = PROJECT_ROOT / "data" / "communities" / "county_adjacency.npz"
ADJ_FIPS = PROJECT_ROOT / "data" / "communities" / "county_adjacency.fips.txt"
OUT_PATH = PROJECT_ROOT / "data" / "communities" / "county_community_assignments.parquet"

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

# Training = pres 16→20 (cols 0-2) + mid 18→22 (cols 6-8)
TRAIN_IDX = [0, 1, 2, 6, 7, 8]
# Holdout = pres 20→24 (cols 3-5)
HOLDOUT_IDX = [3, 4, 5]


def within_cluster_variance(shifts: np.ndarray, labels: np.ndarray) -> float:
    """Weighted mean within-cluster variance across all clusters."""
    total_var = 0.0
    total_weight = 0.0
    for label in np.unique(labels):
        mask = labels == label
        count = int(mask.sum())
        var = float(np.var(shifts[mask], ddof=0)) if count > 1 else 0.0
        total_var += var * count
        total_weight += count
    return total_var / total_weight if total_weight > 0 else 0.0


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
        log.warning("Filling %d counties with NaN shifts using column means", n_missing)
        col_means = aligned[SHIFT_COLS].mean()
        aligned[SHIFT_COLS] = aligned[SHIFT_COLS].fillna(col_means)

    all_shifts = aligned[SHIFT_COLS].values  # (293, 9)

    # ── Split into training and holdout ───────────────────────────────────────
    train_shifts = all_shifts[:, TRAIN_IDX]   # (293, 6): pres 16→20 + mid 18→22
    holdout_shifts = all_shifts[:, HOLDOUT_IDX]  # (293, 3): pres 20→24

    log.info(
        "Data: %d counties, train dims=%d, holdout dims=%d",
        len(fips_list), train_shifts.shape[1], holdout_shifts.shape[1],
    )

    # ── Normalize training shifts ─────────────────────────────────────────────
    scaler = StandardScaler()
    train_norm = scaler.fit_transform(train_shifts)

    # ── Fit full Ward tree (n_clusters=1 builds the full dendrogram) ──────────
    log.info("Fitting Ward dendrogram (n_clusters=1)...")
    model = AgglomerativeClustering(
        linkage="ward",
        connectivity=W,
        n_clusters=1,
        compute_distances=True,
    )
    model.fit(train_norm)
    n_leaves = len(fips_list)

    # ── Sweep k=3..50 ─────────────────────────────────────────────────────────
    log.info("Sweeping k=3..50 for elbow detection...")
    k_values = list(range(3, 51))
    variances = []
    valid_k = []
    for k in k_values:
        if k >= n_leaves:
            continue
        labels = _hc_cut(k, model.children_, n_leaves)
        var = within_cluster_variance(train_norm, labels)
        variances.append(var)
        valid_k.append(k)

    k_arr = np.array(valid_k)
    var_arr = np.array(variances)

    # ── Print variance curve ───────────────────────────────────────────────────
    print("\n=== Variance Curve (k, within-cluster variance on training dims) ===")
    print(f"{'k':>4}  {'variance':>12}")
    for k, v in zip(k_arr, var_arr):
        print(f"{k:>4}  {v:>12.6f}")

    # ── Find elbow ────────────────────────────────────────────────────────────
    try:
        from kneed import KneeLocator
        kl = KneeLocator(
            x=k_arr.tolist(),
            y=var_arr.tolist(),
            curve="convex",
            direction="decreasing",
            S=1.0,
        )
        elbow_k = int(kl.knee) if kl.knee is not None else None
    except Exception as e:
        log.warning("KneeLocator failed: %s", e)
        elbow_k = None

    print(f"\nElbow k: {elbow_k}")

    # ── Cluster at elbow (or k=10 if elbow unclear) ───────────────────────────
    k_target = elbow_k if elbow_k is not None else 10
    log.info("Clustering at k=%d...", k_target)
    labels_final = _hc_cut(k_target, model.children_, n_leaves)

    # Also run k=10 for comparison
    labels_k10 = _hc_cut(10, model.children_, n_leaves)
    print(f"\nk={k_target} cluster sizes: {dict(zip(*np.unique(labels_final, return_counts=True)))}")
    if k_target != 10:
        print(f"k=10 cluster sizes: {dict(zip(*np.unique(labels_k10, return_counts=True)))}")

    # ── Save assignments (Layer 1) ────────────────────────────────────────────
    # Canonical column name is community_id; keep 'community' for backward compat.
    assignments = pd.DataFrame({
        "county_fips": fips_list,
        "community_id": labels_final,
        "community": labels_final,  # legacy alias — downstream scripts may use this
    })
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_parquet(OUT_PATH, index=False)
    log.info("Layer 1 community assignments saved to %s", OUT_PATH)

    print(f"\nFinal: {k_target} communities assigned to {len(fips_list)} counties")

    # ── Produce Layer 2 stub (type assignments) ───────────────────────────────
    # Full NMF implementation is Phase 1 work. This stub preserves the pipeline
    # end-to-end so downstream code can depend on the Layer 2 output format.
    try:
        from src.models.type_classifier import run_type_classification
        type_stub_path = OUT_PATH.parent / "county_type_assignments_stub.parquet"
        # Default J=7 matches the historical NMF K=7 canonical choice
        run_type_classification(
            shifts_path=SHIFTS_PATH,
            assignments_path=OUT_PATH,
            shift_cols=SHIFT_COLS,
            j=7,
            output_path=type_stub_path,
        )
        log.info("Layer 2 stub type assignments saved to %s", type_stub_path)
    except Exception as exc:
        log.warning("Layer 2 stub generation failed (non-fatal): %s", exc)


if __name__ == "__main__":
    main()
