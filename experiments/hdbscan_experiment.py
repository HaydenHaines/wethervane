"""HDBSCAN clustering experiment -- P3.4.

Tests HDBSCAN (density-based clustering) against KMeans J=43 on the production
shift matrix. HDBSCAN can find arbitrary-shaped clusters and automatically
determines the number of clusters, but produces noise points (label=-1).

The county-prior holdout r is evaluated identically to the KMeans baseline:
  1. Fit HDBSCAN on training shifts (2008+, presidential x2.5).
  2. Compute county-prior prediction: each county's predicted holdout shift =
     weighted sum of cluster means, weighted by temperature-scaled inverse
     distance to cluster centroids (T=10, same as production).
  3. Noise points (-1) are assigned to the nearest cluster for prediction.
  4. Pearson r vs actual 2020->2024 holdout shifts.

Usage:
    uv run python experiments/hdbscan_experiment.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore", category=SyntaxWarning)
import hdbscan  # noqa: E402 — must come after warnings filter

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HOLDOUT_COLUMNS = [
    "pres_d_shift_20_24",
    "pres_r_shift_20_24",
    "pres_turnout_shift_20_24",
]

# Production KMeans baseline from spectral experiment (KMeans J=43, mean per-column r)
# Note: CLAUDE.md reports 0.828 from validate_types (leave-one-pair-out CV);
# the spectral experiment measured 0.8428 on the direct 2020->2024 holdout.
# We use 0.8428 here as the apple-to-apple comparison baseline.
KMEANS_J = 43
KMEANS_HOLDOUT_R = 0.8428
MIN_YEAR = 2008
TEMPERATURE = 10.0

# HDBSCAN min_cluster_size values to sweep
MIN_CLUSTER_SIZES = [5, 10, 15, 20, 30]


def load_shift_matrix() -> tuple[np.ndarray, np.ndarray]:
    """Load training matrix (presidential x2.5 weighted) and holdout matrix."""
    shifts_path = PROJECT_ROOT / "data" / "shifts" / "county_shifts_multiyear.parquet"
    df = pd.read_parquet(shifts_path)

    all_shift_cols = [
        c for c in df.columns if c != "county_fips" and c not in HOLDOUT_COLUMNS
    ]

    shift_cols = []
    for c in all_shift_cols:
        parts = c.split("_")
        try:
            y1 = int("20" + parts[-2])
        except (ValueError, IndexError):
            continue
        if y1 >= MIN_YEAR:
            shift_cols.append(c)

    if not shift_cols:
        shift_cols = all_shift_cols

    matrix = df[shift_cols].values.copy()
    pres_mask = np.array(["pres_" in c for c in shift_cols])
    matrix[:, pres_mask] *= 2.5  # Presidential x2.5 weighting (production)

    holdout = df[HOLDOUT_COLUMNS].values

    n_counties, n_dims = matrix.shape
    print(f"Shift matrix: {n_counties} counties x {n_dims} dims (min_year={MIN_YEAR})")
    print(f"  Presidential columns (x2.5): {pres_mask.sum()}")
    print(f"  Governor/Senate columns: {(~pres_mask).sum()}")
    print(f"  Holdout columns: {len(HOLDOUT_COLUMNS)}")
    return matrix, holdout


def temperature_soft_membership(dists: np.ndarray, T: float) -> np.ndarray:
    """Temperature-sharpened soft membership from centroid distances.

    Identical to production implementation in src/discovery/run_type_discovery.py.
    """
    N, J = dists.shape
    eps = 1e-10

    if T >= 500.0:
        scores = np.zeros((N, J))
        nearest = np.argmin(dists, axis=1)
        scores[np.arange(N), nearest] = 1.0
        return scores

    log_weights = -T * np.log(dists + eps)
    log_weights -= log_weights.max(axis=1, keepdims=True)
    powered = np.exp(log_weights)
    row_sums = powered.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return powered / row_sums


def compute_holdout_r_from_labels(
    labels: np.ndarray,
    shift_matrix: np.ndarray,
    holdout: np.ndarray,
    n_clusters: int,
) -> float:
    """County-prior prediction holdout r using temperature-scaled soft membership.

    Cluster centroids are computed from training data. Soft membership uses
    T=10 inverse-distance weighting (production default). Noise points (label=-1)
    are assigned to the nearest centroid.

    Parameters
    ----------
    labels : array of shape (N,)
        Hard cluster assignments from HDBSCAN (-1 = noise).
    shift_matrix : ndarray of shape (N, D)
        Training shift matrix (presidential x2.5 applied).
    holdout : ndarray of shape (N, H)
        Holdout columns (2020->2024 shifts).
    n_clusters : int
        Number of non-noise clusters.

    Returns
    -------
    Pearson r between county-prior predictions and actual holdout shifts.
    """
    if n_clusters == 0:
        return float("nan")

    # Compute cluster centroids (exclude noise)
    cluster_ids = [i for i in range(n_clusters)]
    centroids = np.zeros((n_clusters, shift_matrix.shape[1]))
    for cid in cluster_ids:
        mask = labels == cid
        if mask.sum() > 0:
            centroids[cid] = shift_matrix[mask].mean(axis=0)

    # Assign noise points to nearest centroid for distance computation
    effective_labels = labels.copy()
    noise_mask = labels == -1
    if noise_mask.any():
        noise_dists = np.stack(
            [np.linalg.norm(shift_matrix[noise_mask] - centroids[cid], axis=1)
             for cid in cluster_ids],
            axis=1,
        )
        effective_labels[noise_mask] = np.argmin(noise_dists, axis=1)

    # Compute soft membership via T=10 inverse-distance
    dists = np.zeros((len(shift_matrix), n_clusters))
    for cid in cluster_ids:
        dists[:, cid] = np.linalg.norm(shift_matrix - centroids[cid], axis=1)
    soft_scores = temperature_soft_membership(dists, T=TEMPERATURE)  # (N, K)

    # Type means on holdout using soft_scores
    weight_sums = soft_scores.sum(axis=0, keepdims=True).T  # (K, 1)
    weight_sums = np.where(weight_sums == 0, 1.0, weight_sums)
    type_means = (soft_scores.T @ holdout) / weight_sums  # (K, H)

    # County-prior predictions
    predicted = soft_scores @ type_means  # (N, H)

    # Mean of per-column Pearson r (matches spectral experiment methodology)
    # Each of the 3 holdout dimensions (pres_d, pres_r, pres_turnout) gets its own r,
    # then we average — this is the same metric used in data/validation/spectral_experiment_results.parquet.
    r_values = []
    for d in range(holdout.shape[1]):
        if np.std(predicted[:, d]) < 1e-10 or np.std(holdout[:, d]) < 1e-10:
            r_values.append(0.0)
        else:
            r, _ = pearsonr(predicted[:, d], holdout[:, d])
            r_values.append(float(r))
    return float(np.mean(r_values))


def run_kmeans_baseline(shift_matrix: np.ndarray, holdout: np.ndarray) -> float:
    """Reproduce KMeans J=43 holdout r as a local sanity check."""
    km = KMeans(n_clusters=KMEANS_J, random_state=42, n_init=10)
    labels = km.fit_predict(shift_matrix)
    return compute_holdout_r_from_labels(labels, shift_matrix, holdout, KMEANS_J)


def run_hdbscan_sweep(
    shift_matrix: np.ndarray,
    holdout: np.ndarray,
) -> list[dict]:
    """Run HDBSCAN across min_cluster_size values and evaluate each."""
    results = []

    for mcs in MIN_CLUSTER_SIZES:
        print(f"\n  Running HDBSCAN(min_cluster_size={mcs})...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            metric="euclidean",
            cluster_selection_method="eom",  # Excess of Mass — HDBSCAN default
        )
        labels = clusterer.fit_predict(shift_matrix)

        n_noise = int((labels == -1).sum())
        unique_labels = set(labels) - {-1}
        n_clusters = len(unique_labels)

        if n_clusters == 0:
            print(f"    -> All {len(labels)} counties flagged as noise! Skipping.")
            results.append({
                "min_cluster_size": mcs,
                "n_clusters": 0,
                "n_noise": n_noise,
                "noise_pct": 100.0 * n_noise / len(labels),
                "holdout_r": float("nan"),
                "delta_vs_kmeans": float("nan"),
            })
            continue

        holdout_r = compute_holdout_r_from_labels(
            labels, shift_matrix, holdout, n_clusters
        )
        delta = holdout_r - KMEANS_HOLDOUT_R

        print(f"    -> n_clusters={n_clusters}, n_noise={n_noise} ({100.*n_noise/len(labels):.1f}%), holdout_r={holdout_r:.4f}, delta={delta:+.4f}")

        results.append({
            "min_cluster_size": mcs,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_pct": round(100.0 * n_noise / len(labels), 1),
            "holdout_r": round(holdout_r, 4),
            "delta_vs_kmeans": round(delta, 4),
        })

    return results


def print_summary_table(results: list[dict], kmeans_local_r: float) -> None:
    """Print a formatted results table."""
    print("\n" + "=" * 75)
    print("HDBSCAN Experiment Results -- P3.4")
    print("=" * 75)
    print(f"\nKMeans J=43 baseline (production):  holdout r = {KMEANS_HOLDOUT_R:.4f}")
    print(f"KMeans J=43 (local verification):   holdout r = {kmeans_local_r:.4f}")
    print()
    print(
        f"{'min_cluster_size':>17} | {'n_clusters':>10} | {'n_noise':>7} | {'noise_pct':>9} | {'holdout_r':>9} | {'delta':>8} | {'vs_baseline':>12}"
    )
    print("-" * 75)
    for r in results:
        if r["n_clusters"] == 0:
            print(
                f"{r['min_cluster_size']:>17} | {'0':>10} | {r['n_noise']:>7} | {'100.0':>9} | {'N/A':>9} | {'N/A':>8} | {'N/A':>12}"
            )
        else:
            winner = "BEATS KMeans" if r["delta_vs_kmeans"] > 0 else "below KMeans"
            print(
                f"{r['min_cluster_size']:>17} | {r['n_clusters']:>10} | {r['n_noise']:>7} | "
                f"{r['noise_pct']:>8.1f}% | {r['holdout_r']:>9.4f} | {r['delta_vs_kmeans']:>+8.4f} | {winner:>12}"
            )
    print("=" * 75)

    valid = [r for r in results if not np.isnan(r.get("holdout_r", float("nan")))]
    if valid:
        best = max(valid, key=lambda x: x["holdout_r"])
        print(f"\nBest HDBSCAN result: min_cluster_size={best['min_cluster_size']}, "
              f"n_clusters={best['n_clusters']}, holdout_r={best['holdout_r']:.4f}")
        if best["delta_vs_kmeans"] > 0:
            print(f"-> HDBSCAN BEATS KMeans J=43 baseline by {best['delta_vs_kmeans']:+.4f}")
        else:
            print(f"-> HDBSCAN does NOT beat KMeans J=43 baseline (best delta={best['delta_vs_kmeans']:+.4f})")


def main() -> None:
    print("=" * 75)
    print("HDBSCAN Clustering Experiment -- P3.4")
    print("=" * 75)

    # Load data
    shift_matrix, holdout = load_shift_matrix()

    # Local KMeans verification
    print("\nVerifying KMeans J=43 baseline locally...")
    kmeans_local_r = run_kmeans_baseline(shift_matrix, holdout)
    print(f"  KMeans J=43 local holdout r = {kmeans_local_r:.4f}  (reported: {KMEANS_HOLDOUT_R:.4f})")

    # HDBSCAN sweep
    print(f"\nRunning HDBSCAN sweep: min_cluster_size = {MIN_CLUSTER_SIZES}")
    results = run_hdbscan_sweep(shift_matrix, holdout)

    # Print summary
    print_summary_table(results, kmeans_local_r)

    # Save results
    out_dir = PROJECT_ROOT / "data" / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hdbscan_experiment_results.parquet"
    pd.DataFrame(results).to_parquet(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
