"""Cluster census tracts into communities using shift-vector similarity.

Performs constrained agglomerative clustering on political shift vectors with
Queen contiguity as the connectivity constraint. The dendrogram is saved to
disk for downstream threshold selection.

Outputs:
    data/communities/community_assignments.parquet  — tract → cluster label
    data/communities/dendrogram.pkl                 — scipy linkage matrix + metadata

Usage:
    uv run python src/discovery/cluster_communities.py
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMMUNITIES_DIR = PROJECT_ROOT / "data" / "communities"

# ── Core functions ─────────────────────────────────────────────────────────────


def normalize_shifts(
    shifts: np.ndarray,
    n_presidential_dims: int = 6,
) -> np.ndarray:
    """Zero-mean, unit-variance normalisation with midterm column upweighting.

    Presidential election dimensions (cols 0..n_presidential_dims-1) are
    standardised to mean=0, std=1.  Midterm dimensions (cols
    n_presidential_dims..) are additionally scaled by sqrt(2) so that their
    variance is 2, compensating for the higher noise in off-cycle elections.

    Parameters
    ----------
    shifts:
        Array of shape (n_tracts, n_dims).
    n_presidential_dims:
        Number of leading columns that correspond to presidential elections.

    Returns
    -------
    np.ndarray
        Normalised array of the same shape.
    """
    normed = shifts.copy().astype(float)
    n_cols = normed.shape[1]

    # Zero-mean, unit-variance per column
    col_means = normed.mean(axis=0)
    col_stds = normed.std(axis=0, ddof=0)
    # Guard against zero-variance columns
    col_stds = np.where(col_stds == 0, 1.0, col_stds)
    normed = (normed - col_means) / col_stds

    # Scale midterm columns by sqrt(2)
    if n_presidential_dims < n_cols:
        normed[:, n_presidential_dims:] *= np.sqrt(2)

    return normed


def cluster_at_threshold(
    shifts: np.ndarray,
    connectivity,
    threshold: float | None = None,
    n_clusters: int | None = None,
):
    """Run Ward agglomerative clustering with spatial connectivity constraint.

    Exactly one of *threshold* or *n_clusters* must be provided (or both may
    be provided — sklearn accepts both but *n_clusters* takes precedence).

    Parameters
    ----------
    shifts:
        Normalised shift array of shape (n_tracts, n_dims).
    connectivity:
        Sparse adjacency matrix (scipy CSR) used as the connectivity constraint.
    threshold:
        Distance threshold at which the dendrogram is cut.
    n_clusters:
        Target number of clusters (overrides threshold when both are given).

    Returns
    -------
    labels : np.ndarray
        Integer cluster label per tract.
    model : AgglomerativeClustering
        Fitted sklearn model (exposes ``children_`` and ``distances_``).
    """
    from sklearn.cluster import AgglomerativeClustering

    kwargs: dict = {
        "linkage": "ward",
        "connectivity": connectivity,
        "compute_distances": True,
    }
    if n_clusters is not None:
        kwargs["n_clusters"] = n_clusters
        # Remove distance_threshold default so sklearn does not complain
        kwargs.pop("distance_threshold", None)
    elif threshold is not None:
        kwargs["n_clusters"] = None
        kwargs["distance_threshold"] = threshold
    else:
        raise ValueError("Provide threshold or n_clusters.")

    model = AgglomerativeClustering(**kwargs)
    labels = model.fit_predict(shifts)
    return labels, model


def build_linkage_matrix(model) -> np.ndarray:
    """Convert a fitted AgglomerativeClustering model to scipy linkage format.

    The scipy linkage format is a (n-1, 4) array where each row is
    [left_child, right_child, distance, cluster_size].

    Connectivity-constrained Ward clustering may produce non-monotonic merge
    distances.  This function sorts merges by distance and re-indexes internal
    node IDs so that the result satisfies the scipy monotonicity requirement
    (used by ``scipy.cluster.hierarchy.dendrogram`` and ``fcluster``).

    Parameters
    ----------
    model:
        A fitted ``AgglomerativeClustering`` instance with ``compute_distances=True``.

    Returns
    -------
    np.ndarray
        Linkage matrix of shape (n_samples - 1, 4), rows sorted by distance.
    """
    children = model.children_.copy()  # (n-1, 2)
    distances = model.distances_.copy()  # (n-1,)
    n_samples = len(children) + 1

    # Sort merges by ascending distance for scipy compatibility
    sort_order = np.argsort(distances, kind="stable")
    children = children[sort_order]
    distances = distances[sort_order]

    # Re-map internal node IDs: internal node i (originally n_samples + i)
    # is now the k-th merge in sorted order, so its new ID is n_samples + k.
    old_to_new = np.empty(n_samples + len(children), dtype=int)
    old_to_new[:n_samples] = np.arange(n_samples)  # leaf nodes unchanged
    for new_idx, old_idx in enumerate(sort_order):
        old_to_new[n_samples + old_idx] = n_samples + new_idx

    # Remap child references
    children_remapped = old_to_new[children]

    # Compute cluster sizes in the new order
    sizes = np.ones(n_samples + len(children), dtype=float)
    for k, (left, right) in enumerate(children_remapped):
        sizes[n_samples + k] = sizes[left] + sizes[right]

    linkage = np.column_stack([
        children_remapped[:, 0].astype(float),
        children_remapped[:, 1].astype(float),
        distances,
        sizes[n_samples:],
    ])
    return linkage


def find_elbow(n_communities: np.ndarray, variances: np.ndarray) -> int | None:
    """Find the elbow in the variance-vs-communities curve.

    Uses the Kneedle algorithm to locate the point of maximum curvature on a
    convex, decreasing curve.

    Parameters
    ----------
    n_communities:
        Array of community counts (typically decreasing).
    variances:
        Corresponding within-cluster variance values.

    Returns
    -------
    int or None
        The community count at the elbow, or None if kneed cannot find one.
    """
    from kneed import KneeLocator

    kl = KneeLocator(
        x=n_communities,
        y=variances,
        curve="convex",
        direction="decreasing",
        S=1.0,
    )
    if kl.knee is None:
        return None
    return int(kl.knee)


def sweep_thresholds(
    shifts: np.ndarray,
    connectivity,
    linkage: np.ndarray,
    n_steps: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Cut dendrogram at evenly-spaced heights and compute within-cluster variance.

    Parameters
    ----------
    shifts:
        Normalised shift array of shape (n_tracts, n_dims).
    connectivity:
        Sparse adjacency matrix.
    linkage:
        Scipy-format linkage matrix from :func:`build_linkage_matrix`.
    n_steps:
        Number of threshold values to evaluate.

    Returns
    -------
    n_communities : np.ndarray
        Number of communities at each threshold.
    variances : np.ndarray
        Weighted mean within-cluster variance at each threshold.
    """
    from scipy.cluster.hierarchy import fcluster

    min_dist = linkage[:, 2].min()
    max_dist = linkage[:, 2].max()
    thresholds = np.linspace(min_dist + 1e-6, max_dist, n_steps)

    n_communities_list = []
    variances_list = []

    for t in thresholds:
        flat_labels = fcluster(linkage, t, criterion="distance")
        unique_labels, counts = np.unique(flat_labels, return_counts=True)
        k = len(unique_labels)

        # Weighted mean within-cluster variance
        total_var = 0.0
        total_weight = 0.0
        for label, count in zip(unique_labels, counts):
            mask = flat_labels == label
            cluster_data = shifts[mask]
            if count > 1:
                var = np.var(cluster_data, ddof=0)
            else:
                var = 0.0
            total_var += var * count
            total_weight += count

        weighted_var = total_var / total_weight if total_weight > 0 else 0.0
        n_communities_list.append(k)
        variances_list.append(weighted_var)

    return np.array(n_communities_list), np.array(variances_list)


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    """Load data, normalize, cluster, and save community assignments."""
    import pandas as pd
    from scipy.sparse import load_npz

    adjacency_path = COMMUNITIES_DIR / "adjacency.npz"
    geoids_path = COMMUNITIES_DIR / "adjacency.geoids.txt"
    shifts_path = COMMUNITIES_DIR / "shift_vectors.parquet"

    log.info("Loading adjacency matrix …")
    W = load_npz(str(adjacency_path))
    geoids = geoids_path.read_text().splitlines()

    log.info("Loading shift vectors …")
    shifts_df = pd.read_parquet(shifts_path)
    shift_cols = [c for c in shifts_df.columns if c != "tract_geoid"]
    shifts = shifts_df[shift_cols].values

    log.info("Normalising shift vectors …")
    shifts_norm = normalize_shifts(shifts, n_presidential_dims=6)

    log.info("Building full dendrogram (n_clusters=1) …")
    _, model = cluster_at_threshold(shifts_norm, W, n_clusters=1)
    linkage = build_linkage_matrix(model)

    log.info("Sweeping thresholds to find elbow …")
    n_comms, variances = sweep_thresholds(shifts_norm, W, linkage)
    elbow_k = find_elbow(n_comms, variances)
    log.info("Elbow community count: %s", elbow_k)

    # Cluster at elbow or fallback default
    k_target = elbow_k if elbow_k is not None else 50
    log.info("Clustering at k=%d communities …", k_target)
    labels, _ = cluster_at_threshold(shifts_norm, W, n_clusters=k_target)

    COMMUNITIES_DIR.mkdir(parents=True, exist_ok=True)

    # Save assignments
    assignments_df = pd.DataFrame({"tract_geoid": geoids, "community": labels})
    out_path = COMMUNITIES_DIR / "community_assignments.parquet"
    assignments_df.to_parquet(out_path, index=False)
    log.info("Community assignments saved to %s", out_path)

    # Save dendrogram
    dendro_path = COMMUNITIES_DIR / "dendrogram.pkl"
    with open(dendro_path, "wb") as f:
        pickle.dump({"linkage": linkage, "n_communities": n_comms, "variances": variances}, f)
    log.info("Dendrogram saved to %s", dendro_path)


if __name__ == "__main__":
    main()
