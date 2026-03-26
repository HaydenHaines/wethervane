"""Hierarchical nesting of fine types into super-types.

Uses Ward HAC on type feature vectors to group fine types into a smaller
number of super-types for public interpretability. When J is large (>20),
nesting on demographic profiles produces better-balanced clusters than
nesting on shift-space centroids.

Usage:
    python -m src.discovery.nest_types
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


@dataclass
class NestingResult:
    mapping: dict[int, int]  # fine_type_id -> super_type_id
    best_s: int  # selected number of super-types
    silhouette_scores: dict[int, float]  # s -> silhouette score


def nest_types(
    type_features: np.ndarray,
    s_candidates: list[int] | None = None,
) -> NestingResult:
    """Group fine types into super-types via Ward HAC on feature vectors.

    Parameters
    ----------
    type_features : ndarray of shape (J, D)
        Feature vectors for each type. Can be shift-space centroids,
        demographic profiles, or any numeric representation.
    s_candidates : list[int], optional
        Candidate numbers of super-types. Defaults to [5, 6, 7, 8].

    Returns
    -------
    NestingResult
        Mapping from fine type to super-type, best S, and silhouette scores.
    """
    if s_candidates is None:
        s_candidates = [5, 6, 7, 8]

    j = type_features.shape[0]

    # Ward linkage on feature vectors
    Z = linkage(type_features, method="ward")

    # Evaluate each candidate S
    sil_scores: dict[int, float] = {}
    label_maps: dict[int, np.ndarray] = {}

    for s in s_candidates:
        if s >= j:
            # Can't have more super-types than fine types
            sil_scores[s] = -1.0
            label_maps[s] = np.arange(j)
            continue

        labels = fcluster(Z, t=s, criterion="maxclust")
        # Re-index to 0-based contiguous
        unique_labels = sorted(set(labels))
        remap = {old: new for new, old in enumerate(unique_labels)}
        labels_0 = np.array([remap[l] for l in labels])
        label_maps[s] = labels_0

        if len(set(labels_0)) < 2:
            sil_scores[s] = -1.0
        else:
            sil_scores[s] = float(silhouette_score(type_features, labels_0))

    # Select S with highest silhouette score
    best_s = max(sil_scores, key=lambda s: sil_scores[s])
    best_labels = label_maps[best_s]

    mapping = {i: int(best_labels[i]) for i in range(j)}

    return NestingResult(
        mapping=mapping,
        best_s=best_s,
        silhouette_scores=sil_scores,
    )


# ── CLI entry point ──────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]


DEMO_NESTING_FEATURES = [
    "pct_white_nh",
    "pct_black",
    "pct_hispanic",
    "pct_asian",
    "pct_bachelors_plus",
    "median_hh_income",
    "median_age",
    "evangelical_share",
    "catholic_share",
    "black_protestant_share",
    "religious_adherence_rate",
]


def main() -> None:
    import pandas as pd

    # Load config
    config_path = PROJECT_ROOT / "config" / "model.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    s_candidates = config["types"]["super_type_count_candidates"]

    # Load type assignments
    type_path = PROJECT_ROOT / "data" / "communities" / "type_assignments.parquet"
    if not type_path.exists():
        raise FileNotFoundError(
            f"Run type discovery first: python -m src.discovery.run_type_discovery\n"
            f"Expected: {type_path}"
        )
    type_df = pd.read_parquet(type_path)

    # Load type profiles for demographic-based nesting
    profiles_path = PROJECT_ROOT / "data" / "communities" / "type_profiles.parquet"
    if not profiles_path.exists():
        raise FileNotFoundError(
            f"Run describe_types first: python -m src.description.describe_types\n"
            f"Expected: {profiles_path}"
        )
    profiles_df = pd.read_parquet(profiles_path)
    profiles_df = profiles_df.sort_values("type_id").reset_index(drop=True)

    # Build feature matrix from demographic profiles
    available = [c for c in DEMO_NESTING_FEATURES if c in profiles_df.columns]
    print(f"Nesting features ({len(available)}): {available}")
    feature_matrix = profiles_df[available].values

    # StandardScaler so all demographics are on equal footing
    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(feature_matrix)

    # Handle any NaN (missing demographics for some types)
    nan_mask = np.isnan(feature_matrix)
    if nan_mask.any():
        col_means = np.nanmean(feature_matrix, axis=0)
        for col_idx in range(feature_matrix.shape[1]):
            feature_matrix[nan_mask[:, col_idx], col_idx] = col_means[col_idx]

    j = feature_matrix.shape[0]
    print(f"Nesting {j} fine types into {s_candidates} super-type candidates...")
    nesting = nest_types(feature_matrix, s_candidates=s_candidates)

    print(f"\nSilhouette scores: {nesting.silhouette_scores}")
    print(f"Best S: {nesting.best_s}")

    # Show distribution
    from collections import Counter
    dist = Counter(nesting.mapping.values())
    print(f"Super-type distribution: {dict(sorted(dist.items()))}")

    # Save nesting to the type assignments
    type_df["super_type"] = type_df["dominant_type"].map(nesting.mapping)
    type_df.to_parquet(type_path, index=False)
    print(f"Updated {type_path} with super_type column")


if __name__ == "__main__":
    main()
