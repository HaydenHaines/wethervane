"""Pairwise type similarity and geodesic distance computation for SCI validation.

Contains the core computation functions:
- haversine_km: vectorized great-circle distance
- compute_pairwise_type_similarity: cosine sim of soft type vectors per pair
- add_geodesic_distance: attach haversine distance to a pair DataFrame
- compute_partial_correlation: Pearson r of x and y controlling for z

These are pure numpy/scipy computations with no I/O dependencies, making
them independently testable.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

log = logging.getLogger(__name__)

# Haversine Earth radius in km
_EARTH_RADIUS_KM = 6371.0

# Distance bins for the proximity control (km)
DISTANCE_BINS = [0, 100, 250, 500, 1000, 2000, 5000]
DISTANCE_LABELS = [
    "0-100km",
    "100-250km",
    "250-500km",
    "500-1000km",
    "1000-2000km",
    "2000-5000km",
]


def haversine_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Vectorized haversine distance in km."""
    lat1_r, lat2_r = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def compute_pairwise_type_similarity(
    sci_pairs: pd.DataFrame,
    fips_to_idx: dict[str, int],
    score_matrix: np.ndarray,
    dominant_types: np.ndarray,
) -> pd.DataFrame:
    """Add type similarity columns to SCI pair DataFrame.

    Adds:
    - same_type: bool, whether both counties share the same dominant type
    - cosine_sim: cosine similarity of soft membership vectors
    - log_sci: log10(scaled_sci)
    - same_state: bool, whether both counties are in the same state
    """
    log.info("Computing pairwise type similarity for %d pairs...", len(sci_pairs))

    user_idx = sci_pairs["user_fips"].map(fips_to_idx).values
    friend_idx = sci_pairs["friend_fips"].map(fips_to_idx).values

    # Same dominant type
    sci_pairs = sci_pairs.copy()
    sci_pairs["same_type"] = dominant_types[user_idx] == dominant_types[friend_idx]

    # Cosine similarity of soft membership vectors (vectorized)
    user_vecs = score_matrix[user_idx]  # (N_pairs, J)
    friend_vecs = score_matrix[friend_idx]  # (N_pairs, J)
    # cosine_sim = dot(u, v) / (||u|| * ||v||)
    dot_products = np.sum(user_vecs * friend_vecs, axis=1)
    user_norms = np.linalg.norm(user_vecs, axis=1)
    friend_norms = np.linalg.norm(friend_vecs, axis=1)
    # Avoid division by zero (shouldn't happen with valid scores)
    denom = user_norms * friend_norms
    denom = np.where(denom > 0, denom, 1.0)
    sci_pairs["cosine_sim"] = dot_products / denom

    # Log-scaled SCI
    sci_pairs["log_sci"] = np.log10(sci_pairs["scaled_sci"].clip(lower=1))

    # Same-state indicator
    sci_pairs["same_state"] = (
        sci_pairs["user_fips"].str[:2] == sci_pairs["friend_fips"].str[:2]
    )

    log.info(
        "  Same-type pairs: %d (%.1f%%), different-type: %d",
        sci_pairs["same_type"].sum(),
        100 * sci_pairs["same_type"].mean(),
        (~sci_pairs["same_type"]).sum(),
    )
    return sci_pairs


def add_geodesic_distance(
    pairs: pd.DataFrame,
    centroids: pd.DataFrame,
) -> pd.DataFrame:
    """Add geodesic distance (km) between county centroids to pair DataFrame."""
    log.info("Computing geodesic distances for %d pairs...", len(pairs))

    centroid_map = centroids.set_index("county_fips")[["latitude", "longitude"]]

    pairs = pairs.copy()
    user_coords = pairs["user_fips"].map(centroid_map["latitude"]).values
    user_lon = pairs["user_fips"].map(centroid_map["longitude"]).values
    friend_coords = pairs["friend_fips"].map(centroid_map["latitude"]).values
    friend_lon = pairs["friend_fips"].map(centroid_map["longitude"]).values

    # Drop pairs where centroids are missing
    valid_mask = ~(
        np.isnan(user_coords)
        | np.isnan(user_lon)
        | np.isnan(friend_coords)
        | np.isnan(friend_lon)
    )
    if not valid_mask.all():
        n_dropped = (~valid_mask).sum()
        log.warning("  %d pairs lack centroid data, dropping", n_dropped)
        pairs = pairs[valid_mask].copy()
        user_coords = user_coords[valid_mask]
        user_lon = user_lon[valid_mask]
        friend_coords = friend_coords[valid_mask]
        friend_lon = friend_lon[valid_mask]

    pairs["distance_km"] = haversine_km(
        user_coords, user_lon, friend_coords, friend_lon
    )
    pairs["log_distance"] = np.log10(pairs["distance_km"].clip(lower=1))

    log.info("  Distance range: %.0f - %.0f km", pairs["distance_km"].min(), pairs["distance_km"].max())
    return pairs


def compute_partial_correlation(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[float, float]:
    """Partial Pearson correlation of x and y controlling for z.

    Uses the standard formula: regress x on z, regress y on z, correlate
    residuals.
    """
    # Residualize x on z
    z_with_const = np.column_stack([z, np.ones_like(z)])
    beta_x = np.linalg.lstsq(z_with_const, x, rcond=None)[0]
    resid_x = x - z_with_const @ beta_x

    beta_y = np.linalg.lstsq(z_with_const, y, rcond=None)[0]
    resid_y = y - z_with_const @ beta_y

    return pearsonr(resid_x, resid_y)
