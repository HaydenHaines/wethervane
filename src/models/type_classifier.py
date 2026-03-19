"""Layer 2: Electoral type classification via NMF on community shift profiles.

Takes Layer 1 community assignments + shift vectors, computes per-community
mean shift profiles, then fits NMF to discover J electoral types.

Types are abstract archetypes that can appear in multiple non-contiguous communities.
Rural Georgia and rural Washington may be different communities but the same type.

This is a STUB — full implementation is Phase 1.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def compute_community_profiles(
    shifts: pd.DataFrame,
    assignments: pd.DataFrame,
    shift_cols: list[str],
) -> pd.DataFrame:
    """Compute mean shift vector for each community.

    Parameters
    ----------
    shifts : DataFrame with county_fips + shift columns
    assignments : DataFrame with county_fips + community_id
    shift_cols : list of shift column names to average

    Returns
    -------
    DataFrame with community_id + one column per shift dim (mean across member counties)
    """
    merged = shifts.merge(assignments[["county_fips", "community_id"]], on="county_fips")
    profiles = (
        merged.groupby("community_id")[shift_cols]
        .mean()
        .reset_index()
    )
    return profiles


def classify_types(
    community_profiles: pd.DataFrame,
    shift_cols: list[str],
    j: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """Fit NMF to community shift profiles -> J electoral types.

    STUB: Returns placeholder uniform assignments.
    Replace with real NMF implementation in Phase 1.

    Returns
    -------
    DataFrame with community_id + type_weight_{j} for j in range(J)
    + dominant_type_id (argmax of weights)
    """
    log.warning(
        "type_classifier.classify_types is a STUB. "
        "Returns uniform placeholder weights. Implement NMF in Phase 1."
    )
    n_communities = len(community_profiles)
    weight_cols = {f"type_weight_{j_idx}": 1.0 / j for j_idx in range(j)}
    result = community_profiles[["community_id"]].copy()
    for col, val in weight_cols.items():
        result[col] = val
    result["dominant_type_id"] = 0  # placeholder
    return result


def run_type_classification(
    shifts_path: Path,
    assignments_path: Path,
    shift_cols: list[str],
    j: int,
    output_path: Path,
) -> pd.DataFrame:
    """End-to-end type classification pipeline.

    STUB -- wires compute_community_profiles -> classify_types -> save.
    """
    shifts = pd.read_parquet(shifts_path)
    assignments = pd.read_parquet(assignments_path)

    profiles = compute_community_profiles(shifts, assignments, shift_cols)
    type_assignments = classify_types(profiles, shift_cols, j)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    type_assignments.to_parquet(output_path, index=False)
    log.info("Type assignments (stub) saved -> %s", output_path)
    return type_assignments
