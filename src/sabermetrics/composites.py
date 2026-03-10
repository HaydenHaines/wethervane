"""Composite scores, fit scoring, and talent pipeline.

Aggregates per-race and per-Congress stats into career summaries.
Computes candidate-district fit scores for recruitment scouting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


def compute_career_summary(
    electoral_stats: "pd.DataFrame",
    campaign_stats: "pd.DataFrame",
    legislative_stats: "pd.DataFrame",
    constituent_stats: "pd.DataFrame | None" = None,
) -> dict:
    """Aggregate per-race/per-Congress stats into a career summary.

    Returns
    -------
    dict
        Career-level aggregates: mean_mvd, best_mvd, cec,
        career_ctov (weighted average), mean_les, mean_sdr,
        career_bdr, career_sdl, mean_car, n_races, n_congresses.
    """
    raise NotImplementedError


def compute_fit_score(
    candidate_ctov: "np.ndarray",
    district_type_composition: "np.ndarray",
) -> float:
    """Candidate-district fit score.

    fit = dot(CTOV, district_type_composition)

    Higher = candidate's community-type strengths align with the
    district's community-type composition. A candidate who
    overperforms in suburban-professional types has high fit in
    districts with high suburban-professional weight.

    Parameters
    ----------
    candidate_ctov : np.ndarray
        Candidate's career-average CTOV (K_types,).
    district_type_composition : np.ndarray
        Target district's community-type weight vector (K_types,).

    Returns
    -------
    float
        Fit score (higher = better match).
    """
    raise NotImplementedError


def rank_candidates_for_district(
    candidate_pool: "pd.DataFrame",
    ctov_matrix: "np.ndarray",
    target_district_W: "np.ndarray",
    min_races: int = 2,
) -> "pd.DataFrame":
    """Rank candidates by fit score for a target district.

    The Moneyball scouting report: given a district's community-type
    composition, which candidates in the pool have the best
    skill-to-district match?

    Parameters
    ----------
    candidate_pool : pd.DataFrame
        Columns: candidate_id, name, party, current_office, n_races.
    ctov_matrix : np.ndarray
        Career-average CTOV for each candidate (N_candidates x K_types).
    target_district_W : np.ndarray
        Target district's community-type weights (K_types,).
    min_races : int
        Minimum races required for reliable CTOV estimate.

    Returns
    -------
    pd.DataFrame
        Columns: candidate_id, name, fit_score, rank,
        top_matching_types (which types drive the fit),
        cec (reliability of the CTOV).
    """
    raise NotImplementedError


def portfolio_analysis(
    candidates: "pd.DataFrame",
    ctov_matrix: "np.ndarray",
    districts: "pd.DataFrame",
    W_matrix: "np.ndarray",
) -> "pd.DataFrame":
    """Portfolio optimization: find candidate slate that covers the most community types.

    Treats candidates as assets with community-type-specific returns.
    Identifies the combination of candidates across districts that
    maximizes total community-type coverage with minimal overlap.

    Parameters
    ----------
    candidates : pd.DataFrame
        Candidate pool with fit scores per district.
    ctov_matrix : np.ndarray
        CTOV vectors for all candidates.
    districts : pd.DataFrame
        Target districts with their type compositions.
    W_matrix : np.ndarray
        Full county x type weight matrix.

    Returns
    -------
    pd.DataFrame
        Optimal and near-optimal candidate assignments per district,
        with total portfolio score and coverage analysis.
    """
    raise NotImplementedError


def project_from_lower_office(
    lower_office_ctov: "np.ndarray",
    lower_office_W: "np.ndarray",
    target_W: "np.ndarray",
    n_races: int,
) -> dict:
    """Project a lower-office CTOV to a higher-office district.

    The talent pipeline: estimate how a state legislator or county
    official would perform in a congressional race, based on their
    community-type overperformance in lower-level races.

    Parameters
    ----------
    lower_office_ctov : np.ndarray
        CTOV from lower-office races (may cover fewer types).
    lower_office_W : np.ndarray
        Type composition of the lower-office district.
    target_W : np.ndarray
        Type composition of the target higher-office district.
    n_races : int
        Number of lower-office races (affects confidence).

    Returns
    -------
    dict
        Keys: projected_fit_score, confidence_interval,
        types_with_evidence (types observed in lower office),
        types_extrapolated (types only in target district),
        projection_reliability.
    """
    raise NotImplementedError
