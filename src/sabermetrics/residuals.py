"""Candidate residual computation.

The residual = actual - expected is the raw material for all
electoral sabermetric stats. When the community-covariance model
is available, residuals are decomposed by community type to produce
the CTOV (Community-Type Overperformance Vector).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


def compute_mvd(
    actual_results: "pd.DataFrame",
    baselines: "pd.DataFrame",
    environment: float,
) -> "pd.DataFrame":
    """Compute Marginal Vote Delivery (MVD) for each candidate.

    MVD = actual_vote_share - (baseline + environment_adjustment)

    Parameters
    ----------
    actual_results : pd.DataFrame
        Columns: candidate_id, district_id, vote_share, votes.
    baselines : pd.DataFrame
        Columns: district_id, baseline, baseline_source.
    environment : float
        National environment adjustment.

    Returns
    -------
    pd.DataFrame
        Columns: candidate_id, district_id, mvd, actual, expected.
    """
    raise NotImplementedError


def compute_ctov(
    actual_results_by_county: "pd.DataFrame",
    W: "np.ndarray",
    type_baselines: "np.ndarray",
    county_to_district: "pd.DataFrame",
) -> "pd.DataFrame":
    """Compute Community-Type Overperformance Vector (CTOV).

    Projects county-level residuals onto community-type basis
    using the weight matrix W. Returns one K-length vector per
    candidate per race.

    Parameters
    ----------
    actual_results_by_county : pd.DataFrame
        County-level results: fips, candidate_id, vote_share.
    W : np.ndarray
        County x type weight matrix (N_counties x K_types).
    type_baselines : np.ndarray
        Type-level expected vote share (K_types,).
    county_to_district : pd.DataFrame
        Mapping from county FIPS to district.

    Returns
    -------
    pd.DataFrame
        One row per candidate x race. Columns: candidate_id,
        race_id, ctov_type_0, ctov_type_1, ..., ctov_type_K.
    """
    raise NotImplementedError


def compute_polling_gap(
    actual_results: "pd.DataFrame",
    final_polling_averages: "pd.DataFrame",
    cycle_systematic_error: float | None = None,
) -> "pd.DataFrame":
    """Compute cycle-adjusted polling gap.

    Raw gap: actual - final_polling_average
    Adjusted: raw_gap - cycle_median_error

    The adjustment removes the within-cycle correlated polling
    error (Gelman et al.), isolating the candidate-specific signal.

    Parameters
    ----------
    actual_results : pd.DataFrame
        Columns: candidate_id, district_id, vote_share.
    final_polling_averages : pd.DataFrame
        Columns: district_id, polling_avg, n_polls.
    cycle_systematic_error : float | None
        Median (actual - polling_avg) across all races in the
        cycle. If None, computed from the data.

    Returns
    -------
    pd.DataFrame
        Columns: candidate_id, district_id, raw_gap, adjusted_gap,
        n_polls_in_district.
    """
    raise NotImplementedError


def compute_cec(ctov_history: list["np.ndarray"]) -> float:
    """Compute Cross-Election Consistency (CEC).

    Mean pairwise Pearson correlation of CTOV vectors across
    elections for the same candidate. High CEC = genuine skill
    with specific community types. Low CEC = lucky or context-driven.

    Parameters
    ----------
    ctov_history : list[np.ndarray]
        List of CTOV vectors from successive elections (each K-length).

    Returns
    -------
    float
        CEC score in [-1, 1]. Typically 0.3-0.7 for skilled candidates.
    """
    raise NotImplementedError
