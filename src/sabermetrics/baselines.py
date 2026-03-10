"""District baseline computation.

Computes the "expected result for a generic candidate" in each
district, which is the denominator of every sabermetric stat.
Three approaches: Cook PVI, Inside Elections Baseline (when
available), and model-native structural baseline (when the
community-covariance model is available).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


def compute_cook_pvi(
    election_returns: "pd.DataFrame",
    most_recent_year: int = 2024,
) -> "pd.DataFrame":
    """Compute Cook PVI from last two presidential elections.

    Formula: 75% weight on most recent + 25% on prior presidential
    election, relative to national two-party vote share.

    Parameters
    ----------
    election_returns : pd.DataFrame
        County/district-level presidential returns with columns:
        fips, year, dem_votes, rep_votes.
    most_recent_year : int
        Most recent presidential election year.

    Returns
    -------
    pd.DataFrame
        Columns: district_id, pvi_score, pvi_label (e.g., "D+3").
    """
    raise NotImplementedError


def compute_structural_baseline(
    W: "np.ndarray",
    type_estimates: "np.ndarray",
    national_environment: float,
) -> "pd.DataFrame":
    """Compute model-native structural baseline per district.

    Uses community-type weight matrix W and type-level estimates
    from the covariance pipeline. This is the model's prediction
    for a generic candidate given the national environment.

    Parameters
    ----------
    W : np.ndarray
        County x type weight matrix (N_counties x K_types).
    type_estimates : np.ndarray
        Type-level expected vote share (K_types,).
    national_environment : float
        National environment adjustment (from fundamentals).

    Returns
    -------
    pd.DataFrame
        Columns: fips, expected_vote_share, expected_turnout,
        by community type contributions.
    """
    raise NotImplementedError


def compute_national_environment(
    generic_ballot: float | None = None,
    presidential_approval: float | None = None,
    gdp_growth: float | None = None,
    midterm: bool = False,
) -> float:
    """Estimate national environment from fundamentals.

    Follows Abramowitz: generic ballot is the primary predictor
    (r=.82 with national House popular vote over postwar midterms).

    Parameters
    ----------
    generic_ballot : float | None
        Generic ballot margin (positive = D advantage).
    presidential_approval : float | None
        Presidential approval rating.
    gdp_growth : float | None
        GDP growth rate (Q2 of election year).
    midterm : bool
        Whether this is a midterm election (different model).

    Returns
    -------
    float
        Estimated national environment (positive = D-favorable).
    """
    raise NotImplementedError


def aggregate_baselines(
    cook_pvi: "pd.DataFrame",
    structural_baseline: "pd.DataFrame | None" = None,
    inside_elections: "pd.DataFrame | None" = None,
) -> "pd.DataFrame":
    """Combine baseline estimates, preferring more comprehensive sources.

    Priority: Inside Elections Baseline > model-native structural
    baseline > Cook PVI.

    Returns
    -------
    pd.DataFrame
        Columns: district_id, baseline, baseline_source, baseline_ci_lower,
        baseline_ci_upper.
    """
    raise NotImplementedError
