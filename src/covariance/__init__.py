"""Historical covariance estimation for community types.

PCA and Bayesian factor models on type-level election results. Produces
structured covariance matrices for vote share and turnout jointly.

This module takes the soft assignment matrix W from the detection stage
and historical election returns, infers type-level election results, and
estimates the covariance structure of type-level swings across elections.

The key challenge is that type-level results are not directly observed --
they must be inferred from county-level results via the mixing model:

    y_county = W @ y_types + noise

where W is the county x type weight matrix. This is a linear inverse
problem, potentially ill-conditioned for types that rarely appear in
isolation.

The covariance matrix is factor-structured (ADR decision: factor-structured
covariance) because K_types >> T_elections makes a full covariance matrix
unidentifiable. With F factors (Assumption A006 suggests F = 3-5):

    Sigma_types = Lambda @ Lambda^T + Psi

where Lambda is K_types x F and Psi is diagonal.

The dual-output model estimates covariance jointly over vote share AND
turnout, enabling persuasion/mobilization/composition decomposition.

Key references:
    - Economist 2020 model (state correlation structure)
    - Gelman et al. 2016 (voter stability priors)
    - Geweke & Zhou 1996 (Bayesian factor models)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


def infer_type_level_results(
    elections: "pd.DataFrame",
    W: "np.ndarray",
    method: str = "ols",
) -> "pd.DataFrame":
    """Infer community-type-level election results from county results.

    Solves y_county = W @ y_types for y_types, given observed county
    results y_county and the mixing matrix W.

    Parameters
    ----------
    elections : pd.DataFrame
        County-level election results (N_counties x T_elections).
        Includes both vote share and turnout columns.
    W : np.ndarray
        Soft assignment matrix (N_counties x K_types), rows sum to 1.
    method : str
        Inference method: "ols" (ordinary least squares), "ridge"
        (ridge regression), "bayesian" (full posterior via Stan).

    Returns
    -------
    pd.DataFrame
        Inferred type-level results (K_types x T_elections) with
        uncertainty estimates if method is "bayesian".
    """
    raise NotImplementedError


def estimate_swing_vectors(type_results: "pd.DataFrame") -> "pd.DataFrame":
    """Compute election-to-election swing vectors at the type level.

    Parameters
    ----------
    type_results : pd.DataFrame
        Type-level election results (K_types x T_elections).

    Returns
    -------
    pd.DataFrame
        Swing vectors (K_types x (T_elections - 1)), with columns for
        both vote-share swing and turnout swing.
    """
    raise NotImplementedError


def pca_covariance(
    swings: "pd.DataFrame",
    n_factors: int | None = None,
) -> dict:
    """Estimate covariance structure via PCA on type-level swings.

    Parameters
    ----------
    swings : pd.DataFrame
        Type-level swing vectors (K_types x T_swings), joint
        vote-share and turnout.
    n_factors : int | None
        Number of factors to retain. If None, select via scree plot /
        variance explained threshold.

    Returns
    -------
    dict
        Keys: "loadings" (K_types x F), "explained_variance",
        "scree_values", "n_factors", "covariance_matrix".
    """
    raise NotImplementedError


def bayesian_factor_model(
    swings: "pd.DataFrame",
    n_factors: int,
    prior_stability: float = 0.8,
) -> dict:
    """Estimate factor-structured covariance via Bayesian factor model.

    Uses Stan to fit:
        swing_t ~ MVN(0, Lambda @ Lambda^T + Psi)

    with priors encoding voter stability (Assumption A001).

    Parameters
    ----------
    swings : pd.DataFrame
        Type-level swing vectors.
    n_factors : int
        Number of latent factors F.
    prior_stability : float
        Prior belief in voter stability (0 = no prior, 1 = very strong
        prior that swings are small). Controls the scale of the prior
        on Lambda.

    Returns
    -------
    dict
        Keys: "lambda_posterior" (samples of factor loadings),
        "psi_posterior" (samples of idiosyncratic variance),
        "covariance_posterior" (samples of full covariance matrix),
        "diagnostics" (Rhat, ESS, divergences).
    """
    raise NotImplementedError


def select_n_factors(
    swings: "pd.DataFrame",
    max_factors: int = 8,
) -> dict:
    """Select the number of latent factors via model comparison.

    Tests Assumption A006 (3-5 factors expected).

    Parameters
    ----------
    swings : pd.DataFrame
        Type-level swing vectors.
    max_factors : int
        Maximum number of factors to evaluate.

    Returns
    -------
    dict
        Keys: "n_factors_evaluated", "waic_values", "loo_values",
        "recommended_n_factors", "variance_explained".
    """
    raise NotImplementedError


def test_covariance_stability(
    elections: "pd.DataFrame",
    W: "np.ndarray",
    window_size: int = 3,
) -> dict:
    """Test whether type-level covariance is stable across election cycles.

    Tests Assumption A002 by computing rolling-window covariance
    estimates and measuring subspace similarity.

    Parameters
    ----------
    elections : pd.DataFrame
        Full county-level election history.
    W : np.ndarray
        Soft assignment matrix.
    window_size : int
        Number of elections per window.

    Returns
    -------
    dict
        Keys: "window_covariances", "subspace_angles",
        "stability_verdict" (stable / drifting / unstable).
    """
    raise NotImplementedError


def test_joint_vs_independent(
    swings: "pd.DataFrame",
    n_factors: int,
) -> dict:
    """Compare joint (vote share + turnout) vs independent covariance models.

    Tests Assumption A007.

    Parameters
    ----------
    swings : pd.DataFrame
        Type-level swing vectors with both vote-share and turnout columns.
    n_factors : int
        Number of factors for the joint model.

    Returns
    -------
    dict
        Keys: "joint_waic", "independent_waic", "joint_loo",
        "independent_loo", "recommendation".
    """
    raise NotImplementedError
