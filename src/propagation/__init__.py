"""Poll propagation model.

Decomposes poll signals into community-type estimates using spectral
unmixing, propagates via covariance structure. Core Bayesian model in
Stan, MRP integration via R.

This is the central inference engine of the model. Given:
    - A new poll (state or sub-state level, with sample composition)
    - The community-type weight matrix W
    - The estimated type-level covariance structure Sigma
    - Prior type-level estimates (from fundamentals or previous polls)

The model:
    1. Decomposes the poll into a noisy observation of a weighted sum
       of type-level opinions (spectral unmixing, Assumption A008).
    2. Updates the posterior over ALL type-level opinions via the
       covariance structure -- a poll in Florida updates estimates for
       similar types in Georgia.
    3. Aggregates updated type estimates back to county/district/state
       predictions via the W matrix.

The covariance-mediated propagation is the key innovation: information
from a poll in one geography "flows" to other geographies through
shared community types, regularized by the historical covariance of
how those types move together.

Key references:
    - Economist 2020 model (architectural template)
    - Linzer 2013 (dynamic Bayesian forecasting)
    - Ghitza & Gelman 2013 (MRP for election forecasting)
    - Keshava & Mustard 2002 (spectral unmixing theory)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


def compute_poll_footprint(
    poll_geography: dict,
    W: "np.ndarray",
    county_populations: "pd.Series",
) -> "np.ndarray":
    """Compute a poll's community-type footprint from its geography.

    Maps a poll's geographic coverage (state, district, metro area) to
    a weight vector over community types, using the W matrix and
    population weights.

    Parameters
    ----------
    poll_geography : dict
        Poll metadata: keys include "state", "district" (optional),
        "counties" (optional list of FIPS), "type" (likely voter /
        registered voter / adult).
    W : np.ndarray
        County x type soft assignment matrix (N_counties x K_types).
    county_populations : pd.Series
        Population by county FIPS.

    Returns
    -------
    np.ndarray
        Type weight vector for this poll, shape (K_types,), sums to 1.
    """
    raise NotImplementedError


def spectral_unmix_poll(
    poll_result: float,
    poll_footprint: "np.ndarray",
    type_priors: "np.ndarray",
    poll_n: int,
) -> dict:
    """Decompose a poll result into type-level opinion estimates.

    Solves the linear mixing model (Assumption A008):
        poll_result = footprint^T @ type_opinions + noise

    This is underdetermined for a single poll, so priors and covariance
    regularize the solution.

    Parameters
    ----------
    poll_result : float
        Observed poll result (e.g., two-party Democratic share).
    poll_footprint : np.ndarray
        Type weight vector for this poll, shape (K_types,).
    type_priors : np.ndarray
        Prior mean type-level opinions, shape (K_types,).
    poll_n : int
        Poll sample size (determines observation noise).

    Returns
    -------
    dict
        Keys: "type_posteriors" (updated means), "type_uncertainties"
        (updated standard deviations), "poll_residual".
    """
    raise NotImplementedError


def propagate_via_covariance(
    type_observation: "np.ndarray",
    observation_uncertainty: float,
    observed_types: "np.ndarray",
    Sigma: "np.ndarray",
    prior_mean: "np.ndarray",
    prior_cov: "np.ndarray",
) -> tuple["np.ndarray", "np.ndarray"]:
    """Update all type estimates given an observation of some types.

    Uses the covariance structure to propagate information from
    observed types to unobserved types (conditional multivariate
    normal update).

    Parameters
    ----------
    type_observation : np.ndarray
        Observed value(s) for certain type combinations.
    observation_uncertainty : float
        Observation noise variance.
    observed_types : np.ndarray
        Weight vector indicating which types were observed (poll footprint).
    Sigma : np.ndarray
        Type-level covariance matrix (K_types x K_types).
    prior_mean : np.ndarray
        Prior mean for all types (K_types,).
    prior_cov : np.ndarray
        Prior covariance for all types (K_types x K_types).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (posterior_mean, posterior_cov) for all types.
    """
    raise NotImplementedError


def build_stan_model(
    model_name: str = "propagation_v1",
) -> object:
    """Compile the Stan propagation model.

    Parameters
    ----------
    model_name : str
        Name of the .stan file in the models/ directory.

    Returns
    -------
    object
        Compiled CmdStanModel object.
    """
    raise NotImplementedError


def fit_full_model(
    polls: "pd.DataFrame",
    W: "np.ndarray",
    Sigma: "np.ndarray",
    county_populations: "pd.Series",
    fundamentals: "pd.DataFrame | None" = None,
    n_chains: int = 4,
    n_samples: int = 2000,
) -> dict:
    """Fit the full Bayesian propagation model via Stan.

    Integrates all polls, the covariance structure, community type
    weights, and optional fundamentals priors into a single posterior
    over type-level opinions.

    Parameters
    ----------
    polls : pd.DataFrame
        Poll data: columns include result, n, state, date, pollster,
        methodology.
    W : np.ndarray
        County x type weight matrix.
    Sigma : np.ndarray
        Type-level covariance matrix (or factor decomposition).
    county_populations : pd.Series
        County populations for footprint computation.
    fundamentals : pd.DataFrame | None
        Optional fundamentals-based priors (economic indicators,
        incumbency, etc.).
    n_chains : int
        Number of MCMC chains.
    n_samples : int
        Number of posterior samples per chain.

    Returns
    -------
    dict
        Keys: "type_posteriors" (K_types x n_samples matrix),
        "county_posteriors", "state_posteriors", "diagnostics".
    """
    raise NotImplementedError


def run_mrp_integration(
    ces_data: "pd.DataFrame",
    W: "np.ndarray",
    type_posteriors: "np.ndarray",
) -> dict:
    """Integrate MRP estimates from CES survey data.

    Calls R via subprocess or rpy2 to run multilevel regression and
    poststratification on CES microdata, then reconciles MRP estimates
    with the poll-propagation model.

    Parameters
    ----------
    ces_data : pd.DataFrame
        CES survey microdata with demographics, geography, and
        political attitudes.
    W : np.ndarray
        County x type weight matrix.
    type_posteriors : np.ndarray
        Current type-level posterior estimates from poll propagation.

    Returns
    -------
    dict
        Keys: "mrp_estimates" (county-level), "reconciled_estimates"
        (combined poll-propagation + MRP), "discrepancies".
    """
    raise NotImplementedError
