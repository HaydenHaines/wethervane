"""Prediction and interpretation.

Aggregates type-level estimates to county/district/state predictions.
Decomposes shifts into persuasion vs turnout vs composition. Generates
conditional forecasts.

This module takes the posterior over type-level opinions (from the
propagation stage) and produces actionable outputs:

    1. Geographic predictions: county, congressional district, state,
       and national vote share and turnout estimates with uncertainty.

    2. Shift decomposition: for any geography, decompose the predicted
       shift (vs. previous election) into three components:
           - Persuasion: type-level vote share change (holding turnout
             and composition constant)
           - Mobilization: type-level turnout change (holding vote share
             and composition constant)
           - Composition: population/registration changes across types

    3. Conditional forecasts: "what if" scenarios (e.g., "what if
       Black turnout matches 2012 levels?" or "what if evangelical
       persuasion shifts 2 points?").

Key references:
    - Ghitza & Gelman 2013 (aggregation from small areas)
    - Grimmer et al. 2024 (caution on ecological inference)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


def aggregate_to_county(
    type_posteriors: "np.ndarray",
    W: "np.ndarray",
    county_populations: "pd.Series",
) -> "pd.DataFrame":
    """Aggregate type-level posteriors to county-level predictions.

    county_prediction = W @ type_opinion (for each posterior sample).

    Parameters
    ----------
    type_posteriors : np.ndarray
        Type-level posterior samples (K_types x n_samples).
    W : np.ndarray
        County x type weight matrix (N_counties x K_types).
    county_populations : pd.Series
        County populations for turnout conversion.

    Returns
    -------
    pd.DataFrame
        County-level predictions: columns for fips, mean, median,
        ci_lower, ci_upper, sd, for both vote share and turnout.
    """
    raise NotImplementedError


def aggregate_to_district(
    county_predictions: "pd.DataFrame",
    county_to_district: "pd.DataFrame",
) -> "pd.DataFrame":
    """Aggregate county predictions to congressional district level.

    Uses county-to-district crosswalk (population-weighted for split
    counties).

    Parameters
    ----------
    county_predictions : pd.DataFrame
        County-level predictions with uncertainty.
    county_to_district : pd.DataFrame
        Crosswalk: fips, district_id, population_weight.

    Returns
    -------
    pd.DataFrame
        District-level predictions with propagated uncertainty.
    """
    raise NotImplementedError


def aggregate_to_state(
    county_predictions: "pd.DataFrame",
) -> "pd.DataFrame":
    """Aggregate county predictions to state level.

    Population-weighted aggregation with proper uncertainty propagation
    (accounting for within-state county correlations).

    Parameters
    ----------
    county_predictions : pd.DataFrame
        County-level predictions with posterior samples.

    Returns
    -------
    pd.DataFrame
        State-level predictions: mean, median, ci_lower, ci_upper, sd,
        win_probability.
    """
    raise NotImplementedError


def decompose_shift(
    type_posteriors_current: "np.ndarray",
    type_results_previous: "np.ndarray",
    W_current: "np.ndarray",
    W_previous: "np.ndarray",
    turnout_posteriors_current: "np.ndarray",
    turnout_previous: "np.ndarray",
    geography_weights: "np.ndarray",
) -> dict:
    """Decompose predicted shift into persuasion, mobilization, composition.

    For a given geography (county, district, state), decompose:
        shift = persuasion_effect + mobilization_effect + composition_effect

    where:
        - persuasion: change in type-level vote share, holding turnout
          and composition at previous levels
        - mobilization: change in type-level turnout, holding vote share
          and composition at previous levels
        - composition: change in type weights (population shifts),
          holding type-level vote share and turnout at previous levels

    Parameters
    ----------
    type_posteriors_current : np.ndarray
        Current type-level vote share posteriors (K_types x n_samples).
    type_results_previous : np.ndarray
        Previous election type-level vote shares (K_types,).
    W_current : np.ndarray
        Current county x type weight matrix.
    W_previous : np.ndarray
        Previous election county x type weight matrix.
    turnout_posteriors_current : np.ndarray
        Current type-level turnout posteriors (K_types x n_samples).
    turnout_previous : np.ndarray
        Previous election type-level turnout (K_types,).
    geography_weights : np.ndarray
        Weights for aggregating counties to the target geography.

    Returns
    -------
    dict
        Keys: "total_shift", "persuasion", "mobilization", "composition",
        each with mean, ci_lower, ci_upper. Also "interaction" (residual
        from non-additivity).
    """
    raise NotImplementedError


def conditional_forecast(
    type_posteriors: "np.ndarray",
    W: "np.ndarray",
    conditions: dict,
    county_populations: "pd.Series",
) -> "pd.DataFrame":
    """Generate conditional "what if" forecasts.

    Modifies the posterior for specified types according to the
    conditions and re-aggregates.

    Parameters
    ----------
    type_posteriors : np.ndarray
        Baseline type-level posteriors (K_types x n_samples).
    W : np.ndarray
        County x type weight matrix.
    conditions : dict
        Scenario specification. Keys are type indices or names, values
        are dicts with optional keys "vote_share_shift", "turnout_shift",
        "turnout_level", "vote_share_level".
        Example: {3: {"turnout_level": 0.65}} sets type 3's turnout
        to 65% across all samples.
    county_populations : pd.Series
        County populations.

    Returns
    -------
    pd.DataFrame
        Conditional predictions at county/state level, comparable to
        baseline predictions.
    """
    raise NotImplementedError


def compute_tipping_point(
    type_posteriors: "np.ndarray",
    W: "np.ndarray",
    county_populations: "pd.Series",
    target_geography: str = "state",
) -> "pd.DataFrame":
    """Identify tipping-point types for each geography.

    For each state/district, find which community types' shifts would
    most efficiently flip the outcome.

    Parameters
    ----------
    type_posteriors : np.ndarray
        Type-level posteriors.
    W : np.ndarray
        County x type weight matrix.
    county_populations : pd.Series
        County populations.
    target_geography : str
        "state" or "district".

    Returns
    -------
    pd.DataFrame
        For each geography: type, marginal_impact (votes per point
        of type shift), current_margin, shift_needed_to_flip.
    """
    raise NotImplementedError
