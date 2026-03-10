"""Visualization.

Community type maps, shift maps, covariance structure plots, interactive
dashboard.

Provides publication-quality static figures (matplotlib/seaborn) and
interactive exploration tools (plotly/folium) for all stages of the
model pipeline.

Map projections use Albers Equal Area for the FL+GA+AL region.
Color palettes are colorblind-safe (viridis family for sequential,
PRGn/RdBu for diverging political scales).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from pathlib import Path


def map_community_types(
    W: "np.ndarray",
    county_geometries: object,
    type_names: list[str] | None = None,
    output_path: "Path | None" = None,
) -> object:
    """Choropleth map of community type assignments.

    For soft assignments, shows either the dominant type per county
    (colored by type, with saturation proportional to dominance) or
    a small-multiple map with one panel per type showing the weight
    surface.

    Parameters
    ----------
    W : np.ndarray
        County x type weight matrix (N_counties x K_types).
    county_geometries : object
        GeoDataFrame with county boundaries and FIPS codes.
    type_names : list[str] | None
        Human-readable type names. If None, use "Type 1", "Type 2", etc.
    output_path : Path | None
        If provided, save figure to this path.

    Returns
    -------
    object
        Matplotlib figure or plotly figure object.
    """
    raise NotImplementedError


def map_predictions(
    predictions: "pd.DataFrame",
    county_geometries: object,
    metric: str = "vote_share",
    output_path: "Path | None" = None,
) -> object:
    """Choropleth map of county-level predictions.

    Parameters
    ----------
    predictions : pd.DataFrame
        County-level predictions with fips, mean, ci_lower, ci_upper.
    county_geometries : object
        GeoDataFrame with county boundaries.
    metric : str
        Which metric to map: "vote_share", "turnout", "margin",
        "uncertainty".
    output_path : Path | None
        If provided, save figure.

    Returns
    -------
    object
        Figure object.
    """
    raise NotImplementedError


def map_shift_decomposition(
    decomposition: dict,
    county_geometries: object,
    output_path: "Path | None" = None,
) -> object:
    """Three-panel map showing persuasion, mobilization, and composition shifts.

    Parameters
    ----------
    decomposition : dict
        Output of prediction.decompose_shift(), containing per-county
        persuasion, mobilization, and composition components.
    county_geometries : object
        GeoDataFrame with county boundaries.
    output_path : Path | None
        If provided, save figure.

    Returns
    -------
    object
        Figure object (3-panel layout).
    """
    raise NotImplementedError


def plot_covariance_structure(
    Sigma: "np.ndarray",
    loadings: "np.ndarray",
    type_names: list[str] | None = None,
    output_path: "Path | None" = None,
) -> object:
    """Visualize the type-level covariance / factor structure.

    Includes:
        - Heatmap of the covariance or correlation matrix
        - Factor loading bar charts
        - Biplot of first two factors with type labels

    Parameters
    ----------
    Sigma : np.ndarray
        Type-level covariance matrix (K_types x K_types).
    loadings : np.ndarray
        Factor loadings (K_types x F_factors).
    type_names : list[str] | None
        Human-readable type names.
    output_path : Path | None
        If provided, save figure.

    Returns
    -------
    object
        Figure object.
    """
    raise NotImplementedError


def plot_type_profiles(
    H: "np.ndarray",
    feature_names: list[str],
    type_names: list[str] | None = None,
    output_path: "Path | None" = None,
) -> object:
    """Radar/bar charts showing the feature profile of each community type.

    Parameters
    ----------
    H : np.ndarray
        Type profile matrix (K_types x P_features) from NMF.
    feature_names : list[str]
        Names of the features.
    type_names : list[str] | None
        Human-readable type names.
    output_path : Path | None
        If provided, save figure.

    Returns
    -------
    object
        Figure object.
    """
    raise NotImplementedError


def plot_validation_comparison(
    hindcast_results: dict,
    output_path: "Path | None" = None,
) -> object:
    """Summary figure comparing model performance to baselines.

    Includes:
        - RMSE comparison bar chart
        - Calibration plot (predicted coverage vs. nominal)
        - Residual map (where does the model do well/poorly?)
        - Scatter of predicted vs. actual

    Parameters
    ----------
    hindcast_results : dict
        Output of validation.hindcast_validation().
    output_path : Path | None
        If provided, save figure.

    Returns
    -------
    object
        Figure object (multi-panel layout).
    """
    raise NotImplementedError


def plot_temporal_stability(
    stability_results: dict,
    output_path: "Path | None" = None,
) -> object:
    """Plot covariance stability across election cycles.

    Shows subspace angles and factor loading drift over time.

    Parameters
    ----------
    stability_results : dict
        Output of covariance.test_covariance_stability().
    output_path : Path | None
        If provided, save figure.

    Returns
    -------
    object
        Figure object.
    """
    raise NotImplementedError


def interactive_dashboard(
    W: "np.ndarray",
    predictions: "pd.DataFrame",
    county_geometries: object,
    type_names: list[str] | None = None,
    port: int = 8050,
) -> None:
    """Launch an interactive Dash/Plotly dashboard.

    Features:
        - Clickable county map with type composition tooltip
        - Type selector to highlight counties by dominant type
        - Prediction slider (scenario exploration)
        - Shift decomposition toggle

    Parameters
    ----------
    W : np.ndarray
        County x type weight matrix.
    predictions : pd.DataFrame
        County-level predictions.
    county_geometries : object
        GeoDataFrame with county boundaries.
    type_names : list[str] | None
        Human-readable type names.
    port : int
        Port for the Dash server.
    """
    raise NotImplementedError
