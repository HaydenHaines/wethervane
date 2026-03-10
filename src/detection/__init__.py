"""Community type discovery from non-political data.

Implements multi-layer network detection (Leiden, SBM, Infomap) and
graph-regularized NMF. Produces soft county x type assignment matrices
and type hierarchy.

This module operates strictly on NON-POLITICAL data (demographics, religion,
commuting, migration, social connectedness). Political data (election returns)
is excluded to prevent circularity -- see DECISIONS_LOG.md, "Two-stage
separation."

Key references:
    - Traag, Waltman, van Eck 2019 (Leiden algorithm)
    - Peixoto 2014 (hierarchical SBM via graph-tool)
    - Cai et al. 2011 (graph-regularized NMF)
    - Lee & Seung 1999 (NMF foundations)
    - Bailey et al. 2018 (Social Connectedness Index)
    - Huckfeldt & Sprague 1995 (community political influence theory)

Output:
    - W matrix (N_counties x K_types): soft assignment weights, rows sum to 1
    - H matrix (K_types x P_features): type profiles / centroids
    - Hard cluster labels (from Leiden/SBM) for comparison
    - Type hierarchy / dendrogram (from hierarchical SBM)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


def build_multiplex_graph(
    edges: "pd.DataFrame",
    features: "pd.DataFrame",
    layers: list[str] | None = None,
) -> object:
    """Construct a multiplex (multi-layer) county graph.

    Combines commuting, migration, and SCI edge layers with optional
    feature-similarity edges into a single multiplex graph object.

    Parameters
    ----------
    edges : pd.DataFrame
        Edge list with columns: source_fips, target_fips, weight, layer.
    features : pd.DataFrame
        County feature matrix (used for optional feature-similarity layer).
    layers : list[str] | None
        Which layers to include. If None, use all available.

    Returns
    -------
    object
        Multiplex graph suitable for community detection (format TBD:
        igraph, graph-tool, or custom).
    """
    raise NotImplementedError


def detect_leiden(graph: object, resolution: float = 1.0) -> "np.ndarray":
    """Run Leiden community detection on the multiplex graph.

    Parameters
    ----------
    graph : object
        Multiplex county graph.
    resolution : float
        Resolution parameter (higher = more communities).

    Returns
    -------
    np.ndarray
        Hard cluster labels, shape (N_counties,).
    """
    raise NotImplementedError


def detect_sbm(graph: object, hierarchical: bool = True) -> dict:
    """Run stochastic block model via graph-tool.

    Parameters
    ----------
    graph : object
        Multiplex county graph.
    hierarchical : bool
        If True, fit nested/hierarchical SBM.

    Returns
    -------
    dict
        Keys: "labels" (hard assignments), "hierarchy" (if hierarchical),
        "description_length" (model quality metric).
    """
    raise NotImplementedError


def detect_infomap(graph: object) -> "np.ndarray":
    """Run Infomap community detection on the multiplex graph.

    Parameters
    ----------
    graph : object
        Multiplex county graph.

    Returns
    -------
    np.ndarray
        Hard cluster labels, shape (N_counties,).
    """
    raise NotImplementedError


def graph_regularized_nmf(
    features: "pd.DataFrame",
    graph: object,
    n_types: int,
    alpha: float = 0.1,
    max_iter: int = 500,
    init_labels: "np.ndarray | None" = None,
) -> tuple["np.ndarray", "np.ndarray"]:
    """Graph-regularized Non-negative Matrix Factorization.

    Decomposes the county feature matrix X into W * H, where:
        - W (N_counties x K_types): soft community membership weights
        - H (K_types x P_features): community type profiles

    The graph regularization term encourages connected counties to have
    similar membership vectors.

    min_{W,H >= 0}  ||X - WH||_F^2 + alpha * tr(W^T L W)

    where L is the graph Laplacian of the community network.

    Parameters
    ----------
    features : pd.DataFrame
        County feature matrix (N_counties x P_features), non-negative.
    graph : object
        Multiplex county graph (used to compute Laplacian).
    n_types : int
        Number of community types K.
    alpha : float
        Regularization strength for graph smoothness.
    max_iter : int
        Maximum optimization iterations.
    init_labels : np.ndarray | None
        If provided, initialize W from hard cluster labels (e.g., from
        Leiden). Aids convergence.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (W, H) where W is row-normalized to sum to 1.

    References
    ----------
    Cai et al. 2011, "Graph Regularized Nonnegative Matrix Factorization
    for Data Representation."
    """
    raise NotImplementedError


def select_n_types(
    features: "pd.DataFrame",
    graph: object,
    k_range: range = range(3, 15),
) -> dict:
    """Select the number of community types K via cross-validation.

    Evaluates reconstruction error and graph modularity across a range
    of K values.

    Parameters
    ----------
    features : pd.DataFrame
        County feature matrix.
    graph : object
        Multiplex county graph.
    k_range : range
        Range of K values to evaluate.

    Returns
    -------
    dict
        Keys: "k_values", "reconstruction_errors", "modularities",
        "recommended_k".
    """
    raise NotImplementedError


def normalize_weights(W: "np.ndarray") -> "np.ndarray":
    """Row-normalize the W matrix so each county's weights sum to 1.

    Parameters
    ----------
    W : np.ndarray
        Raw NMF weight matrix (N_counties x K_types).

    Returns
    -------
    np.ndarray
        Row-normalized W matrix.
    """
    raise NotImplementedError


def compare_hard_vs_soft(
    hard_labels: "np.ndarray",
    soft_weights: "np.ndarray",
    elections: "pd.DataFrame",
) -> dict:
    """Compare hard and soft assignments for political covariance quality.

    Tests Assumption A005: are soft assignments more accurate than hard?

    Parameters
    ----------
    hard_labels : np.ndarray
        Hard cluster labels (N_counties,).
    soft_weights : np.ndarray
        Soft assignment matrix (N_counties x K_types).
    elections : pd.DataFrame
        County election returns for evaluation.

    Returns
    -------
    dict
        Comparison metrics: residual variance, spatial autocorrelation,
        covariance structure quality under each assignment scheme.
    """
    raise NotImplementedError
