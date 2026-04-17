"""Data-driven badge axis discovery via PCA on party-relative CTOV residuals.

Purpose
-------
The legacy badge catalog (badges.py) projects CTOV residuals onto 23 preset
demographic axes.  Those axes capture theoretically motivated dimensions but
may miss empirical structure actually present in candidate differentiation.

This module discovers badge axes from the data itself:

1. Compute career CTOV vectors for all candidates (J=100 types).
2. Subtract party-mean CTOV to get residual fingerprints (remove shared
   coalition signal, same as the legacy system).
3. Run PCA on the residual matrix (N_candidates × J) to find the principal
   axes of candidate differentiation.
4. For each PC axis: find which demographic columns in type_profiles correlate
   most strongly with the PC loadings — these define the badge name.
5. Name each badge from its top 2-3 demographic correlations.
6. Optionally label axes by which well-known candidates score highest on them.
7. Award discovered badges at 1σ threshold (same logic as legacy catalog).
8. Save the fitted PCA to data/sabermetrics/badge_axes.pkl for stability.

The discovered badges are *additive* — they appear alongside the legacy catalog
badges in the API response, with kind="discovered" to distinguish them.

Stability contract
------------------
The first time this module is called with no saved .pkl, it fits PCA and saves
it.  On subsequent calls, it loads the saved model — the axes are frozen and
badges are reproducible across runs.  To re-discover axes, delete the .pkl.

Badge naming heuristics
-----------------------
PC loadings (pca.components_[i]) are a J-length weight vector over the 100
community types.  We measure how each demographic feature co-varies with those
weights using Pearson correlation.  The top-correlated positive features name
the "High" end; top-correlated negative features name the "Low" end.

Example: if PC3 is high-correlated with pct_black and urban density, and
negatively correlated with evangelical_share, the badge is named
"Black Urban vs Evangelical" or shortened to "Black Urban Strength".

The naming logic deliberately prefers 2-3 features to keep labels concise.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Number of PCA components to discover.
_N_COMPONENTS = 12

# Path to the serialized PCA model (pickle).  Delete to re-discover axes.
_BADGE_AXES_PATH = PROJECT_ROOT / "data" / "sabermetrics" / "badge_axes.pkl"

# Demographic columns used for badge naming.  Must match type_profiles.parquet.
# Ordered by interpretability — race/ethnicity first, then geography, education,
# religion, income.  The naming heuristics use this ordering to break ties.
_NAMING_COLUMNS: list[str] = [
    "pct_black",
    "pct_hispanic",
    "pct_asian",
    "pct_white_nh",
    "log_pop_density",
    "pct_transit",
    "pct_bachelors_plus",
    "pct_graduate",
    "pct_management",
    "log_median_hh_income",
    "median_age",
    "pct_owner_occupied",
    "pct_wfh",
    "evangelical_share",
    "mainline_share",
    "catholic_share",
    "black_protestant_share",
    "religious_adherence_rate",
    "net_migration_rate",
    "avg_inflow_income",
    "earnings_share",
    "transfers_share",
    "investment_share",
]

# Human-readable labels for each naming column (for badge name construction).
_COLUMN_LABELS: dict[str, str] = {
    "pct_black": "Black",
    "pct_hispanic": "Hispanic",
    "pct_asian": "Asian",
    "pct_white_nh": "White",
    "log_pop_density": "Urban",
    "pct_transit": "Transit",
    "pct_bachelors_plus": "College-Ed",
    "pct_graduate": "Graduate",
    "pct_management": "Professional",
    "log_median_hh_income": "High-Income",
    "median_age": "Older",
    "pct_owner_occupied": "Homeowner",
    "pct_wfh": "Remote-Work",
    "evangelical_share": "Evangelical",
    "mainline_share": "Mainline",
    "catholic_share": "Catholic",
    "black_protestant_share": "Black Church",
    "religious_adherence_rate": "Devout",
    "net_migration_rate": "Boomtown",
    "avg_inflow_income": "Affluent-Migration",
    "earnings_share": "Wage",
    "transfers_share": "Transfer",
    "investment_share": "Investor",
}

# Opposite labels — used when a column loads negatively (anti-correlated).
_COLUMN_ANTI_LABELS: dict[str, str] = {
    "pct_black": "Non-Black",
    "pct_hispanic": "Non-Hispanic",
    "pct_asian": "Non-Asian",
    "pct_white_nh": "Minority",
    "log_pop_density": "Rural",
    "pct_transit": "Car-Dependent",
    "pct_bachelors_plus": "Non-College",
    "pct_graduate": "Non-Graduate",
    "pct_management": "Working-Class",
    "log_median_hh_income": "Lower-Income",
    "median_age": "Younger",
    "pct_owner_occupied": "Renter",
    "pct_wfh": "In-Person",
    "evangelical_share": "Non-Evangelical",
    "mainline_share": "Non-Mainline",
    "catholic_share": "Non-Catholic",
    "black_protestant_share": "Non-Black-Church",
    "religious_adherence_rate": "Secular",
    "net_migration_rate": "Stable",
    "avg_inflow_income": "Local",
    "earnings_share": "Non-Wage",
    "transfers_share": "Non-Transfer",
    "investment_share": "Non-Investor",
}

# Minimum absolute correlation to include a feature in badge name.
# Below this, the demographic signal is too weak to be meaningful.
_MIN_NAMING_CORR = 0.12

# Maximum features included in a badge name.
_MAX_NAME_FEATURES = 3

# Number of top candidates used to annotate each axis (for logging/reporting).
_N_TOP_CANDIDATES_FOR_ANNOTATION = 5

# Badge award threshold: 1 std dev above within-party mean (same as legacy catalog).
_BADGE_THRESHOLD_STD = 1.0

# Minimum party pool size for within-party thresholding.
_MIN_PARTY_POOL_SIZE = 20

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class DiscoveredAxis(NamedTuple):
    """A single PCA-discovered badge axis.

    Attributes
    ----------
    pc_index : int
        Zero-based index into pca.components_.
    name : str
        Human-readable badge name derived from demographic loadings.
    description : str
        Longer explanation of what this axis captures.
    top_demographics : list[tuple[str, float]]
        Top demographic correlations (column_name, correlation), signed.
    top_candidate_names : list[str]
        Names of candidates scoring highest on this axis (for annotation).
    explained_variance_ratio : float
        Fraction of total variance this axis explains.
    """

    pc_index: int
    name: str
    description: str
    top_demographics: list[tuple[str, float]]
    top_candidate_names: list[str]
    explained_variance_ratio: float


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_type_profiles() -> pd.DataFrame:
    """Load per-type demographic profiles, indexed by type_id and sorted."""
    path = PROJECT_ROOT / "data" / "communities" / "type_profiles.parquet"
    df = pd.read_parquet(path)
    return df.set_index("type_id").sort_index()


def _compute_residual_matrix(
    ctov_df: pd.DataFrame,
) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    """Compute the candidate × type residual matrix for PCA.

    Party-mean CTOV is subtracted so residuals capture what makes each
    candidate *different* from an average member of their party — not the
    shared coalition signal.  This is the same subtraction as in derive_badges().

    Parameters
    ----------
    ctov_df : pd.DataFrame
        Output of compute_ctov() — one row per candidate-race.

    Returns
    -------
    R : ndarray of shape (N_candidates, J)
        Residual CTOV matrix, one row per candidate.
    person_ids : list[str]
        Person IDs corresponding to rows of R.
    person_names : list[str]
        Display names corresponding to rows of R.
    person_parties : list[str]
        Party codes corresponding to rows of R.
    """
    ctov_cols = [c for c in ctov_df.columns if c.startswith("ctov_type_")]
    J = len(ctov_cols)

    # Compute party-mean CTOV vectors.
    party_mean_ctov: dict[str, np.ndarray] = {}
    for p in ctov_df["party"].unique():
        rows = ctov_df[ctov_df["party"] == p]
        party_mean_ctov[p] = rows[ctov_cols].values.mean(axis=0)

    rows_out: list[np.ndarray] = []
    person_ids: list[str] = []
    person_names: list[str] = []
    person_parties: list[str] = []

    for pid, group in ctov_df.groupby("person_id"):
        party = group["party"].iloc[0]
        career_vec = group[ctov_cols].values.mean(axis=0)
        pm = party_mean_ctov.get(party, np.zeros(J))
        rows_out.append(career_vec - pm)
        person_ids.append(str(pid))
        person_names.append(str(group["name"].iloc[0]))
        person_parties.append(party)

    return np.array(rows_out), person_ids, person_names, person_parties


# ---------------------------------------------------------------------------
# Axis naming
# ---------------------------------------------------------------------------


def _name_axis(
    pc_weights: np.ndarray,
    type_profiles: pd.DataFrame,
    available_cols: list[str],
) -> tuple[str, str, list[tuple[str, float]]]:
    """Derive a human-readable name for a PCA axis from demographic correlations.

    The PC weights (length J) tell us how much each community type contributes
    to this axis.  We then correlate those weights with each demographic feature
    across the J types to find which demographics drive the axis.

    For example, if PC weights are high wherever pct_black is high, then this
    axis represents "Black community appeal."

    Parameters
    ----------
    pc_weights : ndarray of shape (J,)
        The PCA loading vector for this component.
    type_profiles : pd.DataFrame
        Demographics per type (J rows), indexed by type_id.
    available_cols : list[str]
        Demographic columns actually present in type_profiles.

    Returns
    -------
    name : str
        Badge display name.
    description : str
        Longer description.
    top_demographics : list[tuple[str, float]]
        (column, signed_correlation) pairs for the naming features.
    """
    # Compute Pearson correlation between PC weights and each demographic feature.
    # We center type_profiles features because we care about relative variation,
    # not absolute levels (same centering as the legacy badge score computation).
    correlations: dict[str, float] = {}
    for col in available_cols:
        feat = type_profiles[col].values.astype(float)
        feat_std = float(np.std(feat))
        if feat_std < 1e-9:
            continue  # Degenerate feature — skip
        corr = float(np.corrcoef(pc_weights, feat)[0, 1])
        if not np.isnan(corr):
            correlations[col] = corr

    # Sort by absolute correlation, descending.
    sorted_corrs = sorted(correlations.items(), key=lambda x: -abs(x[1]))

    # Collect features above the minimum correlation threshold.
    named_features = [(col, corr) for col, corr in sorted_corrs if abs(corr) >= _MIN_NAMING_CORR]
    named_features = named_features[:_MAX_NAME_FEATURES]

    if not named_features:
        # No strong demographic signal — name by variance explained.
        return "Residual Axis", "A candidate differentiation axis with no dominant demographic signal.", []

    # Build the name from the top features.
    # Positive correlation: feature increases with PC score (strength end).
    # Negative correlation: feature decreases (the axis is "against" that demographic).
    parts: list[str] = []
    for col, corr in named_features:
        if corr > 0:
            label = _COLUMN_LABELS.get(col, col)
        else:
            # Negative correlation: use the anti-label to describe the positive end
            # of the badge — e.g. "Non-Black" means the candidate does *well in*
            # types with low pct_black (the positive side of this axis).
            label = _COLUMN_ANTI_LABELS.get(col, f"Low-{col}")
        parts.append(label)

    name = " · ".join(parts)

    # Build a description that clarifies both ends of the axis.
    top_col, top_corr = named_features[0]
    if top_corr > 0:
        pos_label = _COLUMN_LABELS.get(top_col, top_col)
        neg_label = _COLUMN_ANTI_LABELS.get(top_col, f"Non-{top_col}")
    else:
        pos_label = _COLUMN_ANTI_LABELS.get(top_col, f"Non-{top_col}")
        neg_label = _COLUMN_LABELS.get(top_col, top_col)

    description = (
        f"Differentiates candidates by {pos_label.lower()} community performance. "
        f"High score = stronger in {pos_label.lower()} communities; "
        f"low score = stronger in {neg_label.lower()} communities."
    )

    return name, description, named_features


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------


def discover_badge_axes(
    n_components: int = _N_COMPONENTS,
    ctov_df: pd.DataFrame | None = None,
    force_refit: bool = False,
) -> tuple[PCA, list[DiscoveredAxis]]:
    """Discover badge axes from CTOV data via PCA.

    This is the primary entry point for badge axis discovery.  It:
    1. Loads the CTOV parquet (or accepts an already-loaded DataFrame).
    2. Builds the residual matrix (party-mean subtracted).
    3. Loads a saved PCA model if one exists (for stability), or fits a new one.
    4. Names each axis from demographic correlations.
    5. Saves the fitted model to _BADGE_AXES_PATH.

    Parameters
    ----------
    n_components : int
        Number of PCA components to discover.  Default 12.
    ctov_df : pd.DataFrame | None
        Pre-loaded CTOV DataFrame.  If None, loads from disk.
    force_refit : bool
        If True, re-fit PCA even if a saved model exists.  Useful when the
        candidate pool or CTOV data has changed substantially.

    Returns
    -------
    pca : sklearn.decomposition.PCA
        The fitted PCA model (either loaded from disk or freshly fitted).
    axes : list[DiscoveredAxis]
        Named axis descriptors, one per PC component.
    """
    # Load CTOV data if not provided.
    if ctov_df is None:
        ctov_path = PROJECT_ROOT / "data" / "sabermetrics" / "candidate_ctov.parquet"
        if not ctov_path.exists():
            raise FileNotFoundError(
                f"CTOV data not found at {ctov_path}. "
                "Run the sabermetrics pipeline first: "
                "uv run python -m src.sabermetrics.pipeline"
            )
        ctov_df = pd.read_parquet(ctov_path)
        log.info("Loaded CTOV data: %d rows", len(ctov_df))

    type_profiles = _load_type_profiles()
    available_naming_cols = [c for c in _NAMING_COLUMNS if c in type_profiles.columns]

    if not available_naming_cols:
        raise ValueError(
            "No naming columns found in type_profiles.parquet. "
            f"Expected at least some of: {_NAMING_COLUMNS[:5]}..."
        )

    log.info(
        "discover_badge_axes: %d naming columns available out of %d requested",
        len(available_naming_cols),
        len(_NAMING_COLUMNS),
    )

    # Build the residual matrix.
    R, person_ids, person_names, person_parties = _compute_residual_matrix(ctov_df)
    log.info(
        "Residual matrix: %d candidates × %d types",
        R.shape[0],
        R.shape[1],
    )

    # Load or fit the PCA model.
    pca = _load_or_fit_pca(R, n_components=n_components, force_refit=force_refit)

    # Project all candidates onto the discovered axes.
    scores = pca.transform(R)  # (N_candidates, n_components)

    # Build DiscoveredAxis descriptors.
    axes: list[DiscoveredAxis] = []
    for pc_idx in range(pca.n_components_):
        pc_weights = pca.components_[pc_idx]  # (J,)

        name, description, top_demographics = _name_axis(
            pc_weights, type_profiles, available_naming_cols
        )

        # Annotate with top-scoring candidates on this axis (informational only).
        pc_scores = scores[:, pc_idx]
        top_candidate_idxs = np.argsort(-pc_scores)[:_N_TOP_CANDIDATES_FOR_ANNOTATION]
        top_candidate_names = [
            f"{person_names[i]} ({person_parties[i]}, {pc_scores[i]:+.3f})"
            for i in top_candidate_idxs
        ]

        axis = DiscoveredAxis(
            pc_index=pc_idx,
            name=name,
            description=description,
            top_demographics=top_demographics,
            top_candidate_names=top_candidate_names,
            explained_variance_ratio=float(pca.explained_variance_ratio_[pc_idx]),
        )
        axes.append(axis)

        log.info(
            "PC%d (var=%.3f): '%s' | top demographics: %s | top candidates: %s",
            pc_idx + 1,
            axis.explained_variance_ratio,
            name,
            [(col, f"{corr:+.3f}") for col, corr in top_demographics[:2]],
            [person_names[i] for i in top_candidate_idxs[:2]],
        )

    return pca, axes


def _load_or_fit_pca(
    R: np.ndarray,
    n_components: int,
    force_refit: bool,
) -> PCA:
    """Load a saved PCA model or fit a new one.

    Stable axes require a frozen PCA fit.  We save the model as a pickle
    after first fitting and reload it on subsequent calls.  This ensures
    badge assignments don't shift when new candidates are added to the pool.

    The model is keyed by (N_candidates, J, n_components) — if any changes,
    we re-fit automatically even without force_refit=True.  This catches the
    common case where new election data changes the candidate pool.

    Parameters
    ----------
    R : ndarray of shape (N, J)
        Residual CTOV matrix.
    n_components : int
        Number of PCA components.
    force_refit : bool
        If True, always re-fit and overwrite the saved model.

    Returns
    -------
    PCA
        The fitted PCA model.
    """
    _BADGE_AXES_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not force_refit and _BADGE_AXES_PATH.exists():
        try:
            with _BADGE_AXES_PATH.open("rb") as f:
                saved = pickle.load(f)

            saved_pca: PCA = saved["pca"]
            saved_shape: tuple[int, int] = saved["data_shape"]

            if (
                saved_pca.n_components_ == n_components
                and saved_shape == (R.shape[0], R.shape[1])
            ):
                log.info(
                    "Loaded saved PCA model from %s (N=%d, J=%d, n_components=%d)",
                    _BADGE_AXES_PATH,
                    *saved_shape,
                    n_components,
                )
                return saved_pca
            else:
                log.info(
                    "Saved PCA shape %s or n_components %d doesn't match current data "
                    "(%d×%d, n_components=%d) — re-fitting",
                    saved_shape,
                    saved_pca.n_components_,
                    R.shape[0],
                    R.shape[1],
                    n_components,
                )
        except Exception as exc:
            log.warning("Could not load saved PCA model: %s — re-fitting", exc)

    # Fit a new PCA model.
    # random_state is pinned for reproducibility.
    # svd_solver='full' is used instead of 'auto' to ensure deterministic results
    # — 'auto' may switch to randomized SVD for large matrices, which is stochastic.
    pca = PCA(n_components=n_components, random_state=42, svd_solver="full")
    pca.fit(R)

    log.info(
        "Fitted PCA: n_components=%d, cumulative_variance=%.3f",
        n_components,
        float(pca.explained_variance_ratio_.sum()),
    )

    # Save the model with metadata.
    with _BADGE_AXES_PATH.open("wb") as f:
        pickle.dump({"pca": pca, "data_shape": (R.shape[0], R.shape[1])}, f)

    log.info("Saved PCA model to %s", _BADGE_AXES_PATH)
    return pca


# ---------------------------------------------------------------------------
# Badge award computation
# ---------------------------------------------------------------------------


def derive_discovered_badges(
    ctov_df: pd.DataFrame,
    pca: PCA | None = None,
    axes: list[DiscoveredAxis] | None = None,
) -> dict[str, dict]:
    """Compute discovered badge awards for all candidates.

    For each discovered PCA axis, compute each candidate's score by projecting
    their residual CTOV onto the axis.  Award a badge when the score exceeds
    1σ above the within-party mean (same threshold as the legacy catalog).

    Parameters
    ----------
    ctov_df : pd.DataFrame
        Output of compute_ctov().
    pca : PCA | None
        Fitted PCA model.  If None, calls discover_badge_axes() to get one.
    axes : list[DiscoveredAxis] | None
        Axis descriptors from discover_badge_axes().  Must be provided if pca
        is provided.

    Returns
    -------
    dict
        Mapping: person_id → {
            "discovered_badge_details": list[dict],
        }

        Each detail dict has keys: name, score, kind, pc_index, top_demographics,
        provisional, fallback_reason.
    """
    if pca is None or axes is None:
        pca, axes = discover_badge_axes(ctov_df=ctov_df)

    # Build residual matrix (same as for PCA fitting).
    R, person_ids, person_names, person_parties = _compute_residual_matrix(ctov_df)

    # Project onto discovered axes.
    scores_matrix = pca.transform(R)  # (N_candidates, n_components)

    # Build a mapping from person_id → row index for fast lookup.
    person_row_idx: dict[str, int] = {pid: i for i, pid in enumerate(person_ids)}

    # Count races per person for provisional detection.
    n_races_by_person: dict[str, int] = (
        ctov_df.groupby("person_id").size().to_dict()
    )

    # ---------------------------------------------------------------------------
    # Compute within-party mean and std for each discovered axis.
    # Fall back to global stats for small party pools (< _MIN_PARTY_POOL_SIZE).
    # ---------------------------------------------------------------------------
    all_parties = list(set(person_parties))

    # Global stats (fallback for small parties and final threshold computation).
    global_mean = scores_matrix.mean(axis=0)  # (n_components,)
    global_std = scores_matrix.std(axis=0).clip(min=1e-9)  # (n_components,)

    party_row_idxs: dict[str, list[int]] = {}
    for p in all_parties:
        idxs = [i for i, party in enumerate(person_parties) if party == p]
        party_row_idxs[p] = idxs

    party_mean: dict[str, np.ndarray] = {}
    party_std: dict[str, np.ndarray] = {}
    small_parties: set[str] = set()

    for p, idxs in party_row_idxs.items():
        if len(idxs) < _MIN_PARTY_POOL_SIZE:
            small_parties.add(p)
        else:
            party_scores = scores_matrix[idxs, :]
            party_mean[p] = party_scores.mean(axis=0)
            party_std[p] = party_scores.std(axis=0).clip(min=1e-9)

    # ---------------------------------------------------------------------------
    # Award badges.
    # ---------------------------------------------------------------------------
    result: dict[str, dict] = {}

    for pid in person_ids:
        row_idx = person_row_idx[pid]
        p = person_parties[row_idx]
        scores_vec = scores_matrix[row_idx, :]  # (n_components,)
        provisional = n_races_by_person.get(pid, 1) < 2
        is_small_pool = p in small_parties

        details: list[dict] = []

        for axis in axes:
            pc_idx = axis.pc_index
            score = float(scores_vec[pc_idx])

            # Choose within-party or global threshold.
            if is_small_pool or p not in party_mean:
                mean = float(global_mean[pc_idx])
                std = float(global_std[pc_idx])
                fallback_reason: str | None = "small_pool" if is_small_pool else None
            else:
                mean = float(party_mean[p][pc_idx])
                std = float(party_std[p][pc_idx])
                fallback_reason = None

            if std < 1e-9:
                continue

            if score > mean + _BADGE_THRESHOLD_STD * std:
                details.append(
                    {
                        "name": f"PCA: {axis.name}",
                        "score": score,
                        "kind": "discovered",
                        "pc_index": pc_idx,
                        "top_demographics": axis.top_demographics,
                        "provisional": provisional,
                        "fallback_reason": fallback_reason,
                        "explained_variance_ratio": axis.explained_variance_ratio,
                    }
                )
            elif score < mean - _BADGE_THRESHOLD_STD * std:
                details.append(
                    {
                        "name": f"PCA: Low {axis.name}",
                        "score": score,
                        "kind": "discovered",
                        "pc_index": pc_idx,
                        "top_demographics": axis.top_demographics,
                        "provisional": provisional,
                        "fallback_reason": fallback_reason,
                        "explained_variance_ratio": axis.explained_variance_ratio,
                    }
                )

        result[pid] = {"discovered_badge_details": details}

    n_with_discovered = sum(1 for v in result.values() if v["discovered_badge_details"])
    log.info(
        "derive_discovered_badges: %d candidates, %d with at least one discovered badge",
        len(result),
        n_with_discovered,
    )
    return result


# ---------------------------------------------------------------------------
# Candidate fingerprint summary
# ---------------------------------------------------------------------------


def compute_candidate_fingerprint(
    person_id: str,
    ctov_df: pd.DataFrame,
    pca: PCA | None = None,
    axes: list[DiscoveredAxis] | None = None,
) -> dict:
    """Compute a single candidate's full fingerprint across all discovered axes.

    Returns their score on each axis and their relative rank within their party,
    making it easy to see how unique a candidate's profile is.

    Parameters
    ----------
    person_id : str
        Bioguide / registry person ID.
    ctov_df : pd.DataFrame
        Full CTOV DataFrame (all candidates).
    pca : PCA | None
        Fitted PCA model.
    axes : list[DiscoveredAxis] | None
        Axis descriptors.

    Returns
    -------
    dict with keys:
        person_id, name, party,
        axis_scores: list[{axis_name, pc_index, score, party_percentile}]
    """
    if pca is None or axes is None:
        pca, axes = discover_badge_axes(ctov_df=ctov_df)

    R, person_ids, person_names, person_parties = _compute_residual_matrix(ctov_df)
    scores_matrix = pca.transform(R)
    person_row_idx = {pid: i for i, pid in enumerate(person_ids)}

    if person_id not in person_row_idx:
        raise KeyError(f"Person ID '{person_id}' not found in CTOV data")

    row_idx = person_row_idx[person_id]
    p = person_parties[row_idx]
    name = person_names[row_idx]
    scores_vec = scores_matrix[row_idx, :]

    # Compute within-party percentiles for this candidate.
    party_idxs = [i for i, party in enumerate(person_parties) if party == p]
    party_scores = scores_matrix[party_idxs, :]  # (n_party, n_components)

    axis_scores: list[dict] = []
    for axis in axes:
        pc_idx = axis.pc_index
        score = float(scores_vec[pc_idx])
        party_col = party_scores[:, pc_idx]
        # Percentile within party (0–100): what fraction of party peers score lower.
        percentile = float(np.mean(party_col < score) * 100)
        axis_scores.append(
            {
                "axis_name": axis.name,
                "pc_index": pc_idx,
                "score": score,
                "party_percentile": round(percentile, 1),
                "explained_variance_ratio": axis.explained_variance_ratio,
                "top_demographics": axis.top_demographics,
            }
        )

    return {
        "person_id": person_id,
        "name": name,
        "party": p,
        "axis_scores": axis_scores,
    }
