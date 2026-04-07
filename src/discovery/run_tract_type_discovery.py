"""Tract-level electoral type discovery using KMeans on the national shift matrix.

Replaces the county-level type discovery with a tract-level version covering all
51 states. Uses ~80K census tracts as the unit of analysis, with soft KMeans
membership via temperature-scaled inverse distance.

Key design decisions:
- Input: 4 presidential shifts + 22 state-centered off-cycle shifts (26 total)
- Holdout: pres_shift_20_24 excluded from training, used only for evaluation
- NaN→0 fill after dropping all-NaN presidential tracts: defensible because
  0 = state mean for centered columns, and 0 shift for presidential columns
  means "no data → assume typical behavior." This is preferable to dropping
  tracts with partial off-cycle coverage, which would bias against geographies
  with infrequent governor/senate races (e.g. non-battleground states).
- Config C (all 26 dims → 25 training) beats pres-only (r=0.686 vs 0.603).
  See data/experiments/t3_column_selection_results.json.
- PCA(n=15, whiten=True) before KMeans: whitening equalizes component scales
  so distance is not dominated by high-variance presidential components.

Usage:
    uv run python -m src.discovery.run_tract_type_discovery
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler

from src.discovery.run_type_discovery import TypeDiscoveryResult, discover_types

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Column naming conventions in tract_shifts_national.parquet
PRESIDENTIAL_PREFIX = "pres_shift_"
CENTERED_SUFFIX = "_centered"
HOLDOUT_COLUMN = "pres_shift_20_24"


def prepare_shift_matrix(
    shifts_path: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load and preprocess the tract shift matrix for type discovery.

    Loads tract_shifts_national.parquet, identifies training columns (all
    presidential + off-cycle centered shifts, excluding the 2020→2024 holdout),
    drops tracts that have no presidential data at all, and fills remaining
    NaN with 0.0.

    NaN fill rationale:
    - Centered off-cycle columns: 0 = "shifted with the state average," which
      is the most neutral assumption for a tract with no off-cycle data.
    - Presidential training columns: 0 = "no shift detected," again neutral.
    - We drop tracts where ALL presidential training columns are NaN because
      these tracts have no electoral signal at all — not just sparse off-cycle
      coverage, but truly unobserved. Keeping them would add noise rows.
    - We do NOT drop tracts with partial presidential data or missing off-cycle
      data, because that would systematically exclude non-battleground geographies.

    Parameters
    ----------
    shifts_path : str
        Path to the tract shift parquet file.

    Returns
    -------
    shift_matrix : ndarray of shape (N, D)
        N tracts × D training columns, NaN-filled with 0.0.
    tract_geoids : ndarray of shape (N,)
        Census tract GEOIDs (11-digit strings) aligned with shift_matrix rows.
    column_names : list[str]
        Names of the D shift columns in column order.
    """
    df = pd.read_parquet(shifts_path)

    # Identify training columns: presidential shifts + centered off-cycle shifts,
    # excluding the holdout column which is reserved for evaluation only.
    pres_cols = [
        c for c in df.columns
        if c.startswith(PRESIDENTIAL_PREFIX) and c != HOLDOUT_COLUMN
    ]
    offcycle_cols = [c for c in df.columns if c.endswith(CENTERED_SUFFIX)]
    training_cols = pres_cols + offcycle_cols

    if not training_cols:
        raise ValueError(
            f"No training columns found in {shifts_path}. "
            f"Expected columns starting with '{PRESIDENTIAL_PREFIX}' or ending with '{CENTERED_SUFFIX}'."
        )

    logger.info(
        "Training columns: %d presidential + %d off-cycle = %d total",
        len(pres_cols), len(offcycle_cols), len(training_cols),
    )

    # Drop tracts with ALL presidential training columns missing.
    # These tracts have zero electoral signal and would contribute only noise.
    all_pres_nan_mask = df[pres_cols].isna().all(axis=1)
    n_dropped = all_pres_nan_mask.sum()
    if n_dropped > 0:
        logger.info("Dropping %d tracts with all-NaN presidential training data", n_dropped)
        df = df[~all_pres_nan_mask].reset_index(drop=True)

    tract_geoids = df["tract_geoid"].values

    # Fill remaining NaN with 0.0 (see module docstring for rationale).
    # After the all-presidential-NaN filter, remaining NaN are:
    # - Partial presidential coverage (rare): filled with 0 = neutral shift
    # - Off-cycle columns (common, 80-99% NaN): filled with 0 = state mean
    shift_matrix = df[training_cols].fillna(0.0).values.astype(np.float64)

    logger.info(
        "Shift matrix: %d tracts × %d training columns",
        shift_matrix.shape[0], shift_matrix.shape[1],
    )

    return shift_matrix, tract_geoids, training_cols


def scale_and_weight(
    matrix: np.ndarray,
    col_names: list[str],
    presidential_weight: float = 8.0,
) -> np.ndarray:
    """StandardScale the shift matrix and upweight presidential dimensions.

    Scaling is applied first to normalize variance across dimensions —
    without this, off-cycle shifts with different natural variances dominate
    KMeans distance, obscuring the true signal structure.

    Presidential weighting is applied POST-scaling. This preserves the
    presidential:off-cycle signal ratio in KMeans distance space, where each
    standard-scaled column would otherwise contribute equally.

    Parameters
    ----------
    matrix : ndarray of shape (N, D)
        Raw shift matrix from prepare_shift_matrix().
    col_names : list[str]
        Column names corresponding to matrix columns, in order.
    presidential_weight : float
        Multiplicative factor applied to presidential shift columns after scaling.
        Default 8.0 comes from config types.presidential_weight.

    Returns
    -------
    scaled_matrix : ndarray of shape (N, D)
        Scaled (and optionally weighted) shift matrix ready for PCA + KMeans.
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)

    if presidential_weight != 1.0:
        pres_indices = [
            i for i, c in enumerate(col_names) if PRESIDENTIAL_PREFIX in c
        ]
        scaled[:, pres_indices] *= presidential_weight
        logger.info(
            "Applied presidential weight=%.1f to %d columns (post-scaling): %s",
            presidential_weight, len(pres_indices),
            [col_names[i] for i in pres_indices],
        )

    return scaled


def run_discovery(
    shift_matrix: np.ndarray,
    j: int = 100,
    temperature: float = 10.0,
    pca_components: int | None = 15,
    pca_whiten: bool = True,
    random_state: int = 42,
) -> TypeDiscoveryResult:
    """Run KMeans type discovery on the prepared tract shift matrix.

    Thin wrapper around the existing discover_types() function from
    run_type_discovery.py. All clustering logic lives there — this function
    exists to provide a clean interface and logging for the tract pipeline.

    Parameters
    ----------
    shift_matrix : ndarray of shape (N, D)
        Scaled and weighted shift matrix from scale_and_weight().
    j : int
        Number of types (centroids) to discover.
    temperature : float
        Sharpening exponent for soft membership. T=10.0 is validated production
        default — see run_type_discovery.py for derivation.
    pca_components : int or None
        PCA dimensions before KMeans. None disables PCA.
        Config C experiment used n=15 with whiten=True for r=0.686.
    pca_whiten : bool
        If True, PCA whitening is applied (equalizes component variances before
        KMeans). Validated to improve Ridge LOO r by +0.022 in county model.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    TypeDiscoveryResult
        Soft scores, centroids, dominant types, type size fractions, identity
        rotation matrix. See TypeDiscoveryResult docstring.
    """
    logger.info(
        "Running KMeans type discovery: J=%d, T=%.1f, PCA(n=%s, whiten=%s), N=%d tracts",
        j, temperature, pca_components, pca_whiten, shift_matrix.shape[0],
    )

    result = discover_types(
        shift_matrix,
        j=j,
        random_state=random_state,
        temperature=temperature,
        pca_components=pca_components,
        pca_whiten=pca_whiten,
    )

    unique_types, counts = np.unique(result.dominant_types, return_counts=True)
    logger.info(
        "Type size distribution: min=%d, max=%d, median=%.0f, mean=%.0f",
        counts.min(), counts.max(), np.median(counts), counts.mean(),
    )

    return result


def evaluate_holdout(
    shifts_path: str,
    tract_geoids: np.ndarray,
    scores: np.ndarray,
    n_types: int,
) -> float:
    """Evaluate type discovery quality using the held-out 2020→2024 presidential shift.

    The holdout column (pres_shift_20_24) was excluded from type discovery training.
    We evaluate by predicting each tract's holdout shift as a weighted average of
    per-type mean holdout shifts, where weights = soft membership scores.

    Prediction formula:
        predicted_shift[i] = sum_j(scores[i,j] * type_mean_shift[j])

    This is the same LOO-style evaluation used for the county model — it tests
    whether the discovered types generalize to an unseen election cycle.

    Parameters
    ----------
    shifts_path : str
        Path to the tract shift parquet (contains the holdout column).
    tract_geoids : ndarray of shape (N,)
        Tract GEOIDs that were used in type discovery (after all-NaN filtering).
    scores : ndarray of shape (N, J)
        Soft membership scores from discover_types().
    n_types : int
        Number of types J.

    Returns
    -------
    holdout_r : float
        Pearson r between predicted and actual holdout shifts.
        Higher is better; ~0.69 matches Config C experiment result.
    """
    df = pd.read_parquet(shifts_path)

    if HOLDOUT_COLUMN not in df.columns:
        raise ValueError(
            f"Holdout column '{HOLDOUT_COLUMN}' not found in {shifts_path}. "
            f"Available columns: {list(df.columns)}"
        )

    # Align holdout values to the tracts used in discovery (same filter was applied).
    holdout_df = df.set_index("tract_geoid")[[HOLDOUT_COLUMN]]
    holdout_aligned = holdout_df.reindex(tract_geoids)

    # Tracts with no holdout data can't be evaluated — exclude them from r calculation.
    valid_mask = ~holdout_aligned[HOLDOUT_COLUMN].isna().values
    actual = holdout_aligned[HOLDOUT_COLUMN].values[valid_mask]
    scores_valid = scores[valid_mask]  # (N_valid, J)

    if valid_mask.sum() < 100:
        logger.warning(
            "Only %d tracts have valid holdout data — holdout r may be unreliable",
            valid_mask.sum(),
        )

    # Compute per-type mean holdout shift using only the dominant-type assignment.
    # (Soft assignment here would be circular — soft scores are what we're evaluating.)
    # Instead, assign each type's "mean" from actual holdout values of its members.
    dominant_types = np.argmax(scores_valid, axis=1)
    type_mean_holdout = np.zeros(n_types)
    for t in range(n_types):
        mask_t = dominant_types == t
        if mask_t.sum() > 0:
            type_mean_holdout[t] = actual[mask_t].mean()
        # If no tracts in this type (can happen with J=100), type_mean stays 0.

    # Predict each tract's holdout shift as weighted average of type means.
    predicted = scores_valid @ type_mean_holdout  # (N_valid,) dot product

    r, p_value = pearsonr(predicted, actual)
    logger.info(
        "Holdout evaluation: r=%.4f (p=%.2e, N=%d valid tracts)",
        r, p_value, valid_mask.sum(),
    )

    return float(r)


def save_tract_types(
    tract_geoids: np.ndarray,
    result: TypeDiscoveryResult,
    n_types: int,
    output_dir: str,
) -> None:
    """Save tract type assignments and centroids to the data/communities directory.

    Saves:
    - tract_type_assignments.parquet: tract_geoid + type_N_score columns + dominant_type
    - tract_type_centroids.npy: J × D centroid matrix in PCA-reduced space

    Parameters
    ----------
    tract_geoids : ndarray of shape (N,)
        Census tract GEOIDs aligned with result rows.
    result : TypeDiscoveryResult
        Output from run_discovery().
    n_types : int
        Number of types J (used to validate array shapes).
    output_dir : str
        Directory to write output files. Created if it doesn't exist.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if result.scores.shape[1] != n_types:
        raise ValueError(
            f"scores.shape[1]={result.scores.shape[1]} != n_types={n_types}. "
            "n_types must match the J used in run_discovery()."
        )

    # Build assignments dataframe.
    out_df = pd.DataFrame({"tract_geoid": tract_geoids})
    score_cols = {f"type_{i}_score": result.scores[:, i] for i in range(n_types)}
    out_df = pd.concat([out_df, pd.DataFrame(score_cols)], axis=1)
    out_df["dominant_type"] = result.dominant_types

    assignments_path = out_dir / "tract_type_assignments.parquet"
    out_df.to_parquet(assignments_path, index=False)
    logger.info("Saved %d tract assignments to %s", len(out_df), assignments_path)

    # Save centroids (J × D in PCA-reduced space).
    centroids_path = out_dir / "tract_type_centroids.npy"
    np.save(centroids_path, result.loadings)
    logger.info(
        "Saved centroids shape=%s to %s", result.loadings.shape, centroids_path
    )

    # Summary stats.
    _, type_counts = np.unique(result.dominant_types, return_counts=True)
    logger.info(
        "Type size summary: min=%d, max=%d, median=%.0f, mean=%.0f (J=%d, N=%d)",
        type_counts.min(), type_counts.max(),
        np.median(type_counts), type_counts.mean(),
        n_types, len(tract_geoids),
    )


def main() -> None:
    """CLI entry point for tract-level type discovery.

    Reads all parameters from config/model.yaml. Runs the full pipeline:
    prepare → scale_and_weight → run_discovery → evaluate_holdout → save.

    Run as:
        uv run python -m src.discovery.run_tract_type_discovery
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    config_path = PROJECT_ROOT / "config" / "model.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    types_cfg = config.get("types", {})
    j = int(types_cfg.get("j", 100))
    temperature = float(types_cfg.get("temperature", 10.0))
    pca_components = types_cfg.get("pca_components")
    if pca_components is not None:
        pca_components = int(pca_components)
    pca_whiten = bool(types_cfg.get("pca_whiten", False))
    presidential_weight = float(types_cfg.get("presidential_weight", 8.0))

    logger.info(
        "Config: J=%d, T=%.1f, PCA(n=%s, whiten=%s), presidential_weight=%.1f",
        j, temperature, pca_components, pca_whiten, presidential_weight,
    )

    shifts_path = PROJECT_ROOT / "data" / "shifts" / "tract_shifts_national.parquet"
    output_dir = PROJECT_ROOT / "data" / "communities"

    # Step 1: Load and filter.
    shift_matrix, tract_geoids, col_names = prepare_shift_matrix(str(shifts_path))
    logger.info("Loaded %d tracts, %d training dims", len(tract_geoids), len(col_names))

    # Step 2: Scale + presidential weighting.
    weighted_matrix = scale_and_weight(shift_matrix, col_names, presidential_weight)

    # Step 3: KMeans type discovery.
    result = run_discovery(
        weighted_matrix,
        j=j,
        temperature=temperature,
        pca_components=pca_components,
        pca_whiten=pca_whiten,
    )

    # Step 4: Holdout evaluation.
    holdout_r = evaluate_holdout(str(shifts_path), tract_geoids, result.scores, j)
    logger.info("Holdout r = %.4f (county baseline = 0.698)", holdout_r)

    # Step 5: Save outputs.
    save_tract_types(tract_geoids, result, j, str(output_dir))

    print("\nTract type discovery complete:")
    print(f"  Tracts: {len(tract_geoids):,}")
    print(f"  Types: {j}")
    print(f"  Holdout r: {holdout_r:.4f}")
    _, type_counts = np.unique(result.dominant_types, return_counts=True)
    print(f"  Type sizes: min={type_counts.min()}, max={type_counts.max()}, "
          f"median={np.median(type_counts):.0f}")


if __name__ == "__main__":
    main()
