"""YAML-driven experiment runner for electoral type discovery.

Selects features from the registry, applies weighting and holdout exclusion,
runs KMeans clustering with J selection, validates, and saves results.

Usage:
    python -m src.experiments.run_experiment --config experiments/tract_political_only.yaml
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from src.discovery.nest_types import nest_types
from src.discovery.run_type_discovery import discover_types
from src.tracts.feature_registry import REGISTRY, FeatureSpec, select_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class ExperimentConfig:
    name: str
    description: str
    geography_level: str  # "tract" or "county"
    features: dict  # parsed YAML features section
    clustering: dict  # algorithm, j_candidates, etc.
    nesting: dict  # s_candidates, method
    visualization: dict  # bubble_dissolve settings
    holdout: dict  # pairs, metric, threshold


@dataclass
class ExperimentResult:
    name: str
    best_j: int
    holdout_r: float
    coherence: float
    n_types: int
    n_super_types: int
    output_dir: Path
    timestamp: str


# ── Config loading ───────────────────────────────────────────────────────────


def load_config(path: Path) -> ExperimentConfig:
    """Load and validate experiment YAML config."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    exp = raw["experiment"]
    geo = raw["geography"]

    return ExperimentConfig(
        name=exp["name"],
        description=exp["description"],
        geography_level=geo["level"],
        features=raw["features"],
        clustering=raw["clustering"],
        nesting=raw["nesting"],
        visualization=raw["visualization"],
        holdout=raw["holdout"],
    )


# ── Feature selection helpers ────────────────────────────────────────────────


def _select_experiment_features(config: ExperimentConfig) -> list[str]:
    """Select feature names based on experiment config.

    For each category (electoral, demographic, religion):
      - If enabled, include features whose subcategory is enabled in the include map.
    """
    selected: list[str] = []

    for category_name in ("electoral", "demographic", "religion"):
        cat_config = config.features.get(category_name, {})
        if not cat_config.get("enabled", False):
            continue
        include_map = cat_config.get("include", {})
        for spec in REGISTRY:
            if spec.category != category_name:
                continue
            # Check if this subcategory is enabled
            if include_map.get(spec.subcategory, True):
                selected.append(spec.name)

    return selected


def _apply_holdout_exclusion(
    feature_names: list[str], config: ExperimentConfig
) -> list[str]:
    """Remove features whose source_year matches a holdout end year."""
    holdout_end_years = set()
    for pair in config.holdout.get("pairs", []):
        if len(pair) >= 2:
            holdout_end_years.add(pair[1])

    if not holdout_end_years:
        return feature_names

    # Build lookup: name -> source_year
    year_lookup = {s.name: s.source_year for s in REGISTRY}

    return [
        name
        for name in feature_names
        if year_lookup.get(name) not in holdout_end_years
    ]


# ── Weighting helpers ────────────────────────────────────────────────────────


def _apply_category_weights(
    df: pd.DataFrame, weights_map: dict[str, float]
) -> pd.DataFrame:
    """Multiply each column by its weight from weights_map."""
    result = df.copy()
    for col in result.columns:
        if col in weights_map:
            result[col] = result[col] * weights_map[col]
    return result


def _apply_presidential_weight(
    df: pd.DataFrame, presidential_weight: float
) -> pd.DataFrame:
    """Multiply presidential feature columns by the presidential weight."""
    result = df.copy()
    presidential_subcats = {"presidential_shifts", "presidential_lean"}
    presidential_names = {
        s.name for s in REGISTRY if s.subcategory in presidential_subcats
    }
    for col in result.columns:
        if col in presidential_names:
            result[col] = result[col] * presidential_weight
    return result


# ── Scaling ──────────────────────────────────────────────────────────────────


def _minmax_scale(df: pd.DataFrame) -> pd.DataFrame:
    """Min-max scale all columns to [0, 1]."""
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.values)
    return pd.DataFrame(scaled, columns=df.columns, index=df.index)


# ── Main experiment runner ───────────────────────────────────────────────────


def run_experiment(
    config: ExperimentConfig,
    features_path: Path | None = None,
    output_base: Path | None = None,
) -> ExperimentResult:
    """Execute a complete experiment run.

    Parameters
    ----------
    config : ExperimentConfig
        Parsed experiment configuration.
    features_path : Path, optional
        Override path to features parquet. If None, uses default based on
        geography_level.
    output_base : Path, optional
        Base directory for experiment outputs. Defaults to
        PROJECT_ROOT / "data" / "experiments".
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Load feature matrix
    if features_path is None:
        if config.geography_level == "tract":
            features_path = PROJECT_ROOT / "data" / "tracts" / "tract_features.parquet"
        else:
            features_path = (
                PROJECT_ROOT / "data" / "shifts" / "county_shifts_multiyear.parquet"
            )

    df = pd.read_parquet(features_path)

    # Identify geoid column
    geoid_col = None
    for candidate in ("geoid", "county_fips", "GEOID"):
        if candidate in df.columns:
            geoid_col = candidate
            break
    geoids = df[geoid_col].values if geoid_col else np.arange(len(df))

    # 2. Select features by config
    selected_features = _select_experiment_features(config)

    # 3. Holdout leakage prevention
    selected_features = _apply_holdout_exclusion(selected_features, config)

    # Filter to columns that actually exist in the DataFrame
    available = [f for f in selected_features if f in df.columns]
    if not available:
        raise ValueError(
            f"No selected features found in DataFrame. "
            f"Selected: {selected_features[:5]}..., "
            f"Available: {list(df.columns)[:10]}..."
        )

    feature_df = df[available].copy()

    # 4. Apply category weights
    weights_map: dict[str, float] = {}
    cat_lookup = {s.name: s.category for s in REGISTRY}
    for col in feature_df.columns:
        cat = cat_lookup.get(col)
        if cat and cat in config.features:
            weights_map[col] = config.features[cat].get("weight", 1.0)

    feature_df = _apply_category_weights(feature_df, weights_map)

    # Apply presidential weight
    pres_weight = config.clustering.get("presidential_weight", 1.0)
    if pres_weight != 1.0:
        feature_df = _apply_presidential_weight(feature_df, pres_weight)

    # 5. Min-max scale to [0, 1]
    feature_df = _minmax_scale(feature_df)

    # 6. J selection sweep
    feature_matrix = feature_df.values
    j_candidates = config.clustering["j_candidates"]
    n_init = config.clustering.get("n_init", 10)
    random_state = config.clustering.get("random_state", 42)

    # For holdout validation, we need holdout columns
    # Identify holdout shift columns from the registry
    holdout_end_years = set()
    for pair in config.holdout.get("pairs", []):
        if len(pair) >= 2:
            holdout_end_years.add(pair[1])

    # Find holdout column indices in the original df
    holdout_col_names = []
    for spec in REGISTRY:
        if spec.source_year in holdout_end_years and spec.name in df.columns:
            holdout_col_names.append(spec.name)

    # Build holdout matrix from original (unscaled) data if available
    holdout_matrix = None
    if holdout_col_names:
        available_holdout = [c for c in holdout_col_names if c in df.columns]
        if available_holdout:
            holdout_matrix = df[available_holdout].values

    j_results: list[dict] = []
    best_j = j_candidates[0]
    best_r = -999.0

    for j in j_candidates:
        km = KMeans(
            n_clusters=j, n_init=n_init, random_state=random_state
        )
        labels = km.fit_predict(feature_matrix)
        centroids = km.cluster_centers_

        # Compute holdout accuracy if we have holdout data
        mean_r = 0.0
        if holdout_matrix is not None:
            per_dim_r = []
            for dim_idx in range(holdout_matrix.shape[1]):
                actual = holdout_matrix[:, dim_idx]
                # Predict via type means
                type_means = np.zeros(j)
                for t in range(j):
                    mask = labels == t
                    if mask.sum() > 0:
                        type_means[t] = actual[mask].mean()
                predicted = type_means[labels]
                if np.std(actual) > 1e-10 and np.std(predicted) > 1e-10:
                    r, _ = pearsonr(actual, predicted)
                    per_dim_r.append(float(r))
                else:
                    per_dim_r.append(0.0)
            mean_r = float(np.mean(per_dim_r)) if per_dim_r else 0.0

        j_results.append({"j": j, "mean_holdout_r": mean_r})

        if mean_r > best_r:
            best_r = mean_r
            best_j = j

    # 7. Final KMeans with best J
    km_final = KMeans(
        n_clusters=best_j, n_init=n_init, random_state=random_state
    )
    labels_final = km_final.fit_predict(feature_matrix)
    centroids_final = km_final.cluster_centers_

    # Compute soft scores (inverse-distance, row-normalized)
    dists = np.zeros((len(feature_matrix), best_j))
    for t in range(best_j):
        dists[:, t] = np.linalg.norm(feature_matrix - centroids_final[t], axis=1)
    inv_dists = 1.0 / (dists + 1e-10)
    soft_scores = inv_dists / inv_dists.sum(axis=1, keepdims=True)

    # Compute coherence (within-type vs between-type variance on holdout)
    coherence = 0.0
    if holdout_matrix is not None:
        per_dim_ratios = []
        for dim_idx in range(holdout_matrix.shape[1]):
            values = holdout_matrix[:, dim_idx]
            type_variances = []
            type_means_list = []
            for t in range(best_j):
                mask = labels_final == t
                if mask.sum() >= 2:
                    type_variances.append(float(np.var(values[mask], ddof=0)))
                else:
                    type_variances.append(0.0)
                if mask.sum() > 0:
                    type_means_list.append(float(np.mean(values[mask])))
            within_var = float(np.mean(type_variances))
            between_var = float(np.var(type_means_list, ddof=0)) if type_means_list else 0.0
            total = within_var + between_var
            ratio = between_var / total if total > 1e-12 else 0.0
            per_dim_ratios.append(ratio)
        coherence = float(np.mean(per_dim_ratios)) if per_dim_ratios else 0.0

    # 8. Hierarchical nesting
    n_super_types = 0
    super_type_labels = None
    if config.nesting.get("enabled", False):
        s_candidates = config.nesting.get("s_candidates", [5, 6, 7, 8])
        nesting_result = nest_types(centroids_final, s_candidates=s_candidates)
        n_super_types = nesting_result.best_s
        super_type_labels = np.array(
            [nesting_result.mapping[t] for t in labels_final]
        )

    # 9. Save outputs
    if output_base is None:
        output_base = PROJECT_ROOT / "data" / "experiments"
    out_dir = output_base / f"{config.name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # config.yaml (frozen copy)
    frozen_config = {
        "experiment": {"name": config.name, "description": config.description},
        "geography": {"level": config.geography_level},
        "features": config.features,
        "clustering": config.clustering,
        "nesting": config.nesting,
        "visualization": config.visualization,
        "holdout": config.holdout,
    }
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(frozen_config, f, default_flow_style=False)

    # assignments.parquet
    assignments = pd.DataFrame({"geoid": geoids})
    for i in range(best_j):
        assignments[f"type_{i}_score"] = soft_scores[:, i]
    assignments["dominant_type"] = labels_final
    if super_type_labels is not None:
        assignments["super_type"] = super_type_labels
    assignments.to_parquet(out_dir / "assignments.parquet", index=False)

    # type_profiles.parquet (feature means per type)
    profile_data = {}
    for t in range(best_j):
        mask = labels_final == t
        profile_data[f"type_{t}"] = feature_df.loc[mask].mean().to_dict()
    profiles_df = pd.DataFrame(profile_data).T
    profiles_df.index.name = "type"
    profiles_df.to_parquet(out_dir / "type_profiles.parquet")

    # validation.json
    validation = {
        "holdout_r": best_r,
        "coherence": coherence,
        "j_selection_results": j_results,
    }
    with open(out_dir / "validation.json", "w") as f:
        json.dump(validation, f, indent=2)

    # meta.yaml
    git_commit = "unknown"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=5,
        )
        if result.returncode == 0:
            git_commit = result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass

    meta = {
        "name": config.name,
        "best_j": int(best_j),
        "holdout_r": float(best_r),
        "coherence": float(coherence),
        "n_types": int(best_j),
        "n_super_types": int(n_super_types),
        "timestamp": timestamp,
        "git_commit": git_commit,
        "n_features": len(available),
        "n_observations": len(feature_matrix),
    }
    with open(out_dir / "meta.yaml", "w") as f:
        yaml.dump(meta, f, default_flow_style=False)

    # 10. Create symlink: {name}_latest -> timestamped dir
    latest_link = output_base / f"{config.name}_latest"
    if latest_link.is_symlink() or latest_link.exists():
        latest_link.unlink()
    try:
        latest_link.symlink_to(out_dir)
    except OSError:
        pass  # Symlink creation may fail on some systems

    # 11. Return result
    return ExperimentResult(
        name=config.name,
        best_j=best_j,
        holdout_r=best_r,
        coherence=coherence,
        n_types=best_j,
        n_super_types=n_super_types,
        output_dir=out_dir,
        timestamp=timestamp,
    )


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run experiment from YAML config")
    parser.add_argument(
        "--config", required=True, help="Path to experiment YAML config"
    )
    parser.add_argument(
        "--features", default=None, help="Override features parquet path"
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))
    result = run_experiment(
        config, Path(args.features) if args.features else None
    )
    print(f"Experiment complete: {result.name}")
    print(f"  Best J: {result.best_j}")
    print(f"  Holdout r: {result.holdout_r:.4f}")
    print(f"  Coherence: {result.coherence:.4f}")
    print(f"  Output: {result.output_dir}")


if __name__ == "__main__":
    main()
