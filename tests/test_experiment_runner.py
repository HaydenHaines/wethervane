"""Tests for the YAML-driven experiment runner.

Uses small synthetic feature matrices (50 tracts x 10 features) — no real data needed.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.tracts.feature_registry import FeatureSpec, REGISTRY


# ── Helpers ──────────────────────────────────────────────────────────────────


def _minimal_config(
    *,
    name: str = "test_run",
    electoral_enabled: bool = True,
    demographic_enabled: bool = False,
    religion_enabled: bool = False,
    electoral_weight: float = 1.0,
    demographic_weight: float = 1.0,
    religion_weight: float = 1.0,
    presidential_weight: float = 1.0,
    j_candidates: list[int] | None = None,
    holdout_pairs: list[list[int]] | None = None,
) -> dict:
    """Build a minimal experiment YAML dict."""
    return {
        "experiment": {
            "name": name,
            "description": "test experiment",
        },
        "geography": {
            "level": "tract",
            "states": ["FL"],
        },
        "features": {
            "electoral": {
                "enabled": electoral_enabled,
                "weight": electoral_weight,
                "include": {
                    "presidential_shifts": True,
                    "presidential_lean": True,
                    "turnout_level": True,
                    "turnout_shift": True,
                    "vote_density": True,
                    "house_shifts": True,
                    "senate_shifts": True,
                    "split_ticket": True,
                    "donor_density": True,
                    "governor_shift": False,
                    "state_center_nonpresidential": True,
                },
            },
            "demographic": {
                "enabled": demographic_enabled,
                "weight": demographic_weight,
                "include": {
                    "race_ethnicity": True,
                    "white_working_class": True,
                    "foreign_born": True,
                    "income": True,
                    "education": True,
                    "housing": True,
                    "rent_burden": True,
                    "age_household": True,
                    "commute": True,
                    "military": True,
                },
            },
            "religion": {
                "enabled": religion_enabled,
                "weight": religion_weight,
                "include": {
                    "religion": True,
                },
            },
        },
        "clustering": {
            "algorithm": "kmeans",
            "j_candidates": j_candidates or [3, 5],
            "j_selection": "holdout_cv",
            "n_init": 5,
            "random_state": 42,
            "presidential_weight": presidential_weight,
            "min_tracts_per_type": 5,
        },
        "nesting": {
            "enabled": True,
            "s_candidates": [2, 3],
            "method": "ward_hac",
        },
        "visualization": {
            "bubble_dissolve": True,
            "min_polygon_area_sqkm": 0.1,
            "simplify_tolerance": 0.001,
        },
        "holdout": {
            "pairs": holdout_pairs or [[2020, 2024]],
            "metric": "pearson_r",
            "min_threshold": 0.3,
        },
    }


def _write_config(config_dict: dict, tmp_dir: Path) -> Path:
    """Write config dict to a YAML file and return the path."""
    p = tmp_dir / "config.yaml"
    with open(p, "w") as f:
        yaml.dump(config_dict, f)
    return p


def _make_synthetic_features(n: int = 50) -> pd.DataFrame:
    """Build a synthetic feature DataFrame with columns matching REGISTRY names.

    Returns a DataFrame with a 'geoid' column + all registered feature columns.
    Values are random but deterministic.
    """
    rng = np.random.default_rng(123)
    data = {"geoid": [f"1200{i:05d}" for i in range(n)]}
    for spec in REGISTRY:
        data[spec.name] = rng.standard_normal(n)
    return pd.DataFrame(data)


@pytest.fixture
def tmp_dir():
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def synthetic_features(tmp_dir):
    df = _make_synthetic_features(50)
    path = tmp_dir / "features.parquet"
    df.to_parquet(path, index=False)
    return path


# ── Test: load_config ────────────────────────────────────────────────────────


def test_load_config(tmp_dir):
    from src.experiments.run_experiment import ExperimentConfig, load_config

    cfg_dict = _minimal_config(name="my_test")
    cfg_path = _write_config(cfg_dict, tmp_dir)
    config = load_config(cfg_path)

    assert isinstance(config, ExperimentConfig)
    assert config.name == "my_test"
    assert config.description == "test experiment"
    assert config.geography_level == "tract"
    assert config.features["electoral"]["enabled"] is True
    assert config.clustering["algorithm"] == "kmeans"
    assert config.holdout["pairs"] == [[2020, 2024]]


# ── Test: feature selection (electoral only) ─────────────────────────────────


def test_feature_selection_electoral_only(tmp_dir):
    from src.experiments.run_experiment import ExperimentConfig, load_config, _select_experiment_features

    cfg_dict = _minimal_config(electoral_enabled=True, demographic_enabled=False, religion_enabled=False)
    cfg_path = _write_config(cfg_dict, tmp_dir)
    config = load_config(cfg_path)

    selected = _select_experiment_features(config)
    # All selected should be electoral
    electoral_names = {s.name for s in REGISTRY if s.category == "electoral"}
    demographic_names = {s.name for s in REGISTRY if s.category == "demographic"}
    religion_names = {s.name for s in REGISTRY if s.category == "religion"}

    selected_set = set(selected)
    assert selected_set.issubset(electoral_names)
    assert not selected_set.intersection(demographic_names)
    assert not selected_set.intersection(religion_names)
    assert len(selected) > 0


# ── Test: feature selection (nonpolitical only) ──────────────────────────────


def test_feature_selection_nonpolitical_only(tmp_dir):
    from src.experiments.run_experiment import ExperimentConfig, load_config, _select_experiment_features

    cfg_dict = _minimal_config(electoral_enabled=False, demographic_enabled=True, religion_enabled=True)
    cfg_path = _write_config(cfg_dict, tmp_dir)
    config = load_config(cfg_path)

    selected = _select_experiment_features(config)
    electoral_names = {s.name for s in REGISTRY if s.category == "electoral"}

    selected_set = set(selected)
    assert not selected_set.intersection(electoral_names)
    assert len(selected) > 0


# ── Test: holdout exclusion ──────────────────────────────────────────────────


def test_holdout_exclusion(tmp_dir):
    from src.experiments.run_experiment import ExperimentConfig, load_config, _apply_holdout_exclusion

    cfg_dict = _minimal_config(holdout_pairs=[[2020, 2024]])
    cfg_path = _write_config(cfg_dict, tmp_dir)
    config = load_config(cfg_path)

    # Start with all electoral features
    all_electoral = [s.name for s in REGISTRY if s.category == "electoral"]
    excluded = _apply_holdout_exclusion(all_electoral, config)

    # Features with source_year=2024 should be gone
    year_2024_features = {s.name for s in REGISTRY if s.source_year == 2024}
    assert not set(excluded).intersection(year_2024_features)
    # Features without source_year=2024 should still be present
    assert len(excluded) < len(all_electoral)
    assert len(excluded) > 0


# ── Test: category weighting ─────────────────────────────────────────────────


def test_category_weighting():
    from src.experiments.run_experiment import _apply_category_weights

    # Build a small feature matrix with known columns
    rng = np.random.default_rng(99)
    cols = ["pct_white_nh", "pct_black", "evangelical_share"]
    data = rng.standard_normal((10, 3))
    df = pd.DataFrame(data, columns=cols)

    weights_map = {
        "pct_white_nh": 2.0,
        "pct_black": 2.0,
        "evangelical_share": 0.5,
    }

    result = _apply_category_weights(df, weights_map)
    np.testing.assert_allclose(result["pct_white_nh"].values, data[:, 0] * 2.0)
    np.testing.assert_allclose(result["pct_black"].values, data[:, 1] * 2.0)
    np.testing.assert_allclose(result["evangelical_share"].values, data[:, 2] * 0.5)


# ── Test: presidential weighting ─────────────────────────────────────────────


def test_presidential_weighting():
    from src.experiments.run_experiment import _apply_presidential_weight

    cols = ["pres_d_shift_16_20", "pres_r_shift_16_20", "house_d_shift_16_18_sc", "pct_white_nh"]
    rng = np.random.default_rng(77)
    data = rng.standard_normal((10, 4))
    df = pd.DataFrame(data, columns=cols)

    result = _apply_presidential_weight(df, presidential_weight=2.5)

    # Presidential columns multiplied by 2.5
    np.testing.assert_allclose(result["pres_d_shift_16_20"].values, data[:, 0] * 2.5)
    np.testing.assert_allclose(result["pres_r_shift_16_20"].values, data[:, 1] * 2.5)
    # Non-presidential columns unchanged
    np.testing.assert_allclose(result["house_d_shift_16_18_sc"].values, data[:, 2])
    np.testing.assert_allclose(result["pct_white_nh"].values, data[:, 3])


# ── Test: minmax scaling ─────────────────────────────────────────────────────


def test_minmax_scaling():
    from src.experiments.run_experiment import _minmax_scale

    rng = np.random.default_rng(55)
    data = rng.standard_normal((20, 5)) * 10 + 50
    df = pd.DataFrame(data, columns=[f"f{i}" for i in range(5)])

    scaled = _minmax_scale(df)
    assert scaled.shape == df.shape
    for col in scaled.columns:
        assert scaled[col].min() >= -1e-10
        assert scaled[col].max() <= 1.0 + 1e-10


# ── Test: output dir is timestamped ──────────────────────────────────────────


def test_output_dir_timestamped(tmp_dir, synthetic_features):
    from src.experiments.run_experiment import load_config, run_experiment

    cfg_dict = _minimal_config(name="ts_test", j_candidates=[3])
    cfg_path = _write_config(cfg_dict, tmp_dir)
    config = load_config(cfg_path)

    result = run_experiment(config, features_path=synthetic_features, output_base=tmp_dir / "experiments")
    # Output dir name should contain the experiment name and a timestamp
    assert "ts_test" in result.output_dir.name
    assert result.timestamp in result.output_dir.name


# ── Test: config frozen copy ─────────────────────────────────────────────────


def test_config_frozen(tmp_dir, synthetic_features):
    from src.experiments.run_experiment import load_config, run_experiment

    cfg_dict = _minimal_config(name="frozen_test", j_candidates=[3])
    cfg_path = _write_config(cfg_dict, tmp_dir)
    config = load_config(cfg_path)

    result = run_experiment(config, features_path=synthetic_features, output_base=tmp_dir / "experiments")
    frozen_path = result.output_dir / "config.yaml"
    assert frozen_path.exists()

    with open(frozen_path) as f:
        frozen = yaml.safe_load(f)
    assert frozen["experiment"]["name"] == "frozen_test"


# ── Test: meta.yaml has required fields ──────────────────────────────────────


def test_meta_yaml(tmp_dir, synthetic_features):
    from src.experiments.run_experiment import load_config, run_experiment

    cfg_dict = _minimal_config(name="meta_test", j_candidates=[3])
    cfg_path = _write_config(cfg_dict, tmp_dir)
    config = load_config(cfg_path)

    result = run_experiment(config, features_path=synthetic_features, output_base=tmp_dir / "experiments")
    meta_path = result.output_dir / "meta.yaml"
    assert meta_path.exists()

    with open(meta_path) as f:
        meta = yaml.safe_load(f)
    assert "name" in meta
    assert "best_j" in meta
    assert "holdout_r" in meta
    assert "timestamp" in meta
    assert meta["name"] == "meta_test"
    assert meta["best_j"] == result.best_j


# ── Test: compare_runs ARI ───────────────────────────────────────────────────


def test_compare_runs_ari(tmp_dir):
    from src.experiments.compare_runs import compare_runs

    # Create two fake assignment DataFrames
    n = 50
    rng = np.random.default_rng(42)

    dir_a = tmp_dir / "run_a"
    dir_b = tmp_dir / "run_b"
    dir_a.mkdir()
    dir_b.mkdir()

    # Identical assignments → ARI = 1.0
    labels = rng.integers(0, 3, size=n)
    df_a = pd.DataFrame({"geoid": [f"g{i}" for i in range(n)], "dominant_type": labels})
    df_b = pd.DataFrame({"geoid": [f"g{i}" for i in range(n)], "dominant_type": labels})
    df_a.to_parquet(dir_a / "assignments.parquet", index=False)
    df_b.to_parquet(dir_b / "assignments.parquet", index=False)

    result = compare_runs(dir_a, dir_b)
    assert "ari" in result
    assert result["ari"] == pytest.approx(1.0)
    assert "nmi" in result


def test_compare_runs_different(tmp_dir):
    from src.experiments.compare_runs import compare_runs

    n = 50
    rng = np.random.default_rng(42)

    dir_a = tmp_dir / "run_c"
    dir_b = tmp_dir / "run_d"
    dir_a.mkdir()
    dir_b.mkdir()

    labels_a = rng.integers(0, 3, size=n)
    labels_b = rng.integers(0, 3, size=n)
    df_a = pd.DataFrame({"geoid": [f"g{i}" for i in range(n)], "dominant_type": labels_a})
    df_b = pd.DataFrame({"geoid": [f"g{i}" for i in range(n)], "dominant_type": labels_b})
    df_a.to_parquet(dir_a / "assignments.parquet", index=False)
    df_b.to_parquet(dir_b / "assignments.parquet", index=False)

    result = compare_runs(dir_a, dir_b)
    assert "ari" in result
    # Different random labels should have ARI < 1
    assert result["ari"] < 1.0
