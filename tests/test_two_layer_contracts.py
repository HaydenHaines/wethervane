"""Tests for the two-layer architecture data contracts."""
from __future__ import annotations
import pandas as pd
import numpy as np
import pytest
from src.models.contracts import (
    LAYER1_REQUIRED_COLS,
    LAYER2_REQUIRED_COLS,
    layer1_output_path,
    layer2_output_path,
)
from src.models.type_classifier import compute_community_profiles, classify_types


@pytest.fixture
def sample_shifts():
    return pd.DataFrame({
        "county_fips": ["12001", "12003", "13001", "13003", "01001"],
        "pres_d_shift_16_20": [0.05, -0.02, 0.10, 0.08, -0.15],
        "gov_d_shift_14_18": [0.02, -0.01, 0.05, 0.04, -0.10],
    })


@pytest.fixture
def sample_assignments():
    return pd.DataFrame({
        "county_fips": ["12001", "12003", "13001", "13003", "01001"],
        "community_id": [0, 0, 1, 1, 2],
    })


def test_layer1_required_cols():
    assert "county_fips" in LAYER1_REQUIRED_COLS
    assert "community_id" in LAYER1_REQUIRED_COLS


def test_compute_community_profiles(sample_shifts, sample_assignments):
    shift_cols = ["pres_d_shift_16_20", "gov_d_shift_14_18"]
    profiles = compute_community_profiles(sample_shifts, sample_assignments, shift_cols)
    assert len(profiles) == 3  # 3 communities
    assert "community_id" in profiles.columns
    # Community 0 mean pres shift = (0.05 + (-0.02)) / 2 = 0.015
    c0 = profiles[profiles["community_id"] == 0].iloc[0]
    assert abs(c0["pres_d_shift_16_20"] - 0.015) < 1e-10


def test_classify_types_stub(sample_shifts, sample_assignments):
    shift_cols = ["pres_d_shift_16_20", "gov_d_shift_14_18"]
    profiles = compute_community_profiles(sample_shifts, sample_assignments, shift_cols)
    result = classify_types(profiles, shift_cols, j=3)
    assert "community_id" in result.columns
    assert "dominant_type_id" in result.columns
    assert all(f"type_weight_{j}" in result.columns for j in range(3))
    assert len(result) == 3


def test_layer1_output_path():
    path = layer1_output_path("my_model_v1")
    assert "my_model_v1" in path
    assert path.endswith(".parquet")
