"""Tests for the voter behavior layer public API.

Covers:
  - compute_tau with synthetic 2-type / 3-tract data
  - compute_delta with synthetic data
  - τ values fall in the reasonable real-world range (0.3, 1.0)
  - δ values fall in the reasonable real-world range (-0.15, 0.15)
  - compute_and_save produces the correct output files
  - adjust_priors_for_cycle adjusts priors correctly and leaves presidential untouched

For integration tests against real pipeline data see test_voter_behavior.py.
These tests use only small synthetic DataFrames and temporary directories.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.behavior.compute_behavior_params import compute_tau, compute_delta, compute_and_save
from src.behavior.apply_behavior import adjust_priors_for_cycle


# ---------------------------------------------------------------------------
# Shared synthetic data helpers
# ---------------------------------------------------------------------------

def _make_elections_df(
    tracts: list[str],
    pres_turnout: list[int],
    off_turnout: list[int],
    pres_dem_shares: list[float],
    off_dem_shares: list[float],
    pres_year: int = 2020,
    off_year: int = 2022,
) -> pd.DataFrame:
    """Build a minimal tract_elections DataFrame in T.1 format.

    Each tract gets one presidential row and one governor row.

    Args:
        tracts: List of tract GEOIDs.
        pres_turnout: Total votes in the presidential election per tract.
        off_turnout: Total votes in the off-cycle election per tract.
        pres_dem_shares: Dem share in the presidential election per tract.
        off_dem_shares: Dem share in the off-cycle election per tract.
        pres_year: Year label for presidential rows.
        off_year: Year label for governor rows.

    Returns:
        DataFrame with columns: tract_geoid, year, race_type, total_votes,
        dem_votes, rep_votes, dem_share.
    """
    n = len(tracts)
    records = []
    for i, tract in enumerate(tracts):
        p_votes = pres_turnout[i]
        p_dem = pres_dem_shares[i]
        records.append({
            "tract_geoid": tract,
            "year": pres_year,
            "race_type": "PRES",
            "total_votes": p_votes,
            "dem_votes": int(p_votes * p_dem),
            "rep_votes": int(p_votes * (1 - p_dem)),
            "dem_share": p_dem,
        })
        o_votes = off_turnout[i]
        o_dem = off_dem_shares[i]
        records.append({
            "tract_geoid": tract,
            "year": off_year,
            "race_type": "GOV",
            "total_votes": o_votes,
            "dem_votes": int(o_votes * o_dem),
            "rep_votes": int(o_votes * (1 - o_dem)),
            "dem_share": o_dem,
        })
    return pd.DataFrame(records)


def _make_assignments_df(
    tracts: list[str],
    type_0_scores: list[float],
    type_1_scores: list[float],
) -> pd.DataFrame:
    """Build a minimal tract_type_assignments DataFrame in T.3 format.

    Args:
        tracts: List of tract GEOIDs.
        type_0_scores: Membership scores for type 0 per tract.
        type_1_scores: Membership scores for type 1 per tract.

    Returns:
        DataFrame with columns: tract_geoid, type_0_score, type_1_score.
    """
    return pd.DataFrame({
        "tract_geoid": tracts,
        "type_0_score": type_0_scores,
        "type_1_score": type_1_scores,
    })


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_data():
    """3-tract / 2-type synthetic dataset for τ and δ testing.

    Tract layout (designed to produce clear, checkable answers):
      T1: mostly type 0 (score 0.9 / 0.1). Presidential 1000 votes, off-cycle 700 → τ=0.70
      T2: mixed     (score 0.5 / 0.5). Presidential 800 votes,  off-cycle 600 → τ=0.75
      T3: mostly type 1 (score 0.1 / 0.9). Presidential 600 votes,  off-cycle 480 → τ=0.80

    Type 0 tracts go *more Dem* in off-cycle (T1, T2: +0.05).
    Type 1 tracts go *more Rep* in off-cycle (T3: -0.06).
    """
    tracts = ["T1", "T2", "T3"]
    elections = _make_elections_df(
        tracts=tracts,
        pres_turnout=[1000, 800, 600],
        off_turnout=[700, 600, 480],
        pres_dem_shares=[0.45, 0.50, 0.55],
        off_dem_shares=[0.50, 0.55, 0.49],  # T1+T2 up 0.05; T3 down 0.06
    )
    assignments = _make_assignments_df(
        tracts=tracts,
        type_0_scores=[0.9, 0.5, 0.1],
        type_1_scores=[0.1, 0.5, 0.9],
    )
    return elections, assignments


# ---------------------------------------------------------------------------
# Test 1: τ computation with synthetic data
# ---------------------------------------------------------------------------

def test_compute_tau_shape(synthetic_data):
    """compute_tau must return exactly one τ per type."""
    elections, assignments = synthetic_data
    tau = compute_tau(elections, assignments)
    assert tau.shape == (2,), f"Expected shape (2,), got {tau.shape}"


def test_compute_tau_values_reasonable(synthetic_data):
    """τ values for a realistic synthetic dataset must fall in (0.3, 1.0)."""
    elections, assignments = synthetic_data
    tau = compute_tau(elections, assignments)
    # All tracts have off-cycle < presidential turnout, so τ < 1 for both types.
    assert (tau > 0.3).all(), f"τ too low: {tau}"
    assert (tau < 1.0).all(), f"τ >= 1.0: {tau}"


def test_compute_tau_type_ordering(synthetic_data):
    """Type 0 tracts have lower τ (700/1000=0.70) than type 1 (480/600=0.80)."""
    elections, assignments = synthetic_data
    tau = compute_tau(elections, assignments)
    # Type 0 is dominated by T1 (τ=0.70); type 1 is dominated by T3 (τ=0.80).
    # The membership weighting should preserve this ordering.
    assert tau[0] < tau[1], f"Expected τ[0] < τ[1], got τ={tau}"


# ---------------------------------------------------------------------------
# Test 2: δ computation with synthetic data
# ---------------------------------------------------------------------------

def test_compute_delta_shape(synthetic_data):
    """compute_delta must return exactly one δ per type."""
    elections, assignments = synthetic_data
    tau = compute_tau(elections, assignments)
    delta = compute_delta(elections, assignments, tau)
    assert delta.shape == (2,), f"Expected shape (2,), got {delta.shape}"


def test_compute_delta_direction(synthetic_data):
    """Type 0 (+0.05 off-cycle shift) should have δ > 0; type 1 (-0.06) should have δ < 0."""
    elections, assignments = synthetic_data
    tau = compute_tau(elections, assignments)
    delta = compute_delta(elections, assignments, tau)
    assert delta[0] > 0, f"Expected δ[0] > 0 (type 0 went more Dem), got {delta[0]:.4f}"
    assert delta[1] < 0, f"Expected δ[1] < 0 (type 1 went more Rep), got {delta[1]:.4f}"


# ---------------------------------------------------------------------------
# Test 3: τ real-world range guard
# ---------------------------------------------------------------------------

def test_tau_real_world_range(synthetic_data):
    """τ must stay between 0.3 and 1.0 for realistic input data.

    0.3 is the theoretical lower bound even for the most volatile partisan-only turnout
    pattern.  τ > 1.0 would mean off-cycle elections outdraw presidential, which
    essentially never happens at the tract level in the US.
    """
    elections, assignments = synthetic_data
    tau = compute_tau(elections, assignments)
    assert (tau >= 0.3).all(), f"τ below 0.3 — data quality concern: {tau}"
    assert (tau <= 1.0).all(), f"τ above 1.0 — off-cycle exceeds presidential: {tau}"


# ---------------------------------------------------------------------------
# Test 4: δ real-world range guard
# ---------------------------------------------------------------------------

def test_delta_real_world_range(synthetic_data):
    """δ must stay between -0.15 and 0.15 for realistic input data.

    A shift beyond ±15pp after turnout adjustment would indicate data contamination
    or a structural model error (e.g., mixing primary vs general results).
    """
    elections, assignments = synthetic_data
    tau = compute_tau(elections, assignments)
    delta = compute_delta(elections, assignments, tau)
    assert (delta >= -0.15).all(), f"δ below -0.15: {delta}"
    assert (delta <= 0.15).all(), f"δ above +0.15: {delta}"


# ---------------------------------------------------------------------------
# Test 5: compute_and_save output files
# ---------------------------------------------------------------------------

def test_compute_and_save_creates_output_files(synthetic_data, tmp_path):
    """compute_and_save must create tau.npy, delta.npy, and summary.json."""
    elections, assignments = synthetic_data

    # Write synthetic data to parquet files in a temp directory.
    data_dir = tmp_path / "data" / "assembled"
    data_dir.mkdir(parents=True)
    communities_dir = tmp_path / "data" / "communities"
    communities_dir.mkdir(parents=True)
    output_dir = tmp_path / "data" / "behavior"

    elections_path = data_dir / "tract_elections.parquet"
    assignments_path = communities_dir / "tract_type_assignments.parquet"
    elections.to_parquet(elections_path, index=False)
    assignments.to_parquet(assignments_path, index=False)

    # Monkey-patch the module-level default paths so compute_and_save reads our files.
    import src.behavior.compute_behavior_params as cbp
    original_elections = cbp.DEFAULT_ELECTIONS_PATH
    original_assignments = cbp.DEFAULT_ASSIGNMENTS_PATH
    cbp.DEFAULT_ELECTIONS_PATH = str(elections_path)
    cbp.DEFAULT_ASSIGNMENTS_PATH = str(assignments_path)

    try:
        summary = cbp.compute_and_save(output_dir=str(output_dir))
    finally:
        # Always restore originals even if the test fails.
        cbp.DEFAULT_ELECTIONS_PATH = original_elections
        cbp.DEFAULT_ASSIGNMENTS_PATH = original_assignments

    assert (output_dir / "tau.npy").exists(), "tau.npy not created"
    assert (output_dir / "delta.npy").exists(), "delta.npy not created"
    assert (output_dir / "summary.json").exists(), "summary.json not created"

    tau = np.load(output_dir / "tau.npy")
    delta = np.load(output_dir / "delta.npy")
    assert tau.shape == (2,)
    assert delta.shape == (2,)

    with open(output_dir / "summary.json") as f:
        saved = json.load(f)
    assert saved["n_types"] == 2
    assert "n_tracts_used" in saved
    assert "mean_tau" in saved
    assert "std_tau" in saved
    assert "mean_delta" in saved
    assert "std_delta" in saved
    assert "computation_date" in saved

    # Summary returned from the function must match the file.
    assert summary["n_types"] == saved["n_types"]
    assert abs(summary["mean_tau"] - saved["mean_tau"]) < 1e-10


# ---------------------------------------------------------------------------
# Test 6: adjust_priors_for_cycle
# ---------------------------------------------------------------------------

def test_adjust_priors_noop_for_presidential():
    """Priors must be returned unchanged for presidential elections."""
    priors = np.array([0.45, 0.52, 0.48])
    tau = np.array([0.70, 0.85])
    delta = np.array([0.03, -0.02])
    result = adjust_priors_for_cycle(priors, tau, delta, is_presidential=True)
    np.testing.assert_array_equal(result, priors)


def test_adjust_priors_changes_for_offcycle():
    """Priors must differ from input for off-cycle elections with non-zero δ."""
    priors = np.array([0.45, 0.52, 0.48])
    tau = np.array([0.70, 0.85])
    delta = np.array([0.05, -0.02])  # non-zero, non-equal shifts
    result = adjust_priors_for_cycle(priors, tau, delta, is_presidential=False)
    # Result should differ because mean(δ) is non-zero.
    assert not np.allclose(result, priors), "Expected priors to change for off-cycle"


def test_adjust_priors_stays_within_bounds():
    """Adjusted priors must remain in [0, 1] even for extreme δ values."""
    priors = np.array([0.01, 0.99, 0.50])
    tau = np.array([0.60, 0.90])
    delta = np.array([0.20, 0.20])  # large positive shift
    result = adjust_priors_for_cycle(priors, tau, delta, is_presidential=False)
    assert (result >= 0.0).all(), f"Prior below 0: {result}"
    assert (result <= 1.0).all(), f"Prior above 1: {result}"


def test_adjust_priors_zero_delta_is_noop_for_offcycle():
    """If all δ values are zero, off-cycle priors should equal input priors."""
    priors = np.array([0.40, 0.55, 0.50])
    tau = np.array([0.70, 0.85])
    delta = np.array([0.0, 0.0])
    result = adjust_priors_for_cycle(priors, tau, delta, is_presidential=False)
    np.testing.assert_array_almost_equal(result, priors)


def test_adjust_priors_tau_delta_length_mismatch_raises():
    """Mismatched tau/delta lengths must raise a ValueError immediately."""
    priors = np.array([0.45, 0.50])
    tau = np.array([0.70, 0.85, 0.80])   # length 3
    delta = np.array([0.02, -0.01])       # length 2
    with pytest.raises(ValueError, match="same length"):
        adjust_priors_for_cycle(priors, tau, delta, is_presidential=False)
