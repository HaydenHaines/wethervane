"""Tests for src/covariance/run_county_covariance.py"""
import numpy as np
import pandas as pd
import pytest
from src.covariance.run_county_covariance import (
    compute_theta_obs,
    identify_k_ref,
)


@pytest.fixture
def sample_assignments():
    return pd.DataFrame({
        "county_fips": ["12001", "12003", "13001", "13003", "01001"],
        "community_id": [0, 0, 1, 1, 2],
    })


@pytest.fixture
def sample_election_data():
    return pd.DataFrame({
        "county_fips": ["12001", "12003", "13001", "13003", "01001"],
        "dem_share": [0.60, 0.55, 0.40, 0.35, 0.30],
        "total_votes": [50000.0, 30000.0, 45000.0, 20000.0, 15000.0],
    })


def test_compute_theta_obs_shape(sample_assignments, sample_election_data):
    """theta_obs should be K × T."""
    elections = [sample_election_data, sample_election_data]
    theta_obs, theta_se, obs_mask = compute_theta_obs(sample_assignments, elections)
    k = sample_assignments["community_id"].nunique()
    assert theta_obs.shape == (k, 2)
    assert theta_se.shape == (k, 2)
    assert obs_mask.shape == (k, 2)


def test_compute_theta_obs_weighted_mean(sample_assignments, sample_election_data):
    """Community 0 theta = weighted mean of its counties' dem_shares."""
    elections = [sample_election_data]
    theta_obs, _, _ = compute_theta_obs(sample_assignments, elections)
    # Community 0: county 12001 (share=0.60, total=50000) + county 12003 (share=0.55, total=30000)
    expected = (0.60 * 50000 + 0.55 * 30000) / (50000 + 30000)
    assert abs(theta_obs[0, 0] - expected) < 1e-10


def test_obs_mask_all_ones_when_no_missing(sample_assignments, sample_election_data):
    elections = [sample_election_data]
    _, _, obs_mask = compute_theta_obs(sample_assignments, elections)
    assert obs_mask.sum() == obs_mask.size


def test_identify_k_ref_most_democratic(sample_assignments, sample_election_data):
    """k_ref = community with highest mean dem_share across elections."""
    elections = [sample_election_data]
    theta_obs, _, _ = compute_theta_obs(sample_assignments, elections)
    k_ref = identify_k_ref(theta_obs)
    # Community 0 has highest dem_share (~0.5875), so k_ref should be 0
    # Stan is 1-indexed so k_ref = 1
    assert k_ref == 1


def test_obs_mask_one_when_partial_nan(sample_assignments):
    """A community with ONE NaN county but other valid counties is still observed."""
    election_with_partial_nan = pd.DataFrame({
        "county_fips": ["12001", "12003", "13001", "13003", "01001"],
        "dem_share": [float("nan"), 0.55, 0.40, 0.35, 0.30],
        "total_votes": [50000.0, 30000.0, 45000.0, 20000.0, 15000.0],
    })
    _, _, obs_mask = compute_theta_obs(sample_assignments, [election_with_partial_nan])
    # Community 0 has one NaN (12001) but 12003 is valid → obs_mask[0,0] = 1
    assert obs_mask[0, 0] == 1.0


def test_obs_mask_zero_when_all_nan(sample_assignments):
    """obs_mask = 0 when ALL counties in a community have NaN dem_share."""
    election_all_nan_c0 = pd.DataFrame({
        "county_fips": ["12001", "12003", "13001", "13003", "01001"],
        "dem_share": [float("nan"), float("nan"), 0.40, 0.35, 0.30],
        "total_votes": [50000.0, 30000.0, 45000.0, 20000.0, 15000.0],
    })
    _, _, obs_mask = compute_theta_obs(sample_assignments, [election_all_nan_c0])
    assert obs_mask[0, 0] == 0.0
    assert obs_mask[1, 0] == 1.0
