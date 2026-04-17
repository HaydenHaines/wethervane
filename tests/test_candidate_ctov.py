"""Tests for candidate CTOV prior adjustment module."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.prediction.candidate_ctov import (
    apply_ctov_adjustment,
    load_ctov_adjustments,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ctov_data_dir(tmp_path):
    """Create minimal CTOV data files for testing."""
    J = 5  # small J for test speed

    # Crosswalk: one race with a candidate who has historical data
    crosswalk = {
        "_meta": {"description": "test"},
        "races": {
            "2026 SC Senate": {
                "R": [
                    {
                        "name": "Lindsey Graham",
                        "clean_name": "Lindsey Graham",
                        "bioguide_id": "G000359",
                        "historical_races": 2,
                    }
                ]
            },
            "2026 AL Senate": {
                "R": [
                    {
                        "name": "Katie Britt",
                        "clean_name": "Katie Britt",
                        "bioguide_id": None,
                        "historical_races": 0,
                    }
                ]
            },
            "2026 TX Senate": {
                "R": [
                    {
                        "name": "John Cornyn",
                        "clean_name": "John Cornyn",
                        "bioguide_id": "C001056",
                        "historical_races": 2,
                    }
                ]
            },
        },
    }
    xwalk_path = tmp_path / "candidate_2026_crosswalk.json"
    with open(xwalk_path, "w") as f:
        json.dump(crosswalk, f)

    # CTOV parquet: two races for Graham, two for Cornyn
    ctov_cols = [f"ctov_type_{i}" for i in range(J)]
    rows = [
        {
            "person_id": "G000359",
            "name": "Lindsey Graham",
            "party": "R",
            "year": 2020,
            "state": "SC",
            "office": "senate",
            **{ctov_cols[i]: (i - 2) * 0.01 for i in range(J)},
        },
        {
            "person_id": "G000359",
            "name": "Lindsey Graham",
            "party": "R",
            "year": 2014,
            "state": "SC",
            "office": "senate",
            **{ctov_cols[i]: (i - 2) * 0.02 for i in range(J)},
        },
        {
            "person_id": "C001056",
            "name": "John Cornyn",
            "party": "R",
            "year": 2020,
            "state": "TX",
            "office": "senate",
            **{ctov_cols[i]: i * 0.005 for i in range(J)},
        },
        {
            "person_id": "C001056",
            "name": "John Cornyn",
            "party": "R",
            "year": 2014,
            "state": "TX",
            "office": "senate",
            **{ctov_cols[i]: i * 0.01 for i in range(J)},
        },
    ]
    ctov_df = pd.DataFrame(rows)
    ctov_path = tmp_path / "candidate_ctov.parquet"
    ctov_df.to_parquet(ctov_path, index=False)

    # Badges JSON with CEC: Graham CEC=0.6 (above threshold), Cornyn CEC=0.2 (below)
    badges = {
        "G000359": {
            "name": "Lindsey Graham",
            "party": "R",
            "n_races": 2,
            "badges": ["Faith Coalition"],
            "badge_scores": {},
            "cec": 0.6,
        },
        "C001056": {
            "name": "John Cornyn",
            "party": "R",
            "n_races": 2,
            "badges": [],
            "badge_scores": {},
            "cec": 0.2,
        },
    }
    badges_path = tmp_path / "candidate_badges.json"
    with open(badges_path, "w") as f:
        json.dump(badges, f)

    return tmp_path


# ---------------------------------------------------------------------------
# Tests: load_ctov_adjustments
# ---------------------------------------------------------------------------


def test_load_ctov_basic(ctov_data_dir):
    """Loads CTOV adjustments for races with historical candidate data."""
    adjustments = load_ctov_adjustments(
        crosswalk_path=ctov_data_dir / "candidate_2026_crosswalk.json",
        ctov_path=ctov_data_dir / "candidate_ctov.parquet",
        badges_path=ctov_data_dir / "candidate_badges.json",
    )
    # Graham (CEC=0.6 > 0.3): should be included
    assert "2026 SC Senate" in adjustments
    # AL: no candidate with data
    assert "2026 AL Senate" not in adjustments


def test_load_ctov_cec_filter(ctov_data_dir):
    """Candidates with CEC below threshold are excluded."""
    adjustments = load_ctov_adjustments(
        crosswalk_path=ctov_data_dir / "candidate_2026_crosswalk.json",
        ctov_path=ctov_data_dir / "candidate_ctov.parquet",
        badges_path=ctov_data_dir / "candidate_badges.json",
    )
    # Cornyn (CEC=0.2 < 0.3): should be excluded
    assert "2026 TX Senate" not in adjustments


def test_load_ctov_recency_weighting(ctov_data_dir):
    """Multi-race candidates use recency-weighted CTOV average."""
    adjustments = load_ctov_adjustments(
        crosswalk_path=ctov_data_dir / "candidate_2026_crosswalk.json",
        ctov_path=ctov_data_dir / "candidate_ctov.parquet",
        badges_path=ctov_data_dir / "candidate_badges.json",
    )
    ctov_vec = adjustments["2026 SC Senate"]

    # Graham's 2020 CTOV: (i-2)*0.01, weights: [1, 2] normalized to [1/3, 2/3]
    # Graham's 2014 CTOV: (i-2)*0.02
    # Weighted avg for type 0: (1/3)*(-2*0.02) + (2/3)*(-2*0.01) = -0.04/3 + -0.04/3 = -0.04/3 * 2
    # Actually: 2014 is first (sorted by year), 2020 is last (most recent, weight=2)
    # weights = [1, 2], sum=3 → [1/3, 2/3]
    # type_0: (1/3)*(-0.04) + (2/3)*(-0.02) = -0.04/3 - 0.04/3 = -0.08/3 ≈ -0.02667
    expected_type_0 = (1 / 3) * (0 - 2) * 0.02 + (2 / 3) * (0 - 2) * 0.01
    assert abs(ctov_vec[0] - expected_type_0) < 1e-10


def test_load_ctov_missing_files():
    """Returns empty dict when data files are missing."""
    adjustments = load_ctov_adjustments(
        crosswalk_path="/nonexistent/crosswalk.json",
        ctov_path="/nonexistent/ctov.parquet",
    )
    assert adjustments == {}


def test_load_ctov_shape(ctov_data_dir):
    """CTOV vectors have correct J dimension."""
    adjustments = load_ctov_adjustments(
        crosswalk_path=ctov_data_dir / "candidate_2026_crosswalk.json",
        ctov_path=ctov_data_dir / "candidate_ctov.parquet",
        badges_path=ctov_data_dir / "candidate_badges.json",
    )
    for race_id, ctov_vec in adjustments.items():
        assert ctov_vec.shape == (5,), f"Wrong shape for {race_id}"


# ---------------------------------------------------------------------------
# Tests: apply_ctov_adjustment
# ---------------------------------------------------------------------------


def test_apply_ctov_basic():
    """CTOV shifts county priors by scaled, type-weighted amount."""
    N, J = 6, 3
    county_priors = np.full(N, 0.5)
    # Counties 0-2 in state A, 3-5 in state B
    state_mask = np.array([True, True, True, False, False, False])
    # Simple type scores: county 0 = pure type 0, county 1 = pure type 1, etc.
    type_scores = np.zeros((N, J))
    type_scores[0, 0] = 1.0
    type_scores[1, 1] = 1.0
    type_scores[2, 2] = 1.0
    type_scores[3, 0] = 1.0
    type_scores[4, 1] = 1.0
    type_scores[5, 2] = 1.0
    # CTOV: type 0 gets +0.05, type 1 gets -0.03, type 2 gets 0
    ctov_vec = np.array([0.05, -0.03, 0.0])

    # Use scale=1.0, max_shift=1.0 to test raw math without scaling
    result = apply_ctov_adjustment(county_priors, type_scores, ctov_vec, state_mask,
                                   scale=1.0, max_shift=1.0)

    # State A counties should be shifted
    assert abs(result[0] - 0.55) < 1e-10  # +0.05 from type 0
    assert abs(result[1] - 0.47) < 1e-10  # -0.03 from type 1
    assert abs(result[2] - 0.50) < 1e-10  # 0.0 from type 2
    # State B counties should be unchanged
    assert abs(result[3] - 0.50) < 1e-10
    assert abs(result[4] - 0.50) < 1e-10
    assert abs(result[5] - 0.50) < 1e-10


def test_apply_ctov_preserves_original():
    """apply_ctov_adjustment does not modify the input array."""
    N, J = 4, 2
    county_priors = np.full(N, 0.5)
    original = county_priors.copy()
    type_scores = np.eye(N, J)
    ctov_vec = np.array([0.1, -0.1])
    state_mask = np.array([True, True, False, False])

    apply_ctov_adjustment(county_priors, type_scores, ctov_vec, state_mask)

    np.testing.assert_array_equal(county_priors, original)


def test_apply_ctov_mixed_membership():
    """Counties with mixed type membership get blended CTOV shifts."""
    N, J = 2, 3
    county_priors = np.full(N, 0.5)
    state_mask = np.array([True, True])
    # County 0: 60% type 0, 40% type 1
    # County 1: 100% type 2
    type_scores = np.array([
        [0.6, 0.4, 0.0],
        [0.0, 0.0, 1.0],
    ])
    ctov_vec = np.array([0.10, -0.05, 0.02])

    # Use scale=1.0, max_shift=1.0 to test raw math
    result = apply_ctov_adjustment(county_priors, type_scores, ctov_vec, state_mask,
                                   scale=1.0, max_shift=1.0)

    # County 0: 0.5 + 0.6*0.10 + 0.4*(-0.05) = 0.5 + 0.06 - 0.02 = 0.54
    assert abs(result[0] - 0.54) < 1e-10
    # County 1: 0.5 + 1.0*0.02 = 0.52
    assert abs(result[1] - 0.52) < 1e-10


def test_apply_ctov_scale_and_cap():
    """Scale factor and cap prevent extreme shifts."""
    N, J = 2, 2
    county_priors = np.full(N, 0.5)
    state_mask = np.array([True, True])
    # County 0: 100% type 0
    # County 1: 100% type 1
    type_scores = np.array([[1.0, 0.0], [0.0, 1.0]])
    # Large CTOV: type 0 = -0.25 (would be -25pp raw)
    ctov_vec = np.array([-0.25, 0.01])

    # Default scale=0.3, max_shift=0.05
    result = apply_ctov_adjustment(county_priors, type_scores, ctov_vec, state_mask)

    # County 0: raw = -0.25, scaled = -0.075, capped = -0.05
    assert abs(result[0] - 0.45) < 1e-10
    # County 1: raw = 0.01, scaled = 0.003, no cap needed
    assert abs(result[1] - 0.503) < 1e-10
