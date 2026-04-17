# api/tests/test_candidates_list.py
"""Tests for the new candidates list endpoint and expanded profile (races field).

Also tests the predecessor lookup logic and predecessor endpoint.
Covers:
- GET /candidates list endpoint with query params (q, party, office, year, state)
- GET /candidates/{id} races[] field in response
- GET /candidates/{id}/predecessor single-race predecessor logic
- Specific candidate smoke tests: Ossoff, Warnock, Abrams (API correctness, not visual)
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.tests.conftest import _build_test_db, _noop_lifespan

# ── Test data ────────────────────────────────────────────────────────────────

# Minimal badge profiles for testing
_TEST_BADGES = {
    "O000168": {  # Ossoff
        "name": "Jon Ossoff",
        "party": "D",
        "n_races": 2,
        "badges": ["Suburban Professional"],
        "badge_scores": {
            "Suburban Professional": 0.031,
            "Hispanic Appeal": 0.008,
            "Rural Populist": -0.012,
            "Turnout Monster": 0.025,
            "Senior Whisperer": -0.004,
            "Faith Coalition": 0.005,
            "Black Community Strength": 0.018,
        },
        "badge_details": [
            {
                "name": "Suburban Professional",
                "score": 0.031,
                "provisional": False,
                "kind": "catalog",
                "fallback_reason": None,
            }
        ],
        "cec": 0.78,
    },
    "W000790": {  # Warnock
        "name": "Raphael Warnock",
        "party": "D",
        "n_races": 3,
        "badges": ["Black Community Strength", "Turnout Monster"],
        "badge_scores": {
            "Suburban Professional": 0.012,
            "Hispanic Appeal": 0.010,
            "Rural Populist": -0.020,
            "Turnout Monster": 0.055,
            "Senior Whisperer": 0.008,
            "Faith Coalition": 0.040,
            "Black Community Strength": 0.070,
        },
        "badge_details": [
            {
                "name": "Black Community Strength",
                "score": 0.070,
                "provisional": False,
                "kind": "catalog",
                "fallback_reason": None,
            },
            {
                "name": "Turnout Monster",
                "score": 0.055,
                "provisional": False,
                "kind": "catalog",
                "fallback_reason": None,
            },
        ],
        "cec": 0.82,
    },
    "A000370": {  # Abrams
        "name": "Stacey Abrams",
        "party": "D",
        "n_races": 2,
        "badges": ["Turnout Monster", "Black Community Strength"],
        "badge_scores": {
            "Suburban Professional": 0.015,
            "Hispanic Appeal": 0.012,
            "Rural Populist": -0.030,
            "Turnout Monster": 0.060,
            "Senior Whisperer": -0.002,
            "Faith Coalition": 0.020,
            "Black Community Strength": 0.065,
        },
        "badge_details": [
            {
                "name": "Turnout Monster",
                "score": 0.060,
                "provisional": False,
                "kind": "catalog",
                "fallback_reason": None,
            },
            {
                "name": "Black Community Strength",
                "score": 0.065,
                "provisional": False,
                "kind": "catalog",
                "fallback_reason": None,
            },
        ],
        "cec": 0.71,
    },
    "S000001": {  # Single-race candidate for predecessor testing
        "name": "Single Race Candidate",
        "party": "R",
        "n_races": 1,
        "badges": [],
        "badge_scores": {"Rural Populist": 0.020},
        "badge_details": [],
        "cec": 0.0,
    },
    "P000001": {  # Predecessor for single-race candidate
        "name": "Predecessor Candidate",
        "party": "R",
        "n_races": 1,
        "badges": [],
        "badge_scores": {"Rural Populist": 0.015},
        "badge_details": [],
        "cec": 0.0,
    },
    "G000001": {  # Governor candidate for filtering tests
        "name": "Test Governor",
        "party": "R",
        "n_races": 1,
        "badges": [],
        "badge_scores": {"Rural Populist": 0.040},
        "badge_details": [],
        "cec": 0.50,
    },
}

_TEST_REGISTRY = {
    "O000168": {
        "name": "Jon Ossoff",
        "party": "D",
        "bioguide_id": "O000168",
        "races": [
            {
                "year": 2017,
                "state": "GA",
                "office": "Senate",
                "special": True,
                "party": "D",
                "result": "loss",
                "actual_dem_share_2party": 0.4818,
            },
            {
                "year": 2020,
                "state": "GA",
                "office": "Senate",
                "special": False,
                "party": "D",
                "result": "win",
                "actual_dem_share_2party": 0.5049,
            },
        ],
    },
    "W000790": {
        "name": "Raphael Warnock",
        "party": "D",
        "bioguide_id": "W000790",
        "races": [
            {
                "year": 2020,
                "state": "GA",
                "office": "Senate",
                "special": True,
                "party": "D",
                "result": "win",
                "actual_dem_share_2party": 0.5090,
            },
            {
                "year": 2022,
                "state": "GA",
                "office": "Senate",
                "special": False,
                "party": "D",
                "result": "win",
                "actual_dem_share_2party": 0.4952,
            },
        ],
    },
    "A000370": {
        "name": "Stacey Abrams",
        "party": "D",
        "bioguide_id": "A000370",
        "races": [
            {
                "year": 2018,
                "state": "GA",
                "office": "Governor",
                "special": False,
                "party": "D",
                "result": "loss",
                "actual_dem_share_2party": 0.4888,
            },
            {
                "year": 2022,
                "state": "GA",
                "office": "Governor",
                "special": False,
                "party": "D",
                "result": "loss",
                "actual_dem_share_2party": 0.4668,
            },
        ],
    },
    "S000001": {
        "name": "Single Race Candidate",
        "party": "R",
        "bioguide_id": "S000001",
        "races": [
            {
                "year": 2022,
                "state": "NC",
                "office": "Senate",
                "special": False,
                "party": "R",
                "result": "win",
                "actual_dem_share_2party": 0.4590,
            }
        ],
    },
    "P000001": {
        "name": "Predecessor Candidate",
        "party": "R",
        "bioguide_id": "P000001",
        "races": [
            {
                "year": 2016,
                "state": "NC",
                "office": "Senate",
                "special": False,
                "party": "R",
                "result": "win",
                "actual_dem_share_2party": 0.4540,
            }
        ],
    },
    "G000001": {
        "name": "Test Governor",
        "party": "R",
        "bioguide_id": "G000001",
        "races": [
            {
                "year": 2022,
                "state": "FL",
                "office": "Governor",
                "special": False,
                "party": "R",
                "result": "win",
                "actual_dem_share_2party": 0.4020,
            }
        ],
    },
}


@pytest.fixture
def candidates_client():
    """TestClient with patched sabermetrics data for list/profile tests."""
    from api.main import create_app

    test_db = _build_test_db()

    with (
        patch("api.routers.candidates._BADGES", _TEST_BADGES),
        patch("api.routers.candidates._REGISTRY", _TEST_REGISTRY),
        patch(
            "api.routers.candidates._NAME_TO_BIOGUIDE",
            {v["name"]: k for k, v in _TEST_REGISTRY.items()},
        ),
        patch("api.routers.candidates._CTOV", pd.DataFrame()),
        patch("api.routers.candidates._CANDIDATES_2026", {}),
        patch("api.routers.candidates._TYPE_DISPLAY_NAMES", None),
    ):
        test_app = create_app(lifespan_override=_noop_lifespan)
        test_app.state.db = test_db
        test_app.state.version_id = "test_v1"
        test_app.state.K = 3
        test_app.state.sigma = np.eye(3) * 0.01
        test_app.state.mu_prior = np.full(3, 0.42)
        test_app.state.state_weights = pd.DataFrame()
        test_app.state.county_weights = pd.DataFrame()
        test_app.state.contract_ok = True

        with TestClient(test_app, raise_server_exceptions=True) as c:
            yield c

    test_db.close()


# ── Tests: GET /candidates (list endpoint) ───────────────────────────────────


class TestListEndpoint:
    def test_returns_200(self, candidates_client):
        resp = candidates_client.get("/api/v1/candidates")
        assert resp.status_code == 200

    def test_returns_all_candidates_by_default(self, candidates_client):
        resp = candidates_client.get("/api/v1/candidates")
        data = resp.json()
        assert "candidates" in data
        assert "total" in data
        assert data["total"] == len(_TEST_BADGES)

    def test_response_shape_per_candidate(self, candidates_client):
        resp = candidates_client.get("/api/v1/candidates")
        candidates = resp.json()["candidates"]
        assert len(candidates) > 0
        first = candidates[0]
        required_fields = {
            "bioguide_id", "name", "party", "n_races", "cec",
            "badges", "states", "offices", "years",
        }
        assert required_fields.issubset(first.keys()), f"Missing fields: {required_fields - first.keys()}"

    def test_filter_by_party_d(self, candidates_client):
        resp = candidates_client.get("/api/v1/candidates?party=D")
        data = resp.json()
        assert all(c["party"] == "D" for c in data["candidates"])
        # Ossoff, Warnock, Abrams should be in result
        names = {c["name"] for c in data["candidates"]}
        assert "Jon Ossoff" in names
        assert "Raphael Warnock" in names
        assert "Stacey Abrams" in names

    def test_filter_by_party_r(self, candidates_client):
        resp = candidates_client.get("/api/v1/candidates?party=R")
        data = resp.json()
        assert all(c["party"] == "R" for c in data["candidates"])
        assert data["total"] == 3  # S000001, P000001, G000001

    def test_filter_by_office_senate(self, candidates_client):
        resp = candidates_client.get("/api/v1/candidates?office=Senate")
        data = resp.json()
        for c in data["candidates"]:
            assert "Senate" in c["offices"], f"{c['name']} missing Senate office"

    def test_filter_by_office_governor(self, candidates_client):
        resp = candidates_client.get("/api/v1/candidates?office=Governor")
        data = resp.json()
        names = {c["name"] for c in data["candidates"]}
        assert "Stacey Abrams" in names
        assert "Test Governor" in names
        assert "Jon Ossoff" not in names

    def test_filter_by_year(self, candidates_client):
        resp = candidates_client.get("/api/v1/candidates?year=2022")
        data = resp.json()
        for c in data["candidates"]:
            assert 2022 in c["years"], f"{c['name']} missing 2022 in years"

    def test_filter_by_state_ga(self, candidates_client):
        resp = candidates_client.get("/api/v1/candidates?state=GA")
        data = resp.json()
        for c in data["candidates"]:
            assert "GA" in c["states"], f"{c['name']} missing GA in states"
        names = {c["name"] for c in data["candidates"]}
        assert "Jon Ossoff" in names
        assert "Raphael Warnock" in names
        assert "Stacey Abrams" in names

    def test_name_search_case_insensitive(self, candidates_client):
        resp = candidates_client.get("/api/v1/candidates?q=ossoff")
        data = resp.json()
        assert data["total"] == 1
        assert data["candidates"][0]["name"] == "Jon Ossoff"

    def test_name_search_partial_match(self, candidates_client):
        resp = candidates_client.get("/api/v1/candidates?q=warn")
        data = resp.json()
        assert data["total"] == 1
        assert data["candidates"][0]["name"] == "Raphael Warnock"

    def test_combined_filters(self, candidates_client):
        # D + Governor should only return Abrams
        resp = candidates_client.get("/api/v1/candidates?party=D&office=Governor")
        data = resp.json()
        assert data["total"] == 1
        assert data["candidates"][0]["name"] == "Stacey Abrams"

    def test_no_match_returns_empty_list(self, candidates_client):
        resp = candidates_client.get("/api/v1/candidates?q=zzz_no_match_zzz")
        data = resp.json()
        assert data["total"] == 0
        assert data["candidates"] == []

    def test_sorted_alphabetically_by_last_name(self, candidates_client):
        resp = candidates_client.get("/api/v1/candidates?party=D")
        data = resp.json()
        names = [c["name"] for c in data["candidates"]]
        # Last names: Abrams, Ossoff, Warnock — alphabetical
        last_names = [n.split()[-1] for n in names]
        assert last_names == sorted(last_names, key=str.lower)


# ── Tests: GET /candidates/{id} races[] field ────────────────────────────────


class TestProfileRacesField:
    def test_ossoff_profile_has_races(self, candidates_client):
        resp = candidates_client.get("/api/v1/candidates/O000168")
        assert resp.status_code == 200
        data = resp.json()
        assert "races" in data
        assert len(data["races"]) == 2

    def test_races_sorted_by_year_descending(self, candidates_client):
        resp = candidates_client.get("/api/v1/candidates/O000168")
        races = resp.json()["races"]
        years = [r["year"] for r in races]
        assert years == sorted(years, reverse=True)

    def test_races_include_required_fields(self, candidates_client):
        resp = candidates_client.get("/api/v1/candidates/O000168")
        races = resp.json()["races"]
        required = {"year", "state", "office", "special", "party", "result", "actual_dem_share_2party"}
        for race in races:
            assert required.issubset(race.keys()), f"Missing fields: {required - race.keys()}"

    def test_warnock_profile_correctness(self, candidates_client):
        resp = candidates_client.get("/api/v1/candidates/W000790")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Raphael Warnock"
        assert data["party"] == "D"
        assert data["n_races"] == 3
        assert data["cec"] == pytest.approx(0.82)
        # Races in test data
        assert len(data["races"]) == 2
        states = {r["state"] for r in data["races"]}
        assert states == {"GA"}

    def test_abrams_profile_has_governor_races(self, candidates_client):
        resp = candidates_client.get("/api/v1/candidates/A000370")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Stacey Abrams"
        offices = {r["office"] for r in data["races"]}
        assert offices == {"Governor"}

    def test_abrams_profile_results_accurate(self, candidates_client):
        resp = candidates_client.get("/api/v1/candidates/A000370")
        data = resp.json()
        results = {r["year"]: r["result"] for r in data["races"]}
        assert results[2018] == "loss"
        assert results[2022] == "loss"

    def test_dem_share_present_in_races(self, candidates_client):
        resp = candidates_client.get("/api/v1/candidates/O000168")
        races = resp.json()["races"]
        # All test races have actual_dem_share_2party set
        assert all(r["actual_dem_share_2party"] is not None for r in races)

    def test_ossoff_dem_shares_correct(self, candidates_client):
        resp = candidates_client.get("/api/v1/candidates/O000168")
        races = resp.json()["races"]
        shares_by_year = {r["year"]: r["actual_dem_share_2party"] for r in races}
        assert abs(shares_by_year[2020] - 0.5049) < 0.001
        assert abs(shares_by_year[2017] - 0.4818) < 0.001


# ── Tests: GET /candidates/{id}/predecessor ──────────────────────────────────


class TestPredecessorEndpoint:
    def test_single_race_candidate_has_predecessor(self, candidates_client):
        resp = candidates_client.get("/api/v1/candidates/S000001/predecessor")
        assert resp.status_code == 200
        data = resp.json()
        # Predecessor should be P000001 who ran in NC Senate R in 2016 (before 2022)
        assert data is not None
        assert data["bioguide_id"] == "P000001"
        assert data["name"] == "Predecessor Candidate"
        assert data["year"] == 2016

    def test_multi_race_candidate_returns_null_predecessor(self, candidates_client):
        # Ossoff has 2 races — no predecessor needed
        resp = candidates_client.get("/api/v1/candidates/O000168/predecessor")
        assert resp.status_code == 200
        assert resp.json() is None

    def test_predecessor_unknown_candidate_404(self, candidates_client):
        resp = candidates_client.get("/api/v1/candidates/UNKNOWN_XYZ/predecessor")
        assert resp.status_code == 404

    def test_predecessor_has_required_fields(self, candidates_client):
        resp = candidates_client.get("/api/v1/candidates/S000001/predecessor")
        data = resp.json()
        assert data is not None
        assert {"bioguide_id", "name", "year"}.issubset(data.keys())

    def test_predecessor_year_is_before_candidate(self, candidates_client):
        """Predecessor must have run in a strictly earlier year."""
        resp = candidates_client.get("/api/v1/candidates/S000001/predecessor")
        data = resp.json()
        assert data["year"] < 2022  # S000001's only race is 2022
