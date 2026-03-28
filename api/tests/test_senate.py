"""Tests for GET /api/v1/senate/overview."""
from __future__ import annotations

import duckdb
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.routers.senate import (
    DEM_SAFE_SEATS,
    GOP_SAFE_SEATS,
    _build_headline,
    _margin_to_rating,
    _rating_sort_key,
)
from api.tests.conftest import _noop_lifespan


# ── Unit tests for helper functions ────────────────────────────────────────


class TestMarginToRating:
    def test_tossup_within_3pp(self):
        assert _margin_to_rating(0.02) == "tossup"
        assert _margin_to_rating(-0.02) == "tossup"
        assert _margin_to_rating(0.0) == "tossup"

    def test_lean_between_3_and_8pp(self):
        assert _margin_to_rating(0.05) == "lean"
        assert _margin_to_rating(-0.05) == "lean"
        assert _margin_to_rating(0.03) == "lean"

    def test_likely_between_8_and_15pp(self):
        assert _margin_to_rating(0.10) == "likely"
        assert _margin_to_rating(-0.10) == "likely"
        assert _margin_to_rating(0.08) == "likely"

    def test_safe_above_15pp(self):
        assert _margin_to_rating(0.20) == "safe"
        assert _margin_to_rating(-0.20) == "safe"
        assert _margin_to_rating(0.15) == "safe"

    def test_boundary_at_3pp_is_lean(self):
        # abs(margin) == 0.03 is NOT < 0.03, so it's lean
        assert _margin_to_rating(0.03) == "lean"

    def test_boundary_at_8pp_is_likely(self):
        assert _margin_to_rating(0.08) == "likely"

    def test_boundary_at_15pp_is_safe(self):
        assert _margin_to_rating(0.15) == "safe"


class TestRatingSortKey:
    def test_tossup_sorts_first(self):
        assert _rating_sort_key("tossup") < _rating_sort_key("lean")

    def test_safe_sorts_last(self):
        assert _rating_sort_key("safe") > _rating_sort_key("likely")

    def test_full_order(self):
        order = ["tossup", "lean", "likely", "safe"]
        keys = [_rating_sort_key(r) for r in order]
        assert keys == sorted(keys)


class TestBuildHeadline:
    def test_many_tossups(self):
        races = [
            {"rating": "tossup", "margin": 0.01},
            {"rating": "tossup", "margin": -0.01},
            {"rating": "tossup", "margin": 0.02},
        ]
        headline, subtitle = _build_headline(races)
        assert "Competitive" in headline

    def test_gop_favored(self):
        races = [
            {"rating": "lean", "margin": -0.05},
            {"rating": "likely", "margin": -0.10},
            {"rating": "lean", "margin": -0.04},
        ]
        headline, subtitle = _build_headline(races)
        assert "Republican" in headline

    def test_dem_favored(self):
        races = [
            {"rating": "lean", "margin": 0.05},
            {"rating": "likely", "margin": 0.10},
            {"rating": "lean", "margin": 0.04},
        ]
        headline, subtitle = _build_headline(races)
        assert "Democrat" in headline


# ── Fixtures ────────────────────────────────────────────────────────────────


def _build_senate_db() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with synthetic Senate race data."""
    con = duckdb.connect(":memory:")

    con.execute("""
        CREATE TABLE counties (
            county_fips      VARCHAR PRIMARY KEY,
            state_abbr       VARCHAR NOT NULL,
            state_fips       VARCHAR NOT NULL,
            county_name      VARCHAR,
            total_votes_2024 INTEGER
        )
    """)
    counties = [
        ("12001", "FL", "12", "Alachua County, FL", 100000),
        ("12003", "FL", "12", "Baker County, FL",    15000),
        ("13001", "GA", "13", "Appling County, GA",  80000),
        ("13003", "GA", "13", "Atkinson County, GA",  8000),
    ]
    for row in counties:
        con.execute("INSERT INTO counties VALUES (?, ?, ?, ?, ?)", list(row))

    con.execute("""
        CREATE TABLE model_versions (
            version_id VARCHAR PRIMARY KEY,
            role VARCHAR, k INTEGER, j INTEGER,
            shift_type VARCHAR, vote_share_type VARCHAR,
            n_training_dims INTEGER, n_holdout_dims INTEGER,
            holdout_r VARCHAR, geography VARCHAR,
            description VARCHAR, created_at TIMESTAMP
        )
    """)
    con.execute(
        "INSERT INTO model_versions VALUES ('test_v1', 'current', 3, 7, "
        "'logodds', 'total', 30, 3, '0.90', 'test', 'test', '2026-01-01')"
    )

    con.execute("""
        CREATE TABLE predictions (
            county_fips VARCHAR NOT NULL,
            race        VARCHAR NOT NULL,
            version_id  VARCHAR NOT NULL,
            pred_dem_share DOUBLE,
            pred_std       DOUBLE,
            pred_lo90      DOUBLE,
            pred_hi90      DOUBLE,
            state_pred     DOUBLE,
            poll_avg       DOUBLE,
            PRIMARY KEY (county_fips, race, version_id)
        )
    """)
    # FL Senate: tossup (~50% Dem share)
    for fips, share in [("12001", 0.502), ("12003", 0.498)]:
        con.execute(
            "INSERT INTO predictions VALUES (?, '2026 FL Senate', 'test_v1', ?, 0.03, ?, ?, 0.50, 0.50)",
            [fips, share, share - 0.05, share + 0.05],
        )
    # GA Senate: lean GOP (~46% Dem share → margin ~-0.04)
    for fips, share in [("13001", 0.462), ("13003", 0.458)]:
        con.execute(
            "INSERT INTO predictions VALUES (?, '2026 GA Senate', 'test_v1', ?, 0.03, ?, ?, 0.46, 0.46)",
            [fips, share, share - 0.05, share + 0.05],
        )
    # A non-Senate race — should NOT appear in senate/overview
    con.execute(
        "INSERT INTO predictions VALUES ('12001', '2026 FL Governor', 'test_v1', "
        "0.48, 0.03, 0.43, 0.53, 0.48, 0.48)"
    )

    con.execute("""
        CREATE TABLE polls (
            poll_id   VARCHAR NOT NULL,
            race      VARCHAR NOT NULL,
            geography VARCHAR NOT NULL,
            geo_level VARCHAR NOT NULL,
            dem_share FLOAT   NOT NULL,
            n_sample  INTEGER NOT NULL,
            date      VARCHAR,
            pollster  VARCHAR,
            notes     VARCHAR,
            cycle     VARCHAR NOT NULL,
            PRIMARY KEY (poll_id)
        )
    """)
    con.execute(
        "INSERT INTO polls VALUES ('p1', '2026 FL Senate', 'FL', 'state', 0.50, 600, "
        "'2026-01-15', 'Siena', NULL, '2026')"
    )
    con.execute(
        "INSERT INTO polls VALUES ('p2', '2026 FL Senate', 'FL', 'state', 0.49, 700, "
        "'2026-02-01', 'Quinnipiac', NULL, '2026')"
    )
    # No polls for GA Senate

    return con


@pytest.fixture
def senate_client():
    con = _build_senate_db()
    test_app = create_app(lifespan_override=_noop_lifespan)
    test_app.state.db = con
    test_app.state.version_id = "test_v1"
    # Minimal state fields the app may inspect
    test_app.state.contract_ok = True
    with TestClient(test_app, raise_server_exceptions=True) as c:
        yield c
    con.close()


# ── Endpoint tests ──────────────────────────────────────────────────────────


class TestSenateOverview:
    def test_status_200(self, senate_client):
        resp = senate_client.get("/api/v1/senate/overview")
        assert resp.status_code == 200

    def test_response_shape(self, senate_client):
        data = senate_client.get("/api/v1/senate/overview").json()
        assert "headline" in data
        assert "subtitle" in data
        assert "dem_seats_safe" in data
        assert "gop_seats_safe" in data
        assert "races" in data
        assert isinstance(data["races"], list)

    def test_safe_seat_counts(self, senate_client):
        data = senate_client.get("/api/v1/senate/overview").json()
        assert data["dem_seats_safe"] == DEM_SAFE_SEATS
        assert data["gop_seats_safe"] == GOP_SAFE_SEATS

    def test_only_senate_races_returned(self, senate_client):
        data = senate_client.get("/api/v1/senate/overview").json()
        for race in data["races"]:
            assert "senate" in race["race"].lower() or "Senate" in race["race"]

    def test_fl_senate_is_tossup(self, senate_client):
        data = senate_client.get("/api/v1/senate/overview").json()
        fl = next(r for r in data["races"] if r["state"] == "FL")
        assert fl["rating"] == "tossup"
        assert abs(fl["margin"]) < 0.03

    def test_ga_senate_is_lean(self, senate_client):
        data = senate_client.get("/api/v1/senate/overview").json()
        ga = next(r for r in data["races"] if r["state"] == "GA")
        assert ga["rating"] == "lean"
        assert ga["margin"] < 0  # GOP-favored

    def test_slug_format(self, senate_client):
        data = senate_client.get("/api/v1/senate/overview").json()
        for race in data["races"]:
            assert " " not in race["slug"]
            assert race["slug"] == race["race"].lower().replace(" ", "-")

    def test_poll_counts(self, senate_client):
        data = senate_client.get("/api/v1/senate/overview").json()
        fl = next(r for r in data["races"] if r["state"] == "FL")
        ga = next(r for r in data["races"] if r["state"] == "GA")
        assert fl["n_polls"] == 2
        assert ga["n_polls"] == 0

    def test_tossup_sorts_before_lean(self, senate_client):
        data = senate_client.get("/api/v1/senate/overview").json()
        ratings = [r["rating"] for r in data["races"]]
        sort_order = [_rating_sort_key(r) for r in ratings]
        assert sort_order == sorted(sort_order)

    def test_headline_is_string(self, senate_client):
        data = senate_client.get("/api/v1/senate/overview").json()
        assert isinstance(data["headline"], str)
        assert len(data["headline"]) > 0

    def test_no_governor_race_in_output(self, senate_client):
        data = senate_client.get("/api/v1/senate/overview").json()
        for race in data["races"]:
            assert "governor" not in race["race"].lower()

    def test_margin_is_rounded_float(self, senate_client):
        data = senate_client.get("/api/v1/senate/overview").json()
        for race in data["races"]:
            assert isinstance(race["margin"], float)


class TestSenateOverviewEmptyDB:
    """Endpoint gracefully handles a DB with no Senate predictions."""

    def test_returns_empty_races_list(self):
        con = duckdb.connect(":memory:")
        con.execute(
            "CREATE TABLE model_versions (version_id VARCHAR PRIMARY KEY, role VARCHAR, "
            "k INTEGER, j INTEGER, shift_type VARCHAR, vote_share_type VARCHAR, "
            "n_training_dims INTEGER, n_holdout_dims INTEGER, holdout_r VARCHAR, "
            "geography VARCHAR, description VARCHAR, created_at TIMESTAMP)"
        )
        con.execute(
            "INSERT INTO model_versions VALUES ('v1', 'current', 3, 7, 'logodds', "
            "'total', 30, 3, '0.90', 'test', 'test', '2026-01-01')"
        )
        con.execute(
            "CREATE TABLE counties (county_fips VARCHAR PRIMARY KEY, state_abbr VARCHAR, "
            "state_fips VARCHAR, county_name VARCHAR, total_votes_2024 INTEGER)"
        )
        con.execute(
            "CREATE TABLE predictions (county_fips VARCHAR, race VARCHAR, version_id VARCHAR, "
            "pred_dem_share DOUBLE, pred_std DOUBLE, pred_lo90 DOUBLE, pred_hi90 DOUBLE, "
            "state_pred DOUBLE, poll_avg DOUBLE)"
        )
        con.execute(
            "CREATE TABLE polls (poll_id VARCHAR PRIMARY KEY, race VARCHAR, geography VARCHAR, "
            "geo_level VARCHAR, dem_share FLOAT, n_sample INTEGER, date VARCHAR, "
            "pollster VARCHAR, notes VARCHAR, cycle VARCHAR)"
        )

        test_app = create_app(lifespan_override=_noop_lifespan)
        test_app.state.db = con
        test_app.state.version_id = "v1"
        test_app.state.contract_ok = True

        with TestClient(test_app, raise_server_exceptions=True) as c:
            resp = c.get("/api/v1/senate/overview")
        con.close()

        assert resp.status_code == 200
        data = resp.json()
        assert data["races"] == []
        assert data["dem_seats_safe"] == DEM_SAFE_SEATS
        assert data["gop_seats_safe"] == GOP_SAFE_SEATS
