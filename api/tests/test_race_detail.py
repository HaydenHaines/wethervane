"""Tests for /forecast/race/{slug} and /forecast/race-slugs endpoints.

The test DB (from conftest.py) uses race="FL_Senate" which is the legacy format.
We add a supplemental fixture that seeds the "2026 FL Senate" format races
expected by the new slug-based endpoints.
"""
from __future__ import annotations

import duckdb
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.routers.forecast import race_to_slug, slug_to_race
from api.tests.conftest import _build_test_state, _noop_lifespan

# ── Slug conversion unit tests ────────────────────────────────────────────

class TestSlugConversion:
    def test_race_to_slug_governor(self):
        assert race_to_slug("2026 FL Governor") == "2026-fl-governor"

    def test_race_to_slug_senate(self):
        assert race_to_slug("2026 GA Senate") == "2026-ga-senate"

    def test_race_to_slug_lowercase(self):
        assert race_to_slug("2026 AL Governor") == "2026-al-governor"

    def test_slug_to_race_governor(self):
        assert slug_to_race("2026-fl-governor") == "2026 FL Governor"

    def test_slug_to_race_senate(self):
        assert slug_to_race("2026-ga-senate") == "2026 GA Senate"

    def test_slug_roundtrip(self):
        for race in ["2026 FL Governor", "2026 GA Senate", "2026 AL Governor"]:
            assert slug_to_race(race_to_slug(race)) == race

    def test_slug_to_race_too_short(self):
        # Should not raise, just return slug unchanged
        result = slug_to_race("short")
        assert isinstance(result, str)


# ── Fixture with 2026-format race data ────────────────────────────────────

TEST_VERSION = "test_v1"
TEST_FIPS_RACES = ["12001", "12003", "13001", "13003", "01001"]
# Mapping: fips -> (state_abbr, state_fips, county_name)
TEST_COUNTIES = {
    "12001": ("FL", "12", "Alachua County, FL"),
    "12003": ("FL", "12", "Baker County, FL"),
    "13001": ("GA", "13", "Appling County, GA"),
    "13003": ("GA", "13", "Atkinson County, GA"),
    "01001": ("AL", "01", "Autauga County, AL"),
}


def _build_race_detail_db() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with 2026-format races and type data for race detail tests."""
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
    # Synthetic vote totals give FL counties ~10x more votes than AL/GA (urban weight test)
    test_votes = {"12001": 100000, "12003": 15000, "13001": 80000, "13003": 8000, "01001": 28000}
    for fips, (state, sfips, name) in TEST_COUNTIES.items():
        con.execute("INSERT INTO counties VALUES (?, ?, ?, ?, ?)", [fips, state, sfips, name, test_votes[fips]])

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
        "INSERT INTO model_versions VALUES "
        "(?, 'current', 3, 4, 'logodds', 'total', 30, 3, '0.70', 'test', 'test', '2026-01-01')",
        [TEST_VERSION],
    )

    con.execute("""
        CREATE TABLE predictions (
            county_fips VARCHAR NOT NULL,
            race VARCHAR NOT NULL,
            version_id VARCHAR NOT NULL,
            pred_dem_share DOUBLE,
            pred_std DOUBLE,
            pred_lo90 DOUBLE,
            pred_hi90 DOUBLE,
            state_pred DOUBLE,
            poll_avg DOUBLE,
            PRIMARY KEY (county_fips, race, version_id)
        )
    """)
    # FL counties get FL races, GA gets GA races, AL gets AL races, plus "baseline"
    races = {
        "12001": ["2026 FL Governor", "2026 FL Senate"],
        "12003": ["2026 FL Governor", "2026 FL Senate"],
        "13001": ["2026 GA Governor", "2026 GA Senate"],
        "13003": ["2026 GA Governor", "2026 GA Senate"],
        "01001": ["2026 AL Governor"],
    }
    for fips, fips_races in races.items():
        for race in fips_races:
            con.execute(
                "INSERT INTO predictions VALUES (?, ?, ?, 0.42, 0.03, 0.37, 0.47, 0.42, 0.46)",
                [fips, race, TEST_VERSION],
            )
    # Also insert a "baseline" that should NOT appear in race-slugs
    for fips in TEST_FIPS_RACES:
        con.execute(
            "INSERT INTO predictions VALUES (?, 'baseline', ?, 0.42, 0.03, 0.37, 0.47, 0.42, 0.46)",
            [fips, TEST_VERSION],
        )

    con.execute("""
        CREATE TABLE community_assignments (
            county_fips VARCHAR NOT NULL,
            community_id INTEGER NOT NULL,
            k INTEGER NOT NULL,
            version_id VARCHAR NOT NULL,
            PRIMARY KEY (county_fips, k, version_id)
        )
    """)
    for i, fips in enumerate(TEST_FIPS_RACES):
        con.execute("INSERT INTO community_assignments VALUES (?, ?, ?, ?)", [fips, i % 3, 3, TEST_VERSION])

    con.execute("""
        CREATE TABLE community_sigma (
            community_id_row INTEGER NOT NULL,
            community_id_col INTEGER NOT NULL,
            sigma_value DOUBLE,
            version_id VARCHAR NOT NULL
        )
    """)
    for i in range(3):
        for j in range(3):
            val = 0.01 if i == j else 0.005
            con.execute("INSERT INTO community_sigma VALUES (?, ?, ?, ?)", [i, j, val, TEST_VERSION])

    con.execute("""
        CREATE TABLE types (
            type_id INTEGER NOT NULL,
            super_type_id INTEGER NOT NULL,
            display_name VARCHAR NOT NULL,
            median_hh_income DOUBLE,
            pct_bachelors_plus DOUBLE,
            pct_white_nh DOUBLE,
            log_pop_density DOUBLE,
            narrative VARCHAR,
            version_id VARCHAR NOT NULL,
            PRIMARY KEY (type_id, version_id)
        )
    """)
    for tid, name in [(0, "Rural Conservative"), (1, "Suburban Moderate"), (2, "Urban Progressive"), (3, "Small Town")]:
        con.execute(
            "INSERT INTO types VALUES (?, 0, ?, 50000, 0.20, 0.75, 2.0, NULL, ?)",
            [tid, name, TEST_VERSION],
        )

    con.execute("""
        CREATE TABLE super_types (
            super_type_id INTEGER PRIMARY KEY,
            display_name VARCHAR
        )
    """)
    con.execute("INSERT INTO super_types VALUES (0, 'All Types')")

    con.execute("""
        CREATE TABLE county_type_assignments (
            county_fips VARCHAR NOT NULL,
            dominant_type INTEGER NOT NULL,
            super_type INTEGER NOT NULL,
            version_id VARCHAR NOT NULL,
            PRIMARY KEY (county_fips, version_id)
        )
    """)
    type_map = {"12001": 2, "12003": 0, "13001": 1, "13003": 0, "01001": 3}
    for fips, dt in type_map.items():
        con.execute("INSERT INTO county_type_assignments VALUES (?, ?, 0, ?)", [fips, dt, TEST_VERSION])

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
    # FL Senate: 2 pollsters, 2 methodologies (LV + RV) — Medium confidence (2 pollsters < 3 for High)
    # FL Governor: 3 pollsters, 2 methodologies — High confidence
    # AL Governor: no polls — Low confidence
    con.execute("""
        INSERT INTO polls VALUES
        ('p1', '2026 FL Senate', 'FL', 'state', 0.45, 600, '2026-01-15', 'Siena',
         'D=45% R=55%; LV; src=rcp', '2026'),
        ('p2', '2026 FL Senate', 'FL', 'state', 0.47, 800, '2026-02-20', 'Quinnipiac',
         'D=47% R=53%; RV; src=rcp', '2026'),
        ('p3', '2026 FL Governor', 'FL', 'state', 0.48, 700, '2026-01-10', 'Siena',
         'D=48% R=52%; LV; src=rcp', '2026'),
        ('p4', '2026 FL Governor', 'FL', 'state', 0.50, 900, '2026-02-15', 'Quinnipiac',
         'D=50% R=50%; RV; src=rcp', '2026'),
        ('p5', '2026 FL Governor', 'FL', 'state', 0.46, 500, '2026-03-01', 'Emerson',
         'D=46% R=54%; LV; src=rcp', '2026')
    """)

    con.execute(
        "CREATE TABLE poll_crosstabs "
        "(poll_id VARCHAR, demographic_group VARCHAR, group_value VARCHAR, "
        "dem_share FLOAT, n_sample INTEGER)"
    )
    con.execute("CREATE TABLE poll_notes (poll_id VARCHAR, note_type VARCHAR, note_value VARCHAR)")
    con.execute("CREATE TABLE county_shifts (county_fips VARCHAR, version_id VARCHAR, pres_d_shift_00_04 DOUBLE)")
    con.execute(
        "CREATE TABLE county_demographics (county_fips VARCHAR PRIMARY KEY, "
        "pop_total BIGINT, pct_white_nh DOUBLE, pct_black DOUBLE, pct_asian DOUBLE, "
        "pct_hispanic DOUBLE, median_age DOUBLE, median_hh_income BIGINT, "
        "log_median_hh_income DOUBLE, pct_bachelors_plus DOUBLE, pct_graduate DOUBLE, "
        "pct_owner_occupied DOUBLE, pct_wfh DOUBLE, pct_transit DOUBLE, pct_management DOUBLE)"
    )

    return con


@pytest.fixture
def race_client():
    """TestClient seeded with 2026-format race data."""
    db = _build_race_detail_db()
    state = _build_test_state()

    test_app = create_app(lifespan_override=_noop_lifespan)
    test_app.state.db = db
    test_app.state.version_id = TEST_VERSION
    test_app.state.K = 3
    test_app.state.sigma = np.eye(3) * 0.01
    test_app.state.mu_prior = np.full(3, 0.42)
    test_app.state.state_weights = state["state_weights"]
    test_app.state.county_weights = state["county_weights"]
    test_app.state.contract_ok = True

    with TestClient(test_app, raise_server_exceptions=True) as c:
        yield c

    db.close()


# ── Endpoint tests ─────────────────────────────────────────────────────────

class TestGetRaceSlugs:
    def test_returns_list_of_slugs(self, race_client):
        resp = race_client.get("/api/v1/forecast/race-slugs")
        assert resp.status_code == 200
        slugs = resp.json()
        assert isinstance(slugs, list)
        assert len(slugs) > 0

    def test_excludes_baseline(self, race_client):
        resp = race_client.get("/api/v1/forecast/race-slugs")
        slugs = resp.json()
        assert "baseline" not in slugs

    def test_slugs_are_lowercase_hyphenated(self, race_client):
        resp = race_client.get("/api/v1/forecast/race-slugs")
        slugs = resp.json()
        for slug in slugs:
            assert slug == slug.lower()
            assert " " not in slug

    def test_contains_expected_races(self, race_client):
        resp = race_client.get("/api/v1/forecast/race-slugs")
        slugs = resp.json()
        assert "2026-fl-senate" in slugs
        assert "2026-fl-governor" in slugs


class TestGetRaceDetail:
    def test_returns_correct_structure(self, race_client):
        resp = race_client.get("/api/v1/forecast/race/2026-fl-senate")
        assert resp.status_code == 200
        data = resp.json()
        required = {
            "race", "slug", "state_abbr", "race_type", "year",
            "prediction", "n_counties", "polls", "type_breakdown",
        }
        for field in required:
            assert field in data, f"Missing field: {field}"

    def test_race_label_matches_slug(self, race_client):
        resp = race_client.get("/api/v1/forecast/race/2026-fl-senate")
        data = resp.json()
        assert data["race"] == "2026 FL Senate"
        assert data["slug"] == "2026-fl-senate"

    def test_state_abbr_extracted(self, race_client):
        resp = race_client.get("/api/v1/forecast/race/2026-fl-senate")
        data = resp.json()
        assert data["state_abbr"] == "FL"

    def test_race_type_extracted(self, race_client):
        resp = race_client.get("/api/v1/forecast/race/2026-fl-senate")
        data = resp.json()
        assert data["race_type"] == "Senate"

    def test_year_extracted(self, race_client):
        resp = race_client.get("/api/v1/forecast/race/2026-fl-senate")
        data = resp.json()
        assert data["year"] == 2026

    def test_prediction_is_float(self, race_client):
        resp = race_client.get("/api/v1/forecast/race/2026-fl-senate")
        data = resp.json()
        assert isinstance(data["prediction"], float)

    def test_n_counties_matches_state(self, race_client):
        resp = race_client.get("/api/v1/forecast/race/2026-fl-senate")
        data = resp.json()
        # 2 FL counties in test DB
        assert data["n_counties"] == 2

    def test_polls_included(self, race_client):
        resp = race_client.get("/api/v1/forecast/race/2026-fl-senate")
        data = resp.json()
        assert len(data["polls"]) == 2
        poll = data["polls"][0]
        assert "date" in poll
        assert "pollster" in poll
        assert "dem_share" in poll
        assert "n_sample" in poll

    def test_polls_empty_for_race_without_polls(self, race_client):
        resp = race_client.get("/api/v1/forecast/race/2026-al-governor")
        data = resp.json()
        assert data["polls"] == []

    def test_type_breakdown_present(self, race_client):
        resp = race_client.get("/api/v1/forecast/race/2026-fl-senate")
        data = resp.json()
        assert len(data["type_breakdown"]) > 0
        t = data["type_breakdown"][0]
        assert "type_id" in t
        assert "display_name" in t
        assert "n_counties" in t
        assert "mean_pred_dem_share" in t
        # total_votes enables vote-weighted sort order (GitHub #21)
        assert "total_votes" in t

    def test_type_breakdown_limited_to_5(self, race_client):
        resp = race_client.get("/api/v1/forecast/race/2026-fl-senate")
        data = resp.json()
        assert len(data["type_breakdown"]) <= 5

    def test_type_breakdown_sorted_by_vote_contribution(self, race_client):
        """Types must be ordered by total votes descending, not county count.

        FL Senate test data has two types:
          - type 2 (Urban Progressive): county 12001 with 100,000 votes
          - type 0 (Rural Conservative): county 12003 with 15,000 votes

        Urban Progressive has more votes but only 1 county vs Rural Conservative's 1
        county. A county-count sort would be a tie; a vote-weighted sort puts Urban
        Progressive first because it contributes far more electoral weight. This is
        the fix for GitHub issue #21 (Michigan showing only rural types).
        """
        resp = race_client.get("/api/v1/forecast/race/2026-fl-senate")
        data = resp.json()
        breakdown = data["type_breakdown"]
        assert len(breakdown) >= 2, "Need at least 2 types to verify ordering"
        votes = [t["total_votes"] for t in breakdown]
        # Every entry should have total_votes populated from the counties table
        assert all(v is not None for v in votes), "total_votes should be non-null"
        # Types must be in descending vote-contribution order
        assert votes == sorted(votes, reverse=True), (
            f"Types not sorted by vote contribution: {votes}"
        )
        # The heaviest type (Urban Progressive, 100K) must come before the lighter one
        assert votes[0] > votes[-1]

    def test_404_for_invalid_slug(self, race_client):
        resp = race_client.get("/api/v1/forecast/race/9999-xx-fake-race")
        assert resp.status_code == 404

    def test_governor_race(self, race_client):
        resp = race_client.get("/api/v1/forecast/race/2026-fl-governor")
        assert resp.status_code == 200
        data = resp.json()
        assert data["race_type"] == "Governor"
        assert data["state_abbr"] == "FL"


class TestBlendEndpoint:
    """Tests for POST /forecast/race/{slug}/blend — GitHub issue #13."""

    def test_returns_blend_result_structure(self, race_client):
        resp = race_client.post(
            "/api/v1/forecast/race/2026-fl-senate/blend",
            json={"model_prior": 60, "state_polls": 30, "national_polls": 10},
        )
        assert resp.status_code == 200
        data = resp.json()
        # All four keys must be present (may be null if no type model loaded)
        for key in ("prediction", "pred_std", "pred_lo90", "pred_hi90"):
            assert key in data, f"Missing key: {key}"

    def test_no_polls_race_returns_structural_prior(self, race_client):
        """Race without polls returns stored prediction, not null."""
        resp = race_client.post(
            "/api/v1/forecast/race/2026-al-governor/blend",
            json={"model_prior": 80, "state_polls": 15, "national_polls": 5},
        )
        assert resp.status_code == 200
        data = resp.json()
        # Structural prior should be available even without polls
        assert data["prediction"] is not None
        assert isinstance(data["prediction"], float)

    def test_different_weights_accepted(self, race_client):
        """Any valid weight combination (sum need not be 100 for the API, but
        typical usage will pass sliders summing to 100)."""
        for weights in [
            {"model_prior": 100, "state_polls": 0, "national_polls": 0},
            {"model_prior": 0, "state_polls": 100, "national_polls": 0},
            {"model_prior": 33, "state_polls": 33, "national_polls": 34},
        ]:
            resp = race_client.post(
                "/api/v1/forecast/race/2026-fl-senate/blend",
                json=weights,
            )
            assert resp.status_code == 200, f"Failed for weights {weights}: {resp.text}"

    def test_invalid_slug_returns_fallback_not_500(self, race_client):
        """Unknown slug should return a graceful non-5xx response or 404."""
        resp = race_client.post(
            "/api/v1/forecast/race/9999-zz-fake/blend",
            json={"model_prior": 60, "state_polls": 30, "national_polls": 10},
        )
        # Either 404 (race not found) or 200 with null prediction is acceptable.
        # What is NOT acceptable is a 500 internal server error.
        assert resp.status_code != 500

    def test_ci_bounds_bracket_prediction(self, race_client):
        """When prediction and CI bounds are present, lo90 <= prediction <= hi90."""
        resp = race_client.post(
            "/api/v1/forecast/race/2026-al-governor/blend",
            json={"model_prior": 60, "state_polls": 30, "national_polls": 10},
        )
        assert resp.status_code == 200
        data = resp.json()
        if data["prediction"] is not None and data["pred_lo90"] is not None:
            assert data["pred_lo90"] <= data["prediction"] <= data["pred_hi90"], (
                f"CI bounds do not bracket prediction: {data}"
            )


class TestHistoricalContext:
    """Tests for historical context in race detail and standalone /history endpoint."""

    def test_race_detail_includes_historical_context_field(self, race_client):
        """Race detail response always includes historical_context key (may be None)."""
        resp = race_client.get("/api/v1/forecast/race/2026-fl-senate")
        assert resp.status_code == 200
        data = resp.json()
        assert "historical_context" in data

    def test_race_detail_historical_context_none_for_untracked_race(self, race_client):
        """Races not in historical_results.json return historical_context=None."""
        resp = race_client.get("/api/v1/forecast/race/2026-fl-senate")
        data = resp.json()
        # FL Senate is not in the 15 tracked competitive races
        assert data["historical_context"] is None

    def test_history_endpoint_404_for_untracked_race(self, race_client):
        """/history endpoint returns 404 for races not in historical_results.json."""
        resp = race_client.get("/api/v1/forecast/race/2026-fl-senate/history")
        assert resp.status_code == 404

    def test_history_endpoint_mi_senate_data(self, race_client):
        """MI Senate history matches known 2020 result (Peters D+1.9)."""
        resp = race_client.get("/api/v1/forecast/race/2026-mi-senate/history")
        assert resp.status_code == 200
        data = resp.json()
        assert data["last_race"]["year"] == 2020
        assert data["last_race"]["winner"] == "Gary Peters"
        assert data["last_race"]["party"] == "D"
        assert abs(data["last_race"]["margin"] - 1.9) < 0.01

    def test_history_endpoint_pa_governor_data(self, race_client):
        """PA Governor history matches known 2022 result (Shapiro D+14.8)."""
        resp = race_client.get("/api/v1/forecast/race/2026-pa-governor/history")
        assert resp.status_code == 200
        data = resp.json()
        assert data["last_race"]["year"] == 2022
        assert data["last_race"]["winner"] == "Josh Shapiro"
        assert data["last_race"]["party"] == "D"
        assert abs(data["last_race"]["margin"] - 14.8) < 0.01

    def test_history_presidential_margin_sign_convention(self, race_client):
        """Presidential margin is negative for Trump wins (Rep advantage)."""
        resp = race_client.get("/api/v1/forecast/race/2026-mi-senate/history")
        assert resp.status_code == 200
        data = resp.json()
        assert data["presidential_2024"]["party"] == "R"
        assert data["presidential_2024"]["margin"] < 0

    def test_history_forecast_shift_none_in_standalone_endpoint(self, race_client):
        """Standalone /history endpoint returns forecast_shift=None (no prediction context)."""
        resp = race_client.get("/api/v1/forecast/race/2026-mi-senate/history")
        assert resp.status_code == 200
        data = resp.json()
        assert data["forecast_shift"] is None

    def test_history_endpoint_returns_structure(self, race_client):
        """Tracked races return proper historical_context structure with expected fields."""
        resp = race_client.get("/api/v1/forecast/race/2026-mi-senate/history")
        assert resp.status_code == 200
        data = resp.json()
        assert "last_race" in data
        assert "presidential_2024" in data
        last = data["last_race"]
        assert all(k in last for k in ("year", "winner", "party", "margin"))
        pres = data["presidential_2024"]
        assert all(k in pres for k in ("winner", "party", "margin"))


# ── Poll confidence integration tests ────────────────────────────────────────


class TestPollConfidence:
    """Integration tests for poll_confidence field on the race detail response.

    Test data (from _build_race_detail_db):
    - FL Senate:   2 polls, 2 pollsters (Siena + Quinnipiac), 2 methods (LV + RV) → Medium
    - FL Governor: 3 polls, 3 pollsters (Siena + Quinnipiac + Emerson), 2 methods (LV + RV) → High
    - AL Governor: 0 polls → Low
    """

    def test_race_detail_includes_poll_confidence_field(self, race_client):
        """Race detail response always includes poll_confidence key."""
        resp = race_client.get("/api/v1/forecast/race/2026-fl-senate")
        assert resp.status_code == 200
        data = resp.json()
        assert "poll_confidence" in data

    def test_poll_confidence_structure(self, race_client):
        """poll_confidence has the expected sub-fields."""
        resp = race_client.get("/api/v1/forecast/race/2026-fl-senate")
        data = resp.json()
        pc = data["poll_confidence"]
        assert pc is not None
        for field in ("n_polls", "n_pollsters", "n_methodologies", "label", "tooltip"):
            assert field in pc, f"Missing poll_confidence field: {field}"

    def test_medium_confidence_two_pollsters_two_methods(self, race_client):
        """FL Senate: 2 pollsters + 2 methods = Medium (needs 3+ pollsters for High)."""
        resp = race_client.get("/api/v1/forecast/race/2026-fl-senate")
        data = resp.json()
        pc = data["poll_confidence"]
        assert pc["label"] == "Medium"
        assert pc["n_pollsters"] == 2
        assert pc["n_methodologies"] == 2
        assert pc["n_polls"] == 2

    def test_high_confidence_three_pollsters_two_methods(self, race_client):
        """FL Governor: 3 pollsters + 2 methods → High confidence."""
        resp = race_client.get("/api/v1/forecast/race/2026-fl-governor")
        data = resp.json()
        pc = data["poll_confidence"]
        assert pc["label"] == "High"
        assert pc["n_pollsters"] == 3
        assert pc["n_methodologies"] == 2
        assert pc["n_polls"] == 3

    def test_low_confidence_no_polls(self, race_client):
        """AL Governor: no polls → Low confidence with zero counts."""
        resp = race_client.get("/api/v1/forecast/race/2026-al-governor")
        data = resp.json()
        pc = data["poll_confidence"]
        assert pc["label"] == "Low"
        assert pc["n_polls"] == 0
        assert pc["n_pollsters"] == 0

    def test_tooltip_is_human_readable_string(self, race_client):
        """Tooltip must be a non-empty string mentioning pollsters and polls."""
        resp = race_client.get("/api/v1/forecast/race/2026-fl-senate")
        data = resp.json()
        tooltip = data["poll_confidence"]["tooltip"]
        assert isinstance(tooltip, str)
        assert len(tooltip) > 0
        assert "pollster" in tooltip or "pollsters" in tooltip
        assert "poll" in tooltip

    def test_tooltip_no_polls_message(self, race_client):
        """Zero-poll races get a clear 'No polls' tooltip."""
        resp = race_client.get("/api/v1/forecast/race/2026-al-governor")
        data = resp.json()
        assert data["poll_confidence"]["tooltip"] == "No polls"


# ── Poll confidence unit tests ─────────────────────────────────────────────────


class TestPollConfidenceUnit:
    """Unit tests for _compute_poll_confidence and _infer_methodology helpers."""

    def test_infer_methodology_lv(self):
        from api.routers.forecast.race_detail import _infer_methodology

        assert _infer_methodology("D=45% R=55%; LV; src=rcp") == "LV"

    def test_infer_methodology_rv(self):
        from api.routers.forecast.race_detail import _infer_methodology

        assert _infer_methodology("D=47% R=53%; RV; src=wikipedia") == "RV"

    def test_infer_methodology_none(self):
        from api.routers.forecast.race_detail import _infer_methodology

        assert _infer_methodology(None) == "Unknown"

    def test_infer_methodology_no_match(self):
        from api.routers.forecast.race_detail import _infer_methodology

        assert _infer_methodology("D=45% R=55%; src=270towin") == "Unknown"

    def test_compute_confidence_empty_df(self):
        import pandas as pd

        from api.routers.forecast.race_detail import _compute_poll_confidence

        empty = pd.DataFrame(columns=["pollster", "notes"])
        result = _compute_poll_confidence(empty)
        assert result.label == "Low"
        assert result.n_polls == 0
        assert result.n_pollsters == 0
        assert result.tooltip == "No polls"

    def test_compute_confidence_single_pollster_single_method(self):
        import pandas as pd

        from api.routers.forecast.race_detail import _compute_poll_confidence

        df = pd.DataFrame([
            {"pollster": "Siena", "notes": "D=45% R=55%; LV; src=rcp"},
        ])
        result = _compute_poll_confidence(df)
        assert result.label == "Low"
        assert result.n_pollsters == 1
        assert result.n_methodologies == 1

    def test_compute_confidence_two_pollsters_one_method_is_medium(self):
        import pandas as pd

        from api.routers.forecast.race_detail import _compute_poll_confidence

        df = pd.DataFrame([
            {"pollster": "Siena", "notes": "D=45% R=55%; LV; src=rcp"},
            {"pollster": "Quinnipiac", "notes": "D=47% R=53%; LV; src=rcp"},
        ])
        result = _compute_poll_confidence(df)
        # 2 pollsters satisfies the OR condition → Medium
        assert result.label == "Medium"

    def test_compute_confidence_one_pollster_two_methods_is_medium(self):
        import pandas as pd

        from api.routers.forecast.race_detail import _compute_poll_confidence

        df = pd.DataFrame([
            {"pollster": "Siena", "notes": "D=45% R=55%; LV; src=rcp"},
            {"pollster": "Siena", "notes": "D=44% R=56%; RV; src=rcp"},
        ])
        result = _compute_poll_confidence(df)
        # 1 pollster but 2 methods satisfies the OR condition → Medium
        assert result.label == "Medium"
        assert result.n_pollsters == 1
        assert result.n_methodologies == 2

    def test_compute_confidence_high_threshold(self):
        import pandas as pd

        from api.routers.forecast.race_detail import _compute_poll_confidence

        df = pd.DataFrame([
            {"pollster": "Siena", "notes": "D=45% R=55%; LV; src=rcp"},
            {"pollster": "Quinnipiac", "notes": "D=47% R=53%; RV; src=rcp"},
            {"pollster": "Emerson", "notes": "D=46% R=54%; LV; src=rcp"},
        ])
        result = _compute_poll_confidence(df)
        assert result.label == "High"
        assert result.n_pollsters == 3
        assert result.n_methodologies == 2

    def test_tooltip_singular_plural(self):
        import pandas as pd

        from api.routers.forecast.race_detail import _compute_poll_confidence

        df = pd.DataFrame([
            {"pollster": "Siena", "notes": "D=45% R=55%; LV; src=rcp"},
        ])
        result = _compute_poll_confidence(df)
        # singular forms for counts of 1
        assert "1 pollster" in result.tooltip
        assert "1 method" in result.tooltip
        assert "1 poll" in result.tooltip
