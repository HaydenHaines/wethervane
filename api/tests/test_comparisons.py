"""Tests for the /forecast/comparisons endpoint."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.tests.conftest import (
    TEST_K,
    TEST_VERSION,
    _build_test_db,
    _build_test_state,
    _noop_lifespan,
)


@pytest.fixture
def client():
    """Standard TestClient with synthetic data."""
    test_db = _build_test_db()
    state = _build_test_state()

    test_app = create_app(lifespan_override=_noop_lifespan)
    test_app.state.db = test_db
    test_app.state.version_id = TEST_VERSION
    test_app.state.K = TEST_K
    test_app.state.sigma = state["sigma"]
    test_app.state.mu_prior = state["mu_prior"]
    test_app.state.state_weights = state["state_weights"]
    test_app.state.county_weights = state["county_weights"]
    test_app.state.contract_ok = True
    rng = np.random.default_rng(42)
    _raw = rng.random((4, 4))
    _sym = (_raw + _raw.T) / 2
    np.fill_diagonal(_sym, 1.0)
    test_app.state.type_correlation = _sym

    with TestClient(test_app, raise_server_exceptions=True) as c:
        yield c

    test_db.close()


class TestComparisonsEndpoint:
    def test_returns_200(self, client):
        resp = client.get("/api/v1/forecast/comparisons")
        assert resp.status_code == 200

    def test_response_has_required_keys(self, client):
        resp = client.get("/api/v1/forecast/comparisons")
        data = resp.json()
        assert "races" in data
        assert "sources" in data
        assert isinstance(data["races"], list)
        assert isinstance(data["sources"], list)

    def test_race_rows_have_required_fields(self, client):
        resp = client.get("/api/v1/forecast/comparisons")
        data = resp.json()
        races = data["races"]
        # Should have at least the FL_Senate test race
        assert len(races) > 0
        row = races[0]
        for field in ("race_id", "slug", "year", "state_abbr", "race_type", "wethervane"):
            assert field in row, f"Missing field: {field}"
        wv = row["wethervane"]
        for wv_field in ("pred_dem_share", "pred_std", "rating", "n_counties"):
            assert wv_field in wv, f"Missing wethervane field: {wv_field}"

    def test_wethervane_predictions_are_valid(self, client):
        """Predictions should be floats in [0, 1] when present, or null."""
        resp = client.get("/api/v1/forecast/comparisons")
        races = resp.json()["races"]
        for row in races:
            pred = row["wethervane"]["pred_dem_share"]
            if pred is not None:
                assert 0.0 <= pred <= 1.0, f"Prediction out of range: {pred} for {row['race_id']}"

    def test_race_rows_have_slug(self, client):
        """Each race should have a URL slug derived from race_id."""
        resp = client.get("/api/v1/forecast/comparisons")
        for row in resp.json()["races"]:
            assert row["slug"], f"Empty slug for {row['race_id']}"
            assert " " not in row["slug"], f"Slug contains space: {row['slug']}"

    def test_handles_missing_comparisons_file_gracefully(self, client):
        """When the ratings file doesn't exist, endpoint should still return predictions.

        With no file on disk, all external ratings (cook, sabato, inside) must be None —
        the endpoint must not raise, and must not invent rating strings from thin air.
        """
        with patch(
            "api.routers.forecast.comparisons.COMPARISONS_FILE",
            Path("/nonexistent/path/ratings_2026.json"),
        ):
            resp = client.get("/api/v1/forecast/comparisons")
        assert resp.status_code == 200
        data = resp.json()
        # Should still return race rows from the DB
        assert "races" in data
        assert len(data["races"]) > 0
        # All external ratings must be None when the file is missing — not empty strings
        # and not fallback values. This verifies the patch actually changed behavior.
        for row in data["races"]:
            assert row["cook"] is None, (
                f"Expected cook=None when ratings file is missing, got {row['cook']!r}"
            )
            assert row.get("sabato") is None, (
                f"Expected sabato=None when ratings file is missing, got {row.get('sabato')!r}"
            )
            assert row.get("inside") is None, (
                f"Expected inside=None when ratings file is missing, got {row.get('inside')!r}"
            )

    def test_sources_list_from_file(self, client):
        """When ratings file exists, sources should be populated."""
        resp = client.get("/api/v1/forecast/comparisons")
        data = resp.json()
        # In test env, uses the real ratings_2026.json if it exists at the expected path
        # Just verify the structure is correct
        for src in data["sources"]:
            assert "id" in src
            assert "name" in src
            assert "url" in src

    def test_manual_ratings_merged_when_file_exists(self, client):
        """If a mock ratings file is injected, manual ratings should appear in response."""
        mock_ratings = {
            "last_updated": "2026-03-01",
            "sources": [
                {"id": "cook", "name": "Cook Political Report", "url": "https://cookpolitical.com/"}
            ],
            "ratings": {
                "FL_Senate": {
                    "cook": "Lean R",
                    "sabato": "Lean R",
                    "inside": "Tilt R",
                }
            },
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(mock_ratings, f)
            tmp_path = Path(f.name)

        try:
            with patch("api.routers.forecast.comparisons.COMPARISONS_FILE", tmp_path):
                resp = client.get("/api/v1/forecast/comparisons")
            assert resp.status_code == 200
            data = resp.json()
            assert data["last_updated"] == "2026-03-01"
            assert len(data["sources"]) == 1
            assert data["sources"][0]["id"] == "cook"
            # FL_Senate should have external ratings
            fl_senate = next(
                (r for r in data["races"] if r["race_id"] == "FL_Senate"), None
            )
            assert fl_senate is not None
            assert fl_senate["cook"] == "Lean R"
            assert fl_senate["sabato"] == "Lean R"
            assert fl_senate["inside"] == "Tilt R"
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_races_with_predictions_have_rating(self, client):
        """Races that have predictions should have a WetherVane rating assigned."""
        resp = client.get("/api/v1/forecast/comparisons")
        for row in resp.json()["races"]:
            pred = row["wethervane"]["pred_dem_share"]
            rating = row["wethervane"]["rating"]
            if pred is not None:
                assert rating is not None, f"Rating is null for race with pred: {row['race_id']}"
                assert rating in {
                    "safe_d", "likely_d", "lean_d", "tossup",
                    "lean_r", "likely_r", "safe_r",
                }, f"Unknown rating: {rating}"
