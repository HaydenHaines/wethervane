"""Tests for the /api/v1/tracts/{state_abbr}/predictions endpoint."""
from __future__ import annotations

import duckdb
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI

from api.routers.tracts import router


def _make_app_with_db(con: duckdb.DuckDBPyConnection) -> FastAPI:
    """Build a minimal FastAPI app with the tracts router and a fixed DuckDB connection."""
    from api.db import get_db

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")

    # Override get_db dependency to use our test connection
    app.dependency_overrides[get_db] = lambda: con
    return app


def _seed_tract_predictions(con: duckdb.DuckDBPyConnection) -> None:
    """Insert a minimal set of tract_predictions rows into a fresh in-memory DB."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS tract_predictions (
            tract_geoid          VARCHAR NOT NULL,
            race                 VARCHAR NOT NULL,
            forecast_mode        VARCHAR NOT NULL DEFAULT 'national',
            pred_dem_share       DOUBLE,
            state_pred_dem_share DOUBLE,
            state                VARCHAR,
            PRIMARY KEY (tract_geoid, race, forecast_mode)
        )
    """)
    rows = [
        ("01001020100", "2026 AL Senate", "national", 0.40, 0.42, "AL"),
        ("01001020100", "2026 AL Senate", "local",    0.41, 0.42, "AL"),
        ("01001020200", "2026 AL Senate", "national", 0.35, 0.42, "AL"),
        ("06001400100", "2026 CA Senate", "national", 0.65, 0.67, "CA"),
    ]
    for r in rows:
        con.execute(
            "INSERT INTO tract_predictions VALUES (?, ?, ?, ?, ?, ?)",
            list(r),
        )


@pytest.fixture()
def client_with_data():
    """TestClient backed by an in-memory DB with tract_predictions seeded."""
    con = duckdb.connect(":memory:")
    _seed_tract_predictions(con)
    app = _make_app_with_db(con)
    with TestClient(app) as c:
        yield c
    con.close()


@pytest.fixture()
def client_no_table():
    """TestClient backed by an empty in-memory DB (no tract_predictions table)."""
    con = duckdb.connect(":memory:")
    app = _make_app_with_db(con)
    with TestClient(app) as c:
        yield c
    con.close()


def test_returns_predictions_for_valid_state(client_with_data):
    resp = client_with_data.get("/api/v1/tracts/AL/predictions")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 3  # 2 tracts × national + 1 local = 3
    geoids = {r["tract_geoid"] for r in data}
    assert "01001020100" in geoids
    assert "01001020200" in geoids


def test_returns_only_requested_state(client_with_data):
    resp = client_with_data.get("/api/v1/tracts/CA/predictions")
    assert resp.status_code == 200
    data = resp.json()
    assert all(r["state"] == "CA" for r in data)
    assert len(data) == 1


def test_filter_by_forecast_mode(client_with_data):
    resp = client_with_data.get("/api/v1/tracts/AL/predictions?forecast_mode=national")
    assert resp.status_code == 200
    data = resp.json()
    assert all(r["forecast_mode"] == "national" for r in data)
    assert len(data) == 2  # 2 AL tracts in national mode


def test_filter_forecast_mode_local(client_with_data):
    resp = client_with_data.get("/api/v1/tracts/AL/predictions?forecast_mode=local")
    assert resp.status_code == 200
    data = resp.json()
    assert all(r["forecast_mode"] == "local" for r in data)
    assert len(data) == 1


def test_state_case_insensitive(client_with_data):
    """State abbreviation should be normalised to uppercase."""
    resp = client_with_data.get("/api/v1/tracts/al/predictions")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 3


def test_unknown_state_returns_404(client_with_data):
    resp = client_with_data.get("/api/v1/tracts/XX/predictions")
    assert resp.status_code == 404


def test_missing_table_returns_503(client_no_table):
    """When tract_predictions table doesn't exist, the endpoint returns 503."""
    resp = client_no_table.get("/api/v1/tracts/AL/predictions")
    assert resp.status_code == 503


def test_response_fields_present(client_with_data):
    """Each response item has the expected fields."""
    resp = client_with_data.get("/api/v1/tracts/CA/predictions")
    assert resp.status_code == 200
    item = resp.json()[0]
    assert "tract_geoid" in item
    assert "race" in item
    assert "forecast_mode" in item
    assert "pred_dem_share" in item
    assert "state_pred_dem_share" in item
    assert "state" in item
