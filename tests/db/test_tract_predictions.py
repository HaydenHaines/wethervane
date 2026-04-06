"""Tests for tract_predictions DuckDB table ingestion and schema."""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest


def _make_tract_predictions_parquet(tmp_path: Path) -> Path:
    """Write a minimal tract_predictions parquet for testing."""
    predictions_dir = tmp_path / "data" / "predictions"
    predictions_dir.mkdir(parents=True)
    df = pd.DataFrame([
        {"tract_geoid": "01001020100", "state": "AL", "race": "2026 AL Senate", "forecast_mode": "national", "pred_dem_share": 0.40, "state_pred_dem_share": 0.42},
        {"tract_geoid": "01001020100", "state": "AL", "race": "2026 AL Senate", "forecast_mode": "local",    "pred_dem_share": 0.41, "state_pred_dem_share": 0.42},
        {"tract_geoid": "01001020200", "state": "AL", "race": "2026 AL Senate", "forecast_mode": "national", "pred_dem_share": 0.35, "state_pred_dem_share": 0.42},
        {"tract_geoid": "06001400100", "state": "CA", "race": "2026 CA Senate", "forecast_mode": "national", "pred_dem_share": 0.65, "state_pred_dem_share": 0.67},
    ])
    out = predictions_dir / "tract_predictions_2026.parquet"
    df.to_parquet(out)
    return out


def _make_db_with_schema() -> duckdb.DuckDBPyConnection:
    """Create an in-memory DuckDB with the tract_predictions schema applied."""
    con = duckdb.connect(":memory:")
    from src.db.schema import create_schema
    create_schema(con)
    return con


def test_tract_predictions_schema_created():
    """tract_predictions table is created by create_schema with correct columns."""
    con = _make_db_with_schema()
    cols = set(con.execute("SELECT * FROM tract_predictions LIMIT 0").fetchdf().columns)
    assert "tract_geoid" in cols
    assert "race" in cols
    assert "forecast_mode" in cols
    assert "pred_dem_share" in cols
    assert "state_pred_dem_share" in cols
    assert "state" in cols
    con.close()


def test_ingest_tract_predictions_inserts_rows(tmp_path: Path):
    """_ingest_tract_predictions loads all rows from parquet into DuckDB."""
    parquet_path = _make_tract_predictions_parquet(tmp_path)
    con = _make_db_with_schema()

    from src.db.ingest import _ingest_tract_predictions
    paths = {"tract_predictions": parquet_path}
    _ingest_tract_predictions(con, paths)

    n = con.execute("SELECT COUNT(*) FROM tract_predictions").fetchone()[0]
    assert n == 4, f"Expected 4 rows, got {n}"
    con.close()


def test_ingest_tract_predictions_skips_if_missing(tmp_path: Path):
    """_ingest_tract_predictions is a no-op when parquet file doesn't exist."""
    con = _make_db_with_schema()

    from src.db.ingest import _ingest_tract_predictions
    paths = {"tract_predictions": tmp_path / "nonexistent.parquet"}
    _ingest_tract_predictions(con, paths)  # should not raise

    n = con.execute("SELECT COUNT(*) FROM tract_predictions").fetchone()[0]
    assert n == 0
    con.close()


def test_ingest_tract_predictions_skips_if_key_missing():
    """_ingest_tract_predictions gracefully handles missing 'tract_predictions' key."""
    con = _make_db_with_schema()

    from src.db.ingest import _ingest_tract_predictions
    _ingest_tract_predictions(con, {})  # no 'tract_predictions' key

    n = con.execute("SELECT COUNT(*) FROM tract_predictions").fetchone()[0]
    assert n == 0
    con.close()


def test_ingest_tract_predictions_idempotent(tmp_path: Path):
    """Running _ingest_tract_predictions twice produces the same row count (DELETE + INSERT)."""
    parquet_path = _make_tract_predictions_parquet(tmp_path)
    con = _make_db_with_schema()
    paths = {"tract_predictions": parquet_path}

    from src.db.ingest import _ingest_tract_predictions
    _ingest_tract_predictions(con, paths)
    _ingest_tract_predictions(con, paths)

    n = con.execute("SELECT COUNT(*) FROM tract_predictions").fetchone()[0]
    assert n == 4, f"Expected 4 rows after double ingest, got {n}"
    con.close()


def test_tract_predictions_contract_validates_columns():
    """validate_contract recognises tract_predictions as optional and checks its columns."""
    con = duckdb.connect(":memory:")
    # Build bare-minimum required tables to pass contract
    con.execute("CREATE TABLE counties (county_fips VARCHAR, state_abbr VARCHAR, county_name VARCHAR, total_votes_2024 INTEGER)")
    con.execute("INSERT INTO counties VALUES ('12001', 'FL', 'Alachua', NULL)")
    con.execute("CREATE TABLE super_types (super_type_id INTEGER, display_name VARCHAR)")
    con.execute("INSERT INTO super_types VALUES (0, 'Test')")
    con.execute("CREATE TABLE types (type_id INTEGER, super_type_id INTEGER, display_name VARCHAR)")
    con.execute("INSERT INTO types VALUES (0, 0, 'Test Type')")
    con.execute("CREATE TABLE county_type_assignments (county_fips VARCHAR, dominant_type INTEGER, super_type INTEGER)")
    con.execute("INSERT INTO county_type_assignments VALUES ('12001', 0, 0)")
    con.execute("CREATE TABLE tract_type_assignments (tract_geoid VARCHAR PRIMARY KEY, dominant_type INTEGER, super_type INTEGER)")
    con.execute("INSERT INTO tract_type_assignments VALUES ('01001020100', 0, 0)")
    con.execute("CREATE TABLE type_scores (county_fips VARCHAR, type_id INTEGER, score FLOAT, version_id VARCHAR)")
    con.execute("CREATE TABLE type_priors (type_id INTEGER, mean_dem_share FLOAT, version_id VARCHAR)")
    con.execute("CREATE TABLE polls (poll_id VARCHAR, race VARCHAR, geography VARCHAR, geo_level VARCHAR, dem_share FLOAT, n_sample INTEGER, cycle VARCHAR)")
    con.execute("CREATE TABLE poll_crosstabs (poll_id VARCHAR NOT NULL, demographic_group VARCHAR NOT NULL, group_value VARCHAR NOT NULL, dem_share FLOAT, n_sample INTEGER, pct_of_sample FLOAT)")

    from src.db.validate import validate_contract

    # Without tract_predictions: should pass (it's optional)
    errors = validate_contract(con)
    assert not any("tract_predictions" in e for e in errors), f"Unexpected errors: {errors}"

    # With tract_predictions missing required column: should fail
    con.execute("CREATE TABLE tract_predictions (tract_geoid VARCHAR)")  # missing race, forecast_mode, pred_dem_share
    errors = validate_contract(con)
    assert any("MISSING COLUMN: tract_predictions.race" in e for e in errors)
    assert any("MISSING COLUMN: tract_predictions.forecast_mode" in e for e in errors)
    assert any("MISSING COLUMN: tract_predictions.pred_dem_share" in e for e in errors)
    con.close()
