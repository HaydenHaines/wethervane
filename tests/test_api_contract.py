"""Integration tests: validate DuckDB->API->frontend contract.

These tests use the REAL wethervane.duckdb to catch pipeline/API mismatches.
Skip gracefully if the DB file doesn't exist (CI without data).
"""
from pathlib import Path

import duckdb
import pytest
from fastapi.testclient import TestClient

DB_PATH = Path("data/wethervane.duckdb")

pytestmark = pytest.mark.skipif(
    not DB_PATH.exists(),
    reason="data/wethervane.duckdb not found — skip contract integration tests",
)


@pytest.fixture(scope="module")
def real_client():
    """TestClient backed by the real wethervane.duckdb."""
    from api.main import create_app

    app = create_app()  # uses real lifespan
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def real_db():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    yield con
    con.close()


# ── Schema tests ──────────────────────────────────────────────────────────

def test_duckdb_contract(real_db):
    """Required tables and columns exist in wethervane.duckdb."""
    from src.db.build_database import validate_contract

    errors = validate_contract(real_db)
    assert errors == [], f"Contract violations: {errors}"


# ── API response shape tests ──────────────────────────────────────────────

def test_super_types_response(real_client):
    resp = real_client.get("/api/v1/super-types")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) > 0, "super-types must not be empty"
    for st in data:
        assert "super_type_id" in st
        assert "display_name" in st
        assert isinstance(st["display_name"], str)
        assert len(st["display_name"]) > 0


def test_counties_reference_valid_super_types(real_client):
    super_types = {st["super_type_id"] for st in real_client.get("/api/v1/super-types").json()}
    counties = real_client.get("/api/v1/counties").json()
    for c in counties:
        if c["super_type"] is not None:
            assert c["super_type"] in super_types, \
                f"County {c['county_fips']} has super_type={c['super_type']} not in super-types"


def test_forecast_has_state_abbr(real_client):
    rows = real_client.get("/api/v1/forecast").json()
    if not rows:
        pytest.skip("No forecast data")
    for r in rows:
        assert "state_abbr" in r
        assert isinstance(r["state_abbr"], str)
        assert len(r["state_abbr"]) == 2


def test_type_detail_has_dynamic_dicts(real_client):
    types = real_client.get("/api/v1/types").json()
    if not types:
        pytest.skip("No types in database")
    detail = real_client.get(f"/api/v1/types/{types[0]['type_id']}").json()
    assert isinstance(detail["demographics"], dict)
    if detail["shift_profile"] is not None:
        assert isinstance(detail["shift_profile"], dict)
        # Verify no non-numeric metadata leaked into shift_profile
        for key, val in detail["shift_profile"].items():
            assert isinstance(val, (int, float)), \
                f"shift_profile[{key}] is {type(val).__name__}, expected number"


# ── Cross-layer consistency ───────────────────────────────────────────────

def test_super_type_coverage(real_client):
    super_type_ids = {st["super_type_id"] for st in real_client.get("/api/v1/super-types").json()}
    county_super_types = {c["super_type"] for c in real_client.get("/api/v1/counties").json() if c["super_type"] is not None}
    assert county_super_types <= super_type_ids, \
        f"Counties reference super-types not in API: {county_super_types - super_type_ids}"
    unused = super_type_ids - county_super_types
    if unused:
        import warnings
        warnings.warn(f"Super-types with no counties: {unused}")


def test_health_reports_contract_status(real_client):
    resp = real_client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "contract" in data
    assert data["contract"] in ("ok", "degraded")


def test_app_state_reconstruction_shapes(tmp_path):
    """Verify app.state arrays have correct shapes after loading from DuckDB."""
    import duckdb
    import numpy as np
    from api.main import _load_type_data_from_db

    J = 4
    N = 3

    con = duckdb.connect(":memory:")
    con.execute("""
        CREATE TABLE type_scores (county_fips VARCHAR, type_id INTEGER, score FLOAT, version_id VARCHAR)
    """)
    for fips in ["12001", "12003", "13001"]:
        for j in range(J):
            con.execute("INSERT INTO type_scores VALUES (?,?,?,?)", [fips, j, 1.0/J, "v1"])

    con.execute("""
        CREATE TABLE type_covariance (type_i INTEGER, type_j INTEGER, value FLOAT, version_id VARCHAR)
    """)
    for i in range(J):
        for j in range(J):
            con.execute("INSERT INTO type_covariance VALUES (?,?,?,?)", [i, j, float(i==j)*0.01, "v1"])

    con.execute("""
        CREATE TABLE type_priors (type_id INTEGER, mean_dem_share FLOAT, version_id VARCHAR)
    """)
    for j in range(J):
        con.execute("INSERT INTO type_priors VALUES (?,?,?)", [j, 0.45, "v1"])

    scores, fips_list, covariance, priors = _load_type_data_from_db(con, "v1")
    assert scores.shape == (N, J)
    assert covariance.shape == (J, J)
    assert priors.shape == (J,)
    assert len(fips_list) == N
