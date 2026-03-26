# api/main.py
"""FastAPI application factory for the WetherVane API."""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import communities, counties, forecast, meta

log = logging.getLogger(__name__)


def _get_current_version_id(db: duckdb.DuckDBPyConnection) -> str:
    row = db.execute(
        "SELECT version_id FROM model_versions WHERE role = 'current' LIMIT 1"
    ).fetchone()
    if row:
        return row[0]
    # Fallback: latest alphabetically
    row = db.execute(
        "SELECT version_id FROM model_versions ORDER BY version_id DESC LIMIT 1"
    ).fetchone()
    return row[0] if row else "unknown"


def _load_sigma(db: duckdb.DuckDBPyConnection, version_id: str) -> tuple[np.ndarray, int]:
    """Load K×K sigma matrix from community_sigma table. Returns (sigma, K)."""
    rows = db.execute(
        """SELECT community_id_row, community_id_col, sigma_value
           FROM community_sigma WHERE version_id = ?
           ORDER BY community_id_row, community_id_col""",
        [version_id],
    ).fetchdf()
    if rows.empty:
        log.warning("community_sigma table is empty for version %s", version_id)
        return np.eye(1), 1
    K = int(max(rows["community_id_row"].max(), rows["community_id_col"].max())) + 1
    sigma = np.zeros((K, K))
    for _, r in rows.iterrows():
        sigma[int(r["community_id_row"]), int(r["community_id_col"])] = float(r["sigma_value"])
    return sigma, K


def _load_mu_prior(db: duckdb.DuckDBPyConnection, version_id: str, K: int) -> np.ndarray:
    """Compute community-level prior from predictions table (mean pred_dem_share per community)."""
    rows = db.execute(
        """SELECT ca.community_id, AVG(p.pred_dem_share) as mean_share
           FROM predictions p
           JOIN community_assignments ca
             ON p.county_fips = ca.county_fips AND p.version_id = ca.version_id
           WHERE p.version_id = ?
           GROUP BY ca.community_id
           ORDER BY ca.community_id""",
        [version_id],
    ).fetchdf()
    mu = np.full(K, 0.45)
    for _, r in rows.iterrows():
        idx = int(r["community_id"])
        if 0 <= idx < K:
            mu[idx] = float(r["mean_share"])
    return mu


def _load_type_data_from_db(
    db: duckdb.DuckDBPyConnection, version_id: str
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray] | tuple[None, None, None, None]:
    """Load type scores, fips list, covariance, and priors from DuckDB.

    Returns (type_scores, type_county_fips, type_covariance, type_priors)
    or (None, None, None, None) if the tables are empty for this version.
    """
    scores_df = db.execute(
        "SELECT county_fips, type_id, score FROM type_scores WHERE version_id=? ORDER BY county_fips, type_id",
        [version_id],
    ).fetchdf()
    if scores_df.empty:
        return None, None, None, None

    # Pivot: rows=county_fips, cols=type_id (0-indexed contiguous)
    pivot = scores_df.pivot(index="county_fips", columns="type_id", values="score").sort_index()
    pivot = pivot[sorted(pivot.columns)]  # ensure column order 0..J-1
    type_scores = pivot.values.astype(float)
    type_county_fips = pivot.index.tolist()
    J = type_scores.shape[1]

    # Covariance J×J
    cov_df = db.execute(
        "SELECT type_i, type_j, value FROM type_covariance WHERE version_id=? ORDER BY type_i, type_j",
        [version_id],
    ).fetchdf()
    cov_pivot = cov_df.pivot(index="type_i", columns="type_j", values="value").sort_index()
    cov_pivot = cov_pivot[sorted(cov_pivot.columns)]
    type_covariance = cov_pivot.values[:J, :J].astype(float)

    # Priors J-vector
    priors_df = db.execute(
        "SELECT type_id, mean_dem_share FROM type_priors WHERE version_id=? ORDER BY type_id",
        [version_id],
    ).fetchdf()
    type_priors = priors_df["mean_dem_share"].values[:J].astype(float)

    return type_scores, type_county_fips, type_covariance, type_priors


def _load_ridge_priors_from_db(db: duckdb.DuckDBPyConnection, version_id: str) -> dict[str, float]:
    """Load ridge county priors as a dict from DuckDB."""
    df = db.execute(
        "SELECT county_fips, ridge_pred_dem_share FROM ridge_county_priors WHERE version_id=?",
        [version_id],
    ).fetchdf()
    if df.empty:
        return {}
    return dict(zip(df["county_fips"], df["ridge_pred_dem_share"]))


def _load_hac_weights_from_db(
    db: duckdb.DuckDBPyConnection, version_id: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruct state_weights and county_weights DataFrames from DuckDB.

    Returns DataFrames matching the shape that _forecast_poll_hac expects:
    state_weights has columns [state_abbr, community_0, community_1, ...]
    county_weights has columns [county_fips, community_id]
    """
    sw_df = db.execute(
        "SELECT state_abbr, community_id, weight FROM hac_state_weights WHERE version_id=?",
        [version_id],
    ).fetchdf()
    if sw_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    state_weights = sw_df.pivot(index="state_abbr", columns="community_id", values="weight").reset_index()
    state_weights.columns = ["state_abbr"] + [f"community_{i}" for i in state_weights.columns[1:]]

    # `_forecast_poll_hac` only reads county_fips → community_id mapping.
    # Other columns that may exist in the source parquet (state_fips, etc.)
    # are intentionally excluded here.
    cw_df = db.execute(
        "SELECT county_fips, community_id FROM hac_county_weights WHERE version_id=?",
        [version_id],
    ).fetchdf()
    return state_weights, cw_df


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open DB and load in-memory data at startup; close at shutdown."""
    data_dir = Path(os.environ.get("WETHERVANE_DATA_DIR", "data"))
    db_path = Path(os.environ.get("WETHERVANE_DB_PATH", str(data_dir / "wethervane.duckdb")))

    log.info("Opening DuckDB at %s (read_only=True)", db_path)
    # Store the path so each request can open its own connection (avoids DuckDB
    # concurrent-query cancellation when Promise.all fires multiple requests).
    app.state.db_path = str(db_path)

    # Use a short-lived startup connection only for loading in-memory data.
    startup_db = duckdb.connect(str(db_path), read_only=True)

    version_id = _get_current_version_id(startup_db)
    app.state.version_id = version_id
    log.info("Using model version: %s", version_id)

    sigma, K = _load_sigma(startup_db, version_id)
    app.state.sigma = sigma
    app.state.K = K
    log.info("Loaded sigma matrix (%d×%d)", K, K)

    app.state.mu_prior = _load_mu_prior(startup_db, version_id, K)
    log.info("Loaded mu_prior: %s", app.state.mu_prior.round(3))

    db = startup_db

    # ── Type-primary model data ─────────────────────────────────────────────
    try:
        (
            app.state.type_scores,
            app.state.type_county_fips,
            app.state.type_covariance,
            app.state.type_priors,
        ) = _load_type_data_from_db(db, version_id)
        if app.state.type_scores is not None:
            J = app.state.type_scores.shape[1]
            log.info("Loaded type data: %d counties × %d types", app.state.type_scores.shape[0], J)
        else:
            log.warning("No type data in DB for version %s", version_id)
    except Exception as exc:
        log.warning("Could not load type data from DB: %s", exc)
        app.state.type_scores = None
        app.state.type_county_fips = None
        app.state.type_covariance = None
        app.state.type_priors = None

    # ── Ridge county priors ──────────────────────────────────────────────────
    try:
        app.state.ridge_priors = _load_ridge_priors_from_db(db, version_id)
        log.info("Loaded ridge priors: %d counties", len(app.state.ridge_priors))
    except Exception as exc:
        log.warning("Could not load ridge priors from DB: %s", exc)
        app.state.ridge_priors = {}

    # ── HAC fallback weights ─────────────────────────────────────────────────
    try:
        app.state.state_weights, app.state.county_weights = _load_hac_weights_from_db(db, version_id)
        log.info("Loaded HAC weights: %d states", len(app.state.state_weights))
    except Exception as exc:
        log.warning("Could not load HAC weights from DB: %s", exc)
        app.state.state_weights = pd.DataFrame()
        app.state.county_weights = pd.DataFrame()

    # ── Contract check ─────────────────────────────────────────────────────────
    contract_ok = True
    for table_name in ["super_types", "types", "county_type_assignments"]:
        try:
            result = startup_db.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
                [table_name],
            ).fetchone()
            if not result or result[0] == 0:
                log.warning("CONTRACT: missing table %s — frontend will show degraded state", table_name)
                contract_ok = False
        except Exception:
            contract_ok = False
    app.state.contract_ok = contract_ok
    log.info("Contract status: %s", "ok" if contract_ok else "degraded")

    startup_db.close()
    log.info("Startup DuckDB connection closed")

    yield


def create_app(lifespan_override=None) -> FastAPI:
    """Factory function. Pass lifespan_override=_noop_lifespan in tests to skip DB startup."""
    app = FastAPI(
        title="WetherVane API",
        description="Electoral community model data for WetherVane",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        lifespan=lifespan_override if lifespan_override is not None else lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Tighten after launch
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.include_router(meta.router, prefix="/api/v1")
    app.include_router(communities.router, prefix="/api/v1")
    app.include_router(counties.router, prefix="/api/v1")
    app.include_router(forecast.router, prefix="/api/v1")

    return app


app = create_app()  # Production app: uses real lifespan


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
