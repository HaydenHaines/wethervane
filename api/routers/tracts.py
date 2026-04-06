"""Tract-level predictions API endpoints.

Provides access to the tract_predictions table, which contains forward predictions
for all ~80K census tracts across all 2026 races and forecast modes.
"""
from __future__ import annotations

import duckdb
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from api.db import get_db

router = APIRouter(tags=["tracts"])


class TractPrediction(BaseModel):
    tract_geoid: str
    race: str
    forecast_mode: str
    pred_dem_share: float | None
    state_pred_dem_share: float | None
    state: str | None


@router.get(
    "/tracts/{state_abbr}/predictions",
    response_model=list[TractPrediction],
    summary="Get tract predictions for a state",
    description=(
        "Returns 2026 forecast predictions for all census tracts in the given state. "
        "Optionally filter by forecast_mode ('national' or 'local'). "
        "Tracts are keyed by their 11-digit GEOID."
    ),
)
def get_state_tract_predictions(
    state_abbr: str,
    forecast_mode: str | None = Query(
        default=None,
        description="Filter by forecast mode: 'national' or 'local'. Returns all modes if omitted.",
    ),
    db: duckdb.DuckDBPyConnection = Depends(get_db),
) -> list[TractPrediction]:
    """Return tract-level predictions for all races in the specified state."""
    # Validate that tract_predictions table exists (table is optional — built when parquet exists)
    exists = db.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'tract_predictions'"
    ).fetchone()[0]
    if not exists:
        raise HTTPException(
            status_code=503,
            detail="tract_predictions table not built — run build_database.py to populate",
        )

    state_upper = state_abbr.upper()

    if forecast_mode is not None:
        rows = db.execute(
            """
            SELECT tract_geoid, race, forecast_mode, pred_dem_share, state_pred_dem_share, state
            FROM tract_predictions
            WHERE state = ? AND forecast_mode = ?
            ORDER BY tract_geoid, race
            """,
            [state_upper, forecast_mode],
        ).fetchdf()
    else:
        rows = db.execute(
            """
            SELECT tract_geoid, race, forecast_mode, pred_dem_share, state_pred_dem_share, state
            FROM tract_predictions
            WHERE state = ?
            ORDER BY tract_geoid, race, forecast_mode
            """,
            [state_upper],
        ).fetchdf()

    if rows.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No tract predictions found for state '{state_upper}'",
        )

    return [
        TractPrediction(
            tract_geoid=row["tract_geoid"],
            race=row["race"],
            forecast_mode=row["forecast_mode"],
            pred_dem_share=float(row["pred_dem_share"]) if row["pred_dem_share"] is not None else None,
            state_pred_dem_share=float(row["state_pred_dem_share"]) if row["state_pred_dem_share"] is not None else None,
            state=row["state"],
        )
        for _, row in rows.iterrows()
    ]
