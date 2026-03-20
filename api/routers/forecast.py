from __future__ import annotations

import numpy as np
import pandas as pd
import duckdb
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from api.db import get_db
from api.models import ForecastRow, PollInput

router = APIRouter(tags=["forecast"])


@router.get("/forecast", response_model=list[ForecastRow])
def get_forecast(
    request: Request,
    db: duckdb.DuckDBPyConnection = Depends(get_db),
    race: str | None = Query(None, description="Filter by race (e.g. FL_Senate)"),
    state: str | None = Query(None, description="Filter by state abbreviation (e.g. FL)"),
):
    version_id = request.app.state.version_id

    conditions = ["p.version_id = ?"]
    params: list = [version_id]

    if race:
        conditions.append("p.race = ?")
        params.append(race)
    if state:
        conditions.append("c.state_abbr = ?")
        params.append(state)

    where = " AND ".join(conditions)

    rows = db.execute(
        f"""
        SELECT
            p.county_fips,
            c.county_name,
            c.state_abbr,
            p.race,
            p.pred_dem_share,
            p.pred_std,
            p.pred_lo90,
            p.pred_hi90,
            p.state_pred,
            p.poll_avg
        FROM predictions p
        JOIN counties c ON p.county_fips = c.county_fips
        WHERE {where}
        ORDER BY p.race, c.state_abbr, p.county_fips
        """,
        params,
    ).fetchdf()

    if rows.empty:
        return []

    return [
        ForecastRow(
            county_fips=row["county_fips"],
            county_name=row["county_name"] if row["county_name"] else None,
            state_abbr=row["state_abbr"],
            race=row["race"],
            pred_dem_share=None if pd.isna(row["pred_dem_share"]) else float(row["pred_dem_share"]),
            pred_std=None if pd.isna(row["pred_std"]) else float(row["pred_std"]),
            pred_lo90=None if pd.isna(row["pred_lo90"]) else float(row["pred_lo90"]),
            pred_hi90=None if pd.isna(row["pred_hi90"]) else float(row["pred_hi90"]),
            state_pred=None if pd.isna(row["state_pred"]) else float(row["state_pred"]),
            poll_avg=None if pd.isna(row["poll_avg"]) else float(row["poll_avg"]),
        )
        for _, row in rows.iterrows()
    ]


@router.post("/forecast/poll", response_model=list[ForecastRow])
def update_forecast_with_poll(
    poll: PollInput,
    request: Request,
    db: duckdb.DuckDBPyConnection = Depends(get_db),
):
    """Run a Bayesian update: given a new poll, return updated county predictions."""
    sigma = request.app.state.sigma
    K = request.app.state.K
    mu_prior = request.app.state.mu_prior
    state_weights = request.app.state.state_weights
    county_weights = request.app.state.county_weights

    if state_weights.empty or county_weights.empty:
        raise HTTPException(status_code=503, detail="Weight matrices not loaded")

    # Build W row for this state
    weight_cols = sorted([c for c in state_weights.columns if c.startswith("community_")])
    state_row = state_weights[state_weights["state_abbr"] == poll.state]
    if state_row.empty:
        raise HTTPException(status_code=404, detail=f"State {poll.state} not found in weights")

    W = state_row[weight_cols].values  # shape (1, K)
    y = np.array([poll.dem_share])
    sigma_poll = np.array([np.sqrt(poll.dem_share * (1 - poll.dem_share) / poll.n)])

    # Bayesian update
    R = np.diag(sigma_poll ** 2)
    sigma_inv = np.linalg.inv(sigma + np.eye(K) * 1e-8)
    sigma_post_inv = sigma_inv + W.T @ np.linalg.inv(R) @ W
    sigma_post = np.linalg.inv(sigma_post_inv)
    mu_post = sigma_post @ (sigma_inv @ mu_prior + W.T @ np.linalg.solve(R, y))

    # Map community posteriors → county predictions via hard assignment
    # county_weights has columns: county_fips, community_id, state_fips, recent_total
    version_id = request.app.state.version_id
    county_info = db.execute(
        """SELECT c.county_fips, c.county_name, c.state_abbr, ca.community_id
           FROM counties c
           JOIN community_assignments ca ON c.county_fips = ca.county_fips AND ca.version_id = ?
           ORDER BY c.county_fips""",
        [version_id],
    ).fetchdf()

    state_pred_val = float((W @ mu_post).item())
    community_stds = np.sqrt(np.diag(sigma_post))

    results = []
    for _, row in county_info.iterrows():
        cid = int(row["community_id"])
        if 0 <= cid < K:
            pred = float(mu_post[cid])
            std = float(community_stds[cid])
        else:
            pred = float(np.mean(mu_post))
            std = 0.05
        results.append(
            ForecastRow(
                county_fips=row["county_fips"],
                county_name=row["county_name"] if row["county_name"] else None,
                state_abbr=row["state_abbr"],
                race=poll.race,
                pred_dem_share=pred,
                pred_std=std,
                pred_lo90=pred - 1.645 * std,
                pred_hi90=pred + 1.645 * std,
                state_pred=state_pred_val,
                poll_avg=poll.dem_share,
            )
        )

    return results
