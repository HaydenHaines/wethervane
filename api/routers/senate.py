from __future__ import annotations

import duckdb
import pandas as pd
from fastapi import APIRouter, Depends, Request

from api.db import get_db

router = APIRouter(tags=["senate"])

# Seats not up for election in 2026 (Class I + Class III senators already decided)
DEM_SAFE_SEATS = 47
GOP_SAFE_SEATS = 53

# Rating margin thresholds (margin = dem_share - 0.5)
_TOSSUP_MAX = 0.03
_LEAN_MAX = 0.08
_LIKELY_MAX = 0.15


def _margin_to_rating(margin: float) -> str:
    """Convert signed Dem margin to a rating label.

    margin = state_pred - 0.5 (positive = Dem-favored, negative = GOP-favored)
    """
    abs_m = abs(margin)
    if abs_m < _TOSSUP_MAX:
        return "tossup"
    if abs_m < _LEAN_MAX:
        return "lean"
    if abs_m < _LIKELY_MAX:
        return "likely"
    return "safe"


def _rating_sort_key(rating: str) -> int:
    """Sort races with tossups first, safe last."""
    return {"tossup": 0, "lean": 1, "likely": 2, "safe": 3}.get(rating, 4)


def _build_headline(races: list[dict]) -> tuple[str, str]:
    """Derive a headline + subtitle from the current race ratings.

    Returns (headline, subtitle).
    """
    dem_leaning = sum(
        1 for r in races
        if r["margin"] > 0 or (r["rating"] == "tossup")
    )
    gop_leaning = len(races) - dem_leaning

    competitive = [r for r in races if r["rating"] in ("tossup", "lean")]
    n_tossup = sum(1 for r in races if r["rating"] == "tossup")

    if n_tossup >= 3:
        return "Senate Highly Competitive", "multiple tossup races in play"
    if gop_leaning > dem_leaning:
        return "Republicans Favored", "to retain control of the Senate"
    if dem_leaning > gop_leaning:
        return "Democrats Favored", "to flip Senate control"
    return "Senate Battle for Control", "outcome uncertain across competitive races"


@router.get("/senate/overview")
def get_senate_overview(
    request: Request,
    db: duckdb.DuckDBPyConnection = Depends(get_db),
) -> dict:
    """Return the national Senate forecast summary for the landing page.

    Queries all Senate races from the predictions table, computes vote-weighted
    state_pred per race, maps to a rating, and returns aggregate headline data.
    """
    version_id = request.app.state.version_id

    # Vote-weighted state prediction per Senate race
    rows = db.execute(
        """
        SELECT
            p.race,
            CASE WHEN SUM(COALESCE(c.total_votes_2024, 0)) > 0
                 THEN SUM(p.pred_dem_share * COALESCE(c.total_votes_2024, 0))
                      / SUM(COALESCE(c.total_votes_2024, 0))
                 ELSE AVG(p.pred_dem_share)
            END AS state_pred,
            MIN(c.state_abbr) AS state_abbr
        FROM predictions p
        JOIN counties c ON p.county_fips = c.county_fips
        WHERE p.version_id = ?
          AND LOWER(p.race) LIKE '%senate%'
        GROUP BY p.race
        ORDER BY p.race
        """,
        [version_id],
    ).fetchdf()

    if rows.empty:
        return {
            "headline": "No Senate Forecasts Available",
            "subtitle": "predictions not yet loaded",
            "dem_seats_safe": DEM_SAFE_SEATS,
            "gop_seats_safe": GOP_SAFE_SEATS,
            "races": [],
        }

    # Poll counts per race (case-insensitive match)
    try:
        poll_counts_df = db.execute(
            """
            SELECT race, COUNT(*) AS n_polls
            FROM polls
            WHERE LOWER(race) LIKE '%senate%'
            GROUP BY race
            """
        ).fetchdf()
        poll_counts = dict(zip(poll_counts_df["race"], poll_counts_df["n_polls"]))
    except Exception:
        poll_counts = {}

    races = []
    for _, row in rows.iterrows():
        race = row["race"]
        state_pred = float(row["state_pred"]) if not pd.isna(row["state_pred"]) else 0.5
        state_abbr = str(row["state_abbr"])

        margin = state_pred - 0.5
        rating = _margin_to_rating(margin)
        slug = race.lower().replace(" ", "-")

        # Poll lookup: try exact match first, then case-insensitive scan
        n_polls = int(poll_counts.get(race, 0))
        if n_polls == 0:
            for poll_race, count in poll_counts.items():
                if poll_race.lower() == race.lower():
                    n_polls = int(count)
                    break

        races.append({
            "state": state_abbr,
            "race": race,
            "slug": slug,
            "rating": rating,
            "margin": round(margin, 4),
            "n_polls": n_polls,
        })

    # Sort: tossups first, then lean, likely, safe; break ties alphabetically by state
    races.sort(key=lambda r: (_rating_sort_key(r["rating"]), r["state"]))

    headline, subtitle = _build_headline(races)

    return {
        "headline": headline,
        "subtitle": subtitle,
        "dem_seats_safe": DEM_SAFE_SEATS,
        "gop_seats_safe": GOP_SAFE_SEATS,
        "races": races,
    }
