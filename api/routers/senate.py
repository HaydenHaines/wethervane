from __future__ import annotations

import duckdb
import pandas as pd
from fastapi import APIRouter, Depends, Request

from api.db import get_db

router = APIRouter(tags=["senate"])

# Seats not up for election in 2026 (Class I + Class III senators already decided)
DEM_SAFE_SEATS = 47
GOP_SAFE_SEATS = 53

# Current Senate composition: which party holds each state's seats.
# "D" = Democrat, "R" = Republican, "I" = Independent (caucuses with D).
# For map coloring, non-contested states show their overall delegation color.
SENATE_DELEGATION = {
    "AL": "R", "AK": "R", "AZ": "D", "AR": "R", "CA": "D", "CO": "D",
    "CT": "D", "DE": "D", "FL": "R", "GA": "D", "HI": "D", "ID": "R",
    "IL": "D", "IN": "R", "IA": "R", "KS": "R", "KY": "R", "LA": "R",
    "ME": "split", "MD": "D", "MA": "D", "MI": "D", "MN": "D", "MS": "R",
    "MO": "R", "MT": "R", "NE": "R", "NV": "D", "NH": "D", "NJ": "D",
    "NM": "D", "NY": "D", "NC": "R", "ND": "R", "OH": "R", "OK": "R",
    "OR": "D", "PA": "split", "RI": "D", "SC": "R", "SD": "R", "TN": "R",
    "TX": "R", "UT": "R", "VT": "D", "VA": "D", "WA": "D", "WV": "R",
    "WI": "split", "WY": "R", "DC": "D",
}

SENATE_2026_STATES = {
    "AL", "AK", "AR", "CO", "DE", "GA", "IA", "ID", "IL", "KS",
    "KY", "LA", "MA", "ME", "MI", "MN", "MS", "MT", "NC", "NE",
    "NH", "NJ", "NM", "OK", "OR", "RI", "SC", "SD", "TN", "TX",
    "VA", "WV", "WY",
}

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

    Uses tract-level type priors + behavior layer adjustment for off-cycle
    races. Produces vote-weighted state predictions per Senate race.
    """
    # Use tract priors + behavior layer for state-level predictions
    type_scores = getattr(request.app.state, "type_scores", None)
    type_priors = getattr(request.app.state, "type_priors", None)
    ridge_priors = getattr(request.app.state, "ridge_priors", {})
    tract_fips = getattr(request.app.state, "type_county_fips", [])
    tract_states = getattr(request.app.state, "tract_states", {})
    tract_votes = getattr(request.app.state, "tract_votes", {})
    behavior_tau = getattr(request.app.state, "behavior_tau", None)
    behavior_delta = getattr(request.app.state, "behavior_delta", None)

    if type_scores is None or not tract_fips:
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

    # Compute behavior-adjusted baseline predictions per state.
    # Apply τ+δ to tract priors for off-cycle races (Senate = off-cycle).
    import numpy as np
    priors = np.array([ridge_priors.get(f, 0.45) for f in tract_fips])

    if behavior_tau is not None and behavior_delta is not None and type_scores.shape[1] == len(behavior_tau):
        from src.behavior.voter_behavior import apply_behavior_adjustment
        adjusted = apply_behavior_adjustment(priors, type_scores, behavior_tau, behavior_delta, is_offcycle=True)
    else:
        adjusted = priors

    # Vote-weighted state predictions
    state_preds = {}
    for i, fips in enumerate(tract_fips):
        st = tract_states.get(fips)
        if not st:
            continue
        votes = tract_votes.get(fips, 0)
        if st not in state_preds:
            state_preds[st] = {"weighted_sum": 0.0, "total_votes": 0}
        state_preds[st]["weighted_sum"] += adjusted[i] * votes
        state_preds[st]["total_votes"] += votes

    for st in state_preds:
        tv = state_preds[st]["total_votes"]
        state_preds[st] = state_preds[st]["weighted_sum"] / tv if tv > 0 else 0.5

    # Build race list from SENATE_2026_STATES
    rows = []
    for st in sorted(SENATE_2026_STATES):
        race = f"2026 {st} Senate"
        state_pred = state_preds.get(st, 0.5)
        rows.append({"race": race, "state_pred": state_pred, "state_abbr": st})

    races = []
    for row in rows:
        race = row["race"]
        state_pred = row["state_pred"]
        state_abbr = row["state_abbr"]

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

    # Build state_colors map: every state gets a hex color for the map.
    # Contested states → rating-based color. Non-contested → delegation party color.
    # Dusty Ink palette
    _RATING_COLORS = {
        "safe_d": "#2d4a6f", "likely_d": "#4b6d90", "lean_d": "#7e9ab5",
        "tossup": "#b5a995",
        "lean_r": "#c4907a", "likely_r": "#9e5e4e", "safe_r": "#6e3535",
    }
    _PARTY_COLORS = {"D": "#4b6d90", "R": "#9e5e4e", "split": "#b5a995"}

    race_by_state = {r["state"]: r for r in races}
    state_colors = {}
    for st, delegation in SENATE_DELEGATION.items():
        if st in race_by_state:
            # Contested: use race rating color
            state_colors[st] = _RATING_COLORS.get(race_by_state[st]["rating"], "#b5a995")
        else:
            # Not contested in 2026: use delegation color (lighter shade)
            state_colors[st] = _PARTY_COLORS.get(delegation, "#eae7e2")

    return {
        "headline": headline,
        "subtitle": subtitle,
        "dem_seats_safe": DEM_SAFE_SEATS,
        "gop_seats_safe": GOP_SAFE_SEATS,
        "races": races,
        "state_colors": state_colors,
    }
