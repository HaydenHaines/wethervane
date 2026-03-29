from __future__ import annotations

import logging

import duckdb
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, Request

from api.db import get_db

log = logging.getLogger(__name__)

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
    # Count races where the model favors each party (tossups split as contested)
    dem_favored = sum(1 for r in races if r["margin"] > _TOSSUP_MAX)
    gop_favored = sum(1 for r in races if r["margin"] < -_TOSSUP_MAX)
    n_tossup = sum(1 for r in races if r["rating"] == "tossup")
    competitive = [r for r in races if r["rating"] in ("tossup", "lean")]
    n_competitive = len(competitive)

    # Seat projections: safe seats plus clearly-favored contested seats
    dem_projected = DEM_SAFE_SEATS + dem_favored
    gop_projected = GOP_SAFE_SEATS + gop_favored

    seat_diff = dem_projected - gop_projected

    if abs(seat_diff) <= 2:
        subtitle_parts = [f"{n_tossup} tossup" if n_tossup == 1 else f"{n_tossup} tossups"]
        if n_competitive > n_tossup:
            subtitle_parts.append(f"{n_competitive - n_tossup} more lean races")
        return (
            "Senate Control on a Knife's Edge",
            f"{' · '.join(subtitle_parts)} in play",
        )
    if gop_projected > dem_projected:
        if gop_projected >= 55:
            subtitle = f"GOP projected {gop_projected} seats · {n_competitive} competitive races"
            return "Republicans Strongly Favored to Hold the Senate", subtitle
        subtitle = f"GOP projected {gop_projected} seats · {n_competitive} competitive races"
        return "Republicans Favored to Hold the Senate", subtitle
    if gop_projected >= 55:
        subtitle = f"Dems projected {dem_projected} seats · {n_competitive} competitive races"
        return "Democrats Strongly Favored to Flip the Senate", subtitle
    subtitle = f"Dems projected {dem_projected} seats · {n_competitive} competitive races"
    return "Democrats Favored to Flip the Senate", subtitle


@router.get("/senate/overview")
def get_senate_overview(
    request: Request,
    db: duckdb.DuckDBPyConnection = Depends(get_db),
) -> dict:
    """Return the national Senate forecast summary for the landing page.

    Uses tract-level type priors + behavior layer adjustment for off-cycle
    races. Produces vote-weighted state predictions per Senate race.
    """
    # Load type model data from app.state
    type_scores = getattr(request.app.state, "type_scores", None)
    type_covariance = getattr(request.app.state, "type_covariance", None)
    type_priors = getattr(request.app.state, "type_priors", None)
    ridge_priors = getattr(request.app.state, "ridge_priors", {})
    tract_fips = getattr(request.app.state, "type_county_fips", [])
    tract_states_map = getattr(request.app.state, "tract_states", {})
    tract_votes_map = getattr(request.app.state, "tract_votes", {})
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

    # Fetch all senate polls grouped by race
    try:
        polls_df = db.execute(
            """
            SELECT race, geography AS state_abbr, dem_share, n_sample
            FROM polls
            WHERE LOWER(race) LIKE '%senate%'
            """
        ).fetchdf()
    except Exception:
        polls_df = pd.DataFrame()

    polls_by_race: dict[str, list[tuple[float, int, str]]] = {}
    for _, row in polls_df.iterrows():
        race_key = str(row["race"])
        polls_by_race.setdefault(race_key, []).append(
            (float(row["dem_share"]), int(row["n_sample"]), str(row["state_abbr"]))
        )

    # Build tract-level arrays (shared across all races)
    states_list = [tract_states_map.get(f, f[:2]) for f in tract_fips]
    county_names = [f"Tract {f}" for f in tract_fips]
    county_priors = np.array([ridge_priors.get(f, 0.45) for f in tract_fips])

    # Apply behavior adjustment for off-cycle (Senate = off-cycle)
    if (behavior_tau is not None and behavior_delta is not None
            and type_scores.shape[1] == len(behavior_tau)):
        from src.behavior.voter_behavior import apply_behavior_adjustment
        county_priors = apply_behavior_adjustment(
            county_priors, type_scores, behavior_tau, behavior_delta, is_offcycle=True
        )

    # Precompute vote-weighted state priors for races without polls
    state_prior_preds: dict[str, float] = {}
    state_accum: dict[str, tuple[float, float]] = {}  # {st: (weighted_sum, total_votes)}
    for i, fips in enumerate(tract_fips):
        st = tract_states_map.get(fips)
        if not st:
            continue
        votes = tract_votes_map.get(fips, 0)
        ws, tv = state_accum.get(st, (0.0, 0))
        state_accum[st] = (ws + county_priors[i] * votes, tv + votes)
    for st, (ws, tv) in state_accum.items():
        state_prior_preds[st] = ws / tv if tv > 0 else 0.5

    # For races WITH polls, run full Bayesian update via predict_race
    from src.prediction.predict_2026_types import predict_race

    rows = []
    for st in sorted(SENATE_2026_STATES):
        race = f"2026 {st} Senate"
        race_polls = polls_by_race.get(race, [])
        n_polls = len(race_polls)

        if race_polls and type_covariance is not None:
            # Full Bayesian update through type covariance
            result_df = predict_race(
                race=race,
                polls=race_polls,
                type_scores=type_scores,
                type_covariance=type_covariance,
                type_priors=type_priors,
                county_fips=tract_fips,
                states=states_list,
                county_names=county_names,
                county_priors=county_priors,
                prior_weight=1.0,
            )
            # Vote-weighted state prediction
            state_rows = result_df[result_df["state"] == st]
            if not state_rows.empty:
                weights = state_rows["county_fips"].map(
                    lambda f: tract_votes_map.get(f, 0)
                ).values.astype(float)
                total_w = weights.sum()
                if total_w > 0:
                    state_pred = float(
                        (state_rows["pred_dem_share"].values * weights).sum() / total_w
                    )
                else:
                    state_pred = float(state_rows["pred_dem_share"].mean())
            else:
                state_pred = 0.5
            log.info("Senate %s: predict_race with %d polls -> %.3f", st, n_polls, state_pred)
        else:
            # No polls: use behavior-adjusted prior
            state_pred = state_prior_preds.get(st, 0.5)

        rows.append({"race": race, "state_pred": state_pred, "state_abbr": st})

    races = []
    for row in rows:
        race = row["race"]
        state_pred = row["state_pred"]
        state_abbr = row["state_abbr"]

        margin = state_pred - 0.5
        rating = _margin_to_rating(margin)
        slug = race.lower().replace(" ", "-")
        n_polls = len(polls_by_race.get(race, []))

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

    # Freshness: report the most recent poll date scraped, if available
    updated_at: str | None = None
    try:
        row = db.execute(
            "SELECT MAX(date) AS max_date FROM polls WHERE date IS NOT NULL"
        ).fetchone()
        if row and row[0]:
            updated_at = str(row[0])
    except Exception:
        pass

    return {
        "headline": headline,
        "subtitle": subtitle,
        "dem_seats_safe": DEM_SAFE_SEATS,
        "gop_seats_safe": GOP_SAFE_SEATS,
        "races": races,
        "state_colors": state_colors,
        "updated_at": updated_at,
    }
