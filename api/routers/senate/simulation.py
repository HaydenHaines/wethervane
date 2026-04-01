"""GET /senate/chamber-probability — Monte Carlo chamber control simulation."""
from __future__ import annotations

import logging

import duckdb
import numpy as np
from fastapi import APIRouter, Depends, Query, Request

from api.db import get_db
from api.models import ChamberProbabilityResponse, SeatDistributionBucket
from api.routers.senate._helpers import (
    DEM_SAFE_SEATS,
    SENATE_2026_STATES,
    _CLASS_II_INCUMBENT,
    _DEM_HOLDOVER_SEATS,
)

log = logging.getLogger(__name__)

router = APIRouter(tags=["senate"])

# State-level uncertainty parameters -- mirrors the values in forecast.py.
# Keeping them here avoids a cross-router import while ensuring both routers
# use the same calibration.  See forecast.py docs for calibration notes.
_STATE_STD_FLOOR = 0.035      # minimum state-level std
_STATE_STD_CAP = 0.15         # hard cap -- beyond this, the race is essentially a coin flip
_STATE_STD_FALLBACK = 0.065   # used when vote-weighted std is unavailable

# Uncertainty for "safe" seats: the model hasn't predicted these, but we treat
# the incumbent as a strong favorite.  A 5pp std (sigma=0.05) centered at +/-0.25
# from 50% gives roughly a 95% win probability -- clearly safe without being
# a certainty (a small upset tail keeps the simulation honest).
_SAFE_SEAT_STD = 0.05

# Seats in the distribution output: filter to a window around the median
# to keep the payload small while covering >99% of the probability mass.
_DISTRIBUTION_SEAT_RANGE = range(35, 66)  # 35-65 seats covers all realistic outcomes

# Number of Monte Carlo draws.  10,000 gives ~0.5pp standard error on a
# 50/50 probability estimate -- more than enough for display purposes.
_N_SIMULATIONS = 10_000


def _build_empty_response(
    total_dem: int,
    n_sims: int,
) -> ChamberProbabilityResponse:
    """Build a deterministic response when no races can be simulated."""
    return ChamberProbabilityResponse(
        dem_control_pct=100.0 if total_dem >= 50 else 0.0,
        rep_control_pct=100.0 if total_dem < 51 else 0.0,
        dem_majority_pct=100.0 if total_dem >= 51 else 0.0,
        median_dem_seats=total_dem,
        median_rep_seats=100 - total_dem,
        seat_distribution=[SeatDistributionBucket(seats=total_dem, probability=1.0)],
        n_simulations=n_sims,
        n_modeled_races=0,
        n_safe_races=0,
    )


def _run_simulations(
    all_preds: np.ndarray,
    all_stds: np.ndarray,
    dem_holdover: int,
    n_modeled: int,
    n_safe: int,
    n_sims: int,
    rng: np.random.Generator,
) -> ChamberProbabilityResponse:
    """Run Monte Carlo draws and compute chamber control probabilities."""
    n_total_races = len(all_preds)

    # Draw all simulations at once: shape (n_sims, n_total_races).
    draws = rng.normal(loc=all_preds, scale=all_stds, size=(n_sims, n_total_races))
    draws = np.clip(draws, 0.0, 1.0)

    # Dem wins per simulation: each race where draw > 0.5
    dem_wins_per_sim = np.sum(draws > 0.5, axis=1)
    total_dem_per_sim = dem_holdover + dem_wins_per_sim

    dem_control_pct = float(np.mean(total_dem_per_sim >= 50) * 100)
    dem_majority_pct = float(np.mean(total_dem_per_sim >= 51) * 100)
    rep_control_pct = float(np.mean(total_dem_per_sim < 50) * 100)

    median_dem = int(np.median(total_dem_per_sim))

    seat_distribution = []
    for seats in _DISTRIBUTION_SEAT_RANGE:
        prob = float(np.mean(total_dem_per_sim == seats))
        if prob > 0.0001:
            seat_distribution.append(
                SeatDistributionBucket(seats=seats, probability=round(prob, 4)),
            )

    return ChamberProbabilityResponse(
        dem_control_pct=round(dem_control_pct, 1),
        rep_control_pct=round(rep_control_pct, 1),
        dem_majority_pct=round(dem_majority_pct, 1),
        median_dem_seats=median_dem,
        median_rep_seats=100 - median_dem,
        seat_distribution=seat_distribution,
        n_simulations=n_sims,
        n_modeled_races=n_modeled,
        n_safe_races=n_safe,
    )


def _simulate_chamber_probability(
    modeled_races: list[tuple[float, float]],
    safe_dem_wins: int,
    safe_gop_wins: int,
    dem_holdover: int,
    n_sims: int,
    rng_seed: int | None = 42,
) -> ChamberProbabilityResponse:
    """Monte Carlo chamber control simulation.

    For each simulation:
    1. Draw each modeled race from Normal(pred, std) -- clip to [0, 1].
    2. Count Dem wins where draw > 0.5.
    3. Add safe Dem wins (high-confidence).
    4. Add holdover (not up in 2026) Dem seats.
    5. Total Dem seats = holdover + safe wins + modeled wins.

    Args:
        modeled_races: List of (dem_share_pred, pred_std) for contested races
            where the model has a prediction.
        safe_dem_wins: Number of D-held seats with no model prediction (treated
            as safe with _SAFE_SEAT_STD uncertainty).
        safe_gop_wins: Number of R-held seats with no model prediction.
        dem_holdover: Dem seats not up in 2026.
        n_sims: Number of Monte Carlo simulations.
        rng_seed: Optional fixed seed for reproducibility in tests.
    """
    rng = np.random.default_rng(rng_seed)

    n_modeled = len(modeled_races)
    n_safe = safe_dem_wins + safe_gop_wins

    if n_modeled > 0:
        preds = np.array([r[0] for r in modeled_races])
        stds = np.array([r[1] for r in modeled_races])
    else:
        preds = np.empty(0)
        stds = np.empty(0)

    # Safe Dem seats: center at 0.75 (strong favorite), small std
    safe_dem_preds = np.full(safe_dem_wins, 0.75)
    safe_dem_stds = np.full(safe_dem_wins, _SAFE_SEAT_STD)

    # Safe GOP seats: center at 0.25 (strong underdog), small std
    safe_gop_preds = np.full(safe_gop_wins, 0.25)
    safe_gop_stds = np.full(safe_gop_wins, _SAFE_SEAT_STD)

    all_preds = np.concatenate([preds, safe_dem_preds, safe_gop_preds])
    all_stds = np.concatenate([stds, safe_dem_stds, safe_gop_stds])

    if len(all_preds) == 0:
        return _build_empty_response(dem_holdover, n_sims)

    return _run_simulations(
        all_preds, all_stds, dem_holdover, n_modeled, n_safe, n_sims, rng,
    )


def _collect_race_data(
    db: duckdb.DuckDBPyConnection,
    version_id: str,
    mode_filter: str,
) -> tuple[list[tuple[float, float]], int, int]:
    """Collect per-state prediction data for the Monte Carlo simulation.

    Returns (modeled_races, safe_dem_wins, safe_gop_wins).
    """
    modeled_races: list[tuple[float, float]] = []
    safe_dem_wins = 0
    safe_gop_wins = 0

    for st in sorted(SENATE_2026_STATES):
        race = f"2026 {st} Senate"

        county_rows = db.execute(
            f"""
            SELECT
                p.pred_dem_share,
                p.pred_std,
                COALESCE(c.total_votes_2024, 0) AS votes
            FROM predictions p
            JOIN counties c ON p.county_fips = c.county_fips
            WHERE p.version_id = ?
              AND p.race = ?
              AND c.state_abbr = ?
              AND p.pred_dem_share IS NOT NULL
              {mode_filter}
            """,
            [version_id, race, st],
        ).fetchdf()

        if county_rows.empty:
            incumbent = _CLASS_II_INCUMBENT.get(st, "R")
            if incumbent == "D":
                safe_dem_wins += 1
            else:
                safe_gop_wins += 1
            continue

        state_pred, std = _compute_state_prediction(county_rows)
        modeled_races.append((state_pred, std))

    return modeled_races, safe_dem_wins, safe_gop_wins


def _compute_state_prediction(county_rows) -> tuple[float, float]:
    """Compute vote-weighted state prediction and uncertainty from county rows.

    Returns (state_pred, clamped_std).
    """
    preds_arr = county_rows["pred_dem_share"].values.astype(float)
    votes_arr = county_rows["votes"].values.astype(float)
    total_votes = votes_arr.sum()

    if total_votes > 0:
        state_pred = float(np.dot(preds_arr, votes_arr) / total_votes)
        weights = votes_arr / total_votes
        county_var = float(np.sum(weights * (preds_arr - state_pred) ** 2))
        n_eff = max(1.0, 1.0 / np.sum(weights ** 2))
        raw_std = float(np.sqrt(county_var / n_eff))
    else:
        state_pred = float(np.mean(preds_arr))
        raw_std = _STATE_STD_FALLBACK

    std = max(raw_std, _STATE_STD_FLOOR)
    std = min(std, _STATE_STD_CAP)
    return state_pred, std


@router.get("/senate/chamber-probability", response_model=ChamberProbabilityResponse)
def get_chamber_probability(
    request: Request,
    db: duckdb.DuckDBPyConnection = Depends(get_db),
    n_simulations: int = Query(
        _N_SIMULATIONS,
        ge=1000,
        le=100_000,
        description="Number of Monte Carlo simulations (default 10,000)",
    ),
) -> ChamberProbabilityResponse:
    """Compute Monte Carlo chamber control probability for the 2026 Senate.

    Loads vote-weighted state predictions for all 33 Class II seats.  For races
    with model predictions, draws each simulation from Normal(pred, std).  For
    races without predictions (safe seats), treats the incumbent as a strong
    favorite (pred=0.75/0.25, std=0.05).

    The Dem total in each simulation = holdover seats (not up in 2026) + wins
    from the 33 contested races.  Control probability is fraction of sims where
    Dems reach >=50 (with VP) or >=51 (outright majority).
    """
    version_id = getattr(request.app.state, "version_id", None)
    if not version_id:
        return _build_empty_response(DEM_SAFE_SEATS, n_simulations)

    _has_mode = "forecast_mode" in [
        row[0] for row in db.execute("DESCRIBE predictions").fetchall()
    ]
    _mode_filter = "AND p.forecast_mode = 'local'" if _has_mode else ""

    modeled_races, safe_dem_wins, safe_gop_wins = _collect_race_data(
        db, version_id, _mode_filter,
    )

    return _simulate_chamber_probability(
        modeled_races=modeled_races,
        safe_dem_wins=safe_dem_wins,
        safe_gop_wins=safe_gop_wins,
        dem_holdover=_DEM_HOLDOVER_SEATS,
        n_sims=n_simulations,
    )
