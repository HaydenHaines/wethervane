"""GET /forecast/seat-history — Senate seat balance over time.

Reads all forecast snapshots from ``data/forecast_snapshots/`` and reconstructs
the projected D/R seat count as of each snapshot date.  This produces a time
series that the frontend can display as a "Senate balance timeline" chart.

Seat-count logic mirrors the overview endpoint:
  - DEM_SAFE_SEATS / GOP_SAFE_SEATS are the not-up holdover totals.
  - For each contested race, a prediction > 0.5 counts as a Dem win, < 0.5 as GOP.
  - Tossup threshold is intentionally kept at exactly 0.5 here (not _TOSSUP_MAX)
    so that marginal changes in predictions are reflected in the time series rather
    than being collapsed into a flat "tossup" bucket.  The chart is showing
    projected seats, not a rating system.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter

from api.routers.senate._helpers import (
    DEM_SAFE_SEATS,
    GOP_SAFE_SEATS,
    SENATE_2026_STATES,
    _CLASS_II_INCUMBENT,
    _DEM_HOLDOVER_SEATS,
    _GOP_HOLDOVER_SEATS,
    _build_headline,
    classify_race,
)

log = logging.getLogger(__name__)

router = APIRouter(tags=["forecast"])

# Snapshots directory — same path as changelog.py uses.
SNAPSHOTS_DIR = Path(__file__).resolve().parents[3] / "data" / "forecast_snapshots"


def _compute_seat_counts(predictions: dict[str, float]) -> tuple[int, int]:
    """Compute projected D/R seat totals from a snapshot's predictions dict.

    Parameters
    ----------
    predictions:
        Mapping of race name (e.g. "2026 FL Senate") to avg_dem_share (0-1).

    Returns
    -------
    (dem_projected, gop_projected)
        Holdover seats plus the contested Class II seats the model predicts
        each party will win.  Races with predictions exactly at 0.5 are split
        50/50 (extremely rare in practice).
    """
    dem_class2 = 0
    gop_class2 = 0

    for st in SENATE_2026_STATES:
        race = f"2026 {st} Senate"
        pred = predictions.get(race)

        if pred is None:
            # No prediction in this snapshot — fall back to incumbent holds.
            incumbent = _CLASS_II_INCUMBENT.get(st, "R")
            if incumbent == "D":
                dem_class2 += 1
            else:
                gop_class2 += 1
        elif pred > 0.5:
            dem_class2 += 1
        else:
            gop_class2 += 1

    dem_projected = _DEM_HOLDOVER_SEATS + dem_class2
    gop_projected = _GOP_HOLDOVER_SEATS + gop_class2
    return dem_projected, gop_projected


@router.get("/forecast/seat-history")
def get_seat_history() -> list[dict]:
    """Return projected Senate seat counts over time from forecast snapshots.

    Each entry in the returned array represents one snapshot:
    ``{ "date": "2026-04-01", "dem_projected": 49, "gop_projected": 51 }``.

    Entries are sorted chronologically (oldest first).  If no snapshots exist,
    an empty list is returned.
    """
    if not SNAPSHOTS_DIR.exists():
        return []

    snapshot_files = sorted(SNAPSHOTS_DIR.glob("*.json"))
    if not snapshot_files:
        return []

    results: list[dict] = []

    for f in snapshot_files:
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            log.warning("seat_history: could not read snapshot %s — skipping", f.name)
            continue

        date = data.get("date")
        predictions = data.get("predictions", {})

        if not date:
            log.warning("seat_history: snapshot %s has no 'date' field — skipping", f.name)
            continue

        dem_projected, gop_projected = _compute_seat_counts(predictions)
        results.append({
            "date": date,
            "dem_projected": dem_projected,
            "gop_projected": gop_projected,
        })

    return results
