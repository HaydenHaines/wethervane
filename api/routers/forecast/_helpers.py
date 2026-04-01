"""Shared helpers for the forecast router package.

Contains slug conversion, historical results loading, pollster grade lookup,
margin-to-rating conversion, and baseline label formatting.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import Request

# Path to the static historical results data file (lives alongside the api/ package)
_HISTORICAL_RESULTS_PATH = Path(__file__).parent.parent.parent / "data" / "historical_results.json"


def _load_historical_results() -> dict:
    """Load and return the historical results dict from disk.

    Returns an empty dict when the file is missing (graceful degradation).
    Strips comment keys (those starting with '_') used for documentation.
    """
    if not _HISTORICAL_RESULTS_PATH.exists():
        return {}
    with _HISTORICAL_RESULTS_PATH.open() as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


# Loaded once at import time - this file changes only when race data is manually updated
_HISTORICAL_RESULTS: dict = _load_historical_results()

# Uncertainty model parameters — see docs/ARCHITECTURE.md for calibration notes
_STATE_STD_FLOOR = 0.035      # minimum state-level std; prevents over-confidence when counties agree
_STATE_STD_CAP = 0.15         # hard cap; beyond this, the race is essentially a coin flip
_STATE_STD_FALLBACK = 0.065   # used when poll-derived std is unavailable
_MATRIX_JITTER = 1e-8         # Tikhonov regularization keeps covariance PD during matrix inversion
_Z90 = 1.645                  # z-score for 90% confidence interval


def race_to_slug(race: str) -> str:
    """Convert race label to URL slug. "2026 FL Governor" → "2026-fl-governor"."""
    return race.lower().replace(" ", "-")


def slug_to_race(slug: str) -> str:
    """Convert URL slug back to race label. "2026-fl-governor" → "2026 FL Governor"."""
    parts = slug.split("-")
    if len(parts) < 3:
        return slug
    year = parts[0]
    state = parts[1].upper()
    race_type = " ".join(p.capitalize() for p in parts[2:])
    return f"{year} {state} {race_type}"


def _lookup_pollster_grade(request: Request, pollster_name: str | None) -> str | None:
    """Look up Silver Bulletin letter grade for a pollster, with fuzzy matching."""
    if not pollster_name:
        return None
    grades = getattr(request.app.state, "pollster_grades", {})
    norm_grades = getattr(request.app.state, "pollster_grades_normalized", {})
    if not grades:
        return None
    # Exact match
    if pollster_name in grades:
        return grades[pollster_name]
    # Normalized match
    from src.assembly.silver_bulletin_ratings import _normalize, _name_similarity
    norm = _normalize(pollster_name)
    if norm in norm_grades:
        return norm_grades[norm]
    # Fuzzy match (Jaccard > 0.4)
    best_grade, best_sim = None, 0.0
    for nk, grade in norm_grades.items():
        sim = _name_similarity(norm, nk)
        if sim > best_sim:
            best_sim = sim
            best_grade = grade
    return best_grade if best_sim >= 0.4 else None


def _format_baseline_label(pres_baseline: float) -> str:
    """Format the presidential baseline as a party-margin label, e.g. 'R+3.2' or 'D+0.5'.

    The label measures how far the 2024 presidential Dem share deviates from 50/50.
    shift = pres_baseline - 0.5; negative shift → Republican advantage → 'R+X'.
    """
    shift = pres_baseline - 0.5
    magnitude = round(abs(shift) * 100, 1)
    if shift < 0:
        return f"R+{magnitude}"
    return f"D+{magnitude}"


def marginToRating(dem_share: float) -> str:
    """Python equivalent of the frontend marginToRating for API use."""
    margin = dem_share - 0.5
    abs_margin = abs(margin)
    if abs_margin < 0.03:
        return "tossup"
    if margin > 0:
        if abs_margin >= 0.15:
            return "safe_d"
        if abs_margin >= 0.08:
            return "likely_d"
        return "lean_d"
    if abs_margin >= 0.15:
        return "safe_r"
    if abs_margin >= 0.08:
        return "likely_r"
    return "lean_r"
