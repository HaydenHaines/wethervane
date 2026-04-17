# api/routers/candidates.py
"""Sabermetrics candidate endpoints: badges, CTOV, and race-level candidate lookups.

Data is loaded once at import time from static JSON/Parquet files produced by
the sabermetrics pipeline (Phases 1-2).  These files are small (<1MB total)
and change only when the pipeline is re-run.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import duckdb
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request

from api.db import get_db
from api.models import (
    CandidateBadge,
    CandidateBadgesResponse,
    CandidateListItem,
    CandidateListResponse,
    CTOVEntry,
    CTOVResponse,
    PredecessorInfo,
    RaceCandidatesResponse,
    RaceCandidateSummary,
    RaceResult,
)

log = logging.getLogger(__name__)

router = APIRouter(tags=["candidates"])

# ── Static data loaded at import time ────────────────────────────────────────

_DATA_DIR = Path(__file__).parent.parent.parent / "data"
_SABERMETRICS_DIR = _DATA_DIR / "sabermetrics"
_CANDIDATES_2026_PATH = _DATA_DIR / "config" / "candidates_2026.json"

# Badge data: keyed by bioguide ID
_BADGES: dict[str, dict] = {}
_BADGES_PATH = _SABERMETRICS_DIR / "candidate_badges.json"
if _BADGES_PATH.exists():
    with _BADGES_PATH.open() as f:
        _BADGES = json.load(f)
    log.info("Loaded %d candidate badge profiles", len(_BADGES))

# CTOV data: DataFrame with per-candidate, per-race, per-type overperformance
_CTOV: pd.DataFrame = pd.DataFrame()
_CTOV_PATH = _SABERMETRICS_DIR / "candidate_ctov.parquet"
if _CTOV_PATH.exists():
    _CTOV = pd.read_parquet(_CTOV_PATH)
    log.info("Loaded CTOV data: %d rows x %d cols", *_CTOV.shape)

# Candidate registry: maps bioguide IDs to names, parties, races
_REGISTRY: dict[str, dict] = {}
_REGISTRY_PATH = _SABERMETRICS_DIR / "candidate_registry.json"
if _REGISTRY_PATH.exists():
    with _REGISTRY_PATH.open() as f:
        raw = json.load(f)
    _REGISTRY = raw.get("persons", raw) if isinstance(raw, dict) else {}
    log.info("Loaded candidate registry: %d persons", len(_REGISTRY))

# Candidates 2026: race-level candidate data for linking to sabermetrics
_CANDIDATES_2026: dict[str, dict] = {}
if _CANDIDATES_2026_PATH.exists():
    with _CANDIDATES_2026_PATH.open() as f:
        raw_2026 = json.load(f)
    for race_type_key in ("senate", "governor"):
        if race_type_key in raw_2026:
            _CANDIDATES_2026.update(raw_2026[race_type_key])
    log.info("Loaded %d race entries from candidates_2026.json", len(_CANDIDATES_2026))

# ── Name-to-bioguide reverse index ──────────────────────────────────────────
# candidates_2026.json lacks bioguide IDs, so we match by exact name against
# the registry.  This is built once at import time.

_NAME_TO_BIOGUIDE: dict[str, str] = {}
for _pid, _pdata in _REGISTRY.items():
    _name = _pdata.get("name", "")
    if _name:
        _NAME_TO_BIOGUIDE[_name] = _pid


def _resolve_bioguide(candidate_name: str) -> str | None:
    """Resolve a candidate display name to a bioguide ID via the registry.

    Handles the common case where candidates_2026.json includes parenthetical
    annotations like 'Katie Britt (appointed interim)' by stripping them.
    """
    clean = candidate_name.split("(")[0].strip()
    return _NAME_TO_BIOGUIDE.get(clean)


# ── Super-type display names (cached) ───────────────────────────────────────

_TYPE_DISPLAY_NAMES: dict[int, str] | None = None


def _get_type_display_names(db: duckdb.DuckDBPyConnection) -> dict[int, str]:
    """Fetch per-type display names from DuckDB ``types`` table, cached after first call.

    Uses the individual type display names (J=100 types), not super-type names.
    The cache persists for the process lifetime since type names only change on
    retrain (at which point the server is restarted).
    """
    global _TYPE_DISPLAY_NAMES  # noqa: PLW0603
    if _TYPE_DISPLAY_NAMES is not None:
        return _TYPE_DISPLAY_NAMES

    try:
        rows = db.execute("SELECT type_id, display_name FROM types").fetchall()
        _TYPE_DISPLAY_NAMES = {int(r[0]): str(r[1]) for r in rows}
    except Exception:
        log.warning("Could not load type display names from DuckDB")
        _TYPE_DISPLAY_NAMES = {}
    return _TYPE_DISPLAY_NAMES


def _build_badges(badge_data: dict) -> list[CandidateBadge]:
    """Convert raw badge data into a list of CandidateBadge models.

    Reads from badge_details when available (includes provisional, kind, type_id,
    fallback_reason).  Falls back to the legacy badges list + badge_scores for
    artifacts produced before those fields existed.
    """
    details = badge_data.get("badge_details")
    if details:
        scores = badge_data.get("badge_scores", {})
        return [
            CandidateBadge(
                name=d["name"],
                score=d.get("score", scores.get(d["name"], 0.0)),
                provisional=d.get("provisional", False),
                kind=d.get("kind", "catalog"),
                type_id=d.get("type_id"),
                fallback_reason=d.get("fallback_reason"),
            )
            for d in details
        ]
    # Legacy fallback: artifact pre-dates badge_details
    badge_names = badge_data.get("badges", [])
    scores = badge_data.get("badge_scores", {})
    result = []
    for name in badge_names:
        score = scores.get(name, None)
        if score is None and name.startswith("Low "):
            score = scores.get(name[4:], 0.0)
        result.append(CandidateBadge(name=name, score=score or 0.0))
    return result


# ── Predecessor lookup ───────────────────────────────────────────────────────


def _find_predecessor(
    bioguide_id: str,
    state: str,
    office: str,
    party: str,
    before_year: int,
) -> dict | None:
    """Find the most recent prior candidate in the same state/office/party.

    Searches ``_REGISTRY`` for a different person who ran in the same state and
    office for the same party in a year strictly earlier than ``before_year``.
    Returns the registry entry dict for the closest predecessor, or None.

    Single-race candidates use this to show a weak second data point — the CTOV
    shift between the current candidate and their predecessor.  Low-trust signal
    only; the caller must label it clearly in the UI.
    """
    best_year: int = -1
    best_entry: dict | None = None
    best_id: str | None = None

    for pid, pdata in _REGISTRY.items():
        if pid == bioguide_id:
            continue
        for race in pdata.get("races", []):
            if (
                race.get("state") == state
                and race.get("office") == office
                and race.get("party") == party
                and race.get("year", 0) < before_year
                and race.get("year", 0) > best_year
            ):
                best_year = race["year"]
                best_entry = pdata
                best_id = pid

    if best_id is None or best_entry is None:
        return None

    return {
        "bioguide_id": best_id,
        "name": best_entry.get("name", ""),
        "year": best_year,
    }


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.get(
    "/candidates",
    response_model=CandidateListResponse,
    summary="List candidates with optional filtering",
)
def list_candidates(
    q: str | None = None,
    party: str | None = None,
    office: str | None = None,
    year: int | None = None,
    state: str | None = None,
    limit: int = 200,
    offset: int = 0,
) -> CandidateListResponse:
    """Return a filtered, paginated list of candidates from the badge registry.

    All query parameters are optional and additive (AND logic):
    - ``q``: case-insensitive substring match on candidate name
    - ``party``: exact match on party code ('D', 'R', 'I', etc.)
    - ``office``: exact match on office type ('Senate', 'Governor')
    - ``year``: candidate must have run in this election year
    - ``state``: candidate must have run in this state (2-letter code)

    Only candidates present in candidate_badges.json are included; pure registry
    entries without badge data are excluded since the profile page requires badges.
    """
    items: list[CandidateListItem] = []

    for bioguide_id, badge_data in _BADGES.items():
        name = badge_data.get("name", "")
        cand_party = badge_data.get("party", "")

        # Party filter — exact match
        if party and cand_party != party:
            continue

        # Name search — case-insensitive substring
        if q and q.lower() not in name.lower():
            continue

        # Registry data for office/year/state filtering
        reg = _REGISTRY.get(bioguide_id, {})
        races = reg.get("races", [])

        # Office filter — candidate must have run for this office at least once
        if office and not any(r.get("office") == office for r in races):
            continue

        # Year filter — candidate must have run in this year
        if year and not any(r.get("year") == year for r in races):
            continue

        # State filter — candidate must have run in this state
        if state and not any(r.get("state") == state for r in races):
            continue

        # Build summary fields from registry races
        states = sorted({r["state"] for r in races if "state" in r})
        offices = sorted({r["office"] for r in races if "office" in r})
        years = sorted({r["year"] for r in races if "year" in r})
        badge_names = [b.get("name", "") for b in badge_data.get("badge_details", [])] or \
                      badge_data.get("badges", [])

        items.append(
            CandidateListItem(
                bioguide_id=bioguide_id,
                name=name,
                party=cand_party,
                n_races=badge_data.get("n_races", 0),
                cec=badge_data.get("cec", 0.0),
                badges=badge_names,
                states=states,
                offices=offices,
                years=years,
            )
        )

    # Sort alphabetically by last name (last space-separated token), then first name
    items.sort(key=lambda c: (c.name.split()[-1].lower(), c.name.lower()))

    total = len(items)
    page = items[offset : offset + limit]
    return CandidateListResponse(candidates=page, total=total)


@router.get(
    "/candidates/{bioguide_id}",
    response_model=CandidateBadgesResponse,
    summary="Full candidate sabermetrics profile",
)
def get_candidate(bioguide_id: str) -> CandidateBadgesResponse:
    """Return badges, badge scores, CEC, and race history for a candidate by bioguide ID."""
    if bioguide_id not in _BADGES:
        raise HTTPException(status_code=404, detail=f"Candidate {bioguide_id} not found")

    data = _BADGES[bioguide_id]

    # Pull race history from the registry (richer than badge data alone)
    reg = _REGISTRY.get(bioguide_id, {})
    raw_races = reg.get("races", [])
    races = [
        RaceResult(
            year=r["year"],
            state=r["state"],
            office=r["office"],
            special=r.get("special", False),
            party=r.get("party", data.get("party", "")),
            result=r.get("result", "unknown"),
            actual_dem_share_2party=r.get("actual_dem_share_2party"),
        )
        for r in sorted(raw_races, key=lambda x: x.get("year", 0), reverse=True)
    ]

    return CandidateBadgesResponse(
        bioguide_id=bioguide_id,
        name=data["name"],
        party=data["party"],
        n_races=data["n_races"],
        badges=_build_badges(data),
        badge_scores=data.get("badge_scores", {}),
        cec=data.get("cec", 0.0),
        races=races,
    )


@router.get(
    "/candidates/{bioguide_id}/predecessor",
    response_model=PredecessorInfo | None,
    summary="Find the predecessor candidate in the same state/office/party slot",
)
def get_candidate_predecessor(bioguide_id: str) -> PredecessorInfo | None:
    """Return the most recent predecessor for single-race candidates.

    A predecessor is the last person of the same party who ran in the same
    state and office before the current candidate's first race.  Used to give
    single-race candidates a weak cross-cycle comparison signal.

    Returns 404 if the candidate is not found.
    Returns null (HTTP 200 with null body) if no predecessor exists or the
    candidate has more than one race (predecessor comparison not needed).
    """
    if bioguide_id not in _REGISTRY:
        raise HTTPException(status_code=404, detail=f"Candidate {bioguide_id} not found")

    reg = _REGISTRY[bioguide_id]
    races = reg.get("races", [])

    # Only return predecessor for single-race candidates — multi-race candidates
    # already have their own cross-cycle consistency data.
    if len(races) != 1:
        return None

    race = races[0]
    predecessor = _find_predecessor(
        bioguide_id=bioguide_id,
        state=race["state"],
        office=race["office"],
        party=race.get("party", reg.get("party", "")),
        before_year=race["year"],
    )

    if predecessor is None:
        return None

    return PredecessorInfo(
        bioguide_id=predecessor["bioguide_id"],
        name=predecessor["name"],
        year=predecessor["year"],
    )


@router.get(
    "/candidates/{bioguide_id}/ctov",
    response_model=CTOVResponse,
    summary="CTOV vector for a candidate (top 10 types by absolute value)",
)
def get_candidate_ctov(
    request: Request,
    bioguide_id: str,
    year: int | None = None,
    db: duckdb.DuckDBPyConnection = Depends(get_db),
):
    """Return the top 10 community types by absolute CTOV for a candidate.

    If year is specified, returns CTOV for that specific race.
    Otherwise returns the most recent race available.
    """
    if _CTOV.empty:
        raise HTTPException(status_code=404, detail="CTOV data not available")

    # Filter to this candidate
    mask = _CTOV["person_id"] == bioguide_id
    if mask.sum() == 0:
        raise HTTPException(status_code=404, detail=f"No CTOV data for {bioguide_id}")

    candidate_rows = _CTOV[mask]
    if year is not None:
        candidate_rows = candidate_rows[candidate_rows["year"] == year]
        if candidate_rows.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No CTOV data for {bioguide_id} in {year}",
            )

    # Take the most recent race
    row = candidate_rows.sort_values("year", ascending=False).iloc[0]

    # Extract CTOV columns (ctov_type_0 .. ctov_type_99)
    ctov_cols = [c for c in _CTOV.columns if c.startswith("ctov_type_")]
    ctov_values: dict[int, float] = {}
    for col in ctov_cols:
        type_id = int(col.replace("ctov_type_", ""))
        val = float(row[col])
        if pd.notna(val):
            ctov_values[type_id] = val

    # Sort by absolute value and take top 10
    sorted_types = sorted(ctov_values.items(), key=lambda x: abs(x[1]), reverse=True)[:10]

    # Resolve display names from DuckDB super_types table
    type_names = _get_type_display_names(db)

    entries = [
        CTOVEntry(
            type_id=tid,
            display_name=type_names.get(tid, f"Type {tid}"),
            ctov=val,
        )
        for tid, val in sorted_types
    ]

    return CTOVResponse(
        bioguide_id=bioguide_id,
        name=str(row["name"]),
        party=str(row["party"]),
        year=int(row["year"]),
        state=str(row["state"]),
        office=str(row["office"]),
        entries=entries,
    )


@router.get(
    "/races/{race_key:path}/candidates",
    response_model=RaceCandidatesResponse,
    summary="All candidates with badges for a race",
)
def get_race_candidates(race_key: str):
    """Return badge summaries for all candidates in a given race.

    race_key uses the same format as candidates_2026.json keys,
    e.g. '2026 GA Senate'.  The endpoint accepts URL-encoded spaces
    or dashes (converted to spaces).
    """
    # Normalize: convert dashes to spaces for slug-style keys
    normalized_key = race_key.replace("-", " ")

    # Try exact match first, then case-insensitive
    race_data = _CANDIDATES_2026.get(normalized_key)
    if race_data is None:
        # Try case-insensitive
        for k, v in _CANDIDATES_2026.items():
            if k.lower() == normalized_key.lower():
                race_data = v
                normalized_key = k
                break

    if race_data is None:
        raise HTTPException(status_code=404, detail=f"Race '{race_key}' not found")

    candidates_out: list[RaceCandidateSummary] = []

    # Collect all candidate names from the race
    all_candidates: list[tuple[str, str]] = []  # (name, party)
    for party, names in race_data.get("candidates", {}).items():
        for name in names:
            all_candidates.append((name, party))

    # Also check the incumbent (they might not be listed under candidates)
    incumbent = race_data.get("incumbent", {})
    if isinstance(incumbent, dict) and incumbent.get("name"):
        inc_name = incumbent["name"]
        inc_party = incumbent.get("party", "")
        # Avoid duplicates
        already_listed = any(
            inc_name.split("(")[0].strip() == n.split("(")[0].strip()
            for n, _ in all_candidates
        )
        if not already_listed:
            all_candidates.append((inc_name, inc_party))

    for name, party in all_candidates:
        bioguide = _resolve_bioguide(name)
        if bioguide and bioguide in _BADGES:
            badge_data = _BADGES[bioguide]
            candidates_out.append(
                RaceCandidateSummary(
                    bioguide_id=bioguide,
                    name=badge_data["name"],
                    party=badge_data["party"],
                    badges=_build_badges(badge_data),
                    cec=badge_data.get("cec", 0.0),
                    badge_scores=badge_data.get("badge_scores", {}),
                )
            )

    return RaceCandidatesResponse(
        race_key=normalized_key,
        candidates=candidates_out,
    )
