"""Pollster quality scoring: RMSE-based accuracy, Silver Bulletin ratings, 538 grade fallback.

Assigns a quality multiplier to each poll based on pollster reputation.
Priority order when computing the multiplier for a given poll:
  1. RMSE-based accuracy data (from pollster_accuracy.json backtest results):
     Lower empirical RMSE → higher weight.  ``multiplier = clamp(median_rmse /
     pollster_rmse, RMSE_MIN, RMSE_MAX)``.  Fuzzy name matching handles minor
     spelling differences between the accuracy file and the poll CSV.
  2. Silver Bulletin ratings (XLSX): score in [0, 1] rescaled to [SB_MIN, SB_MAX].
  3. 538 numeric grade embedded in poll_notes (format: "...;grade=2.5;...").

Quality multipliers are applied multiplicatively to effective sample size,
so a 1.2x multiplier makes a poll count as 20% larger and a 0.3x multiplier
makes it count as 70% smaller.
"""

from __future__ import annotations

import json
import logging
import statistics
from copy import copy
from pathlib import Path

from src.propagation.propagate_polls import PollObservation

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RMSE-based quality weighting
# ---------------------------------------------------------------------------

# Multiplier bounds for RMSE-based weighting.
# The median pollster gets 1.0.  A pollster with half the median RMSE gets
# RMSE_MAX (capped); a pollster with twice the median RMSE gets ~0.5, floored
# at RMSE_MIN.
_RMSE_MIN_MULTIPLIER: float = 0.3
_RMSE_MAX_MULTIPLIER: float = 1.5

# Fuzzy matching: accept a pollster name match if the ratio of matching tokens
# to total unique tokens exceeds this threshold (Jaccard-like).
_FUZZY_MATCH_THRESHOLD: float = 0.5

# Cached RMSE lookup table: pollster (lowercase) → rmse_pp.
# None = not yet loaded; empty dict = loaded but file not found.
_RMSE_CACHE: dict[str, float] | None = None
_RMSE_MEDIAN: float | None = None


def _tokenize(name: str) -> set[str]:
    """Split a pollster name into lowercase word tokens for fuzzy matching."""
    import re
    return set(re.findall(r"[a-z0-9]+", name.lower()))


def _fuzzy_match_pollster(name: str, lookup: dict[str, float]) -> float | None:
    """Find the RMSE for *name* using fuzzy token matching against *lookup*.

    The lookup keys are already lowercased.  We compute a Jaccard-like
    similarity between the token sets of *name* and each lookup key, and
    return the RMSE of the best match if it exceeds ``_FUZZY_MATCH_THRESHOLD``.
    Returns None when no match meets the threshold.
    """
    query_tokens = _tokenize(name)
    if not query_tokens:
        return None

    best_score = 0.0
    best_rmse: float | None = None

    for key, rmse in lookup.items():
        key_tokens = _tokenize(key)
        if not key_tokens:
            continue
        intersection = len(query_tokens & key_tokens)
        union = len(query_tokens | key_tokens)
        score = intersection / union if union > 0 else 0.0
        if score > best_score:
            best_score = score
            best_rmse = rmse

    if best_score >= _FUZZY_MATCH_THRESHOLD:
        return best_rmse
    return None


def _load_rmse_lookup(accuracy_path: Path) -> tuple[dict[str, float], float | None]:
    """Load pollster→rmse_pp mapping from the accuracy JSON.

    Returns (lookup, median_rmse) where lookup maps lowercased pollster
    names to their rmse_pp values.  Returns ({}, None) if the file is
    missing or malformed.
    """
    if not accuracy_path.exists():
        log.debug("Pollster accuracy file not found: %s", accuracy_path)
        return {}, None

    try:
        with accuracy_path.open(encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        log.warning("Failed to load pollster accuracy file: %s", exc)
        return {}, None

    pollsters = data.get("pollsters", [])
    if not pollsters:
        return {}, None

    lookup: dict[str, float] = {}
    for entry in pollsters:
        name = entry.get("pollster", "").strip()
        rmse = entry.get("rmse_pp")
        if name and isinstance(rmse, (int, float)) and rmse > 0:
            lookup[name.lower()] = float(rmse)

    if not lookup:
        return {}, None

    median_rmse = statistics.median(lookup.values())
    return lookup, median_rmse


def _rmse_to_multiplier(pollster_rmse: float, median_rmse: float) -> float:
    """Convert a pollster's empirical RMSE to a quality multiplier.

    Formula: ``median_rmse / pollster_rmse``, clamped to [RMSE_MIN, RMSE_MAX].
    A pollster at the median gets exactly 1.0.  Better-than-median (lower RMSE)
    gets above 1.0 (up to RMSE_MAX).  Worse-than-median gets below 1.0 (down to
    RMSE_MIN).
    """
    if pollster_rmse <= 0 or median_rmse <= 0:
        return 1.0
    raw = median_rmse / pollster_rmse
    return max(_RMSE_MIN_MULTIPLIER, min(_RMSE_MAX_MULTIPLIER, raw))


def get_rmse_quality(
    pollster_name: str,
    accuracy_path: Path,
) -> float | None:
    """Return an RMSE-based quality multiplier for *pollster_name*.

    Loads (and caches per-path) the accuracy JSON.  Uses fuzzy token matching
    to handle minor spelling differences.  Returns None when the pollster is
    not found in the accuracy data (so the caller can fall back to grade-based
    weighting).
    """
    global _RMSE_CACHE, _RMSE_MEDIAN

    # Lazy-load; key on canonical path string so tests can pass different paths.
    # For production (single path) this is a one-time load.
    if _RMSE_CACHE is None:
        _RMSE_CACHE, _RMSE_MEDIAN = _load_rmse_lookup(accuracy_path)

    if not _RMSE_CACHE or _RMSE_MEDIAN is None:
        return None

    # Exact match first (fast path)
    key = pollster_name.lower()
    if key in _RMSE_CACHE:
        return _rmse_to_multiplier(_RMSE_CACHE[key], _RMSE_MEDIAN)

    # Fuzzy match fallback
    rmse = _fuzzy_match_pollster(pollster_name, _RMSE_CACHE)
    if rmse is not None:
        return _rmse_to_multiplier(rmse, _RMSE_MEDIAN)

    return None


def reset_rmse_cache() -> None:
    """Reset the RMSE lookup cache (for testing)."""
    global _RMSE_CACHE, _RMSE_MEDIAN
    _RMSE_CACHE = None
    _RMSE_MEDIAN = None

# ---------------------------------------------------------------------------
# Silver Bulletin integration
# ---------------------------------------------------------------------------

# Multiplier range for Silver Bulletin quality scores [0, 1] -> [min, max]
# Score 0.0 (banned) -> _SB_MIN_MULTIPLIER; score 1.0 (A+) -> _SB_MAX_MULTIPLIER
_SB_MIN_MULTIPLIER: float = 0.3
_SB_MAX_MULTIPLIER: float = 1.2

# Cached flag: None = not yet checked; True = SB available; False = not available
_SB_AVAILABLE: bool | None = None


def _sb_score_to_multiplier(score: float) -> float:
    """Linearly rescale a Silver Bulletin quality score [0, 1] to a multiplier."""
    return _SB_MIN_MULTIPLIER + score * (_SB_MAX_MULTIPLIER - _SB_MIN_MULTIPLIER)


def _get_sb_quality(pollster_name: str) -> float | None:
    """Return Silver Bulletin quality multiplier for *pollster_name*, or None if unavailable.

    Returns None when the Silver Bulletin XLSX is not present (so caller
    can fall back to 538 grades).  The first successful load caches a
    ``_SB_AVAILABLE=True`` flag; the first failure caches ``False`` to
    avoid repeated FileNotFoundError checks.
    """
    global _SB_AVAILABLE
    if _SB_AVAILABLE is False:
        return None

    try:
        from src.assembly.silver_bulletin_ratings import get_pollster_quality
        score = get_pollster_quality(pollster_name)
        _SB_AVAILABLE = True
        return _sb_score_to_multiplier(score)
    except FileNotFoundError:
        log.debug(
            "Silver Bulletin XLSX not found; falling back to 538 grade-based weighting"
        )
        _SB_AVAILABLE = False
        return None
    except Exception as exc:  # pragma: no cover
        log.warning("Silver Bulletin lookup failed (%s); using 538 grades", exc)
        _SB_AVAILABLE = False
        return None


def reset_sb_cache() -> None:
    """Reset all Silver Bulletin, RMSE, and house effect caches (for testing).

    Resets the availability flag AND the house effect caches so that tests
    calling this function start from a completely clean state — no stale
    house effect data from a previous test run.
    """
    global _SB_AVAILABLE
    _SB_AVAILABLE = None
    reset_rmse_cache()
    # Import here to avoid circular imports at module level — house_effects
    # is a sibling module that manages its own cache.
    from src.propagation.house_effects import reset_house_effect_cache
    reset_house_effect_cache()


# ---------------------------------------------------------------------------
# 538-grade fallback tables
# ---------------------------------------------------------------------------

# 538 numeric_grade -> quality multiplier
# Higher numeric grade = better pollster
# Scale: 3.0 = A+, ~2.5 = A, ~2.0 = A/B, ~1.5 = B, ~1.0 = B/C, ~0.5 = C, <0.5 = D
_DEFAULT_GRADE_MULTIPLIERS: dict[str, float] = {
    "A+": 1.2,
    "A": 1.1,
    "A/B": 1.0,
    "B": 0.9,
    "B/C": 0.8,
    "C": 0.7,
    "C/D": 0.5,
    "D": 0.3,
}

# No grade -> default multiplier
_NO_GRADE_MULTIPLIER = 0.8


def _numeric_grade_to_letter(grade_val: float) -> str:
    """Convert 538 numeric grade (0-3 scale) to letter grade."""
    if grade_val >= 2.8:
        return "A+"
    elif grade_val >= 2.4:
        return "A"
    elif grade_val >= 2.0:
        return "A/B"
    elif grade_val >= 1.5:
        return "B"
    elif grade_val >= 1.0:
        return "B/C"
    elif grade_val >= 0.5:
        return "C"
    elif grade_val >= 0.3:
        return "C/D"
    else:
        return "D"


def extract_grade_from_notes(notes: str) -> str | None:
    """Extract grade value from notes field (format: '...;grade=2.5;...').

    Returns letter grade string (e.g. 'A', 'B/C') or None if not found.
    """
    if not notes:
        return None
    for part in notes.split(";"):
        part = part.strip()
        if part.startswith("grade="):
            try:
                val = float(part[6:])
                return _numeric_grade_to_letter(val)
            except (ValueError, IndexError):
                return None
    return None


def grade_to_multiplier(
    grade: str | None,
    grade_multipliers: dict[str, float] | None = None,
) -> float:
    """Convert a letter grade to a quality multiplier."""
    if grade is None:
        return _NO_GRADE_MULTIPLIER
    table = grade_multipliers or _DEFAULT_GRADE_MULTIPLIERS
    return table.get(grade, _NO_GRADE_MULTIPLIER)


def apply_pollster_quality(
    polls: list[PollObservation],
    poll_notes: list[str] | None = None,
    grade_multipliers: dict[str, float] | None = None,
    use_silver_bulletin: bool = True,
    accuracy_path: Path | None = None,
) -> list[PollObservation]:
    """Adjust effective sample sizes by pollster quality.

    Quality source priority:
      1. RMSE-based accuracy data (when ``accuracy_path`` is provided and the
         pollster appears in the backtest results): empirical prediction error
         mapped to a multiplier via ``clamp(median_rmse / pollster_rmse,
         RMSE_MIN, RMSE_MAX)``.  This replaces the grade-based weight when
         RMSE data is available for the pollster.
      2. Silver Bulletin ratings (when XLSX is present and ``use_silver_bulletin``
         is True): ``get_pollster_quality(poll.pollster)`` returns a [0, 1]
         score, which is linearly rescaled to [0.3, 1.2].
      3. 538 numeric grade from poll_notes (format: "...;grade=2.5;...").
         Applied when both RMSE data and Silver Bulletin are unavailable.

    If poll_notes is None and no other source is available, all polls get
    the no-grade default (0.8x).

    Returns new PollObservation copies with adjusted n_sample.
    """
    result: list[PollObservation] = []

    for i, poll in enumerate(polls):
        multiplier: float | None = None

        # --- Priority 1: RMSE-based accuracy data ---
        if accuracy_path is not None and poll.pollster:
            multiplier = get_rmse_quality(poll.pollster, accuracy_path)

        # --- Priority 2: Silver Bulletin ---
        if multiplier is None and use_silver_bulletin and poll.pollster:
            multiplier = _get_sb_quality(poll.pollster)

        # --- Priority 3: 538 grade from notes ---
        if multiplier is None:
            notes = poll_notes[i] if poll_notes and i < len(poll_notes) else ""
            grade = extract_grade_from_notes(notes)
            multiplier = grade_to_multiplier(grade, grade_multipliers)

        n_effective = int(max(1, round(poll.n_sample * multiplier)))
        new_poll = copy(poll)
        new_poll.n_sample = n_effective
        result.append(new_poll)

    return result
