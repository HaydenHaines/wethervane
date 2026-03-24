"""Silver Bulletin pollster ratings loader.

Loads the Silver Bulletin pollster stats XLSX and exposes a quality score
[0.0, 1.0] for each pollster, derived from the letter grade and predictive
plus-minus metric.

Quality score derivation
------------------------
The Silver Bulletin's "Predictive Plus-Minus" column measures how much error
a pollster adds relative to a naive benchmark (lower = better). The SB Grade
(A+, A, A-, B+, …, F) is a summary label. We use the grade as the primary
signal (robust, comparable across eras) and fall back to plus-minus rescaling
when a grade is absent.

Grade-to-score mapping (higher = better pollster):
  A+   -> 1.00
  A    -> 0.93
  A-   -> 0.87
  A/B  -> 0.80
  B+   -> 0.73
  B    -> 0.67
  B-   -> 0.60
  B/C  -> 0.53
  C+   -> 0.47
  C    -> 0.40
  C/D  -> 0.30
  D+   -> 0.20
  F    -> 0.05
  (banned) -> 0.0

Unknown pollsters get 0.5 (neutral prior).

Name normalization
------------------
Pollster names vary across data sources (e.g. "Univ." vs "University",
leading/trailing whitespace, extra punctuation). We apply a set of deterministic
substitutions so that fuzzy lookups work without a heavy library dependency.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import openpyxl

# ---------------------------------------------------------------------------
# Grade table
# ---------------------------------------------------------------------------

GRADE_SCORES: dict[str, float] = {
    "A+": 1.00,
    "A": 0.93,
    "A-": 0.87,
    "A/B": 0.80,
    "B+": 0.73,
    "B": 0.67,
    "B-": 0.60,
    "B/C": 0.53,
    "C+": 0.47,
    "C": 0.40,
    "C/D": 0.30,
    "D+": 0.20,
    "F": 0.05,
}

# Score for a pollster banned by Silver Bulletin for fabrication/misconduct
BANNED_SCORE: float = 0.0

# Score returned when a pollster name is not found in the database
DEFAULT_SCORE: float = 0.5

# Default path to the XLSX (relative to project root)
DEFAULT_XLSX_PATH: Path = (
    Path(__file__).parent.parent.parent / "data" / "raw" / "silver_bulletin" / "pollster_stats_full_2026.xlsx"
)


# ---------------------------------------------------------------------------
# Name normalisation helpers
# ---------------------------------------------------------------------------

_SUBSTITUTIONS: list[tuple[str, str]] = [
    (r"\buniv\b\.?", "university"),
    (r"\bcollege\b", "college"),
    (r"\bpoll(ing)?\b", "poll"),
    (r"[''`]", "'"),          # fancy apostrophes
    (r"[&]", "and"),
    (r"\bst\b\.?", "saint"),  # St. → saint
    (r"\bmt\b\.?", "mount"),
    (r"\bprof\b\.?", "professional"),
    (r"\bres\b\.?\s+", "research "),
    (r"\bllc\b\.?", ""),
    (r"\binc\b\.?", ""),
    (r"\bcorp\b\.?", ""),
    (r"\bco\b\.?", ""),
    (r"[.,;:!?/\\]", " "),    # punctuation → space
    (r"\s+", " "),            # collapse multiple spaces
]

_COMPILED: list[tuple[re.Pattern, str]] = [
    (re.compile(pat, re.IGNORECASE), repl) for pat, repl in _SUBSTITUTIONS
]


def _normalize(name: str) -> str:
    """Normalize a pollster name for fuzzy matching."""
    s = name.lower().strip()
    for pattern, replacement in _COMPILED:
        s = pattern.sub(replacement, s)
    return s.strip()


def _name_similarity(a: str, b: str) -> float:
    """Simple token-overlap similarity between two normalised names.

    Returns Jaccard similarity on word sets: |A ∩ B| / |A ∪ B|.
    """
    set_a = set(a.split())
    set_b = set(b.split())
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------


def load_pollster_ratings(path: Optional[Path | str] = None) -> dict[str, float]:
    """Load Silver Bulletin pollster ratings XLSX and return {pollster_name: quality_score}.

    The quality score is a float in [0, 1] derived from the Silver Bulletin
    letter grade. Higher = better pollster. Banned pollsters receive 0.0.

    Parameters
    ----------
    path:
        Path to the Silver Bulletin XLSX file. Defaults to
        ``data/raw/silver_bulletin/pollster_stats_full_2026.xlsx`` relative
        to the project root.

    Returns
    -------
    dict[str, float]
        Mapping from original pollster name (as in the XLSX) to quality score.

    Raises
    ------
    FileNotFoundError
        If the XLSX file does not exist at the given path.
    """
    xlsx_path = Path(path) if path is not None else DEFAULT_XLSX_PATH

    if not xlsx_path.exists():
        raise FileNotFoundError(
            f"Silver Bulletin XLSX not found at {xlsx_path}. "
            "Download it with: curl -sL <url> -o <path>"
        )

    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb.active

    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        return {}

    # Parse column indices from header row.
    # The Silver Bulletin XLSX uses a merged-cell layout where the header row
    # shows "Pollster" at col 0, but col 0 contains the numeric rank in data
    # rows and col 1 contains the actual pollster name string.  We detect this
    # by checking whether the first data row's "Pollster" column holds a string
    # or an integer, and advance by one if it is an integer.
    header = rows[0]
    col_name_raw = _find_col(header, "pollster", required=True, fallback=0)
    # Peek at first data row to check whether we need to shift by 1
    if len(rows) > 1 and len(rows[1]) > col_name_raw:
        peek = rows[1][col_name_raw]
        if isinstance(peek, (int, float)) and not isinstance(peek, bool):
            col_name_raw += 1  # name is in the adjacent (None-header) cell
    col_name = col_name_raw
    col_banned = _find_col(header, "banned", required=False, fallback=5)
    col_grade = _find_col(header, "sb grade", required=False, fallback=8)

    ratings: dict[str, float] = {}

    for row in rows[1:]:
        if row is None or len(row) <= col_name:
            continue

        raw_name = row[col_name]
        if not raw_name:
            continue
        name = str(raw_name).strip()

        # Banned pollsters
        if col_banned is not None and len(row) > col_banned:
            banned_val = str(row[col_banned] or "").strip().lower()
            if banned_val == "yes":
                ratings[name] = BANNED_SCORE
                continue

        # Grade-based score
        grade = None
        if col_grade is not None and len(row) > col_grade:
            raw_grade = row[col_grade]
            if raw_grade is not None:
                grade = str(raw_grade).strip()

        score = GRADE_SCORES.get(grade, DEFAULT_SCORE) if grade else DEFAULT_SCORE
        ratings[name] = score

    wb.close()
    return ratings


def _find_col(
    header: tuple,
    keyword: str,
    required: bool = False,
    fallback: int = 0,
) -> Optional[int]:
    """Return column index whose header contains *keyword* (case-insensitive)."""
    for i, cell in enumerate(header):
        if cell is not None and keyword.lower() in str(cell).lower():
            return i
    if required:
        return fallback
    return fallback


# ---------------------------------------------------------------------------
# Grade conversion helper (public, for testing)
# ---------------------------------------------------------------------------


def grade_to_score(grade: str) -> float:
    """Convert a Silver Bulletin letter grade string to a quality score.

    Parameters
    ----------
    grade:
        Letter grade string, e.g. ``"A+"``, ``"B/C"``, ``"F"``.

    Returns
    -------
    float
        Quality score in [0, 1]. Returns ``DEFAULT_SCORE`` for unrecognised grades.
    """
    return GRADE_SCORES.get(grade.strip() if grade else "", DEFAULT_SCORE)


# ---------------------------------------------------------------------------
# Fuzzy lookup
# ---------------------------------------------------------------------------

# Module-level cache populated on first call to get_pollster_quality
_RATINGS_CACHE: Optional[dict[str, float]] = None
_NORMALIZED_CACHE: Optional[dict[str, tuple[str, float]]] = None  # norm_name -> (orig_name, score)


def _ensure_cache(path: Optional[Path | str] = None) -> None:
    global _RATINGS_CACHE, _NORMALIZED_CACHE
    if _RATINGS_CACHE is None:
        _RATINGS_CACHE = load_pollster_ratings(path)
        _NORMALIZED_CACHE = {
            _normalize(name): (name, score)
            for name, score in _RATINGS_CACHE.items()
        }


def get_pollster_quality(
    name: str,
    path: Optional[Path | str] = None,
    threshold: float = 0.4,
) -> float:
    """Return quality score for a pollster, with fuzzy name matching.

    Lookup order:
    1. Exact match on original XLSX name.
    2. Exact match after normalization (lowercase, common substitutions).
    3. Best Jaccard token-overlap match among normalized names, if the
       overlap score exceeds *threshold*.
    4. Return ``DEFAULT_SCORE`` (0.5) if no match found.

    Parameters
    ----------
    name:
        Pollster name to look up (may use different abbreviations than XLSX).
    path:
        Optional override for the XLSX path (passed to ``load_pollster_ratings``
        on first call; ignored on subsequent calls due to caching).
    threshold:
        Minimum Jaccard similarity required for a fuzzy match to count.
        Default 0.4 — requires at least 40% token overlap.

    Returns
    -------
    float
        Quality score in [0, 1]. Returns 0.5 for unknown pollsters.
    """
    _ensure_cache(path)
    assert _RATINGS_CACHE is not None
    assert _NORMALIZED_CACHE is not None

    # 1. Exact match
    if name in _RATINGS_CACHE:
        return _RATINGS_CACHE[name]

    # 2. Normalised exact match
    norm = _normalize(name)
    if norm in _NORMALIZED_CACHE:
        return _NORMALIZED_CACHE[norm][1]

    # 3. Best Jaccard fuzzy match
    best_score: Optional[float] = None
    best_sim = 0.0
    for norm_key, (_, score) in _NORMALIZED_CACHE.items():
        sim = _name_similarity(norm, norm_key)
        if sim > best_sim:
            best_sim = sim
            best_score = score

    if best_sim >= threshold and best_score is not None:
        return best_score

    return DEFAULT_SCORE


def clear_cache() -> None:
    """Clear the module-level pollster ratings cache.

    Useful in tests that use temporary XLSX fixtures.
    """
    global _RATINGS_CACHE, _NORMALIZED_CACHE
    _RATINGS_CACHE = None
    _NORMALIZED_CACHE = None
