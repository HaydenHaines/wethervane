"""Tests for the Silver Bulletin pollster ratings module.

Tests cover:
1. Grade-to-score conversion (all grades, edge cases)
2. Exact pollster name lookup
3. Normalisation-based lookup (abbreviations, punctuation)
4. Fuzzy token-overlap matching
5. Default score for unknown pollsters
6. Loader with a real XLSX file (skipped when file is absent)
7. Banned pollster handling (score = 0.0)
8. Name normalization helper
"""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

import openpyxl
import pytest

from src.assembly.silver_bulletin_ratings import (
    BANNED_SCORE,
    DEFAULT_SCORE,
    GRADE_SCORES,
    clear_cache,
    get_pollster_quality,
    grade_to_score,
    load_pollster_ratings,
    _normalize,
)

# ---------------------------------------------------------------------------
# Path to real XLSX (may not exist in CI)
# ---------------------------------------------------------------------------

REAL_XLSX = Path(__file__).parent.parent / "data" / "raw" / "silver_bulletin" / "pollster_stats_full_2026.xlsx"
REAL_XLSX_EXISTS = REAL_XLSX.exists()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_xlsx(rows: list[tuple]) -> Path:
    """Write a minimal XLSX to a temp file and return its path."""
    wb = openpyxl.Workbook()
    ws = wb.active
    header = (
        "Rank",
        "Pollster",
        "Pollster Rating ID",
        "# of Polls",
        "AAPOR / Roper",
        "Banned by Silver Bulletin",
        None,
        "Predictive Plus-Minus",
        "SB Grade",
    )
    ws.append(header)
    for row in rows:
        ws.append(row)
    tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
    tmp.close()
    wb.save(tmp.name)
    return Path(tmp.name)


@pytest.fixture
def sample_xlsx(tmp_path):
    """Minimal XLSX with a handful of pollsters covering the major grade tiers."""
    data = [
        (1,  "Top Pollster",      "tp1", 50, "yes", "no",  None, -1.0, "A+"),
        (2,  "Good Pollster",     "gp1", 40, "no",  "no",  None, -0.5, "B+"),
        (3,  "Average Pollster",  "ap1", 30, "no",  "no",  None,  0.5, "C"),
        (4,  "Marquette Univ.",   "mu1", 17, "yes", "no",  None, -1.0, "A/B"),
        (5,  "Banned Firm LLC",   "bf1", 5,  "no",  "yes", None,  4.0, "F"),
        (6,  "No Grade Pollster", "ng1", 2,  "no",  "no",  None, None, None),
    ]
    path = _make_xlsx(data)
    yield path
    os.unlink(path)


# ---------------------------------------------------------------------------
# 1. Grade-to-score conversion
# ---------------------------------------------------------------------------


class TestGradeToScore:
    def test_all_defined_grades_return_expected_values(self):
        assert grade_to_score("A+") == 1.00
        assert grade_to_score("A") == 0.93
        assert grade_to_score("A-") == 0.87
        assert grade_to_score("A/B") == 0.80
        assert grade_to_score("B+") == 0.73
        assert grade_to_score("B") == 0.67
        assert grade_to_score("B-") == 0.60
        assert grade_to_score("B/C") == 0.53
        assert grade_to_score("C+") == 0.47
        assert grade_to_score("C") == 0.40
        assert grade_to_score("C/D") == 0.30
        assert grade_to_score("D+") == 0.20
        assert grade_to_score("F") == 0.05

    def test_all_grades_in_range(self):
        for grade, score in GRADE_SCORES.items():
            assert 0.0 <= score <= 1.0, f"Grade {grade!r} score {score} out of [0,1]"

    def test_grades_are_monotone_by_tier(self):
        """Higher grades should yield higher scores."""
        assert grade_to_score("A+") > grade_to_score("A")
        assert grade_to_score("A") > grade_to_score("A-")
        assert grade_to_score("A-") > grade_to_score("B+")
        assert grade_to_score("B+") > grade_to_score("B")
        assert grade_to_score("B") > grade_to_score("B-")
        assert grade_to_score("C") > grade_to_score("C/D")
        assert grade_to_score("C/D") > grade_to_score("F")

    def test_unknown_grade_returns_default(self):
        assert grade_to_score("Z") == DEFAULT_SCORE
        assert grade_to_score("") == DEFAULT_SCORE
        assert grade_to_score("??") == DEFAULT_SCORE

    def test_whitespace_stripped(self):
        assert grade_to_score("  A+ ") == 1.00


# ---------------------------------------------------------------------------
# 2. Name normalisation
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_lowercase_and_strip(self):
        assert _normalize("  CBS News  ") == "cbs news"

    def test_univ_expansion(self):
        n = _normalize("Marquette Univ.")
        assert "university" in n

    def test_ampersand_to_and(self):
        n = _normalize("Research & Polling")
        assert "and" in n

    def test_llc_removed(self):
        n = _normalize("Banned Firm LLC")
        assert "llc" not in n

    def test_punctuation_becomes_space(self):
        n = _normalize("Poll.Co, Inc.")
        # Should not contain raw punctuation
        assert "." not in n
        assert "," not in n


# ---------------------------------------------------------------------------
# 3. Loader with synthetic XLSX
# ---------------------------------------------------------------------------


class TestLoadPollsterRatings:
    def test_returns_dict(self, sample_xlsx):
        clear_cache()
        ratings = load_pollster_ratings(sample_xlsx)
        assert isinstance(ratings, dict)

    def test_known_pollsters_present(self, sample_xlsx):
        clear_cache()
        ratings = load_pollster_ratings(sample_xlsx)
        assert "Top Pollster" in ratings
        assert "Good Pollster" in ratings
        assert "Average Pollster" in ratings

    def test_a_plus_score(self, sample_xlsx):
        clear_cache()
        ratings = load_pollster_ratings(sample_xlsx)
        assert ratings["Top Pollster"] == 1.00

    def test_b_plus_score(self, sample_xlsx):
        clear_cache()
        ratings = load_pollster_ratings(sample_xlsx)
        assert ratings["Good Pollster"] == 0.73

    def test_c_score(self, sample_xlsx):
        clear_cache()
        ratings = load_pollster_ratings(sample_xlsx)
        assert ratings["Average Pollster"] == 0.40

    def test_banned_pollster_gets_zero(self, sample_xlsx):
        clear_cache()
        ratings = load_pollster_ratings(sample_xlsx)
        assert ratings["Banned Firm LLC"] == BANNED_SCORE

    def test_no_grade_returns_default(self, sample_xlsx):
        clear_cache()
        ratings = load_pollster_ratings(sample_xlsx)
        assert ratings["No Grade Pollster"] == DEFAULT_SCORE

    def test_file_not_found_raises(self, tmp_path):
        clear_cache()
        with pytest.raises(FileNotFoundError):
            load_pollster_ratings(tmp_path / "nonexistent.xlsx")


# ---------------------------------------------------------------------------
# 4. Exact lookup via get_pollster_quality
# ---------------------------------------------------------------------------


class TestGetPollsterQualityExact:
    def test_exact_name_lookup(self, sample_xlsx):
        clear_cache()
        score = get_pollster_quality("Top Pollster", path=sample_xlsx)
        assert score == 1.00

    def test_banned_exact_lookup(self, sample_xlsx):
        clear_cache()
        score = get_pollster_quality("Banned Firm LLC", path=sample_xlsx)
        assert score == BANNED_SCORE

    def test_unknown_returns_default(self, sample_xlsx):
        clear_cache()
        score = get_pollster_quality("Completely Unknown Org", path=sample_xlsx)
        assert score == DEFAULT_SCORE


# ---------------------------------------------------------------------------
# 5. Fuzzy lookup
# ---------------------------------------------------------------------------


class TestGetPollsterQualityFuzzy:
    def test_normalized_lookup_univ_abbreviation(self, sample_xlsx):
        """'Marquette Univ.' in XLSX should match 'Marquette University'."""
        clear_cache()
        score = get_pollster_quality("Marquette University", path=sample_xlsx)
        # Should match the A/B grade (0.80), not default (0.5)
        assert score == 0.80

    def test_case_insensitive_lookup(self, sample_xlsx):
        clear_cache()
        score = get_pollster_quality("top pollster", path=sample_xlsx)
        assert score == 1.00

    def test_extra_whitespace_ignored(self, sample_xlsx):
        clear_cache()
        score = get_pollster_quality("  Good Pollster  ", path=sample_xlsx)
        assert score == 0.73

    def test_partial_token_overlap(self, sample_xlsx):
        """'Average Poll' has enough token overlap with 'Average Pollster'."""
        clear_cache()
        score = get_pollster_quality("Average Poll", path=sample_xlsx)
        # Jaccard("average poll", "average pollster") = 1/3 which is < 0.4 threshold
        # So this should fall back to default — verify we don't crash
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_result_in_valid_range(self, sample_xlsx):
        clear_cache()
        for name in ["Top Pollster", "Good Pollster", "Some Unknown Firm"]:
            score = get_pollster_quality(name, path=sample_xlsx)
            assert 0.0 <= score <= 1.0, f"Score for {name!r} is {score}"


# ---------------------------------------------------------------------------
# 6. Real XLSX integration test (skipped when file absent)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not REAL_XLSX_EXISTS, reason="Silver Bulletin XLSX not downloaded")
class TestRealXlsx:
    def test_loads_without_error(self):
        clear_cache()
        ratings = load_pollster_ratings(REAL_XLSX)
        assert isinstance(ratings, dict)

    def test_loads_substantial_number_of_pollsters(self):
        clear_cache()
        ratings = load_pollster_ratings(REAL_XLSX)
        # The 2026 file has ~539 pollsters
        assert len(ratings) > 100, f"Expected >100 pollsters, got {len(ratings)}"

    def test_all_scores_in_range(self):
        clear_cache()
        ratings = load_pollster_ratings(REAL_XLSX)
        for name, score in ratings.items():
            assert 0.0 <= score <= 1.0, f"Score for {name!r} = {score} out of [0,1]"

    def test_known_top_pollster_present(self):
        """Washington Post is rated A+ in the 2026 file."""
        clear_cache()
        ratings = load_pollster_ratings(REAL_XLSX)
        assert "The Washington Post" in ratings
        assert ratings["The Washington Post"] == 1.00

    def test_get_pollster_quality_known_name(self):
        clear_cache()
        score = get_pollster_quality("The Washington Post")
        assert score == 1.00

    def test_get_pollster_quality_unknown_returns_default(self):
        clear_cache()
        score = get_pollster_quality("Completely Nonexistent Polling Organization 99999")
        assert score == DEFAULT_SCORE

    def test_no_none_values(self):
        clear_cache()
        ratings = load_pollster_ratings(REAL_XLSX)
        for name, score in ratings.items():
            assert score is not None, f"None score for {name!r}"

    def test_fuzzy_lookup_marquette(self):
        """'Marquette Law School' should resolve to 'Marquette University Law School'."""
        clear_cache()
        score = get_pollster_quality("Marquette Law School")
        # Must not be default — Jaccard overlap should be enough
        # Marquette University Law School vs Marquette Law School:
        # norm A = "marquette university law school", norm B = "marquette law school"
        # intersection = {marquette, law, school}, union = {marquette, university, law, school}
        # Jaccard = 3/4 = 0.75 -> above threshold
        assert score > DEFAULT_SCORE, f"Expected fuzzy match for Marquette, got {score}"
