"""Tests for RMSE-based pollster quality weighting.

Coverage:
  - RMSE → multiplier mapping (known pollsters get expected weights)
  - Fallback behavior when pollster not in RMSE data
  - Fuzzy name matching
  - End-to-end: accurate pollsters get higher effective weight than inaccurate ones
  - Edge cases: empty file, single pollster, all same RMSE, missing file
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

from src.propagation.poll_quality import (
    _RMSE_MAX_MULTIPLIER,
    _RMSE_MIN_MULTIPLIER,
    _rmse_to_multiplier,
    get_rmse_quality,
    reset_rmse_cache,
    reset_sb_cache,
)
from src.propagation.poll_weighting import (
    apply_all_weights,
    apply_pollster_quality,
)
from src.propagation.propagate_polls import PollObservation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_accuracy_json(pollsters: list[dict], path: Path) -> Path:
    """Write a minimal pollster_accuracy.json for testing."""
    data = {
        "description": "Test accuracy data",
        "n_pollsters": len(pollsters),
        "pollsters": pollsters,
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _make_poll(
    pollster: str = "TestPollster",
    n_sample: int = 1000,
    dem_share: float = 0.50,
    date: str = "2026-10-01",
) -> PollObservation:
    return PollObservation(
        geography="GA",
        dem_share=dem_share,
        n_sample=n_sample,
        race="2026 GA Senate",
        date=date,
        pollster=pollster,
        geo_level="state",
    )


@pytest.fixture(autouse=True)
def _reset_caches():
    """Ensure RMSE and SB caches are cleared between tests."""
    reset_sb_cache()
    yield
    reset_sb_cache()


# ---------------------------------------------------------------------------
# _rmse_to_multiplier unit tests
# ---------------------------------------------------------------------------


class TestRmseToMultiplier:
    def test_median_pollster_gets_one(self):
        """A pollster exactly at the median RMSE gets multiplier 1.0."""
        median = 2.5
        result = _rmse_to_multiplier(median, median)
        assert result == pytest.approx(1.0)

    def test_better_than_median_gets_above_one(self):
        """A pollster with half the median RMSE gets multiplier > 1.0."""
        median = 2.0
        better_rmse = 1.0  # half the median → ratio 2.0
        result = _rmse_to_multiplier(better_rmse, median)
        assert result > 1.0

    def test_worse_than_median_gets_below_one(self):
        """A pollster with double the median RMSE gets multiplier < 1.0."""
        median = 2.0
        worse_rmse = 4.0  # double the median → ratio 0.5
        result = _rmse_to_multiplier(worse_rmse, median)
        assert result < 1.0

    def test_clamp_at_max(self):
        """Extremely accurate pollsters are capped at RMSE_MAX_MULTIPLIER."""
        result = _rmse_to_multiplier(0.001, 2.0)  # ratio = 2000
        assert result == _RMSE_MAX_MULTIPLIER

    def test_clamp_at_min(self):
        """Extremely inaccurate pollsters are floored at RMSE_MIN_MULTIPLIER."""
        result = _rmse_to_multiplier(1000.0, 2.0)  # ratio = 0.002
        assert result == _RMSE_MIN_MULTIPLIER

    def test_inverse_relationship(self):
        """Lower RMSE always produces a higher multiplier than higher RMSE."""
        median = 3.0
        m_accurate = _rmse_to_multiplier(1.0, median)
        m_inaccurate = _rmse_to_multiplier(5.0, median)
        assert m_accurate > m_inaccurate

    def test_zero_rmse_returns_one(self):
        """Guard: zero RMSE returns 1.0 rather than crashing."""
        result = _rmse_to_multiplier(0.0, 2.0)
        assert result == 1.0

    def test_zero_median_returns_one(self):
        """Guard: zero median returns 1.0 rather than crashing."""
        result = _rmse_to_multiplier(2.0, 0.0)
        assert result == 1.0


# ---------------------------------------------------------------------------
# get_rmse_quality: exact and fuzzy matching
# ---------------------------------------------------------------------------


class TestGetRmseQuality:
    def _two_pollster_file(self, tmp_path: Path) -> Path:
        """Write a two-pollster accuracy file: Good (1.0 pp) and Bad (4.0 pp)."""
        return _make_accuracy_json(
            [
                {"pollster": "Good Pollster", "n_polls": 5, "n_races": 2, "rmse_pp": 1.0, "rank": 1},
                {"pollster": "Bad Pollster", "n_polls": 5, "n_races": 2, "rmse_pp": 4.0, "rank": 2},
            ],
            tmp_path / "accuracy.json",
        )

    def test_exact_match_known_pollster(self, tmp_path):
        """Exact pollster name match returns a non-None multiplier."""
        path = self._two_pollster_file(tmp_path)
        result = get_rmse_quality("Good Pollster", path)
        assert result is not None

    def test_known_pollster_gets_expected_multiplier(self, tmp_path):
        """Good Pollster (RMSE=1.0) with median 2.5 → ratio=2.5, capped at RMSE_MAX."""
        path = self._two_pollster_file(tmp_path)
        median = statistics.median([1.0, 4.0])  # 2.5
        expected = min(_RMSE_MAX_MULTIPLIER, median / 1.0)  # 2.5, capped
        result = get_rmse_quality("Good Pollster", path)
        assert result == pytest.approx(expected, abs=1e-6)

    def test_bad_pollster_gets_lower_multiplier(self, tmp_path):
        """Bad Pollster (RMSE=4.0) should get a lower multiplier than Good Pollster."""
        path = self._two_pollster_file(tmp_path)
        good = get_rmse_quality("Good Pollster", path)
        bad = get_rmse_quality("Bad Pollster", path)
        assert good > bad

    def test_unknown_pollster_returns_none(self, tmp_path):
        """A pollster not in the accuracy file returns None (triggers fallback)."""
        path = self._two_pollster_file(tmp_path)
        result = get_rmse_quality("Unknown Firm Inc", path)
        assert result is None

    def test_case_insensitive_exact_match(self, tmp_path):
        """Matching is case-insensitive."""
        path = self._two_pollster_file(tmp_path)
        result = get_rmse_quality("good pollster", path)
        assert result is not None

    def test_fuzzy_match_partial_name(self, tmp_path):
        """Slight name variation is handled via fuzzy token matching."""
        path = _make_accuracy_json(
            [{"pollster": "New York Times/Siena College", "n_polls": 2, "n_races": 2, "rmse_pp": 1.5, "rank": 1}],
            tmp_path / "accuracy.json",
        )
        # Poll CSV might list it without "New"
        result = get_rmse_quality("New York Times / Siena College", path)
        assert result is not None

    def test_missing_file_returns_none(self, tmp_path):
        """Missing accuracy file returns None without crashing."""
        path = tmp_path / "nonexistent.json"
        result = get_rmse_quality("Emerson College", path)
        assert result is None

    def test_empty_pollsters_list_returns_none(self, tmp_path):
        """Accuracy file with empty pollsters list returns None."""
        path = _make_accuracy_json([], tmp_path / "accuracy.json")
        result = get_rmse_quality("Emerson College", path)
        assert result is None

    def test_single_pollster_gets_multiplier_one(self, tmp_path):
        """With a single pollster the median equals its RMSE → multiplier exactly 1.0."""
        path = _make_accuracy_json(
            [{"pollster": "Only Pollster", "n_polls": 3, "n_races": 1, "rmse_pp": 2.0, "rank": 1}],
            tmp_path / "accuracy.json",
        )
        result = get_rmse_quality("Only Pollster", path)
        assert result == pytest.approx(1.0)

    def test_all_same_rmse_all_get_one(self, tmp_path):
        """When all pollsters have the same RMSE, all get multiplier 1.0."""
        path = _make_accuracy_json(
            [
                {"pollster": "Alpha", "n_polls": 2, "n_races": 1, "rmse_pp": 3.0, "rank": 1},
                {"pollster": "Beta", "n_polls": 2, "n_races": 1, "rmse_pp": 3.0, "rank": 2},
                {"pollster": "Gamma", "n_polls": 2, "n_races": 1, "rmse_pp": 3.0, "rank": 3},
            ],
            tmp_path / "accuracy.json",
        )
        for name in ("Alpha", "Beta", "Gamma"):
            result = get_rmse_quality(name, path)
            assert result == pytest.approx(1.0), f"{name} should get 1.0"


# ---------------------------------------------------------------------------
# apply_pollster_quality: RMSE replaces grade when available
# ---------------------------------------------------------------------------


class TestApplyPollsterQualityWithRmse:
    def test_rmse_takes_priority_over_grade(self, tmp_path):
        """When RMSE data is available, it replaces the grade-based multiplier."""
        # Accuracy data: median 2.5, this pollster RMSE=1.0 → multiplier=min(1.5, 2.5)=1.5
        path = _make_accuracy_json(
            [
                {"pollster": "RMSEPollster", "n_polls": 3, "n_races": 2, "rmse_pp": 1.0, "rank": 1},
                {"pollster": "Other", "n_polls": 3, "n_races": 2, "rmse_pp": 4.0, "rank": 2},
            ],
            tmp_path / "accuracy.json",
        )
        poll = _make_poll(pollster="RMSEPollster", n_sample=1000)
        # With RMSE: multiplier = min(RMSE_MAX, 2.5/1.0) = 1.5 → n=1500
        result = apply_pollster_quality(
            [poll],
            poll_notes=["grade=1.5"],  # grade=1.5 → B → 0.9x — should be ignored
            use_silver_bulletin=False,
            accuracy_path=path,
        )
        # RMSE multiplier (1.5) is larger than grade multiplier (0.9)
        assert result[0].n_sample > 900
        # Should be at least 1000 (RMSE data beats grade)
        assert result[0].n_sample >= 1000

    def test_fallback_to_grade_when_not_in_rmse(self, tmp_path):
        """Pollsters absent from RMSE data fall back to grade-based weighting."""
        path = _make_accuracy_json(
            [{"pollster": "SomeOtherPollster", "n_polls": 1, "n_races": 1, "rmse_pp": 2.0, "rank": 1}],
            tmp_path / "accuracy.json",
        )
        poll = _make_poll(pollster="Unknown Pollster", n_sample=1000)
        # grade=3.0 → A+ → 1.2x multiplier
        result = apply_pollster_quality(
            [poll],
            poll_notes=["grade=3.0"],
            use_silver_bulletin=False,
            accuracy_path=path,
        )
        assert result[0].n_sample == 1200  # grade-based 1.2x applied

    def test_no_accuracy_path_uses_grade(self):
        """When no accuracy_path is provided, grade-based weighting is used as before."""
        poll = _make_poll(n_sample=1000)
        result = apply_pollster_quality(
            [poll],
            poll_notes=["grade=3.0"],
            use_silver_bulletin=False,
            accuracy_path=None,
        )
        assert result[0].n_sample == 1200  # A+ grade → 1.2x


# ---------------------------------------------------------------------------
# End-to-end: accurate pollsters have higher effective weight
# ---------------------------------------------------------------------------


class TestRmseWeightingEndToEnd:
    """Accurate pollster (low RMSE) should have meaningfully higher n_sample
    than inaccurate pollster (high RMSE) after apply_all_weights."""

    def test_accurate_pollster_outweighs_inaccurate(self, tmp_path):
        """After applying RMSE weights, accurate pollster effective N > inaccurate effective N."""
        path = _make_accuracy_json(
            [
                {"pollster": "Accurate Firm", "n_polls": 5, "n_races": 3, "rmse_pp": 0.5, "rank": 1},
                {"pollster": "Inaccurate Firm", "n_polls": 5, "n_races": 3, "rmse_pp": 8.0, "rank": 2},
            ],
            tmp_path / "accuracy.json",
        )

        accurate_poll = _make_poll(pollster="Accurate Firm", n_sample=1000)
        inaccurate_poll = _make_poll(pollster="Inaccurate Firm", n_sample=1000)

        # Same date → same time decay; only RMSE quality differs
        result = apply_all_weights(
            [accurate_poll, inaccurate_poll],
            reference_date="2026-10-01",
            apply_house_effects=False,
            use_primary_discount=False,
            use_silver_bulletin=False,
            accuracy_path=path,
        )
        accurate_n = result[0].n_sample
        inaccurate_n = result[1].n_sample
        assert accurate_n > inaccurate_n, (
            f"Expected accurate pollster n ({accurate_n}) > inaccurate n ({inaccurate_n})"
        )

    def test_unknown_pollster_gets_fallback_weight(self, tmp_path):
        """A pollster not in RMSE data falls back to grade/default weighting."""
        path = _make_accuracy_json(
            [{"pollster": "Known Firm", "n_polls": 1, "n_races": 1, "rmse_pp": 2.0, "rank": 1}],
            tmp_path / "accuracy.json",
        )
        poll = _make_poll(pollster="Unknown Firm", n_sample=1000)
        result = apply_all_weights(
            [poll],
            reference_date="2026-10-01",
            poll_notes=[""],
            apply_house_effects=False,
            use_primary_discount=False,
            use_silver_bulletin=False,
            accuracy_path=path,
        )
        # No grade in notes → default multiplier 0.8
        from src.propagation.poll_quality import _NO_GRADE_MULTIPLIER
        expected = int(max(1, round(1000 * _NO_GRADE_MULTIPLIER)))
        assert result[0].n_sample == expected
