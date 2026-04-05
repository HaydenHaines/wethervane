"""Tests for pollster house effects adjustment.

Coverage:
  - load_empirical_house_effects: loading from JSON, min_polls filter, missing file
  - _lookup_empirical_bias: exact match, case-insensitive, fuzzy, not found
  - _lookup_house_effect: empirical takes priority over Silver Bulletin and 538
  - apply_house_effect_correction: dem_share adjusted correctly, clamping, no-pollster passthrough
  - Cache reset: reset_house_effect_cache clears _EMPIRICAL_BIAS
  - Integration: empirical bias flows through apply_all_weights pipeline
"""
from __future__ import annotations

import json
from copy import copy
from pathlib import Path

import pytest

from src.propagation.house_effects import (
    _HE_DEM_SHARE_MAX,
    _HE_DEM_SHARE_MIN,
    _lookup_empirical_bias,
    _lookup_house_effect,
    apply_house_effect_correction,
    load_empirical_house_effects,
    reset_house_effect_cache,
)
from src.propagation.poll_weighting import apply_all_weights
from src.propagation.propagate_polls import PollObservation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_accuracy_json(pollsters: list[dict], path: Path) -> Path:
    """Write a minimal pollster_accuracy.json for testing."""
    data = {
        "description": "Test pollster accuracy",
        "n_pollsters": len(pollsters),
        "pollsters": pollsters,
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _make_accuracy_entry(
    pollster: str,
    mean_error_pp: float,
    n_polls: int = 5,
    rmse_pp: float = 2.0,
    n_races: int = 2,
    rank: int = 1,
) -> dict:
    """Build a single pollster entry for the accuracy JSON."""
    return {
        "pollster": pollster,
        "n_polls": n_polls,
        "n_races": n_races,
        "rmse_pp": rmse_pp,
        "mean_error_pp": mean_error_pp,
        "rank": rank,
    }


def _make_poll(
    pollster: str = "TestPollster",
    dem_share: float = 0.50,
    n_sample: int = 1000,
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
def _reset_cache():
    """Ensure house effect caches are cleared before and after each test."""
    reset_house_effect_cache()
    yield
    reset_house_effect_cache()


# ---------------------------------------------------------------------------
# load_empirical_house_effects
# ---------------------------------------------------------------------------


class TestLoadEmpiricalHouseEffects:
    def test_loads_bias_from_json(self, tmp_path):
        """Reads mean_error_pp from each pollster entry."""
        path = _write_accuracy_json(
            [
                _make_accuracy_entry("Emerson College", mean_error_pp=1.5, n_polls=5),
                _make_accuracy_entry("Rasmussen", mean_error_pp=-3.2, n_polls=8),
            ],
            tmp_path / "accuracy.json",
        )
        result = load_empirical_house_effects(path)
        assert result["Emerson College"] == pytest.approx(1.5)
        assert result["Rasmussen"] == pytest.approx(-3.2)

    def test_missing_file_returns_empty(self, tmp_path):
        """Missing file yields empty dict without raising."""
        result = load_empirical_house_effects(tmp_path / "nonexistent.json")
        assert result == {}

    def test_min_polls_filter(self, tmp_path):
        """Pollsters with fewer than min_polls polls are excluded."""
        path = _write_accuracy_json(
            [
                _make_accuracy_entry("BigPollster", mean_error_pp=2.0, n_polls=10),
                _make_accuracy_entry("TinyPollster", mean_error_pp=-1.0, n_polls=1),
            ],
            tmp_path / "accuracy.json",
        )
        # Default min_polls=2; TinyPollster (1 poll) should be excluded
        result = load_empirical_house_effects(path)
        assert "BigPollster" in result
        assert "TinyPollster" not in result

    def test_custom_min_polls(self, tmp_path):
        """Custom min_polls threshold is respected."""
        path = _write_accuracy_json(
            [
                _make_accuracy_entry("OncePollster", mean_error_pp=0.5, n_polls=1),
            ],
            tmp_path / "accuracy.json",
        )
        # With min_polls=1, even a single-poll pollster is included
        result = load_empirical_house_effects(path, min_polls=1)
        assert "OncePollster" in result

    def test_empty_pollsters_list(self, tmp_path):
        """Empty pollsters list yields empty dict."""
        path = _write_accuracy_json([], tmp_path / "accuracy.json")
        result = load_empirical_house_effects(path)
        assert result == {}

    def test_returns_float_values(self, tmp_path):
        """Values are floats even when JSON encodes them as ints."""
        path = _write_accuracy_json(
            [_make_accuracy_entry("Firm", mean_error_pp=2, n_polls=5)],
            tmp_path / "accuracy.json",
        )
        result = load_empirical_house_effects(path)
        assert isinstance(result["Firm"], float)

    def test_negative_bias_preserved(self, tmp_path):
        """Negative bias (R-leaning pollsters) is preserved correctly."""
        path = _write_accuracy_json(
            [_make_accuracy_entry("RasmReports", mean_error_pp=-4.1, n_polls=6)],
            tmp_path / "accuracy.json",
        )
        result = load_empirical_house_effects(path)
        assert result["RasmReports"] == pytest.approx(-4.1)

    def test_malformed_json_returns_empty(self, tmp_path):
        """Malformed JSON file yields empty dict without raising."""
        path = tmp_path / "bad.json"
        path.write_text("not valid json{{{", encoding="utf-8")
        result = load_empirical_house_effects(path)
        assert result == {}


# ---------------------------------------------------------------------------
# _lookup_empirical_bias
# ---------------------------------------------------------------------------


class TestLookupEmpiricalBias:
    def _bias(self) -> dict[str, float]:
        return {
            "Emerson College": 1.5,
            "Rasmussen Reports": -3.2,
            "New York Times/Siena College": 0.8,
        }

    def test_exact_match(self):
        """Exact pollster name returns the correct bias."""
        result = _lookup_empirical_bias("Emerson College", self._bias())
        assert result == pytest.approx(1.5)

    def test_case_insensitive_match(self):
        """Case-insensitive matching works for name variations."""
        result = _lookup_empirical_bias("emerson college", self._bias())
        assert result == pytest.approx(1.5)

    def test_not_found_returns_none(self):
        """Unknown pollster returns None."""
        result = _lookup_empirical_bias("Unknown Firm LLC", self._bias())
        assert result is None

    def test_empty_bias_dict_returns_none(self):
        """Empty bias dict always returns None."""
        result = _lookup_empirical_bias("Emerson College", {})
        assert result is None

    def test_fuzzy_match_partial_name(self):
        """Fuzzy matching handles minor name variations."""
        bias = {"New York Times Siena College": 1.0}
        # Token overlap between "New York Times/Siena College" and "New York Times Siena College"
        result = _lookup_empirical_bias("New York Times Siena College Poll", bias)
        # Should match via fuzzy (high token overlap)
        assert result is not None

    def test_fuzzy_no_match_below_threshold(self):
        """Completely different name doesn't fuzzy-match."""
        bias = {"Emerson College": 1.5}
        result = _lookup_empirical_bias("Rasmussen Reports", bias)
        assert result is None


# ---------------------------------------------------------------------------
# _lookup_house_effect: priority order
# ---------------------------------------------------------------------------


class TestLookupHouseEffectPriority:
    """Empirical bias should take priority over Silver Bulletin and 538."""

    def test_empirical_takes_priority_over_sb(self):
        """When empirical bias is available, it beats Silver Bulletin."""
        # Empirical says +2pp; if SB were used, it would return something else
        # We can't easily simulate SB without the XLSX, so pass empty SB + nonempty empirical
        effect, source = _lookup_house_effect(
            "Emerson College",
            sb_house_effects={},
            bias_538={},
            empirical_bias={"Emerson College": 2.0},
        )
        assert source == "empirical"
        assert effect == pytest.approx(2.0)

    def test_empirical_takes_priority_over_538(self):
        """Empirical bias beats 538 bias when both are present."""
        effect, source = _lookup_house_effect(
            "Rasmussen",
            sb_house_effects={},
            bias_538={"Rasmussen": -5.0},
            empirical_bias={"Rasmussen": -3.2},
        )
        assert source == "empirical"
        assert effect == pytest.approx(-3.2)

    def test_falls_through_to_538_when_no_empirical(self):
        """When empirical is empty, 538 bias is used."""
        effect, source = _lookup_house_effect(
            "Rasmussen",
            sb_house_effects={},
            bias_538={"Rasmussen": -5.0},
            empirical_bias={},
        )
        assert source == "538"
        assert effect == pytest.approx(-5.0)

    def test_returns_none_source_when_not_found(self):
        """Unknown pollster with empty all sources returns (0.0, 'none')."""
        effect, source = _lookup_house_effect(
            "UnknownPollsterXYZ",
            sb_house_effects={},
            bias_538={},
            empirical_bias={},
        )
        assert source == "none"
        assert effect == pytest.approx(0.0)

    def test_negative_empirical_bias_returned(self):
        """Negative (R-leaning) empirical bias is returned correctly."""
        effect, source = _lookup_house_effect(
            "Rasmussen",
            sb_house_effects={},
            bias_538={},
            empirical_bias={"Rasmussen": -4.0},
        )
        assert source == "empirical"
        assert effect == pytest.approx(-4.0)


# ---------------------------------------------------------------------------
# apply_house_effect_correction
# ---------------------------------------------------------------------------


class TestApplyHouseEffectCorrection:
    """Test that poll dem_share is adjusted correctly by empirical bias."""

    def test_dem_leaning_pollster_reduces_dem_share(self):
        """Positive bias (Dem-leaning pollster) reduces the dem_share."""
        # Pollster over-estimates Dems by 2pp → corrected share is lower
        poll = _make_poll(pollster="DemLean", dem_share=0.52)
        result = apply_house_effect_correction(
            [poll],
            sb_house_effects={},
            bias_538={},
            empirical_bias={"DemLean": 2.0},  # +2pp bias
        )
        assert result[0].dem_share == pytest.approx(0.52 - 0.02, abs=1e-6)

    def test_rep_leaning_pollster_increases_dem_share(self):
        """Negative bias (R-leaning pollster) increases the dem_share."""
        # Pollster under-estimates Dems by 3pp → corrected share is higher
        poll = _make_poll(pollster="RepLean", dem_share=0.48)
        result = apply_house_effect_correction(
            [poll],
            sb_house_effects={},
            bias_538={},
            empirical_bias={"RepLean": -3.0},  # -3pp bias
        )
        assert result[0].dem_share == pytest.approx(0.48 + 0.03, abs=1e-6)

    def test_zero_bias_leaves_dem_share_unchanged(self):
        """A pollster with zero bias leaves dem_share unchanged."""
        poll = _make_poll(pollster="NeutralFirm", dem_share=0.50)
        result = apply_house_effect_correction(
            [poll],
            sb_house_effects={},
            bias_538={},
            empirical_bias={"NeutralFirm": 0.0},
        )
        assert result[0].dem_share == pytest.approx(0.50, abs=1e-6)

    def test_unknown_pollster_left_unchanged(self):
        """A pollster not in any source is not adjusted."""
        poll = _make_poll(pollster="UnknownFirm", dem_share=0.55)
        result = apply_house_effect_correction(
            [poll],
            sb_house_effects={},
            bias_538={},
            empirical_bias={},
        )
        assert result[0].dem_share == pytest.approx(0.55, abs=1e-6)

    def test_dem_share_clamped_at_max(self):
        """Large negative bias can't push dem_share above _HE_DEM_SHARE_MAX."""
        poll = _make_poll(pollster="ExtremeR", dem_share=0.98)
        result = apply_house_effect_correction(
            [poll],
            sb_house_effects={},
            bias_538={},
            empirical_bias={"ExtremeR": -5.0},  # -5pp → would push to 1.03
        )
        assert result[0].dem_share <= _HE_DEM_SHARE_MAX

    def test_dem_share_clamped_at_min(self):
        """Large positive bias can't push dem_share below _HE_DEM_SHARE_MIN."""
        poll = _make_poll(pollster="ExtremeD", dem_share=0.02)
        result = apply_house_effect_correction(
            [poll],
            sb_house_effects={},
            bias_538={},
            empirical_bias={"ExtremeD": 5.0},  # +5pp → would push to -0.03
        )
        assert result[0].dem_share >= _HE_DEM_SHARE_MIN

    def test_no_pollster_field_left_unchanged(self):
        """Poll with empty pollster string is not adjusted."""
        poll = _make_poll(pollster="", dem_share=0.52)
        result = apply_house_effect_correction(
            [poll],
            sb_house_effects={},
            bias_538={},
            empirical_bias={"": 99.0},  # Even if empty key exists, skip
        )
        assert result[0].dem_share == pytest.approx(0.52, abs=1e-6)

    def test_returns_new_copies_not_mutated_originals(self):
        """Input PollObservation objects are not mutated."""
        poll = _make_poll(pollster="TestFirm", dem_share=0.50)
        original_share = poll.dem_share
        apply_house_effect_correction(
            [poll],
            sb_house_effects={},
            bias_538={},
            empirical_bias={"TestFirm": 2.0},
        )
        # Original must be unchanged
        assert poll.dem_share == pytest.approx(original_share, abs=1e-6)

    def test_multiple_polls_each_corrected_independently(self):
        """Each poll in a list is corrected by its own pollster's bias."""
        polls = [
            _make_poll(pollster="DemFirm", dem_share=0.55),
            _make_poll(pollster="RepFirm", dem_share=0.45),
        ]
        bias = {"DemFirm": 3.0, "RepFirm": -2.0}
        result = apply_house_effect_correction(
            polls, sb_house_effects={}, bias_538={}, empirical_bias=bias
        )
        assert result[0].dem_share == pytest.approx(0.55 - 0.03, abs=1e-6)
        assert result[1].dem_share == pytest.approx(0.45 + 0.02, abs=1e-6)

    def test_empty_poll_list_returns_empty(self):
        """Empty input produces empty output."""
        result = apply_house_effect_correction(
            [], sb_house_effects={}, bias_538={}, empirical_bias={}
        )
        assert result == []


# ---------------------------------------------------------------------------
# Cache reset
# ---------------------------------------------------------------------------


class TestResetHouseEffectCache:
    def test_reset_clears_empirical_cache(self, tmp_path):
        """After reset, a fresh load picks up new empirical data."""
        import src.propagation.house_effects as he_mod

        # Prime cache with custom data
        path = _write_accuracy_json(
            [_make_accuracy_entry("PollsterA", mean_error_pp=1.0, n_polls=5)],
            tmp_path / "accuracy.json",
        )
        he_mod._EMPIRICAL_BIAS = load_empirical_house_effects(path)
        assert he_mod._EMPIRICAL_BIAS is not None

        # Reset should clear it
        reset_house_effect_cache()
        assert he_mod._EMPIRICAL_BIAS is None


# ---------------------------------------------------------------------------
# Integration: empirical bias through apply_all_weights
# ---------------------------------------------------------------------------


class TestEmpiricalBiasIntegration:
    """Verify that empirical house effects flow through apply_all_weights.

    We inject the empirical bias directly into the module-level cache so the
    integration test doesn't depend on the accuracy file being on disk.
    """

    def test_empirical_bias_applied_via_apply_all_weights(self):
        """Empirical bias adjusts dem_share even through the full pipeline."""
        import src.propagation.house_effects as he_mod

        # Inject empirical bias for our test pollster
        he_mod._EMPIRICAL_BIAS = {"BiasedFirm": 4.0}  # +4pp D-lean
        he_mod._SB_HOUSE_EFFECTS = {}
        he_mod._538_BIAS = {}

        poll = _make_poll(pollster="BiasedFirm", dem_share=0.54, date="2026-10-01")
        result = apply_all_weights(
            [poll],
            reference_date="2026-10-01",
            apply_house_effects=True,
            apply_quality=False,
            use_silver_bulletin=False,
            use_primary_discount=False,
        )

        # After correction, dem_share should be 0.54 - 0.04 = 0.50
        assert result[0].dem_share == pytest.approx(0.50, abs=1e-6)

    def test_no_house_effects_flag_skips_empirical(self):
        """apply_house_effects=False bypasses empirical correction."""
        import src.propagation.house_effects as he_mod

        he_mod._EMPIRICAL_BIAS = {"BiasedFirm": 4.0}
        he_mod._SB_HOUSE_EFFECTS = {}
        he_mod._538_BIAS = {}

        poll = _make_poll(pollster="BiasedFirm", dem_share=0.54, date="2026-10-01")
        result = apply_all_weights(
            [poll],
            reference_date="2026-10-01",
            apply_house_effects=False,
            apply_quality=False,
            use_silver_bulletin=False,
            use_primary_discount=False,
        )

        # No house effect applied — dem_share unchanged
        assert result[0].dem_share == pytest.approx(0.54, abs=1e-6)

    def test_load_from_real_accuracy_file(self):
        """load_empirical_house_effects works with the real pollster_accuracy.json if present."""
        from pathlib import Path
        from src.propagation.house_effects import _DEFAULT_ACCURACY_PATH

        if not _DEFAULT_ACCURACY_PATH.exists():
            pytest.skip("Real pollster_accuracy.json not available in test environment")

        result = load_empirical_house_effects()
        # Should have loaded at least some pollsters
        assert len(result) > 0
        # All values must be floats (bias in pp)
        for name, bias in result.items():
            assert isinstance(bias, float), f"Bias for {name!r} is not float: {bias!r}"
