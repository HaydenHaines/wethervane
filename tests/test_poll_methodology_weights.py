"""Tests for methodology-based poll quality weighting.

Covers:
  1. methodology_to_multiplier: correct multipliers for each methodology value
  2. apply_methodology_weights: n_sample adjusted correctly
  3. load_methodology_weights: loads from prediction_params.json correctly
  4. apply_all_weights: methodology step applied when poll_methodologies supplied
  5. prepare_polls: methodology tag extracted from poll dict and applied
  6. prediction_params.json: methodology_weights section present and valid
"""

from __future__ import annotations

import json
from copy import copy
from pathlib import Path

import numpy as np
import pytest

from src.propagation.poll_methodology import (
    _DEFAULT_METHODOLOGY_WEIGHTS,
    _MISSING_METHODOLOGY_MULTIPLIER,
    _VALID_METHODOLOGIES,
    apply_methodology_weights,
    load_methodology_weights,
    methodology_to_multiplier,
)
from src.propagation.poll_pipeline import apply_all_weights
from src.propagation.propagate_polls import PollObservation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = PROJECT_ROOT / "data" / "config" / "prediction_params.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_poll(
    n_sample: int = 1000,
    dem_share: float = 0.50,
    date: str = "2026-04-04",
    geography: str = "FL",
    pollster: str = "TestPollster",
    race: str = "2026 FL Senate",
) -> PollObservation:
    return PollObservation(
        geography=geography,
        dem_share=dem_share,
        n_sample=n_sample,
        race=race,
        date=date,
        pollster=pollster,
        geo_level="state",
    )


# ---------------------------------------------------------------------------
# 1. methodology_to_multiplier
# ---------------------------------------------------------------------------


class TestMethodologyToMultiplier:
    """Unit tests for the multiplier lookup function."""

    def test_phone_multiplier_is_highest(self):
        """Phone polls should get the highest multiplier (> 1.0)."""
        m = methodology_to_multiplier("phone")
        assert m > 1.0, f"Expected phone multiplier > 1.0, got {m}"

    def test_ivr_multiplier_is_lowest(self):
        """IVR polls should get the lowest multiplier (< 1.0)."""
        m = methodology_to_multiplier("IVR")
        assert m < 1.0, f"Expected IVR multiplier < 1.0, got {m}"

    def test_ordering_phone_gt_mixed_gt_online_gt_ivr(self):
        """phone > mixed > online > IVR in multiplier order."""
        phone = methodology_to_multiplier("phone")
        mixed = methodology_to_multiplier("mixed")
        online = methodology_to_multiplier("online")
        ivr = methodology_to_multiplier("IVR")
        assert phone > mixed, f"phone={phone} should beat mixed={mixed}"
        assert mixed > online, f"mixed={mixed} should beat online={online}"
        assert online > ivr, f"online={online} should beat IVR={ivr}"

    def test_unknown_is_neutral(self):
        """Unknown methodology should return 1.0 (no adjustment)."""
        assert methodology_to_multiplier("unknown") == pytest.approx(1.0)

    def test_none_is_neutral(self):
        """None (missing methodology) should return the missing-methodology default (1.0)."""
        assert methodology_to_multiplier(None) == pytest.approx(_MISSING_METHODOLOGY_MULTIPLIER)

    def test_empty_string_is_neutral(self):
        """Empty string should be treated the same as None."""
        assert methodology_to_multiplier("") == pytest.approx(_MISSING_METHODOLOGY_MULTIPLIER)

    def test_unrecognized_value_is_neutral(self):
        """An unrecognized methodology string falls back to 1.0 (neutral)."""
        assert methodology_to_multiplier("mail") == pytest.approx(1.0)
        assert methodology_to_multiplier("random-mode") == pytest.approx(1.0)

    def test_custom_weights_override_defaults(self):
        """Caller-supplied weights should override defaults."""
        custom = {"phone": 2.0, "online": 0.5}
        assert methodology_to_multiplier("phone", weights=custom) == pytest.approx(2.0)
        assert methodology_to_multiplier("online", weights=custom) == pytest.approx(0.5)

    def test_default_values_match_spec(self):
        """Verify the specific multiplier values match the documented spec."""
        assert methodology_to_multiplier("phone") == pytest.approx(1.15)
        assert methodology_to_multiplier("mixed") == pytest.approx(1.05)
        assert methodology_to_multiplier("online") == pytest.approx(0.90)
        assert methodology_to_multiplier("IVR") == pytest.approx(0.85)
        assert methodology_to_multiplier("unknown") == pytest.approx(1.00)


# ---------------------------------------------------------------------------
# 2. apply_methodology_weights
# ---------------------------------------------------------------------------


class TestApplyMethodologyWeights:
    """Unit tests for the batch weighting function."""

    def test_phone_poll_gets_boosted(self):
        """Phone polls should receive a higher n_sample than they started with."""
        poll = _make_poll(n_sample=1000)
        result = apply_methodology_weights([poll], ["phone"])
        assert result[0].n_sample > 1000

    def test_ivr_poll_gets_reduced(self):
        """IVR polls should receive a lower n_sample than they started with."""
        poll = _make_poll(n_sample=1000)
        result = apply_methodology_weights([poll], ["IVR"])
        assert result[0].n_sample < 1000

    def test_unknown_poll_unchanged(self):
        """Unknown methodology should not change n_sample."""
        poll = _make_poll(n_sample=1000)
        result = apply_methodology_weights([poll], ["unknown"])
        assert result[0].n_sample == 1000

    def test_none_methodology_unchanged(self):
        """None methodology should not change n_sample."""
        poll = _make_poll(n_sample=1000)
        result = apply_methodology_weights([poll], [None])
        assert result[0].n_sample == 1000

    def test_correct_math_phone(self):
        """Phone multiplier 1.15 × 1000 = 1150 (rounded)."""
        poll = _make_poll(n_sample=1000)
        result = apply_methodology_weights([poll], ["phone"])
        expected = int(round(1000 * 1.15))
        assert result[0].n_sample == expected

    def test_correct_math_ivr(self):
        """IVR multiplier 0.85 × 1000 = 850 (rounded)."""
        poll = _make_poll(n_sample=1000)
        result = apply_methodology_weights([poll], ["IVR"])
        expected = int(round(1000 * 0.85))
        assert result[0].n_sample == expected

    def test_other_fields_unchanged(self):
        """dem_share, geography, pollster should not be altered."""
        poll = _make_poll(n_sample=1000, dem_share=0.55, geography="GA")
        result = apply_methodology_weights([poll], ["online"])
        assert result[0].dem_share == pytest.approx(0.55)
        assert result[0].geography == "GA"

    def test_returns_new_objects(self):
        """apply_methodology_weights must return new copies, not mutate originals."""
        poll = _make_poll(n_sample=1000)
        result = apply_methodology_weights([poll], ["phone"])
        assert result[0] is not poll
        assert poll.n_sample == 1000  # original unchanged

    def test_multiple_polls_different_methodologies(self):
        """Each poll gets its own multiplier independently."""
        polls = [_make_poll(n_sample=1000), _make_poll(n_sample=1000)]
        methods = ["phone", "IVR"]
        result = apply_methodology_weights(polls, methods)
        assert result[0].n_sample > result[1].n_sample

    def test_length_mismatch_raises(self):
        """Mismatched list lengths should raise ValueError."""
        polls = [_make_poll(), _make_poll()]
        with pytest.raises(ValueError, match="same length"):
            apply_methodology_weights(polls, ["phone"])

    def test_minimum_n_sample_is_one(self):
        """Even with an extreme downweight, n_sample must be at least 1."""
        poll = _make_poll(n_sample=1)
        result = apply_methodology_weights([poll], ["IVR"])
        assert result[0].n_sample >= 1

    def test_custom_weights(self):
        """Custom weights should override the defaults."""
        poll = _make_poll(n_sample=1000)
        custom = {"phone": 2.0}
        result = apply_methodology_weights([poll], ["phone"], weights=custom)
        assert result[0].n_sample == int(round(1000 * 2.0))


# ---------------------------------------------------------------------------
# 3. load_methodology_weights
# ---------------------------------------------------------------------------


class TestLoadMethodologyWeights:
    """Tests for config file loading."""

    def test_loads_from_real_params_file(self):
        """load_methodology_weights should return non-empty dict from real config."""
        weights = load_methodology_weights(PARAMS_PATH)
        assert isinstance(weights, dict)
        assert len(weights) > 0

    def test_real_config_contains_all_methodologies(self):
        """All five methodology types should be present in the loaded weights."""
        weights = load_methodology_weights(PARAMS_PATH)
        for m in ["phone", "mixed", "online", "IVR", "unknown"]:
            assert m in weights, f"Methodology {m!r} missing from loaded weights"

    def test_real_config_values_match_defaults(self):
        """Values in prediction_params.json should match the module defaults."""
        weights = load_methodology_weights(PARAMS_PATH)
        for key, default_val in _DEFAULT_METHODOLOGY_WEIGHTS.items():
            assert weights[key] == pytest.approx(default_val), (
                f"Loaded weight for {key!r} ({weights[key]}) differs from "
                f"default ({default_val})"
            )

    def test_missing_file_returns_defaults(self, tmp_path):
        """When the params file doesn't exist, return the default weights."""
        nonexistent = tmp_path / "no_such_file.json"
        weights = load_methodology_weights(nonexistent)
        assert weights == _DEFAULT_METHODOLOGY_WEIGHTS

    def test_missing_key_returns_defaults(self, tmp_path):
        """When methodology_weights key is absent, return the default weights."""
        params = {"poll_weighting": {"half_life_days": 30.0}}
        path = tmp_path / "params.json"
        path.write_text(json.dumps(params), encoding="utf-8")
        weights = load_methodology_weights(path)
        assert weights == _DEFAULT_METHODOLOGY_WEIGHTS

    def test_partial_override_merges_with_defaults(self, tmp_path):
        """Partial config merges with defaults; missing keys keep defaults."""
        params = {"poll_weighting": {"methodology_weights": {"phone": 1.25}}}
        path = tmp_path / "params.json"
        path.write_text(json.dumps(params), encoding="utf-8")
        weights = load_methodology_weights(path)
        assert weights["phone"] == pytest.approx(1.25)
        # Non-overridden keys fall back to defaults
        assert weights["IVR"] == pytest.approx(_DEFAULT_METHODOLOGY_WEIGHTS["IVR"])

    def test_malformed_json_returns_defaults(self, tmp_path):
        """Malformed JSON file should not raise; return defaults."""
        path = tmp_path / "bad.json"
        path.write_text("{not valid json", encoding="utf-8")
        weights = load_methodology_weights(path)
        assert weights == _DEFAULT_METHODOLOGY_WEIGHTS


# ---------------------------------------------------------------------------
# 4. apply_all_weights integration
# ---------------------------------------------------------------------------


class TestApplyAllWeightsMethodology:
    """Verify methodology step integrates correctly into the pipeline."""

    def test_methodology_step_applied_when_methodologies_supplied(self):
        """With phone methodology, n_sample should be boosted vs no methodology."""
        poll = _make_poll(n_sample=1000)
        ref = "2026-04-04"

        result_method = apply_all_weights(
            [poll], reference_date=ref,
            apply_quality=False, apply_house_effects=False, use_primary_discount=False,
            poll_methodologies=["phone"],
        )
        result_no_method = apply_all_weights(
            [poll], reference_date=ref,
            apply_quality=False, apply_house_effects=False, use_primary_discount=False,
            poll_methodologies=None,
        )
        assert result_method[0].n_sample > result_no_method[0].n_sample

    def test_methodology_step_skipped_when_none(self):
        """When poll_methodologies is None, n_sample is unchanged by methodology."""
        poll = _make_poll(n_sample=1000)
        ref = "2026-04-04"
        result = apply_all_weights(
            [poll], reference_date=ref,
            apply_quality=False, apply_house_effects=False, use_primary_discount=False,
            poll_methodologies=None,
        )
        # No methodology adjustment applied; n_sample reflects only time decay
        # (poll is same-day so decay factor ~1.0 → unchanged)
        assert result[0].n_sample == 1000

    def test_apply_methodology_false_skips_step(self):
        """apply_methodology=False should skip the methodology step."""
        poll = _make_poll(n_sample=1000)
        ref = "2026-04-04"
        result = apply_all_weights(
            [poll], reference_date=ref,
            apply_quality=False, apply_house_effects=False, use_primary_discount=False,
            poll_methodologies=["IVR"],
            apply_methodology=False,
        )
        assert result[0].n_sample == 1000

    def test_methodology_stacks_on_top_of_time_decay(self):
        """Methodology multiplier stacks multiplicatively with time decay."""
        # Poll is 30 days old → decay = 0.5 → n=500; then phone 1.15x → ~575
        poll = _make_poll(n_sample=1000, date="2026-03-05")
        ref = "2026-04-04"

        result_phone = apply_all_weights(
            [poll], reference_date=ref, half_life_days=30.0,
            apply_quality=False, apply_house_effects=False, use_primary_discount=False,
            poll_methodologies=["phone"],
        )
        result_no_method = apply_all_weights(
            [poll], reference_date=ref, half_life_days=30.0,
            apply_quality=False, apply_house_effects=False, use_primary_discount=False,
            poll_methodologies=None,
        )
        # Phone-boosted result should be higher than un-adjusted result
        assert result_phone[0].n_sample > result_no_method[0].n_sample

    def test_phone_higher_than_ivr_in_pipeline(self):
        """Same poll, phone methodology vs IVR should produce different n_sample."""
        poll = _make_poll(n_sample=1000)
        ref = "2026-04-04"
        phone_result = apply_all_weights(
            [poll], reference_date=ref,
            apply_quality=False, apply_house_effects=False, use_primary_discount=False,
            poll_methodologies=["phone"],
        )
        ivr_result = apply_all_weights(
            [poll], reference_date=ref,
            apply_quality=False, apply_house_effects=False, use_primary_discount=False,
            poll_methodologies=["IVR"],
        )
        assert phone_result[0].n_sample > ivr_result[0].n_sample


# ---------------------------------------------------------------------------
# 5. prepare_polls integration
# ---------------------------------------------------------------------------


class TestPreparePollsMethodology:
    """Verify prepare_polls extracts and applies methodology from poll dicts."""

    def test_prepare_polls_applies_phone_boost(self):
        """A phone poll should have higher n_sample after prepare_polls."""
        from src.prediction.forecast_engine import prepare_polls

        polls_phone = {
            "2026 FL Senate": [{
                "dem_share": 0.52,
                "n_sample": 1000,
                "state": "FL",
                "date": "2026-04-04",
                "pollster": "Quinnipiac University",
                "notes": "",
                "methodology": "phone",
            }]
        }
        polls_ivr = {
            "2026 FL Senate": [{
                "dem_share": 0.52,
                "n_sample": 1000,
                "state": "FL",
                "date": "2026-04-04",
                "pollster": "Cygnal",
                "notes": "",
                "methodology": "IVR",
            }]
        }

        result_phone = prepare_polls(
            polls_phone, "2026-04-04",
            methodology_weights=_DEFAULT_METHODOLOGY_WEIGHTS,
        )
        result_ivr = prepare_polls(
            polls_ivr, "2026-04-04",
            methodology_weights=_DEFAULT_METHODOLOGY_WEIGHTS,
        )

        n_phone = result_phone["2026 FL Senate"][0]["n_sample"]
        n_ivr = result_ivr["2026 FL Senate"][0]["n_sample"]

        assert n_phone > n_ivr, (
            f"Phone poll should have higher effective n than IVR; "
            f"got phone={n_phone}, IVR={n_ivr}"
        )

    def test_prepare_polls_missing_methodology_is_neutral(self):
        """Polls without a methodology key should be treated as neutral (1.0)."""
        from src.prediction.forecast_engine import prepare_polls

        polls_no_method = {
            "2026 FL Senate": [{
                "dem_share": 0.52,
                "n_sample": 1000,
                "state": "FL",
                "date": "2026-04-04",
                "pollster": "TestPollster",
                "notes": "",
                # no "methodology" key
            }]
        }
        polls_unknown = {
            "2026 FL Senate": [{
                "dem_share": 0.52,
                "n_sample": 1000,
                "state": "FL",
                "date": "2026-04-04",
                "pollster": "TestPollster",
                "notes": "",
                "methodology": "unknown",
            }]
        }

        result_none = prepare_polls(
            polls_no_method, "2026-04-04",
            methodology_weights=_DEFAULT_METHODOLOGY_WEIGHTS,
        )
        result_unknown = prepare_polls(
            polls_unknown, "2026-04-04",
            methodology_weights=_DEFAULT_METHODOLOGY_WEIGHTS,
        )

        n_none = result_none["2026 FL Senate"][0]["n_sample"]
        n_unknown = result_unknown["2026 FL Senate"][0]["n_sample"]

        # Both should produce the same n_sample (neutral multiplier)
        assert n_none == n_unknown

    def test_prepare_polls_methodology_none_skips_step(self):
        """When methodology_weights is None, no methodology adjustment is applied."""
        from src.prediction.forecast_engine import prepare_polls

        polls = {
            "2026 FL Senate": [{
                "dem_share": 0.52,
                "n_sample": 1000,
                "state": "FL",
                "date": "2026-04-04",
                "pollster": "TestPollster",
                "notes": "",
                "methodology": "IVR",
            }]
        }

        result_no_mw = prepare_polls(
            polls, "2026-04-04",
            methodology_weights=None,  # skip methodology step
        )
        result_with_mw = prepare_polls(
            polls, "2026-04-04",
            methodology_weights=_DEFAULT_METHODOLOGY_WEIGHTS,
        )

        n_no_mw = result_no_mw["2026 FL Senate"][0]["n_sample"]
        n_with_mw = result_with_mw["2026 FL Senate"][0]["n_sample"]

        # Without methodology weights: IVR poll gets no penalty → n=1000
        # With methodology weights: IVR poll is penalized → n<1000
        assert n_no_mw > n_with_mw


# ---------------------------------------------------------------------------
# 6. prediction_params.json schema validation
# ---------------------------------------------------------------------------


class TestPredictionParamsMethodologySection:
    """prediction_params.json contains a valid methodology_weights section."""

    def test_params_file_exists(self):
        assert PARAMS_PATH.exists(), f"prediction_params.json not found: {PARAMS_PATH}"

    def test_methodology_weights_present(self):
        params = json.loads(PARAMS_PATH.read_text())
        pw = params.get("poll_weighting", {})
        assert "methodology_weights" in pw, (
            "prediction_params.json poll_weighting section is missing 'methodology_weights'"
        )

    def test_all_required_methodology_keys_present(self):
        params = json.loads(PARAMS_PATH.read_text())
        mw = params["poll_weighting"]["methodology_weights"]
        for method in ["phone", "mixed", "online", "IVR", "unknown"]:
            assert method in mw, f"methodology_weights missing key {method!r}"

    def test_all_values_are_positive_floats(self):
        params = json.loads(PARAMS_PATH.read_text())
        mw = params["poll_weighting"]["methodology_weights"]
        for k, v in mw.items():
            assert isinstance(v, (int, float)), f"{k}: expected numeric value, got {type(v)}"
            assert v > 0, f"{k}: weight must be positive, got {v}"

    def test_phone_weight_greater_than_ivr_weight(self):
        """Config must encode phone > IVR ordering."""
        params = json.loads(PARAMS_PATH.read_text())
        mw = params["poll_weighting"]["methodology_weights"]
        assert mw["phone"] > mw["IVR"], (
            f"phone weight ({mw['phone']}) should exceed IVR ({mw['IVR']})"
        )

    def test_unknown_weight_is_one(self):
        """unknown methodology should be neutral (1.0)."""
        params = json.loads(PARAMS_PATH.read_text())
        mw = params["poll_weighting"]["methodology_weights"]
        assert mw["unknown"] == pytest.approx(1.0), (
            f"unknown weight should be 1.0 (neutral), got {mw['unknown']}"
        )

    def test_existing_params_still_intact(self):
        """Adding methodology_weights must not break existing config keys."""
        params = json.loads(PARAMS_PATH.read_text())
        pw = params["poll_weighting"]
        assert "half_life_days" in pw
        assert "pre_primary_discount" in pw
        assert "use_pollster_rmse_weights" in pw
        assert "forecast" in params
        assert "lam" in params["forecast"]
