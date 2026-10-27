"""Source-pinned A2018 aggregate check with guarded synthetic interaction fixtures.

No model import, owner job, context/problem rows, or forecast states are read.
This is a research diagnostic, not an admission or qualification gate.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

HERE = Path(__file__).resolve().parent
PLAN = Path("/home/hayden/projects/wethervane-redesign-20261006/integration/research/senate-stack-integration-20261020/national-multicycle-execution-astra-v1/A2018/UNADMITTED-plan.json")
LEDGER = Path("/home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024/m01-a2018-observation-ownership-luna-v2/aggregate-ledger.json")
SOURCE = Path("/home/hayden/projects/wethervane-redesign-20261006/integration/research/senate-stack-integration-20261020/national-2018-gpu-preparation-root-v1/gpu-production-v1/attempt1/source-snapshots/117-components.py")
LEGACY_RESULT = HERE / "legacy-result-preserved.json"
PRIOR_REVIEW = Path("/home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024/m01-a2018-layer-interaction-review-sol-v1/REVIEW.md")
PRIOR_REVIEW_TEST = PRIOR_REVIEW.parent / "test_review.py"
PRIOR_REPRODUCED_RESULT = PRIOR_REVIEW.parent / "reproduced-result.json"
PRIOR_V2 = HERE.parent / "m01-a2018-layer-interaction-audit-luna-v2"

PINNED = {
    PLAN: ("3e33169c4e1b0f2811d43d3afbb8d8a8f74132b132db866d65ca119567e27003", 87475),
    LEDGER: ("077e620e255fdb12474497f040afa061fcebb17c506b173692ced2acb438e423", 50261),
    SOURCE: ("1574e7831344009f08470b9287b257556a9c031fb7e33e29938a70d51b1a854d", 27626),
    HERE / "legacy-result-preserved.json": ("c5795829d7ded8d616baf9213d9091c9942318c4f0f8bc02f0100c1bebeba0db", 3198),
    PRIOR_REVIEW: ("017017f3facbe88a797bee45be553e3cfe5e635ded1cca28063582d5893e0385", 6494),
    PRIOR_REVIEW_TEST: ("0c0ddb01d716ddc684f87261603d55481e6e68e15a04f1b9eac145acbc7f2d41", 2226),
    PRIOR_REPRODUCED_RESULT: ("c5795829d7ded8d616baf9213d9091c9942318c4f0f8bc02f0100c1bebeba0db", 3198),
    PRIOR_V2 / "check.py": ("21b77fb57d9ead011be9a12c5b4df6d4a27b3985925624e8762d7f54fbcc87b3", 16305),
    PRIOR_V2 / "test_check.py": ("854165e363f2a339ebee146874faa7590149284b7fd8f9ef15522654bb920ea0", 5443),
    PRIOR_V2 / "result.json": ("da8b2f49c9eb4ddcb1bf00448fb9d85e3751a159cfa75009af45a60c980e22e8", 8688),
    PRIOR_V2 / "source-pins.json": ("5bf3fc920e5044fc6263bd44582d6380ea802e968386f16a6fbc754ab81b7d76", 3338),
}
EXPECTED_ROLES = [
    ["population", "protected_full_parent_exposure_v1"],
    ["community", "demographic_profile_drift_v1"],
    ["fundamentals", "lean_carryover_context_incumbency_v1"],
    ["environment", "separated_lkj2_cycle_student5_race_v1"],
    ["participation", "profile_turnout_response_observed_total_gamma_v1"],
    ["choice", "candidate_iid_profile_choice_v1"],
    ["counts_rules", "single_mark_category_allocation_poisson_v1"],
    ["measurement", "joint_CES_ranked_intended_report_F1a_pair_v1"],
]

# Reviewed synthetic examples. Both are invented, not fitted parameter states.
ORIGINAL_PAIRS = ((1.4, 0.0), (0.0, 1.2))
CENTERED_PAIRS = ((0.7, -0.7), (-0.6, 0.6))
OTHER_UTILITY_ENDS = (-1.0, 1.0)
WEIGHT_CASES = ((1.0, 1.0), (9.0, 1.0))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_pinned(path: Path, expected_sha256: str, expected_bytes: int | None = None) -> bytes:
    raw = Path(path).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    require(actual == expected_sha256, f"Pinned source hash differs: {path}")
    if expected_bytes is not None:
        require(len(raw) == expected_bytes, f"Pinned source size differs: {path}")
    return raw


def _finite_real(value: Any, label: str) -> float:
    require(type(value) in (int, float), f"{label} must be a finite real number")
    try:
        converted = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label} must be representable as a finite float") from exc
    require(math.isfinite(converted), f"{label} must be finite")
    return converted


def _logsumexp(values: Sequence[float], label: str) -> float:
    require(len(values) > 0, f"{label} must be nonempty")
    maximum = max(values)
    require(math.isfinite(maximum), f"{label} has no finite terms")
    # Differences can be -inf only when finite endpoints exceed representable
    # subtraction; exp(-inf)=0 is the correct limiting contribution.
    total = sum(math.exp(value - maximum) for value in values)
    require(math.isfinite(total) and total > 0.0, f"{label} is not numerically representable")
    result = maximum + math.log(total)
    require(math.isfinite(result), f"{label} is not numerically representable")
    return result


def softmax3(d: float, r: float, o: float) -> tuple[float, float, float]:
    """Stable three-category softmax; probabilities may underflow to zero."""
    utilities = tuple(_finite_real(x, "utility") for x in (d, r, o))
    maximum = max(utilities)
    exponentials = tuple(math.exp(x - maximum) for x in utilities)
    total = sum(exponentials)
    require(math.isfinite(total) and total > 0.0, "softmax denominator is invalid")
    probabilities = tuple(value / total for value in exponentials)
    require(all(math.isfinite(p) and 0.0 <= p <= 1.0 for p in probabilities),
            "softmax probabilities are invalid")
    return probabilities  # type: ignore[return-value]


def ordinary_poll_logit(weights: Sequence[float], pair_utilities: Sequence[Sequence[float]],
                        other_utility: float, response: float, event_g: float) -> float:
    """Compute log aggregate D/R rates in log space for a synthetic poll.

    The common response and event_g scales are required to be finite/positive,
    then cancel symbolically from this ordinary pair log ratio. This avoids
    multiplying rates into overflow/underflow and does not claim those scales
    cancel in other likelihoods.
    """
    require(isinstance(weights, (tuple, list)) and len(weights) > 0,
            "weights must be a nonempty vector")
    require(isinstance(pair_utilities, (tuple, list)) and len(pair_utilities) > 0,
            "profile utilities must be a nonempty vector")
    require(len(weights) == len(pair_utilities), "profile and weight vector lengths differ")
    w = tuple(_finite_real(value, "profile weight") for value in weights)
    require(all(value > 0.0 for value in w), "profile weights must be positive")
    profiles: list[tuple[float, float, float]] = []
    for profile in pair_utilities:
        require(isinstance(profile, (tuple, list)) and len(profile) == 2,
                "each profile must contain exactly two pair utilities")
        d = _finite_real(profile[0], "D utility")
        r = _finite_real(profile[1], "R utility")
        profiles.append((d, r, _finite_real(other_utility, "other-candidate utility")))
    response_value = _finite_real(response, "response scale")
    event_value = _finite_real(event_g, "event scale")
    require(response_value > 0.0 and event_value > 0.0,
            "response and event scales must be positive")

    log_weights = tuple(math.log(value) for value in w)
    log_d_terms: list[float] = []
    log_r_terms: list[float] = []
    for log_weight, (d, r, o) in zip(log_weights, profiles, strict=True):
        # Center before adding the log normalizer. Computing max+log(sumexp)
        # first and then subtracting it from d/r loses the normalizer when all
        # utilities share a large gauge offset (for example, around 1e16).
        maximum = max(d, r, o)
        relative_d, relative_r, relative_o = d - maximum, r - maximum, o - maximum
        log_normalizer = _logsumexp((relative_d, relative_r, relative_o),
                                    "centered profile softmax normalizer")
        log_d_terms.append(log_weight + relative_d - log_normalizer)
        log_r_terms.append(log_weight + relative_r - log_normalizer)
    # Center the two candidate term sets together as well, so their shared
    # large negative log-rate offset is removed before the final subtraction.
    common_peak = max(log_d_terms + log_r_terms)
    scaled_log_d = _logsumexp(tuple(term - common_peak for term in log_d_terms),
                              "scaled aggregate D rate")
    scaled_log_r = _logsumexp(tuple(term - common_peak for term in log_r_terms),
                              "scaled aggregate R rate")
    result = scaled_log_d - scaled_log_r
    require(math.isfinite(result), "ordinary poll logit is not finite")
    return result


def _source_facts(plan: dict[str, Any], ledger: dict[str, Any], source: str) -> dict[str, Any]:
    config = plan["provider_contract"]["composition"]["config"]
    require(config.get("selections") == EXPECTED_ROLES, "plan selections differ")
    require(plan.get("parent_masses") == [0.125] * 8, "plan parent masses differ")
    require(plan.get("forecast_qualified") is False and plan.get("national_complete") is False
            and plan.get("publication_authorized") is False, "plan qualification flags differ")
    require(ledger["selected_composition"].get("selections") == EXPECTED_ROLES,
            "ledger selections differ")
    masses = [item["mass"] for item in ledger["parent_masses"]]
    require(masses == [0.125] * 8, "ledger parent masses differ")
    counts = ledger["counts"]
    require(counts.get("poll_blocks") == 1 and counts.get("poll_rows") == 274,
            "poll block counts differ")
    require(ledger["poll_row_counts_by_event"].get("event_az2018_general") == 20,
            "Arizona poll row count differs")
    require(counts.get("official_count_observations") == 0
            and counts.get("participation_report_observations") == 0,
            "unexpected count/report observations")
    closure = ledger["eventG_likelihood_closure"]["closure_checks"]
    expected_closure = {
        "open_person_absent": True,
        "core_ranked_observation_count": 0,
        "official_count_observation_count": 0,
        "participation_report_count": 0,
        "ranked_observation_count": 0,
        "ranked_poll_channel_count": 0,
        "runoff_observation_count": 0,
        "survey_loop_eventG_free": True,
        "ordinary_poll_formula_verified": True,
        "contest_response_is_scalar": True,
    }
    for key, expected in expected_closure.items():
        require(closure.get(key) == expected, f"ledger closure differs: {key}")
    fragments = (
        'intensity = v["exposure"][:, :, None] * v["community"] * q',
        'pm.math.softmax(utilities[name], axis=-1)',
        'response * v["candidate_probability"][name]',
        'v["cell_intensity"][c.event_index[contest["event_id"]], :, :, None] * categories[name]',
        'v["eventG"][c.event_index[contest["event_id"]]] * v["base_category_rates"][name]',
        'means.append(pt.log(rates[first]) - pt.log(rates[second]))',
    )
    for fragment in fragments:
        require(fragment in source, f"source anchor missing: {fragment}")
    return {
        "selected_roles": [role for role, _ in EXPECTED_ROLES],
        "selected_choice": EXPECTED_ROLES[5][1],
        "protected_parent_masses": plan["parent_masses"],
        "az2018_paired_poll_rows": 20,
        "one_poll_covariance_block_all_rows": 274,
        "official_counts": 0,
        "participation_reports": 0,
        "forecast_qualified": False,
        "source_closure_checks_passed": [
            "open_person_absent", "no_official_counts_or_reports", "no_ranked_or_runoff_observations",
            "survey_loop_eventG_free", "ordinary_poll_formula_verified", "contest_response_is_scalar",
        ],
        "aggregate_eventG_closure_claim_present": ledger["eventG_likelihood_closure"]["claim"].startswith(
            "For the exact pinned A2018 problem/source"),
        "compiled_graph_authenticated": False,
    }


def _fixture_sensitivity(pairs: Sequence[Sequence[float]]) -> dict[str, Any]:
    deltas: dict[str, float] = {}
    for label, weights in (("equal", WEIGHT_CASES[0]), ("shifted", WEIGHT_CASES[1])):
        before = ordinary_poll_logit(weights, pairs, OTHER_UTILITY_ENDS[0], 0.72, 1.5)
        after = ordinary_poll_logit(weights, pairs, OTHER_UTILITY_ENDS[1], 0.72, 1.5)
        deltas[label] = after - before
    return {
        "pair_utilities_by_profile": [list(row) for row in pairs],
        "other_utility_change": list(OTHER_UTILITY_ENDS),
        "relative_profile_intensity_equal": list(WEIGHT_CASES[0]),
        "relative_profile_intensity_participation_shift": list(WEIGHT_CASES[1]),
        "ordinary_poll_logit_before_equal": ordinary_poll_logit(WEIGHT_CASES[0], pairs, -1.0, 0.72, 1.5),
        "ordinary_poll_logit_after_equal": ordinary_poll_logit(WEIGHT_CASES[0], pairs, 1.0, 0.72, 1.5),
        "minor_utility_delta_equal": deltas["equal"],
        "ordinary_poll_logit_before_shifted": ordinary_poll_logit(WEIGHT_CASES[1], pairs, -1.0, 0.72, 1.5),
        "ordinary_poll_logit_after_shifted": ordinary_poll_logit(WEIGHT_CASES[1], pairs, 1.0, 0.72, 1.5),
        "minor_utility_delta_shifted": deltas["shifted"],
        "difference_in_differences": deltas["shifted"] - deltas["equal"],
        "input_status": "invented utility and relative profile-intensity fixture; not source rows, posterior states, or a forecast",
    }


def build_result() -> dict[str, Any]:
    raw: dict[Path, bytes] = {}
    for path, (digest, size) in PINNED.items():
        raw[path] = read_pinned(path, digest, size)
    plan = json.loads(raw[PLAN])
    ledger = json.loads(raw[LEDGER])
    source = raw[SOURCE].decode("utf-8")
    old_result = json.loads(raw[LEGACY_RESULT])
    aggregate = _source_facts(plan, ledger, source)
    require(ledger["eventG_likelihood_closure"]["claim"].startswith(
        "For the exact pinned A2018 problem/source"), "ledger closure claim differs")

    original = _fixture_sensitivity(ORIGINAL_PAIRS)
    centered = _fixture_sensitivity(CENTERED_PAIRS)
    require(abs(original["difference_in_differences"] - (-0.014237568016383512)) < 1e-12,
            "original fixture output drifted")
    require(abs(centered["difference_in_differences"] - (-0.0068878407953504705)) < 1e-12,
            "reviewed centered fixture output drifted")
    for weights in WEIGHT_CASES:
        for other in OTHER_UTILITY_ENDS:
            baseline = ordinary_poll_logit(weights, ORIGINAL_PAIRS, other, 0.72, 1.5)
            scaled = ordinary_poll_logit(weights, ORIGINAL_PAIRS, other, 0.19, 6.4)
            require(math.isclose(baseline, scaled, rel_tol=0.0, abs_tol=1e-14),
                    "common scalar cancellation check failed")
    constant_gap = ((1.0, 0.0), (2.0, 1.0))
    constant_delta = (ordinary_poll_logit((1.0, 1.0), constant_gap, 1.0, 0.72, 1.5)
                      - ordinary_poll_logit((1.0, 1.0), constant_gap, -1.0, 0.72, 1.5))
    require(abs(constant_delta) < 1e-14, "constant D/R contrast cancellation failed")
    for weights in WEIGHT_CASES:
        for other in OTHER_UTILITY_ENDS:
            base = ordinary_poll_logit(weights, ORIGINAL_PAIRS, other, 0.72, 1.5)
            scaled_weights = tuple(value * 17.0 for value in weights)
            scaled = ordinary_poll_logit(scaled_weights, ORIGINAL_PAIRS, other, 0.72, 1.5)
            shifted_profiles = tuple((d + 11.0, r + 11.0) for d, r in ORIGINAL_PAIRS)
            shifted = ordinary_poll_logit(weights, shifted_profiles, other + 11.0, 0.72, 1.5)
            require(math.isclose(base, scaled, rel_tol=0.0, abs_tol=1e-14),
                    "common weight scale invariance failed")
            require(math.isclose(base, shifted, rel_tol=0.0, abs_tol=1e-14),
                    "profile common utility shift invariance failed")
    gauge_pairs = ((4.0, -4.0), (-4.0, 4.0))
    gauge_base = ordinary_poll_logit((1.0, 9.0), gauge_pairs, 0.0, 0.72, 1.5)
    gauge_shift = ordinary_poll_logit((1.0, 9.0),
        tuple((d + 1e16, r + 1e16) for d, r in gauge_pairs), 1e16, 0.72, 1.5)
    require(math.isclose(gauge_base, gauge_shift, rel_tol=0.0, abs_tol=1e-14),
            "large common utility gauge invariance failed")

    # Carry forward the frozen legacy output as an unchanged nested object.
    require(old_result.get("schema") == "A2018PinnedAggregateAndInventedInteractionCheckV1",
            "preserved legacy result schema differs")
    legacy_fixture = old_result["invented_equation_sensitivity"]
    require(abs(legacy_fixture["difference_in_differences"] - original["difference_in_differences"]) < 1e-12,
            "preserved legacy result does not match this source-pinned calculation")
    return {
        "schema": "A2018PinnedAggregateAndStableInteractionCheckV3",
        "verified_source_hashes": {str(path): PINNED[path][0] for path in PINNED},
        "aggregate_only_a2018": aggregate,
        "legacy_fixed_fixture_output_preserved": old_result,
        "invented_equation_sensitivity": original,
        "reviewed_centered_contrast_sensitivity": centered,
        "cancellation_checks": {
            "positive_response_and_event_scale_cancellation_checked": True,
            "common_weight_scale_invariance_checked": True,
            "profile_common_utility_shift_invariance_checked": True,
            "large_common_utility_gauge_invariance_checked": True,
            "constant_pair_gap_minor_cancellation_checked": True,
        },
        "limits": [
            "source expression presence and prepared aggregate counts do not authenticate original observations or sample independence",
            "synthetic sensitivities show possible mechanisms, not their magnitude or direction in actual A2018 posterior draws",
            "centered contrast fixture is not a solved full native parameter state or evidence of posterior support",
            "no compiled graph, prediction, calibration, qualification, or weakest-layer ranking",
        ],
    }


def main() -> None:
    print(json.dumps(build_result(), sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
