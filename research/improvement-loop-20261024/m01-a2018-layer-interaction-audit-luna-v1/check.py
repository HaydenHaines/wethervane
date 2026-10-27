"""Pinned A2018 aggregate closure plus invented source-equation interaction check.

Reads only the compact plan, compact aggregate ledger, and selected provider source.
No problem/context arrays, individual records, model imports, or fit artifacts.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

PLAN = Path("/home/hayden/projects/wethervane-redesign-20261006/integration/research/senate-stack-integration-20261020/national-multicycle-execution-astra-v1/A2018/UNADMITTED-plan.json")
LEDGER = Path("/home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024/m01-a2018-observation-ownership-luna-v2/aggregate-ledger.json")
SOURCE = Path("/home/hayden/projects/wethervane-redesign-20261006/integration/research/senate-stack-integration-20261020/national-2018-gpu-preparation-root-v1/gpu-production-v1/attempt1/source-snapshots/117-components.py")
EXPECTED = {
    PLAN: "3e33169c4e1b0f2811d43d3afbb8d8a8f74132b132db866d65ca119567e27003",
    LEDGER: "077e620e255fdb12474497f040afa061fcebb17c506b173692ced2acb438e423",
    SOURCE: "1574e7831344009f08470b9287b257556a9c031fb7e33e29938a70d51b1a854d",
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


def softmax3(d: float, r: float, o: float) -> tuple[float, float, float]:
    vals = (math.exp(d), math.exp(r), math.exp(o))
    total = sum(vals)
    return tuple(x / total for x in vals)


def ordinary_poll_logit(weights: tuple[float, float], pair_utilities, other_utility: float,
                        response: float, event_g: float) -> float:
    """Source-equation toy: log(sum I*G*response*pD) - log(sum I*G*response*pR)."""
    rates = [0.0, 0.0]
    for weight, (d_utility, r_utility) in zip(weights, pair_utilities):
        p_d, p_r, _p_o = softmax3(d_utility, r_utility, other_utility)
        rates[0] += weight * event_g * response * p_d
        rates[1] += weight * event_g * response * p_r
    return math.log(rates[0]) - math.log(rates[1])


def main() -> None:
    raw = {path: path.read_bytes() for path in EXPECTED}
    for path, digest in EXPECTED.items():
        actual = hashlib.sha256(raw[path]).hexdigest()
        assert actual == digest, (str(path), actual, digest)
    plan = json.loads(raw[PLAN])
    ledger = json.loads(raw[LEDGER])
    source = raw[SOURCE].decode("utf-8")
    config = plan["provider_contract"]["composition"]["config"]
    assert config["selections"] == EXPECTED_ROLES
    assert config["selections"][5][1] == "candidate_iid_profile_choice_v1"
    assert plan["parent_masses"] == [0.125] * 8
    assert plan["forecast_qualified"] is False
    assert plan["national_complete"] is False
    assert plan["publication_authorized"] is False
    assert ledger["selected_composition"]["selections"] == EXPECTED_ROLES
    ledger_masses = [item["mass"] for item in ledger["parent_masses"]]
    assert ledger_masses == [0.125] * 8
    assert ledger["counts"]["poll_blocks"] == 1
    assert ledger["counts"]["poll_rows"] == 274
    assert ledger["poll_row_counts_by_event"]["event_az2018_general"] == 20
    assert ledger["counts"]["official_count_observations"] == 0
    assert ledger["counts"]["participation_report_observations"] == 0
    closure = ledger["eventG_likelihood_closure"]["closure_checks"]
    assert closure["open_person_absent"] is True
    assert closure["core_ranked_observation_count"] == 0
    assert closure["official_count_observation_count"] == 0
    assert closure["participation_report_count"] == 0
    assert closure["ranked_observation_count"] == 0
    assert closure["ranked_poll_channel_count"] == 0
    assert closure["survey_loop_eventG_free"] is True
    assert closure["ordinary_poll_formula_verified"] is True
    assert closure["contest_response_is_scalar"] is True

    # Source anchors: shared propensity/intensity, full-candidate softmax, scalar
    # candidate response, event scaling, and ordinary pair log-rate difference.
    for fragment in (
        'intensity = v["exposure"][:, :, None] * v["community"] * q',
        'pm.math.softmax(utilities[name], axis=-1)',
        'response * v["candidate_probability"][name]',
        'v["cell_intensity"][c.event_index[contest["event_id"]], :, :, None] * categories[name]',
        'v["eventG"][c.event_index[contest["event_id"]]] * v["base_category_rates"][name]',
        'means.append(pt.log(rates[first]) - pt.log(rates[second]))',
    ):
        assert fragment in source, fragment

    # Fixed invented utilities: profile-specific D/R gaps differ, so changing
    # the O utility changes profile weights inside aggregate D/R rates.
    heterogeneous = ((1.4, 0.0), (0.0, 1.2))
    equal_weights = (1.0, 1.0)
    shifted_participation = (9.0, 1.0)  # relative cell_intensity, not observed q
    before_equal = ordinary_poll_logit(equal_weights, heterogeneous, -1.0, .72, 1.5)
    after_equal = ordinary_poll_logit(equal_weights, heterogeneous, 1.0, .72, 1.5)
    before_shifted = ordinary_poll_logit(shifted_participation, heterogeneous, -1.0, .72, 1.5)
    after_shifted = ordinary_poll_logit(shifted_participation, heterogeneous, 1.0, .72, 1.5)
    delta_equal = after_equal - before_equal
    delta_shifted = after_shifted - before_shifted
    assert delta_equal > 0 and delta_shifted > 0
    interaction = delta_shifted - delta_equal
    assert abs(interaction) > 1e-3

    # EventG and the single contest response multiply every candidate rate and
    # cancel from the ordinary pair logit. Constant D/R utility contrast across
    # profiles also makes O's softmax denominator cancel in the aggregate.
    for weights in (equal_weights, shifted_participation):
        for o in (-1.0, 1.0):
            base = ordinary_poll_logit(weights, heterogeneous, o, .72, 1.5)
            changed = ordinary_poll_logit(weights, heterogeneous, o, .19, 6.4)
            assert math.isclose(base, changed, rel_tol=0, abs_tol=1e-14)
    constant_gap = ((1.0, 0.0), (2.0, 1.0))
    constant_delta = ordinary_poll_logit(equal_weights, constant_gap, 1.0, .72, 1.5) - ordinary_poll_logit(equal_weights, constant_gap, -1.0, .72, 1.5)
    assert abs(constant_delta) < 1e-14

    print(json.dumps({
        "schema": "A2018PinnedAggregateAndInventedInteractionCheckV1",
        "verified_pins": {str(path): EXPECTED[path] for path in EXPECTED},
        "aggregate_only_a2018": {
            "selected_roles": [role for role, _ in EXPECTED_ROLES],
            "selected_choice": EXPECTED_ROLES[5][1],
            "protected_parent_masses": plan["parent_masses"],
            "az2018_paired_poll_rows": ledger["poll_row_counts_by_event"]["event_az2018_general"],
            "one_poll_covariance_block_all_rows": ledger["counts"]["poll_rows"],
            "official_counts": ledger["counts"]["official_count_observations"],
            "participation_reports": ledger["counts"]["participation_report_observations"],
            "forecast_qualified": plan["forecast_qualified"],
            "source_closure_checks_passed": [
                "open_person_absent", "no_official_counts_or_reports", "no_ranked_or_runoff_observations",
                "survey_loop_eventG_free", "ordinary_poll_formula_verified", "contest_response_is_scalar",
            ],
            "aggregate_eventG_closure_claim_present": ledger["eventG_likelihood_closure"]["claim"].startswith("For the exact pinned A2018 problem/source"),
            "compiled_graph_authenticated": False,
        },
        "invented_equation_sensitivity": {
            "pair_utilities_by_profile": [list(row) for row in heterogeneous],
            "other_utility_change": [-1.0, 1.0],
            "relative_profile_intensity_equal": list(equal_weights),
            "relative_profile_intensity_participation_shift": list(shifted_participation),
            "ordinary_poll_logit_before_equal": before_equal,
            "ordinary_poll_logit_after_equal": after_equal,
            "minor_utility_delta_equal": delta_equal,
            "ordinary_poll_logit_before_shifted": before_shifted,
            "ordinary_poll_logit_after_shifted": after_shifted,
            "minor_utility_delta_shifted": delta_shifted,
            "difference_in_differences": interaction,
            "response_event_scale_cancellation_checked": True,
            "constant_pair_gap_minor_cancellation_checked": True,
            "input_status": "invented utility and relative profile-intensity fixtures; not source rows, posterior states, or a forecast",
        },
        "limits": [
            "source expression presence and prepared aggregate counts do not authenticate original observations or sample independence",
            "synthetic sensitivity demonstrates a possible mechanism, not its magnitude or direction in actual A2018 posterior draws",
            "no compiled graph, prediction, calibration, qualification, or weakest-layer ranking",
        ],
    }, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
