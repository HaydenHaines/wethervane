#!/usr/bin/env python3
"""Stage the ten reviewed October 26 Senate polling editions from frozen native artifacts.

This importer reads pinned forecasts and their original joint rows. It does not fit,
resample, publish, or replace the six fundamentals-only and historical bundles.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from scripts.export_current_comparisons import LABELS as COMPARISON_LABELS
from scripts.export_current_explorer import (
    _race_outcomes, _scenario_payload, _summary_from_counts,
)
from scripts.export_site_snapshot import SnapshotError, _check_races, _distribution, _model, _race
from scripts.export_tipping_points import _chamber_groups, _flatten_draws, _model_snapshot

HANDOFF = Path("research/senate-polling-refresh-20261026/frontend-handoff-v2.json")
HANDOFF_SHA256 = "53ecb8ea7c322932a64784453a00bd4181c7e95cea474c2f843e5b0e0bcd356b"
EXPECTED_ARMS = {
    "standard": ("current-polling", "fundamentals", "standard", "standard", 1.0, 1.0),
    "standard_all_low": ("current-error-low", "fundamentals", "standard", "error-low", 2 / 3, 1.0),
    "standard_all_high": ("current-error-high", "fundamentals", "standard", "error-high", 1.5, 1.0),
    "standard_cycle_low": ("current-cycle-low", "fundamentals", "standard", "cycle-low", 1.0, 0.5),
    "standard_cycle_high": ("current-cycle-high", "fundamentals", "standard", "cycle-high", 1.0, 2.0),
    **{
        arm: (f"current-comparison-{arm}-fp", arm, "standard", "standard", 1.0, 1.0)
        for arm in COMPARISON_LABELS
    },
}
STANDARD_TARGET = "national_senate_aggregate_poll_v1"
COMPARISON_TARGET = "senate2026_principal_pair_fundamentals_comparison_v1"
STANDARD_LAW = "senate_principal_pair_aggregate_v1"  # Existing explicit public projected law.
COMPARISON_LAW = "principal_pair_empirical_logit_shared_normal_student_t5_v1"
POLL_CUTOFF = "2026-10-26T15:03:57.212405+00:00"
STRUCTURAL_CUTOFF = "2026-10-20T15:21:23Z"
SCENARIOS = {
    "campaign_pledge_unaligned",
    "independent_opposition_caucuses_D",
    "independent_opposition_caucuses_R",
}


class RefreshImportError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RefreshImportError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def pinned(root: Path, reference: dict[str, str]) -> Path:
    relative = Path(reference["path"])
    require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe source path")
    path = root / relative
    require(path.resolve().is_relative_to(root.resolve()), "Unsafe source path")
    require(path.is_file() and sha(path) == reference["sha256"], f"Source hash mismatch: {relative}")
    return path


def _unique_notes(*groups: Any) -> list[str]:
    notes: list[str] = []
    for group in groups:
        if isinstance(group, str):
            group = [group]
        require(isinstance(group, list) and all(isinstance(x, str) and x for x in group),
                "Malformed source qualifications")
        notes.extend(x for x in group if x not in notes)
    return notes


def _verify_handoff(root: Path, handoff: dict[str, Any]) -> dict[str, str]:
    require(handoff.get("status") == "numerically_qualified_and_independently_reviewed_not_published",
            "Handoff status changed")
    require(handoff.get("polling_input_cutoff_utc") == POLL_CUTOFF
            and handoff.get("structural_input_cutoff_utc") == STRUCTURAL_CUTOFF,
            "Handoff cutoffs changed")
    arms = handoff.get("arms")
    require(isinstance(arms, list) and len(arms) == 10
            and {a.get("arm") for a in arms} == set(EXPECTED_ARMS), "Handoff arm set changed")
    source = read(pinned(root, handoff["root_source_review"]))
    review = read(pinned(root, handoff["independent_output_review"]))
    require(source.get("status") == "admitted_for_existing_aggregate_target"
            and source.get("candidate_selection_approved") is True, "Source admission not approved")
    checks = {key: read(pinned(root, source[key])) for key in
              ("independent_intake_review", "independent_runner_review", "runner_validation")}
    require(checks["independent_intake_review"].get("status") == "passed_no_blocking_findings"
            and checks["runner_validation"].get("status") == "passed_without_sampling"
            and checks["independent_runner_review"].get("reviewed_runner_unchanged_during_review") is True
            and checks["independent_runner_review"].get("blocking_findings") == [],
            "Independent admission or runner review failed")
    require(review.get("status") == "passed" and set(review.get("arms", {})) == set(EXPECTED_ARMS)
            and all(row.get("status") == "passed" for row in review["arms"].values()),
            "Independent all-arm review not passed")
    for arm in arms:
        reviewed = review["arms"][arm["arm"]]
        require(reviewed.get("forecast") == arm["files"]["forecast.json"]
                and reviewed.get("draws") == arm["retained_joint_draws"]
                and reviewed.get("races") == 35,
                f"{arm['arm']} reviewed artifact/shape differs from handoff")
    for key in ("execution_plan", "intake"):
        pinned(root, handoff[key])
    return {"handoffSha256": sha(root / HANDOFF),
            "sourceAdmissionReviewSha256": handoff["root_source_review"]["sha256"],
            "independentAllArmReviewSha256": handoff["independent_output_review"]["sha256"]}


def _validate_identity(arm: dict[str, Any], forecast: dict[str, Any], spec: dict[str, Any],
                       receipt: dict[str, Any], reference: dict[str, Any]) -> tuple[str, str, str, str]:
    name = arm["arm"]
    run_id, base, polling, uncertainty, all_scale, cycle_scale = EXPECTED_ARMS[name]
    expected_target = STANDARD_TARGET if name.startswith("standard") else COMPARISON_TARGET
    expected_law = STANDARD_LAW if name.startswith("standard") else COMPARISON_LAW
    require(arm.get("target_id") == forecast.get("target_id") == spec.get("target_id")
            == receipt.get("target_id") == expected_target, f"{name} target mismatch")
    require(forecast.get("schema") == "SenatePollingRefreshForecastV1"
            and forecast.get("mode") == "FP" and forecast.get("comparison_arm") == name
            and receipt.get("arm") == name and receipt.get("status") == "passed"
            and forecast.get("numerical_gate_status") == "passed", f"{name} forecast identity/status mismatch")
    if name.startswith("standard"):
        require(forecast.get("law_id") is None, f"{name} unexpected native law")
        require(arm.get("existing_frontend_id") == run_id, f"{name} public ID mismatch")
    else:
        require(forecast.get("law_id") == reference.get("law_id") == expected_law,
                f"{name} law mismatch")
        require(forecast.get("implementation_id") == arm.get("implementation_id"),
                f"{name} implementation mismatch")
        require(arm.get("existing_frontend_id") is None, f"{name} unexpected existing ID")
        # Recent-calendar and refreshed-history have the same implementation but distinct bases.
        require(forecast["comparison_arm"] == base, f"{name} public base mismatch")
    require(forecast.get("input_cutoff_utc") == forecast.get("poll_batch_input_cutoff_utc")
            == forecast.get("latest_input_acquisition_cutoff_utc") == spec.get("polling_input_cutoff_utc")
            == receipt.get("polling_input_cutoff_utc") == POLL_CUTOFF, f"{name} polling cutoff mismatch")
    require(forecast.get("nonpoll_input_cutoff_utc") == spec.get("structural_input_cutoff_utc")
            == receipt.get("structural_input_cutoff_utc") == STRUCTURAL_CUTOFF,
            f"{name} structural cutoff mismatch")
    require(isinstance(forecast.get("executed_utc"), str) and forecast["executed_utc"].startswith("2026-10-26T"),
            f"{name} missing Oct 26 issuance")
    require(forecast.get("unchanged_F_reference") == arm.get("unchanged_fundamentals_only")
            == receipt.get("unchanged_F_reference"), f"{name} F reference mismatch")
    require(math.isclose(float(spec.get("all_scale_multiplier")), all_scale, abs_tol=1e-12, rel_tol=0)
            and math.isclose(float(spec.get("cycle_scale_multiplier")), cycle_scale, abs_tol=1e-12, rel_tol=0),
            f"{name} covariance sensitivity mismatch")
    require(spec.get("admitted_observations") == forecast.get("admitted_poll_observations") == 131
            and spec.get("admitted_races") == forecast.get("admitted_races") == 25,
            f"{name} admission counts mismatch")
    return run_id, base, polling, uncertainty


def _adapt_model(forecast: dict[str, Any], reference: dict[str, Any], previous: dict[str, Any],
                 handoff: dict[str, Any], standard: bool) -> dict[str, Any]:
    # _model is the existing public projection for the standard target. Alternative
    # targets use the same _race/_distribution projections without coercing target_id.
    base_notes = reference.get("limitations", reference.get("qualifications", []))
    previous_notes = previous.get("limitations", previous.get("qualifications", []))
    if isinstance(previous_notes, dict):
        previous_notes = [f"{key.replace('_', ' ')}: {value}" for key, value in previous_notes.items()]
    metadata = forecast.get("posterior_metadata", {})
    require(isinstance(metadata, dict), "Malformed posterior metadata")
    notes = _unique_notes(
        [f"Unchanged fundamentals reference: {x}" for x in base_notes],
        [f"Superseded polling edition: {x}" for x in previous_notes],
        [f"Original posterior qualification: {x}" for x in metadata.get("original_prepared_qualifications", [])],
        [f"Original posterior qualification: {x}" for x in metadata.get("qualifications", [])],
        forecast.get("qualifications", []),
        forecast.get("calibration_reuse", {}).get("reuse_qualification", ""),
        handoff.get("source_limits", []),
    )
    if standard:
        normalized = {**forecast, "execution_completed_utc": forecast["executed_utc"], "limitations": notes}
        try:
            return _model(normalized, "polling")
        except SnapshotError as exc:
            raise RefreshImportError(f"Standard forecast public projection failed: {exc}") from exc
    races = [_race(r, "polling") for r in forecast["races"]]
    _check_races(races, "polling")
    scenario_rows = forecast["caucus_scenarios"]
    require({row.get("id") for row in scenario_rows} == set(forecast["control_scenarios"]) == SCENARIOS,
            "Caucus scenario set changed")
    descriptions = {row["id"]: row["description"] for row in scenario_rows}
    controls = {}
    caucus = {}
    for sid, source in forecast["control_scenarios"].items():
        values = {"democraticMajorityProbability": source["democratic_majority_probability"],
                  "republicanMajorityProbability": source["republican_majority_probability"],
                  "unresolvedOrganizingProbability": source["unresolved_organizing_probability"]}
        require(all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in values.values())
                and abs(sum(values.values()) - 1) <= 1e-8, f"{sid} invalid control law")
        controls[sid] = {"description": descriptions[sid], **values}
        caucus[sid] = {p: _distribution(forecast["chamber"][f"scenario:{sid}:{p}"], sid+p)
                       for p in ("D", "R")}
    return {"targetId": forecast["target_id"], "issuedAt": forecast["executed_utc"],
            "cutoffAt": forecast["input_cutoff_utc"], "races": races, "controls": controls,
            "chamberDistributions": {"caucus": caucus}, "limitations": notes}


def _load_arrays(paths: dict[str, Path], forecast: dict[str, Any], count: int, label: str):
    race_ids = [row["race_id"] for row in forecast["races"]]
    require(len(race_ids) == 35 and len(set(race_ids)) == 35, f"{label} race roster changed")
    with np.load(paths["joint-draws.npz"], allow_pickle=False) as joint:
        if "race_ids" in joint:
            require(joint["race_ids"].tolist() == race_ids, f"{label} joint race order mismatch")
        require(joint["shares"].shape == joint["first_wins"].shape
                and joint["shares"].shape[-1] == 35, f"{label} draw shape mismatch")
        shares, wins = _flatten_draws(joint["shares"], joint["first_wins"], 35, label)
    expected_count = 40000 if label.startswith("standard") else 80000
    require(shares.shape[0] == count == expected_count, f"{label} joint draw count changed")
    with np.load(paths["chamber-draws.npz"], allow_pickle=False) as chamber:
        groups = {name: values.reshape(-1) for name, values in _chamber_groups(chamber, label).items()}
    require(all(len(values) == count for values in groups.values()), f"{label} chamber row count changed")
    require(np.array_equal(groups["party_D"] + groups["party_R"] + groups["party_I"],
                           np.full(count, 100)), f"{label} party seats do not conserve 100")
    for party in ("D", "R", "I"):
        source = _summary_from_counts(groups[f"party_{party}"])
        declared = forecast["chamber"][f"party_{party}"]
        require(np.allclose(source["pmf"], declared["pmf"], atol=1e-12, rtol=0)
                and math.isclose(source["expectedSeats"], declared["expected_seats"], abs_tol=1e-10),
                f"{label} party {party} distribution differs from joint rows")
    quantiles = np.quantile(shares, [0.05, .5, .95], axis=0)
    declared = np.array([[race["first_share_quantiles"][q] for race in forecast["races"]]
                         for q in ("p05", "p50", "p95")])
    require(np.allclose(quantiles, declared, atol=1e-12, rtol=0), f"{label} share quantiles differ")
    return shares, wins, groups


def build_one(root: Path, handoff: dict[str, Any], arm: dict[str, Any], common_pins: dict[str, str]):
    name = arm["arm"]
    paths = {key: pinned(root, ref) for key, ref in arm["files"].items()}
    input_path = pinned(root, arm["frozen_roster_snapshot"])
    reference_path = pinned(root, arm["unchanged_fundamentals_only"])
    previous_path = pinned(root, arm["supersedes_current_polling_reference"])
    forecast, inputs, reference = read(paths["forecast.json"]), read(input_path), read(reference_path)
    previous = read(previous_path)
    spec, receipt, gates = (read(paths[key]) for key in ("run-specification.json", "receipt.json", "numerical-gates.json"))
    require(gates.get("status") == "passed" and gates.get("checks")
            and all(value is True for value in gates["checks"].values()),
            f"{name} numerical gates not passed")
    require(all(receipt.get("artifacts", {}).get(key) == arm["files"][key]["sha256"]
                for key in arm["files"] if key != "receipt.json"), f"{name} receipt artifact pins differ")
    require(receipt.get("plan") == handoff["execution_plan"] and forecast.get("plan") == handoff["execution_plan"],
            f"{name} execution plan mismatch")
    run_id, base, polling, uncertainty = _validate_identity(arm, forecast, spec, receipt, reference)
    model = _adapt_model(forecast, reference, previous, handoff, name.startswith("standard"))
    shares, wins, groups = _load_arrays(paths, forecast, arm["retained_joint_draws"], name)
    tipping = _model_snapshot(label=name, forecast=forecast, input_data=inputs, shares=shares,
                              first_wins=wins, chamber_groups=groups)
    scenarios = [_scenario_payload(model, tipping, row, groups) for row in forecast["caucus_scenarios"]]
    source_by_id = {r["race_id"]: r for r in forecast["races"]}
    races = []
    share_notes = {
        "democratic_two_party": "Democratic candidate's share among the two named contenders (democratic_two_party).",
        "opposition_proxy": "Named opposition contender's share against the Republican among the two named contenders; this contender is independent, not a Democrat (opposition_proxy).",
    }
    for row in model["races"]:
        source = source_by_id[row["id"]]
        public = _race_outcomes(row, source)
        share_note = source.get("share_interpretation")
        require(share_note in share_notes, f"{name}/{row['id']} unknown share interpretation")
        require((share_note == "opposition_proxy") == (row["candidates"][0]["party"] == "I"),
                f"{name}/{row['id']} share interpretation/party mismatch")
        public["notes"] = _unique_notes(public["notes"], share_notes[share_note])
        races.append(public)
    labels = {"standard": "Standard polling", "standard_all_low": "Polling error: all scales × 2/3",
              "standard_all_high": "Polling error: all scales × 1.5",
              "standard_cycle_low": "Polling error: common-cycle SD × 0.5",
              "standard_cycle_high": "Polling error: common-cycle SD × 2"}
    label = labels.get(name, f"{COMPARISON_LABELS.get(name, name)} · with polls")
    law = STANDARD_LAW if name.startswith("standard") else COMPARISON_LAW
    run = {"id": run_id, "year": 2026, "label": label, "targetId": model["targetId"], "lawId": law,
           "dateId": "2026-10-26", "dateLabel": "October 26, 2026", "forecastAt": POLL_CUTOFF,
           "issuedAt": model["issuedAt"], "cutoffAt": POLL_CUTOFF, "horizonDays": None,
           "components": {"base": base, "polling": polling, "uncertainty": uncertainty},
           "uncertaintyLabel": "Saved fitted joint uncertainty" if not name.startswith("standard")
           else "Saved polling-error scales",
           "uncertaintyDescription": "Shares, winners, chamber seats and tipping derive from the same saved joint rows; polling covariance reuses historical calibration without retraining.",
           "historical": False, "sensitivity": name != "standard", "simulationCount": len(shares),
           "strictAsOf": False,
           "inputCutoffs": [{"label": "Structural inputs", "at": STRUCTURAL_CUTOFF},
                            {"label": "Polling inputs", "at": POLL_CUTOFF}]}
    if not name.startswith("standard"):
        run["researchEvidence"] = {
            "status": "experimental", "numericalStatus": "passed",
            "forecastSkill": "unobserved_2026",
            "qualification": "Saved current comparison. Conservative availability bounds include 22 selected observations without exact posting dates; strict as-of is not established.",
        }
    sources = {**common_pins, "forecastSha256": arm["files"]["forecast.json"]["sha256"],
               "jointDrawsSha256": arm["files"]["joint-draws.npz"]["sha256"],
               "chamberDrawsSha256": arm["files"]["chamber-draws.npz"]["sha256"],
               "runSpecificationSha256": arm["files"]["run-specification.json"]["sha256"],
               "numericalGatesSha256": arm["files"]["numerical-gates.json"]["sha256"],
               "receiptSha256": arm["files"]["receipt.json"]["sha256"],
               "frozenInputsSha256": arm["frozen_roster_snapshot"]["sha256"],
               "unchangedFundamentalsSha256": arm["unchanged_fundamentals_only"]["sha256"],
               "supersededForecastSha256": arm["supersedes_current_polling_reference"]["sha256"]}
    return {"schemaVersion": 1, "run": run, "races": races, "scenarios": scenarios,
            "defaultScenarioId": "campaign_pledge_unaligned", "tipping": tipping,
            "limitations": model["limitations"], "historicalNote": None,
            "provenance": {"sources": sources}}


def build(root: Path):
    require(sha(root / HANDOFF) == HANDOFF_SHA256, "Handoff hash changed")
    handoff = read(root / HANDOFF)
    common_pins = _verify_handoff(root, handoff)
    bundles = {bundle["run"]["id"]: bundle for arm in handoff["arms"]
               for bundle in [build_one(root, handoff, arm, common_pins)]}
    require(set(bundles) == {row[0] for row in EXPECTED_ARMS.values()}, "Public refresh IDs changed")
    first = {r["id"]: r for r in bundles["current-polling"]["races"]}
    for bundle in bundles.values():
        require({r["id"] for r in bundle["races"]} == set(first), "Refresh race roster differs")
        for row in bundle["races"]:
            target = first[row["id"]]
            require([x["id"] for x in row["outcomes"]] == [x["id"] for x in target["outcomes"]],
                    f"Refresh candidate identities differ in {row['id']}")
    return bundles


def export(root: Path, output: Path) -> list[Path]:
    public_data = (Path(__file__).resolve().parents[1] / "public/data").resolve()
    require(not output.resolve().is_relative_to(public_data),
            "Stage to an isolated output directory, not public/data")
    bundles = build(root)
    output.mkdir(parents=True, exist_ok=True)
    written = []
    entries = []
    for run_id, bundle in bundles.items():
        path = output / f"{run_id}.json"
        data = (json.dumps(bundle, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
        path.write_bytes(data)
        written.append(path)
        entries.append({**bundle["run"], "href": f"/data/explorer/{run_id}.json",
                        "sha256": hashlib.sha256(data).hexdigest()})
    index_path = output / "senate-refresh-index.json"
    index_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return [*written, index_path]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in export(args.source_root, args.output):
        print(f"Staged {path}")


if __name__ == "__main__":
    main()
