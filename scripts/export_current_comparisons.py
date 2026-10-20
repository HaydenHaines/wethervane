#!/usr/bin/env python3
"""Project ten reviewed saved Senate comparisons; never fit or resample a model."""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.export_site_snapshot import _race, _check_races, _distribution
from scripts.export_current_explorer import _race_outcomes, _scenario_payload
from scripts.export_tipping_points import _flatten_draws, _chamber_groups, _model_snapshot

PACKET = Path("research/fundamentals-iteration-20261020/frontend-contract-root-v2/interface-v1")
INTERFACE_RECEIPT_SHA256 = "85f38310f650fc7eb0b18739f36e36a3a7457602cb5e42f328a89f1b58bb1ece"
TARGET = "senate2026_principal_pair_fundamentals_comparison_v1"
LAW = "principal_pair_empirical_logit_shared_normal_student_t5_v1"
LABELS = {
    "recent_calendar_history": "Recent calendar history",
    "senate_history_refreshed_v1": "Refreshed history",
    "senate_presidential_slope_v1": "Learned presidential slope",
    "senate_relative_context_v1": "Relative lean with presidential context",
    "senate_relative_lean_v1": "Relative state and national lean",
}


class ComparisonError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ComparisonError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def pinned(root, reference):
    relative = Path(reference["path"])
    require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe source path")
    path = root / relative
    require(path.is_file() and sha(path) == reference["sha256"], f"Source hash mismatch: {relative}")
    return path


def build_one(root, entry, packet_hashes):
    paths = {name: pinned(root, ref) for name, ref in entry["files"].items()}
    receipt_path = pinned(root, entry["receipt"])
    receipt = read(receipt_path)
    plan_path = pinned(root, receipt["plan"])
    plan = read(plan_path)
    # Scientific inputs are independently checked, not merely referenced by name.
    for ref in plan["inputs"].values():
        pinned(root, ref)
    inputs_path = pinned(root, plan["inputs"]["frozen_inputs"])
    inputs = read(inputs_path)
    forecast = read(paths["forecast.json"])
    gates = read(paths["numerical-gates.json"])
    require(entry["configuration"] in LABELS and entry["mode"] in {"F", "FP"}, "Unreviewed comparison")
    require(entry["custom_composition_driver_available"] is False, "Unexpected custom chamber execution claim")
    require(
        entry["numerical_status"] == gates["status"] == forecast["numerical_gate_status"] == "passed",
        "Numerical gates did not pass",
    )
    require(all(gates["checks"].values()), "Failed numerical check")
    require(forecast["target_id"] == entry["target_id"] == TARGET and forecast["law_id"] == LAW, "Target/law mismatch")
    require(
        forecast["comparison_arm"] == entry["configuration"] and forecast["mode"] == entry["mode"],
        "Comparison identity mismatch",
    )
    for key in ("input_cutoff_utc", "nonpoll_input_cutoff_utc", "poll_batch_input_cutoff_utc"):
        require(forecast[key] == entry[key], f"Cutoff mismatch: {key}")
    key = "polling" if entry["mode"] == "FP" else "fundamentals"
    races = [_race(race, key) for race in forecast["races"]]
    _check_races(races, key)
    with np.load(paths["joint-draws.npz"], allow_pickle=False) as joint:
        shares, wins = _flatten_draws(joint["shares"], joint["first_wins"], len(races), "comparison")
    source_quantiles = np.array(
        [[race["first_share_quantiles"][q] for race in forecast["races"]] for q in ("p05", "p50", "p95")]
    )
    require(
        np.allclose(np.quantile(shares, [0.05, 0.5, 0.95], axis=0), source_quantiles, rtol=0, atol=1e-12),
        "Share quantiles do not match joint draws",
    )
    with np.load(paths["chamber-draws.npz"], allow_pickle=False) as chamber:
        groups = {name: values.reshape(-1) for name, values in _chamber_groups(chamber, "comparison").items()}
    require(
        np.array_equal(groups["party_D"] + groups["party_R"] + groups["party_I"], np.full(shares.shape[0], 100)),
        "Party draws do not conserve 100 seats",
    )
    for party in ("D", "R", "I"):
        counts = groups[f"party_{party}"]
        expected = np.bincount(counts, minlength=101) / len(counts)
        require(
            np.allclose(expected, forecast["chamber"][f"party_{party}"]["pmf"], rtol=0, atol=1e-12),
            "Party PMF does not match joint draws",
        )
    tipping = _model_snapshot(
        label="comparison", forecast=forecast, input_data=inputs, shares=shares, first_wins=wins, chamber_groups=groups
    )
    # The shared scenario adapter validates PMFs, quantiles and seat conservation against joint draws.
    controls = {}
    caucus = {}
    for scenario in forecast["caucus_scenarios"]:
        sid = scenario["id"]
        source = forecast["control_scenarios"][sid]
        controls[sid] = dict(
            description=scenario["description"],
            democraticMajorityProbability=source["democratic_majority_probability"],
            republicanMajorityProbability=source["republican_majority_probability"],
            unresolvedOrganizingProbability=source["unresolved_organizing_probability"],
        )
        require(abs(sum(source.values()) - 1) < 1e-8, "Control probabilities do not sum to one")
        caucus[sid] = {p: _distribution(forecast["chamber"][f"scenario:{sid}:{p}"], sid + p) for p in ("D", "R")}
    model = {"controls": controls, "chamberDistributions": {"caucus": caucus}}
    scenarios = [_scenario_payload(model, tipping, scenario, groups) for scenario in forecast["caucus_scenarios"]]
    run_id = f"current-comparison-{entry['configuration']}-{entry['mode'].lower()}"
    cutoffs = [{"label": "Nonpoll inputs", "at": forecast["nonpoll_input_cutoff_utc"]}]
    if forecast["poll_batch_input_cutoff_utc"]:
        cutoffs.append({"label": "Polling data", "at": forecast["poll_batch_input_cutoff_utc"]})
    run = dict(
        id=run_id,
        year=2026,
        label=f"{LABELS[entry['configuration']]} · {'with polls' if key == 'polling' else 'without polls'}",
        targetId=TARGET,
        lawId=LAW,
        dateId="2026-10-20",
        dateLabel="October 20, 2026",
        forecastAt=forecast["input_cutoff_utc"],
        issuedAt=forecast["executed_utc"],
        cutoffAt=forecast["input_cutoff_utc"],
        horizonDays=None,
        components={
            "base": entry["configuration"],
            "polling": "standard" if key == "polling" else "none",
            "uncertainty": "standard",
        },
        uncertaintyLabel="Saved fitted joint uncertainty",
        uncertaintyDescription="Shared national shifts and race-specific variation from this exact saved comparison.",
        historical=False,
        sensitivity=True,
        simulationCount=tipping["simulationCount"],
        strictAsOf=False,
        inputCutoffs=cutoffs,
        researchEvidence={
            "status": "experimental",
            "numericalStatus": "passed",
            "forecastSkill": entry["forecast_skill"],
            "qualification": "Saved current comparison, not a replacement for the preserved default and not arbitrary custom chamber execution.",
        },
    )
    public_races = [_race_outcomes(race, source) for race, source in zip(races, forecast["races"])]
    sources = {
        **packet_hashes,
        "receiptSha256": sha(receipt_path),
        "planSha256": sha(plan_path),
        "frozenInputsSha256": sha(inputs_path),
        **{name.replace(".", "_"): sha(path) for name, path in paths.items()},
        "forecastSha256": sha(paths["forecast.json"]),
        "jointDrawsSha256": sha(paths["joint-draws.npz"]),
        "chamberDrawsSha256": sha(paths["chamber-draws.npz"]),
    }
    return dict(
        schemaVersion=1,
        run=run,
        races=public_races,
        scenarios=scenarios,
        defaultScenarioId="campaign_pledge_unaligned",
        tipping=tipping,
        limitations=forecast["qualifications"],
        historicalNote=None,
        provenance={"sources": sources},
    )


def export(root, output):
    receipt_path = root / PACKET / "receipt.json"
    require(sha(receipt_path) == INTERFACE_RECEIPT_SHA256, "Interface receipt differs from reviewed handoff")
    receipt = read(receipt_path)
    index_ref = next(ref for ref in receipt["outputs"] if ref["path"].endswith("/current-comparisons.json"))
    index_path = pinned(root, index_ref)
    seal_path = pinned(root, receipt["current_final_seal"])
    require(read(seal_path)["status"] == "passed", "Final execution seal did not pass")
    entries = read(index_path)
    require(
        len(entries) == 10
        and {(e["configuration"], e["mode"]) for e in entries} == {(c, m) for c in LABELS for m in ("F", "FP")},
        "Expected five reviewed F/FP pairs",
    )
    packet_hashes = {
        "interfaceReceiptSha256": sha(receipt_path),
        "comparisonIndexSha256": sha(index_path),
        "executionSealSha256": sha(seal_path),
    }
    # Build all results before writing any output.
    bundles = [build_one(root, entry, packet_hashes) for entry in entries]
    output.mkdir(parents=True, exist_ok=True)
    index = []
    for bundle in bundles:
        path = output / (bundle["run"]["id"] + ".json")
        path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        index.append({**bundle["run"], "href": "/data/explorer/" + path.name, "sha256": sha(path)})
    (output / "current-comparisons-index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n")
    return bundles


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("public/data/explorer"))
    a = p.parse_args()
    print(f"Exported {len(export(a.source_root, a.output))} saved comparisons")
