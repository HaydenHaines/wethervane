#!/usr/bin/env python3
"""Export the two reviewed, race-only 2018 component integration results.

Presentation only: no model imports, refits, resampling, or chamber inference.
Observed outcomes are joined after loading the immutable forecast summaries.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path

from scripts.export_component_catalog import (
    PACKET,
    INTERFACE_SHA256,
    READY_SHA256,
    CatalogError,
    export_catalog,
    pinned,
    read,
    require,
    sha,
    validate_ready,
)

DENOMINATOR = "Democratic and Republican candidate votes only"


def probability(value, label):
    require(
        not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1,
        f"Invalid probability: {label}",
    )
    return value


def integer(value, label):
    try:
        number = float(value)
    except (ValueError, TypeError):
        raise CatalogError(f"Invalid count: {label}") from None
    require(math.isfinite(number) and number >= 0 and number.is_integer(), f"Invalid count: {label}")
    return int(number)


def unique_rows(rows, race_ids, label):
    by_id = {}
    for row in rows:
        race_id = row.get("race_id")
        require(race_id in race_ids and race_id not in by_id, f"Duplicate or unexpected {label} race")
        by_id[race_id] = row
    require(set(by_id) == set(race_ids), f"Incomplete {label} join")
    return by_id


def project_races(summary, features, outcomes):
    """Exact race-ID joins; D/R IDs identify scoped outcomes, not people."""
    ids = summary["race_ids"]
    require(len(ids) == len(set(ids)) == 30, "Expected 30 unique scoped races")
    feature_by_id = unique_rows(features, ids, "feature")
    outcome_by_id = unique_rows(outcomes, ids, "observed outcome")
    levels = summary["quantile_levels"]
    require(levels == [0.025, 0.1, 0.5, 0.9, 0.975], "Unexpected quantile levels")
    require(
        len(summary["dem_win_probability"]) == len(summary["mean_dem_share"]) == 30
        and len(summary["share_quantiles"]) == len(levels)
        and all(len(row) == 30 for row in summary["share_quantiles"]),
        "Invalid summary array shape",
    )
    races = []
    for index, race_id in enumerate(ids):
        feature, observed = feature_by_id[race_id], outcome_by_id[race_id]
        require(
            feature["cycle"] == feature["year"] == observed["cycle"] == "2018"
            and feature["eligible"] == "True"
            and feature["special"] == "False",
            "Unsupported feature/outcome scope",
        )
        state = feature["state_abbrev"]
        require(
            race_id == f"2018:{state}:senate:regular" and feature["state"] == state,
            "Feature state/race identity mismatch",
        )
        p = probability(summary["dem_win_probability"][index], race_id)
        mean = probability(summary["mean_dem_share"][index], race_id)
        quantiles = [probability(values[index], race_id) for values in summary["share_quantiles"]]
        require(quantiles == sorted(quantiles), "Unordered share quantiles")
        candidates = []
        for party, prefix, win, average, values in (
            ("D", "dem", p, mean, quantiles),
            ("R", "rep", 1 - p, 1 - mean, [1 - value for value in reversed(quantiles)]),
        ):
            name, incumbent = feature[f"{prefix}_candidate_name"], feature[f"{prefix}_incumbent"]
            require(
                isinstance(name, str) and bool(name.strip()) and incumbent in {"True", "False"},
                "Candidate name/incumbency missing",
            )
            candidates.append(
                dict(
                    id=party,
                    name=name,
                    party=party,
                    incumbent=incumbent == "True",
                    winProbability=win,
                    share={
                        "mean": average,
                        "denominator": DENOMINATOR,
                        "quantiles": [{"level": level, "value": value} for level, value in zip(levels, values)],
                    },
                )
            )
        d = integer(observed["dem_votes"], race_id + " Democratic votes")
        r = integer(observed["rep_votes"], race_id + " Republican votes")
        total = integer(observed["two_party_votes"], race_id + " two-party votes")
        require(total == d + r and total > 0, "Observed two-party counts do not reconcile")
        actual_share = probability(float(observed["actual_dem_share"]), race_id + " observed share")
        require(abs(actual_share - d / total) < 1e-12, "Observed share differs from vote counts")
        require(
            feature["roster_vintage"] == "final_result_roster_proxy;not_dated_ballot_snapshot",
            "Roster qualification changed",
        )
        races.append(
            dict(
                id=race_id,
                state=state,
                seatId=feature["seat_id"],
                candidates=candidates,
                observed={
                    "democraticVotes": d,
                    "republicanVotes": r,
                    "twoPartyVotes": total,
                    "democraticTwoPartyShare": actual_share,
                },
                notes=[
                    "Candidate names are final-result roster proxies, not a verified ballot snapshot available on the forecast date.",
                    "Candidate IDs D and R identify the two scoped model outcomes; they are not durable person identifiers.",
                ],
            )
        )
    return races


def build_results(root):
    # Reuse complete receipt, provider, dependency and immutable artifact checks.
    metadata = export_catalog(root)
    ready_receipt = read(root / PACKET / "ready-v1/receipt.json")
    interface = read(root / PACKET / "interface-v1/receipt.json")
    interface_outputs = {Path(ref["path"]).name: pinned(root, ref) for ref in interface["outputs"]}
    ready_outputs = {Path(ref["path"]).name: pinned(root, ref) for ref in ready_receipt["outputs"]}
    index_path = pinned(root, ready_receipt["result_index"])
    index = read(index_path)
    results = []
    for arm in index["arms"]:
        mode = arm["arm"]
        request = read(interface_outputs[f"{mode}-request.json"])
        ready_path = ready_outputs[f"{mode}-ready.json"]
        ready = read(ready_path)
        validate_ready(ready, request, arm)
        bundle_path = pinned(root, request["data_bundle"])
        bundle = read(bundle_path)
        feature_path = pinned(root, bundle["target_features"])
        outcome_path = pinned(root, bundle["training_outcomes"])
        panel_path = pinned(root, bundle["target_receipt"])
        with feature_path.open(newline="") as handle:
            features = list(csv.DictReader(handle))
        with outcome_path.open(newline="") as handle:
            # Outcomes are evaluation data only, never forecast input here.
            outcomes = [row for row in csv.DictReader(handle) if row["cycle"] == "2018"]
        races = project_races(ready["summary"], features, outcomes)
        require(
            bundle["target_admission"]["support_id"] == ready["configuration"]["support_id"]
            and bundle["target_admission"]["roster_qualification"] == "qualified_retrospective_proxy",
            "Target admission mismatch",
        )
        context = next(
            context
            for context in metadata["supportedContexts"]
            if context["id"] == f"ordinary-senate-2018-{mode.lower()}"
        )
        results.append(
            dict(
                schemaVersion=1,
                id=f"ordinary-2018-{mode.lower()}",
                contextId=context["id"],
                label=f"2018 ordinary Senate races · {'with polls' if mode == 'FP' else 'without polls'}",
                year=2018,
                scope="ordinary_dr_race_subset",
                informationCutoff=request["information_cutoff"],
                targetId=request["target_id"],
                outputContractId=request["output_contract_id"],
                requestSha256=ready["request_sha256"],
                configurationSha256=ready["record"]["configuration_sha256"],
                artifactId=ready["artifact"]["artifact_id"],
                selections=copy.deepcopy(request["selections"]),
                status={"lifecycle": "ready", "numerical": "passed", "evidence": "unassessed", "experimental": True},
                qualification=ready["summary"]["qualification"],
                strictAsOf=False,
                capabilities={
                    "raceProbabilities": True,
                    "conditionalTwoPartyShares": True,
                    "chamberControl": False,
                    "tipping": False,
                    "counts": False,
                },
                races=races,
                quantileLevels=ready["summary"]["quantile_levels"],
                sourceAttribution=[
                    {
                        "label": "Source-bound research panel",
                        "description": "Names and race membership come from the pinned 2018 ordinary D/R feature panel. Final-result names are retrospective proxies, not proof of historical ballot availability.",
                        "urls": [],
                    },
                    {
                        "label": "Observed two-party outcomes",
                        "description": "D/R vote counts come from the separately pinned research outcome labels, including documented official/MEDSL revisions. These are two-party counts, not full-ballot totals.",
                        "urls": [],
                    },
                    {
                        "label": "MEDSL historical source documentation",
                        "description": "Background documentation for the historical returns collection. These public references do not replace the exact source fingerprints for the derived research panel.",
                        "urls": [
                            "https://github.com/MEDSL/constituency-returns",
                            "https://github.com/MEDSL/constituency-returns/blob/master/codebook-us-senate-1976%E2%80%932018.md",
                        ],
                    },
                ],
                sourceFingerprints={
                    "interfaceReceipt": INTERFACE_SHA256,
                    "readyReceipt": READY_SHA256,
                    "readyResult": sha(ready_path),
                    "resultIndex": sha(index_path),
                    "summary": arm["summary"]["sha256"],
                    "dataBundle": sha(bundle_path),
                    "targetFeatures": sha(feature_path),
                    "observedOutcomeLabels": sha(outcome_path),
                    "panelReceipt": sha(panel_path),
                    "scientificPlan": request["context"]["scientific_plan"]["sha256"],
                    "trainingLaw": ready["summary"]["training_law_sha256"],
                    "recipe": ready["summary"]["recipe_sha256"],
                },
                limitations=[
                    *ready["record"]["limitations"],
                    "Observed outcomes are joined only to evaluate this saved prediction; they were not added to model fitting by this exporter.",
                    "This result covers 30 ordinary D/R races. It contains no chamber-control, count-model or tipping-point forecast.",
                    "Reported quantiles are the 2.5th, 10th, 50th, 90th and 97.5th percentiles. The available central ranges are 80% and 95%; no 90% interval is supplied.",
                    "Historical poll availability and roster inputs use qualified retrospective proxies; this is not strict-as-of validation.",
                    "Numerical checks passed, but predictive evidence for these integration examples remains unassessed.",
                ],
            )
        )
    require(len(results) == 2, "Expected exactly two scoped results")
    return results


def export(root, output):
    results = build_results(root)
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    for result in results:
        path = output / ("fp.json" if result["id"].endswith("-fp") else "f.json")
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        entries.append(
            dict(
                id=result["id"],
                file=path.name,
                sha256=sha(path),
                contextId=result["contextId"],
                configurationSha256=result["configurationSha256"],
            )
        )
    (output / "index.json").write_text(json.dumps({"schemaVersion": 1, "results": entries}, indent=2) + "\n")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/component-results"))
    args = parser.parse_args()
    print(f"Exported {len(export(args.source_root, args.output))} immutable scoped results; no models run")


if __name__ == "__main__":
    main()
