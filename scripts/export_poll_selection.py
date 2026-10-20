#!/usr/bin/env python3
"""Export a sanitized receipt of the saved October 20 polling selection.

Reads frozen evidence only. Does not import a model runner, fit, or recompute a
forecast. Public records identify questions/releases, not shared survey samples.
"""

import argparse
import ast
from collections import Counter
import csv
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse


DEPLOYMENT = Path(__file__).resolve().parents[1]
RESEARCH = Path("research/polling-layer-implementation-20261020")
RUNS = {
    "current-polling": "forecast-v2",
    "current-error-low": "forecast-sensitivity-low-v1",
    "current-error-high": "forecast-sensitivity-high-v1",
    "current-cycle-low": "forecast-sensitivity-cycle-low-v1",
    "current-cycle-high": "forecast-sensitivity-cycle-high-v1",
}
UNIVERSES = {
    "LV": "Likely voters",
    "RV": "Registered voters",
    "voters_unspecified": "Voters (registration and likely-voter screen unspecified by the publisher)",
}
QUESTION_KINDS = {
    "initial_or_unspecified_ballot": "Initial or unspecified ballot question",
    "leaners_included": "Ballot question including voters who lean toward a candidate",
    "unspecified_ballot": "Ballot question (stage unspecified)",
    "head_to_head": "Head-to-head ballot question",
    "multicandidate_initial": "Initial ballot question with multiple candidates",
    "first_preference_two_named_plus_other_not_RCV_transfer": "First preference, two named candidates plus other (not an RCV transfer round)",
    "head_to_head_two_candidates_not_RCV_transfer": "Two-candidate head-to-head (not an RCV transfer round)",
    "ranked_choice_final": "Ranked-choice final preference",
    "first_choice": "First-choice ballot question",
}
# These decisions are documented in the pinned catalog, not a general quality
# model. Unknown alternatives must receive a new reviewed explanation.
ALTERNATIVES = {
    "advanced-targeting-research-id-2026-09-13-2026-09-16:two_candidate": (
        "three_candidate", "choose three-candidate test once",
        "The three-candidate question from this same survey was selected; this dependent two-candidate alternative was not added as another observation.",
    ),
    "targoz-market-research-tn-2026-08-15-2026-08-26:senate_total": (
        "senate_LV", "LV subgroup607 and total Senate base1157 belong to same1200 RV survey",
        "The likely-voter question from this same survey was selected; the overlapping full registered-voter question was not added as another observation.",
    ),
    "co-efficient-oh-2026-10-05-2026-10-06:Q7": (
        "Q6", "Select multi-candidate Q6 once; Q7 dependent alternative",
        "The multicandidate Q6 question from this same survey was selected; dependent Q7 was not added as another observation.",
    ),
    "co-efficient-ia-2026-10-15-2026-10-19:Q9": (
        "Q8", "Ballot variants share one sample",
        "The multicandidate Q8 question from this same survey was selected; the dependent head-to-head alternative was not added as another observation.",
    ),
    "co-efficient-ne-2026-09-28-2026-09-30:Q7": (
        "Q6", "Ballot variants share one sample",
        "The multicandidate Q6 question from this same survey was selected; the dependent head-to-head alternative was not added as another observation.",
    ),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def selection_function(path):
    functions = [node for node in ast.parse(path.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name == "current_observations"]
    require(len(functions) == 1, "Selection function missing")
    return ast.dump(functions[0], include_attributes=False)


def unselected_reason(row, selected_by_sample):
    """Fail closed when an exclusion is not explained by this frozen catalog."""
    qualifiers = " ".join(row["qualifiers"])
    status = row["admission_status"]
    if status == "held_publication_date_not_verified":
        require("Publication date requires fresh direct-release verification" in qualifiers, "Unverified publication hold explanation changed")
        return "Held because the publication date was provisional and had not been verified against a direct release."
    if status == "held_original_source_unavailable":
        require("Secondary-source lead only" in qualifiers and "original release403" in qualifiers, "Original-source hold explanation changed")
        return "Held because the original release was unavailable; only a secondary-source lead and proxy publication date were available."
    if status == "primary_documented_excluded_first_choice_vs_eventual_RCV_target":
        require(row["state"] == "AK" and row["question_kind"] in {"first_choice", "head_to_head"}
                and "not a measured RCV elimination/transfer sequence" in qualifiers, "RCV exclusion explanation changed")
        return "Not selected for the eventual ranked-choice winner target: this first-choice or head-to-head question does not measure the elimination and transfer sequence."
    alternative = ALTERNATIVES.get(row["release_id"])
    require(alternative is not None, f"Unmatched unselected release: {row['release_id']}")
    selected_question, qualifier, explanation = alternative
    sibling = selected_by_sample.get(row["sample_id"])
    require(status == "primary_documented_qualified_aggregate_only" and qualifier in qualifiers
            and sibling is not None and sibling["question_id"] == selected_question,
            f"Dependent-question selection changed: {row['release_id']}")
    return explanation


def export(source_root, registry_path):
    root = source_root.resolve()
    research = root / RESEARCH
    catalog_path = research / "current-intake-sol/current-poll-catalog-v1.json"
    plan_path = research / "standard-plan-root-v1.json"
    catalog, plan = read(catalog_path), read(plan_path)
    cutoff = plan["current_poll_cutoff_utc"]
    datetime.fromisoformat(cutoff.replace("Z", "+00:00"))
    require(datetime.strptime(catalog["acquisition_cutoff_utc"], "%Y-%m-%d %H:%M:%S UTC")
            <= datetime.fromisoformat(cutoff.replace("Z", "+00:00")).replace(tzinfo=None), "Acquisition after cutoff")
    records = catalog["records"]
    require(len(records) == 63 and len({row["release_id"] for row in records}) == 63, "Frozen question catalog changed")
    selected = [row for row in records if row["selected_for_aggregate"]]
    require(len(selected) == 54 and len({row["sample_id"] for row in selected}) == 54, "Selected sample ownership changed")
    selected_by_sample = {row["sample_id"]: row for row in selected}
    selected_by_release = {row["release_id"]: row for row in selected}
    adapter_path = research / "current-intake-sol/aggregate-observations-v1.csv"
    with adapter_path.open(newline="") as handle:
        adapter_rows = list(csv.DictReader(handle))
    require(len(adapter_rows) == len(selected) and {row["release_id"] for row in adapter_rows} == selected_by_release.keys(), "Metadata adapter ownership changed")
    providers = {row["sample_id"]: row["fieldwork_provider_id"] for row in adapter_rows}
    latest = {}
    for row in sorted(selected, key=lambda row: (row["field_end"], row["publication_date"], row["sample_id"])):
        provider = str(providers.get(row["sample_id"]) or row["pollster"]).strip().casefold()
        latest[row["race_id"], provider] = row["release_id"]
    registry = read(registry_path)
    registry_runs = {run["id"]: run for run in registry["runs"]}
    frozen_runner = research / "forecast-v2/source-snapshots/00-standard_runner.py"
    require(selection_function(frozen_runner) == selection_function(research / "standard_runner.py"), "Live and saved selection logic differ")
    source_hashes = {"catalog": sha(catalog_path), "metadataAdapter": sha(adapter_path), "selectionPlan": sha(plan_path), "selectionCode": sha(frozen_runner)}
    applicable = {}
    admitted_ids, excluded_samples, included_counts = None, None, None
    for run_id, folder in RUNS.items():
        run = registry_runs[run_id]
        bundle_path = DEPLOYMENT / "public" / run["href"].removeprefix("/")
        require(sha(bundle_path) == run["sha256"], f"Registry fingerprint mismatch: {run_id}")
        bundle = read(bundle_path)
        require(bundle["run"]["id"] == run_id and run["cutoffAt"] == cutoff
                and bundle["run"]["cutoffAt"] == cutoff, f"Run identity/cutoff mismatch: {run_id}")
        folder_path = research / folder
        receipt_path = folder_path / "input-receipt.json"
        receipt = read(receipt_path)
        for filename in ["standard_runner.py", "standard-plan-root-v1.json", "current-poll-catalog-v1.json", "aggregate-observations-v1.csv"]:
            entries = [entry for entry in receipt["files"] if Path(entry["path"]).name == filename]
            require(len(entries) == 1, f"Missing frozen input: {filename}")
            entry = entries[0]
            require(sha(folder_path / entry["snapshot"]) == entry["sha256"], f"Frozen input snapshot changed: {filename}")
            require((filename == "standard_runner.py" and entry["sha256"] == source_hashes["selectionCode"])
                    or (filename != "standard_runner.py" and sha(root / entry["path"]) == entry["sha256"]),
                    f"Frozen input changed: {filename}")
        admitted_path, excluded_path = folder_path / "admitted-observations.json", folder_path / "excluded-observations.json"
        admitted, excluded = read(admitted_path), read(excluded_path)
        ids = {row["release_id"] for row in admitted}
        older = {row["sample_id"] for row in excluded}
        require(len(ids) == len(admitted) == 49 and ids <= selected_by_release.keys(), f"Admitted release ownership mismatch: {run_id}")
        require(ids == set(latest.values()), "Saved latest-provider sample selection does not reproduce")
        require(len(older) == len(excluded) == 5 and all(row["reason"] == "earlier same-provider race sample" for row in excluded), "Unknown runner exclusion reason")
        require(older == {row["sample_id"] for row in selected if row["release_id"] not in ids}, "Runner exclusions do not partition selected questions")
        # The saved runner adds derived columns; all original catalog fields must
        # remain byte-equivalent as JSON values for every admitted question.
        for row in admitted:
            original = selected_by_release[row["release_id"]]
            require(all(row[key] == value for key, value in original.items()), "Admitted question differs from catalog")
        counts = Counter(row["race_id"] for row in admitted)
        source_forecast = read(folder_path / "forecast.json")
        require(source_forecast["input_cutoff_utc"] == cutoff, "Source forecast cutoff changed")
        require({row["race_id"]: row["admitted_poll_observations"] for row in source_forecast["races"]}
                == {row["id"]: counts[row["id"]] for row in bundle["races"]}, "Source race poll counts differ")
        require({row["id"]: row["pollCount"] for row in bundle["races"]}
                == {row["race_id"]: counts[row["race_id"]] for row in source_forecast["races"]}, f"Public race poll counts differ: {run_id}")
        if admitted_ids is not None:
            require(ids == admitted_ids and older == excluded_samples and counts == included_counts, "Polling variants use different question selections")
        admitted_ids, excluded_samples, included_counts = ids, older, counts
        applicable[run_id] = run["sha256"]
        source_hashes[f"{run_id}:admittedObservations"] = sha(admitted_path)
        source_hashes[f"{run_id}:excludedObservations"] = sha(excluded_path)
        source_hashes[f"{run_id}:inputReceipt"] = sha(receipt_path)
        source_hashes[f"{run_id}:forecast"] = sha(folder_path / "forecast.json")
    evidence_hashes = set()
    for row in records:
        for evidence in row["source_files"]:
            evidence_path = (root / evidence["path"]).resolve()
            require(evidence_path.is_relative_to(root) and sha(evidence_path) == evidence["sha256"], "Acquired evidence fingerprint mismatch")
            evidence_hashes.add(evidence["sha256"])
    source_hashes.update({f"acquiredEvidence{index:02d}": digest for index, digest in enumerate(sorted(evidence_hashes), 1)})
    output = []
    for row in records:
        if row["release_id"] in admitted_ids:
            status, reason = "included", "Selected ballot question from the latest retained sample for this race and fieldwork provider."
        elif row["selected_for_aggregate"]:
            require(row["sample_id"] in excluded_samples, "Unknown selected-question exclusion")
            status, reason = "excluded", "A newer sample from the same fieldwork provider for this race was retained instead."
        else:
            status, reason = "excluded", unselected_reason(row, selected_by_sample)
        require(row["question_universe"] in UNIVERSES and row["question_kind"] in QUESTION_KINDS, "Unknown question universe or kind")
        for key in ["field_start", "field_end", "publication_date"]:
            date.fromisoformat(row[key])
        require(row["field_start"] <= row["field_end"] <= cutoff[:10] and row["publication_date"] <= cutoff[:10], "Question dates exceed cutoff")
        require(urlparse(row["source_url"]).scheme in {"http", "https"} and urlparse(row["source_url"]).netloc, "Invalid publisher URL")
        output.append({
            "id": row["release_id"], "raceId": row["race_id"], "state": row["state"], "pollster": row["pollster"],
            "fieldStart": row["field_start"], "fieldEnd": row["field_end"], "publicationDate": row["publication_date"],
            "sourceUrl": row["source_url"], "status": status, "reason": reason,
            "questionUniverse": UNIVERSES[row["question_universe"]], "questionKind": QUESTION_KINDS[row["question_kind"]],
        })
    output.sort(key=lambda row: (row["raceId"], row["fieldEnd"], row["id"]))
    return {"schemaVersion": 1, "cutoffAt": cutoff, "applicableRuns": applicable,
            "sourceHashes": source_hashes, "records": output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=DEPLOYMENT / "public/data/explorer/registry.json")
    parser.add_argument("--output", type=Path, default=DEPLOYMENT / "data/polling-selection-20261020.json")
    args = parser.parse_args()
    receipt = export(args.source_root, args.registry)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"records": len(receipt["records"]), "included": sum(row["status"] == "included" for row in receipt["records"]), "applicableRuns": list(receipt["applicableRuns"])}))


if __name__ == "__main__":
    main()
