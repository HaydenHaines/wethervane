"""Publish the reviewed Oct 26 poll receipt without changing selection or forecasts.

Run after the refreshed Explorer bundles exist. Native admissions and the source
review, rather than reconstructed polling heuristics, determine inclusion.
"""
import argparse
from collections import Counter
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

from scripts.export_poll_selection import UNIVERSES, QUESTION_KINDS

HANDOFF = "research/senate-polling-refresh-20261026/frontend-handoff-v2.json"
HANDOFF_SHA = "53ecb8ea7c322932a64784453a00bd4181c7e95cea474c2f843e5b0e0bcd356b"
KINDS = {**QUESTION_KINDS, **dict.fromkeys([
    "allocated_leaners", "ballot_with_leaners", "ballot_including_leaners",
    "full_ballot_with_leaners", "modeled_LV_ballot_with_leaners",
], "Ballot question including voters who lean toward a candidate"),
    **dict.fromkeys(["initial_ballot", "initial_ballot_without_leaners", "named_multicandidate_ballot"],
                   "Initial ballot question"),
    **dict.fromkeys(["two_candidate_app_preference", "two_candidate_app_poll"], "Two-candidate question in a polling app"),
    **dict.fromkeys(["reported_vote_and_current_ballot_with_leaners", "ballot_with_leaners_and_reported_early_vote", "ballot_with_leaners_and_early_vote"],
                   "Ballot question combining reported early votes and current preferences, including leaners"),
    **dict.fromkeys(["informed_post_biography"], "Ballot question asked after candidate biographies"),
    "ranked_choice_first_preference": "Ranked-choice first preference",
    "ranked_choice_first_preference_only": "Ranked-choice first preference only",
    "ranked_choice_after_excluding_not_sure": "Ranked-choice question excluding unsure respondents",
    "hypothetical_RCV_exhausted_ballots_reassigned_to_incumbent": "Hypothetical ranked-choice result that reassigns exhausted ballots",
    "definitely_and_probably_combined": "Ballot question combining definite and probable support",
    "full_ballot_definitely_probably_combined": "Full ballot combining definite and probable support",
    "head_to_head_definitely_probably_combined": "Two-candidate question combining definite and probable support",
    **dict.fromkeys(["head_to_head_with_leaners", "head_to_head_including_leaners"], "Two-candidate question including leaners"),
    **dict.fromkeys(["ballot_with_forced_followup_for_unsure", "forced_choice_undecided_followup", "ballot_with_forced_followup", "forced_push_ballot"],
                   "Ballot question with a follow-up asking unsure respondents to choose"),
    "probabilistic_likely_voter_model_ballot": "Ballot estimate weighted by estimated likelihood of voting",
}
HOLDS = {
    "held_publication_date_not_verified": "The recorded publication date has not been verified and a qualifying availability record was not established.",
    "held_original_source_unavailable": "The original release was unavailable; the retained lead alone did not satisfy the source checks.",
    "primary_documented_held_state_metadata_conflict": "The source contains conflicting state information that remains unresolved.",
    "primary_documented_held_conflicting_LV_sample_sizes": "The source gives conflicting likely-voter sample sizes that remain unresolved.",
    "primary_ballot_verified_missing_primary_sample_metadata": "The ballot figures were verified, but required fieldwork or sample details were not verified in the primary source.",
    "primary_documented_excluded_first_choice_vs_eventual_RCV_target": "This question measures first choice or a head-to-head matchup, not the final ranked-choice preference used by this model.",
    "primary_documented_held_estimand_incompatible_with_current_AK_final_RCV_proxy": "This question measures first choice, not the final ranked-choice preference used by this model.",
    "excluded_informed_ballot_same_sample": "This question follows candidate biographies and measures prompted responses; it is not added as another independent ballot poll.",
}
ALTERNATIVES = {"primary_documented_qualified_aggregate_only", "primary_documented_alternative_not_independent", "same_sample_alternate_not_selected"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def pinned(root, ref):
    relative = Path(ref["path"])
    path = (root / relative).resolve()
    require(not relative.is_absolute() and ".." not in relative.parts and path.is_relative_to(root), "Unsafe source reference")
    require(path.is_file() and sha(path) == ref["sha256"], f"Source hash mismatch: {relative}")
    return path


def explain(row, selected, variants, earlier, records):
    rid = row["release_id"]
    if rid in selected:
        require(row["selected_for_aggregate"] and variants[rid]["disposition"] == "candidate_latest_provider_race_sample", "Admission/selection mismatch")
        return "included", "Selected question from the latest retained sample for this race and polling-provider grouping."
    require(not row["selected_for_aggregate"], "Selected row absent from admitted observations")
    if rid in earlier:
        successor = records[earlier[rid]["successor_release_id"]]
        require(successor["race_id"] == row["race_id"] and successor["field_end"] >= row["field_end"], "Invalid later-sample exclusion")
        return "excluded", "A newer sample in the same polling-provider grouping for this race was retained instead."
    status = row["admission_status"]
    if status in HOLDS:
        return "excluded", HOLDS[status]
    require(status in ALTERNATIVES and any(v["release_id"] != rid and v["source_selected"]
            for v in variants.values() if records[v["release_id"]]["sample_id"] == row["sample_id"]),
            f"No reviewed explanation for excluded question: {rid}")
    if row["state"] == "AK":
        return "excluded", "This survey also reports final ranked-choice preferences, which were selected for this model. Its first-choice, earlier-round and hypothetical alternatives are not added as separate polls."
    return "excluded", "Another question or voter group from the same survey was selected. These overlapping responses are not counted as another poll."


def build(source_root, bundles):
    root = source_root.resolve()
    handoff = read(pinned(root, {"path": HANDOFF, "sha256": HANDOFF_SHA}))
    intake = read(pinned(root, handoff["intake"]))
    review = read(pinned(root, intake["source_review"]))
    require(intake["status"] == review["status"] == "admitted_for_existing_aggregate_target"
            and review["candidate_selection_approved"] is True, "Source collection has not been admitted")
    catalog = read(pinned(root, intake["catalog"]))
    report = read(pinned(root, catalog["reconciliation_report"]))
    pinned(root, intake["policy"])
    rows = catalog["records"]
    records = {r["release_id"]: r for r in rows}
    require(len(rows) == len(records) == review["preserved_variants"] == 187, "Question collection changed")
    variants = {v["release_id"]: v for group in report["same_sample_variants"].values() for v in group}
    require(variants.keys() == records.keys(), "Reconciliation does not cover every question")
    earlier = {v["release_id"]: v for v in report["latest_provider_exclusions"]}
    standard = next(arm for arm in handoff["arms"] if arm["arm"] == "standard")
    admitted = read(pinned(root, standard["files"]["admitted-observations.json"]))
    selected = {r["release_id"] for r in admitted}
    require(len(selected) == len(admitted) == handoff["observations"] == 131, "Admitted question count changed")
    for row in admitted:
        require(all(row.get(k) == v for k, v in records[row["release_id"]].items()), "Admitted record differs from its reviewed source")
    counts = Counter(r["race_id"] for r in admitted)
    require(len(counts) == handoff["polled_races"] == 25, "Polled race count changed")
    applicable = {}
    sources = {"handoff": HANDOFF_SHA, "intake": handoff["intake"]["sha256"], "catalog": intake["catalog"]["sha256"],
               "sourceReview": intake["source_review"]["sha256"], "policy": intake["policy"]["sha256"],
               "reconciliation": catalog["reconciliation_report"]["sha256"]}
    for arm in handoff["arms"]:
        run_id = arm["existing_frontend_id"] or f"current-comparison-{arm['arm']}-fp"
        path = bundles / f"{run_id}.json"
        bundle = read(path)
        observation_path = pinned(root, arm["files"]["admitted-observations.json"])
        require(sha(observation_path) == standard["files"]["admitted-observations.json"]["sha256"], "Arms use different admitted observations")
        require(bundle["run"]["id"] == run_id and bundle["run"]["cutoffAt"] == handoff["polling_input_cutoff_utc"]
                and bundle["provenance"]["sources"]["forecastSha256"] == arm["files"]["forecast.json"]["sha256"], "Public forecast binding mismatch")
        require(len(bundle["races"]) == 35 and all(r["pollCount"] == counts[r["id"]] for r in bundle["races"]), "Public race polling counts differ")
        applicable[run_id] = sha(path)
        sources[f"{run_id}:admittedObservations"] = sha(observation_path)
        sources[f"{run_id}:forecast"] = arm["files"]["forecast.json"]["sha256"]
    cutoff = datetime.fromisoformat(handoff["polling_input_cutoff_utc"])
    output = []
    for row in rows:
        status, reason = explain(row, selected, variants, earlier, records)
        require(row["question_universe"] in UNIVERSES and row["question_kind"] in KINDS, "Unexplained question type")
        require(urlparse(row["source_url"]).scheme in {"https", "http"} and urlparse(row["source_url"]).netloc, "Invalid source URL")
        for key in ("field_start", "field_end", "publication_date"):
            if row.get(key):
                require(date.fromisoformat(row[key]) <= cutoff.date(), "Future question date")
        bound = row.get("available_by_utc")
        if bound:
            require(datetime.fromisoformat(bound) <= cutoff, "Future availability bound")
        if status == "included":
            require(row.get("publication_date") or bound, "Included question lacks availability evidence")
        for ref in row["source_files"]:
            pinned(root, ref)
            sources[f"evidence:{ref['sha256']}"] = ref["sha256"]
        output.append({"id": row["release_id"], "raceId": row["race_id"], "state": row["state"], "pollster": row["pollster"],
                       "fieldStart": row.get("field_start"), "fieldEnd": row.get("field_end"), "publicationDate": row.get("publication_date"),
                       "publicationDateVerified": bool(row.get("publication_date")) and row["admission_status"] not in {"held_publication_date_not_verified", "held_original_source_unavailable"},
                       "publicationBasis": row.get("publication_basis"), "availableBy": bound,
                       "availabilityBasis": row.get("availability_basis"),
                       "sourceUrl": row["source_url"], "status": status, "reason": reason,
                       "questionUniverse": UNIVERSES[row["question_universe"]], "questionKind": KINDS[row["question_kind"]],
                       "providerGrouping": row["fieldwork_provider_id"], "providerIdentityStatus": row["fieldwork_provider_identity_status"],
                       "actualFieldworkProvider": row.get("actual_fieldwork_provider_id"), "questionSampleSize": row.get("question_specific_n"),
                       "nominalSampleSize": row.get("nominal_sample_n"),
                       "nominalSampleUniverse": UNIVERSES.get(row.get("nominal_sample_universe")),
                       "sourceQualifications": row.get("qualifiers", []),
                       "originalDocumentBytesAcquired": row["original_html_or_pdf_bytes_acquired"],
                       "preferredSurveyQuestions": [{"id": other["release_id"], "questionKind": KINDS[other["question_kind"]],
                           "questionUniverse": UNIVERSES[other["question_universe"]], "included": other["release_id"] in selected}
                           for other in rows if other["sample_id"] == row["sample_id"] and other["release_id"] != row["release_id"]
                           and variants[other["release_id"]]["source_selected"]]})
    output.sort(key=lambda r: (r["raceId"], r["fieldEnd"] or "", r["id"]))
    return {"schemaVersion": 2, "cutoffAt": handoff["polling_input_cutoff_utc"], "applicableRuns": applicable,
            "sourceHashes": sources, "unknownPublicationDates": sum(not r.get("publication_date") for r in admitted), "records": output}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--bundles", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    receipt = build(args.source_root, args.bundles)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"records": len(receipt["records"]), "included": sum(r["status"] == "included" for r in receipt["records"]), "runs": len(receipt["applicableRuns"])}))
