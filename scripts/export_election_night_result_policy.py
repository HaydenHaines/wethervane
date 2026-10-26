#!/usr/bin/env python3
"""Bind the reviewed pre-result Senate pair policy to the current F and FP editions.

This importer admits policy and forecast identity metadata only. It never reads
actual election results or claims result-source admission or an evaluation lock.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

HANDOFF_SHA256 = "1d0bae622c90c79c0ce6e66ff796c0011625a38c48e2aa3b25a39140d4e6d4e1"
HANDOFF_REL = Path("research/senate-polling-refresh-20261026/result-policy-frontend-handoff-root-v1.json")
POLICY_REL = Path("research/senate-polling-refresh-20261026/result-policy-special-rules-sol-v1/policy-v1.json")
POLICY_SHA256 = "efe1bf193e2959284b73ea833af0a136f6c592001a2faf50505e3258e25cf0cd"
MAPPING_REL = Path("research/senate-polling-refresh-20261026/result-policy-special-rules-sol-v1/locked-35-result-policy-mapping-v1.json")
MAPPING_SHA256 = "244e9db017957d0a5361f5f1b7a2498f066c43d9b94a59f1277e8cff91b6afa3"
OPERATOR_REL = Path("research/senate-polling-refresh-20261026/result-policy-special-rules-sol-v1/pair_operator.py")
OPERATOR_SHA256 = "81cf4c04c974faab4388c6d623a93cb8ddfd1d6ce0aa7659e4c7cd330eb9830e"
FREEZE_REL = Path("research/senate-polling-refresh-20261026/result-policy-special-rules-sol-v1/freeze-manifest-v1.json")
FREEZE_SHA256 = "c890eda18924215ebe049af38725bb1389998b25f77f1ab91d84a4234d71a7d8"
ROUTE_REGISTRY_SHA256 = "428fcd0d87057a0cc729495e90d632f7ac90678fb6e01eec3a9c415a017a9415"
SPECIAL_SOURCES_SHA256 = "64c4b9d1bf2c979b22ddf1c7bef08c868b7206d3b839866d902d506d3a34311b"
ROUTE_EVIDENCE_SHA256 = "e9cea130e48b82575030936cdd6ad32efa203ee5e7d9a62e6b3c2a214320dc28"
COMPAT_MAPPING_SHA256 = "9024069ac0f397c2d379663d653d384e32532548ef26772035eda61ce55265e0"
NATIVE_INPUTS_SHA256 = "49d175f0c442b911e92710bb8f6d2082bf4ed8fdb675e2c57f306a194266262c"
F_PUBLIC_SHA256 = "37af627c934f519f32b2cf77a326f5658758a24153ee574fdc1a0ba46bfd51c1"
FP_PUBLIC_SHA256 = "905ce512c44104d3e48900745cec0f430c9e3954736614468e1a8b728b6da5f0"
F_NATIVE_SHA256 = "c02bdc63a0df27824578c07d6e2b752b831ae4bbd858cabc432ee00fbbf94136"
FP_NATIVE_SHA256 = "a1ffba60d937bfc81d06df51af6ed917a7ddc71fd9860c1d702de7c7c5db67cb"
REGISTRY_SHA256 = "c1e6d85bd4886768975d91d45f57a151114564dfb8f73645b484f089a5214031"
TARGET_IDS = {
    "F": "national_senate_aggregate_baseline_v1",
    "FP": "national_senate_aggregate_poll_v1",
}
DENOMINATOR = "D=first+second authenticated counts in the admitted decisive event/round; D>0. Margin_pp=100*(first-second)/D; no all-candidate, turnout, registration or exhausted-ballot denominator substitution."
EXTERNAL = Path("/home/hayden/projects/wethervane-redesign-20261006/integration")


class ResultPolicyExportError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ResultPolicyExportError(message)


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ResultPolicyExportError(f"Unable to read JSON object: {path}") from error
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def _safe_source(root: Path, relative: str | Path) -> Path:
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts, f"Unsafe source path: {relative}")
    resolved_root, resolved = root.resolve(), (root / path).resolve()
    require(resolved.is_relative_to(resolved_root), f"Unsafe source path: {relative}")
    require(resolved.is_file(), f"Missing source file: {relative}")
    return resolved


def pinned(root: Path, relative: str | Path, expected_sha: str, expected_bytes: int | None = None) -> Path:
    path = _safe_source(root, relative)
    require(sha(path) == expected_sha, f"Source hash mismatch: {relative}")
    if expected_bytes is not None:
        require(path.stat().st_size == expected_bytes, f"Source byte length mismatch: {relative}")
    return path


def _valid_url(value: Any, *, optional: bool = False) -> bool:
    return (optional and value is None) or (isinstance(value, str) and value.startswith("https://"))


def _verify_manifest_files(root: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    files = manifest.get("files")
    require(manifest.get("schema") == "SenatePreResultPolicyFreezeManifestV1"
            and isinstance(files, list) and len(files) == 16,
            "Freeze manifest identity/file count changed")
    seen: set[str] = set()
    trace: list[dict[str, Any]] = []
    for item in files:
        require(isinstance(item, dict) and isinstance(item.get("path"), str)
                and isinstance(item.get("sha256"), str) and isinstance(item.get("bytes"), int),
                "Malformed freeze-manifest row")
        relative = item["path"]
        require(relative not in seen and len(item["sha256"]) == 64
                and all(c in "0123456789abcdef" for c in item["sha256"]),
                f"Duplicate or invalid freeze-manifest pin: {relative}")
        seen.add(relative)
        pinned(root, relative, item["sha256"], item["bytes"])
        trace.append({"path": relative, "sha256": item["sha256"]})
    return trace


def _validate_policy(policy: dict[str, Any], mapping: dict[str, Any]) -> None:
    choices = policy.get("root_policy_choices", {})
    frontend = policy.get("frontend_binding", {})
    require(policy.get("schema") == "Senate35PreResultOperatorPolicyV1"
            and policy.get("status") == "root_requested_policy_choices_implemented_for_review; no_frontend_lock_or_actual_source_admission"
            and choices.get("fixed_rows", "").startswith("Keep all35 locked rows")
            and choices.get("plurality", "").startswith("Certified decisive November3 general counts")
            and choices.get("AK_ME", "").startswith("Require an official observed final round")
            and choices.get("GA", "").startswith("Use certified decisive general counts")
            and choices.get("ties", "").startswith("Certified equal pair counts permit margin0/share.5")
            and choices.get("denominator") == DENOMINATOR
            and frontend.get("same_actual_result_version_for_F_FP") is True
            and frontend.get("draw_banks", "").startswith("Preserve each bank and native weights independently")
            and frontend.get("semantic_labels", "").startswith("Pair margin and first-versus-second relative lead"),
            "Pre-result policy schema or selected rule choices changed")
    require(policy.get("operations", {}).get("actual_result_operator_called_on_2026_outcomes") is False
            and policy.get("operations", {}).get("new_forecasts_fits_or_grades") == 0
            and policy.get("acquisition_boundary", {}).get("actual2026_general_result_tables_requested_or_used") is False,
            "Policy now contains actual outcomes or grades")
    require(mapping.get("schema") == "LockedSenate35PreResultMappingV1"
            and mapping.get("race_count") == 35,
            "Locked mapping identity changed")


def _bundle_rows(bundle: dict[str, Any], run: dict[str, Any], run_id: str,
                 expected_target: str) -> list[dict[str, Any]]:
    races = bundle.get("races")
    require(isinstance(bundle.get("run"), dict) and isinstance(races, list) and len(races) == 35
            and bundle.get("run", {}).get("id") == run_id
            and bundle.get("run", {}).get("targetId") == expected_target
            and bundle.get("run", {}).get("lawId") == "senate_principal_pair_aggregate_v1"
            and bundle.get("run", {}).get("year") == 2026
            and bundle.get("run", {}).get("historical") is False,
            f"{run_id} public forecast identity changed")
    require(run.get("id") == run_id and run.get("targetId") == expected_target
            and run.get("lawId") == "senate_principal_pair_aggregate_v1"
            and run.get("href") == f"/data/explorer/{run_id}.json"
            and run.get("sha256") == (F_PUBLIC_SHA256 if run_id == "current-fundamentals" else FP_PUBLIC_SHA256),
            f"{run_id} registry entry changed")
    return races


def build(integration_root: Path, repo_root: Path) -> dict[str, Any]:
    root = integration_root.resolve()
    require(root == EXTERNAL.resolve(), "Unexpected integration source root")
    handoff_path = pinned(root, HANDOFF_REL, HANDOFF_SHA256)
    handoff = read(handoff_path)
    require(handoff.get("schema") == "SenatePreResultPolicyRootHandoffV1"
            and handoff.get("status") == "reviewed_pre_result_policy_and_synthetic_operator_ready_for_owner_integration"
            and handoff.get("forecast_or_default_changes") is False,
            "Root handoff status or scope changed")
    for key, expected_path, expected_hash in (
        ("policy", POLICY_REL, POLICY_SHA256), ("mapping", MAPPING_REL, MAPPING_SHA256),
        ("operator", OPERATOR_REL, OPERATOR_SHA256), ("freeze_manifest", FREEZE_REL, FREEZE_SHA256),
    ):
        spec = handoff.get(key, {})
        require(spec.get("path") == str(root / expected_path) and spec.get("sha256") == expected_hash,
                f"Handoff {key} pin changed")
    policy = read(pinned(root, POLICY_REL, POLICY_SHA256))
    mapping = read(pinned(root, MAPPING_REL, MAPPING_SHA256))
    operator_path = pinned(root, OPERATOR_REL, OPERATOR_SHA256)
    freeze_path = pinned(root, FREEZE_REL, FREEZE_SHA256)
    freeze = read(freeze_path)
    manifest_files = _verify_manifest_files(root, freeze)

    dependency_specs = [
        mapping.get("native_inputs", {}), mapping.get("compatibility_mapping", {}),
        mapping.get("official_route_registry", {}),
    ]
    dependency_files: list[dict[str, Any]] = []
    for spec in dependency_specs:
        require(isinstance(spec.get("path"), str) and isinstance(spec.get("sha256"), str),
                "Malformed mapping dependency pin")
        pinned(root, spec["path"], spec["sha256"], spec.get("bytes"))
        dependency_files.append({"path": spec["path"], "sha256": spec["sha256"]})
    require(dependency_specs[1]["sha256"] == COMPAT_MAPPING_SHA256
            and dependency_specs[2]["sha256"] == ROUTE_REGISTRY_SHA256,
            "Mapping compatibility/route registry dependency changed")
    route_evidence_rel = Path("research/senate-polling-refresh-20261026/result-policy-sources-luna-v1/web-evidence.json")
    special_sources_rel = Path("research/senate-polling-refresh-20261026/result-policy-special-rules-sol-v1/official-special-rule-sources-v1.json")
    pinned(root, route_evidence_rel, ROUTE_EVIDENCE_SHA256)
    pinned(root, special_sources_rel, SPECIAL_SOURCES_SHA256)

    _validate_policy(policy, mapping)
    compatibility = read(root / dependency_specs[1]["path"])
    route_registry = read(root / dependency_specs[2]["path"])
    require(compatibility.get("schema") == "SenateFToFPExplicitScorecardCompatibilityMappingV1"
            and compatibility.get("status") == "reviewed_native_pair_proxy_compatibility_only"
            and compatibility.get("race_count") == 35
            and compatibility.get("source_target_ids_unchanged") == TARGET_IDS
            and route_registry.get("schema") == "official_state_result_source_registry_partial_v1"
            and route_registry.get("status") == "partial_source_discovery_checkpoint_not_complete"
            and route_registry.get("race_count") == 35,
            "Pinned roster/event or route-registry contract changed")

    cohort_sets = mapping.get("fixed_cohorts", {})
    full_ids = cohort_sets.get("full_35_locked_rows")
    core_ids = cohort_sets.get("plurality_core_32")
    conditional_ids = cohort_sets.get("special_conditional_3")
    require(isinstance(full_ids, list) and len(full_ids) == 35 and len(set(full_ids)) == 35
            and isinstance(core_ids, list) and len(core_ids) == 32 and len(set(core_ids)) == 32
            and isinstance(conditional_ids, list) and len(conditional_ids) == 3 and len(set(conditional_ids)) == 3
            and set(core_ids).isdisjoint(conditional_ids) and set(full_ids) == set(core_ids) | set(conditional_ids),
            "Fixed 35/32/3 cohorts no longer form an exact partition")
    require(policy.get("frontend_binding", {}).get("fixed_cohorts", {}) == cohort_sets,
            "Policy cohort arrays differ from locked mapping")
    raw_rows = mapping.get("races")
    routes = route_registry.get("races")
    compatible_rows = compatibility.get("races")
    require(isinstance(raw_rows, list) and len(raw_rows) == 35
            and isinstance(routes, list) and len(routes) == 35
            and isinstance(compatible_rows, list) and len(compatible_rows) == 35,
            "Expected 35 mapping, route, and compatibility rows")

    registry_path = repo_root / "public/data/explorer/registry.json"
    f_path = repo_root / "public/data/explorer/current-fundamentals.json"
    fp_path = repo_root / "public/data/explorer/current-polling.json"
    require(sha(registry_path) == REGISTRY_SHA256 and sha(f_path) == F_PUBLIC_SHA256
            and sha(fp_path) == FP_PUBLIC_SHA256,
            "Current F/FP public forecast or registry bytes changed")
    registry = read(registry_path)
    f_entry = next((row for row in registry.get("runs", []) if row.get("id") == "current-fundamentals"), None)
    fp_entry = next((row for row in registry.get("runs", []) if row.get("id") == "current-polling"), None)
    f_bundle, fp_bundle = read(f_path), read(fp_path)
    f_races = _bundle_rows(f_bundle, f_entry or {}, "current-fundamentals", TARGET_IDS["F"])
    fp_races = _bundle_rows(fp_bundle, fp_entry or {}, "current-polling", TARGET_IDS["FP"])
    require(f_bundle.get("provenance", {}).get("sources", {}).get("forecastSha256") == F_NATIVE_SHA256
            and fp_bundle.get("provenance", {}).get("sources", {}).get("forecastSha256") == FP_NATIVE_SHA256,
            "F/FP native forecast provenance changed")

    candidate_ids = {row["race_id"]: (row["first_candidate_id"], row["second_candidate_id"]) for row in raw_rows}
    require(len(candidate_ids) == 35, "Duplicate mapping race identity")
    rows: list[dict[str, Any]] = []
    for index, (raw, route, compatible, f_race, fp_race) in enumerate(zip(raw_rows, routes, compatible_rows, f_races, fp_races, strict=True)):
        race_id = full_ids[index]
        require(raw.get("race_id") == route.get("race_id") == compatible.get("race_id")
                == f_race.get("id") == fp_race.get("id") == race_id
                and raw.get("state") == f_race.get("state") == fp_race.get("state")
                and raw.get("seat_id") == f_race.get("seatId") == fp_race.get("seatId")
                and raw.get("event_kind") == f_race.get("electionLabel", "").removesuffix(" election").lower()
                and raw.get("event_kind") == fp_race.get("electionLabel", "").removesuffix(" election").lower(),
                f"row[{index}] race/seat/event identity differs across sources")
        require(raw.get("fixed_full_35_member") is True
                and raw.get("fixed_plurality_core_32_member") is (race_id in set(core_ids))
                and raw.get("official_future_feed_candidate_ids") is None
                and raw.get("official_future_feed_contest_id") is None
                and raw.get("source_route_is_not_actual_result_admission") is True
                and raw.get("initial_result_status") == "unavailable_no_actual_result_read",
                f"{race_id} future-feed or pre-result status changed")
        locked_candidates = raw.get("locked_candidates", [])
        require(isinstance(locked_candidates, list) and len(locked_candidates) == 2,
                f"{race_id} locked pair must have exactly two candidates")
        first, second = locked_candidates
        require(first.get("id") == raw.get("first_candidate_id")
                and second.get("id") == raw.get("second_candidate_id")
                and first.get("id") != second.get("id"), f"{race_id} locked pair malformed")
        f_outcomes, fp_outcomes = f_race.get("outcomes", []), fp_race.get("outcomes", [])
        require(len(f_outcomes) == len(fp_outcomes) == 2,
                f"{race_id} forecasts no longer expose exact pair outcomes")
        for contender, f_outcome, fp_outcome in zip((first, second), f_outcomes, fp_outcomes, strict=True):
            support = contender.get("official_roster_source_support", {})
            urls = support.get("source_urls", [])
            require(contender.get("id") == f_outcome.get("id") == fp_outcome.get("id")
                    and contender.get("name") == f_outcome.get("name") == fp_outcome.get("name")
                    and contender.get("party") == f_outcome.get("party") == fp_outcome.get("party")
                    and contender.get("name") == f_outcome.get("members", [{}])[0].get("name")
                    and contender.get("id") == f_outcome.get("members", [{}])[0].get("id")
                    and contender.get("name") == fp_outcome.get("members", [{}])[0].get("name")
                    and contender.get("id") == fp_outcome.get("members", [{}])[0].get("id")
                    and f_outcome.get("kind") == fp_outcome.get("kind") == "candidate"
                    and isinstance(urls, list) and all(_valid_url(url) for url in urls if url is not None),
                    f"{race_id} candidate ID/name/party differs from F or FP")
        require(compatible.get("candidate_pairs_exactly_identical") is True
                and compatible.get("first_candidate") == {"id": first["id"], "name": first["name"], "party": first["party"]}
                and compatible.get("second_candidate") == {"id": second["id"], "name": second["name"], "party": second["party"]}
                and compatible.get("F_source_target_id") == TARGET_IDS["F"]
                and compatible.get("FP_source_target_id") == TARGET_IDS["FP"],
                f"{race_id} prior F/FP pair compatibility mapping changed")
        route_meta = raw.get("source_route_metadata", {})
        require(route_meta.get("official_state_election_authority_url") == route.get("official_state_election_authority_url")
                and route_meta.get("official_results_or_certification_landing_url") == route.get("official_results_or_certification_landing_url")
                and route_meta.get("route_status") == route.get("route_status")
                and route.get("internal_contender_ids_are_not_official_feed_keys") is True,
                f"{race_id} route metadata does not match pinned registry")
        roster_urls = []
        for contender in (first, second):
            urls = contender.get("official_roster_source_support", {}).get("source_urls", [])
            roster_urls.append([url for url in urls if url is not None])
        rows.append({
            "raceId": race_id, "state": raw["state"], "seatId": raw["seat_id"],
            "eventKind": raw["event_kind"], "generalDate": raw["scheduled_general_event_date"],
            "runoffDate": raw["scheduled_runoff_event_date"],
            "firstCandidateId": first["id"], "secondCandidateId": second["id"],
            "firstCandidateName": first["name"], "secondCandidateName": second["name"],
            "firstParty": first["party"], "secondParty": second["party"],
            "operatorFamily": raw["operator_family"], "denominator": DENOMINATOR,
            "coreMember": race_id in set(core_ids),
            "officialAuthorityUrl": route_meta["official_state_election_authority_url"],
            "officialResultsUrl": route_meta["official_results_or_certification_landing_url"],
            "sourceRouteStatus": route_meta["route_status"],
            "crosswalkStatus": raw["future_result_crosswalk_status"],
            "firstCandidateRosterSourceUrls": roster_urls[0],
            "secondCandidateRosterSourceUrls": roster_urls[1],
            "officialFeedCandidateIds": None, "officialFeedContestId": None,
        })

    require([row["raceId"] for row in rows] == full_ids
            and [row["raceId"] for row in rows if row["coreMember"]] == core_ids
            and [row["raceId"] for row in rows if not row["coreMember"]] == conditional_ids,
            "Normalized rows no longer preserve fixed ordered cohorts")
    extra_files = [
        {"path": str(spec["path"]), "sha256": spec["sha256"]} for spec in dependency_specs
    ] + [
        {"path": str(route_evidence_rel), "sha256": ROUTE_EVIDENCE_SHA256},
        {"path": str(special_sources_rel), "sha256": SPECIAL_SOURCES_SHA256},
    ]
    source_files_by_path = {row["path"]: row for row in manifest_files}
    source_files_by_path.update({row["path"]: row for row in extra_files})
    return {
        "schemaVersion": 1,
        "status": "preparation_not_evaluation_lock",
        "evaluationLock": False,
        "observedResultsUsed": False,
        "actualGradesAssigned": False,
        "sourceHashes": {
            "handoffSha256": HANDOFF_SHA256, "policySha256": POLICY_SHA256,
            "mappingSha256": MAPPING_SHA256, "operatorSha256": OPERATOR_SHA256,
            "freezeManifestSha256": FREEZE_SHA256,
            "FPublicSha256": F_PUBLIC_SHA256, "FPPublicSha256": FP_PUBLIC_SHA256,
            "registrySha256": REGISTRY_SHA256,
        },
        "sourceFiles": list(source_files_by_path.values()),
        "forecastBindings": {
            "F": {"runId": "current-fundamentals", "targetId": TARGET_IDS["F"],
                  "lawId": "senate_principal_pair_aggregate_v1", "publicSha256": F_PUBLIC_SHA256,
                  "nativeForecastSha256": F_NATIVE_SHA256},
            "FP": {"runId": "current-polling", "targetId": TARGET_IDS["FP"],
                   "lawId": "senate_principal_pair_aggregate_v1", "publicSha256": FP_PUBLIC_SHA256,
                   "nativeForecastSha256": FP_NATIVE_SHA256},
        },
        "fixedCohorts": {"full35": full_ids, "pluralityCore32": core_ids, "conditional3": conditional_ids},
        "cohortSemantics": {
            "primaryRecommendation": "32-row plurality core is the proposed primary grade cohort; owner evaluation lock remains separate.",
            "full35": "Keep all35 named principal pairs visible. The 32/core and 3/conditional cohorts are fixed before results.",
            "conditional3": "Alaska and Maine require an official observed exact two-candidate final round. Georgia requires a certified decisive no-runoff general or qualifying exact-principal runoff.",
            "gradeTarget": "Named-pair margin and first-versus-second relative lead, not unconditional all-candidate or legal office winner.",
            "sameResultRevision": "F and FP must use the same admitted official result record revision and frozen cohort.",
        },
        "sourceGaps": {"policy": policy.get("source_gaps", []),
                       "resultRouteRegistry": route_registry.get("known_exceptions", [])},
        "sourceFinalityPolicy": policy.get("source_finality_policy", {}),
        "rows": rows,
    }


def export(integration_root: Path, repo_root: Path, output: Path) -> dict[str, Any]:
    catalog = build(integration_root, repo_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(catalog, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return catalog


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integration-root", type=Path, default=EXTERNAL)
    parser.add_argument("--repo-root", type=Path, default=repo)
    parser.add_argument("--output", type=Path, default=repo / "data/election-night-senate-result-policy.json")
    args = parser.parse_args()
    result = export(args.integration_root, args.repo_root, args.output)
    print(f"Exported pre-result Senate policy for {len(result['rows'])} frozen races")


if __name__ == "__main__":
    main()
