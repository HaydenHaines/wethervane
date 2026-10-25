"""Typed M02 metadata/review bridge; no fitting, rescoring or model activation."""

import argparse
import copy
import hashlib
import math
import os
import sqlite3
import tempfile
from pathlib import Path

import execution
from ledger import Ledger, canonical
from run_receipts import decode

ROOT_ROLES = {"admission", "scoring_terminal", "scoring_receipt", "scores", "independent_review", "root_acceptance", "root_decision"}
CYCLES = [2018, 2020, 2022, 2024]
TIERS = {"A": [2018, 2022], "B": [2020, 2024]}
FITTED_ARMS = {"FS5", "FS6", "LP5"}
SCORE_ARMS = FITTED_ARMS | {"PO5", "F0_archived_predictive"}
METRICS = {"brier", "crps", "negative_log", "wis"}


def require(value, message):
    execution.require(value, message)


def identity(pin):
    require(isinstance(pin, dict) and isinstance(pin.get("path"), str) and Path(pin["path"]).is_absolute()
            and isinstance(pin.get("sha256"), str) and len(pin["sha256"]) == 64, "absolute SHA pin required")
    return {"path": pin["path"], "sha256": pin["sha256"]}


def same(left, right):
    return identity(left) == identity(right)


def request_pin(path, expected_sha256):
    item = execution.pin(Path(path).resolve())
    require(item["sha256"] == expected_sha256, "request/receipt SHA mismatch")
    return item


class Metadata:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.documents = {}
        self.pins = {}

    def read(self, item):
        expected = identity(item)
        path = Path(expected["path"])
        require(path.resolve() == path and path.is_relative_to(self.root) and path.suffix == ".json" and not path.is_symlink(),
                "metadata must be canonical JSON inside declared source root")
        key = str(path)
        require(key not in self.pins or self.pins[key]["sha256"] == expected["sha256"], "conflicting metadata pin")
        if key not in self.documents:
            data = path.read_bytes()
            require(hashlib.sha256(data).hexdigest() == expected["sha256"], "metadata fingerprint mismatch: " + key)
            value = decode(data)
            require(isinstance(value, dict), "metadata document must be an object")
            self.pins[key] = dict(expected, size=len(data))
            self.documents[key] = value
        for size_key in ("size", "bytes"):
            require(size_key not in item or item[size_key] == self.pins[key]["size"], "metadata byte count mismatch")
        return self.documents[key]


def inspect(request):
    """Authenticate one supported native graph; prior scientific review stays separate."""
    request_data = decode(execution.checked(request))
    require(set(request_data) == {"schema", "source_root", "cycle_id", "cycle_binding", "sources"}
            and request_data["schema"] == "ReviewedMomentumComparisonRequestV1", "unsupported comparison request")
    require(set(request_data["sources"]) == ROOT_ROLES, "comparison source roles mismatch")
    reader = Metadata(request_data["source_root"])
    sources = request_data["sources"]
    documents = {role: reader.read(pin) for role, pin in sources.items()}
    admission, terminal, receipt, scores = (documents[k] for k in ("admission", "scoring_terminal", "scoring_receipt", "scores"))
    review, acceptance, decision = (documents[k] for k in ("independent_review", "root_acceptance", "root_decision"))
    require(admission["schema"] == "MomentumForecastScoringAdmissionV1", "unsupported scoring admission")
    require(terminal["schema"] == "MomentumScoringTerminalV1" and terminal["status"] == "complete_scores"
            and terminal["numeric_labels_opened"] is True, "incomplete scoring terminal")
    require(receipt["schema"] == "MomentumScoringOutputReceiptV1" and terminal["receipt"] == receipt
            and same(terminal["admission"], sources["admission"]) and same(receipt["admission"], sources["admission"]), "scorer admission/receipt binding mismatch")
    require(receipt["post_label_sampling"] is False and receipt["source_and_numeric_status_failures_distinct"] is True,
            "unsupported scoring policy")
    require(scores["schema"] == "MomentumMatchedForecastScoresV1" and scores["target"] == "conditional_aggregate_DR_share_no_full_chamber_claim"
            and scores["post_label_sampling"] is False, "unsupported scoring scope")
    require(set(receipt["artifacts"]) == {"scores.json", "cycle-influences.npz", "cycle-influence-scales.json", "launch.json", "label-access.json"}, "unsupported scoring artifact roster")
    require(same(receipt["artifacts"]["scores.json"], sources["scores"]), "scores receipt binding mismatch")
    for name in ("launch.json", "label-access.json", "cycle-influence-scales.json"):
        reader.read(receipt["artifacts"][name])
    launch = reader.read(receipt["artifacts"]["launch.json"])
    require(same(launch["admission"], sources["admission"]) and launch["numeric_labels_opened"] is False
            and launch["post_label_sampling"] is False, "scoring launch policy mismatch")
    access = reader.read(receipt["artifacts"]["label-access.json"])
    require(access["schema"] == "MomentumNumericLabelAccessV1" and access["all_required_receipts_verified"] is True
            and same(access["label_source"], admission["label_source"]), "numeric-label access metadata mismatch")
    require(scores["label_audit"] == ["all_prediction_reference_mask_status_receipts_verified_before_numeric_label_read",
            "numeric_outcome_source_read_after_all_receipts"], "unsupported label audit")
    selection = reader.read(admission["prelabel_selection"])
    masks = reader.read(admission["mask_receipt"])
    references = reader.read(admission["references_terminal"])
    require(selection["schema"] == "MomentumFrozenPrelabelAttemptSelectionV1" and selection["core_bulk_bank_all_pass"] is True
            and selection["win_lose_tail_all_pass"] is True, "unsupported prelabel selection")
    require(masks["schema"] == "MomentumFrozenScoreMasksV1" and same(masks["prediction_selection"], admission["prelabel_selection"])
            and same(masks["references_terminal"], admission["references_terminal"]), "frozen mask binding mismatch")
    require(references["schema"] == "MomentumPrelabelReferencesTerminalV1", "unsupported prelabel references")
    for key in ("execution_manifest", "scoring_manifest", "independent_scoring_review", "runtime_receipt", "scoring_runtime"):
        reader.read(admission[key])
    reader.read(admission["execution_identity"]["independent_execution_review"])
    require(admission["runtime_receipt"]["sha256"] == admission["scoring_runtime"]["sha256"]
            and reader.read(admission["runtime_receipt"]) == reader.read(admission["scoring_runtime"]), "runtime pin mismatch")
    folds = admission["selected_folds"]
    require([fold["cycle"] for fold in folds] == CYCLES and set(scores["cycles"]) == {str(c) for c in CYCLES}
            and set(scores["tiers"]) == set(TIERS), "matched cycle roster mismatch")
    forecast_statuses = []
    selected_terminals = []
    for fold in folds:
        require(set(fold["jobs"]) == FITTED_ARMS, "native fitted arm roster mismatch")
        require(scores["cycles"][str(fold["cycle"])] ["selected_attempt"] == fold["attempt"]
                == selection["selected_attempt_by_cycle"][str(fold["cycle"])], "selected attempt mismatch")
        for arm, pins in fold["jobs"].items():
            selected_terminals.append(identity(pins["terminal"]))
            native = reader.read(pins["terminal"])
            seal = reader.read(pins["prediction_seal"])
            numerical = reader.read(pins["numerical_status"])
            require(numerical["common_polled_race_ids"] == masks["folds"][str(fold["cycle"])]["common_polled_race_ids"], "forecast common race support mismatch")
            require(native["schema"] == "MomentumForecastJobTerminalV1" and native["status"] == "complete_prelabel"
                    and native["numerical_core_bulk_passed"] is True, "forecast numerical gate failed")
            require(native["job"]["cycle"] == fold["cycle"] and native["job"]["arm"] == arm
                    and native["job"]["attempt"] == fold["attempt"] and seal["job"] == native["job"], "forecast job cross-binding mismatch")
            require(seal["schema"] == "MomentumPollingPredictionSealV1" and numerical["schema"] == "MomentumPollingPrelabelNumericalSummaryV1"
                    and same(native["prelabel_summary"], pins["numerical_status"]), "forecast numerical/seal binding mismatch")
            require(native["numeric_target_labels"] is False and seal["numeric_target_labels"] is False, "prelabel target access mismatch")
            require(set(numerical["gates"]) == {"core_state", "bulk_absolute_win_probability", "bulk_mean_share", "bank_chain_consistency"}
                    and all(value is True for value in numerical["gates"].values()) and numerical["tail_qualification"] == {"win": True, "lose": True}
                    and numerical["inherited_bank_warning"] is False and numerical["bank"]["gates"]
                    and all(value is True for value in numerical["bank"]["gates"].values()),
                    "native core/bulk/bank/tail gate mismatch")
            provenance = native["provenance"]
            require(provenance["execution_group_id"] == admission["execution_identity"]["execution_group_id"]
                    and provenance["forecast_design_sha256"] == admission["forecast_design_sha256"]
                    and provenance["numerical_clarification_sha256"] == admission["numerical_clarification_sha256"]
                    and provenance["source_identity"] == native["source_identity"] == seal["source_identity"] == numerical["source_identity"], "native source/design identity mismatch")
            executed = reader.read(provenance["execution_receipt"])
            for key, expected in admission["execution_identity"].items():
                require(same(executed[key], expected) if isinstance(expected, dict) else executed[key] == expected,
                        "execution receipt/scoring identity mismatch")
            native_launch = reader.read(native["artifacts"]["launch"])
            require(native_launch["job"] == native["job"] and same(native_launch["provenance"]["execution_receipt"], provenance["execution_receipt"]), "forecast launch binding mismatch")
            for name in ("original", "trace"):
                require(same(native["artifacts"][name], seal["artifacts"][name]), "prediction seal artifact binding mismatch")
            forecast_statuses.append(dict(cycle=fold["cycle"], arm=arm, attempt=fold["attempt"],
                                          core_bulk_prior_gate=True, interval_display_qualified=numerical["interval_display_qualified"]))
    require(sorted(selected_terminals, key=canonical) == sorted((identity(p) for p in selection["selected_terminals"]), key=canonical), "frozen selected terminal roster mismatch")
    require(review["schema"] == "MomentumActualForecastResultsIndependentReviewSolV1"
            and review["disposition"] == "verified_completed_score_artifacts_mixed_A_B_retain_FS5_comparator", "unsupported independent comparison review")
    require(same(review["actual_terminal"], sources["scoring_terminal"]) and same(review["actual_output_receipt"], sources["scoring_receipt"])
            and same(review["actual_artifacts"]["scores.json"], sources["scores"]) and same(review["admission"], sources["admission"]), "independent review cross-binding mismatch")
    require(set(review["actual_artifacts"]) == set(receipt["artifacts"])
            and all(same(pin, receipt["artifacts"][name]) for name, pin in review["actual_artifacts"].items()), "independent review artifact cross-binding mismatch")
    require(review["promotion"] is False and review["full_count_qualification"] is False, "unsupported positive review claim")
    require(acceptance["schema"] == "MomentumActualForecastResultsRootAcceptanceV1"
            and acceptance["disposition"] == "accepted_verified_conditional_scores_no_full_count_or_public_qualification"
            and acceptance["full_count_or_chamber_qualification"] is False and acceptance["sampling_or_retuning_after_labels"] is False,
            "unsupported root acceptance")
    require(acceptance["no_computational_inconsistency_found_in_bounded_review"] is True, "root acceptance retains an unresolved computational inconsistency")
    require(same(acceptance["actual_terminal"], sources["scoring_terminal"]) and same(acceptance["actual_receipt"], sources["scoring_receipt"])
            and same(acceptance["independent_review"], sources["independent_review"]), "root acceptance cross-binding mismatch")
    require(decision["schema"] == "MomentumForecastDecisionV1" and decision["decision"] == "retain", "unsupported decision/schema; only scoped retain is supported")
    require(decision["experiment"] == request_data["cycle_id"] and same(decision["acceptance"], sources["root_acceptance"])
            and decision["incumbent_comparator"].startswith("FS5 ") and decision["challenger"].startswith("FS6 "), "decision scope/binding mismatch")
    require(decision["preservation"]["new_active_layer_promotion"] is False and decision["preservation"]["public_default_changed"] is False
            and decision["preservation"]["incumbent_retained"] is True, "unsupported activation/preservation claim")
    binding = request_data["cycle_binding"]
    require(set(binding) == {"incumbent_config", "scope", "target_layer", "hypothesis"}, "full journal cycle binding required")
    scope = binding["scope"]
    require(scope["A"] == TIERS["A"] and scope["B"] == TIERS["B"] and scope["horizon_days"] == 14
            and scope["aggregate_comparator_only"] is True and scope["strict_asof_qualified"] is False
            and binding["incumbent_config"]["arm"] == "FS5" and binding["incumbent_config"]["design_sha256"] == admission["forecast_design_sha256"]
            and binding["target_layer"] == "measurement_and_temporal_dependence_aggregate_comparison", "unsupported journal scientific scope")
    require(selection["design_sha256"] == masks["forecast_design_sha256"] == admission["forecast_design_sha256"], "forecast design binding mismatch")
    require(scope["common_polled_races"] == [len(masks["folds"][str(year)]["common_polled_race_ids"]) for year in CYCLES], "journal common race support mismatch")
    primary = {}
    require({row["tier"] for row in decision["primary_common_crps"]} == set(TIERS) and len(decision["primary_common_crps"]) == 2,
            "decision primary tier roster mismatch")
    for row in decision["primary_common_crps"]:
        tier = row["tier"]
        require(row["cycles"] == TIERS[tier] and row["mask"] == "common" and row["metric"] == "crps", "decision tier/mask mismatch")
        arms = scores["tiers"][tier]["common"]["arms"]
        difference = scores["primary_differences"][tier]["common"]["crps"]
        for arm in ("FS5", "FS6"):
            cell = arms[arm]["crps"]
            require(cell["cycles"] == TIERS[tier] and cell["value"] == row[arm] and cell["qualified"] == row[arm + "_qualified"], "saved score/decision cell mismatch")
        require(difference["value"] == row["FS6_minus_FS5"] and difference["three_mcse"] == row["difference_three_mcse"]
                and difference["numerically_resolved"] == row["difference_numerically_resolved"], "saved score/decision difference mismatch")
        primary[tier] = copy.deepcopy(row)
    # Keep every cell status, including empty/unqualified references; never relabel
    # the scorer's `qualified` as scientific model qualification.
    cells = []
    for year, cycle in scores["cycles"].items():
        require(set(cycle["masks"]) == {"common", "competitive", "near_even"}, "cycle mask roster mismatch")
        for mask, item in cycle["masks"].items():
            expected_ids = masks["folds"][year][{"common": "common_polled_race_ids", "competitive": "competitive_race_ids", "near_even": "near_even_race_ids"}[mask]]
            require(item["race_ids"] == expected_ids and item["empty_mask_unavailable"] == (not expected_ids)
                    and set(item["arms"]) == SCORE_ARMS, "fixed race mask/score arm mismatch")
            for arm, value in item["arms"].items():
                require(set(value["aggregates"]) == METRICS and value["mask_race_ids"] == expected_ids, "cycle metric/race roster mismatch")
                for metric, cell in value["aggregates"].items():
                    check_cell(cell)
                    cells.append(dict(cycle=int(year), mask=mask, arm=arm, metric=metric, value=cell["value"], status=cell["status"],
                                      conditional_numerical_precision=cell["qualified"], native_cell=copy.deepcopy(cell)))
    for tier, masks_by_name in scores["tiers"].items():
        require(set(masks_by_name) == {"common", "competitive", "near_even"}, "score mask roster mismatch")
        for mask, item in masks_by_name.items():
            require(set(item["arms"]) == SCORE_ARMS, "tier score arm roster mismatch")
            for arm, metrics in item["arms"].items():
                require(set(metrics) == METRICS, "tier metric roster mismatch")
                for metric, cell in metrics.items():
                    check_cell(cell)
                    require(cell["cycles"] == TIERS[tier], "metric cycle mismatch")
                    cycle_cells = [scores["cycles"][str(year)]["masks"][mask]["arms"][arm]["aggregates"][metric] for year in TIERS[tier]]
                    if any(source["value"] is None for source in cycle_cells):
                        require(cell["value"] is None and cell["qualified"] is False and cell["status"] == "metric_unresolved",
                                "unavailable cycle must propagate to unresolved equal-cycle tier")
                    cells.append(dict(tier=tier, mask=mask, arm=arm, metric=metric, value=cell["value"], status=cell["status"],
                                      conditional_numerical_precision=cell["qualified"], native_cell=copy.deepcopy(cell)))
    report = dict(schema="ReviewedMomentumComparisonReceiptV1", status="verified_scoped_retain", request=request,
                  cycle_id=request_data["cycle_id"], cycle_binding=binding, sources=sources, recommendation="retain",
                  primary_common_crps=primary, metric_cells=cells, forecast_prior_numerical_statuses=forecast_statuses,
                  interval_display_limitations=selection["interval_display_limitations"],
                  prior_independent_review=dict(receipt=sources["independent_review"], arithmetic_checks=review["arithmetic_checks"],
                    positive_point_regressions=review["positive_point_regressions"], numerically_resolved_positive_regressions=review["numerically_resolved_positive_regressions"],
                    unresolved_scientific_limits=review["unresolved_scientific_limits"]),
                  root_decision_scope=decision["scope"], unresolved_limits=decision["unresolved_limits"],
                  verified_metadata=[reader.pins[k] for k in sorted(reader.pins)],
                  verification_scope="JSON metadata bytes and typed cross-bindings; prior independent scientific review preserved, not recomputed",
                  raw_prediction_tensors_loaded=False, raw_outcome_source_loaded=False, raw_bank_revalidated=False,
                  linked_not_reopened=dict(raw_label_source=admission["label_source"], cycle_influence_array=receipt["artifacts"]["cycle-influences.npz"]),
                  score_cell_qualification_meaning="conditional_numerical_precision_only",
                  full_count_qualification=False, scientific_model_promotion=False, publication_authorized=False)
    return report, reader


def check_cell(cell):
    statuses = {"deterministic_analytic_no_MCMC", "empty_binary_or_race_mask", "qualified_conditional_numerical",
                "source_chain_dispersion_only", "uncertainty_unresolved", "metric_unresolved"}
    require(cell["status"] in statuses and type(cell["qualified"]) is bool, "unsupported numerical cell status")
    require(cell["qualified"] == (cell["status"] in {"deterministic_analytic_no_MCMC", "qualified_conditional_numerical"}), "cell precision/status contradiction")
    value = cell["value"]
    require(value is None or type(value) in (int, float) and math.isfinite(value), "unsupported score value")
    require(not cell["qualified"] or value is not None, "qualified numerical cell has no value")
    if cell["status"] == "empty_binary_or_race_mask":
        require(value is None and cell["qualified"] is False and cell["count"] == 0, "empty score cell requires null value and zero count")


def validate(request, output):
    require(not Path(output).exists(), "comparison output must be fresh")
    report, _ = inspect(request)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    execution.publish(output / "receipt.json", report)
    return execution.pin(output / "receipt.json")


def journal_snapshot(root, cycle_id, command_id=None):
    """A missing journal/cycle fails without creating files or opening a writer."""
    root = Path(root).resolve()
    require((root / "ledger.sqlite3").is_file(), "journal database does not exist")
    connection = sqlite3.connect((root / "ledger.sqlite3").as_uri() + "?mode=ro", uri=True)
    try:
        connection.execute("BEGIN")
        row = connection.execute("SELECT binding FROM cycles WHERE id=?", (cycle_id,)).fetchone()
        require(row is not None, "journal cycle does not exist")
        history = [dict(revision=revision, status=status, body=decode(body)) for revision, status, body in connection.execute(
            "SELECT revision,status,body FROM checkpoints WHERE cycle=? ORDER BY revision", (cycle_id,))]
        previous = connection.execute("SELECT payload,response FROM commands WHERE id=?", (command_id,)).fetchone() if command_id else None
        require([item["revision"] for item in history] == list(range(len(history))) and history, "invalid journal checkpoint history")
        digests = set(history[-1]["body"]["references"].values())
        if previous:
            digests.update(decode(previous[0]).get("body", {}).get("references", {}).values())
        sizes = {}
        for digest in digests:
            artifact = connection.execute("SELECT size FROM artifacts WHERE hash=?", (digest,)).fetchone()
            require(artifact is not None, "journal reference is not a registered artifact")
            sizes[digest] = artifact[0]
        return dict(binding=decode(row[0]), history=history, artifact_sizes=sizes,
                    previous_command=(dict(payload=decode(previous[0]), response=decode(previous[1])) if previous else None))
    finally:
        connection.close()


def reconcile(comparison_receipt, expected_request, journal_root, cycle_id, expected_revision, command_id, output):
    require(not Path(output).exists(), "reconciliation output must be fresh")
    supplied = decode(execution.checked(comparison_receipt))
    require(isinstance(supplied, dict) and "request" in supplied, "comparison receipt missing original request")
    require(same(supplied["request"], expected_request), "comparison receipt differs from independently expected request")
    execution.checked(expected_request)
    report, reader = inspect(supplied["request"])
    require(canonical(supplied) == canonical(report), "normalized comparison receipt differs from reconstructed original request")
    require(cycle_id == report["cycle_id"] and type(expected_revision) is int and expected_revision >= 0
            and isinstance(command_id, str) and command_id.strip(), "explicit cycle/revision/command required")
    snapshot = journal_snapshot(journal_root, cycle_id, command_id)
    require(canonical(snapshot["binding"]) == canonical(report["cycle_binding"]), "journal full cycle binding mismatch")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    current = snapshot["history"][-1]
    references = {"evaluation": report["sources"]["root_decision"]["sha256"], "review": report["sources"]["independent_review"]["sha256"]}
    note = "Reviewed M02 scoped retain; comparison=" + comparison_receipt["sha256"] + "; request=" + report["request"]["sha256"]
    payload = dict(operation="checkpoint", cycle=cycle_id, expected_revision=expected_revision, status="decided",
                   body=dict(note=note, references=references, outcome="success", recommendation="retain"))
    previous = snapshot["previous_command"]
    if previous:
        require(canonical(previous["payload"]) == canonical(payload), "command ID conflicts with comparison payload")
        require(isinstance(previous["response"], dict) and type(previous["response"].get("revision")) is int
                and previous["response"] == {"cycle": cycle_id, "revision": expected_revision + 1, "status": "decided"}
                and current["revision"] == expected_revision + 1, "recorded command cycle/revision moved")
        recorded = snapshot["history"][previous["response"]["revision"]]
        require(recorded["status"] == "decided" and all(recorded["body"].get(k) == v for k, v in payload["body"].items()), "recorded command/checkpoint mismatch")
        verify_journal_objects(journal_root, snapshot, references)
        mode, response = "recorded_idempotent", previous["response"]
    elif current["status"] == "decided":
        require(current["revision"] == expected_revision and current["body"]["recommendation"] == "retain"
                and current["body"]["outcome"] == "success" and current["body"]["references"] == references,
                "decided journal differs from exact reviewed decision")
        verify_journal_objects(journal_root, snapshot, references)
        mode, response = "already_recorded", dict(cycle=cycle_id, revision=current["revision"], status="decided")
    else:
        require(current["revision"] == expected_revision and current["status"] in ("evaluating", "reviewing"), "stale revision or unsupported predecision phase")
        # Stage every metadata byte and the original request/report BEFORE opening
        # the journal writer. Re-inspect after staging to close source drift gaps.
        with tempfile.TemporaryDirectory() as temporary:
            staged = []
            pins = [report["request"], comparison_receipt, *report["verified_metadata"]]
            for index, pin in enumerate(pins):
                path = Path(temporary) / str(index)
                path.write_bytes(execution.checked(pin))
                staged.append((path, pin["sha256"]))
            fresh, _ = inspect(report["request"])
            require(canonical(fresh) == canonical(report), "comparison sources drifted before journal mutation")
            ledger = Ledger(journal_root)
            try:
                for path, sha in staged:
                    ledger.put_artifact(command_id + ":artifact:" + sha, path)
                require(canonical(ledger.export(cycle_id)["binding"]) == canonical(report["cycle_binding"]), "journal binding changed")
                response = ledger.checkpoint(command_id, cycle_id, expected_revision, "decided", note, references,
                                             outcome="success", recommendation="retain")
            finally:
                ledger.close()
        mode = "recorded"
    result = dict(schema="ReviewedMomentumJournalReconciliationV1", status=mode, comparison=comparison_receipt,
                  request=report["request"], journal_result=response, recommendation="retain", journal_read_only=mode != "recorded",
                  scientific_model_promotion=False, publication_authorized=False)
    execution.publish(output / "receipt.json", result)
    return execution.pin(output / "receipt.json")


def verify_journal_objects(root, snapshot, references):
    for digest in references.values():
        require(digest in snapshot["artifact_sizes"], "journal reference artifact registration missing")
        execution.verify_pin(dict(path=str(Path(root).resolve() / "objects" / digest), sha256=digest,
                                  size=snapshot["artifact_sizes"][digest]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    validator = sub.add_parser("validate")
    validator.add_argument("--request", required=True, type=Path)
    validator.add_argument("--sha256", required=True)
    writer = sub.add_parser("reconcile")
    writer.add_argument("--comparison-receipt", required=True, type=Path)
    writer.add_argument("--request", required=True, type=Path)
    writer.add_argument("--request-sha256", required=True)
    writer.add_argument("--sha256", required=True)
    writer.add_argument("--journal-root", required=True, type=Path)
    writer.add_argument("--cycle", required=True)
    writer.add_argument("--expected-revision", required=True, type=int)
    writer.add_argument("--command-id", required=True)
    for child in (validator, writer):
        child.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.operation == "validate":
        result = validate(request_pin(args.request, args.sha256), args.output)
    else:
        result = reconcile(request_pin(args.comparison_receipt, args.sha256), request_pin(args.request, args.request_sha256), args.journal_root, args.cycle,
                           args.expected_revision, args.command_id, args.output)
    print(canonical(result))


if __name__ == "__main__":
    main()
