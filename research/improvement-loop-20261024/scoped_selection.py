"""Reviewed count research selections, separate from journal recommendations/publication.

Native readiness is re-opened through the fixed native adapter. Scientific review
authority is independently pinned; this module invents no score threshold.
"""

import argparse
import hashlib
import math
import os
import sqlite3
import tempfile
from pathlib import Path

import execution
import native_selection
from ledger import canonical
from run_receipts import decode

OPS = {"bootstrap": ("retain", "supports_numerical_research_baseline"),
       "promote": ("promote", "supports_matched_comparison_transfer"),
       "retain": ("retain", "supports_existing_selection"),
       "rollback": ("revise", "supports_reviewed_reversal")}
TARGET = "population_cut_v1"
OUTPUT = "whole_parent_joint_candidate_counts_v1"


def require(value, message):
    execution.require(value, message)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read_pin(pin):
    require(isinstance(pin, dict) and set(pin) in ({"path", "sha256"}, {"path", "sha256", "size"}), "exact file pin required")
    path = Path(pin["path"])
    require(path.is_absolute() and path.resolve() == path and not path.is_symlink() and path.is_file(), "canonical regular evidence file required")
    require(path.stat().st_size <= 32 * 1024 * 1024, "compact evidence manifest required; raw banks stay in native storage")
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == pin["sha256"] and ("size" not in pin or len(data) == pin["size"]), "evidence fingerprint mismatch")
    return data


def document(pin):
    value = decode(read_pin(pin))
    require(isinstance(value, dict), "evidence must be an object")
    return value


def task_scope(native):
    physical = native["taskScope"]
    require(native["taskScopeSha256"] == digest(physical) and physical["targetId"] == TARGET
            and physical["outputContractId"] == OUTPUT and native["request"]["targetId"] == TARGET
            and native["request"]["outputContractId"] == OUTPUT, "native physical prediction task mismatch")
    require(set(physical) == {"targetId", "outputContractId", "predictionEvents", "predictionContests"}, "unsupported physical task fields")
    outputs = native["configuration"].get("outputs")
    require(isinstance(outputs, list), "native output contracts missing")
    matching = [row for row in outputs if isinstance(row, dict) and row.get("contract_id") == OUTPUT]
    require(len(matching) == 1, "unique matching native output contract required")
    output = matching[0]
    require(output.get("unit") == "integer_counts"
            and output.get("denominator") == "declared_event_participants_and_contest_category_universes"
            and output.get("estimand") == "joint candidate/category counts and shared event participants",
            "unsupported native count output semantics")
    return dict(predictionTask=physical, informationCutoff=native["request"]["informationCutoff"],
                outputUnits=output["unit"], denominator=output["denominator"], outputEstimand=output["estimand"])


def preparation(spec_pin):
    spec = document(spec_pin)
    candidate = native_selection.inspect_count_candidate(spec)
    require(isinstance(candidate, native_selection.VerifiedNativeCandidate), "trusted native adapter capability required")
    native = candidate.receipt()
    require(candidate.receipt_sha256 == digest(native) and native["schema"] == "VerifiedNativeCountCandidateV1", "native receipt identity mismatch")
    scope = task_scope(native)
    result = native["result"]
    ready = candidate.status == "ready" and result is not None
    if ready:
        require(result["record"]["numerical_status"] == "passed" and result["predictionRef"] is not None
                and result["terminalReceipt"] is not None and result["commonDrawIdentity"], "complete native result required")
    reference = dict(schema="NativeCountResearchReferenceV1", nativeSpec=spec_pin, nativeReceipt=native) if ready else None
    return dict(schema="ScopedCountSelectionPreparationV1", status="review_required" if ready else "blocked_requires_computation",
                taskScope=scope, taskKey=digest(scope), nativeSpec=spec_pin, nativeReceipt=native,
                candidateRef=reference, candidateRefSha256=digest(reference) if reference else None,
                nativeNumericalStatus="passed" if ready else "no_complete_result", empiricalStatus="unassessed",
                selectionEstablished=False, publicationAuthorized=False)


def prepare(spec_pin, output):
    require(not Path(output).exists(), "fresh preparation output required")
    report = preparation(spec_pin)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    execution.publish(output / "receipt.json", report)
    return execution.pin(output / "receipt.json")


def _connect(root, readonly=False):
    path = Path(root).resolve() / "selection.sqlite3"
    require(path.is_file(), "selection registry does not exist")
    connection = sqlite3.connect(path.as_uri() + ("?mode=ro" if readonly else "?mode=rw"), uri=True, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout=5000")
    return connection


def initialize(root, domain):
    require(domain in ("research", "synthetic"), "explicit research/synthetic domain required")
    root = Path(root).resolve()
    execution.ensure_directory(root / "objects")
    config = dict(schema="ScopedCountSelectionRegistryV1", domain=domain,
                  selectionSource=execution.pin(Path(__file__).resolve()), nativeAdapter=execution.pin(Path(native_selection.__file__).resolve()))
    db = sqlite3.connect(root / "selection.sqlite3", isolation_level=None)
    try:
        db.execute("PRAGMA synchronous=FULL")
        db.executescript("""
        CREATE TABLE IF NOT EXISTS configuration(id INTEGER PRIMARY KEY CHECK(id=1), body TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS objects(hash TEXT PRIMARY KEY, size INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS events(scope TEXT NOT NULL, revision INTEGER NOT NULL, hash TEXT UNIQUE NOT NULL,
          body TEXT NOT NULL, PRIMARY KEY(scope,revision));
        CREATE TABLE IF NOT EXISTS commands(id TEXT PRIMARY KEY, payload TEXT NOT NULL, response TEXT NOT NULL);
        """)
        for table, keys in {"configuration": ("id",), "objects": ("hash",), "events": ("scope", "revision"), "commands": ("id",)}.items():
            for operation in ("UPDATE", "DELETE"):
                db.execute(f"CREATE TRIGGER IF NOT EXISTS immutable_{table}_{operation} BEFORE {operation} ON {table} BEGIN SELECT RAISE(ABORT,'append-only'); END")
            predicate = " AND ".join(f"{key}=NEW.{key}" for key in keys)
            db.execute(f"CREATE TRIGGER IF NOT EXISTS immutable_{table}_INSERT BEFORE INSERT ON {table} WHEN EXISTS(SELECT 1 FROM {table} WHERE {predicate}) BEGIN SELECT RAISE(ABORT,'append-only duplicate'); END")
        db.execute("BEGIN IMMEDIATE")
        old = db.execute("SELECT body FROM configuration WHERE id=1").fetchone()
        if old:
            require(old[0] == canonical(config), "registry domain/source identity differs")
        else:
            db.execute("INSERT INTO configuration VALUES(1,?)", (canonical(config),))
        db.execute("COMMIT")
    finally:
        db.close()
    execution.sync_directory(root)
    return config


def _config(db):
    config = decode(db.execute("SELECT body FROM configuration WHERE id=1").fetchone()[0])
    require(config["schema"] == "ScopedCountSelectionRegistryV1" and config["domain"] in ("research", "synthetic"), "unsupported registry")
    execution.verify_pin(config["selectionSource"])
    execution.verify_pin(config["nativeAdapter"])
    return config


def _object(root, db, sha):
    row = db.execute("SELECT size FROM objects WHERE hash=?", (sha,)).fetchone()
    require(row is not None, "registry object not registered")
    return execution.checked(dict(path=str(Path(root).resolve() / "objects" / sha), sha256=sha, size=row[0]))


def _event(root, db, row):
    if row is None:
        return None
    body = decode(row["body"])
    require(digest(body) == row["hash"] and body["taskKey"] == row["scope"] == digest(body["taskScope"])
            and body["revision"] == row["revision"] and digest(body["candidateRef"]) == body["candidateRefSha256"], "registry event corruption")
    for pin in body["retainedEvidence"]:
        _object(root, db, pin["sha256"])
    return dict(hash=row["hash"], **body)


def _head(root, db, key):
    return _event(root, db, db.execute("SELECT * FROM events WHERE scope=? ORDER BY revision DESC LIMIT 1", (key,)).fetchone())


def predecessor(event):
    return None if event is None else dict(eventSha256=event["hash"], candidateRefSha256=event["candidateRefSha256"])


def _stage(root, evidence):
    objects = Path(root).resolve() / "objects"
    rows = {}
    for data in evidence:
        sha = hashlib.sha256(data).hexdigest()
        target = objects / sha
        if target.exists():
            require(not target.is_symlink() and target.read_bytes() == data, "registry object corruption")
        else:
            name = None
            try:
                with tempfile.NamedTemporaryFile(dir=objects, delete=False) as stream:
                    name = stream.name
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                try:
                    os.link(name, target)
                    target.chmod(0o444)
                except FileExistsError:
                    require(target.read_bytes() == data and not target.is_symlink(), "registry object race/corruption")
            finally:
                if name:
                    os.unlink(name)
        rows[sha] = len(data)
    execution.sync_directory(objects)
    return rows


def _transfer(review, report, previous, evidence):
    mapping = review["comparisonTransfer"]
    require(isinstance(mapping, dict) and set(mapping) == {"targetCandidateRefSha256", "targetIncumbentRefSha256", "candidateDefinition", "incumbentDefinition", "evaluations", "transferReview"}, "explicit comparison/transfer mapping required")
    require(mapping["targetCandidateRefSha256"] == report["candidateRefSha256"] and previous
            and mapping["targetIncumbentRefSha256"] == previous["candidateRefSha256"], "target comparison identity mismatch")
    def typed(pin, schema):
        raw = read_pin(pin)
        value = decode(raw)
        require(isinstance(value, dict) and value.get("schema") == schema, "unsupported typed comparison evidence: " + schema)
        evidence.append(raw)
        return value
    for key, native in (("candidateDefinition", report["nativeReceipt"]), ("incumbentDefinition", previous["candidateRef"]["nativeReceipt"])):
        definition = typed(mapping[key], "ScopedCountModelDefinitionV1")
        require(definition["providerSelections"] == native["request"]["selections"] and isinstance(definition["sourceEvidence"], list)
                and definition["sourceEvidence"], "target model definition mapping mismatch")
        evidence.extend(read_pin(pin) for pin in definition["sourceEvidence"])
    transfer = typed(mapping["transferReview"], "ScopedCountTargetTransferReviewV1")
    require(transfer["taskKey"] == report["taskKey"] and transfer["candidateRefSha256"] == report["candidateRefSha256"]
            and transfer["incumbentRefSha256"] == previous["candidateRefSha256"]
            and transfer["candidateDefinition"] == mapping["candidateDefinition"] and transfer["incumbentDefinition"] == mapping["incumbentDefinition"]
            and transfer["disposition"] == "supports_scoped_target_change", "reviewed target transfer identity mismatch")
    target_inputs = transfer["targetInputRelationship"]
    require(target_inputs["candidateObservationSupportSha256"] == report["nativeReceipt"]["observationSupportSha256"]
            and target_inputs["incumbentObservationSupportSha256"] == previous["candidateRef"]["nativeReceipt"]["observationSupportSha256"], "target observation support mapping mismatch")
    _information_relationship(target_inputs)
    rows = mapping["evaluations"]
    require(isinstance(rows, list) and {(row["tier"], row["cycle"]) for row in rows} == {("A", 2018), ("A", 2022), ("B", 2020), ("B", 2024)} and len(rows) == 4,
            "complete matched A2018/22 B2020/24 comparison required")
    require(transfer["evaluationMappings"] == [{key: row[key] for key in ("tier", "cycle", "candidateDefinition", "incumbentDefinition", "scores", "inputBudgetReview")} for row in rows], "historical-to-target model transfer mapping mismatch")
    summaries = []
    for row in rows:
        require(set(row) == {"tier", "cycle", "evaluationScope", "incumbentDefinition", "candidateDefinition", "incumbentConfiguration", "candidateConfiguration", "incumbentResult", "candidateResult", "comparisonPlan", "scores", "numericalReview", "inputBudgetReview"}, "unsupported comparison row")
        scope = row["evaluationScope"]
        require(isinstance(scope, dict) and set(scope) == {"targetId", "outputContractId", "cycle", "office", "geography", "raceIds", "informationCutoff", "horizonDays", "informationBudget"}
                and scope["targetId"] == TARGET and scope["outputContractId"] == OUTPUT
                and scope["cycle"] == row["cycle"] and isinstance(scope["raceIds"], list) and scope["raceIds"]
                and len(set(scope["raceIds"])) == len(scope["raceIds"]) and type(scope["horizonDays"]) is int and scope["horizonDays"] > 0
                and all(isinstance(scope[key], str) and scope[key].strip() for key in ("office", "geography", "informationCutoff", "informationBudget")), "matched historical scope must be explicit")
        native_results = {}
        for side in ("incumbent", "candidate"):
            typed(row[side + "Definition"], "ScopedCountModelDefinitionV1")
            cfg = typed(row[side + "Configuration"], "ScopedCountEvaluationConfigurationV1")
            result = typed(row[side + "Result"], "ScopedCountEvaluationResultV1")
            native_results[side] = result
            require(cfg["scope"] == scope and cfg["modelDefinition"] == row[side + "Definition"]
                    and result["scope"] == scope and result["configuration"] == row[side + "Configuration"], "historical native configuration/result mapping mismatch")
            record = document(result["nativeRecord"])
            require(record["schema_version"] == "composition-result-v1" and record["configuration_sha256"] == cfg["nativeConfigurationSha256"]
                    and record["numerical_status"] == "passed", "historical native result-record/configuration mismatch")
            evidence.append(read_pin(result["nativeRecord"]))
            terminal = typed(result["terminalReceipt"], "ScopedCountEvaluationTerminalV1")
            require(set(terminal) == {"schema", "scope", "nativeConfigurationSha256", "nativeRecord", "outputIdentity", "status"}
                    and terminal["scope"] == scope and terminal["nativeConfigurationSha256"] == cfg["nativeConfigurationSha256"]
                    and terminal["nativeRecord"] == result["nativeRecord"] and terminal["status"] == "complete",
                    "historical terminal/configuration/record mismatch")
            output = typed(terminal["outputIdentity"], "ScopedCountEvaluationOutputIdentityV1")
            require(set(output) == {"schema", "scope", "nativeConfigurationSha256", "nativeRecord", "predictionArtifact", "parentIds", "commonDrawIdentity"}
                    and output["scope"] == scope and output["nativeConfigurationSha256"] == cfg["nativeConfigurationSha256"]
                    and output["nativeRecord"] == result["nativeRecord"]
                    and isinstance(output["predictionArtifact"], dict) and set(output["predictionArtifact"]) == {"kind", "artifact_id"}
                    and all(isinstance(output["predictionArtifact"][key], str) and output["predictionArtifact"][key].strip() for key in ("kind", "artifact_id"))
                    and isinstance(output["parentIds"], list) and output["parentIds"] and len(set(output["parentIds"])) == len(output["parentIds"])
                    and all(isinstance(parent, str) and parent.strip() for parent in output["parentIds"])
                    and isinstance(output["commonDrawIdentity"], str) and output["commonDrawIdentity"].strip(),
                    "historical output identity/configuration/record mismatch")
        plan = typed(row["comparisonPlan"], "ScopedCountMatchedEvaluationPlanV1")
        score = typed(row["scores"], "ScopedCountMatchedEvaluationScoresV1")
        numerical = typed(row["numericalReview"], "ScopedCountNumericalReviewV1")
        budget = typed(row["inputBudgetReview"], "ScopedCountInformationMatchReviewV1")
        require(plan["scope"] == score["scope"] == numerical["scope"] == budget["scope"] == scope
                and score["plan"] == budget["plan"] == row["comparisonPlan"], "matched plan/score/review scope mismatch")
        for side in ("incumbent", "candidate"):
            require(plan[side + "Configuration"] == row[side + "Configuration"] and plan[side + "Definition"] == row[side + "Definition"]
                    and score[side + "Result"] == numerical[side + "Result"] == budget[side + "Result"] == row[side + "Result"], "comparison result/definition cross-binding mismatch")
            require(budget[side + "ObservationSupportSha256"] == native_results[side]["observationSupportSha256"], "historical observation support mapping mismatch")
        require(set(budget) == {"schema", "scope", "plan", "incumbentResult", "candidateResult", "disposition", "candidateObservationSupportSha256", "incumbentObservationSupportSha256", "inputRelationship", "reason"}, "unsupported information review fields")
        _information_relationship({key: budget[key] for key in ("candidateObservationSupportSha256", "incumbentObservationSupportSha256", "inputRelationship", "reason")})
        require(isinstance(plan["metricIds"], list) and plan["metricIds"] and all(isinstance(metric, str) and metric.strip() for metric in plan["metricIds"])
                and len(set(plan["metricIds"])) == len(plan["metricIds"]) and set(score["metricScores"]) == set(plan["metricIds"])
                and numerical["disposition"] == "accepted_declared_gates" and budget["disposition"] == "matched_inputs_or_explicit_data_arm",
                "declared metric/numerical/information review mismatch")
        require(plan["primaryMetricId"] in plan["metricIds"], "primary metric must be frozen in comparison plan")
        for metric, cell in score["metricScores"].items():
            _score_cell(cell)
        require(score["metricScores"][plan["primaryMetricId"]]["status"] == "available", "required primary comparison is unavailable")
        limits = plan["limitations"]
        require(isinstance(limits, dict) and set(limits) == {"benchmarkExposure", "asOfStatus", "reasons"}
                and limits["benchmarkExposure"] == "retrospective_development" and limits["asOfStatus"] in ("reviewed_asof_replay", "unassessed")
                and isinstance(limits["reasons"], list) and limits["reasons"] and all(isinstance(s, str) and s.strip() for s in limits["reasons"]),
                "retrospective/as-of limits must stay explicit")
        summaries.append(dict(tier=row["tier"], cycle=row["cycle"], scope=scope, primaryMetricId=plan["primaryMetricId"],
                              primaryCell=score["metricScores"][plan["primaryMetricId"]], metricCells=score["metricScores"], limitations=limits))
    return dict(transferReview=mapping["transferReview"], targetInputRelationship=target_inputs, evaluations=summaries)


def _information_relationship(value):
    require(isinstance(value, dict) and set(value) == {"candidateObservationSupportSha256", "incumbentObservationSupportSha256", "inputRelationship", "reason"}, "unsupported input relationship fields")
    require(value["inputRelationship"] in ("matched", "declared_data_arm") and isinstance(value["reason"], str) and value["reason"].strip(), "explicit matched-data/input-budget judgment required")
    for key in ("candidateObservationSupportSha256", "incumbentObservationSupportSha256"):
        require(isinstance(value[key], str) and len(value[key]) == 64, "observation support fingerprint required")
    if value["inputRelationship"] == "matched":
        require(value["candidateObservationSupportSha256"] == value["incumbentObservationSupportSha256"], "different observation supports require an explicit data-arm review")


def _score_cell(cell):
    require(isinstance(cell, dict) and set(cell) == {"status", "supportCount", "supportUnit", "incumbent", "candidate", "pairedDifference", "uncertainty", "reason"}, "typed comparison score cell required")
    require(cell["status"] in ("available", "unavailable") and type(cell["supportCount"]) is int and cell["supportCount"] >= 0
            and isinstance(cell["supportUnit"], str) and cell["supportUnit"].strip() and isinstance(cell["reason"], str) and cell["reason"].strip(), "score support/status/reason missing")
    uncertainty = cell["uncertainty"]
    require(isinstance(uncertainty, dict) and set(uncertainty) == {"status", "standardError", "reason"}
            and uncertainty["status"] in ("estimated", "unresolved") and isinstance(uncertainty["reason"], str) and uncertainty["reason"].strip(), "score uncertainty status/reason missing")
    finite = lambda x: type(x) in (int, float) and math.isfinite(x)
    require((uncertainty["status"] == "unresolved" and uncertainty["standardError"] is None)
            or (uncertainty["status"] == "estimated" and finite(uncertainty["standardError"]) and uncertainty["standardError"] >= 0), "invalid score uncertainty")
    if cell["status"] == "available":
        require(cell["supportCount"] > 0 and all(finite(cell[k]) for k in ("incumbent", "candidate", "pairedDifference"))
                and math.isclose(cell["pairedDifference"], cell["candidate"]-cell["incumbent"], rel_tol=1e-12, abs_tol=1e-12), "available comparison needs finite paired estimates")
    else:
        require(cell["supportCount"] == 0 and all(cell[k] is None for k in ("incumbent", "candidate", "pairedDifference"))
                and uncertainty["status"] == "unresolved", "unavailable comparison must remain explicit and valueless")


def _authorize(decision_pin, report, previous, domain):
    decision = document(decision_pin)
    require(set(decision) == {"schema", "domain", "operation", "taskKey", "candidateRefSha256", "expectedRevision", "expectedPredecessor", "journal", "review", "empiricalStatus", "reason", "restoreEventSha256", "primaryLayer", "changedRoles", "interactionScope"}
            and decision["schema"] == "ScopedCountSelectionDecisionV1", "unsupported selection decision")
    operation = decision["operation"]
    require(operation in OPS and decision["domain"] == domain and decision["taskKey"] == report["taskKey"]
            and decision["candidateRefSha256"] == report["candidateRefSha256"] and decision["expectedPredecessor"] == predecessor(previous), "selection decision target/predecessor mismatch")
    require(isinstance(decision["reason"], str) and decision["reason"].strip(), "explicit reviewed reason required")
    review = document(decision["review"])
    require(set(review) == {"schema", "domain", "operation", "taskKey", "candidateRefSha256", "predecessor", "disposition", "nativeCompatibility", "evidence", "comparisonTransfer", "primaryLayer", "changedRoles", "interactionScope"}
            and review["schema"] == "ScopedCountSelectionReviewV1" and review["domain"] == domain
            and review["operation"] == operation and review["taskKey"] == report["taskKey"] and review["candidateRefSha256"] == report["candidateRefSha256"]
            and review["predecessor"] == predecessor(previous) and review["disposition"] == OPS[operation][1], "independent selection review mismatch")
    require(all(review[key] == decision[key] for key in ("primaryLayer", "changedRoles", "interactionScope"))
            and isinstance(decision["primaryLayer"], str) and decision["primaryLayer"].strip()
            and isinstance(decision["changedRoles"], list) and len(set(decision["changedRoles"])) == len(decision["changedRoles"])
            and set(decision["changedRoles"]) <= set(native_selection.ROLES), "reviewed primary layer/changed roles mismatch")
    native = report["nativeReceipt"]
    require(review["nativeCompatibility"] == {"candidateProblemIdentity": native["scope"]["problemIdentity"],
            "candidateSupportId": native["configuration"]["supportId"], "incumbentProblemIdentity": previous["candidateRef"]["nativeReceipt"]["scope"]["problemIdentity"] if previous else None,
            "taskScopeSha256": report["taskKey"]}, "explicit native encoding/task compatibility review required")
    require(isinstance(review["evidence"], list) and review["evidence"], "source-bound review evidence required")
    evidence = [read_pin(decision_pin), read_pin(decision["review"]), *[read_pin(pin) for pin in review["evidence"]]]
    evaluation_basis = previous.get("evaluationBasis") if previous else None
    if operation == "bootstrap":
        require(previous is None and decision["empiricalStatus"] == "unassessed" and review["comparisonTransfer"] is None
                and decision["restoreEventSha256"] is None, "baseline needs explicit absent predecessor/unassessed authority")
    elif operation == "promote":
        require(previous is not None and previous["candidateRefSha256"] != report["candidateRefSha256"]
                and decision["empiricalStatus"] == "reviewed_comparative_support" and decision["restoreEventSha256"] is None
                and decision["changedRoles"], "promotion basis is unsupported")
        evaluation_basis = _transfer(review, report, previous, evidence)
    elif operation == "retain":
        require(previous is not None and previous["candidateRefSha256"] == report["candidateRefSha256"]
                and decision["empiricalStatus"] == previous["empiricalStatus"] and decision["restoreEventSha256"] is None
                and review["comparisonTransfer"] is None, "retain must preserve the existing complete selection and evidence status")
    else:
        require(previous is not None and isinstance(decision["restoreEventSha256"], str) and review["comparisonTransfer"] is None
                and decision["empiricalStatus"] in ("unassessed", "reviewed_comparative_support"), "explicit reviewed reversal required")
    require(decision["journal"]["binding"]["scope"] == {"taskScope": report["taskScope"], "domain": domain}
            and decision["journal"]["binding"]["incumbent_config"] == (previous["candidateRef"] if previous else report["candidateRef"])
            and decision["journal"]["binding"]["target_layer"] == decision["primaryLayer"]
            and isinstance(decision["journal"]["binding"]["hypothesis"], str) and decision["journal"]["binding"]["hypothesis"].strip(), "journal full selection scope/incumbent mismatch")
    return decision, evidence, evaluation_basis


def _journal(db, decision, decision_pin):
    journal = decision["journal"]
    require(set(journal) == {"root", "cycle", "revision", "binding"}, "exact journal expectation required")
    require(Path(journal["root"]).is_absolute() and (Path(journal["root"]) / "ledger.sqlite3").is_file(), "journal missing")
    require(Path(journal["root"]).resolve() != Path(db.execute("PRAGMA database_list").fetchone()["file"]).parent, "selection objects require a separate registry directory")
    db.execute("ATTACH DATABASE ? AS journal", ((Path(journal["root"]) / "ledger.sqlite3").as_uri() + "?mode=ro",))
    require(db.execute("PRAGMA journal.journal_mode").fetchone()[0] in ("delete", "truncate", "persist"), "WAL/nonrollback journal unsupported for atomic selection")
    return journal


def _journal_check(db, journal, decision, decision_pin, replay=False):
    cycle = db.execute("SELECT binding FROM journal.cycles WHERE id=?", (journal["cycle"],)).fetchone()
    row = db.execute("SELECT revision,status,body FROM journal.checkpoints WHERE cycle=? AND revision=?" if replay else
                     "SELECT revision,status,body FROM journal.checkpoints WHERE cycle=? ORDER BY revision DESC LIMIT 1",
                     (journal["cycle"], journal["revision"]) if replay else (journal["cycle"],)).fetchone()
    require(db.execute("PRAGMA journal.journal_mode").fetchone()[0] in ("delete", "truncate", "persist"), "WAL/nonrollback journal unsupported for atomic selection")
    require(cycle is not None and row is not None and canonical(decode(cycle[0])) == canonical(journal["binding"]), "journal cycle/binding mismatch")
    body = decode(row["body"])
    refs = {"evaluation": decision_pin["sha256"], "review": decision["review"]["sha256"]}
    require(row["revision"] == journal["revision"] and row["status"] == "decided" and body["outcome"] == "success"
            and body["recommendation"] == OPS[decision["operation"]][0] and body["references"] == refs, "journal does not contain exact current reviewed decision")
    for sha in refs.values():
        registered = db.execute("SELECT size FROM journal.artifacts WHERE hash=?", (sha,)).fetchone()
        require(registered is not None, "journal decision artifact not registered")
        execution.verify_pin(dict(path=str(Path(journal["root"]) / "objects" / sha), sha256=sha, size=registered[0]))


def apply(root, preparation_pin, expected_spec_pin, decision_pin, expected_revision, expected_predecessor, command_id, output):
    require(not Path(output).exists(), "fresh selection output required")
    require(type(expected_revision) is int and expected_revision >= -1 and isinstance(command_id, str) and command_id.strip(), "CAS revision/command required")
    supplied = document(preparation_pin)
    require(supplied["nativeSpec"] == expected_spec_pin, "independent native spec anchor mismatch")
    report = preparation(expected_spec_pin)
    require(canonical(supplied) == canonical(report), "preparation differs from fresh trusted native inspection")
    require(report["status"] == "review_required" and report["candidateRef"] is not None, "no complete native count result; selection blocked")
    root = Path(root).resolve()
    db = _connect(root)
    try:
        config = _config(db)
        native = report["nativeReceipt"]
        require((config["domain"] == "research" and native["scope"]["kind"] == "actual_research" and native["result"]["record"]["evidence_status"] == "unassessed")
                or (config["domain"] == "synthetic" and native["scope"]["kind"] == "synthetic_lifecycle" and native["result"]["record"]["evidence_status"] == "synthetic"), "synthetic/actual selection domains must remain separate")
        prior_command = db.execute("SELECT payload,response FROM commands WHERE id=?", (command_id,)).fetchone()
        current = _head(root, db, report["taskKey"])
        if not prior_command:
            require((current["revision"] if current else -1) == expected_revision and predecessor(current) == expected_predecessor,
                    "stale scoped selection predecessor/revision")
        # An exact command replay validates the original predecessor instead of
        # treating today's head as permission to repeat an old state change.
        prior = current
        if prior_command:
            response = decode(prior_command["response"])
            require(isinstance(response, dict) and type(response.get("revision")) is int and response["revision"] >= 0, "malformed recorded command response")
            event = _event(root, db, db.execute("SELECT * FROM events WHERE scope=? AND revision=?", (report["taskKey"], response["revision"])).fetchone())
            require(event is not None, "recorded selection command has no event")
            require(response == {"taskKey": event["taskKey"], "revision": event["revision"], "eventSha256": event["hash"], "candidateRefSha256": event["candidateRefSha256"]}, "recorded command response/event mismatch")
            prior = _event(root, db, db.execute("SELECT * FROM events WHERE scope=? AND revision=?", (report["taskKey"], event["revision"] - 1)).fetchone())
        decision, evidence, evaluation_basis = _authorize(decision_pin, report, prior, config["domain"])
        require(decision["expectedRevision"] == expected_revision and decision["expectedPredecessor"] == expected_predecessor, "external CAS differs from authorization")
        payload = dict(preparation=preparation_pin, nativeSpec=expected_spec_pin, decision=decision_pin,
                       taskKey=report["taskKey"], expectedRevision=expected_revision, expectedPredecessor=expected_predecessor)
        output = Path(output).resolve()
        output.mkdir(parents=True, exist_ok=False)
        evidence.extend([read_pin(expected_spec_pin), read_pin(preparation_pin), canonical(report["candidateRef"]).encode()])
        retained = _stage(root, evidence)
        require(canonical(preparation(expected_spec_pin)) == canonical(report), "native result changed before transaction")
        journal = _journal(db, decision, decision_pin)
        db.execute("BEGIN IMMEDIATE")
        try:
            previous = db.execute("SELECT payload,response FROM commands WHERE id=?", (command_id,)).fetchone()
            _journal_check(db, journal, decision, decision_pin, replay=previous is not None)
            head = _head(root, db, report["taskKey"])
            if previous:
                require(previous["payload"] == canonical(payload), "selection command ID conflicts with payload")
                response = decode(previous["response"])
                mode = "already_applied"
            else:
                require((head["revision"] if head else -1) == expected_revision and predecessor(head) == expected_predecessor,
                        "stale scoped selection predecessor/revision")
                require(predecessor(head) == predecessor(prior), "selection head changed during verification")
                if decision["operation"] == "rollback":
                    old = _event(root, db, db.execute("SELECT * FROM events WHERE scope=? AND hash=?", (report["taskKey"], decision["restoreEventSha256"])).fetchone())
                    require(old is not None and old["revision"] < head["revision"] and old["candidateRefSha256"] == report["candidateRefSha256"], "reversal must select an exact retained historical result")
                    require(decision["empiricalStatus"] == old["empiricalStatus"], "reversal cannot relabel the historical evidence status")
                    evaluation_basis = old.get("evaluationBasis")
                for sha, size in retained.items():
                    old = db.execute("SELECT size FROM objects WHERE hash=?", (sha,)).fetchone()
                    if old is None:
                        db.execute("INSERT INTO objects VALUES(?,?)", (sha, size))
                    else:
                        require(old[0] == size, "registry object registration corruption")
                body = dict(schema="ScopedCountSelectionEventV1", domain=config["domain"], taskKey=report["taskKey"], taskScope=report["taskScope"],
                            revision=expected_revision + 1, predecessor=expected_predecessor, operation=decision["operation"],
                            candidateRef=report["candidateRef"], candidateRefSha256=report["candidateRefSha256"],
                            basis=decision["operation"], empiricalStatus=decision["empiricalStatus"], rootDecision=decision_pin,
                            independentReview=decision["review"], journal=dict(cycle=journal["cycle"], revision=journal["revision"], binding=journal["binding"]),
                            reason=decision["reason"], restoreEventSha256=decision["restoreEventSha256"],
                            primaryLayer=decision["primaryLayer"], changedRoles=decision["changedRoles"], interactionScope=decision["interactionScope"],
                            evaluationBasis=evaluation_basis,
                            retainedEvidence=[dict(sha256=sha, size=size) for sha, size in sorted(retained.items())], publicationAuthorized=False)
                event_sha = digest(body)
                db.execute("INSERT INTO events VALUES(?,?,?,?)", (report["taskKey"], expected_revision + 1, event_sha, canonical(body)))
                response = dict(taskKey=report["taskKey"], revision=expected_revision + 1, eventSha256=event_sha, candidateRefSha256=report["candidateRefSha256"])
                db.execute("INSERT INTO commands VALUES(?,?,?)", (command_id, canonical(payload), canonical(response)))
                mode = "retained" if decision["operation"] == "retain" else "selected"
            db.execute("COMMIT")
        except BaseException:
            db.execute("ROLLBACK")
            raise
        current_head = _head(root, db, report["taskKey"])
        result = dict(schema="ScopedCountSelectionApplicationV1", status=mode, commandEvent=response,
                      currentHead=dict(revision=current_head["revision"], eventSha256=current_head["hash"], candidateRefSha256=current_head["candidateRefSha256"]),
                      domain=config["domain"], publicationAuthorized=False)
        execution.publish(output / "receipt.json", result)
        return execution.pin(output / "receipt.json")
    finally:
        db.close()


def resolve_selected(root, scope):
    root = Path(root).resolve()
    db = _connect(root, readonly=True)
    try:
        config = _config(db)
        event = _head(root, db, digest(scope))
        require(event is not None and event["taskScope"] == scope, "no selected reference for exact prediction task")
        ref = event["candidateRef"]
        spec = decode(_object(root, db, ref["nativeSpec"]["sha256"]))
        candidate = native_selection.inspect_count_candidate(spec)
        require(isinstance(candidate, native_selection.VerifiedNativeCandidate) and candidate.status == "ready"
                and canonical(candidate.receipt()) == canonical(ref["nativeReceipt"]), "selected native request/result no longer validates")
        return dict(schema="ResolvedScopedCountSelectionV1", domain=config["domain"], taskKey=event["taskKey"], revision=event["revision"],
                    eventSha256=event["hash"], candidateRef=ref, basis=event["basis"], empiricalStatus=event["empiricalStatus"],
                    independentReview=event["independentReview"], evaluationBasis=event["evaluationBasis"], publicationAuthorized=False)
    finally:
        db.close()


def history(root, scope):
    """Audit immutable references; only resolve_selected makes a usable claim."""
    db = _connect(root, readonly=True)
    try:
        config = _config(db)
        events = [_event(root, db, row) for row in db.execute("SELECT * FROM events WHERE scope=? ORDER BY revision", (digest(scope),))]
        return dict(schema="ScopedCountSelectionHistoryV1", domain=config["domain"], taskKey=digest(scope), events=events,
                    nativeResultsReopened=False, publicationAuthorized=False)
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    init = sub.add_parser("initialize")
    init.add_argument("--root", required=True, type=Path)
    init.add_argument("--domain", required=True, choices=("research", "synthetic"))
    prep = sub.add_parser("prepare")
    prep.add_argument("--native-spec", required=True, type=Path)
    prep.add_argument("--native-spec-sha256", required=True)
    prep.add_argument("--output", required=True, type=Path)
    commit = sub.add_parser("apply")
    for name in ("root", "preparation", "native-spec", "decision", "output"):
        commit.add_argument("--" + name, required=True, type=Path)
    for name in ("preparation-sha256", "native-spec-sha256", "decision-sha256", "command-id"):
        commit.add_argument("--" + name, required=True)
    commit.add_argument("--expected-revision", required=True, type=int)
    commit.add_argument("--expected-predecessor", required=True, help="JSON null or exact predecessor object")
    for name in ("resolve", "history"):
        load = sub.add_parser(name)
        load.add_argument("--root", required=True, type=Path)
        load.add_argument("--scope", required=True, type=Path)
    args = parser.parse_args()
    def pinned(path, sha):
        item = execution.pin(path.resolve())
        require(item["sha256"] == sha, "explicit input SHA mismatch")
        return item
    if args.operation == "initialize":
        result = initialize(args.root, args.domain)
    elif args.operation == "prepare":
        result = prepare(pinned(args.native_spec, args.native_spec_sha256), args.output)
    elif args.operation in ("resolve", "history"):
        result = (resolve_selected if args.operation == "resolve" else history)(args.root, decode(args.scope.read_bytes()))
    else:
        result = apply(args.root, pinned(args.preparation, args.preparation_sha256), pinned(args.native_spec, args.native_spec_sha256),
                       pinned(args.decision, args.decision_sha256), args.expected_revision, decode(args.expected_predecessor), args.command_id, args.output)
    print(canonical(result))


if __name__ == "__main__":
    main()
