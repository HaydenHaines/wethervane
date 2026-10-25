"""Read-only adapters for native Senate failed attempts and reviewed historical House cells.

Not a scientific verifier, execution manager, liveness monitor, or qualifier.
Positive count-stack runs and unrecognized receipt versions are unsupported. Hashes bind
bytes, not the truth of scientific claims. No scientific output is executed.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ledger import Ledger, canonical

ADAPTER = "senate_v16_failed_pilot_adapter_v1"
SCOPE_KEYS = (
    "purpose",
    "cutoff_utc",
    "problem_identity",
    "provider_identity",
    "upstream_identity",
    "parent_ids",
    "parent_masses",
    "scientific_target_ids",
    "provider_configuration",
    "settings",
)


class ReceiptError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ReceiptError(message)


def decode(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    return json.loads(
        data,
        object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ReceiptError("nonfinite JSON")),
    )


@dataclass(frozen=True)
class Inspection:
    """Immutable output of inspection; attachment checks the caller's report against it."""

    root: Path
    report_bytes: bytes
    pins: tuple

    @property
    def artifacts(self):
        return {path: {"sha256": digest, "size": size} for path, digest, size in self.pins}


class Bundle:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self._artifacts = {}

    @property
    def artifacts(self):
        return {path: dict(pin) for path, pin in self._artifacts.items()}

    def resolve(self, name):
        path = (self.root / name).resolve()
        require(path.is_relative_to(self.root), "path outside source root")
        return path

    def freeze(self, report):
        pins = tuple((path, pin["sha256"], pin["size"]) for path, pin in sorted(self._artifacts.items()))
        return report, Inspection(self.root, canonical(report).encode(), pins)

    def read(self, pin):
        require(isinstance(pin, dict) and isinstance(pin.get("path"), str), "missing path pin")
        path = self.resolve(pin["path"])
        name = path.relative_to(self.root).as_posix()
        digest = pin.get("sha256")
        require(isinstance(digest, str) and len(digest) == 64, "missing SHA-256 pin")
        previous = self._artifacts.get(name)
        require(previous is None or previous["sha256"] == digest, f"conflicting pin: {name}")
        data = path.read_bytes()
        require(hashlib.sha256(data).hexdigest() == digest, f"hash mismatch: {pin['path']}")
        self._artifacts[name] = {"sha256": digest, "size": len(data)}
        return data

    def document(self, pin):
        value = decode(self.read(pin))
        require(isinstance(value, dict), "receipt must be an object")
        return value

    def nested_pins(self, value):
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                self.read(value)
            for child in value.values():
                self.nested_pins(child)
        elif isinstance(value, list):
            for child in value:
                self.nested_pins(child)


def inspect_receipts(root, checkpoint_pin, expected_plan_sha256, declared_scope, native_attempt=None):
    """Return (normalized report, frozen Inspection); never infer a pass."""
    bundle = Bundle(root)
    checkpoint = None
    if checkpoint_pin is not None:
        checkpoint = bundle.document(checkpoint_pin)
        require(checkpoint.get("schema") == "SenateStackCheckpointV16", "unsupported checkpoint schema")
        terminal = checkpoint["terminal_attempt"]
        require(
            checkpoint.get("status") == "partial_affine_worker_startup_failed_repair_underway",
            "unsupported checkpoint outcome",
        )
        require(
            all(
                terminal.get(key) is False for key in ("complete_cut", "qualified_counts", "full_fit_followup_allowed")
            ),
            "contradictory terminal claims",
        )
        require(
            type(terminal.get("returned_draw_records")) is int and terminal["returned_draw_records"] == 0,
            "unsupported returned draws",
        )
    else:
        require(isinstance(native_attempt, dict), "native attempt pins required")
        terminal = native_attempt
    # The native writer has one fixed layout. Equal error strings are not attempt identity.
    attempt_root = bundle.resolve(terminal["failure"]["path"]).parent
    expected_paths = {
        "failure": attempt_root / "failure.json",
        "fit_failure": attempt_root / "fit/failure.json",
        "execution": attempt_root / "fit/parent-0-execution.json",
        "observer": attempt_root / "fit/parent-0-observer-receipt.json",
        "log": Path(str(attempt_root) + ".log"),
    }
    for key, path in expected_paths.items():
        require(bundle.resolve(terminal[key]["path"]) == path, f"native attempt path mismatch: {key}")
    if checkpoint is not None or "path" in terminal:
        require(bundle.resolve(terminal["path"]) == attempt_root, "terminal attempt path mismatch")
    failure = bundle.document(terminal["failure"])
    if checkpoint is not None or "error" in terminal:
        require(terminal.get("error") == failure.get("error"), "terminal attempt error mismatch")
    require(failure.get("status") == "incomplete_preserved_attempt", "unsupported failure schema")
    require(
        failure.get("forecast_qualified") is False and failure.get("publication_authorized") is False,
        "failure cannot qualify",
    )
    require(failure["plan"]["sha256"] == expected_plan_sha256, "plan binding mismatch")
    plan = bundle.document(failure["plan"])
    require(plan.get("schema") == "actual_TX_role_execution_plan_v2", "unsupported plan schema")
    require(set(declared_scope) == set(SCOPE_KEYS), "declared scope keys mismatch")
    scope = {key: plan[key] for key in SCOPE_KEYS}
    require(canonical(scope) == canonical(declared_scope), "declared scope mismatch")
    require(
        plan.get("forecast_qualified") is False and plan.get("publication_authorized") is False,
        "unsupported positive plan",
    )
    require(
        plan.get("execution_adapter_version") == "share_coordinate_bounded_execution_v1",
        "unsupported execution adapter",
    )
    require(plan["execution_policy"]["mode"] == "diagnostic_first_parent", "unsupported execution mode")
    fit = bundle.document(terminal["fit_failure"])
    require(
        fit.get("status") == "incomplete_preserved_attempt" and fit.get("complete_cut") is False,
        "unsupported fit outcome",
    )
    require(
        fit.get("completed_parents") == [] and fit.get("required_parent_ids") == plan["parent_ids"],
        "parent support mismatch",
    )
    require(
        failure.get("no_automatic_retry") is True and fit.get("no_automatic_retry") is True, "unsupported retry policy"
    )
    require(
        fit["execution_policy"]["version"] == "share_coordinate_bounded_first_parent_execution_v1"
        and fit["execution_policy"]["policy"] == plan["execution_policy"],
        "execution policy mismatch",
    )
    execution = bundle.document(terminal["execution"])
    observer = bundle.document(terminal["observer"])
    require(execution.get("schema") == "conditional_process_observation_v1", "unsupported process schema")
    require(
        observer.get("schema") == "sampling_observer_receipt_v1"
        and observer.get("version") == "returned_draw_and_observer_heartbeat_v1",
        "unsupported observer schema",
    )
    require(
        observer.get("returned_draw_records") == 0 and observer.get("sampling_returned_to_coordinator") is False,
        "observer contradicts failed fit",
    )
    require(observer.get("parent_id") == plan["parent_ids"][0], "observer parent mismatch")
    require(observer.get("observer_liveness_is_not_sampler_progress") is True, "observer scope mismatch")
    require(
        failure.get("error") == fit.get("error") == execution.get("error") == observer.get("sampling_error"),
        "failure causes mismatch",
    )
    bundle.read(terminal["log"])
    bundle.nested_pins(plan)
    bundle.nested_pins(failure.get("admission"))
    for path, digest in plan["source_pins"].items():
        bundle.read({"path": path, "sha256": digest})
    # Observer files are relative to the native fit directory, never to cwd.
    fit_directory = attempt_root / "fit"

    def read_inventory(files):
        require(isinstance(files, dict), "invalid native inventory")
        for name, digest in files.items():
            require(Path(name).name == name and name not in (".", ".."), "invalid observer file name")
            bundle.read({"path": str(fit_directory / name), "sha256": digest})

    read_inventory(observer["files"])
    if "observer_inventory" in fit:
        inventory = fit["observer_inventory"]
        require(inventory.get("complete") is True and inventory.get("errors") == [], "incomplete observer inventory")
        files = inventory["files"]
        require(
            files.get("parent-0-observer-receipt.json") == terminal["observer"]["sha256"]
            and all(files.get(name) == digest for name, digest in observer["files"].items()),
            "observer inventory binding mismatch",
        )
        read_inventory(files)
    if "progress_files" in fit:
        read_inventory(fit["progress_files"])
    report = dict(
        schema=ADAPTER,
        checkpoint=checkpoint_pin,
        plan=failure["plan"],
        scope=scope,
        provider=plan["provider"],
        source_inputs=plan["source_inputs"],
        source_pins=plan["source_pins"],
        runtime=plan["runtime"],
        execution_policy=plan["execution_policy"],
        owner_observation=None
        if checkpoint is None
        else {
            "status": checkpoint["status"],
            "goal_status": checkpoint.get("goal_status"),
            "issued_utc": checkpoint["issued_utc"],
            "is_liveness": False,
        },
        dimensions={
            "mathematical_check": "not_established",
            "numerical_fit": "failed",
            "prediction_precision": "not_established",
            "empirical_evaluation": "not_established",
            "publication": "not_authorized",
        },
        error=failure["error"],
        qualification=False,
        recommendation=None,
        verified_artifacts=bundle.artifacts,
    )
    return bundle.freeze(report)


COMPLETED_FAILURE_VERSION = "senate_completed_conditional_failure_adapter_v1"
COMPLETION_EXECUTION = "share_coordinate_completion_execution_v2"
COMPLETION_FAILURE_CLASSIFICATION = "COMPLETED_FIRST_CONDITIONAL_SAMPLING_FAILED_NUMERICAL_CONVERGENCE"


def inspect_completed_failure(root, terminal_pin, expected_plan_sha256, declared_scope):
    """Inspect the completed-first-conditional failure contract; never load the pickle."""
    bundle = Bundle(root)
    terminal = bundle.document(terminal_pin)
    require(terminal.get("schema") == "FullActualFitTerminalRootV1", "unsupported terminal schema")
    require(terminal.get("classification") == COMPLETION_FAILURE_CLASSIFICATION, "unsupported terminal classification")
    require(
        type(terminal.get("authoritative_exit_code")) is int and terminal["authoritative_exit_code"] == 1,
        "unsupported exit code",
    )
    require(terminal.get("terminal_confirmed_by_tool_handle") is True, "missing owner terminal observation")
    require(
        terminal.get("complete_cut") is False and terminal.get("no_automatic_retry") is True,
        "contradictory terminal outcome",
    )
    require(terminal.get("no_numerical_gates_relaxed") is True, "unsupported gate change")
    require(terminal.get("resource_budget_exceeded") is False, "unsupported budget failure")
    require(
        type(terminal.get("completed_population_conditionals")) is int
        and terminal["completed_population_conditionals"] == 1
        and type(terminal.get("numerically_qualified_population_conditionals")) is int
        and terminal["numerically_qualified_population_conditionals"] == 0,
        "unsupported conditional counts",
    )
    pins = terminal.get("preserved_files")
    require(isinstance(pins, list) and len(pins) == 8, "unsupported terminal inventory")
    by_path = {bundle.resolve(pin["path"]): pin for pin in pins}
    require(len(by_path) == len(pins), "duplicate terminal inventory path")
    candidates = [path for path in by_path if path.name == "failure.json" and path.parent.name != "fit"]
    require(len(candidates) == 1, "ambiguous native attempt")
    attempt = candidates[0].parent
    require(bundle.resolve(terminal_pin["path"]).parent == attempt.parent, "terminal attempt directory mismatch")
    names = [
        "failure.json",
        "fit/failure.json",
        "fit/parent-0-result.json",
        "fit/parent-0-diagnostics.csv",
        "fit/parent-0-returned-inference.json",
        "fit/parent-0-returned-inference.pkl",
        "fit/parent-0-execution.json",
        "fit/parent-0-sampling-budget.json",
    ]
    require(set(by_path) == {attempt / name for name in names}, "native attempt path mismatch")
    data = {name: bundle.read(by_path[attempt / name]) for name in names}
    documents = {name: decode(raw) for name, raw in data.items() if name.endswith(".json")}
    failure, fit, result = (
        documents[name] for name in ["failure.json", "fit/failure.json", "fit/parent-0-result.json"]
    )
    for record in [failure, fit]:
        require(
            record.get("status") == "incomplete_preserved_attempt" and record.get("no_automatic_retry") is True,
            "unsupported failure outcome",
        )
    require(
        failure.get("forecast_qualified") is False and failure.get("publication_authorized") is False,
        "failure cannot qualify",
    )
    require(fit.get("complete_cut") is False and failure.get("error") == fit.get("error"), "failure identity mismatch")
    require(fit.get("returned_inference_preservation_error") is None, "inference preservation failed")
    require(failure["plan"]["sha256"] == expected_plan_sha256, "plan binding mismatch")
    plan = bundle.document(failure["plan"])
    bundle.read(failure["admission"])
    require(
        plan.get("schema") == "actual_TX_role_execution_plan_v2"
        and plan.get("execution_adapter_version") == COMPLETION_EXECUTION,
        "unsupported full execution plan",
    )
    require(
        plan.get("forecast_qualified") is False
        and plan.get("publication_authorized") is False
        and plan.get("automatic_retry") is False,
        "unsupported positive or retry plan",
    )
    require(set(declared_scope) == set(SCOPE_KEYS), "declared scope keys mismatch")
    scope = {key: plan[key] for key in SCOPE_KEYS}
    require(canonical(scope) == canonical(declared_scope), "declared scope mismatch")
    parents = plan["parent_ids"]
    masses = plan["parent_masses"]
    targets = plan["scientific_target_ids"]
    require(
        isinstance(parents, list) and len(parents) > 1 and len(set(parents)) == len(parents),
        "invalid original parent support",
    )
    require(
        len(masses) == len(targets) == len(parents)
        and all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in masses)
        and math.isclose(sum(masses), 1, rel_tol=0, abs_tol=1e-12),
        "invalid original parent masses",
    )
    require(
        type(terminal.get("required_population_conditionals")) is int
        and terminal["required_population_conditionals"] == len(parents)
        and fit.get("required_parent_ids") == parents,
        "parent support mismatch",
    )
    require(
        fit.get("completed_parents") == [result]
        and result.get("parent_id") == parents[0]
        and result.get("scientific_target_id") == targets[0]
        and result.get("conditional_bank_only") is True,
        "completed result identity mismatch",
    )
    coordinate = plan.get("provider", {}).get("coordinate_version")
    require(
        isinstance(coordinate, str) and coordinate and result.get("coordinate_version") == coordinate,
        "coordinate identity mismatch",
    )
    policy = plan["execution_policy"]
    require(
        policy.get("mode") == "full_all_parents" and plan.get("complete_cut_expected") is True,
        "unsupported execution scope",
    )
    require(
        fit["execution_policy"].get("version") == COMPLETION_EXECUTION
        and fit["execution_policy"].get("policy") == policy,
        "fit execution policy mismatch",
    )
    budget = documents["fit/parent-0-sampling-budget.json"]
    require(
        result.get("sampling_budget") == budget
        and budget.get("version") == COMPLETION_EXECUTION
        and budget.get("policy") == policy,
        "sampling budget binding mismatch",
    )
    require(
        budget.get("sampler_returned") is True
        and budget.get("budget_exceeded") is False
        and budget.get("error") is None
        and budget.get("automatic_retry") is False
        and budget.get("publication_authorized") is False,
        "unsupported sampling outcome",
    )
    settings = plan["settings"]
    for key in ["chains", "draws", "tune"]:
        require(type(settings.get(key)) is int and settings[key] > 0, "invalid sampling dimensions")
    require(
        terminal.get("chains") == settings["chains"]
        and terminal.get("retained_per_chain") == settings["draws"]
        and terminal.get("warmup_per_chain") == settings["tune"],
        "terminal sampling mismatch",
    )
    require(
        type(budget.get("returned_transitions")) is int
        and budget["returned_transitions"] == settings["chains"] * (settings["draws"] + settings["tune"]),
        "returned transitions mismatch",
    )
    archive = documents["fit/parent-0-returned-inference.json"]
    require(
        result.get("returned_inference_archive") == archive
        and fit.get("complete_returned_inference_archive") == archive,
        "archive binding mismatch",
    )
    require(
        archive.get("complete") is True
        and archive.get("returned_object_available") is True
        and archive.get("numerical_qualification") is False
        and archive.get("error") is None
        and archive.get("format") == "python_pickle_complete_InferenceData_same_pinned_runtime_v1"
        and archive.get("groups") == ["posterior", "sample_stats"],
        "unsupported returned archive",
    )
    require(
        archive.get("files")
        == {"parent-0-returned-inference.pkl": by_path[attempt / "fit/parent-0-returned-inference.pkl"]["sha256"]},
        "archive file mismatch",
    )
    execution = documents["fit/parent-0-execution.json"]
    require(
        execution.get("schema") == "conditional_process_observation_v1" and execution.get("error") is None,
        "unsupported process observation",
    )
    require(
        result.get("execution_observation")
        == {"file": "parent-0-execution.json", "sha256": by_path[attempt / "fit/parent-0-execution.json"]["sha256"]},
        "execution observation mismatch",
    )
    diagnostics = result["diagnostics"]
    require(
        diagnostics.get("passed") is False
        and diagnostics.get("finite_diagnostics") is True
        and diagnostics.get("gates") == settings,
        "unsupported diagnostics",
    )
    terminal_diagnostic_keys = {
        "passed",
        "finite_diagnostics",
        "max_rhat",
        "min_bulk_ess",
        "min_tail_ess",
        "divergent",
        "depth_saturation",
        "gates",
    }
    require(
        isinstance(terminal.get("diagnostics"), dict) and set(terminal["diagnostics"]) == terminal_diagnostic_keys,
        "terminal diagnostics keys mismatch",
    )
    require(
        all(terminal["diagnostics"][key] == diagnostics.get(key) for key in terminal_diagnostic_keys),
        "terminal diagnostics mismatch",
    )
    # Preserve declared observation evidence as bytes. These dated records are not
    # fresh process checks or independent verification of the numerical diagnostics.
    observation_files = {}
    if "observer_inventory" in fit:
        inventory = fit["observer_inventory"]
        require(inventory.get("complete") is True and inventory.get("errors") == [], "incomplete observer inventory")
        observation_files.update(inventory["files"])
    for inventory in [observation_files, fit.get("progress_files", {})]:
        require(isinstance(inventory, dict), "invalid observation inventory")
        for name, digest in inventory.items():
            require(Path(name).name == name and name not in (".", ".."), "invalid observation file name")
            bundle.read({"path": str(attempt / "fit" / name), "sha256": digest})
    rows = list(csv.DictReader(io.StringIO(data["fit/parent-0-diagnostics.csv"].decode())))
    require(
        rows and len({row.get("") for row in rows}) == len(rows) and all(row.get("") for row in rows),
        "invalid diagnostic row identities",
    )
    values = {key: [float(row[key]) for row in rows] for key in ["r_hat", "ess_bulk", "ess_tail"]}
    require(all(math.isfinite(x) and x > 0 for column in values.values() for x in column), "invalid diagnostic values")
    extrema = {
        "max_rhat": max(values["r_hat"]),
        "min_bulk_ess": min(values["ess_bulk"]),
        "min_tail_ess": min(values["ess_tail"]),
    }
    require(
        all(math.isclose(value, diagnostics[key], rel_tol=1e-12, abs_tol=1e-12) for key, value in extrema.items()),
        "diagnostic extrema mismatch",
    )
    require(
        extrema["max_rhat"] > settings["max_rhat"]
        or extrema["min_bulk_ess"] < settings["min_ess"]
        or extrema["min_tail_ess"] < settings["min_ess"],
        "no recorded convergence failure",
    )
    report = dict(
        schema=COMPLETED_FAILURE_VERSION,
        terminal=terminal_pin,
        plan=failure["plan"],
        scope=scope,
        verification_scope="Terminal/native failure/result/CSV/archive bytes and internal bindings; "
        "CSV extrema checked, diagnostics not recomputed from draws; transitive scientific ancestry not replayed.",
        owner_observation={
            "issued_utc": terminal["issued_utc"],
            "reported_exit_code": 1,
            "is_liveness": False,
            "tool_handle_independently_polled": False,
        },
        dimensions={
            "execution": "first_conditional_sampling_returned",
            "mathematical_check": "not_established",
            "numerical_fit": "failed",
            "complete_population_mixture": "incomplete",
            "prediction_precision": "not_established",
            "empirical_evaluation": "not_established",
            "publication": "not_authorized",
        },
        completed_population_conditionals=1,
        required_population_conditionals=len(parents),
        original_parent_masses=masses,
        numerical_diagnostics=diagnostics,
        csv_rows=len(rows),
        error=failure["error"],
        qualification=False,
        recommendation=None,
        automatic_retry=False,
        verified_artifacts=bundle.artifacts,
    )
    return bundle.freeze(report)


HOUSE_REVIEW_SHA = "70f371b56d63909c847d0aa539efb28a5c4a538c3ed2e33052ea4f0a28eea9ee"


def inspect_house(root, review_pin, cycle, configuration, declared_plan):
    """Recognize one cell of the exact independently reviewed historical handoff."""
    require(review_pin.get("sha256") == HOUSE_REVIEW_SHA, "untrusted House review")
    bundle = Bundle(root)
    review = bundle.document(review_pin)
    require(review.get("schema") == "house-actual-output-review-v1", "unsupported House review")
    cells = [cell for cell in review["cells"] if cell["cycle"] == cycle and cell["configuration"] == configuration]
    require(len(cells) == 1, "unknown or ambiguous reviewed cell")
    cell = cells[0]
    require(
        cell["numerical_status"] == "passed" and cell["structural_and_score_audit_passed"] is True,
        "review did not qualify race evidence",
    )
    fit = bundle.document(cell["fit_receipt"])
    require(fit.get("schema") == "house-fit-receipt-v1", "unsupported House fit receipt")
    plan = bundle.document(fit["plan"])
    require(plan.get("schema") == "house-fit-plan-v1", "unsupported House plan")
    require(canonical(plan) == canonical(declared_plan), "declared plan mismatch")
    require(
        plan["target_id"] == "house_aggregate_fundamentals_retrospective_v1"
        and plan["target_cycle"] == cycle
        and plan["house_configuration_id"] == configuration,
        "target mismatch",
    )
    require(
        fit["diagnostics"]["passed"] is True and fit["race_prediction_precision_passed"] is True,
        "race fit not qualified",
    )
    require(
        fit["subset_seat_precision_passed"] is False and cell["subset_seat_precision_passed"] is False,
        "unexpected seat capability",
    )
    require(fit["prediction_diagnostics"]["checks"]["subtotal_mcse"] is False, "unexpected subtotal gate")
    suffix = f"/{configuration}-{cycle}-h14.json"
    outputs = [pin for pin in review["outputs"] if pin["path"].endswith(suffix)]
    require(len(outputs) == 1, "review result binding missing")
    result = bundle.document(outputs[0])
    require(result.get("schemaVersion") == 1 and result.get("reviewStatus") == "reviewed", "unsupported House result")
    require(
        result["run"]["id"] == f"{configuration}-{cycle}-h14"
        and result["run"]["configurationSha256"] == fit["plan"]["sha256"],
        "result plan mismatch",
    )
    require(
        result["run"]["historical"] is True
        and result["run"]["experimental"] is True
        and result["metadata"]["full_chamber"] is False,
        "unsupported result scope",
    )
    # Verify the loaded plan itself, including its actual implementation pins.
    source_pins = plan.get("source_pins")
    require(isinstance(source_pins, list) and source_pins, "unsupported House source pin inventory")
    require(
        len({str(bundle.resolve(pin["path"])) for pin in source_pins}) == len(source_pins),
        "duplicate House source pin",
    )
    bundle.nested_pins(plan)
    # Verify declared fit output bytes without executing/unpickling any arrays.
    bundle.nested_pins(fit)
    for key in ("panel", "scores", "review_source"):
        bundle.read(review[key])
    report = dict(
        schema="house_historical_review_adapter_v1",
        verification_scope=(
            "selected review/fit/result and direct loaded-plan pins; transitive ancestors not replayed"
        ),
        scope=plan,
        independent_review=review_pin,
        fit_receipt=cell["fit_receipt"],
        result=outputs[0],
        native_fit=fit,
        dimensions={
            "mathematical_check": "not_established",
            "numerical_fit": "independently_reviewed_race_pass",
            "prediction_precision": "race_pass_subtotal_fail",
            "empirical_evaluation": "independently_reviewed_historical_aggregate",
            "publication": "not_activated",
        },
        qualification=False,
        recommendation=None,
        owner_observation=None,
        limitations=result["limitations"],
        verified_artifacts=bundle.artifacts,
    )
    return bundle.freeze(report)


def attach_evidence(ledger, command_id, report, bundle):
    """Copy verified evidence and normalized report only; never append a phase."""
    require(isinstance(bundle, Inspection), "attachment requires a completed inspection")
    require(canonical(report).encode() == bundle.report_bytes, "report changed since inspection")
    require(report.get("verified_artifacts") == bundle.artifacts, "inspection pin identity mismatch")
    # Stage and verify all source bytes before writing any ledger artifacts. Immutable
    # staged bytes also close the verify/copy race on caller-owned source files.
    verifier = Bundle(bundle.root)
    with tempfile.TemporaryDirectory() as directory:
        staged = []
        for index, (path, digest, size) in enumerate(bundle.pins):
            data = verifier.read({"path": path, "sha256": digest})
            require(len(data) == size, "artifact size mismatch")
            target = Path(directory) / str(index)
            target.write_bytes(data)
            staged.append((target, digest))
        for target, digest in staged:
            copied = ledger.put_artifact(f"{command_id}:source:{digest}", target)
            require(copied["sha256"] == digest, "artifact changed during attachment")
        path = Path(directory) / "normalized.json"
        path.write_bytes(bundle.report_bytes)
        return ledger.put_artifact(command_id + ":report", path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--request", required=True, help="Versioned adapter request JSON file")
    parser.add_argument("--journal-root")
    parser.add_argument("--command-id")
    args = parser.parse_args()
    require(bool(args.journal_root) == bool(args.command_id), "journal root and command id must be supplied together")
    request = decode(Path(args.request).read_bytes())
    require(request.pop("schema", None) == "run_receipt_request_v1", "unsupported request schema")
    kind = request.pop("kind", None)
    adapters = {
        "senate_failed": inspect_receipts,
        "house_historical": inspect_house,
        "senate_completed_failure": inspect_completed_failure,
    }
    require(kind in adapters, "unsupported receipt adapter")
    function = adapters[kind]
    report, bundle = function(args.source_root, **request)
    if args.journal_root:
        ledger = Ledger(args.journal_root)
        try:
            evidence = attach_evidence(ledger, args.command_id, report, bundle)
            print(canonical({"report": report, "journal_evidence": evidence}))
        finally:
            ledger.close()
    else:
        print(canonical(report))


if __name__ == "__main__":
    main()
