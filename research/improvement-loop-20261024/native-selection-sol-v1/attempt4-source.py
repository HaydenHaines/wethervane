"""Read-only native count candidate inspection for a future scoped research decision.

This module prepares and resolves a complete owner configuration. It never selects
an incumbent, approves evidence, starts inference, or publishes a forecast.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Mapping


SCHEMA = "NativeCountCandidateSpecV1"
RECEIPT_SCHEMA = "VerifiedNativeCountCandidateV1"
TARGET = "population_cut_v1"
OUTPUT = "whole_parent_joint_candidate_counts_v1"
RESULT_KIND = "composition-application-result-v1"
ROLES = ("population", "community", "fundamentals", "environment", "measurement", "participation", "choice", "counts_rules")
NATIVE_MODULE_PINS = {
    "driver": ("research/senate-stack-integration-20261020/count-application-driver-astra-v1/driver.py", "e89e82ecd39b213683ee6172be0d74ac31c646df42dd6f00c471958af1d05c88"),
    "backend": ("research/senate-stack-integration-20261020/count-application-driver-astra-v1/completion_backend.py", "c25d116601c5659f961ac648cae2280524a003ef70166844bb0543f07f3374ec"),
    "application": ("src/forecasting/composition_application_v1.py", "0510a10ffdde85136e44aa09702cd9f6838749ec64430b2dc5b62491e25402f1"),
    "registry": ("src/forecasting/model_registry_v1.py", "a028374c4cc068e5e65aa26d99c26b38a02d6bd6b101eb119a19d6f1a981f1ed"),
    "artifacts": ("src/model_stack/artifacts.py", "a5d4533ebab5007e8ab05cc5d63ea2abf2eb904f6a7d37d99124890625a4ffce"),
    "contracts": ("src/model_stack/contracts.py", "2ff6f84315fbef60fb223815f8125f63da0244aa01704b083f4c451fdd2686f8"),
    "banks": ("src/senate_count_stack_v1/banks.py", "3b47727fde988ab9be41519a2c02522f60c9d681488eb06081863b47898165b8"),
    "runtime_arviz": (".redesign-runtime/response-20261007/reference-dependencies/arviz/__init__.py", "20e754eee2096c7bab7fefa62b3b6e47862af9f54748bc94fe778be272bcf4a1"),
}
_TOKEN = object()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _pin(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {"path", "sha256"} and isinstance(value["path"], str) and _hash(value["sha256"])


def _read_pin(root: Path, pin: dict[str, str], limit: int = 200_000) -> bytes:
    _require(_pin(pin), "Invalid source pin")
    name = Path(pin["path"])
    _require(not name.is_absolute() and all(part not in ("", ".", "..") for part in name.parts), "Unsafe source path")
    path = (root / name).resolve(strict=True)
    _require(path.is_relative_to(root) and path.is_file() and not (root / name).is_symlink(), "Source path escapes owner root")
    _require(path.stat().st_size <= limit, "Source metadata exceeds limit")
    data = path.read_bytes()
    _require(_sha(data) == pin["sha256"], "Source fingerprint mismatch: " + pin["path"])
    return data


def _parse_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    _require(isinstance(spec, dict) and set(spec) == {"schema", "ownerRoot", "pythonExecutable", "pythonSha256",
        "sources", "expectedRequestSha256", "expectedConfigurationSha256", "artifactStoreRoot", "artifactRef"}, "Invalid native candidate spec fields")
    _require(spec["schema"] == SCHEMA and isinstance(spec["ownerRoot"], str) and Path(spec["ownerRoot"]).is_absolute(), "Invalid owner root")
    _require(isinstance(spec["pythonExecutable"], str) and Path(spec["pythonExecutable"]).is_absolute() and _hash(spec["pythonSha256"]), "Invalid Python identity")
    _require(isinstance(spec["artifactStoreRoot"], str) and Path(spec["artifactStoreRoot"]).is_absolute(), "Invalid artifact store root")
    _require(_hash(spec["expectedRequestSha256"]) and _hash(spec["expectedConfigurationSha256"]), "Invalid expected native digests")
    sources = spec["sources"]
    _require(isinstance(sources, dict) and set(sources) == {"request", "bundle", "registration", *NATIVE_MODULE_PINS}, "Incomplete native source pins")
    _require(all(_pin(pin) for pin in sources.values()), "Invalid native source pin")
    for name, (path, digest) in NATIVE_MODULE_PINS.items():
        _require(sources[name] == {"path": path, "sha256": digest}, "Unreviewed native module: " + name)
    ref = spec["artifactRef"]
    _require(ref is None or (isinstance(ref, dict) and set(ref) == {"kind", "artifact_id"} and
        ref["kind"] == RESULT_KIND and _hash(ref["artifact_id"])), "Invalid count result artifact reference")
    owner = Path(spec["ownerRoot"]).resolve(strict=True)
    _require(owner.is_dir(), "Owner root is not a directory")
    executable = Path(spec["pythonExecutable"])
    _require(executable.exists() and executable.is_file() and os.access(executable, os.X_OK), "Owner Python is unavailable")
    _require(_sha(executable.resolve(strict=True).read_bytes()) == spec["pythonSha256"], "Owner Python fingerprint mismatch")
    for name, pin in sources.items():
        _read_pin(owner, pin, 200_000)
    return dict(spec)


@dataclass(frozen=True, slots=True)
class VerifiedNativeCandidate:
    """Immutable receipt carrier; callers must obtain it through inspection, not construct it."""

    status: str
    request_sha256: str
    configuration_sha256: str
    artifact_id: str | None
    receipt_sha256: str
    _receipt_bytes: bytes = field(repr=False)
    _token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _require(self._token is _TOKEN, "Use inspect_count_candidate to obtain verified native evidence")

    def receipt(self) -> dict[str, Any]:
        return json.loads(self._receipt_bytes)


def inspect_count_candidate(spec: Mapping[str, Any], *, timeout_seconds: int = 300) -> VerifiedNativeCandidate:
    """Inspect via a fresh owner interpreter so lab imports cannot contaminate src/driver."""
    checked = _parse_spec(spec)
    adapter_sha = _sha(Path(__file__).resolve().read_bytes())
    command = [checked["pythonExecutable"], "-I", "-B", str(Path(__file__).resolve()), "--native-worker"]
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONNOUSERSITE": "1",
        "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
    try:
        completed = subprocess.run(command, input=_canonical(checked), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False, timeout=timeout_seconds, env=env)
    except subprocess.TimeoutExpired as exc:
        # subprocess.run terminates and waits for this worker before raising.
        raise TimeoutError("Native count inspection timed out; its isolated worker was terminated") from exc
    _require(completed.returncode == 0, "Native inspection failed: " + completed.stderr.decode("utf-8", "replace")[-1200:])
    _require(len(completed.stdout) <= 200_000, "Native inspection returned oversized metadata")
    receipt = json.loads(completed.stdout)
    _require(isinstance(receipt, dict) and receipt.get("schema") == RECEIPT_SCHEMA and
        receipt.get("adapterSha256") == adapter_sha == _sha(Path(__file__).resolve().read_bytes()) and
        receipt.get("specSha256") == _sha(_canonical(checked)) and
        receipt.get("taskScopeSha256") == _sha(_canonical(receipt.get("taskScope"))) and
        receipt.get("request", {}).get("sha256") == checked["expectedRequestSha256"] and
        receipt.get("configuration", {}).get("sha256") == checked["expectedConfigurationSha256"] and
        receipt.get("status") in ("requires_computation", "ready"), "Native inspection identity mismatch")
    result = receipt.get("result")
    _require((checked["artifactRef"] is None and receipt["status"] == "requires_computation" and result is None) or
        (checked["artifactRef"] is not None and receipt["status"] == "ready" and isinstance(result, dict) and
         result.get("artifactRef") == checked["artifactRef"]), "Native result presence mismatch")
    payload = _canonical(receipt)
    return VerifiedNativeCandidate(receipt["status"], receipt["request"]["sha256"],
        receipt["configuration"]["sha256"], None if result is None else result["artifactRef"]["artifact_id"],
        _sha(payload), payload, _TOKEN)


def _native_worker() -> None:
    spec = _parse_spec(json.loads(sys.stdin.buffer.read(200_001)))
    root = Path(spec["ownerRoot"]).resolve(strict=True)
    _require(not any(name == "src" or name.startswith("src.") or name in ("driver", "completion_backend") for name in sys.modules),
        "Foreign native module already loaded")
    package = root / "research/senate-stack-integration-20261020/count-application-driver-astra-v1"
    reference_dependencies = root / ".redesign-runtime/response-20261007/reference-dependencies"
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(package))
    sys.path.insert(0, str(reference_dependencies))
    import arviz
    arviz_pin = spec["sources"]["runtime_arviz"]
    _require(Path(arviz.__file__).resolve(strict=True) == (root / arviz_pin["path"]).resolve(strict=True) and
        _sha(Path(arviz.__file__).read_bytes()) == arviz_pin["sha256"] and arviz.__version__ == "0.22.0",
        "Native runtime ArviZ differs from pinned plan")
    from src.forecasting.composition_application_v1 import CompositionRequest
    from src.forecasting.model_registry_v1 import SourcePin
    from src.model_stack.artifacts import ArtifactRef, ArtifactStore, canonical, digest
    from completion_backend import CompletionBackend
    from driver import application

    class ReadOnlyArtifactStore(ArtifactStore):
        def put(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("Native candidate inspection cannot store an artifact")

    source = spec["sources"]
    backend = CompletionBackend(root, SourcePin(**source["registration"]))
    app, driver = application(root, ReadOnlyArtifactStore(spec["artifactStoreRoot"]), backend, SourcePin(**source["bundle"]))
    request = CompositionRequest.model_validate_json(_read_pin(root, source["request"]))
    _require(request.model_dump(mode="json") == driver.request().model_dump(mode="json"), "Pinned request differs from bound driver request")
    _require(request.target_id == TARGET and request.output_contract_id == OUTPUT and request.request_sha256 == spec["expectedRequestSha256"],
        "Wrong native request target/output/identity")
    if spec["artifactRef"] is not None:
        app.remember(request, ArtifactRef.model_validate(spec["artifactRef"]))
    resolved = app.resolve(request)
    _require(resolved.status == ("ready" if spec["artifactRef"] else "requires_computation") and resolved.configuration is not None,
        "Native application did not resolve the expected state: " + resolved.status + " " + resolved.reason)
    configuration = resolved.configuration
    bound = driver.initial
    _require(configuration.configuration_sha256 == spec["expectedConfigurationSha256"] and
        configuration.request.request_sha256 == request.request_sha256 and
        configuration.support_id == bound.problem_identity and
        configuration.scientific_plan == bound.plan, "Native prepared configuration identity changed")
    _require(bound.scope == "actual_research" and set(bound.selections) == set(ROLES), "Count scope/roles are not the actual research registration")
    plan = json.loads(_read_pin(root, bound.plan.model_dump(mode="json")))
    problem_pin = plan.get("source_inputs", {}).get("problem")
    _require(_pin(problem_pin) and plan.get("problem_identity") == bound.problem_identity and
        plan.get("cutoff_utc") == request.information_cutoff.isoformat().replace("+00:00", "Z"),
        "Native plan lacks the prepared physical problem binding")
    problem = json.loads(_read_pin(root, problem_pin, 3_000_000))
    _require(problem.get("purpose") == "actual_current" and problem.get("cutoff_utc") == plan["cutoff_utc"] and
        isinstance(problem.get("events"), list) and isinstance(problem.get("contests"), list),
        "Native physical problem scope changed")
    prediction_events = []
    for event in problem["events"]:
        if event.get("role") == "prediction":
            _require(isinstance(event.get("id"), str) and isinstance(event.get("state"), str) and
                type(event.get("cycle")) is int and isinstance(event.get("date"), str), "Malformed prediction event")
            prediction_events.append({key: event[key] for key in ("id", "state", "cycle", "date")})
    _require(prediction_events and len({row["id"] for row in prediction_events}) == len(prediction_events),
        "No unique prediction events")
    event_ids = {row["id"] for row in prediction_events}
    prediction_contests = []
    for contest in problem["contests"]:
        if contest.get("event_id") not in event_ids:
            continue
        candidates, categories = contest.get("candidates"), contest.get("noncandidate_categories")
        _require(isinstance(candidates, list) and candidates and isinstance(categories, list) and
            all(isinstance(row, dict) and isinstance(row.get("id"), str) for row in candidates) and
            all(isinstance(row, dict) and isinstance(row.get("id"), str) and isinstance(row.get("kind"), str) for row in categories),
            "Malformed prediction ballot menu")
        row = {"id": contest["id"], "eventId": contest["event_id"],
            "kind": contest["kind"], "rule": contest["rule"]}
        row["candidateIds"] = [candidate["id"] for candidate in candidates]
        row["noncandidateCategories"] = [{"id": category["id"], "kind": category["kind"]} for category in categories]
        _require(len(set(row["candidateIds"])) == len(row["candidateIds"]) and
            len({category["id"] for category in row["noncandidateCategories"]}) == len(row["noncandidateCategories"]),
            "Duplicate physical ballot option")
        prediction_contests.append(row)
    _require(prediction_contests and len({row["id"] for row in prediction_contests}) == len(prediction_contests),
        "No unique prediction contests")
    task_scope = {"targetId": TARGET, "outputContractId": OUTPUT,
        "predictionEvents": prediction_events, "predictionContests": prediction_contests}
    native_result = None
    if resolved.status == "ready":
        record, summary = resolved.record, resolved.summary
        _require(record is not None and record.numerical_status == "passed" and record.evidence_status == "unassessed" and
            resolved.artifact is not None and isinstance(summary, dict), "Native count result failed numerical/evidence checks")
        _require(summary.get("scope") == bound.scope and summary.get("target") == TARGET and
            summary.get("status") == "numerically_complete_research_counts" and
            summary.get("problem_identity") == bound.problem_identity and
            summary.get("upstream_identity") == bound.upstream_identity and
            summary.get("composition_identity") == bound.composition_identity and
            summary.get("parent_ids") == list(bound.parent_ids) and
            summary.get("parent_masses") == list(bound.parent_masses) and
            summary.get("scientific_target_ids") == list(bound.scientific_target_ids) and
            summary.get("common_draw_identity") == "original_parent_chain_draw_shared_across_all_event_contest_counts" and
            summary.get("forecast_qualified") is False and summary.get("publication_authorized") is False,
            "Native saved summary does not preserve complete count scope/draw identity")
        native_result = {
            "artifactRef": resolved.artifact.model_dump(mode="json"),
            "record": record.model_dump(mode="json"),
            "summarySha256": digest(summary),
            "predictionRef": summary.get("prediction"),
            "terminalReceipt": summary.get("terminal_receipt"),
            "commonDrawIdentity": summary["common_draw_identity"],
            "forecastQualifiedByNativeResult": False,
            "publicationAuthorizedByNativeResult": False,
        }
    receipt = {
        "schema": RECEIPT_SCHEMA, "adapterSha256": _sha(Path(__file__).resolve().read_bytes()),
        "specSha256": _sha(_canonical(spec)), "status": resolved.status,
        "request": {"sha256": request.request_sha256, "targetId": TARGET, "outputContractId": OUTPUT,
            "informationCutoff": request.information_cutoff.isoformat().replace("+00:00", "Z"),
            "dataBundle": request.data_bundle.model_dump(mode="json"),
            "selections": [choice.model_dump(mode="json") for choice in request.selections]},
        "configuration": {"sha256": configuration.configuration_sha256, "driverId": configuration.driver_id,
            "driverVersion": configuration.driver_version, "lawSha256": configuration.law_sha256,
            "graphSha256": configuration.graph_sha256, "supportId": configuration.support_id,
            "scientificPlan": configuration.scientific_plan.model_dump(mode="json"),
            "dependenciesSha256": digest([pin.model_dump(mode="json") for pin in configuration.dependencies]),
            "outputs": [port.model_dump(mode="json") for port in configuration.outputs]},
        "scope": {"kind": bound.scope, "problemIdentity": bound.problem_identity,
            "upstreamIdentity": bound.upstream_identity, "compositionIdentity": bound.composition_identity,
            "scientificTargetIds": list(bound.scientific_target_ids), "parentIds": list(bound.parent_ids),
            "parentMasses": list(bound.parent_masses), "parentProjectionDigest": bound.parent_projection_digest,
            "executionVariant": bound.execution_variant},
        "taskScope": task_scope, "taskScopeSha256": digest(task_scope),
        "sources": {name: source[name] for name in sorted(source)},
        "result": native_result,
        "qualification": {"independentReviewVerified": False, "empiricalForecastQualified": False,
            "selectedResearchIncumbent": False, "publiclyAvailable": False},
    }
    sys.stdout.buffer.write(canonical(receipt))


if __name__ == "__main__":
    if sys.argv[1:] != ["--native-worker"]:
        raise SystemExit("Use inspect_count_candidate from a trusted local caller")
    try:
        _native_worker()
    except Exception as exc:
        print("Native count candidate inspection rejected: " + str(exc)[:1000], file=sys.stderr)
        raise SystemExit(1) from None
