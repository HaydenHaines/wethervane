"""Compile separately reviewed count-selection phases into V3 queue plans for new stores.

This module emits data only. Existing queue/executor and native selection checks
remain authoritative; no scheduler, native inspector hook or scientific policy is
added here.
"""

import argparse
import copy
import sys
from pathlib import Path

QUEUE_DIR = Path(__file__).resolve().parent
HERE = QUEUE_DIR.parent
sys.path.insert(1, str(HERE))

import execution
import experiment_queue as queue
import scoped_selection as selection
from ledger import canonical
from run_receipts import decode

SOURCE_NAMES = ("selection_queue.py", "scoped_selection.py", "native_selection.py",
                "experiment_queue.py", "execution.py", "ledger.py", "run_receipts.py")


def require(value, message):
    execution.require(value, message)


def _pins(items):
    """Preserve explicit input anchors, rejecting conflicting path identities."""
    values = {}
    for item in items:
        execution.verify_pin(item)
        old = values.get(item["path"])
        require(old is None or old == item, "conflicting input pins")
        values[item["path"]] = item
    return list(values.values())


def _source_pins():
    # The recipe compiler uses queue-v3 imports; the child selection CLI
    # separately imports helpers from the parent directory.
    queue.current_sources()
    return [*[execution.pin(QUEUE_DIR / name) for name in
              ("selection_queue.py", "experiment_queue.py", "execution.py", "ledger.py", "run_receipts.py")],
            *[execution.pin(HERE / name) for name in
              ("scoped_selection.py", "native_selection.py", "execution.py", "ledger.py", "run_receipts.py")]]


def _require_v3_sources(spec):
    inputs = _pins(spec["input_pins"])
    require(all(pin in inputs for pin in _source_pins()),
            "V3 recipe source pins required; legacy plans cannot be adopted")


def compile_plan(request_pin):
    request = selection.document(request_pin)
    require(set(request) == {"schema", "phase", "queueId", "cycleId", "jobKey", "attemptId", "nativeSpec",
                             "pythonExecutable", "outputDirectory", "timeoutSeconds", "application"}
            and request["schema"] == "ScopedSelectionQueueRecipeV1" and request["phase"] in ("prepare", "apply"),
            "unsupported selection queue recipe")
    native_spec = request["nativeSpec"]
    selection.document(native_spec)  # Authenticate only; native validation runs in the owner command.
    output = execution.absolute(request["outputDirectory"])
    require(not output.exists(), "fresh selection owner output required")
    argv = [request["pythonExecutable"], str(HERE / "scoped_selection.py"), request["phase"]]
    inputs = [request_pin, native_spec, *_source_pins()]
    if request["phase"] == "prepare":
        require(request["application"] is None, "preparation cannot carry selection authority")
        argv.extend(["--native-spec", native_spec["path"], "--native-spec-sha256", native_spec["sha256"]])
        fields = dict(schema="ScopedCountSelectionPreparationV1", selectionEstablished=False,
                      empiricalStatus="unassessed", publicationAuthorized=False)
        role = "native-preparation"
    else:
        app = request["application"]
        require(isinstance(app, dict) and set(app) == {"registryRoot", "preparation", "decision", "expectedRevision", "expectedPredecessor", "commandId"},
                "explicit independently reviewed application required")
        execution.absolute(app["registryRoot"])
        preparation = selection.document(app["preparation"])
        decision = selection.document(app["decision"])
        require(preparation.get("schema") == "ScopedCountSelectionPreparationV1" and preparation.get("status") == "review_required"
                and preparation.get("candidateRef") is not None and preparation.get("nativeSpec") == native_spec,
                "application requires exact completed preparation; pending cannot establish selection")
        require(decision.get("schema") == "ScopedCountSelectionDecisionV1"
                and decision.get("taskKey") == preparation.get("taskKey")
                and decision.get("candidateRefSha256") == preparation.get("candidateRefSha256")
                and decision.get("expectedRevision") == app["expectedRevision"]
                and decision.get("expectedPredecessor") == app["expectedPredecessor"]
                and isinstance(decision.get("journal"), dict) and decision["journal"].get("cycle") == request["cycleId"]
                and decision.get("domain") in ("research", "synthetic"), "recipe/decision/preparation binding mismatch")
        require(type(app["expectedRevision"]) is int and app["expectedRevision"] >= -1
                and isinstance(app["commandId"], str) and app["commandId"].strip(), "explicit CAS/command identity required")
        # These are recipe identity checks, not scientific or journal admission.
        # scoped_selection.apply reconstructs native readiness and authority.
        argv.extend(["--root", app["registryRoot"], "--preparation", app["preparation"]["path"],
                     "--preparation-sha256", app["preparation"]["sha256"], "--native-spec", native_spec["path"],
                     "--native-spec-sha256", native_spec["sha256"], "--decision", app["decision"]["path"],
                     "--decision-sha256", app["decision"]["sha256"], "--expected-revision", str(app["expectedRevision"]),
                     "--expected-predecessor", canonical(app["expectedPredecessor"]), "--command-id", app["commandId"]])
        inputs.extend([app["preparation"], app["decision"]])
        fields = dict(schema="ScopedCountSelectionApplicationV1", domain=decision["domain"], publicationAuthorized=False)
        role = "selection-application"
    argv.extend(["--output", str(output)])
    spec = dict(schema="LocalExecutionSpecV1", cycle_id=request["cycleId"], job_key=request["jobKey"], attempt_id=request["attemptId"],
                argv=argv, executable=execution.executable_pin(request["pythonExecutable"]), cwd=str(HERE),
                environment={"PYTHONDONTWRITEBYTECODE": "1"}, input_pins=_pins(inputs), output_directory=str(output),
                native_receipts=[dict(path="receipt.json", required_fields=fields)], timeout_seconds=request["timeoutSeconds"],
                grace_seconds=3, previous_attempt=None, retry_reason="")
    job = dict(id=request["jobKey"], execution_spec=spec, dependencies=[],
               outputs=[dict(role=role, path="receipt.json", minimum_bytes=1)], resume_prelaunch=None)
    plan = dict(schema="LocalExperimentQueuePlanV1", queue_id=request["queueId"], jobs=[job])
    queue.validate_plan(plan)
    return plan


def compile_retry(previous_plan_pin, terminal_pin, queue_id, attempt_id, output_directory, reason):
    """Copy an exact command/payload, changing only execution retry identities."""
    previous = selection.document(previous_plan_pin)
    queue.validate_plan(previous)
    require(len(previous["jobs"]) == 1 and not previous["jobs"][0]["dependencies"], "one fixed selection phase required")
    job = previous["jobs"][0]
    old = job["execution_spec"]
    _require_v3_sources(old)
    argv = old["argv"]
    require(len(argv) > 3 and argv[1] == str(HERE / "scoped_selection.py") and argv[2] in ("prepare", "apply")
            and argv[-2:] == ["--output", old["output_directory"]], "unsupported selection command layout")
    terminal = selection.document(terminal_pin)
    execution.validate_terminal(terminal, old)
    require(selection.document(terminal["specification"]) == old and terminal["cleanup"]["verified"],
            "retry needs exact previous specification and verified cleanup")
    for item in terminal["evidence"]:
        execution.verify_pin(item)
    require(isinstance(reason, str) and reason.strip(), "explicit retry reason required")
    require(queue_id != previous["queue_id"] and attempt_id != old["attempt_id"], "retry requires new queue and attempt identities")
    output = execution.absolute(output_directory)
    require(str(output) != old["output_directory"] and not output.exists(), "retry requires fresh output; preserve previous partial output")
    plan = copy.deepcopy(previous)
    plan["queue_id"] = queue_id
    # A launched successor owns a new attempt/output. An earlier prelaunch
    # proof belongs only to the unchanged, previously unused attempt.
    plan["jobs"][0]["resume_prelaunch"] = None
    spec = plan["jobs"][0]["execution_spec"]
    spec.update(attempt_id=attempt_id, output_directory=str(output),
                previous_attempt=dict(attempt_id=old["attempt_id"], receipt=terminal_pin), retry_reason=reason)
    spec["argv"][-1] = str(output)
    spec["input_pins"] = _pins([*spec["input_pins"], previous_plan_pin, terminal_pin])
    queue.validate_plan(plan)
    return plan


def compile_resume(previous_plan_pin, binding_pin, result_pin, release_pin, queue_id, reason):
    """Copy the existing queue's proof for a sealed, never-launched deferral."""
    previous = selection.document(previous_plan_pin)
    queue.validate_plan(previous)
    require(len(previous["jobs"]) == 1 and not previous["jobs"][0]["dependencies"], "one fixed selection phase required")
    job = previous["jobs"][0]
    spec = job["execution_spec"]
    _require_v3_sources(spec)
    argv = spec["argv"]
    require(len(argv) > 3 and argv[1] == str(HERE / "scoped_selection.py") and argv[2] in ("prepare", "apply")
            and argv[-2:] == ["--output", spec["output_directory"]], "unsupported selection command layout")
    binding, result, release = map(selection.document, (binding_pin, result_pin, release_pin))
    require(binding.get("schema") == "LocalQueueExecutionBindingV1" and binding.get("job_id") == job["id"]
            and selection.document(binding["plan"]) == previous and binding.get("execution_spec") == spec
            and binding.get("specification_sha256") == queue.digest(spec), "prelaunch binding/plan mismatch")
    require(result.get("schema") == "LocalQueueJobResultV1" and result.get("job_id") == job["id"]
            and result.get("outcome") == "deferred_prelaunch_integrity" and result.get("executor_terminal") is None
            and release.get("never_launched") is True and release.get("result") == result_pin
            and selection.document(release["reservation"])["binding"] == binding_pin,
            "sealed never-launched queue evidence required")
    require(queue_id != previous["queue_id"] and isinstance(reason, str) and reason.strip(), "new resume queue/reason required")
    require(not Path(spec["output_directory"]).exists(), "unused output must remain fresh")
    plan = copy.deepcopy(previous)
    plan["queue_id"] = queue_id
    plan["jobs"][0]["resume_prelaunch"] = dict(queue_id=previous["queue_id"], job_id=job["id"],
        binding=binding_pin, result=result_pin, release=release_pin, reason=reason)
    # Ownership, every previous reference, admission pins and absence of launch
    # intent are checked under the existing queue's dispatcher lock at enqueue.
    queue.validate_plan(plan)
    return plan


def write_plan(plan, output):
    require(not Path(output).exists(), "fresh recipe output required")
    queue.validate_plan(plan)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    execution.publish(output / "plan.json", plan)
    result = dict(schema="ScopedSelectionQueuePlanReceiptV1", plan=execution.pin(output / "plan.json"),
                  canonicalPlanSha256=queue.digest(plan), qualification="unassessed", selectionEstablished=False,
                  publicationAuthorized=False)
    execution.publish(output / "receipt.json", result)
    return execution.pin(output / "receipt.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    build = sub.add_parser("build")
    build.add_argument("--request", type=Path, required=True)
    build.add_argument("--sha256", required=True)
    retry = sub.add_parser("retry")
    for name in ("previous-plan", "terminal"):
        retry.add_argument("--" + name, type=Path, required=True)
        retry.add_argument("--" + name + "-sha256", required=True)
    for name in ("queue-id", "attempt-id", "owner-output", "reason"):
        retry.add_argument("--" + name, required=True)
    resume = sub.add_parser("resume")
    for name in ("previous-plan", "binding", "result", "release"):
        resume.add_argument("--" + name, type=Path, required=True)
        resume.add_argument("--" + name + "-sha256", required=True)
    for name in ("queue-id", "reason"):
        resume.add_argument("--" + name, required=True)
    for command in (build, retry, resume):
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    def anchor(path, sha):
        pin = execution.pin(path.resolve())
        require(pin["sha256"] == sha, "explicit recipe input SHA mismatch")
        return pin
    if args.operation == "build":
        plan = compile_plan(anchor(args.request, args.sha256))
    elif args.operation == "retry":
        plan = compile_retry(anchor(args.previous_plan, args.previous_plan_sha256), anchor(args.terminal, args.terminal_sha256),
                             args.queue_id, args.attempt_id, args.owner_output, args.reason)
    else:
        plan = compile_resume(anchor(args.previous_plan, args.previous_plan_sha256), anchor(args.binding, args.binding_sha256),
                              anchor(args.result, args.result_sha256), anchor(args.release, args.release_sha256), args.queue_id, args.reason)
    print(canonical(write_plan(plan, args.output)))


if __name__ == "__main__":
    main()
