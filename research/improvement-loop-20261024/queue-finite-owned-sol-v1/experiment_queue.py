"""Persistent bounded DAG dispatch for a dedicated cooperating execution domain.

All command launches in store/executions must pass through tick(). Controllers run
outside that substore. Workflow admission never establishes scientific qualification.
"""

import argparse
import copy
import fcntl
import hashlib
import os
import sys
import time
from pathlib import Path

import finite_intake
import receipt_projection
import execution as executor
import ledger as queue_ledger
import run_receipts as queue_receipts
from ledger import canonical
from run_receipts import decode

SHA_ARGUMENT = "<dependency-sha256>"


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def document(path):
    return decode(Path(path).read_bytes())


def require(value, message):
    executor.require(value, message)


def current_sources():
    directory = Path(__file__).resolve().parent
    for module, filename in ((executor, "execution.py"), (queue_ledger, "ledger.py"),
                             (queue_receipts, "run_receipts.py"), (finite_intake, "finite_intake.py"), (receipt_projection, "receipt_projection.py")):
        require(Path(module.__file__).resolve() == directory / filename,
                "imported queue dependency differs from running dispatcher")
    return [executor.pin(directory / name) for name in
            ("experiment_queue.py", "execution.py", "ledger.py", "run_receipts.py", "finite_intake.py", "receipt_projection.py")]


def initialize(root, capacity):
    require(type(capacity) is int and 1 <= capacity <= 2, "capacity must be one or two")
    sources = current_sources()
    root = Path(root).resolve()
    path = root / "store.json"
    if path.exists():
        config = configuration(root)
        require(config["capacity"] == capacity, "queue capacity is immutable")
        executor.ensure_directory(root / "queues")
        config, descriptor = dispatcher(root)
        try:
            finite_intake.initialize(root)
            executor.initialize_domain(root)
        finally:
            os.close(descriptor)
        return executor.pin(path)
    executor.ensure_directory(root / "queues")
    config = dict(schema="LocalExperimentQueueStoreFiniteOwnedResearchV1", capacity=capacity, root=str(root),
                  execution_root=str(root / "executions"), qualification="unassessed",
                  sources=sources)
    try:
        executor.publish(path, config)
    except FileExistsError:
        require(configuration(root) == config, "conflicting queue initialization")
    config, descriptor = dispatcher(root)
    try:
        finite_intake.initialize(root)
        executor.initialize_domain(root)
    finally:
        os.close(descriptor)
    return executor.pin(path)


def configuration(root):
    root = Path(root).resolve()
    config = document(root / "store.json")
    require(config["schema"] == "LocalExperimentQueueStoreFiniteOwnedResearchV1" and config["root"] == str(root)
            and config["execution_root"] == str(root / "executions"), "unsupported or moved queue store")
    expected_sources = current_sources()
    require(config["sources"] == expected_sources, "queue store source identity differs from running dispatcher")
    for item in config["sources"]:
        executor.verify_pin(item)
    return config


def dispatcher(root):
    root = Path(root).resolve()
    config = configuration(root)
    descriptor = os.open(Path(root) / "dispatcher.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        executor.bounded_flock(descriptor)
    except BaseException:
        os.close(descriptor)
        raise
    return config, descriptor


def outputs_overlap(left, right):
    a,b=Path(left),Path(right)
    return a==b or a.is_relative_to(b) or b.is_relative_to(a)

def validate_plan(plan):
    require(set(plan) == {"schema", "queue_id", "jobs"} and plan["schema"] == "LocalExperimentQueuePlanV1", "unsupported queue plan")
    executor.identifier(plan["queue_id"])
    require(isinstance(plan["jobs"], list) and plan["jobs"], "nonempty fixed job list required")
    jobs = {}
    for job in plan["jobs"]:
        require(set(job) == {"id", "execution_spec", "dependencies", "outputs", "resume_prelaunch"}, "unknown job fields")
        executor.identifier(job["id"])
        require(job["id"] not in jobs, "duplicate job ID")
        jobs[job["id"]] = job
        executor.validate(job["execution_spec"])
        resume = job["resume_prelaunch"]
        require(resume is None or isinstance(resume, dict) and set(resume) == {"queue_id", "job_id", "result", "release", "binding", "reason"}
                and isinstance(resume["reason"], str) and resume["reason"].strip(), "explicit prelaunch resume proof/reason required")
        if resume:
            executor.identifier(resume["queue_id"])
            executor.identifier(resume["job_id"])
            require(not job["dependencies"], "prelaunch resumption uses already materialized exact input pins")
        require(isinstance(job["dependencies"], list) and isinstance(job["outputs"], list), "explicit dependencies/outputs required")
        roles, paths = set(), set()
        for output in job["outputs"]:
            require(set(output) == {"role", "path", "minimum_bytes"}, "invalid output declaration")
            executor.identifier(output["role"])
            name = output["path"]
            require(isinstance(name, str) and name and not Path(name).is_absolute() and ".." not in Path(name).parts,
                    "output path must stay in native output directory")
            require(output["role"] not in roles and name not in paths, "duplicate output role/path")
            require(name in {r["path"] for r in job["execution_spec"]["native_receipts"]}, "output roles must name authenticated native receipts")
            require(type(output["minimum_bytes"]) is int and output["minimum_bytes"] >= 0, "invalid output byte gate")
            roles.add(output["role"])
            paths.add(name)
    for field in ("attempt_id", "output_directory"):
        values = [job["execution_spec"][field] for job in jobs.values()]
        require(len(set(values)) == len(values), "duplicate queued " + field)
    output_paths=[j['execution_spec']['output_directory'] for j in jobs.values()]
    for i,a in enumerate(output_paths):
        for b in output_paths[i+1:]:
            require(not outputs_overlap(a,b),'overlapping output directories in plan')
    identities = [(j["execution_spec"]["cycle_id"], j["execution_spec"]["job_key"]) for j in jobs.values()]
    require(len(set(identities)) == len(identities), "one attempt per cycle/job per plan")
    visited, visiting = set(), set()
    def visit(job_id):
        require(job_id not in visiting, "dependency cycle")
        if job_id in visited:
            return
        visiting.add(job_id)
        job = jobs[job_id]
        indices, edges = set(), set()
        argv = job["execution_spec"]["argv"]
        for dependency in job["dependencies"]:
            require(set(dependency) == {"job_id", "role", "path_arguments", "sha256_arguments"}, "invalid dependency declaration")
            producer = dependency["job_id"]
            require(producer in jobs and producer != job_id, "unknown/self dependency")
            require(dependency["role"] in {o["role"] for o in jobs[producer]["outputs"]}, "unknown producer output role")
            edge = (producer, dependency["role"])
            require(edge not in edges, "duplicate dependency edge")
            edges.add(edge)
            require(isinstance(dependency["sha256_arguments"], list) and dependency["sha256_arguments"], "nonempty SHA argument indices required")
            output = next(o for o in jobs[producer]["outputs"] if o["role"] == dependency["role"])
            expected_path = str(Path(jobs[producer]["execution_spec"]["output_directory"]) / output["path"])
            require(isinstance(dependency["path_arguments"], list) and dependency["path_arguments"], "nonempty fixed path argument indices required")
            for index in dependency["path_arguments"]:
                require(type(index) is int and 0 < index < len(argv) and index not in indices and argv[index] == expected_path,
                        "dependency argv path must equal selected native receipt path")
                indices.add(index)
            for index in dependency["sha256_arguments"]:
                require(type(index) is int and 0 < index < len(argv) and index not in indices and argv[index] == SHA_ARGUMENT,
                        "SHA substitution must name one declared sentinel argument")
                indices.add(index)
            visit(producer)
        require({i for i in indices if argv[i] == SHA_ARGUMENT} == {i for i, value in enumerate(argv) if value == SHA_ARGUMENT}, "unbound SHA sentinel")
        visiting.remove(job_id)
        visited.add(job_id)
    for job_id in jobs:
        visit(job_id)
    return jobs


def entries(root, problems=None):
    root = Path(root).resolve()
    for directory in sorted((Path(root) / "queues").iterdir()):
        if directory.is_dir() and (directory / "receipt.json").is_file():
            try:
                receipt = document(directory / "receipt.json")
                require(receipt["schema"] == "LocalExperimentQueueAdmissionV1" and receipt["qualification"] == "unassessed"
                        and receipt["store"] == executor.pin(Path(root) / "store.json"), "invalid queue admission")
                require(receipt["plan"]["path"] == str(directory / "plan.json"), "admission plan path mismatch")
                plan = decode(executor.checked(receipt["plan"]))
                require(plan["schema"] == "LocalExperimentQueuePlanV1" and plan["queue_id"] == directory.name, "invalid admitted plan")
                yield directory, plan
            except (OSError, ValueError, KeyError, TypeError) as exc:
                if problems is None:
                    raise
                problems.append(dict(queue_id=directory.name, error=str(exc)))


def _enqueue_locked(root, plan, expected_sha256, config, files=True):
    root = Path(root).resolve()
    jobs = validate_plan(plan)
    require(digest(plan) == expected_sha256, "queue plan digest mismatch")
    directory = Path(root).resolve() / "queues" / plan["queue_id"]
    if directory.exists():
        if (directory / "plan.json").exists():
            require(digest(document(directory / "plan.json")) == digest(plan), "queue ID reused with changed plan")
        else:
            names = {p.name for p in directory.iterdir()}
            require(names <= {"jobs"} and (not names or (directory / "jobs").is_dir() and not any((directory / "jobs").iterdir())),
                    "partial admission contains unknown files")
        if (directory / "receipt.json").exists():
            require(document(directory / "receipt.json")["plan"] == executor.pin(directory / "plan.json"), "queue admission plan changed")
            return executor.pin(directory / "receipt.json")
    # A durable partial plan also owns its attempt/output identities.
    # entries() excludes unsealed admissions; validate those claims explicitly
    # so another ID cannot steal them before exact interrupted admission repair.
    for pending in (root / "queues").iterdir():
        require(pending.is_dir(), "unknown admission-store entry")
        if pending == directory or (pending / "receipt.json").exists():
            continue
        require((pending / "plan.json").is_file(), "unresolved admission scaffold requires repair")
        pending_plan = document(pending / "plan.json")
        pending_jobs = validate_plan(pending_plan)
        require(pending_plan["queue_id"] == pending.name, "partial admission identity mismatch")
        for job in jobs.values():
            spec = job["execution_spec"]
            require(not any(old["execution_spec"]["attempt_id"] == spec["attempt_id"] or
                            outputs_overlap(old["execution_spec"]["output_directory"], spec["output_directory"])
                            for old in pending_jobs.values()),
                    "attempt/output belongs to a partial admission; repair that exact plan first")
    old_entries = list(entries(root))
    for job in jobs.values():
        spec = job["execution_spec"]
        references = [(old_dir, old_job) for old_dir, old_plan in old_entries for old_job in old_plan["jobs"]
                      if old_job["execution_spec"]["attempt_id"] == spec["attempt_id"] or outputs_overlap(old_job["execution_spec"]["output_directory"], spec["output_directory"])]
        if job["resume_prelaunch"]:
            validate_resume(root, job, references, config, plan["queue_id"])
        else:
            require(not references, "attempt/output already belongs to another queue plan")
            require(not executor.attempt(config["execution_root"], spec["attempt_id"]).exists(), "external/preexisting attempt is not admitted")
        require(not Path(spec["output_directory"]).exists(), "queued owner output must be fresh")
        if spec["previous_attempt"]:
            previous = spec["previous_attempt"]
            state = executor.inspect(config["execution_root"], previous["attempt_id"])
            _, prior_spec, _ = executor.load(config["execution_root"], previous["attempt_id"])
            require(state["terminal_receipt"] == previous["receipt"] and
                    (prior_spec["cycle_id"], prior_spec["job_key"]) == (spec["cycle_id"], spec["job_key"]), "explicit predecessor scope/receipt mismatch")
    if not files:
        return
    finite_intake.crash("admission_before_scaffold")
    executor.ensure_directory(directory / "jobs")
    finite_intake.crash("admission_after_scaffold")
    if not (directory / "plan.json").exists():
        executor.publish(directory / "plan.json", plan)
    finite_intake.crash("admission_after_plan")
    executor.publish(directory / "receipt.json", dict(schema="LocalExperimentQueueAdmissionV1", plan=executor.pin(directory / "plan.json"),
                     store=executor.pin(Path(root) / "store.json"), qualification="unassessed", recorded_at=executor.now()))
    finite_intake.crash("admission_after_receipt")
    for job_id in jobs:
        executor.ensure_directory(directory / "jobs" / job_id)
    return executor.pin(directory / "receipt.json")


def job_result(directory):
    path = directory / "result.json"
    return document(path) if path.exists() else None


def verified_result(directory, job):
    value = job_result(directory)
    if value is None:
        require(not (directory / "release.json").exists(), "release record exists without its result")
        return None
    require(value["schema"] == "LocalQueueJobResultV1" and value["job_id"] == job["id"] and value["qualification"] == "unassessed",
            "invalid queue result identity")
    require(value["outcome"] in {"accepted_execution", "deferred_unknown_exit", "failed_timeout", "failed_process", "failed_native_outputs",
            "failed_declared_outputs", "blocked_predecessor", "deferred_prelaunch_integrity"}, "unknown queue result outcome")
    reservation = directory / "reservation.json"
    release = directory / "release.json"
    require(not release.exists() or reservation.exists(), "release record exists without its reservation")
    if reservation.exists():
        verify_binding(directory, job, verify_dependency_bytes=value["outcome"] != "deferred_prelaunch_integrity")
    if reservation.exists() and release.exists():
        binding = document(directory / "binding.json")
        require(document(reservation)["binding"] == executor.pin(directory / "binding.json") and binding["job_id"] == job["id"], "reservation binding mismatch")
        released = document(release)
        require(released["reservation"] == executor.pin(reservation) and released["result"] == executor.pin(directory / "result.json"), "release result/reservation mismatch")
        if not released.get("never_launched"):
            require(released["executor_terminal"] == value["executor_terminal"], "release terminal mismatch")
        else:
            require(value["outcome"] == "deferred_prelaunch_integrity" and value["executor_terminal"] is None, "invalid never-launched release")
    if value["outcome"] == "accepted_execution":
        require(reservation.exists() and release.exists() and value["executor_terminal"] is not None, "accepted execution needs released terminal binding")
        executor.verify_pin(value["executor_terminal"])
        require(set(value["outputs"]) == {o["role"] for o in job["outputs"]}, "result output roster mismatch")
    elif value["executor_terminal"] is None:
        require(value["outcome"] in ("blocked_predecessor", "deferred_prelaunch_integrity"), "missing executor terminal")
    if value["executor_terminal"] is not None:
        terminal = decode(executor.checked(value["executor_terminal"]))
        executor.validate_terminal(terminal, job["execution_spec"])
        require(terminal["schema"] == "LocalExecutionTerminalV1" and terminal["attempt_id"] == job["execution_spec"]["attempt_id"]
                and terminal["qualification"] == "unassessed", "queue result terminal identity mismatch")
        require(terminal["specification"]["sha256"] == document(directory / "binding.json")["specification_sha256"], "result terminal/effective specification mismatch")
        expected = "deferred_unknown_exit" if terminal["return_code"] is None else "failed_timeout" if terminal["timed_out"] else "failed_process" if terminal["return_code"] != 0 else "failed_native_outputs" if terminal["native_outputs"] != "metadata_verified_science_unassessed" else "accepted_execution"
        require(value["outcome"] == expected or expected == "accepted_execution" and value["outcome"] == "failed_declared_outputs", "queue outcome contradicts executor terminal")
    if reservation.exists():
        ownership_for_job(directory,job,required=value['executor_terminal']is not None)
    if value["outcome"] == "accepted_execution":
        declarations = {o["role"]: o for o in job["outputs"]}
        for role, output in value["outputs"].items():
            executor.verify_pin(output)
            declared = declarations[role]
            require(output["path"] == str(Path(job["execution_spec"]["output_directory"]) / declared["path"])
                    and output["size"] >= declared["minimum_bytes"], "cached output role/path/byte gate mismatch")
            require(output in terminal["native_receipts"], "role pin differs from authenticated native receipt")
    return value


def verify_binding(directory, job, verify_dependency_bytes=True):
    binding = document(directory / "binding.json")
    plan_path = directory.parent.parent / "plan.json"
    require(binding["schema"] == "LocalQueueExecutionBindingV1" and binding["plan"] == executor.pin(plan_path)
            and binding["job_id"] == job["id"] and binding["resume_prelaunch"] == job["resume_prelaunch"], "binding differs from pinned queue plan")
    require(document(directory / "reservation.json")["binding"] == executor.pin(directory / "binding.json"), "reservation binding fingerprint mismatch")
    expected = copy.deepcopy(job["execution_spec"])
    pins = {p["path"]: p for p in expected["input_pins"]}
    require(len(binding["dependencies"]) == len(job["dependencies"]), "bound dependency roster differs from plan")
    plan = document(plan_path)
    for edge, saved in zip(job["dependencies"], binding["dependencies"]):
        require(saved["edge"] == edge, "bound dependency edge differs from plan")
        producer = next(j for j in plan["jobs"] if j["id"] == edge["job_id"])
        declaration = next(o for o in producer["outputs"] if o["role"] == edge["role"])
        require(saved["output"]["path"] == str(Path(producer["execution_spec"]["output_directory"]) / declaration["path"]), "bound producer role path differs from plan")
        for key in ("producer_result", "producer_terminal", "output"):
            item = saved[key]
            if verify_dependency_bytes:
                executor.verify_pin(item)
            require(item["path"] not in pins or pins[item["path"]] == item, "conflicting frozen dependency pin")
            pins[item["path"]] = item
        for index in edge["sha256_arguments"]:
            expected["argv"][index] = saved["output"]["sha256"]
    expected["input_pins"] = list(pins.values())
    require(digest(expected) == binding["specification_sha256"] == digest(binding["execution_spec"]), "effective execution specification differs from pinned plan")
    return binding


def validate_resume(root, job, references, config, admitting_queue):
    root = Path(root).resolve()
    proof, spec = job["resume_prelaunch"], job["execution_spec"]
    source = Path(root).resolve() / "queues" / proof["queue_id"] / "jobs" / proof["job_id"]
    require(references and any(old_dir / "jobs" / old_job["id"] == source for old_dir, old_job in references), "resume must name an existing queued deferral")
    for directory in (Path(root) / "queues").iterdir():
        require(directory.name == admitting_queue or not directory.is_dir() or (directory / "receipt.json").exists(), "partial admission blocks prelaunch resumption")
    for old_dir, old_job in references:
        directory = old_dir / "jobs" / old_job["id"]
        value = verified_result(directory, old_job)
        require(value and value["outcome"] == "deferred_prelaunch_integrity" and (directory / "release.json").is_file(), "all prior queue references must be sealed prelaunch deferrals")
        require(not any((directory / name).exists() for name in ("reconciliation-error.json", "dependency-error.json")), "quarantined prior reference cannot resume")
        require(document(directory / "release.json").get("never_launched") is True, "prior queue attempt may have launched")
        require(digest(document(directory / "binding.json")["execution_spec"]) == digest(spec) and canonical(old_job["outputs"]) == canonical(job["outputs"]), "prelaunch resume specification/native roles must be identical")
    for key, name in (("result", "result.json"), ("release", "release.json"), ("binding", "binding.json")):
        require(proof[key] == executor.pin(source / name), "prelaunch resume evidence pin mismatch")
    require(not Path(spec["output_directory"]).exists(), "prelaunch resume owner output must be fresh")
    execution_dir = executor.attempt(config["execution_root"], spec["attempt_id"])
    require(not (execution_dir / "launch-intent.json").exists(), "recorded launch intent cannot resume prelaunch")
    if execution_dir.exists():
        state = executor.inspect(config["execution_root"], spec["attempt_id"])
        executor.registered(config["execution_root"], execution_dir, spec, require_leaf=True)
        _, old_spec, _ = executor.load(config["execution_root"], spec["attempt_id"])
        require(digest(old_spec) == digest(spec) and state["state"] == "prepared" and not state["job_lock_held"] and not state["session_members"]
                and not state["process_observation"], "prelaunch executor attempt is not proven unused")


def bind(directory, job, plan_pin):
    spec = copy.deepcopy(job["execution_spec"])
    pins, dependencies = {p["path"]: p for p in spec["input_pins"]}, []
    for edge in job["dependencies"]:
        producer = directory.parent / edge["job_id"]
        producer_plan = document(directory.parent.parent / "plan.json")
        producer_job = next(j for j in producer_plan["jobs"] if j["id"] == edge["job_id"])
        result = verified_result(producer, producer_job)
        require(result and result["outcome"] == "accepted_execution", "producer has no accepted execution result")
        result_pin = executor.pin(producer / "result.json")
        output = result["outputs"][edge["role"]]
        for item in (result_pin, result["executor_terminal"], output):
            executor.verify_pin(item)
            require(item["path"] not in pins or pins[item["path"]] == item, "conflicting dependency pin")
            pins[item["path"]] = item
        for index in edge["sha256_arguments"]:
            spec["argv"][index] = output["sha256"]
        dependencies.append(dict(edge=edge, producer_result=result_pin, producer_terminal=result["executor_terminal"], output=output))
    spec["input_pins"] = list(pins.values())
    binding = dict(schema="LocalQueueExecutionBindingV1", plan=plan_pin, job_id=job["id"], execution_spec=spec,
                   specification_sha256=digest(spec), dependencies=dependencies, resume_prelaunch=job["resume_prelaunch"])
    path = directory / "binding.json"
    if path.exists():
        require(document(path) == binding, "bound dependency bytes changed; new immutable plan required")
    else:
        executor.publish(path, binding)
    return binding


def result(directory, job, outcome, terminal=None, outputs=None, error=None):
    executor.publish(directory / "result.json", dict(schema="LocalQueueJobResultV1", job_id=job["id"], outcome=outcome,
                     executor_terminal=terminal, outputs=outputs or {}, error=error, qualification="unassessed", recorded_at=executor.now()))


def finish(root, directory, job, config):
    """Return True only when a reservation can durably release its capacity."""
    spec = job["execution_spec"]
    state = executor.inspect(config["execution_root"], spec["attempt_id"])
    terminal = state["attempt_terminal"]
    if not terminal or state["job_lock_held"] or state["session_members"] or any(
            o["state"] == "alive" for o in state["process_observation"].values()):
        return False
    require(terminal["cleanup"]["verified"], "terminal cleanup is unverified")
    ownership_for_job(directory,job,required=True)
    code, outputs, error = terminal["return_code"], {}, None
    outcome = "deferred_unknown_exit" if code is None else "failed_timeout" if terminal["timed_out"] else "failed_process" if code != 0 else "failed_native_outputs" if terminal["native_outputs"] != "metadata_verified_science_unassessed" else "accepted_execution"
    if outcome == "accepted_execution":
        try:
            for declared in job["outputs"]:
                path = Path(spec["output_directory"]) / declared["path"]
                require(path.resolve().is_relative_to(Path(spec["output_directory"])), "declared output escaped owner directory")
                item = executor.pin(path)
                require(item in terminal["native_receipts"], "output role does not match authenticated native receipt bytes")
                require(item["size"] >= declared["minimum_bytes"], "declared output byte gate failed")
                outputs[declared["role"]] = item
        except (OSError, ValueError) as exc:
            outcome, error = "failed_declared_outputs", str(exc)
    existing = job_result(directory)
    if existing:
        require(existing["schema"] == "LocalQueueJobResultV1" and existing["job_id"] == job["id"]
                and existing["qualification"] == "unassessed" and existing["outcome"] == outcome
                and existing["executor_terminal"] == state["terminal_receipt"] and existing["outputs"] == outputs,
                "cached result differs from verified executor outcome")
    else:
        result(directory, job, outcome, state["terminal_receipt"], outputs, error)
    executor.publish(directory / "release.json", dict(reservation=executor.pin(directory / "reservation.json"),
                     result=executor.pin(directory / "result.json"), executor_terminal=state["terminal_receipt"], recorded_at=executor.now()))
    return True


def defer_unlaunched(directory, job, config, error):
    execution_dir = executor.attempt(config["execution_root"], job["execution_spec"]["attempt_id"])
    if (execution_dir / "launch-intent.json").exists():
        return False
    if not job_result(directory):
        result(directory, job, "deferred_prelaunch_integrity", error=str(error))
    if (directory / "reservation.json").exists() and not (directory / "release.json").exists():
        executor.publish(directory / "release.json", dict(reservation=executor.pin(directory / "reservation.json"),
                         result=executor.pin(directory / "result.json"), never_launched=True, recorded_at=executor.now()))
    return True


def _tick_locked(root, config):
    root = Path(root).resolve()
    problems = []
    queues = list(entries(root, problems))
    if problems:
        return _status_snapshot(root, verify_config=False)
    if any(not (reservation.parents[2] / "receipt.json").exists()
           for reservation in (Path(root) / "queues").glob("*/jobs/*/reservation.json")):
        return _status_snapshot(root, verify_config=False)
    active = []
    # Reconcile reservations before counting capacity or selecting new work.
    for queue_dir, plan in queues:
        for job in plan["jobs"]:
            directory = queue_dir / "jobs" / job["id"]
            executor.ensure_directory(directory)
            if (directory / "reservation.json").exists():
                spec = job["execution_spec"]
                execution_dir = executor.attempt(config["execution_root"], spec["attempt_id"])
                if (directory / "release.json").exists():
                    try:
                        verified_result(directory, job)
                        continue
                    except executor.LockBusy:
                        active.append((directory,job));continue
                    except (OSError, ValueError, KeyError, TypeError) as exc:
                        active.append((directory, job))
                        error_path = directory / "reconciliation-error.json"
                        if not error_path.exists():
                            executor.publish(error_path, dict(error=str(exc), recorded_at=executor.now()))
                        continue
                active.append((directory, job))
                try:
                    cached = job_result(directory)
                    if cached and cached.get("outcome") == "deferred_prelaunch_integrity":
                        verified_result(directory, job)
                        if defer_unlaunched(directory, job, config, cached["error"]):
                            active.pop()
                            continue
                    if (execution_dir / "prepared.json").exists() and finish(root, directory, job, config):
                        active.pop()
                except executor.LockBusy:
                    continue
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    error_path = directory / "reconciliation-error.json"
                    if not error_path.exists():
                        executor.publish(error_path, dict(error=str(exc), recorded_at=executor.now()))
    if len(active) > config["capacity"]:
        return _status_snapshot(root, verify_config=False)
    for queue_dir, plan in queues:
        plan_pin = executor.pin(queue_dir / "plan.json")
        for job in plan["jobs"]:
            directory = queue_dir / "jobs" / job["id"]
            if (directory / "release.json").exists() and not (directory / "reservation.json").exists():
                error_path = directory / "reconciliation-error.json"
                if not error_path.exists():
                    executor.publish(error_path, dict(error="release record exists without its reservation", recorded_at=executor.now()))
                continue
            if job_result(directory) or (directory / "reconciliation-error.json").exists():
                continue
            try:
                dependencies = [verified_result(queue_dir / "jobs" / edge["job_id"], next(j for j in plan["jobs"] if j["id"] == edge["job_id"])) for edge in job["dependencies"]]
            except (OSError, ValueError, KeyError, TypeError) as exc:
                if defer_unlaunched(directory, job, config, exc):
                    active = [(d, j) for d, j in active if d != directory]
                    continue
                error_path = directory / "dependency-error.json"
                if not error_path.exists():
                    executor.publish(error_path, dict(error=str(exc), recorded_at=executor.now()))
                continue
            if any(d and d["outcome"] != "accepted_execution" for d in dependencies):
                result(directory, job, "blocked_predecessor")
                continue
            if any(d is None for d in dependencies):
                continue
            reserved = (directory / "reservation.json").exists()
            if not reserved and len(active) >= config["capacity"]:
                continue
            try:
                binding = bind(directory, job, plan_pin)
                if not reserved:
                    executor.publish(directory / "reservation.json", dict(schema="LocalQueueCapacityReservationV1",
                                     binding=executor.pin(directory / "binding.json"), attempt_id=job["execution_spec"]["attempt_id"], recorded_at=executor.now()))
                    active.append((directory, job))
                else:
                    require(document(directory / "reservation.json")["binding"] == executor.pin(directory / "binding.json"), "reservation binding changed")
                spec = binding["execution_spec"]
                executor.prepare(config["execution_root"], spec, binding["specification_sha256"])
                snapshot=executor.ownership_audit(config['execution_root'],ownership_expected(root))
                owned=[pin for pin in snapshot['evidence']if pin['path']==str(Path(config['execution_root'])/'ownership-intents'/(spec['attempt_id']+'.json')) or pin['path']==str(executor.attempt(config['execution_root'],spec['attempt_id'])/'prepared.json')]
                executor.publish_exact(directory/'ownership-ack.json',dict(schema='QueueOwnedPreparationAckV1',binding=executor.pin(directory/'binding.json'),reservation=executor.pin(directory/'reservation.json'),evidence=owned))
                finite_intake.crash('after_ownership_ack')
                if not (directory / "dispatch-intent.json").exists():
                    executor.publish(directory / "dispatch-intent.json", dict(reservation=executor.pin(directory / "reservation.json"), recorded_at=executor.now()))
                finite_intake.crash('before_executor_launch')
                executor.launch(config["execution_root"], spec["attempt_id"])
            except (executor.LockBusy,OSError):
                continue
            except executor.LiveInputDrift as exc:
                execution_dir=executor.attempt(config['execution_root'],job['execution_spec']['attempt_id'])
                if defer_unlaunched(directory,job,config,exc):active=[(d,j)for d,j in active if d!=directory]
            except ValueError as exc:
                error_path=directory/'dispatch-error.json'
                if not error_path.exists():executor.publish(error_path,dict(error=str(exc),recorded_at=executor.now()))
    return _status_snapshot(root, verify_config=False)


def _status_snapshot(root, verify_config=True):
    root = Path(root).resolve()
    config = configuration(root) if verify_config else document(Path(root) / "store.json")
    queues, reserved, problems = {}, 0, []
    for directory, plan in entries(root, problems):
        states = {}
        for job in plan["jobs"]:
            job_dir = directory / "jobs" / job["id"]
            error = None
            try:
                outcome = verified_result(job_dir, job)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                outcome, error = None, str(exc)
            held = (job_dir / "reservation.json").exists() and (not (job_dir / "release.json").exists() or error is not None)
            reserved += int(held)
            if (job_dir / "reconciliation-error.json").exists():
                error = document(job_dir / "reconciliation-error.json")["error"]
            if (job_dir / "dependency-error.json").exists():
                error = document(job_dir / "dependency-error.json")["error"]
            states[job["id"]] = dict(outcome="quarantined_integrity" if error else outcome["outcome"] if outcome else "reserved" if held else "waiting", error=error, qualification="unassessed")
        queues[plan["queue_id"]] = states
    partial = [p.name for p in (Path(root) / "queues").iterdir() if p.is_dir() and not (p / "receipt.json").exists()]
    unknown_reservations = sum(1 for p in (Path(root) / "queues").glob("*/jobs/*/reservation.json")
                               if p.parents[2].name in {problem["queue_id"] for problem in problems} or p.parents[2].name in partial)
    reserved += unknown_reservations
    complete = not reserved and not partial and not problems and all(j["outcome"] not in ("reserved", "waiting", "quarantined_integrity") for jobs in queues.values() for j in jobs.values())
    return dict(schema="LocalExperimentQueueObservationV1", capacity=config["capacity"], reserved=reserved, queues=queues,
                all_jobs_recorded=complete, partial_admissions=partial, quarantined_admissions=problems,
                unknown_reservations=unknown_reservations, qualification="unassessed")


def ownership_expected(root):
    values={}
    for directory,plan in entries(root):
        for job in plan['jobs']:
            leaf=directory/'jobs'/job['id'];binding=leaf/'binding.json';reservation=leaf/'reservation.json'
            if not reservation.exists():continue
            require(document(reservation)['binding']==executor.pin(binding),'ownership reservation binding drift')
            bound=document(binding);require(bound['job_id']==job['id']and bound['plan']==executor.pin(directory/'plan.json'),'ownership effective binding identity drift')
            required=(leaf/'ownership-ack.json').exists()or(leaf/'dispatch-intent.json').exists()or(executor.attempt(Path(root)/'executions',job['execution_spec']['attempt_id'])/'launch-intent.json').exists()
            key=job['execution_spec']['attempt_id'];entry=dict(spec=bound['execution_spec'],required=required)
            if key in values:require(values[key]['spec']==entry['spec'],'conflicting effective ownership specs');entry['required']|=values[key]['required']
            values[key]=entry
    return values

def ownership_for_job(directory,job,required=False):
    root=directory.parents[3];spec=document(directory/'binding.json')['execution_spec']
    snapshot=executor.ownership_audit(root/'executions',ownership_expected(root))
    ack=directory/'ownership-ack.json'
    if required:require(ack.exists(),'executed queue result lacks ownership acknowledgment')
    if ack.exists():
        value=document(ack);require(value['schema']=='QueueOwnedPreparationAckV1' and value['binding']==executor.pin(directory/'binding.json') and value['reservation']==executor.pin(directory/'reservation.json'),'ownership ack binding mismatch')
        require(len(value['evidence'])==2,'ownership ack evidence roster differs')
        expected={str(root/'executions'/'ownership-intents'/(spec['attempt_id']+'.json')),str(executor.attempt(root/'executions',spec['attempt_id'])/'prepared.json')}
        require({p['path']for p in value['evidence']}==expected,'ownership ack evidence paths differ')
        for item in value['evidence']:executor.verify_pin(item);require(item in snapshot['evidence'],'ownership acknowledgment not in audited namespace')
    return snapshot

def status(root):
    config,fd=dispatcher(root)
    try:
        finite_intake.frontier(root)
        value=_status_snapshot(root,False)
        value['intake']=finite_intake.info(root)
        value['finality']='metadata snapshot; only finalize authenticates final receipt'
        return value
    finally:os.close(fd)

def enqueue(root, plan, expected_sha256):
    validate_plan(plan)
    require(digest(plan)==expected_sha256,"queue plan digest mismatch")
    config, fd=dispatcher(root)
    try:
        return finite_intake.enqueue(root,plan,config,_enqueue_locked)
    finally:os.close(fd)

def seal(root, allow_empty=False):
    config,fd=dispatcher(root)
    try:return finite_intake.seal(root,allow_empty)
    finally:os.close(fd)

def tick(root):
    config,fd=dispatcher(root)
    try:
        finite_intake.frontier(root)
        finite_intake.require_accepted(root)
        value=_tick_locked(root,config)
        value["intake"]=finite_intake.info(root)
        return value
    finally:os.close(fd)

def finalize(root):
    config,fd=dispatcher(root)
    try:
        finite_intake.frontier(root)
        finite_intake.require_accepted(root)
        value=_tick_locked(root,config)
        snapshot=executor.ownership_audit(Path(root)/'executions',ownership_expected(root))
        value['ownership']=snapshot
        return finite_intake.finalize(root,value)
    finally:os.close(fd)

def work(root,output,poll_seconds,max_runtime):
    configuration(root)
    require(0<poll_seconds<=60 and 0<max_runtime<=604800,"invalid worker bounds")
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    deadline=time.monotonic()+max_runtime
    while True:
        state=tick(root)
        if state["intake"]["state"]!="OPEN" and state["all_jobs_recorded"]:
            final=finalize(root);status_value="finalized";break
        if time.monotonic()>=deadline:
            final=None;status_value="idle_open" if state["intake"]["state"]=="OPEN" and state["all_jobs_recorded"] else "deadline_pending";break
        time.sleep(min(poll_seconds,max(0,deadline-time.monotonic())))
    executor.publish(output/"receipt.json",dict(schema="FiniteResearchWorkerV1",status=status_value,final=final,observation=state,qualification="unassessed"))
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    sub = parser.add_subparsers(dest="operation", required=True)
    sub.add_parser("initialize").add_argument("--capacity", required=True, type=int)
    add = sub.add_parser("enqueue")
    add.add_argument("--plan", required=True, type=Path)
    add.add_argument("--sha256", required=True)
    sub.add_parser("seal").add_argument("--allow-empty",action="store_true")
    sub.add_parser("finalize")
    sub.add_parser("tick")
    sub.add_parser("status")
    worker = sub.add_parser("work")
    worker.add_argument("--output", required=True, type=Path)
    worker.add_argument("--poll-seconds", required=True, type=float)
    worker.add_argument("--max-runtime", required=True, type=float)
    args = parser.parse_args()
    if args.operation == "initialize":
        value = initialize(args.root, args.capacity)
    elif args.operation == "enqueue":
        value = enqueue(args.root, document(args.plan), args.sha256)
    elif args.operation == "seal":
        value=seal(args.root,args.allow_empty)
    elif args.operation == "work":
        value = work(args.root, args.output, args.poll_seconds, args.max_runtime)
    else:
        value = globals()[args.operation](args.root)
    print(canonical(value))


if __name__ == "__main__":
    main()
