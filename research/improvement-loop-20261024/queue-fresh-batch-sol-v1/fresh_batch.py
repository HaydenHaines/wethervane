"""Pinned admission of independent, already known fresh plans; never dispatches.

One cooperative dispatcher lock and one authenticated history scan per batch.
Interrupted admission is a committed prefix, not an atomic batch transaction.
Recover individual plans with the unchanged experiment_queue.enqueue API.
"""
import argparse
import copy
import os
from pathlib import Path
import sys

CORE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE))
import experiment_queue as queue
import execution as executor

MAX_PLANS = 1000


def sources():
    return [executor.pin(path) for path in [Path(__file__).resolve()] +
            [CORE / name for name in ("experiment_queue.py", "execution.py", "ledger.py", "run_receipts.py")]]


def no_symlinks(path):
    path = Path(path).absolute()
    for part in [path, *path.parents]:
        queue.require(not part.is_symlink(), "symlink path is not admitted")


def absent(path):
    no_symlinks(path)
    queue.require(not os.path.lexists(path), "fresh batch path already exists")


def trusted(pin):
    no_symlinks(pin["path"])
    executor.verify_pin(pin)


def admit(root, manifest_path, expected_sha256):
    """Expected manifest byte SHA must come from an independently trusted caller."""
    no_symlinks(root)
    root = Path(root).resolve()
    manifest_path = Path(manifest_path).absolute()
    no_symlinks(manifest_path)
    manifest_pin = executor.pin(manifest_path)
    queue.require(manifest_pin["sha256"] == expected_sha256, "untrusted manifest digest")
    trusted(manifest_pin)
    manifest = copy.deepcopy(queue.decode(executor.checked(manifest_pin)))
    queue.require(set(manifest) == {"schema", "batch_id", "store", "sources", "plans"}
                  and manifest["schema"] == "FreshQueueBatchPlanV1", "unsupported batch manifest")
    executor.identifier(manifest["batch_id"])
    queue.require(manifest["sources"] == sources(), "exact trusted batch/core source roster required")
    queue.require(isinstance(manifest["plans"], list) and 1 <= len(manifest["plans"]) <= MAX_PLANS,
                  "bounded nonempty fixed plan roster required")
    for pin in manifest["sources"]:
        trusted(pin)
    no_symlinks(root / "dispatcher.lock")
    config, descriptor = queue.dispatcher(root)
    try:
        queue.require(manifest["store"] == executor.pin(root / "store.json"), "batch store pin mismatch")
        trusted(manifest["store"])
        no_symlinks(root / "dispatcher.lock")
        no_symlinks(root / "queues")
        destination = root / "batch-admissions" / manifest["batch_id"]
        absent(destination)
        # Partial/quarantined admission is recovery work, never silently skipped.
        for directory in (root / "queues").iterdir():
            no_symlinks(directory)
            queue.require(directory.is_dir() and (directory / "receipt.json").is_file()
                          and (directory / "plan.json").is_file() and (directory / "jobs").is_dir(),
                          "malformed/partial admission requires ordinary recovery")
            for name in ("receipt.json", "plan.json", "jobs"):
                no_symlinks(directory / name)
        history = list(queue.entries(root))
        attempts, outputs = set(), set()
        for _, plan in history:
            queue.require(isinstance(plan["jobs"], list) and plan["jobs"], "malformed historical job roster")
            for job in plan["jobs"]:
                # Historical deferred/retried ownership may appear more than once.
                spec = job["execution_spec"]
                executor.identifier(spec["attempt_id"])
                queue.require(isinstance(spec["output_directory"], str)
                              and Path(spec["output_directory"]).is_absolute(), "malformed historical output")
                attempts.add(spec["attempt_id"])
                outputs.add(spec["output_directory"])
        plans, queue_ids = [], set()
        for pin in manifest["plans"]:
            trusted(pin)
            plan = copy.deepcopy(queue.decode(executor.checked(pin)))
            jobs = queue.validate_plan(plan)
            queue.require(plan["queue_id"] not in queue_ids, "duplicate batch queue ID")
            queue_ids.add(plan["queue_id"])
            absent(root / "queues" / plan["queue_id"])
            for job in jobs.values():
                spec = job["execution_spec"]
                queue.require(job["resume_prelaunch"] is None and spec["previous_attempt"] is None,
                              "resume/retry requires ordinary enqueue")
                queue.require(spec["attempt_id"] not in attempts and spec["output_directory"] not in outputs,
                              "attempt/output already belongs to another queue plan")
                absent(executor.attempt(config["execution_root"], spec["attempt_id"]))
                absent(spec["output_directory"])
                attempts.add(spec["attempt_id"])
                outputs.add(spec["output_directory"])
            plans.append((pin, plan, jobs))
        # No admission writes until the complete roster has passed preflight.
        trusted(manifest_pin)
        for pin in manifest["sources"]:
            trusted(pin)
        executor.ensure_directory(destination)
        executor.publish(destination / "intent.json", dict(schema="FreshQueueBatchIntentV1",
                         manifest=manifest_pin, plan=manifest, qualification="unassessed",
                         recorded_at=executor.now()))
        receipts = []
        for pin, plan, jobs in plans:
            trusted(manifest_pin)
            trusted(pin)
            for source in manifest["sources"]:
                trusted(source)
            # Cooperative clients cannot mutate while locked; recheck filesystem
            # freshness and pinned inputs immediately before each admission.
            for job in jobs.values():
                executor.validate(job["execution_spec"])
                absent(executor.attempt(config["execution_root"], job["execution_spec"]["attempt_id"]))
                absent(job["execution_spec"]["output_directory"])
            directory = root / "queues" / plan["queue_id"]
            absent(directory)
            executor.ensure_directory(directory / "jobs")
            executor.publish(directory / "plan.json", plan)
            executor.publish(directory / "receipt.json", dict(schema="LocalExperimentQueueAdmissionV1",
                             plan=executor.pin(directory / "plan.json"), store=executor.pin(root / "store.json"),
                             qualification="unassessed", recorded_at=executor.now()))
            for job_id in jobs:
                executor.ensure_directory(directory / "jobs" / job_id)
            receipts.append(executor.pin(directory / "receipt.json"))
        executor.publish(destination / "receipt.json", dict(schema="FreshQueueBatchAdmissionV1",
                         intent=executor.pin(destination / "intent.json"), admissions=receipts,
                         qualification="unassessed", recorded_at=executor.now()))
        return executor.pin(destination / "receipt.json")
    finally:
        os.close(descriptor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sha256", required=True, help="independently trusted manifest byte SHA256")
    args = parser.parse_args()
    print(queue.canonical(admit(args.root, args.manifest, args.sha256)))
