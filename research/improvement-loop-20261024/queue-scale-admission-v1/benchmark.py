"""Bounded admission-only scale probe; it never dispatches a job or fit.

The fixture runs in a fresh temporary store and is cleaned up afterward. The
measurement covers durable queue-plan admission and read-only status, not full
scientific cycles, long-term storage, throughput under load, or unattended work.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import experiment_queue as queue  # noqa: E402
from test_experiment_queue import QueueTests  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--seconds", type=float, default=20)
    args = parser.parse_args()
    if not 1 <= args.limit <= 1000 or not 1 <= args.seconds <= 45:
        parser.error("limit must be 1..1000 and seconds 1..45")
    if args.output.exists():
        parser.error("output must be a new file")
    os.nice(10)
    test = QueueTests("runTest")
    test.setUp()
    start = time.monotonic()
    steps = []
    count = 0
    first = None
    try:
        for index in range(args.limit):
            if time.monotonic() - start >= args.seconds:
                break
            name = f"scale-{index:04d}"
            plan = {"schema": "LocalExperimentQueuePlanV1", "queue_id": name,
                    "jobs": [test.job(name)]}
            mark = time.monotonic()
            receipt = queue.enqueue(test.root, plan, queue.digest(plan))
            count += 1
            if first is None:
                first = (plan, receipt)
            if count in {1, 10, 100, 250, 500, 750, 1000} or count == args.limit:
                elapsed = time.monotonic() - start
                status_start = time.monotonic()
                status = queue.status(test.root)
                assert len(status["queues"]) == count
                assert status["reserved"] == 0 and not status["all_jobs_recorded"]
                assert not status["partial_admissions"] and not status["quarantined_admissions"]
                assert all(item["outcome"] == "waiting" and item["qualification"] == "unassessed"
                           for group in status["queues"].values() for item in group.values())
                steps.append({"admitted": count, "cumulativeSeconds": elapsed,
                              "lastAdmissionSeconds": status_start - mark,
                              "statusSeconds": time.monotonic() - status_start})
        assert first and queue.enqueue(test.root, first[0], queue.digest(first[0])) == first[1]
        assert not any((test.root / "executions" / "attempts").glob("*"))
        result = {"schema": "QueueScaleAdmissionProbeV1", "requested": args.limit,
                  "admitted": count, "deadlineSeconds": args.seconds,
                  "stoppedAtDeadline": count < args.limit, "steps": steps,
                  "identicalReadmissionPreservedReceipt": True,
                  "launchedAttempts": 0, "scienceQualification": "unassessed",
                  "scope": "Fresh local synthetic pending plans only; no scientific jobs, completed cycles, parallel clients, crash recovery, model comparisons, or deployment were tested.",
                  "sourceHashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                   [ROOT / "experiment_queue.py", ROOT / "execution.py", ROOT / "test_experiment_queue.py", Path(__file__)]}}
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"admitted": count, "stoppedAtDeadline": count < args.limit, "steps": steps}))
    finally:
        test.doCleanups()


if __name__ == "__main__":
    main()
