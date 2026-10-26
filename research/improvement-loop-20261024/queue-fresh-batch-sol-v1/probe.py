"""Matched bounded fresh-store admission probe; no dispatch, no scientific jobs."""
import argparse
import json
import time
from pathlib import Path
from unittest.mock import patch

import fresh_batch as batch
import experiment_queue as queue
import execution as executor
import test_experiment_queue as fixtures


def measure(count, batched):
    fixture = fixtures.QueueTests("runTest")
    fixture.setUp()
    try:
        pins, plans = [], []
        for i in range(count):
            plan = dict(schema="LocalExperimentQueuePlanV1", queue_id="probe-" + str(i),
                        jobs=[fixture.job("probe-" + str(i))])
            path = fixture.base / ("plan-" + str(i) + ".json")
            path.write_text(json.dumps(plan))
            pins.append(executor.pin(path))
            plans.append(plan)
        manifest = dict(schema="FreshQueueBatchPlanV1", batch_id="probe",
                        store=executor.pin(fixture.root / "store.json"), sources=batch.sources(), plans=pins)
        path = fixture.base / "manifest.json"
        path.write_text(json.dumps(manifest))
        sha = executor.pin(path)["sha256"]
        start = time.monotonic()
        with patch.object(queue, "entries", wraps=queue.entries) as scan:
            if batched:
                batch.admit(fixture.root, path, sha)
            else:
                for plan in plans:
                    queue.enqueue(fixture.root, plan, queue.digest(plan))
        seconds = time.monotonic() - start
        assert len(list(queue.entries(fixture.root))) == count
        assert not (fixture.root / "executions").exists()
        return dict(plans=count, seconds=seconds, historyScans=scan.call_count, launchedAttempts=0)
    finally:
        fixture.doCleanups()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", type=int, default=250)
    args = parser.parse_args()
    if not 1 <= args.count <= 250 or args.output.exists():
        parser.error("count must be1..250 and output must be new")
    result = dict(schema="FreshQueueBatchBoundedProbeV1", batch=measure(args.count, True),
                  ordinary=measure(args.count, False), qualification="unassessed",
                  sources=batch.sources() + [executor.pin(Path(__file__)),
                           executor.pin(Path(__file__).with_name("test_fresh_batch.py"))],
                  scope="Two fresh isolated temporary stores; fixed known independent pending plans only. "
                  "No adaptive cycles, models, production stores, dispatch, or scientific qualification.")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
