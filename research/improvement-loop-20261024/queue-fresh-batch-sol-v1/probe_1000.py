"""Bounded batch-only synthetic admission on the separate research volume."""
import argparse
import json
import os
import resource
import shutil
import time
from pathlib import Path
from unittest.mock import patch

import fresh_batch as batch
import experiment_queue as queue
import execution as executor
import test_experiment_queue as fixtures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    base = args.directory.absolute()
    queue.require(base.is_relative_to(Path("/mnt/research-storage")) and not base.exists(),
                  "new separate-volume probe directory required")
    queue.require(not args.output.exists(), "new evidence output required")
    batch.no_symlinks(base)
    queue.require(shutil.disk_usage(base.parent).free >= 512 * 1024**2, "512MiB free-space gate")
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
    resource.setrlimit(resource.RLIMIT_CPU, (45, 45))
    os.nice(19)
    base.mkdir()
    fixture = fixtures.QueueTests("runTest")
    fixture.base, fixture.root, fixture.source = base, base / "queue", base / "owner.py"
    fixture.source.write_text(fixtures.FIXTURE)
    queue.initialize(fixture.root, 2)
    (base / "plans").mkdir()
    pins = []
    for i in range(1000):
        plan = dict(schema="LocalExperimentQueuePlanV1", queue_id="fixed-" + str(i),
                    jobs=[fixture.job("fixed-" + str(i))])
        path = base / "plans" / (str(i) + ".json")
        path.write_text(json.dumps(plan))
        pins.append(executor.pin(path))
    manifest = dict(schema="FreshQueueBatchPlanV1", batch_id="fixed-1000",
                    store=executor.pin(fixture.root / "store.json"), sources=batch.sources(), plans=pins)
    path = base / "manifest.json"
    path.write_text(json.dumps(manifest))
    manifest_pin = executor.pin(path)
    start = time.monotonic()
    with patch.object(queue, "entries", wraps=queue.entries) as scan:
        receipt = batch.admit(fixture.root, path, manifest_pin["sha256"])
    elapsed = time.monotonic() - start
    result = queue.document(receipt["path"])
    assert len(result["admissions"]) == 1000 and scan.call_count == 1
    for pin in result["admissions"]:
        executor.verify_pin(pin)
    assert not (fixture.root / "executions").exists()
    evidence = dict(schema="FreshQueueBatchThousandProbeV1", plans=1000,
                    batchAdmissionSeconds=elapsed, historyScans=scan.call_count,
                    peakRssKiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    nice=os.getpriority(os.PRIO_PROCESS, 0), cpuAffinity=sorted(os.sched_getaffinity(0)),
                    root=str(base), manifest=manifest_pin, batchReceipt=receipt,
                    batchIntent=executor.pin(fixture.root / "batch-admissions/fixed-1000/intent.json"),
                    sources=batch.sources() + [executor.pin(Path(__file__)),
                             executor.pin(Path(fixtures.__file__)),
                             executor.pin(Path(__file__).with_name("test_fresh_batch.py")),
                             executor.pin(Path(__file__).with_name("test_post_receipt_crash.py"))],
                    qualification="unassessed", launchedAttempts=0,
                    scope="1000 already known independent synthetic pending plans, one batch, separate research volume. "
                    "No normal-queue1000 run, adaptive sequential cycles, scientific jobs, dispatch, or production stores. "
                    "Synthetic store retained as evidence.")
    args.output.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(dict(plans=1000, seconds=elapsed, scans=scan.call_count,
                          peakRssKiB=evidence["peakRssKiB"], output=executor.pin(args.output))))
