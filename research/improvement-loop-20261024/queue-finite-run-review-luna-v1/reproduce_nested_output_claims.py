"""Admission-only test for output path containment in a selected queue source."""
import sys
from pathlib import Path
source = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(source))
import experiment_queue as queue
from test_experiment_queue import QueueTests
fixture = QueueTests(methodName="runTest")
fixture.setUp()
try:
    outer = fixture.base / "shared-output" / "parent"
    inner = outer / "child"
    for name, output in (("parent", outer), ("child", inner)):
        job = fixture.job(name)
        job["execution_spec"]["output_directory"] = str(output)
        job["execution_spec"]["argv"][2] = str(output)
        plan = {"schema": "LocalExperimentQueuePlanV1", "queue_id": "q-" + name, "jobs": [job]}
        queue.enqueue(fixture.root, plan, queue.digest(plan))
        print(name, "accepted", str(output))
    assert not outer.exists() and not inner.exists()
    print("both_nonexistent_output_roots_accepted_before_launch", True)
finally:
    fixture.doCleanups()
