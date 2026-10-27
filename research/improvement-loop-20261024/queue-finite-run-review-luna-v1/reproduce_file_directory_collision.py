"""Admission-only collision: one owner's output file is another's output directory."""
import sys
from pathlib import Path
source = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(source))
import experiment_queue as queue
from test_experiment_queue import QueueTests
fixture = QueueTests(methodName="runTest")
fixture.setUp()
try:
    shared = fixture.base / "shared-output"
    first_file = shared / "data.txt"
    for name, output in (("writer", shared), ("nested", first_file)):
        job = fixture.job(name)
        job["execution_spec"]["output_directory"] = str(output)
        job["execution_spec"]["argv"][2] = str(output)
        plan = {"schema": "LocalExperimentQueuePlanV1", "queue_id": "q-" + name, "jobs": [job]}
        queue.enqueue(fixture.root, plan, queue.digest(plan))
        print(name, "accepted", str(output))
    assert not shared.exists()
    print("first_fixture_writes_data_txt_as_file", True)
    print("second_fixture_requires_data_txt_as_directory", True)
finally:
    fixture.doCleanups()
