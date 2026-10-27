"""Test whether a pending ADMITTING claim blocks a nested output directory."""
import json, os, subprocess, sys
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
    def make(name, output):
        job = fixture.job(name)
        job["execution_spec"]["output_directory"] = str(output)
        job["execution_spec"]["argv"][2] = str(output)
        return {"schema": "LocalExperimentQueuePlanV1", "queue_id": "q-" + name, "jobs": [job]}
    first = make("parent", outer)
    code = "import json,sys;from pathlib import Path;import experiment_queue as q;r=Path(sys.argv[1]);p=json.loads(sys.argv[2]);q.enqueue(r,p,q.digest(p))"
    env = dict(os.environ, PYTHONPATH=str(source), FINITE_QUEUE_CRASH="intent_after_commit", PYTHONDONTWRITEBYTECODE="1")
    crash = subprocess.run([sys.executable, "-B", "-c", code, str(fixture.root), json.dumps(first)], env=env, capture_output=True)
    assert crash.returncode == 91, crash.returncode
    print("first_plan_left_admitting_intent", True)
    second = make("child", inner)
    queue.enqueue(fixture.root, second, queue.digest(second))
    state = queue.status(fixture.root)["intake"]
    print("nested_second_accepted_with_pending_first", state["intents"] == 2 and state["admitting"] == 1)
    assert state["intents"] == 2 and state["admitting"] == 1
    print("no_output_paths_materialized", not outer.exists() and not inner.exists())
finally:
    fixture.doCleanups()
