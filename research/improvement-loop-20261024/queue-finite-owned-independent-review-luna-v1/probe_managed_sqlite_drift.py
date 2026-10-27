"""Tiny negative probe against a temporary fixture; no production or model work."""
import os, sqlite3, sys
from pathlib import Path
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "queue-finite-owned-sol-v1"
sys.path.insert(0, str(SOURCE))
import experiment_queue as q
import execution as ex
from test_experiment_queue import QueueTests

fx = QueueTests(methodName="runTest")
fx.setUp()
try:
    root = fx.root
    plan = dict(schema="LocalExperimentQueuePlanV1", queue_id="sqlite-corrupt",
                jobs=[fx.job("job")])
    q.enqueue(root, plan, q.digest(plan))
    # Stop after reservation/preparation, before the real launch call.
    with patch.object(ex, "launch", side_effect=SystemExit("stop before child")):
        try:
            q.tick(root)
        except SystemExit:
            pass
    attempt = "job"
    launch_intent = root / "executions" / "attempts" / attempt / "launch-intent.json"
    assert not launch_intent.exists()
    # The store is isolated under QueueTests' TemporaryDirectory.
    db = sqlite3.connect(root / "intake.sqlite")
    db.execute("UPDATE intents SET state='ADMITTING' WHERE id=?", ("sqlite-corrupt",))
    db.commit()
    db.close()
    config, fd = q.dispatcher(root)
    try:
        try:
            ex.launch(root / "executions", attempt)
        except ValueError as error:
            rejection = str(error)
        else:
            raise AssertionError("managed launch accepted non-ACCEPTED lifecycle row")
    finally:
        os.close(fd)
    assert not launch_intent.exists()
    reservation = root / "queues" / "sqlite-corrupt" / "jobs" / "job" / "reservation.json"
    assert reservation.exists()
    print("negative_probe=PASS")
    print("rejection=" + rejection)
    print("launch_intent_written=false")
    print("reservation_retained=true")
finally:
    fx.doCleanups()
