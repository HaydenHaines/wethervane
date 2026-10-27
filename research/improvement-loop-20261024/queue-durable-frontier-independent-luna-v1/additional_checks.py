"""Two bounded supplemental checks for queue-durable-frontier-sol-v1.

All stores and tiny subprocess outputs live in TemporaryDirectory instances.
Run with python3 -B from any working directory; requires the adjacent research
checkout structure and uses its pinned fixture only, never an owner/scientific job.
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
FRONTIER = LAB / "queue-durable-frontier-sol-v1"
sys.path.insert(0, str(FRONTIER))
sys.path.insert(0, str(LAB))

import dispatch as d
import test_experiment_queue as fixtures


def stale_source_check():
    with tempfile.TemporaryDirectory(prefix="luna-frontier-source-") as td:
        root = d.initialize(Path(td) / "queue")
        config_path = root / "dispatch.json"
        config = json.loads(config_path.read_text())
        config["sources"][0]["sha256"] = "0" * 64
        config_path.write_text(json.dumps(config))
        try:
            d.tick(root)
        except ValueError as exc:
            assert "source identity" in str(exc), str(exc)
        else:
            raise AssertionError("stale source pin accepted")
        assert not (root / "dispatcher.lock").exists()
        return {"status": "passed", "stale_source_pin_rejected_before_lock": True}


def shared_producer_multi_role_check():
    with tempfile.TemporaryDirectory(prefix="luna-frontier-multirole-") as td:
        base = Path(td)
        root = d.initialize(base / "queue")
        fixture = fixtures.QueueTests("runTest")
        fixture.base = base
        fixture.root = root
        fixture.source = base / "owner.py"
        fixture.source.write_text(fixtures.FIXTURE)
        producer = fixture.job("producer")
        producer["execution_spec"]["native_receipts"].append(
            dict(path="second.json", required_fields=dict(schema="QueueFixtureV1", plan_sha256="fixed-plan", variant="second"))
        )
        producer["outputs"].append(dict(role="second", path="second.json", minimum_bytes=1))
        consumers = []
        for name in ("consumer-a", "consumer-b"):
            consumer = fixture.job(name, producer="producer")
            consumer["execution_spec"]["argv"].extend(
                [str(base / "output-producer" / "second.json"), d.q.SHA_ARGUMENT]
            )
            consumer["dependencies"].append(
                dict(job_id="producer", role="second", path_arguments=[6], sha256_arguments=[7])
            )
            consumers.append(consumer)
        plan = dict(schema="LocalExperimentQueuePlanV1", queue_id="multi-role-shared-producer", jobs=[*consumers, producer])
        d.enqueue(root, plan, d.q.digest(plan))
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            observation = d.tick(root)
            with d.opened(root / "admissions.sqlite") as db:
                released = db.execute("SELECT count(*) FROM releases").fetchone()[0]
                pending = db.execute("SELECT attempt,remaining,blocked FROM pending ORDER BY seq").fetchall()
            if released == 3:
                break
            time.sleep(0.02)
        assert released == 3, {"released": released, "observation": observation, "pending": pending}
        assert pending == [], pending
        bound_roles = {}
        for name in ("consumer-a", "consumer-b"):
            bound = json.loads((root / "evidence" / name / "binding.json").read_text())
            bound_roles[name] = sorted(item["edge"]["role"] for item in bound["dependencies"])
            assert bound_roles[name] == ["receipt", "second"], bound_roles[name]
        with d.locked(root) as (_, db):
            audit = d.audit_frontier(db)
        assert audit["released"] == 3 and audit["pending"] == 0 and audit["active"] == 0, audit
        fixture.cleanup()
        return {
            "status": "passed",
            "producer_consumer_edges": 1,
            "producer_roles_per_consumer": bound_roles,
            "shared_producer_consumers": 2,
            "audit": audit,
            "stores": "TemporaryDirectory",
        }


if __name__ == "__main__":
    result = {
        "schema": "LunaFrontierAdditionalChecksV1",
        "source": str(Path(__file__).resolve()),
        "checks": {
            "stale_source_pin": stale_source_check(),
            "shared_producer_multiple_roles": shared_producer_multi_role_check(),
        },
        "scope": "Invented local fixtures only; no owner stores or scientific work.",
    }
    out = HERE / "additional_checks_result.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
