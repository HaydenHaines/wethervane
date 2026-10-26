"""Real invented subprocess DAGs test durable dispatch; no scientific jobs."""

import copy
import contextlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import execution as executor
import experiment_queue as queue

FIXTURE = r'''
import hashlib,json,pathlib,sys,time
output,mode=pathlib.Path(sys.argv[1]),sys.argv[2]
output.mkdir()
value=mode
if len(sys.argv)>3:
    data=pathlib.Path(sys.argv[3]).read_bytes()
    assert hashlib.sha256(data).hexdigest()==sys.argv[4], "dependency SHA argument differs"
    receipt=json.loads(data)
    pin=receipt["data"]
    raw=pathlib.Path(pin["path"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==pin["sha256"]
    value=raw.decode()+":consumed"
if mode!="missing_data": (output/"data.txt").write_text(value)
if mode!="missing_receipt":
    receipt={"schema":"QueueFixtureV1","plan_sha256":"wrong" if mode=="wrong_metadata" else "fixed-plan","status":"computed","data":{"path":str(output/"data.txt"),"sha256":hashlib.sha256(value.encode()).hexdigest(),"size":len(value.encode())}}
    (output/"receipt.json").write_text(json.dumps(receipt))
    (output/"second.json").write_text(json.dumps(dict(receipt,variant="second")))
if mode=="hold": time.sleep(60)
if mode=="nonzero": sys.exit(7)
'''


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "queue"
        self.source = self.base / "owner.py"
        self.source.write_text(FIXTURE)
        queue.initialize(self.root, 2)
        self.addCleanup(self.cleanup)

    def cleanup(self):
        stores = [self.root / "executions", self.base / "controllers"]
        for store in stores:
            for path in store.glob("attempts/*/supervisor.json"):
                identity = json.loads(path.read_text())["identity"]
                if executor.observe(identity)["state"] == "alive":
                    fd = executor.open_pidfd(identity)
                    signal.pidfd_send_signal(fd, signal.SIGKILL)
                    os.close(fd)
            for path in store.glob("attempts/*/child.json"):
                identity = json.loads(path.read_text())["identity"]
                if executor.observe(identity)["state"] == "alive":
                    fd = executor.open_pidfd(identity)
                    signal.pidfd_send_signal(fd, signal.SIGKILL)
                    os.close(fd)

    def wait(self, predicate, timeout=10):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            time.sleep(.02)
        logs = [p.read_text() for p in self.root.glob("executions/attempts/*/supervisor.log")]
        self.fail("queue fixture timeout: " + repr(logs))

    def job(self, name, mode="seed", producer=None):
        output = self.base / ("output-" + name)
        argv = [sys.executable, str(self.source), str(output), mode]
        dependencies = []
        if producer:
            argv.extend([str(self.base / ("output-" + producer) / "receipt.json"), queue.SHA_ARGUMENT])
            dependencies = [dict(job_id=producer, role="receipt", path_arguments=[4], sha256_arguments=[5])]
        spec = dict(schema="LocalExecutionSpecV1", cycle_id="fixture-cycle", job_key=name, attempt_id=name,
                    argv=argv, executable=executor.executable_pin(sys.executable), cwd=str(self.base), environment={},
                    input_pins=[executor.pin(self.source)], output_directory=str(output),
                    native_receipts=[dict(path="receipt.json", required_fields=dict(schema="QueueFixtureV1", plan_sha256="fixed-plan"))],
                    timeout_seconds=10, grace_seconds=.05, previous_attempt=None, retry_reason="")
        return dict(id=name, execution_spec=spec, dependencies=dependencies, outputs=[dict(role="receipt", path="receipt.json", minimum_bytes=1)], resume_prelaunch=None)

    def admit(self, jobs, name="q0"):
        plan = dict(schema="LocalExperimentQueuePlanV1", queue_id=name, jobs=jobs)
        queue.enqueue(self.root, plan, queue.digest(plan))
        return plan

    def result(self, job, name="q0"):
        return queue.document(self.root / "queues" / name / "jobs" / job / "result.json")

    def run_queue(self):
        def done():
            state = queue.tick(self.root)
            return state if state["all_jobs_recorded"] and not state["reserved"] else None
        return self.wait(done)

    def stop_orphan(self, attempt_id):
        store = self.root / "executions"
        directory = executor.attempt(store, attempt_id)
        self.wait(lambda: (directory / "exec-intent.json").exists())
        supervisor = queue.document(directory / "supervisor.json")["identity"]
        fd = executor.open_pidfd(supervisor)
        signal.pidfd_send_signal(fd, signal.SIGKILL)
        os.close(fd)
        self.wait(lambda: executor.observe(supervisor)["state"] != "alive")
        executor.stop(store, attempt_id)
        self.wait(lambda: not executor.inspect(store, attempt_id)["job_lock_held"])
        return executor.recover(store, attempt_id)

    def test_killed_partial_admission_keeps_ownership_and_allows_unrelated_work(self):
        """A durable plan must retain its claim even when its writer dies before the receipt."""
        original = dict(schema="LocalExperimentQueuePlanV1", queue_id="original", jobs=[self.job("first")])
        plan_file = self.base / "admission-plan.json"
        plan_file.write_text(json.dumps(original))
        code = '''
import json, os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import experiment_queue as q
root = Path(sys.argv[2]); plan = json.loads(Path(sys.argv[3]).read_text())
publish = q.executor.publish
def die_after_plan(path, value):
    result = publish(path, value)
    if Path(path) == root / "queues" / plan["queue_id"] / "plan.json":
        os._exit(82)
    return result
q.executor.publish = die_after_plan
q.enqueue(root, plan, q.digest(plan))
'''
        child = subprocess.run([sys.executable, "-B", "-c", code, str(Path(queue.__file__).parent), str(self.root), str(plan_file)],
                               capture_output=True, timeout=10)
        self.assertEqual(child.returncode, 82, child.stderr.decode())
        directory = self.root / "queues" / "original"
        self.assertTrue((directory / "plan.json").exists())
        self.assertFalse((directory / "receipt.json").exists())
        for field in ("attempt_id", "output_directory"):
            contender = dict(schema="LocalExperimentQueuePlanV1", queue_id="collision-" + field, jobs=[self.job("other")])
            contender["jobs"][0]["execution_spec"][field] = original["jobs"][0]["execution_spec"][field]
            with self.assertRaisesRegex(ValueError, "partial admission"):
                queue.enqueue(self.root, contender, queue.digest(contender))
        unrelated = dict(schema="LocalExperimentQueuePlanV1", queue_id="unrelated", jobs=[self.job("second")])
        queue.enqueue(self.root, unrelated, queue.digest(unrelated))
        queue.enqueue(self.root, original, queue.digest(original))
        state = self.run_queue()
        self.assertEqual(state["partial_admissions"], [])
        self.assertEqual(self.result("first", "original")["outcome"], "accepted_execution")
        self.assertEqual(self.result("second", "unrelated")["outcome"], "accepted_execution")
        self.assertEqual(len(list((self.root / "executions/attempts").glob("*/launch-intent.json"))), 2)

    def test_release_without_result_never_frees_live_capacity(self):
        """Corrupt release evidence cannot release a running job's reservation."""
        self.root = self.base / "capacity-one"
        queue.initialize(self.root, 1)
        self.admit([self.job("first", "hold"), self.job("second")])
        queue.tick(self.root)
        execution = executor.attempt(self.root / "executions", "first")
        child = self.wait(lambda: queue.document(execution / "child.json") if (execution / "child.json").exists() else None)
        self.wait(lambda: executor.observe(child["identity"])["state"] == "alive")
        first = self.root / "queues/q0/jobs/first"
        self.assertFalse((first / "result.json").exists())
        (first / "release.json").write_text('{"forged":true}\n')
        state = queue.tick(self.root)
        self.assertEqual(state["reserved"], 1)
        self.assertEqual(state["queues"]["q0"]["first"]["outcome"], "quarantined_integrity")
        self.assertIn("release record exists without its result", state["queues"]["q0"]["first"]["error"])
        self.assertFalse((executor.attempt(self.root / "executions", "second") / "launch-intent.json").exists())
        self.assertEqual(executor.observe(child["identity"])["state"], "alive")

    def test_release_without_reservation_is_not_a_valid_result(self):
        job = self.job("orphan")
        directory = self.base / "orphan-result"
        directory.mkdir()
        queue.result(directory, job, "blocked_predecessor")
        (directory / "release.json").write_text('{}\n')
        with self.assertRaisesRegex(ValueError, "release record exists without its reservation"):
            queue.verified_result(directory, job)

    def test_store_versions_cannot_dispatch_each_others_plans(self):
        """Different code versions cannot silently operate on each other's pinned stores."""
        current = Path(queue.__file__).resolve()
        original = current.parent.parent / "experiment_queue.py"
        old_root = self.base / "old-queue"
        initialized = subprocess.run([sys.executable, "-B", str(original), "--root", str(old_root), "initialize", "--capacity", "1"],
                                     capture_output=True, text=True, timeout=10)
        self.assertEqual(initialized.returncode, 0, initialized.stderr)
        plan = dict(schema="LocalExperimentQueuePlanV1", queue_id="cross-version", jobs=[self.job("cross")])
        plan_file = self.base / "cross-version.json"
        plan_file.write_text(json.dumps(plan))
        digest = queue.digest(plan)
        for module, store in ((current, old_root), (original, self.root)):
            before = (store / "store.json").read_bytes()
            failed = subprocess.run([sys.executable, "-B", str(module), "--root", str(store), "enqueue",
                                     "--plan", str(plan_file), "--sha256", digest],
                                    capture_output=True, text=True, timeout=10)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("unsupported or moved queue store", failed.stderr)
            self.assertEqual((store / "store.json").read_bytes(), before)
            self.assertFalse((store / "queues" / "cross-version").exists())
        (old_root / "queues").rmdir()
        failed_init = subprocess.run([sys.executable, "-B", str(current), "--root", str(old_root), "initialize", "--capacity", "1"],
                                     capture_output=True, text=True, timeout=10)
        self.assertNotEqual(failed_init.returncode, 0)
        self.assertIn("unsupported or moved queue store", failed_init.stderr)
        self.assertFalse((old_root / "queues").exists())
        worker_output = self.base / "cross-version-worker"
        failed_work = subprocess.run([sys.executable, "-B", str(current), "--root", str(old_root), "work",
                                      "--output", str(worker_output), "--poll-seconds", ".01", "--max-runtime", "1"],
                                     capture_output=True, text=True, timeout=10)
        self.assertNotEqual(failed_work.returncode, 0)
        self.assertIn("unsupported or moved queue store", failed_work.stderr)
        self.assertFalse(worker_output.exists())

    def test_v3_rejects_valid_pins_for_a_different_dispatcher_path(self):
        config_path = self.root / "store.json"
        config = json.loads(config_path.read_text())
        original = Path(queue.__file__).resolve().parent.parent / "experiment_queue.py"
        config["sources"][0] = executor.pin(original)
        config_path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "source identity differs"):
            queue.configuration(self.root)

    def test_v3_rejects_dependencies_preloaded_from_another_version(self):
        code = '''
import importlib, importlib.util, sys
from pathlib import Path
old, new, name, root = sys.argv[1:]
sys.path.insert(0, old)
importlib.import_module(name)
sys.path.insert(0, new)
spec = importlib.util.spec_from_file_location("queue_v2_mixed", Path(new) / "experiment_queue.py")
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)
q.initialize(Path(root), 1)
'''
        current = Path(queue.__file__).resolve().parent
        for dependency in ("execution", "ledger", "run_receipts"):
            with self.subTest(dependency=dependency):
                root = self.base / ("mixed-" + dependency)
                result = subprocess.run([sys.executable, "-B", "-c", code, str(current.parent), str(current), dependency, str(root)],
                                        capture_output=True, text=True, timeout=10)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("imported queue dependency differs", result.stderr)
                self.assertFalse(root.exists())

    def test_v2_and_v3_store_schemas_cannot_be_mixed(self):
        current = Path(queue.__file__).resolve()
        older = current.parent.parent / "queue-v2" / "experiment_queue.py"
        older_root = self.base / "v2-queue"
        initialized = subprocess.run([sys.executable, "-B", str(older), "--root", str(older_root), "initialize", "--capacity", "1"],
                                     capture_output=True, text=True, timeout=10)
        self.assertEqual(initialized.returncode, 0, initialized.stderr)
        for module, store in ((current, older_root), (older, self.root)):
            before = (store / "store.json").read_bytes()
            rejected = subprocess.run([sys.executable, "-B", str(module), "--root", str(store), "status"],
                                      capture_output=True, text=True, timeout=10)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("unsupported or moved queue store", rejected.stderr)
            self.assertEqual((store / "store.json").read_bytes(), before)

    def test_real_dependency_chain_consumes_exact_output_and_sha_argument(self):
        self.admit([self.job("producer"), self.job("consumer", producer="producer"), self.job("independent")])
        state = queue.work(self.root, self.base / "worker", .02, 8)
        self.assertTrue(state["all_jobs_recorded"])
        self.assertEqual((self.base / "output-consumer/data.txt").read_text(), "seed:consumed")
        producer = self.result("producer")
        binding = queue.document(self.root / "queues/q0/jobs/consumer/binding.json")
        self.assertEqual(binding["execution_spec"]["argv"][5], producer["outputs"]["receipt"]["sha256"])
        self.assertIn(producer["outputs"]["receipt"], binding["execution_spec"]["input_pins"])
        self.assertIn(producer["executor_terminal"], binding["execution_spec"]["input_pins"])
        self.assertTrue(all(j["qualification"] == "unassessed" for jobs in state["queues"].values() for j in jobs.values()))

    def test_competing_killed_clients_share_capacity_across_two_queues(self):
        self.admit([self.job("a", "hold"), self.job("b", "hold")], "one")
        self.admit([self.job("c", "hold"), self.job("d", "hold")], "two")
        code = "import sys,os,signal;sys.path.insert(0,sys.argv[1]);import experiment_queue as q;q.tick(sys.argv[2]);os.kill(os.getpid(),signal.SIGKILL)"
        clients = [subprocess.Popen([sys.executable, "-B", "-c", code, str(Path(queue.__file__).parent), str(self.root)],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(2)]
        for client in clients:
            stdout, stderr = client.communicate(timeout=10)
            self.assertEqual(client.returncode, -signal.SIGKILL, (stdout, stderr))
        self.wait(lambda: len(list((self.root / "executions").glob("attempts/*/child.json"))) == 2)
        state = queue.tick(self.root)
        self.assertEqual(state["reserved"], 2)
        self.assertEqual(len(list((self.root / "executions").glob("attempts/*/launch-intent.json"))), 2)
        self.assertEqual(len(list(self.root.glob("queues/*/jobs/*/reservation.json"))), 2)

    def test_resume_same_reservation_after_prepare_and_launch_crashes(self):
        for index, phase in enumerate(("before_prepare", "after_prepare", "after_launch")):
            name = "phase" + str(index)
            self.admit([self.job(name)], name)
            original_prepare, original_launch = executor.prepare, executor.launch
            def prepare(*args):
                if phase == "before_prepare":
                    raise SystemExit("invented caller crash")
                value = original_prepare(*args)
                if phase == "after_prepare":
                    raise SystemExit("invented caller crash")
                return value
            def launch(*args):
                value = original_launch(*args)
                raise SystemExit("invented caller crash")
            with patch.object(executor, "prepare", side_effect=prepare), patch.object(executor, "launch", side_effect=launch if phase == "after_launch" else original_launch):
                with self.assertRaises(SystemExit):
                    queue.tick(self.root)
            reservation = executor.pin(self.root / "queues" / name / "jobs" / name / "reservation.json")
            self.run_queue()
            self.assertEqual(reservation, executor.pin(reservation["path"]))
            self.assertEqual(self.result(name, name)["outcome"], "accepted_execution")
            self.assertEqual(len(list((self.root / "executions" / "attempts" / name).glob("launch-intent.json"))), 1)

    def test_unknown_live_remains_reserved_then_recovered_unknown_blocks_downstream(self):
        self.admit([self.job("producer", "hold"), self.job("consumer", producer="producer")])
        queue.tick(self.root)
        directory = self.root / "executions/attempts/producer"
        self.wait(lambda: (directory / "exec-intent.json").exists() and (self.base / "output-producer/receipt.json").exists())
        supervisor = queue.document(directory / "supervisor.json")["identity"]
        fd = executor.open_pidfd(supervisor)
        signal.pidfd_send_signal(fd, signal.SIGKILL)
        os.close(fd)
        self.wait(lambda: executor.observe(supervisor)["state"] != "alive")
        self.assertEqual(queue.tick(self.root)["reserved"], 1)
        self.assertFalse((self.root / "executions/attempts/consumer").exists())
        executor.stop(self.root / "executions", "producer")
        self.wait(lambda: not executor.inspect(self.root / "executions", "producer")["job_lock_held"])
        executor.recover(self.root / "executions", "producer")
        state = queue.tick(self.root)
        self.assertEqual(state["reserved"], 0)
        self.assertEqual(self.result("producer")["outcome"], "deferred_unknown_exit")
        self.assertEqual(self.result("consumer")["outcome"], "blocked_predecessor")

    def test_nonzero_invalid_native_and_declared_output_gates_block_dependents(self):
        jobs = []
        for index, mode in enumerate(("nonzero", "wrong_metadata", "seed")):
            name = "bad" + str(index)
            producer = self.job(name, mode)
            if index == 2:
                producer["outputs"][0]["minimum_bytes"] = 1_000_000
            jobs.extend([producer, self.job(name + "-child", producer=name)])
        self.admit(jobs)
        self.run_queue()
        for index, expected in enumerate(("failed_process", "failed_native_outputs", "failed_declared_outputs")):
            name = "bad" + str(index)
            self.assertEqual(self.result(name)["outcome"], expected)
            self.assertEqual(self.result(name + "-child")["outcome"], "blocked_predecessor")

    def test_filesystem_exists_without_producer_terminal_never_admits_dependency(self):
        self.admit([self.job("producer"), self.job("consumer", producer="producer")])
        output = self.base / "output-producer"
        output.mkdir()
        (output / "data.txt").write_text("preexisting bytes")
        self.run_queue()
        self.assertEqual(self.result("producer")["outcome"], "deferred_prelaunch_integrity")
        self.assertFalse((self.root / "queues/q0/jobs/consumer/binding.json").exists())
        self.assertFalse((self.root / "executions/attempts/consumer").exists())

    def test_bound_dependency_drift_does_not_rebind_or_launch_consumer(self):
        self.admit([self.job("producer"), self.job("consumer", producer="producer")])
        original = executor.prepare
        def before_consumer(root, spec, sha):
            if spec["attempt_id"] == "consumer":
                raise SystemExit("crash after dependency binding/reservation")
            return original(root, spec, sha)
        with patch.object(executor, "prepare", side_effect=before_consumer):
            with self.assertRaises(SystemExit):
                self.wait(lambda: queue.tick(self.root) and (self.root / "queues/q0/jobs/consumer/binding.json").exists())
        binding_pin = executor.pin(self.root / "queues/q0/jobs/consumer/binding.json")
        (self.base / "output-producer/receipt.json").write_text("changed after immutable binding")
        state = queue.tick(self.root)
        self.assertEqual(self.result("consumer")["outcome"], "deferred_prelaunch_integrity")
        self.assertEqual(binding_pin, executor.pin(binding_pin["path"]))
        self.assertFalse((self.root / "executions/attempts/consumer/launch-intent.json").exists())
        self.assertTrue((self.root / "queues/q0/jobs/consumer/release.json").exists())
        self.assertEqual(state["reserved"], 1)  # Corrupt producer evidence is quarantined.
        self.admit([self.job("independent-after-drift")], "unrelated")
        self.wait(lambda: queue.tick(self.root) and (self.root / "queues/unrelated/jobs/independent-after-drift/result.json").exists())
        self.assertEqual(self.result("independent-after-drift", "unrelated")["outcome"], "accepted_execution")

    def test_invalid_graph_duplicate_attempt_and_unbound_sha_rejected(self):
        a, b = self.job("a"), self.job("b", producer="a")
        a["execution_spec"]["argv"].extend([str(self.base / "output-b/receipt.json"), queue.SHA_ARGUMENT])
        a["dependencies"] = [dict(job_id="b", role="receipt", path_arguments=[4], sha256_arguments=[5])]
        with self.assertRaisesRegex(ValueError, "cycle"):
            self.admit([a, b])
        duplicate = self.job("other")
        duplicate["execution_spec"]["attempt_id"] = "a"
        with self.assertRaisesRegex(ValueError, "duplicate queued attempt"):
            self.admit([self.job("a"), duplicate])
        unbound = self.job("unbound")
        unbound["execution_spec"]["argv"].append(queue.SHA_ARGUMENT)
        with self.assertRaisesRegex(ValueError, "unbound SHA"):
            self.admit([unbound])

    def test_worker_runs_under_separate_detached_executor_controller(self):
        self.admit([self.job("producer"), self.job("consumer", producer="producer")])
        controller = self.job("controller")["execution_spec"]
        controller["argv"] = [sys.executable, str(Path(queue.__file__)), "--root", str(self.root), "work", "--output", controller["output_directory"], "--poll-seconds", ".02", "--max-runtime", "8"]
        controller["input_pins"] = [executor.pin(Path(queue.__file__)), executor.pin(self.root / "store.json")]
        controller["native_receipts"] = [dict(path="receipt.json", required_fields=dict(schema="LocalExperimentQueueWorkerReceiptV1", status="finished"))]
        store = self.base / "controllers"
        executor.prepare(store, controller, queue.digest(controller))
        executor.launch(store, "controller")
        def finished():
            state = executor.inspect(store, "controller")
            return state if state["attempt_terminal"] and not state["job_lock_held"] else None
        terminal = self.wait(finished)["attempt_terminal"]
        self.assertEqual(terminal["return_code"], 0)
        self.assertEqual(self.result("consumer")["outcome"], "accepted_execution")
        self.assertEqual(queue.status(self.root)["reserved"], 0)
        self.assertFalse((self.root / "executions/attempts/controller").exists())

    def test_partial_admission_resumes_same_plan_and_is_visible_before_resume(self):
        plan = dict(schema="LocalExperimentQueuePlanV1", queue_id="partial", jobs=[self.job("producer")])
        original = executor.publish
        def crash(path, value):
            if Path(path) == self.root / "queues/partial/receipt.json":
                raise SystemExit("crash after plan publication")
            return original(path, value)
        with patch.object(executor, "publish", side_effect=crash):
            with self.assertRaises(SystemExit):
                queue.enqueue(self.root, plan, queue.digest(plan))
        self.assertEqual(queue.status(self.root)["partial_admissions"], ["partial"])
        self.assertFalse(queue.status(self.root)["all_jobs_recorded"])
        queue.enqueue(self.root, plan, queue.digest(plan))
        self.run_queue()
        self.assertEqual(self.result("producer", "partial")["outcome"], "accepted_execution")

    def test_result_before_release_is_incomplete_and_reconciles_exact_result(self):
        self.admit([self.job("producer")])
        original = executor.publish
        def crash(path, value):
            if Path(path).name == "release.json":
                raise SystemExit("crash after result publication")
            return original(path, value)
        with patch.object(executor, "publish", side_effect=crash):
            with self.assertRaises(SystemExit):
                self.wait(lambda: queue.tick(self.root) and False)
        result_pin = executor.pin(self.root / "queues/q0/jobs/producer/result.json")
        state = queue.status(self.root)
        self.assertFalse(state["all_jobs_recorded"])
        self.assertEqual(state["reserved"], 1)
        self.run_queue()
        self.assertEqual(result_pin, executor.pin(result_pin["path"]))

    def test_corrupted_cached_result_or_release_retains_capacity_and_quarantines(self):
        self.admit([self.job("producer")])
        original = executor.publish
        def crash(path, value):
            if Path(path).name == "release.json":
                raise SystemExit("crash before release")
            return original(path, value)
        with patch.object(executor, "publish", side_effect=crash):
            with self.assertRaises(SystemExit):
                self.wait(lambda: queue.tick(self.root) and False)
        path = self.root / "queues/q0/jobs/producer/result.json"
        value = queue.document(path)
        value["executor_terminal"]["sha256"] = "0" * 64
        path.write_text(json.dumps(value))
        state = queue.tick(self.root)
        self.assertEqual(state["reserved"], 1)
        self.assertFalse(state["all_jobs_recorded"])
        self.assertEqual(state["queues"]["q0"]["producer"]["outcome"], "quarantined_integrity")
        self.assertFalse((path.parent / "release.json").exists())

    def test_corrupted_prepared_attempt_still_counts_capacity(self):
        self.admit([self.job("one", "hold"), self.job("two", "hold"), self.job("three", "hold")])
        queue.tick(self.root)
        self.wait(lambda: (self.root / "executions/attempts/one/child.json").exists())
        (self.root / "executions/attempts/one/prepared.json").write_text("invalid")
        state = queue.tick(self.root)
        self.assertEqual(state["reserved"], 2)
        self.assertEqual(state["queues"]["q0"]["one"]["outcome"], "quarantined_integrity")
        self.assertFalse((self.root / "executions/attempts/three").exists())

    def test_prepared_input_drift_defers_without_terminal_and_unrelated_job_runs(self):
        first, other = self.job("first"), self.job("other")
        independent_source = self.base / "other_owner.py"
        independent_source.write_text(FIXTURE)
        other["execution_spec"]["argv"][1] = str(independent_source)
        other["execution_spec"]["input_pins"] = [executor.pin(independent_source)]
        self.admit([first, other])
        original = executor.prepare
        def crash(root, spec, sha):
            value = original(root, spec, sha)
            if spec["attempt_id"] == "first":
                raise SystemExit("crash after preparing first")
            return value
        with patch.object(executor, "prepare", side_effect=crash):
            with self.assertRaises(SystemExit):
                queue.tick(self.root)
        self.source.write_text(FIXTURE + "\n# source drift\n")
        self.run_queue()
        self.assertEqual(self.result("first")["outcome"], "deferred_prelaunch_integrity")
        self.assertEqual(self.result("other")["outcome"], "accepted_execution")
        directory = self.root / "executions/attempts/first"
        self.assertTrue((directory / "prepared.json").exists())
        self.assertFalse((directory / "launch-intent.json").exists())
        self.assertFalse((directory / "terminal.json").exists())

    def test_retry_is_new_plan_attempt_and_outputs_without_rebinding_old_consumer(self):
        plan = self.admit([self.job("producer"), self.job("consumer", producer="producer")])
        self.run_queue()
        old_binding = executor.pin(self.root / "queues/q0/jobs/consumer/binding.json")
        old_result = self.result("producer")
        prior = self.job("producer2", "changed-seed")
        prior["execution_spec"]["job_key"] = "producer"
        prior["execution_spec"]["previous_attempt"] = dict(attempt_id="producer", receipt=old_result["executor_terminal"])
        prior["execution_spec"]["retry_reason"] = "explicit new fixture comparison"
        self.admit([prior, self.job("consumer2", producer="producer2")], "retry")
        self.run_queue()
        self.assertNotEqual(old_result["outputs"]["receipt"]["sha256"], self.result("producer2", "retry")["outputs"]["receipt"]["sha256"])
        self.assertEqual(old_binding, executor.pin(old_binding["path"]))
        changed = copy.deepcopy(plan)
        changed["jobs"][1]["execution_spec"]["environment"]["NEW"] = "changed"
        with self.assertRaisesRegex(ValueError, "changed plan"):
            queue.enqueue(self.root, changed, queue.digest(changed))

    def test_receipt_only_roles_exact_consumer_path_and_sha_are_required(self):
        producer, consumer = self.job("producer"), self.job("consumer", producer="producer")
        invalid = copy.deepcopy(producer)
        invalid["outputs"][0]["path"] = "data.txt"
        with self.assertRaisesRegex(ValueError, "authenticated native"):
            self.admit([invalid])
        consumer["execution_spec"]["argv"][4] = str(self.base / "unrelated.json")
        with self.assertRaisesRegex(ValueError, "argv path"):
            self.admit([producer, consumer])
        consumer = self.job("consumer", producer="producer")
        consumer["dependencies"][0]["sha256_arguments"] = []
        with self.assertRaisesRegex(ValueError, "nonempty SHA"):
            self.admit([producer, consumer])

    def prelaunch_deferral(self):
        job = self.job("deferred")
        self.admit([job])
        original = executor.prepare
        def crash(*args):
            original(*args)
            raise SystemExit("crash with registered unlaunched attempt")
        with patch.object(executor, "prepare", side_effect=crash):
            with self.assertRaises(SystemExit):
                queue.tick(self.root)
        self.source.write_text(FIXTURE + "\n# drift\n")
        queue.tick(self.root)
        self.source.write_text(FIXTURE)
        directory = self.root / "queues/q0/jobs/deferred"
        proof = dict(queue_id="q0", job_id="deferred", reason="explicitly restore original pinned inputs",
                     result=executor.pin(directory / "result.json"), release=executor.pin(directory / "release.json"),
                     binding=executor.pin(directory / "binding.json"))
        resumed = copy.deepcopy(job)
        resumed["id"] = "resumed"
        resumed["execution_spec"] = queue.document(directory / "binding.json")["execution_spec"]
        resumed["resume_prelaunch"] = proof
        return resumed

    def test_exact_prelaunch_resume_in_new_plan_launches_once_and_preserves_deferral(self):
        resumed = self.prelaunch_deferral()
        old_pins = [resumed["resume_prelaunch"][key] for key in ("binding", "result", "release")]
        for change in ("argv", "inputs", "roles"):
            invalid = copy.deepcopy(resumed)
            if change == "argv":
                invalid["execution_spec"]["argv"].append("changed")
            elif change == "inputs":
                extra = self.base / "extra.json"
                extra.write_text("{}")
                invalid["execution_spec"]["input_pins"].append(executor.pin(extra))
            else:
                invalid["outputs"][0]["minimum_bytes"] += 1
            with self.assertRaisesRegex(ValueError, "must be identical"):
                self.admit([invalid], "invalid-" + change)
        self.admit([resumed], "resumption")
        competing = copy.deepcopy(resumed)
        competing["id"] = "competitor"
        with self.assertRaisesRegex(ValueError, "all prior queue references"):
            self.admit([competing], "competing-resumption")
        self.run_queue()
        self.assertEqual(self.result("resumed", "resumption")["outcome"], "accepted_execution")
        for pin in old_pins:
            self.assertEqual(pin, executor.pin(pin["path"]))
        self.assertEqual(len(list((self.root / "executions/attempts").iterdir())), 1)
        self.assertEqual(len(list(self.root.glob("executions/attempts/*/launch-intent.json"))), 1)

    def test_prelaunch_resume_rejects_output_unsealed_proof_and_launch_intent(self):
        resumed = self.prelaunch_deferral()
        output = Path(resumed["execution_spec"]["output_directory"])
        output.mkdir()
        (output / "external.txt").write_text("preserve")
        with self.assertRaisesRegex(ValueError, "output must be fresh"):
            self.admit([resumed], "output-conflict")
        output.rename(self.base / "preserved-output")
        proof = resumed["resume_prelaunch"]
        release = Path(proof["release"]["path"])
        saved = release.with_suffix(".preserved")
        release.rename(saved)
        with self.assertRaisesRegex(ValueError, "sealed prelaunch"):
            self.admit([resumed], "unsealed-conflict")
        saved.rename(release)
        executor.publish(self.root / "executions/attempts/deferred/launch-intent.json", {"fixture":"lost launch intent"})
        with self.assertRaisesRegex(ValueError, "launch intent"):
            self.admit([resumed], "launched-conflict")

    def test_empty_scaffold_admission_and_failed_terminal_drift_are_explicit(self):
        (self.root / "queues/scaffold/jobs").mkdir(parents=True)
        self.admit([self.job("failure", "nonzero")], "scaffold")
        self.run_queue()
        terminal = self.result("failure", "scaffold")["executor_terminal"]
        Path(terminal["path"]).write_text("changed failed terminal evidence")
        state = queue.tick(self.root)
        self.assertFalse(state["all_jobs_recorded"])
        self.assertEqual(state["reserved"], 1)
        self.assertEqual(state["queues"]["scaffold"]["failure"]["outcome"], "quarantined_integrity")

    def test_malformed_admission_preserves_unknown_reservation_count(self):
        self.admit([self.job("live", "hold")])
        queue.tick(self.root)
        self.wait(lambda: (self.root / "executions/attempts/live/child.json").exists())
        (self.root / "queues/q0/receipt.json").write_text("malformed")
        state = queue.tick(self.root)
        self.assertEqual(state["reserved"], 1)
        self.assertEqual(state["unknown_reservations"], 1)
        self.assertEqual(state["quarantined_admissions"][0]["queue_id"], "q0")
        self.assertFalse(state["all_jobs_recorded"])

    def test_cached_native_roles_cannot_swap_and_effective_binding_matches_plan(self):
        job = self.job("producer")
        job["execution_spec"]["native_receipts"].append(dict(path="second.json", required_fields=dict(schema="QueueFixtureV1", plan_sha256="fixed-plan", variant="second")))
        job["outputs"].append(dict(role="second", path="second.json", minimum_bytes=1))
        self.admit([job])
        self.run_queue()
        directory = self.root / "queues/q0/jobs/producer"
        result_path, release_path = directory / "result.json", directory / "release.json"
        original_result, original_release = result_path.read_bytes(), release_path.read_bytes()
        value = queue.document(result_path)
        value["outputs"]["receipt"], value["outputs"]["second"] = value["outputs"]["second"], value["outputs"]["receipt"]
        result_path.write_text(json.dumps(value))
        release = queue.document(release_path)
        release["result"] = executor.pin(result_path)
        release_path.write_text(json.dumps(release))
        self.assertIn("role/path/byte", queue.status(self.root)["queues"]["q0"]["producer"]["error"])
        result_path.write_bytes(original_result)
        release_path.write_bytes(original_release)
        binding_path, reservation_path = directory / "binding.json", directory / "reservation.json"
        binding = queue.document(binding_path)
        binding["execution_spec"]["environment"]["FORGED"] = "changed"
        binding_path.write_text(json.dumps(binding))
        reservation = queue.document(reservation_path)
        reservation["binding"] = executor.pin(binding_path)
        reservation_path.write_text(json.dumps(reservation))
        release = queue.document(release_path)
        release["reservation"] = executor.pin(reservation_path)
        release_path.write_text(json.dumps(release))
        self.assertIn("differs from pinned plan", queue.status(self.root)["queues"]["q0"]["producer"]["error"])

    def test_prelaunch_deferred_result_before_release_reconciles_without_launch(self):
        self.admit([self.job("producer")])
        original_prepare = executor.prepare
        def prepared_then_crash(*args):
            original_prepare(*args)
            raise SystemExit("crash after registered preparation")
        with patch.object(executor, "prepare", side_effect=prepared_then_crash):
            with self.assertRaises(SystemExit):
                queue.tick(self.root)
        self.source.write_text(FIXTURE + "\n# drift before launch\n")
        original_publish = executor.publish
        def crash_before_release(path, value):
            if Path(path).name == "release.json":
                raise SystemExit("crash after prelaunch deferral result")
            return original_publish(path, value)
        with patch.object(executor, "publish", side_effect=crash_before_release):
            with self.assertRaises(SystemExit):
                queue.tick(self.root)
        directory = self.root / "queues/q0/jobs/producer"
        result_pin, binding_pin = executor.pin(directory / "result.json"), executor.pin(directory / "binding.json")
        self.assertEqual(queue.status(self.root)["reserved"], 1)
        self.assertFalse(queue.status(self.root)["all_jobs_recorded"])
        state = queue.tick(self.root)
        self.assertEqual(state["reserved"], 0)
        self.assertTrue(state["all_jobs_recorded"])
        self.assertEqual(result_pin, executor.pin(result_pin["path"]))
        self.assertEqual(binding_pin, executor.pin(binding_pin["path"]))
        self.assertTrue(queue.document(directory / "release.json")["never_launched"])
        self.assertFalse((self.root / "executions/attempts/producer/launch-intent.json").exists())

    def test_missing_admission_with_reservation_freezes_other_queue_dispatch(self):
        self.admit([self.job("live", "hold")], "first")
        queue.tick(self.root)
        self.wait(lambda: (self.root / "executions/attempts/live/child.json").exists())
        receipt = self.root / "queues/first/receipt.json"
        receipt.rename(receipt.with_suffix(".preserved"))
        self.admit([self.job("must-wait", "hold")], "second")
        state = queue.tick(self.root)
        self.assertEqual(state["reserved"], 1)
        self.assertEqual(state["unknown_reservations"], 1)
        self.assertEqual(state["partial_admissions"], ["first"])
        self.assertFalse((self.root / "executions/attempts/must-wait").exists())

    def test_relative_root_cli_and_api_execute_same_absolutely_admitted_dag(self):
        self.admit([self.job("producer"), self.job("consumer", producer="producer")])
        command = [sys.executable, "-B", str(Path(queue.__file__).resolve()), "--root", "queue"]
        def cli(*arguments):
            process = subprocess.run([*command, *arguments], cwd=self.base, capture_output=True, text=True, timeout=15)
            self.assertEqual(process.returncode, 0, process.stderr)
            return json.loads(process.stdout)
        self.assertEqual(cli("initialize", "--capacity", "2"), executor.pin(self.root / "store.json"))
        before = cli("status")
        self.assertEqual(before["quarantined_admissions"], [])
        self.assertEqual(before["partial_admissions"], [])
        cli("tick")
        state = cli("work", "--output", "relative-worker", "--poll-seconds", ".02", "--max-runtime", "8")
        self.assertTrue(state["all_jobs_recorded"])
        self.assertEqual(state["reserved"], 0)
        self.assertEqual((self.base / "output-consumer/data.txt").read_text(), "seed:consumed")
        # Also exercise relative API admission/dispatch and CLI admission retry.
        plan = dict(schema="LocalExperimentQueuePlanV1", queue_id="relative-api", jobs=[self.job("api-job")])
        plan_path = self.base / "relative-api-plan.json"
        executor.publish(plan_path, plan)
        with contextlib.chdir(self.base):
            admitted = queue.enqueue("queue", plan, queue.digest(plan))
            self.assertEqual(queue.status("queue")["quarantined_admissions"], [])
            queue.tick(Path("queue"))
            state = queue.work("queue", "relative-api-worker", .02, 8)
        self.assertTrue(state["all_jobs_recorded"])
        self.assertEqual(cli("enqueue", "--plan", str(plan_path), "--sha256", queue.digest(plan)), admitted)
        self.assertEqual(cli("status")["quarantined_admissions"], [])


if __name__ == "__main__":
    unittest.main()
