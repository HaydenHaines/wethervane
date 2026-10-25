"""Real tiny subprocesses exercise execution/recovery, never scientific jobs."""

import copy
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import venv
from pathlib import Path
from unittest.mock import patch

import execution as runner
from ledger import Ledger, canonical

OWNER = r'''
import json, os, pathlib, signal, subprocess, sys, time
output, plan_sha, mode = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
output.mkdir()
(output / "invocation.json").write_text(json.dumps({"executable":sys.executable,"prefix":sys.prefix,"secret":os.environ.get("EXECUTION_TEST_SECRET"),"explicit":os.environ.get("EXPLICIT")}))
if mode in ("tree", "leave_descendant", "orphan_exit"):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    descriptors=[]
    for item in pathlib.Path("/proc/self/fd").iterdir():
        try:
            fd=int(item.name)
            if fd>2: os.fstat(fd); descriptors.append(fd)
        except OSError: pass
    code="import os,pathlib,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); pathlib.Path("+repr(str(output / "descendant.pid"))+").write_text(str(os.getpid())); time.sleep(60)"
    subprocess.Popen([sys.executable,"-c",code],pass_fds=descriptors)
if mode != "missing":
    value={"schema":"SyntheticNativeV1","plan_sha256":plan_sha,"status":"scientific_failure_unassessed"}
    if mode == "wrong": value["plan_sha256"]="wrong"
    (output / "receipt.json").write_text("malformed" if mode=="malformed" else json.dumps(value))
if mode == "leave_descendant":
    while not (output / "descendant.pid").exists(): time.sleep(.01)
if mode == "orphan_exit":
    while not (output / "allow-exit").exists(): time.sleep(.01)
if mode in ("hold","tree"): time.sleep(60)
if mode == "nonzero": sys.exit(7)
'''


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "manager"
        self.owner = self.base / "owner.py"
        self.owner.write_text(OWNER)
        self.plan = self.base / "plan.json"
        self.plan.write_text('{"scope":"invented execution fixture"}')
        self.attempts = []
        self.addCleanup(self.cleanup_processes)

    def cleanup_processes(self):
        for attempt_id in self.attempts:
            directory = runner.attempt(self.root, attempt_id)
            for name in ("supervisor", "child"):
                path = directory / (name + ".json")
                if path.exists():
                    identity = json.loads(path.read_text())["identity"]
                    if runner.observe(identity)["state"] == "alive":
                        descriptor = os.pidfd_open(identity["pid"])
                        try:
                            if runner.observe(identity)["state"] == "alive":
                                signal.pidfd_send_signal(descriptor, signal.SIGKILL)
                        finally:
                            os.close(descriptor)

    def spec(self, attempt_id="a0", mode="ok", job="job0", previous=None, executable=None):
        executable = executable or sys.executable
        plan_sha = runner.pin(self.plan)["sha256"]
        return dict(schema="LocalExecutionSpecV1", cycle_id="synthetic-cycle", job_key=job, attempt_id=attempt_id,
                    argv=[executable, str(self.owner), str(self.base / ("output-" + attempt_id)), plan_sha, mode],
                    executable=runner.executable_pin(executable), cwd=str(self.base), environment={"EXPLICIT":"yes"},
                    input_pins=[runner.pin(self.owner), runner.pin(self.plan)], output_directory=str(self.base / ("output-" + attempt_id)),
                    native_receipts=[dict(path="receipt.json", required_fields=dict(schema="SyntheticNativeV1", plan_sha256=plan_sha))],
                    timeout_seconds=10, grace_seconds=0.05, previous_attempt=previous,
                    retry_reason="new declared fixture attempt" if previous else "")

    def prepare(self, spec):
        result = runner.prepare(self.root, spec, hashlib.sha256(canonical(spec).encode()).hexdigest())
        if spec["attempt_id"] not in self.attempts:
            self.attempts.append(spec["attempt_id"])
        return result

    def wait_for(self, condition, timeout=12):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            value = condition()
            if value:
                return value
            time.sleep(0.02)
        logs = [p.read_text() for p in self.root.glob("attempts/*/supervisor.log")]
        self.fail("synthetic subprocess condition timed out: " + repr(logs))

    def terminal(self, attempt_id):
        def finished():
            state = runner.inspect(self.root, attempt_id)
            return state if state["attempt_terminal"] and not state["job_lock_held"] and all(
                p["state"] != "alive" for p in state["process_observation"].values()) else None
        return self.wait_for(finished)

    def child_running(self, attempt_id):
        directory = runner.attempt(self.root, attempt_id)
        self.wait_for(lambda: (directory / "exec-intent.json").exists() and (self.base / ("output-" + attempt_id) / "receipt.json").exists())
        return json.loads((directory / "child.json").read_text())["identity"]

    def kill_supervisor(self, attempt_id):
        identity = json.loads((runner.attempt(self.root, attempt_id) / "supervisor.json").read_text())["identity"]
        descriptor = os.pidfd_open(identity["pid"])
        signal.pidfd_send_signal(descriptor, signal.SIGKILL)
        os.close(descriptor)
        self.wait_for(lambda: runner.observe(identity)["state"] != "alive")

    def test_real_command_minimal_environment_idempotence_and_artifact_only_attachment(self):
        spec = self.spec()
        self.assertEqual(self.prepare(spec), self.prepare(spec))
        with patch.dict(os.environ, EXECUTION_TEST_SECRET="must not leak"):
            runner.launch(self.root, "a0")
        state = self.terminal("a0")
        terminal = state["attempt_terminal"]
        self.assertEqual(terminal["return_code"], 0)
        self.assertEqual(terminal["native_outputs"], "metadata_verified_science_unassessed")
        self.assertEqual(terminal["qualification"], "unassessed")
        invocation = json.loads((self.base / "output-a0/invocation.json").read_text())
        self.assertIsNone(invocation["secret"])
        self.assertEqual(invocation["explicit"], "yes")
        self.assertEqual(runner.launch(self.root, "a0")["terminal_receipt"], state["terminal_receipt"])
        changed = copy.deepcopy(spec)
        changed["environment"]["EXPLICIT"] = "changed"
        with self.assertRaisesRegex(runner.ExecutionError, "different specification"):
            self.prepare(changed)
        ledger = Ledger(self.base / "ledger")
        self.addCleanup(ledger.close)
        ledger.create_cycle("create", "fixture", {"model":"incumbent"}, {"test":True}, "execution", "invented fixture")
        before = ledger.export("fixture")
        receipt = runner.attach(ledger, "capture", self.root, "a0")
        self.assertEqual(receipt, runner.attach(ledger, "capture", self.root, "a0"))
        self.assertEqual(before, ledger.export("fixture"))
        self.assertEqual(json.loads(ledger._artifact(receipt["sha256"]))["return_code"], 0)
        (self.base / "output-a0/receipt.json").write_text("changed")
        with self.assertRaisesRegex(runner.ExecutionError, "fingerprint"):
            runner.inspect(self.root, "a0")

    def test_exit_codes_missing_malformed_and_wrong_plan_are_separate(self):
        for index, mode in enumerate(("missing", "malformed", "wrong", "nonzero")):
            attempt_id = "case" + str(index)
            self.prepare(self.spec(attempt_id, mode, job=attempt_id))
            runner.launch(self.root, attempt_id)
            result = self.terminal(attempt_id)["attempt_terminal"]
            self.assertEqual(result["return_code"], 7 if mode == "nonzero" else 0)
            self.assertEqual(result["native_outputs"], "incomplete" if mode == "missing" else "metadata_verified_science_unassessed" if mode == "nonzero" else "invalid")
            self.assertEqual(result["qualification"], "unassessed")

    def test_caller_loss_and_two_concurrent_launches_produce_one_child(self):
        self.prepare(self.spec(mode="hold"))
        command = [sys.executable, "-B", str(Path(runner.__file__)), "--root", str(self.root), "launch", "a0"]
        callers = [subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(2)]
        results = [process.communicate(timeout=10) for process in callers]
        self.assertTrue(any(p.returncode == 0 for p in callers), results)
        identity = self.child_running("a0")
        self.assertEqual(runner.observe(identity)["state"], "alive")
        self.assertEqual(len(list(self.root.glob("attempts/*/child.json"))), 1)
        runner.launch(self.root, "a0")
        with self.assertRaisesRegex(runner.ExecutionError, "lock held"):
            runner.recover(self.root, "a0")
        with self.assertRaisesRegex(runner.ExecutionError, "lock held"):
            self.prepare(self.spec("a1", previous={"attempt_id":"a0", "receipt":None}))

    def test_supervisor_loss_keeps_lock_then_pidfd_stop_and_unknown_recovery(self):
        spec = self.spec(mode="tree")
        self.prepare(spec)
        runner.launch(self.root, "a0")
        self.child_running("a0")
        descendant_path = self.base / "output-a0/descendant.pid"
        self.wait_for(descendant_path.exists)
        descendant = runner.process_identity(int(descendant_path.read_text()))
        self.kill_supervisor("a0")
        state = runner.inspect(self.root, "a0")
        self.assertTrue(state["job_lock_held"])
        self.assertIsNone(state["attempt_terminal"])
        with self.assertRaisesRegex(runner.ExecutionError, "lock held"):
            runner.recover(self.root, "a0")
        runner.stop(self.root, "a0")
        self.wait_for(lambda: runner.observe(descendant)["state"] != "alive")
        self.wait_for(lambda: not runner.inspect(self.root, "a0")["job_lock_held"])
        receipt = runner.recover(self.root, "a0")
        terminal = runner.inspect(self.root, "a0")["attempt_terminal"]
        self.assertEqual(terminal["process_outcome"], "exit_unknown")
        self.assertIsNone(terminal["return_code"])
        self.assertEqual(terminal["native_outputs"], "metadata_verified_science_unassessed")
        successor = self.spec("a1", previous={"attempt_id":"a0", "receipt":receipt})
        missing_reason = dict(successor, retry_reason="   ")
        with self.assertRaisesRegex(runner.ExecutionError, "retry reason"):
            self.prepare(missing_reason)
        self.prepare(successor)
        runner.launch(self.root, "a1")
        self.assertEqual(self.terminal("a1")["attempt_terminal"]["return_code"], 0)
        self.assertTrue((self.base / "output-a0/receipt.json").exists())

    def test_timeout_cleans_owned_descendant_session(self):
        spec = self.spec(mode="tree")
        spec["timeout_seconds"] = 0.6
        self.prepare(spec)
        runner.launch(self.root, "a0")
        terminal = self.terminal("a0")["attempt_terminal"]
        self.assertTrue(terminal["timed_out"])
        self.assertEqual(terminal["return_code"], -signal.SIGKILL)
        self.assertTrue(terminal["cleanup"]["verified"])
        self.assertEqual(terminal["cleanup"]["remaining"], [])

    def test_prepare_registration_gap_and_launch_intent_crash_do_not_relaunch(self):
        spec = self.spec()
        original = runner.publish
        def crash(path, value):
            if Path(path).name.startswith("claim-"):
                raise OSError("invented prepare crash")
            return original(path, value)
        with patch.object(runner, "publish", side_effect=crash):
            with self.assertRaisesRegex(OSError, "prepare crash"):
                self.prepare(spec)
        with self.assertRaises(FileNotFoundError):
            runner.launch(self.root, "a0")
        # A different initial attempt cannot make the unregistered one launchable.
        self.prepare(self.spec("a1"))
        with self.assertRaises(FileNotFoundError):
            runner.launch(self.root, "a0")
        directory = runner.attempt(self.root, "a1")
        runner.publish(directory / "launch-intent.json", {"fixture":"crash before spawn"})
        self.assertEqual(runner.launch(self.root, "a1")["state"], "orphaned_incomplete")
        recovered = runner.recover(self.root, "a1")
        self.assertEqual(runner.inspect(self.root, "a1")["attempt_terminal"]["process_outcome"], "exit_unknown")
        self.assertEqual(recovered, runner.recover(self.root, "a1"))
        self.assertFalse((self.base / "output-a1").exists())

    def test_identity_reuse_is_not_liveness_and_malformed_terminal_rejected(self):
        identity = runner.process_identity(os.getpid())
        replacement = dict(identity, start_ticks=identity["start_ticks"] + 1)
        with patch.object(runner, "process_identity", return_value=replacement):
            self.assertEqual(runner.observe(identity)["state"], "identity_changed")
        self.prepare(self.spec())
        runner.launch(self.root, "a0")
        self.terminal("a0")
        path = runner.attempt(self.root, "a0") / "terminal.json"
        value = json.loads(path.read_text())
        value["qualification"] = "qualified"
        path.write_text(canonical(value))
        with self.assertRaisesRegex(runner.ExecutionError, "qualification"):
            runner.inspect(self.root, "a0")
        value["qualification"] = "unassessed"
        value["missing_native_receipts"] = ["receipt.json"]
        path.write_text(canonical(value))
        with self.assertRaisesRegex(runner.ExecutionError, "inventory"):
            runner.inspect(self.root, "a0")

    def test_helper_crashes_and_pipe_eof_before_gate_never_execute_owner(self):
        for stage in ("before_identity", "after_identity", "after_release_record"):
            spec = self.spec(stage, job=stage)
            self.prepare(spec)
            directory = runner.attempt(self.root, stage)
            runner.publish(directory / "launch-intent.json", {"fixture":"durable intent before spawn"})
            descriptor, _ = runner.lock(self.root, spec["cycle_id"], stage)
            read_gate, write_gate = os.pipe()
            code = "import sys,os; sys.path.insert(0," + repr(str(Path(runner.__file__).parent)) + "); import execution as e; "
            if stage == "before_identity":
                code += "e.publish=lambda *args: os._exit(31); "
            code += "e.child_main(" + repr(str(self.root)) + "," + repr(stage) + "," + str(descriptor) + "," + str(read_gate) + ")"
            process = subprocess.Popen([sys.executable, "-B", "-c", code], env={}, pass_fds=(descriptor, read_gate),
                                       start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            os.close(read_gate)
            if stage != "before_identity":
                self.wait_for(lambda: (directory / "child.json").exists())
                self.assertTrue(runner.inspect(self.root, stage)["job_lock_held"])
            if stage == "after_release_record":
                runner.publish(directory / "release.json", {"fixture":"durable release before pipe byte"})
            os.close(write_gate)
            stdout, stderr = process.communicate(timeout=10)
            os.close(descriptor)
            self.assertEqual(process.returncode, 31 if stage == "before_identity" else 0, (stdout, stderr))
            self.assertFalse((self.base / ("output-" + stage)).exists())
            self.assertEqual(runner.launch(self.root, stage)["state"], "orphaned_incomplete")
            runner.recover(self.root, stage)
            self.assertIsNone(runner.inspect(self.root, stage)["attempt_terminal"]["return_code"])

    def test_pidfd_rechecks_birth_after_open_before_any_signal(self):
        identity = runner.process_identity(os.getpid())
        actual = runner.observe(identity)
        changed = dict(actual, state="identity_changed")
        with patch.object(runner, "observe", side_effect=[actual, changed]), patch.object(signal, "pidfd_send_signal") as send:
            with self.assertRaisesRegex(runner.ExecutionError, "changed during pidfd"):
                runner.open_pidfd(identity)
            send.assert_not_called()

    def test_sources_are_stream_verified_and_drift_blocks_launch(self):
        spec = self.spec()
        self.prepare(spec)
        original = Path.read_bytes
        def compact_only(path):
            if path == self.owner or path == self.plan or path == Path(spec["executable"]["target"]["path"]):
                raise AssertionError("input verification must not read whole source/runtime files")
            return original(path)
        with patch.object(Path, "read_bytes", compact_only):
            runner.validate(spec)
        self.owner.write_text(OWNER + "\n# modified\n")
        with self.assertRaisesRegex(runner.ExecutionError, "fingerprint"):
            runner.launch(self.root, "a0")
        self.assertFalse((runner.attempt(self.root, "a0") / "launch-intent.json").exists())

    def test_same_job_key_in_new_cycle_and_cross_cycle_predecessor_rejection(self):
        self.prepare(self.spec())
        runner.launch(self.root, "a0")
        previous = self.terminal("a0")["terminal_receipt"]
        second = self.spec("other-cycle")
        second["cycle_id"] = "another-cycle"
        self.prepare(second)
        runner.launch(self.root, "other-cycle")
        self.terminal("other-cycle")
        wrong = self.spec("wrong-cycle", previous={"attempt_id":"a0", "receipt":previous})
        wrong["cycle_id"] = "third-cycle"
        with self.assertRaisesRegex(runner.ExecutionError, "different job"):
            self.prepare(wrong)

    def test_departed_leader_descendant_is_stopped_using_saved_birth_identity(self):
        self.prepare(self.spec(mode="leave_descendant"))
        runner.launch(self.root, "a0")
        directory = runner.attempt(self.root, "a0")
        self.wait_for(lambda: (directory / "supervisor-error.json").exists())
        self.wait_for(lambda: runner.inspect(self.root, "a0")["process_observation"]["supervisor"]["state"] != "alive")
        state = runner.inspect(self.root, "a0")
        self.assertNotEqual(state["process_observation"]["child"]["state"], "alive")
        self.assertTrue(state["job_lock_held"])
        self.assertIsNone(state["attempt_terminal"])
        runner.stop(self.root, "a0")
        self.wait_for(lambda: not runner.inspect(self.root, "a0")["session_members"])
        runner.recover(self.root, "a0")
        terminal = runner.inspect(self.root, "a0")["attempt_terminal"]
        self.assertEqual(terminal["return_code"], 0)
        self.assertTrue(any("stop-" in item["path"] for item in terminal["evidence"]))

    def test_killed_supervisor_then_departed_leader_uses_exact_inherited_flock(self):
        self.prepare(self.spec(mode="orphan_exit"))
        runner.launch(self.root, "a0")
        child = self.child_running("a0")
        output = self.base / "output-a0"
        self.wait_for(lambda: (output / "descendant.pid").exists())
        descendant = runner.process_identity(int((output / "descendant.pid").read_text()))
        self.kill_supervisor("a0")
        (output / "allow-exit").touch()
        self.wait_for(lambda: runner.observe(child)["state"] != "alive")
        self.assertFalse((runner.attempt(self.root, "a0") / "supervisor-error.json").exists())
        prepared = json.loads((runner.attempt(self.root, "a0") / "prepared.json").read_text())
        self.assertTrue(runner.holds_lock(descendant, prepared["job_lock"]))
        # A separate open of the same inode does not prove inherited ownership.
        separate = os.open(runner.job_directory(self.root, "synthetic-cycle", "job0") / "lock", os.O_RDONLY)
        try:
            self.assertFalse(runner.holds_lock(runner.process_identity(os.getpid()), prepared["job_lock"]))
        finally:
            os.close(separate)
        with patch.object(runner, "holds_lock", side_effect=[True, False]), patch.object(signal, "pidfd_send_signal") as send:
            with self.assertRaisesRegex(runner.ExecutionError, "ownership unresolved"):
                runner.stop(self.root, "a0")
            send.assert_not_called()
        runner.stop(self.root, "a0")
        self.wait_for(lambda: runner.observe(descendant)["state"] != "alive")
        self.wait_for(lambda: not runner.inspect(self.root, "a0")["job_lock_held"])
        runner.recover(self.root, "a0")
        state = runner.inspect(self.root, "a0")
        self.assertIsNone(state["attempt_terminal"]["return_code"])
        stop_receipt = next(runner.attempt(self.root, "a0").glob("stop-*.json"))
        self.assertEqual(json.loads(stop_receipt.read_text())["targets"][0]["ownership"], "inherited_flock_verified")

    def test_caller_killed_immediately_after_supervisor_spawn_does_not_duplicate(self):
        self.prepare(self.spec(mode="hold"))
        code = """import sys,os,signal
sys.path.insert(0,sys.argv[1])
import execution as e
original=e.subprocess.Popen
def spawn(*args,**kwargs):
    process=original(*args,**kwargs)
    os.kill(os.getpid(),signal.SIGKILL)
e.subprocess.Popen=spawn
e.launch(sys.argv[2],'a0')
"""
        caller = subprocess.run([sys.executable, "-B", "-c", code, str(Path(runner.__file__).parent), str(self.root)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        self.assertEqual(caller.returncode, -signal.SIGKILL)
        child = self.child_running("a0")
        self.assertEqual(runner.observe(child)["state"], "alive")
        runner.launch(self.root, "a0")
        self.assertEqual(len(list(self.root.glob("attempts/*/child.json"))), 1)
        self.kill_supervisor("a0")
        runner.stop(self.root, "a0")
        self.wait_for(lambda: not runner.inspect(self.root, "a0")["job_lock_held"])
        runner.recover(self.root, "a0")

    def test_venv_symlink_invocation_preserved_and_retarget_rejected(self):
        environment = self.base / "venv"
        venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
        executable = str(environment / "bin/python")
        spec = self.spec(executable=executable)
        self.prepare(spec)
        runner.launch(self.root, "a0")
        self.terminal("a0")
        invocation = json.loads((self.base / "output-a0/invocation.json").read_text())
        self.assertEqual(invocation["executable"], executable)
        self.assertEqual(invocation["prefix"], str(environment))
        pending = self.spec("other", job="other-job", executable=executable)
        self.prepare(pending)
        other = self.base / "other-binary"
        other.write_bytes(Path(sys.executable).resolve().read_bytes())
        Path(executable).unlink()
        Path(executable).symlink_to(other)
        with self.assertRaisesRegex(runner.ExecutionError, "target changed"):
            runner.launch(self.root, "other")


if __name__ == "__main__":
    unittest.main()
