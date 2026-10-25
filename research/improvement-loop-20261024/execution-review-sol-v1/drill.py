"""Independent CLI-only orphan/lock-holder recovery drill. No scientific command runs."""

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
LOOP = HERE.parent
MANAGER = LOOP / "execution.py"
OWNER_SOURCE = r'''
import json, os, pathlib, subprocess, sys, time

output = pathlib.Path(sys.argv[1])
plan_sha = sys.argv[2]
expected_lock = pathlib.Path(sys.argv[3])
output.mkdir()
with (output / "invocations.txt").open("a") as stream:
    stream.write("invoked\n")

# Carry only the manager's own lock descriptor into the grandchild.
flocks = []
for path in pathlib.Path("/proc/self/fd").iterdir():
    if path.name.isdecimal() and int(path.name) > 2:
        try:
            if pathlib.Path(os.readlink(path)) == expected_lock:
                flocks.append(int(path.name))
        except OSError:
            pass
if len(flocks) != 1:
    raise RuntimeError("expected one inherited job flock, found " + repr(flocks))

grandchild = (
    "import os,pathlib,time; "
    "pathlib.Path(" + repr(str(output / "grandchild.pid")) + ").write_text(str(os.getpid())); "
    "\nwhile True: time.sleep(1)"
)
subprocess.Popen([sys.executable, "-c", grandchild], pass_fds=tuple(flocks))
(output / "receipt.json").write_text(json.dumps({
    "schema": "SyntheticNativeV1",
    "plan_sha256": plan_sha,
    "status": "scientific_failure_unassessed",
}))
while not (output / "allow-leader-exit").exists():
    time.sleep(0.02)
'''


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def pin(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": digest(path), "size": path.stat().st_size}


def executable_pin(path):
    return {"path": str(path), "target": pin(path.resolve(strict=True))}


def cli(store, action, *args, succeeds=True):
    command = [sys.executable, "-B", str(MANAGER), "--root", str(store), action, *map(str, args)]
    result = subprocess.run(command, cwd=LOOP, env={}, capture_output=True, text=True, timeout=25)
    if succeeds:
        if result.returncode:
            raise AssertionError(f"{action} failed: {result.stderr[-1200:]}")
        return json.loads(result.stdout)
    if result.returncode == 0:
        raise AssertionError(f"{action} unexpectedly succeeded: {result.stdout[-1200:]}")
    return result.stderr[-1200:]


def await_condition(label, fn, seconds=12):
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        value = fn()
        if value:
            return value
        time.sleep(0.025)
    raise AssertionError(f"timed out waiting for {label}")


def proc_identity(pid):
    path = Path(f"/proc/{pid}/stat")
    try:
        fields = path.read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return None
    return {
        "pid": pid,
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "start_ticks": int(fields[19]),
        "session": int(fields[3]),
        "state": fields[0],
    }


def is_live(identity):
    current = proc_identity(identity["pid"])
    return current is not None and current["state"] != "Z" and all(
        current[key] == identity[key] for key in ("pid", "boot_id", "start_ticks", "session")
    )


def kill_own_supervisor(store, attempt_dir):
    identity = json.loads((attempt_dir / "supervisor.json").read_text())["identity"]
    assert is_live(identity), "recorded supervisor is not live"
    cmdline = Path(f"/proc/{identity['pid']}/cmdline").read_bytes().split(b"\0")
    expected = [str(MANAGER).encode(), b"_supervise", str(store).encode(), b"orphan-lock-v1"]
    assert all(item in cmdline for item in expected), "recorded PID is not this drill's supervisor"
    descriptor = os.pidfd_open(identity["pid"])
    try:
        assert is_live(identity), "supervisor birth changed during PIDFD acquisition"
        signal.pidfd_send_signal(descriptor, signal.SIGKILL)
    finally:
        os.close(descriptor)
    await_condition("supervisor exit", lambda: not is_live(identity))
    return identity


def main():
    run = HERE / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run.mkdir(parents=True)
    store = run / "store"
    owner = run / "owner.py"
    plan = run / "plan.json"
    output = run / "owner-output"
    owner.write_text(OWNER_SOURCE)
    plan.write_text(canonical({"schema": "SyntheticPlanV1", "scope": "independent execution recovery drill"}))
    invoked = Path(sys.executable)
    job_lock = store / "jobs" / hashlib.sha256(canonical([
        "execution-review-sol-v1", "inherited-lock-grandchild"]).encode()).hexdigest() / "lock"
    spec = {
        "schema": "LocalExecutionSpecV1", "cycle_id": "execution-review-sol-v1",
        "job_key": "inherited-lock-grandchild", "attempt_id": "orphan-lock-v1",
        "argv": [str(invoked), str(owner), str(output), digest(plan), str(job_lock)],
        "executable": executable_pin(invoked), "cwd": str(run), "environment": {},
        "input_pins": [pin(owner), pin(plan)], "output_directory": str(output),
        "native_receipts": [{"path": "receipt.json", "required_fields": {
            "schema": "SyntheticNativeV1", "plan_sha256": digest(plan)}}],
        "timeout_seconds": 90, "grace_seconds": 0.1,
        "previous_attempt": None, "retry_reason": "",
    }
    spec_path = run / "specification.json"
    spec_path.write_text(canonical(spec))
    report = {
        "schema": "ExecutionRecoveryDrillSolV1", "run_directory": str(run),
        "source_pins": {name: pin(LOOP / name) for name in
                        ("execution.py", "test_execution.py", "execution.md", "ledger.py", "run_receipts.py")},
        "drill_source": pin(HERE / "drill.py"), "synthetic_owner": pin(owner), "synthetic_plan": pin(plan),
        "specification_sha256": hashlib.sha256(canonical(spec).encode()).hexdigest(),
    }
    attempt_dir = store / "attempts" / spec["attempt_id"]
    try:
        report["prepared_pin"] = cli(store, "prepare", "--specification", spec_path,
                                     "--sha256", report["specification_sha256"])
        cli(store, "launch", spec["attempt_id"])
        await_condition("native leader and grandchild ready", lambda:
                        (attempt_dir / "supervisor.json").is_file()
                        and (attempt_dir / "child.json").is_file()
                        and (output / "grandchild.pid").is_file()
                        and (output / "receipt.json").is_file())
        grandchild_id = proc_identity(int((output / "grandchild.pid").read_text()))
        assert grandchild_id and is_live(grandchild_id)
        report["supervisor_identity"] = kill_own_supervisor(store, attempt_dir)
        (output / "allow-leader-exit").touch()
        leader_id = json.loads((attempt_dir / "child.json").read_text())["identity"]
        await_condition("native leader exit", lambda: not is_live(leader_id))
        assert is_live(grandchild_id), "grandchild left before recovery test"
        before = cli(store, "inspect", spec["attempt_id"])
        assert before["job_lock_held"] and before["attempt_terminal"] is None
        assert before["state"] == "running_or_starting"
        assert any(item["pid"] == grandchild_id["pid"] for item in before["session_members"])
        assert cli(store, "launch", spec["attempt_id"])["attempt_terminal"] is None
        assert (output / "invocations.txt").read_text().splitlines() == ["invoked"]
        refusal = cli(store, "recover", spec["attempt_id"], succeeds=False)
        assert "lock held" in refusal, refusal
        report["live_refusal"] = {"launch_did_not_duplicate": True,
                                  "recover_blocked_by_lock": True,
                                  "grandchild_alive": True}
        cli(store, "stop", spec["attempt_id"])
        await_condition("grandchild stopped and lock released", lambda:
                        not is_live(grandchild_id) and not cli(store, "inspect", spec["attempt_id"])["job_lock_held"])
        stop_paths = sorted(attempt_dir.glob("stop-*.json"))
        assert len(stop_paths) == 1
        stop_record = json.loads(stop_paths[0].read_text())
        assert stop_record["stopped_session_stable"] is True
        assert any(target["ownership"] == "inherited_flock_verified" and
                   target["identity"]["pid"] == grandchild_id["pid"] and
                   target["sigkill"] == "delivered" for target in stop_record["targets"])
        report["stop_observation"] = pin(stop_paths[0])
        report["inherited_lock_anchor_verified"] = True
        recovery_pin = cli(store, "recover", spec["attempt_id"])
        after = cli(store, "inspect", spec["attempt_id"])
        terminal = after["attempt_terminal"]
        assert after["state"] == "terminal" and not after["job_lock_held"]
        assert terminal["process_outcome"] == "exit_unknown" and terminal["return_code"] is None
        assert terminal["native_outputs"] == "metadata_verified_science_unassessed"
        assert terminal["qualification"] == after["qualification"] == "unassessed"
        assert any(item["path"] == str(output / "receipt.json") for item in terminal["native_receipts"])
        assert (output / "receipt.json").is_file() and (attempt_dir / "command.log").is_file()
        assert cli(store, "recover", spec["attempt_id"]) == recovery_pin
        report["recovery_pin"] = recovery_pin
        report["terminal"] = {key: terminal[key] for key in
                              ("process_outcome", "return_code", "native_outputs", "qualification")}
        report["retained_artifacts"] = {
            "native_receipt": pin(output / "receipt.json"),
            "command_log": pin(attempt_dir / "command.log"),
            "recovery_record": recovery_pin,
        }
        report["result"] = "passed"
    except BaseException as exc:
        report["result"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (run / "report.json").write_text(canonical(report) + "\n")
        print(run / "report.json")


if __name__ == "__main__":
    main()
