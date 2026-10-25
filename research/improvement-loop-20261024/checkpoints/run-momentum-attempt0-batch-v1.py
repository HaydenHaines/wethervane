"""Run the 12 already selected receipts once, serially; stop on any exception.

This is a fixed execution batch, not an experiment scheduler or retry policy.
Root records the returned tool handle separately after launching this script.
"""

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
INVENTORY = HERE / "momentum-forecast-root-receipts-v1/inventory.json"
INVENTORY_SHA = "7bd18c7c5fe2d7145b1f98ecb931dfc3b686ddff4010589ff1360569d4d8735f"
RECORDS = HERE / "momentum-attempt0-batch-records-v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(name, value):
    with (RECORDS / name).open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    if sha(INVENTORY) != INVENTORY_SHA:
        raise ValueError("Root inventory changed")
    inventory = json.loads(INVENTORY.read_text())
    for entry in inventory["items"]:
        if sha(entry["receipt"]["path"]) != entry["receipt"]["sha256"]:
            raise ValueError("Root execution receipt changed")
    if sha(inventory["runner"]["path"]) != inventory["runner"]["sha256"]:
        raise ValueError("Accepted runner changed")
    reference = next(r for r in inventory["items"] if r["action"] == "references")
    reference_terminal_path = Path(reference["output"]) / "terminal.json"
    reference_terminal = json.loads(reference_terminal_path.read_text())
    if reference_terminal["status"] != "complete_prelabel_references":
        raise ValueError("Reference generation has not completed")
    jobs = [r for r in inventory["items"] if r["action"] == "job"]
    if len(jobs) != 12 or any(Path(r["output"]).exists() for r in jobs):
        raise ValueError("Exactly 12 fresh attempt0 outputs required; never restart this batch")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise ValueError("Frozen one-thread environment required")
    RECORDS.mkdir(exist_ok=False)
    write("launch.json", dict(
        schema="MomentumRootFixedBatchLaunchV1", utc=utc(), pid=os.getpid(), ppid=os.getppid(),
        inventory=dict(path=str(INVENTORY), sha256=INVENTORY_SHA),
        launcher=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)),
        reference_terminal=dict(path=str(reference_terminal_path), sha256=sha(reference_terminal_path)),
        jobs=[r["name"] for r in jobs], restart_authorized=False,
        numeric_target_labels_authorized=False, scores_authorized=False,
    ))
    for entry in jobs:
        command = [inventory["executable"], inventory["runner"]["path"],
                   "--receipt", entry["receipt"]["path"], "--receipt-sha256", entry["receipt"]["sha256"]]
        write(entry["name"] + "-start.json", dict(utc=utc(), command=command, receipt=entry["receipt"]))
        print(json.dumps(dict(utc=utc(), starting=entry["name"])), flush=True)
        with (RECORDS / (entry["name"] + "-output.log")).open("x") as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            write(entry["name"] + "-process.json", dict(utc=utc(), pid=process.pid, ppid=os.getpid()))
            code = process.wait()
        terminal_path = Path(entry["output"]) / "terminal.json"
        terminal_pin = (dict(path=str(terminal_path), sha256=sha(terminal_path))
                        if terminal_path.exists() else None)
        write(entry["name"] + "-return.json", dict(utc=utc(), returncode=code, terminal=terminal_pin))
        print(json.dumps(dict(utc=utc(), returned=entry["name"], returncode=code, terminal=terminal_pin)), flush=True)
        if code:
            raise SystemExit(code)
    write("terminal.json", dict(utc=utc(), status="all_12_attempt0_processes_returned_zero",
                               convergence_accepted=False, scores_authorized=False))


if __name__ == "__main__":
    main()
