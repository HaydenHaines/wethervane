"""One bounded CPU-only deterministic check process; no scientific jobs."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
attempt = sys.argv[1] if len(sys.argv) == 2 else None
if not attempt or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", attempt):
    raise SystemExit("Usage: python3 run_tests.py <fresh-attempt-id>")
launch = HERE / f"{attempt}-launch.json"
terminal = HERE / f"{attempt}-terminal.json"
if launch.exists() or terminal.exists():
    raise SystemExit("Attempt already exists; preserve it and use a fresh identifier")
output = Path(tempfile.mkdtemp(prefix=f"ballotmancy-m01-{attempt}-", dir="/mnt/research-storage"))
snapshot = output / "source-snapshots"
snapshot.mkdir()
sources = {}
for name in ["prototype.py", "test_prototype.py", "run_tests.py"]:
    shutil.copyfile(HERE / name, snapshot / name)
    sources[name] = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
           MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", M01_OUTPUT=str(output),
           PYTENSOR_FLAGS=f"base_compiledir={output}/pytensor,cxx=,floatX=float64")
# The existing immutable environment is repository-root relative.
python = HERE.parents[2] / ".improvement-ledger/runtime/calibration-v1-env/bin/python"
command = ["/usr/bin/time", "-v", str(python), "-B", str(HERE / "test_prototype.py")]
record = dict(schema="M01PrototypeBoundedLaunchV2", output=str(output), command=command,
              cwd=str(HERE), wallLimitSeconds=120, peakRssLimitMiB=512,
              sourceHashes=sources, started=time.time(), samplerAuthorized=False,
              environment={key: env[key] for key in ["PYTHONDONTWRITEBYTECODE", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "M01_OUTPUT", "PYTENSOR_FLAGS"]})
with launch.open("x") as file:
    file.write(json.dumps(record, indent=2) + "\n")
with (output / "tests.log").open("xb") as log:
    process = subprocess.Popen(command, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        code = process.wait(timeout=120)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        record["timedOut"] = True
        code = process.returncode
text = (output / "tests.log").read_text()
rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", text)
maximum_rss = int(rss.group(1)) if rss else None
record.update(exitCode=code, elapsedSeconds=time.time()-record["started"], processTerminal=process.poll() is not None,
              maximumRssKiB=maximum_rss, withinRssBound=maximum_rss is not None and maximum_rss <= 512*1024,
              sourcesUnchanged=all(hashlib.sha256((HERE/name).read_bytes()).hexdigest() == value for name, value in sources.items()))
record["acceptedComputation"] = code == 0 and not record.get("timedOut", False) and record["withinRssBound"] and record["sourcesUnchanged"]
with terminal.open("x") as file:
    file.write(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
print(text[-6000:])
raise SystemExit(0 if record["acceptedComputation"] else 1)
