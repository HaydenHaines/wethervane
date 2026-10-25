
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
