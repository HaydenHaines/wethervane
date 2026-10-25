
import json, os, pathlib, subprocess, sys, time

output = pathlib.Path(sys.argv[1])
plan_sha = sys.argv[2]
output.mkdir()
with (output / "invocations.txt").open("a") as stream:
    stream.write("invoked\n")

# Only the manager's inherited flock descriptor qualifies as the ownership anchor.
flocks = []
for path in pathlib.Path("/proc/self/fdinfo").iterdir():
    if path.name.isdecimal() and int(path.name) > 2:
        try:
            if any("FLOCK ADVISORY WRITE" in line for line in path.read_text().splitlines()):
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
