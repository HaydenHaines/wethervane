"""Small function-only timing; this is not a queue or forecast benchmark."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
OLD = ROOT / "queue-finite-owned-sol-v1" / "execution.py"
NEW = ROOT / "queue-finite-ancestor-sol-v1" / "execution.py"


def module(path, name):
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        result = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(result)
        return result
    finally:
        sys.path.pop(0)


def main():
    old = module(OLD, "overlap_old")
    new = module(NEW, "overlap_new")
    timings = []
    for size in (50, 150, 300):
        paths = [f"/tmp/ballotmancy-microbench/owner-{i:06d}" for i in range(size)]
        old_times, new_times = [], []
        for _ in range(3):
            start = time.perf_counter()
            prior = any(old.overlap(a, b) for i, a in enumerate(paths) for b in paths[i + 1:])
            old_times.append(time.perf_counter() - start)
            start = time.perf_counter()
            successor = new.overlapping_outputs(paths)
            new_times.append(time.perf_counter() - start)
            if prior or successor or prior != successor:
                raise RuntimeError("unexpected overlapping paths or disagreement")
        timings.append({"paths": size, "pair_checks": size * (size - 1) // 2,
                        "old_seconds": old_times, "new_seconds": new_times,
                        "old_median_seconds": statistics.median(old_times),
                        "new_median_seconds": statistics.median(new_times)})
    print(json.dumps({"scope": "synthetic paths; isolated overlap functions only; no queue or model work",
                      "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (OLD, NEW)},
                      "affinity": sorted(os.sched_getaffinity(0)),
                      "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      "measurements": timings,
                      "limitations": "one warm process; shared host; not queue throughput or scientific cycles"}, indent=2))


if __name__ == "__main__":
    main()
