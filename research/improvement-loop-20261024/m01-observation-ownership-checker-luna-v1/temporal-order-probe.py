"""Scratch reproducer: source timestamp precedes its claimed observed period."""
import json, sys
sys.path.insert(0, "/home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024/observation-ownership-checker-sol-v1")
from checker import audit, digest
from test_checker import manifest, seal

m = manifest()
o = m["factors"][0]["components"][0]["observations"][0]
o["period_start"] = "2020-01-15T00:00:00Z"
o["period_end"] = "2020-01-15T00:00:00Z"
seal(m["factors"][0])
r = audit(m, digest(m))
print(json.dumps({
    "source_available_at": m["sources"][0]["available_at"],
    "observation_period_end": o["period_end"],
    "accepted_status": r["status"],
    "source_admission_authenticated": r["source_admission_authenticated"],
}, sort_keys=True))

