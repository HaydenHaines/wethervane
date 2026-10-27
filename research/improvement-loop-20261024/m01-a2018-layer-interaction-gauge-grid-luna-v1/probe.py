"""Independent deterministic gauge-offset check using frozen v3 arithmetic."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import math
import random
import sys
from pathlib import Path

CHECK = Path("/home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024/m01-a2018-layer-interaction-audit-luna-v3/check.py")
EXPECTED_SHA256 = "a0812ddd5fabb99bf3b5d358764691b63248b461e0e8f9688eccd6a1e222f7c1"
raw = CHECK.read_bytes()
if hashlib.sha256(raw).hexdigest() != EXPECTED_SHA256:
    raise ValueError("frozen v3 check.py hash mismatch")
spec = importlib.util.spec_from_file_location("pinned_v3_check", CHECK)
if spec is None or spec.loader is None:
    raise ValueError("could not load frozen v3 checker")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ordinary_poll_logit = module.ordinary_poll_logit

rng = random.Random(20261027)
rows = []
for offset, quantum, n in [(1e16, 2.0, 200), (1e20, 32768.0, 200)]:
    maxerr = 0.0
    for i in range(n):
        pairs = tuple(tuple(rng.randint(-8, 8) * quantum for _ in range(2)) for __ in range(3))
        other = rng.randint(-8, 8) * quantum
        weights = tuple(10.0 ** rng.uniform(-4, 4) for _ in range(3))
        shifted = tuple(tuple(v + offset for v in row) for row in pairs)
        shifted_o = other + offset
        if tuple(tuple(v - offset for v in row) for row in shifted) != pairs or shifted_o - offset != other:
            raise ValueError("fixture lost utility bits under offset")
        a = ordinary_poll_logit(weights, pairs, other, 0.7, 2.3)
        b = ordinary_poll_logit(weights, shifted, shifted_o, 0.7, 2.3)
        err = abs(a - b)
        maxerr = max(maxerr, err)
        if err > 2e-12:
            raise ValueError(f"gauge invariance failed: offset={offset}, case={i}, error={err}")
    rows.append({"offset": offset, "cases": n, "max_abs_invariance_error": maxerr,
                 "bit_preservation_checked": True})
print(json.dumps({"seed": 20261027, "checks": rows}, indent=2))
