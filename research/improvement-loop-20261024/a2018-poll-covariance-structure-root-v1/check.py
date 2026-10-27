"""Inspect only aggregate covariance structure; no observation identifiers are exported."""
import hashlib
import json
from pathlib import Path

import numpy as np

SOURCE = Path("/home/hayden/projects/wethervane-redesign-20261006/integration/research/senate-stack-integration-20261020/historical-national-2018-assembly-astra-v1/problem.json")
SHA256 = "abcaf347e682422303780c5ff12bb3e6179e4b98362686c01338e29df06a385a"
SIZE = 6979007


def inspect(matrix, contests):
    a = np.asarray(matrix, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] != len(contests) or not contests:
        raise ValueError("matrix and contest vector dimensions differ")
    if any(not isinstance(c, str) or not c for c in contests):
        raise ValueError("invalid contest identity")
    if not np.isfinite(a).all() or not np.array_equal(a, a.T) or (np.diag(a) <= 0).any():
        raise ValueError("invalid covariance")
    correlation = a / np.sqrt(np.outer(np.diag(a), np.diag(a)))
    if not np.isfinite(correlation).all() or np.any(np.abs(correlation) > 1 + 1e-12):
        raise ValueError("invalid correlation range")
    counts = {"same_contest": [], "different_contest": []}
    for i, contest in enumerate(contests):
        for j in range(i):
            category = "same_contest" if contest == contests[j] else "different_contest"
            counts[category].append(float(correlation[i, j]))

    def summarize(values):
        return {"pairs": len(values), "positive": sum(v > 0 for v in values),
                "negative": sum(v < 0 for v in values), "exact_zero": sum(v == 0 for v in values),
                "minimum_correlation": min(values) if values else None,
                "maximum_correlation": max(values) if values else None}
    return {"rows": len(contests), "distinct_contests": len(set(contests)),
            **{key: summarize(value) for key, value in counts.items()}}


def frozen_check(path=SOURCE):
    raw = Path(path).read_bytes()
    if len(raw) != SIZE or hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError("prepared problem byte pin differs")
    problem = json.loads(raw)
    blocks = problem["polls"]
    if len(blocks) != 1 or len(blocks[0]["rows"]) != 274:
        raise ValueError("unexpected poll block structure")
    block = blocks[0]
    values = inspect(block["covariance"], [row["contest_id"] for row in block["rows"]])
    return {"schema": "A2018PreparedPollCovarianceStructureV1", "problem_sha256": SHA256,
            "structure": values, "compiled_graph_authenticated": False,
            "raw_sample_overlap_verified": False, "covariance_scientifically_qualified": False,
            "forecast_qualified": False}


if __name__ == "__main__":
    print(json.dumps(frozen_check(), indent=2, sort_keys=True))
