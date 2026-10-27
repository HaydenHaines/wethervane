"""Aggregate-only numerical check of one exact prepared covariance; not model admission."""
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np

SOURCE = Path("/home/hayden/projects/wethervane-redesign-20261006/integration/research/senate-stack-integration-20261020/historical-national-2018-assembly-astra-v1/problem.json")
SOURCE_SIZE = 6979007
SOURCE_SHA256 = "abcaf347e682422303780c5ff12bb3e6179e4b98362686c01338e29df06a385a"


def inspect(matrix):
    """Check an invented or prepared matrix; this helper authenticates no source."""
    a = np.asarray(matrix, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] == 0:
        raise ValueError("covariance must be nonempty and square")
    if not np.isfinite(a).all() or not np.array_equal(a, a.T):
        raise ValueError("covariance must be finite and exactly symmetric")
    if (np.diag(a) <= 0).any():
        raise ValueError("diagonal entries must be positive")
    eig = np.linalg.eigvalsh(a)
    scale = float(np.max(np.abs(eig)))
    lower = float(eig[0])
    numerical_margin = scale * np.finfo(np.float64).eps * a.shape[0]
    if not math.isfinite(scale) or lower <= numerical_margin:
        raise ValueError("positive definiteness lacks a safe float64 eigenvalue margin")
    try:
        factor = np.linalg.cholesky(a)
    except np.linalg.LinAlgError as exc:
        raise ValueError("float64 Cholesky failed") from exc
    residual = float(np.linalg.norm(a - factor @ factor.T, ord="fro") / np.linalg.norm(a, ord="fro"))
    return {"dimension": int(a.shape[0]), "minimum_eigenvalue": lower,
            "maximum_eigenvalue": float(eig[-1]), "eigenvalue_ratio": float(eig[-1] / lower),
            "float64_eigenvalue_margin": float(numerical_margin), "cholesky_relative_frobenius_residual": residual}


def frozen_check(path=SOURCE):
    raw = Path(path).read_bytes()
    if len(raw) != SOURCE_SIZE or hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("prepared problem byte pin differs")
    problem = json.loads(raw)
    blocks = problem["polls"]
    if len(blocks) != 1 or len(blocks[0]["rows"]) != 274 or len(blocks[0]["covariance"]) != 274:
        raise ValueError("unexpected prepared poll-block structure")
    matrix = blocks[0]["covariance"]
    encoded = json.dumps(matrix, separators=(",", ":"), allow_nan=False).encode()
    stats = inspect(matrix)
    if stats["dimension"] != 274:
        raise ValueError("unexpected prepared covariance dimension")
    return {"schema": "A2018PreparedPollCovarianceNumericsV1", "problem_sha256": SOURCE_SHA256,
            "covariance_canonical_sha256": hashlib.sha256(encoded).hexdigest(),
            "result": "float64_numerical_spd_pass", "statistics": stats,
            "numpy_version": np.__version__, "python_version": platform.python_version(),
            "compiled_graph_authenticated": False, "covariance_scientifically_qualified": False,
            "forecast_qualified": False}


if __name__ == "__main__":
    print(json.dumps(frozen_check(), indent=2, sort_keys=True))
