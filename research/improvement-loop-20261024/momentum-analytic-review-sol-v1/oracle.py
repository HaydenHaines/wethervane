"""Independent supplied-point Gaussian oracles; no optimizer, RNG or scores."""

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRODUCER = HERE.parent / "momentum-analytic-prototype-sol-v1"
EXPECTED = "01b76622f1c97c10abcd71e4134a5dd359ccb9ae22c1b92c815d32952438fd51"
manifest_bytes = (PRODUCER / "artifact-manifest.json").read_bytes()
assert hashlib.sha256(manifest_bytes).hexdigest() == EXPECTED
manifest = json.loads(manifest_bytes)
for item in manifest["files"]:
    body = (PRODUCER / item["path"]).read_bytes()
    assert hashlib.sha256(body).hexdigest() == item["sha256"]
sys.path.insert(0, str(PRODUCER))
from unittest.mock import patch  # noqa: E402

import momentum as m  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import multivariate_normal  # noqa: E402


def kernels(c):
    n = len(c.observation_ids)
    result = np.zeros((6, n, n))
    for i in range(n):
        for j in range(n):
            cycle = c.cycles[i] == c.cycles[j]
            race = cycle and c.race_ids[i] == c.race_ids[j]
            result[:, i, j] = [
                i == j,
                cycle,
                race,
                cycle and c.pollsters[i] == c.pollsters[j],
                min(c.days_to_election[i], c.days_to_election[j]) / 30 if race else 0,
                c.days_to_election[i] * c.days_to_election[j] / 900 if race else 0,
            ]
    return result


c = m.PollCoordinatesV1(
    tuple(f"independent-{i}" for i in range(7)),
    ("a", "a", "b", "a", "a", "b", "b"),
    ("2016",) * 3 + ("2020",) * 4,
    ("h", "j", "h", "h", "h", "j", "h"),
    [75, 12, 0, 50, 8, 31, 2],
)
k = kernels(c)
np.testing.assert_allclose(c.kernels(), k, rtol=0, atol=1e-15)
r = np.array([0.2, -0.13, 0.08, -0.17, 0.05, 0.11, -0.04])
medians = np.array([0.11, 0.045, 0.055, 0.025, 0.065, 0.04])
point = np.log(medians) + np.array([0.13, -0.18, 0.27, -0.09, 0.32, -0.21])
block = m.ResidualBlockV1(c, r)


def reference(logs):
    covariance = np.tensordot(np.exp(2 * logs), k, axes=1)
    return -multivariate_normal.logpdf(r, mean=np.zeros(7), cov=covariance) + 0.5 * np.sum(
        ((logs - np.log(medians)) / 0.7) ** 2
    )


with (
    patch("scipy.optimize.minimize", side_effect=AssertionError("no optimizer")),
    patch("numpy.random.default_rng", side_effect=AssertionError("no RNG")),
):
    value, gradient = m.gaussian_error_objective(point, [block], prior_medians=medians, prior_log_sd=0.7)
    gradient_oracle = []
    for axis in np.eye(6):
        step = 2e-5 * axis
        # Five-point stencil of independent SciPy density, not producer objective.
        gradient_oracle.append(
            (
                reference(point - 2 * step)
                - 8 * reference(point - step)
                + 8 * reference(point + step)
                - reference(point + 2 * step)
            )
            / (12 * 2e-5)
        )
    np.testing.assert_allclose(gradient, gradient_oracle, rtol=2e-8, atol=2e-9)
    assert abs(value - reference(point)) < 1e-12
    scales = dict(zip(m.COMPONENTS, np.exp(point), strict=True))
    zero = m.error_covariance(c, {**scales, "persistent_linear": 0})
    old = m.INCUMBENT.error_covariance(c.incumbent_coordinates(), {x: scales[x] for x in m.INCUMBENT.COMPONENTS})
    np.testing.assert_array_equal(zero, old)
    base = np.tensordot(np.exp(2 * point[:5]), k[:5], axes=1)
    groups = list(zip(c.cycles, c.race_ids, strict=True))
    design = np.array(
        [[c.days_to_election[i] / 30 if groups[i] == g else 0 for g in sorted(set(groups))] for i in range(7)]
    )
    u = design * np.exp(point[5])
    middle = np.eye(u.shape[1]) + u.T @ np.linalg.solve(base, u)
    inv = np.linalg.inv(base) - np.linalg.solve(base, u) @ np.linalg.solve(middle, u.T @ np.linalg.inv(base))
    logdet = np.linalg.slogdet(base)[1] + np.linalg.slogdet(middle)[1]
    latent_nll = 0.5 * (7 * np.log(2 * np.pi) + logdet + r @ inv @ r)
    direct_nll = -multivariate_normal.logpdf(r, mean=np.zeros(7), cov=base + u @ u.T)
    assert abs(latent_nll - direct_nll) < 1e-12
    supplied = m.error_covariance(c, scales)
    np.testing.assert_allclose(supplied, base + u @ u.T, rtol=1e-14, atol=1e-16)
    loose = m.synthetic_conditional_example(0.06)
    tight = m.synthetic_conditional_example(0.006)
    assert abs(tight["posterior_mean_slope"]) < abs(loose["posterior_mean_slope"])
    np.testing.assert_array_equal(
        loose["mirrored_posterior_mean"], -np.array([loose["posterior_mean_theta"], loose["posterior_mean_slope"]])
    )
result = dict(
    producer_manifest_sha256=EXPECTED,
    artifacts_verified=len(manifest["files"]),
    fixture="new seven-observation/two-cycle supplied fixture",
    objective_error=abs(value - reference(point)),
    gradient_max_error=float(np.max(abs(gradient - gradient_oracle))),
    latent_marginal_nll_error=abs(latent_nll - direct_nll),
    zero_covariance_exact=True,
    optimizer_invoked=False,
    sampler_invoked=False,
    forecast_scores_computed=False,
)
(HERE / "oracle-results.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
