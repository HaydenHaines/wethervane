"""Actual supplied-point dense Gaussian oracle and authenticated reload; no fit."""

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "momentum-calibration-preparation-sol-v1"
PLAN = SOURCE / "prepared-v1/plan.json"
SHA = "f5201d12ec69f378e03cf14af6aa0835ef84cfbbe213e6101dedc575297d16c9"
assert hashlib.sha256(PLAN.read_bytes()).hexdigest() == SHA
sys.path.insert(0, str(SOURCE))
from unittest.mock import patch  # noqa: E402

import numpy as np  # noqa: E402
import prepare as p  # noqa: E402

cache = {}
errors = []
records = []
with (
    patch("scipy.optimize.minimize", side_effect=AssertionError("no optimizer")),
    patch("numpy.random.default_rng", side_effect=AssertionError("no RNG")),
):
    for target in (2018, 2020, 2022, 2024):
        plan, fold, blocks = p.load_prepared(PLAN, SHA, target)
        for block in blocks:
            assert not block.residual.flags.writeable and not block.coordinates.days_to_election.flags.writeable
            try:
                block.residual.setflags(write=True)
            except ValueError:
                pass
            else:
                raise AssertionError("mutable reloaded residual")
        points = json.loads(Path(fold["objective_points"]["path"]).read_bytes())
        for point in points:
            for width, key in ((5, "five"), (6, "six")):
                ell = np.array(point["log_scales"][:width])
                values = []
                gradients = []
                for block in blocks:
                    cycle = block.coordinates.cycles[0]
                    cachekey = (cycle, width, tuple(ell))
                    if cachekey not in cache:
                        c = block.coordinates
                        race = np.array(c.race_ids)
                        firm = np.array(c.pollsters)
                        age = np.array(c.days_to_election)
                        n = len(age)
                        # Complete cycle: all cycle indicators are one, all houses remain jointly linked.
                        same_race = race[:, None] == race
                        kernels = np.array(
                            [
                                np.eye(n),
                                np.ones((n, n)),
                                same_race,
                                firm[:, None] == firm,
                                same_race * np.minimum.outer(age, age) / 30,
                                same_race * np.outer(age / 30, age / 30),
                            ]
                        )[:width]
                        variance = np.exp(2 * ell)
                        sigma = np.tensordot(variance, kernels, axes=1)
                        sign, logdet = np.linalg.slogdet(sigma)
                        assert sign == 1
                        inverse = np.linalg.solve(sigma, np.eye(n))
                        alpha = inverse @ block.residual
                        value = 0.5 * (n * np.log(2 * np.pi) + logdet + block.residual @ alpha)
                        gradient = np.array(
                            [
                                variance[i] * (np.trace(inverse @ kernel) - alpha @ kernel @ alpha)
                                for i, kernel in enumerate(kernels)
                            ]
                        )
                        cache[cachekey] = (value, gradient)
                    value, gradient = cache[cachekey]
                    values.append(value)
                    gradients.append(gradient)
                medians = np.array([plan["regularizer"]["medians"][k] for k in p.math_model.COMPONENTS[:width]])
                standardized = (ell - np.log(medians)) / 0.7
                value = sum(values) + 0.5 * standardized @ standardized
                gradient = np.sum(gradients, axis=0) + (ell - np.log(medians)) / 0.7**2
                value_error = abs(value - point[key]["value"])
                gradient_error = float(np.max(abs(gradient - point[key]["gradient"])))
                assert value_error < 1e-9 and gradient_error < 1e-8
                errors.append((value_error, gradient_error))
        records.append(
            dict(target=target, blocks=[len(b.residual) for b in blocks], rows=fold["rows"], all_arrays_immutable=True)
        )
result = dict(
    plan_sha256=SHA,
    folds=records,
    supplied_objectives_checked=len(errors),
    max_NLL_penalty_error=max(x[0] for x in errors),
    max_joint_gradient_error=max(x[1] for x in errors),
    unique_dense_cycle_checks=len(cache),
    optimizer=False,
    sampler=False,
    forecast_scores=False,
)
(HERE / "objective-results.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
