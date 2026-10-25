"""Evaluate explicit synthetic coordinates only; exclusive output, no fitting."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import momentum as model
import numpy as np
import scipy

HERE = Path(__file__).resolve().parent


def main(output):
    model.source_guard.verify()
    output = Path(output).resolve()
    model.require(output.is_relative_to(HERE), "output must remain in isolated mathematical package")
    model.require(not output.exists(), "exclusive supplied-point output required")
    coordinates = model.PollCoordinatesV1(
        tuple(f"synthetic-{i}" for i in range(6)),
        ("a", "a", "a", "b", "b", "a"),
        ("2020",) * 5 + ("2022",),
        ("x", "x", "y", "x", "z", "x"),
        [60.0, 30.0, 0.0, 30.0, 7.0, 10.0],
    )
    residual = np.array([0.14, -0.07, 0.03, 0.11, -0.1, 0.05])
    block = model.ResidualBlockV1(coordinates, residual)
    medians = np.array([0.1, 0.04, 0.04, 0.03, 0.05, 0.05])
    sd = 0.7
    points = []
    for offset in (np.zeros(6), np.array([0.2, -0.1, 0.15, -0.25, 0.3, -0.2])):
        point = np.log(medians) + offset
        value, gradient = model.gaussian_error_objective(point, [block], prior_medians=medians, prior_log_sd=sd)
        numeric = []
        for k in range(6):
            step = np.eye(6)[k] * 1e-6
            plus = model.gaussian_error_objective(point + step, [block], prior_medians=medians, prior_log_sd=sd)[0]
            minus = model.gaussian_error_objective(point - step, [block], prior_medians=medians, prior_log_sd=sd)[0]
            numeric.append((plus - minus) / 2e-6)
        maximum = float(np.max(np.abs(gradient - numeric)))
        model.require(maximum < 1e-7, "supplied-point gradient mismatch")
        scales = dict(zip(model.COMPONENTS, np.exp(point), strict=True))
        points.append(
            dict(
                log_scales=point.tolist(),
                scales=scales,
                objective=value,
                gradient=gradient.tolist(),
                central_difference_step=1e-6,
                gradient_max_absolute_error=maximum,
                geometry=model.kernel_geometry(coordinates, scales),
            )
        )
    data = dict(
        schema="MomentumAnalyticSuppliedSyntheticPointsV1",
        synthetic=True,
        source_observations_used=False,
        components=model.COMPONENTS,
        days_before_election=coordinates.days_to_election.tolist(),
        observation_ids=coordinates.observation_ids,
        race_ids=coordinates.race_ids,
        cycles=coordinates.cycles,
        pollsters=coordinates.pollsters,
        supplied_residuals=residual.tolist(),
        prior_medians=dict(zip(model.COMPONENTS, medians.tolist(), strict=True)),
        prior_log_sd=sd,
        objective_measure=(
            "log-SD coordinates; normalized Gaussian residual NLL plus quadratic penalty; "
            "fixed prior normalizer omitted exactly as incumbent; no natural-SD Jacobian"
        ),
        points=points,
        conditional_examples=[model.synthetic_conditional_example(s) for s in (0.05, 0.005)],
        source_bindings_sha256=hashlib.sha256((HERE / "source-bindings.json").read_bytes()).hexdigest(),
        environment=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__),
        optimized=False,
        sampled=False,
        outcome_scored=False,
        actual_forecast=False,
        interval_approximation=False,
        forecast_qualified=False,
        promotion_authorized=False,
    )
    with output.open("x") as stream:
        stream.write(json.dumps(data, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            dict(
                path=str(output),
                sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                gradient_max_absolute_errors=[p["gradient_max_absolute_error"] for p in points],
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "supplied-point-results.json")
    main(parser.parse_args().output)
