"""Independent bounded gates/Hessian and real-point wrapper checks; no real fit."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "momentum-calibration-execution-sol-v1"


def main(expected):
    body = (SOURCE / "artifact-manifest.json").read_bytes()
    assert hashlib.sha256(body).hexdigest() == expected
    manifest = json.loads(body)
    for item in manifest["files"]:
        body = (SOURCE / item["path"]).read_bytes()
        assert hashlib.sha256(body).hexdigest() == item["sha256"]
        assert len(body) == item["bytes"]
    sys.path.insert(0, str(SOURCE))
    from unittest.mock import patch

    import numpy as np
    import runner as r

    # SciPy builds distribution documentation with an internal RNG at import;
    # authenticate/import preparation before prohibiting model RNG use.
    prep = r.preparation()

    center = np.array([-2.1, -2.7, -3.0, -3.2, -2.5, -2.9])
    h = np.diag([2.0, 3.0, 5.0, 7.0, 11.0, 13.0]) + np.outer(np.arange(1, 7) / 10, np.arange(1, 7) / 10)

    def quadratic(x):
        delta = x - center
        return 0.5 * delta @ h @ delta, h @ delta

    with (
        patch.object(r, "minimize", side_effect=AssertionError("no optimizer in review oracle")),
        patch("numpy.random.default_rng", side_effect=AssertionError("no RNG")),
    ):
        curvature = r.hessian_diagnostics(quadratic, center)
        for step in curvature["steps"]:
            np.testing.assert_allclose(step["symmetrized_hessian"], h, rtol=1e-10, atol=1e-10)
        inverse = np.linalg.inv(h)
        np.testing.assert_allclose(curvature["inverse_observed_penalized_hessian"], inverse, rtol=1e-10, atol=1e-10)
        success = dict(
            optimizer_success=True,
            finite=True,
            objective=0.0,
            log_scales=center.tolist(),
            gradient_inf=0.0,
            minimum_bound_log_distance=1.0,
        )
        passed = r.gates_for([success, success, {"optimizer_success": False, "finite": False}], curvature)
        assert passed["passed"]
        assert not r.gates_for([{**success, "gradient_inf": 1.0}] * 3, curvature)["passed"]
        assert not r.gates_for([{**success, "minimum_bound_log_distance": 0.0}] * 3, curvature)["passed"]
        assert not r.gates_for(
            [success, {**success, "objective": 1.0}, {"optimizer_success": False, "finite": False}], curvature
        )["passed"]
        plan_path = r.DESIGN["references"][1]["path"]
        plan_sha = r.DESIGN["references"][1]["sha256"]
        plan, _, _ = prep.load_prepared(plan_path, plan_sha, 2022)
        r.verify_alias(plan)
        changed = json.loads(json.dumps(plan))
        changed["folds"][-1]["input_identity_sha256"] = "0" * 64
        try:
            r.verify_alias(changed)
        except ValueError:
            pass
        else:
            raise AssertionError("changed alias accepted")
        errors = []
        for target in r.DESIGN["unique_training_targets"]:
            _, fold, blocks = prep.load_prepared(plan_path, plan_sha, target)
            points = json.loads(Path(fold["objective_points"]["path"]).read_bytes())
            for width, key in [(5, "five"), (6, "six")]:
                objective, medians = r.objective_for(prep, blocks, width)
                assert len(medians) == width
                for point in points:
                    value, gradient = objective(np.array(point["log_scales"][:width]))
                    errors.append(
                        (abs(value - point[key]["value"]), float(np.max(abs(gradient - point[key]["gradient"]))))
                    )
        assert max(x[0] for x in errors) < 1e-10 and max(x[1] for x in errors) < 1e-10
    result = dict(
        manifest_sha256=expected,
        artifact_count=len(manifest["files"]),
        synthetic_full_Hessian_max_error=float(np.max(abs(np.array(curvature["steps"][0]["symmetrized_hessian"]) - h))),
        all_gate_oracles_pass=True,
        alias_tamper_rejected=True,
        actual_supplied_wrapper_points=len(errors),
        max_objective_difference=max(x[0] for x in errors),
        max_gradient_difference=max(x[1] for x in errors),
        optimizer_invoked=False,
        sampler_invoked=False,
        forecast_scores=False,
    )
    (HERE / "oracle-results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-sha256", required=True)
    main(parser.parse_args().manifest_sha256)
