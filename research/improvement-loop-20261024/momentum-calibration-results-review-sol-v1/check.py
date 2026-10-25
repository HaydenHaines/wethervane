"""Read-only terminal audit; saved-point algebra only, no optimization."""

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
SOURCE = LAB / "momentum-calibration-execution-sol-v1"
RUN = SOURCE / "attempt-root-v1"
RECEIPT = LAB / "checkpoints/momentum-calibration-execution-receipt-v1.json"
PINS = {}


def pin(path):
    data = Path(path).read_bytes()
    value = hashlib.sha256(data).hexdigest()
    PINS[str(path)] = value
    return value


def read(path):
    pin(path)
    return json.loads(Path(path).read_bytes())


def checked(item):
    assert pin(item["path"]) == item["sha256"]


def lines(path):
    pin(path)
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


assert pin(RECEIPT) == "4c150bc181a474a64d75b121770bb9bf5da63e5cd1357ee950a8b3c4a480d79d"
assert pin(SOURCE / "artifact-manifest.json") == "fcaccb33a483045e16d179ce5266370520db853f4c997cb3e0e5afcb4c991ef2"
sys.path.insert(0, str(SOURCE))
from unittest.mock import patch  # noqa: E402

import numpy as np  # noqa: E402
import runner as r  # noqa: E402

root = read(RECEIPT)
r.sources.verify_execution(dict(path=str(RECEIPT), sha256=PINS[str(RECEIPT)]), RUN)
for path in RUN.rglob("*"):
    if path.is_file():
        pin(path)
terminal = read(RUN / "terminal.json")
summary = read(RUN / "summary.json")
launch = read(RUN / "launch.json")
checked(terminal["summary"])
assert launch["pid"] == 1567524 and launch["run_id"] == root["run_id"] == terminal["run_id"]
assert not terminal["promotion_authorized"] and not terminal["forecast_scored"]
assert summary["fits"] == 6 and summary["fixed_starts"] == 18
prep = r.preparation()
plan_path = r.DESIGN["references"][1]["path"]
plan_sha = r.DESIGN["references"][1]["sha256"]
records = []
start_count = 0
max_value_error = max_gradient_error = 0.0
dense_value_errors = []
dense_gradient_errors = []
with (
    patch.object(r, "minimize", side_effect=AssertionError("no optimizer")),
    patch("numpy.random.default_rng", side_effect=AssertionError("no model RNG")),
):
    for target in (2018, 2020, 2022):
        _, fold, blocks = prep.load_prepared(plan_path, plan_sha, target)
        for width, arm in zip((5, 6), r.DESIGN["arms"], strict=True):
            folder = RUN / "fits" / str(target) / arm
            result = read(folder / "result.json")
            fit_terminal = read(folder / "terminal.json")
            checked(fit_terminal["result"])
            objective, medians = r.objective_for(prep, blocks, width)
            attempts = []
            ftol = []
            for i, multiplier in enumerate((0.5, 1.0, 2.0)):
                directory = folder / f"start-{i}"
                a = read(directory / "terminal.json")
                start_launch = read(directory / "launch.json")
                np.testing.assert_allclose(start_launch["start"], np.log(medians * multiplier), rtol=0, atol=0)
                assert a == result["attempts"][i] and a["optimizer_success"] and a["finite"]
                assert len(a["log_scales"]) == len(a["gradient"]) == width
                np.testing.assert_allclose(np.exp(a["log_scales"]), a["standard_deviations"], rtol=0, atol=0)
                assert abs(np.max(abs(np.array(a["gradient"]))) - a["gradient_inf"]) < 1e-15
                evals = lines(directory / "evaluations.jsonl")
                iters = lines(directory / "iterations.jsonl")
                started = [x for x in evals if x["phase"] == "evaluation_started"]
                returned = [x for x in evals if x["phase"] == "evaluation_returned"]
                assert len(started) == len(returned) == a["wrapper_evaluations"]
                assert [x["evaluation"] for x in started] == list(range(1, len(started) + 1))
                assert [x["evaluation"] for x in returned] == list(range(1, len(returned) + 1))
                assert all(x["finite"] and len(x["log_scales"]) == len(x["gradient"]) == width for x in returned)
                assert len(iters) == a["nit"] and a["final_recomputation_evaluations"] == 1
                before, after = iters[-2:]
                relative = (before["value"] - after["value"]) / max(abs(before["value"]), abs(after["value"]), 1)
                ftol.append(
                    dict(
                        start=i,
                        last_relative_reduction=relative,
                        ftol=1e-11,
                        terminated_on_ftol="RELATIVE REDUCTION" in a["message"],
                        gradient_inf=a["gradient_inf"],
                    )
                )
                attempts.append(a)
                start_count += 1
            best_index = min(range(3), key=lambda i: attempts[i]["objective"])
            best = attempts[best_index]
            assert result["gate"]["best_attempt"] == best_index
            x = np.array(best["log_scales"])
            value, gradient = objective(x)
            max_value_error = max(max_value_error, abs(value - best["objective"]))
            max_gradient_error = max(max_gradient_error, float(np.max(abs(gradient - best["gradient"]))))
            hessian = read(folder / "hessian.json")
            geometry = read(folder / "geometry.json")
            matrices = []
            for h in hessian["steps"]:
                raw = np.array(h["raw_gradient_jacobian"])
                matrix = np.array(h["symmetrized_hessian"])
                np.testing.assert_allclose(matrix, (raw + raw.T) / 2, rtol=0, atol=0)
                np.testing.assert_allclose(np.linalg.eigvalsh(matrix), h["eigenvalues"], rtol=1e-12, atol=1e-10)
                matrices.append(matrix)
            difference = np.linalg.norm(matrices[0] - matrices[1], "fro") / max(
                map(lambda m: np.linalg.norm(m, "fro"), matrices)
            )
            assert abs(difference - hessian["relative_frobenius_difference"]) < 1e-14
            inverse = np.linalg.inv(matrices[-1])
            correlation = inverse / np.sqrt(np.outer(np.diag(inverse), np.diag(inverse)))
            np.testing.assert_allclose(inverse, hessian["inverse_observed_penalized_hessian"], rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(
                correlation, hessian["local_log_sd_curvature_correlation"], rtol=1e-10, atol=1e-12
            )
            gate = r.gates_for(attempts, hessian)
            gate["checks"]["geometry_finite_and_SPD"] = all(
                g["covariance_min_eigenvalue"] > 0 and np.isfinite(g["covariance_condition"]) for g in geometry
            )
            gate["passed"] = all(gate["checks"].values())
            assert gate == result["gate"] and fit_terminal["passed"] == gate["passed"]
            declared = r.DESIGN["gates"]
            agreement = sum(
                abs(a["objective"] - best["objective"]) <= declared["agreement_objective_abs"]
                and np.max(abs(np.array(a["log_scales"]) - x)) <= declared["agreement_log_scale_inf"]
                for a in attempts
            )
            bound = np.min(np.column_stack((x - np.log(1e-5), np.log(2) - x)))
            independent_checks = dict(
                finite=np.isfinite(value) and np.isfinite(gradient).all(),
                optimizer_success=best["optimizer_success"],
                gradient=np.max(abs(gradient)) <= 1e-4,
                bounds=bound >= 1e-4,
                agreeing_starts=agreement >= 2,
                hessian_finite=all(np.isfinite(np.linalg.cond(m)) for m in matrices),
                hessian_stable=difference <= 0.001,
                hessian_positive=all((np.linalg.eigvalsh(m) > 0).all() for m in matrices),
                geometry_finite_and_SPD=all(
                    g["covariance_min_eigenvalue"] > 0 and np.isfinite(g["covariance_condition"]) for g in geometry
                ),
            )
            assert independent_checks == result["gate"]["checks"]
            # Rebuild complete-cycle covariance geometry independently at saved point.
            dense_nll = 0.0
            dense_gradient = np.zeros(width)
            for block, g in zip(blocks, geometry, strict=True):
                c = block.coordinates
                n = len(c.observation_ids)
                race = np.array(c.race_ids)
                house = np.array(c.pollsters)
                age = np.array(c.days_to_election)
                same = race[:, None] == race
                kernels = np.array(
                    [
                        np.eye(n),
                        np.ones((n, n)),
                        same,
                        house[:, None] == house,
                        same * np.minimum.outer(age, age) / 30,
                        same * np.outer(age / 30, age / 30),
                    ]
                )[:width]
                covariance = np.tensordot(np.exp(2 * x), kernels, axes=1)
                eigen = np.linalg.eigvalsh(covariance)
                assert abs(eigen.min() - g["covariance_min_eigenvalue"]) < 1e-10
                np.testing.assert_allclose(eigen.max() / eigen.min(), g["covariance_condition"], rtol=1e-9)
                inverse_cov = np.linalg.solve(covariance, np.eye(n))
                alpha = inverse_cov @ block.residual
                sign, det = np.linalg.slogdet(covariance)
                assert sign == 1
                dense_nll += 0.5 * (n * np.log(2 * np.pi) + det + block.residual @ alpha)
                dense_gradient += np.exp(2 * x) * np.array(
                    [np.einsum("ij,ji", inverse_cov, kernel) - alpha @ kernel @ alpha for kernel in kernels]
                )
            normalized = (x - np.log(medians)) / 0.7
            dense_nll += 0.5 * normalized @ normalized
            dense_gradient += (x - np.log(medians)) / 0.7**2
            dense_value_errors.append(abs(dense_nll - value))
            dense_gradient_errors.append(float(np.max(abs(dense_gradient - gradient))))
            records.append(
                dict(
                    target=target,
                    arm=arm,
                    passed=gate["passed"],
                    failed_checks=[k for k, v in gate["checks"].items() if not v],
                    best_attempt=best_index,
                    gradient_inf=best["gradient_inf"],
                    standard_deviations=best["standard_deviations"],
                    all_starts=ftol,
                    local_brownian_slope_curvature=None if width == 5 else float(correlation[4, 5]),
                    local_quadratic_remaining_decrease_estimate=float(0.5 * gradient @ inverse @ gradient),
                )
            )
    for target in (2018, 2020, 2022, 2024):
        for arm in r.DESIGN["arms"]:
            alias = read(RUN / "folds" / str(target) / (arm + ".json"))
            checked(alias["fit_terminal"])
            assert alias["target"] == target and alias["calibration_target"] == (2022 if target == 2024 else target)
            assert alias["aliased"] == (target == 2024)
assert start_count == 18 and len(records) == 6 and sum(x["passed"] for x in records) == 5
assert max_value_error < 1e-9 and max_gradient_error < 1e-9
assert max(dense_value_errors) < 1e-9 and max(dense_gradient_errors) < 1e-8
old_calibration = read(
    Path("/home/hayden/projects/wethervane-redesign-20261006/integration")
    / "research/polling-layer-implementation-20261020/calibration-v1/calibration.json"
)
control = next(v for v in records if v["target"] == 2022 and v["arm"] == r.DESIGN["arms"][0])
assert control["standard_deviations"] == [
    old_calibration["standard_deviations"][name] for name in ("independent", "cycle", "race", "house", "movement")
]
output = dict(
    inspected_pins=PINS,
    execution_receipt_sha256=PINS[str(RECEIPT)],
    run_id=root["run_id"],
    terminal_status=terminal["status"],
    fits=6,
    starts=18,
    fold_receipts=8,
    passed_fits=5,
    records=records,
    max_saved_objective_error=max_value_error,
    max_saved_gradient_error=max_gradient_error,
    max_dense_objective_error=max(dense_value_errors),
    max_dense_gradient_error=max(dense_gradient_errors),
    original_selected_five_scale_control_exact=True,
    optimizer_invoked=False,
    sampler_invoked=False,
    forecast_scores=False,
)
(HERE / "results.json").write_text(json.dumps(output, indent=2) + "\n")
print(
    json.dumps(
        {
            k: output[k]
            for k in [
                "fits",
                "starts",
                "fold_receipts",
                "passed_fits",
                "max_saved_objective_error",
                "max_saved_gradient_error",
            ]
        }
    )
)
