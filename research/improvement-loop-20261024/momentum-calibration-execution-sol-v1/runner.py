"""Matched calibration execution. Actual CLI requires an external pinned receipt.

Author development uses synthetic solver tests and supplied real points only.
"""

import argparse
import importlib.util
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import execution_sources as sources

DESIGN, BINDINGS = sources.verify_sources()
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

HERE = Path(__file__).resolve().parent
require = sources.require


def now():
    return datetime.now(timezone.utc).isoformat()


def serial(value):
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if isinstance(value, np.ndarray):
        return serial(value.tolist())
    if isinstance(value, np.generic):
        return serial(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(serial(value), stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def preparation():
    sources.verify_sources()
    path = Path(DESIGN["references"][0]["path"]).parent / "prepare.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("reviewed_calibration_preparation", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def objective_for(prep, blocks, width):
    medians = np.array([prep.SELECTION["regularizer"]["medians"][k] for k in prep.math_model.COMPONENTS])
    if width == 5:
        data = [(b.residual, b.coordinates.kernels()[:5]) for b in blocks]

        def objective(point):
            return prep.math_model.INCUMBENT.gaussian_error_objective(
                point, data, prior_medians=medians[:5], prior_log_sd=0.7
            )
    else:
        require(width == 6, "joint five/six objective required")

        def objective(point):
            return prep.math_model.gaussian_error_objective(point, blocks, prior_medians=medians, prior_log_sd=0.7)

    return objective, medians[:width]


def bound_distances(point):
    low, high = np.log(DESIGN["optimizer"]["log_sd_bounds"])
    return np.column_stack((point - low, high - point))


def attempt(objective, initial, folder, metadata):
    folder.mkdir(parents=True, exist_ok=False)
    write(folder / "launch.json", dict(start=initial, metadata=metadata, optimizer=DESIGN["optimizer"], utc=now()))
    calls = 0
    iterations = 0
    cache = {}
    started = time.monotonic()
    result = None
    try:
        with (
            (folder / "evaluations.jsonl").open("x") as evaluations,
            (folder / "iterations.jsonl").open("x") as progress,
        ):

            def tracked(point):
                nonlocal calls
                calls += 1
                evaluations.write(
                    json.dumps(
                        serial(dict(phase="evaluation_started", evaluation=calls, log_scales=point, utc=now())),
                        allow_nan=False,
                    )
                    + "\n"
                )
                evaluations.flush()
                value, gradient = objective(point)
                value = float(value)
                gradient = np.asarray(gradient, float)
                finite = bool(
                    np.asarray(point).shape == initial.shape
                    and gradient.shape == initial.shape
                    and np.isfinite(value)
                    and np.isfinite(gradient).all()
                )
                evaluations.write(
                    json.dumps(
                        serial(
                            dict(
                                phase="evaluation_returned",
                                evaluation=calls,
                                log_scales=point,
                                value=value,
                                gradient=gradient,
                                finite=finite,
                                utc=now(),
                            )
                        ),
                        allow_nan=False,
                    )
                    + "\n"
                )
                evaluations.flush()
                require(finite, "finite aligned objective and gradient required")
                cache[np.asarray(point, float).tobytes()] = (value, gradient.copy())
                return value, gradient

            def callback(point):
                nonlocal iterations
                iterations += 1
                value, gradient = cache.get(np.asarray(point, float).tobytes(), (None, None))
                if value is None:
                    value, gradient = tracked(point)
                item = dict(iteration=iterations, log_scales=point, value=value, gradient=gradient, utc=now())
                progress.write(json.dumps(serial(item), allow_nan=False) + "\n")
                progress.flush()
                if iterations % 10 == 0:
                    print(json.dumps(serial(dict(phase="optimizer_progress", metadata=metadata, **item))), flush=True)

            result = minimize(
                tracked,
                initial.copy(),
                jac=True,
                method=DESIGN["optimizer"]["method"],
                bounds=[tuple(np.log(DESIGN["optimizer"]["log_sd_bounds"]))] * len(initial),
                options=DESIGN["optimizer"]["options"].copy(),
                callback=callback,
            )
            point = np.asarray(result.x, float)
            require(
                point.shape == initial.shape and np.isfinite(point).all(), "finite full optimizer coordinates required"
            )
            # Fresh recomputation: never trust an optimizer's fun/jac.
            value, gradient = objective(point)
            gradient = np.asarray(gradient, float)
            finite = bool(
                np.isfinite(value)
                and gradient.shape == initial.shape
                and np.isfinite(gradient).all()
                and np.isfinite(result.fun)
                and np.asarray(result.jac).shape == initial.shape
                and np.isfinite(np.asarray(result.jac, float)).all()
            )
            record = dict(
                status="optimizer_returned",
                optimizer_success=bool(result.success),
                finite=finite,
                log_scales=point,
                standard_deviations=np.exp(point),
                objective=float(value),
                gradient=gradient,
                gradient_inf=float(np.max(abs(gradient))),
                bound_log_distances=bound_distances(point),
                minimum_bound_log_distance=float(bound_distances(point).min()),
                optimizer_fun=result.fun,
                optimizer_jac=result.jac,
                optimizer_fun_minus_recomputed=float(result.fun - value),
                status_code=int(result.status),
                message=str(result.message),
                nit=int(result.nit),
                nfev=int(result.nfev),
                njev=int(result.njev),
                wrapper_evaluations=calls,
                final_recomputation_evaluations=1,
                elapsed_seconds=time.monotonic() - started,
                metadata=metadata,
            )
            write(folder / "terminal.json", record)
            print(
                json.dumps(
                    serial(
                        dict(
                            phase="attempt_completed",
                            metadata=metadata,
                            success=record["optimizer_success"],
                            objective=record["objective"],
                            gradient_inf=record["gradient_inf"],
                        )
                    )
                ),
                flush=True,
            )
            return record
    except BaseException as error:
        record = dict(
            status="preserved_attempt_failure",
            optimizer_success=False,
            finite=False,
            error=repr(error),
            traceback=traceback.format_exc(),
            wrapper_evaluations=calls,
            callback_iterations=iterations,
            elapsed_seconds=time.monotonic() - started,
            metadata=metadata,
            automatic_retry=False,
            optimizer_result=None
            if result is None
            else {
                k: getattr(result, k, None)
                for k in ("x", "fun", "jac", "success", "status", "message", "nit", "nfev", "njev")
            },
        )
        write(folder / "failure.json", record)
        if not isinstance(error, Exception):
            raise
        return record


def hessian_diagnostics(objective, point):
    results = []
    for step in DESIGN["gates"]["hessian_gradient_central_difference_steps"]:
        columns = []
        for i in range(len(point)):
            delta = np.eye(len(point))[i] * step
            plus = np.asarray(objective(point + delta)[1], float)
            minus = np.asarray(objective(point - delta)[1], float)
            columns.append((plus - minus) / (2 * step))
        raw = np.column_stack(columns)
        require(np.isfinite(raw).all(), "finite full Hessian required")
        matrix = (raw + raw.T) / 2
        eigen = np.linalg.eigvalsh(matrix)
        results.append(
            dict(
                step=step,
                raw_gradient_jacobian=raw,
                symmetrized_hessian=matrix,
                eigenvalues=eigen,
                condition_number=float(np.linalg.cond(matrix)),
                positive=bool((eigen > 0).all()),
            )
        )
    first, second = (r["symmetrized_hessian"] for r in results)
    denominator = max(np.linalg.norm(first, "fro"), np.linalg.norm(second, "fro"))
    difference = float(np.linalg.norm(first - second, "fro") / denominator) if denominator > 0 else 0.0
    positive = all(r["positive"] for r in results)
    inverse = None
    correlation = None
    if positive:
        inverse = np.linalg.solve(second, np.eye(len(point)))
        require(
            np.isfinite(inverse).all() and (np.diag(inverse) > 0).all(),
            "finite positive local curvature covariance required",
        )
        correlation = inverse / np.sqrt(np.outer(np.diag(inverse), np.diag(inverse)))
        require(np.isfinite(correlation).all(), "finite local curvature correlation required")
    return dict(
        steps=results,
        relative_frobenius_difference=difference,
        relative_difference_denominator="max(Frobenius_norm(H_step1),Frobenius_norm(H_step2))",
        positive_eigenvalues=positive,
        inverse_observed_penalized_hessian=inverse,
        local_log_sd_curvature_correlation=correlation,
        brownian_slope_local_log_sd_curvature_correlation=None
        if len(point) == 5 or correlation is None
        else float(correlation[4, 5]),
        interpretation=(
            "Local inverse observed penalized Hessian in log-SD coordinates; never empirical posterior correlation"
        ),
    )


def gates_for(attempts, hessian=None):
    successful = [(i, a) for i, a in enumerate(attempts) if a["optimizer_success"] and a["finite"]]
    if not successful:
        return dict(
            passed=False,
            best_attempt=None,
            checks=dict(
                finite=False,
                optimizer_success=False,
                gradient=False,
                bounds=False,
                agreeing_starts=False,
                hessian_finite=False,
                hessian_stable=False,
                hessian_positive=False,
            ),
            agreement=[],
        )
    index, best = min(successful, key=lambda pair: pair[1]["objective"])
    agreement = []
    for i, a in enumerate(attempts):
        if a["optimizer_success"] and a["finite"]:
            difference = abs(a["objective"] - best["objective"])
            distance = float(np.max(abs(np.asarray(a["log_scales"]) - best["log_scales"])))
            agrees = (
                difference <= DESIGN["gates"]["agreement_objective_abs"]
                and distance <= DESIGN["gates"]["agreement_log_scale_inf"]
            )
            agreement.append(
                dict(attempt=i, agrees=bool(agrees), objective_difference=difference, log_scale_inf_distance=distance)
            )
        else:
            agreement.append(dict(attempt=i, agrees=False, reason="unsuccessful_or_nonfinite"))
    hessian_finite = (
        hessian is not None
        and "error" not in hessian
        and np.isfinite(hessian["relative_frobenius_difference"])
        and all(np.isfinite(r["condition_number"]) and np.isfinite(r["eigenvalues"]).all() for r in hessian["steps"])
    )
    checks = dict(
        finite=bool(best["finite"]),
        optimizer_success=bool(best["optimizer_success"]),
        gradient=best["gradient_inf"] <= DESIGN["gates"]["best_gradient_inf_max"],
        bounds=best["minimum_bound_log_distance"] >= DESIGN["gates"]["bound_log_distance_min"],
        agreeing_starts=sum(a["agrees"] for a in agreement) >= DESIGN["gates"]["minimum_agreeing_successful_starts"],
        hessian_finite=bool(hessian_finite),
        hessian_stable=bool(
            hessian_finite
            and hessian["relative_frobenius_difference"] <= DESIGN["gates"]["hessian_relative_frobenius_difference_max"]
        ),
        hessian_positive=bool(hessian_finite and hessian["positive_eigenvalues"]),
    )
    return dict(passed=all(checks.values()), best_attempt=index, checks=checks, agreement=agreement)


def fit_job(objective, medians, folder, metadata, geometry):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    write(
        folder / "launch.json", dict(metadata=metadata, optimizer=DESIGN["optimizer"], gates=DESIGN["gates"], utc=now())
    )
    attempts = []
    try:
        for i, multiplier in enumerate(DESIGN["optimizer"]["multiplicative_prior_median_starts"]):
            attempts.append(
                attempt(
                    objective,
                    np.log(medians * multiplier),
                    folder / f"start-{i}",
                    {**metadata, "start_index": i, "prior_median_multiplier": multiplier},
                )
            )
        gate = gates_for(attempts)
        hessian = None
        geometries = None
        if gate["best_attempt"] is not None:
            point = np.asarray(attempts[gate["best_attempt"]]["log_scales"])
            try:
                hessian = hessian_diagnostics(objective, point)
            except Exception as error:
                hessian = dict(error=repr(error), traceback=traceback.format_exc())
            try:
                geometries = geometry(np.exp(point))
            except Exception as error:
                geometries = dict(error=repr(error), traceback=traceback.format_exc())
        gate = gates_for(attempts, hessian)
        gate["checks"]["geometry_finite_and_SPD"] = bool(
            isinstance(geometries, list)
            and len(geometries) > 0
            and all(
                np.isfinite(g["covariance_condition"])
                and np.isfinite(g["covariance_min_eigenvalue"])
                and g["covariance_min_eigenvalue"] > 0
                for g in geometries
            )
        )
        gate["passed"] = all(gate["checks"].values())
        write(folder / "hessian.json", hessian)
        write(folder / "geometry.json", geometries)
        summary = dict(
            metadata=metadata,
            gate=gate,
            attempts=attempts,
            fitted_scales=None
            if gate["best_attempt"] is None
            else attempts[gate["best_attempt"]]["standard_deviations"],
            conditional_MAP_only=True,
            forecast_improvement_claim=False,
            automatic_retry=False,
        )
        write(folder / "result.json", summary)
        write(
            folder / "terminal.json",
            dict(
                status="calibration_gates_passed" if gate["passed"] else "calibration_gates_failed_preserved",
                passed=gate["passed"],
                result=sources.pin(folder / "result.json"),
                metadata=metadata,
                forecast_scored=False,
                promotion_authorized=False,
            ),
        )
        return summary
    except BaseException as error:
        write(
            folder / "failure.json",
            dict(
                error=repr(error),
                traceback=traceback.format_exc(),
                completed_attempts=len(attempts),
                automatic_retry=False,
                forecast_scored=False,
            ),
        )
        raise


def geometry_for(prep, blocks, width):
    def geometry(scales):
        named = dict(zip(prep.math_model.COMPONENTS[:width], scales, strict=True))
        if width == 5:
            named["persistent_linear"] = 0.0
        result = []
        for block in blocks:
            data = prep.math_model.kernel_geometry(block.coordinates, named)
            active = block.coordinates.kernels()[:width].reshape(width, -1)
            result.append(
                {
                    **data,
                    "cycle": block.coordinates.cycles[0],
                    "observations": len(block.residual),
                    "active_components": prep.math_model.COMPONENTS[:width],
                    "active_kernel_design_rank": int(np.linalg.matrix_rank(active)),
                }
            )
        return result

    return geometry


def verify_alias(plan):
    folds = {f["target"]: f for f in plan["folds"]}
    a, b = folds[2022], folds[2024]
    require(
        a["input_identity_sha256"] == b["input_identity_sha256"]
        and a["input_payload"]["sha256"] == b["input_payload"]["sha256"]
        and [i["sha256"] for i in a["blocks"]] == [i["sha256"] for i in b["blocks"]]
        and a["selected"]["sha256"] == b["selected"]["sha256"]
        and a["earlier_outcomes"]["sha256"] == b["earlier_outcomes"]["sha256"],
        "exact 2022/2024 alias identity required",
    )
    return folds


def run(receipt_pin, output):
    receipt, _ = sources.verify_execution(receipt_pin, output)
    prep = preparation()
    plan_path = DESIGN["references"][1]["path"]
    plan_sha = DESIGN["references"][1]["sha256"]
    plan, _, _ = prep.load_prepared(plan_path, plan_sha, 2022)
    folds = verify_alias(plan)
    prep.load_prepared(plan_path, plan_sha, 2024)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write(
        output / "launch.json",
        dict(
            run_id=receipt["run_id"],
            execution_receipt=receipt_pin,
            execution_design_sha256=sources.DESIGN_SHA,
            pid=os.getpid(),
            ppid=os.getppid(),
            utc=now(),
            runtime=sources.runtime_identity(),
            external_exec_handle="root_records_after_launch; not fabricated in advance",
        ),
    )
    print(
        json.dumps(
            dict(phase="launched", run_id=receipt["run_id"], pid=os.getpid(), ppid=os.getppid(), output=str(output))
        ),
        flush=True,
    )
    results = []
    fit_paths = {}
    try:
        for target in DESIGN["unique_training_targets"]:
            _, fold, blocks = prep.load_prepared(plan_path, plan_sha, target)
            for width, arm in zip((5, 6), DESIGN["arms"], strict=True):
                objective, medians = objective_for(prep, blocks, width)
                folder = output / "fits" / str(target) / arm
                metadata = dict(
                    target=target,
                    arm=arm,
                    components=prep.math_model.COMPONENTS[:width],
                    input_identity_sha256=fold["input_identity_sha256"],
                    rows=fold["rows"],
                    synthetic=False,
                )
                result = fit_job(objective, medians, folder, metadata, geometry_for(prep, blocks, width))
                results.append(
                    dict(
                        target=target,
                        arm=arm,
                        passed=result["gate"]["passed"],
                        terminal=sources.pin(folder / "terminal.json"),
                    )
                )
                fit_paths[(target, arm)] = folder
        for target in (2018, 2020, 2022, 2024):
            source = 2022 if target == 2024 else target
            folder = output / "folds" / str(target)
            folder.mkdir(parents=True, exist_ok=False)
            for arm in DESIGN["arms"]:
                write(
                    folder / (arm + ".json"),
                    dict(
                        target=target,
                        calibration_target=source,
                        aliased=target != source,
                        input_identity_sha256=folds[target]["input_identity_sha256"],
                        input_payload=folds[target]["input_payload"],
                        fit_terminal=sources.pin(fit_paths[(source, arm)] / "terminal.json"),
                        interpretation=(
                            "Exact calibration input reuse only; target fundamentals/features/forecasts are separate"
                        ),
                    ),
                )
        write(
            output / "summary.json",
            dict(
                results=results,
                fits=6,
                fixed_starts=18,
                alias_2024_to_2022=True,
                all_calibration_gates_passed=all(r["passed"] for r in results),
                objective_values_are_not_forecast_scores=True,
                forecast_scored=False,
                promotion_authorized=False,
            ),
        )
        write(
            output / "terminal.json",
            dict(
                status="completed_calibration"
                if all(r["passed"] for r in results)
                else "completed_with_calibration_failures_preserved",
                run_id=receipt["run_id"],
                summary=sources.pin(output / "summary.json"),
                automatic_retry=False,
                forecast_scored=False,
                promotion_authorized=False,
            ),
        )
    except BaseException as error:
        write(
            output / "failure.json",
            dict(
                run_id=receipt["run_id"],
                error=repr(error),
                traceback=traceback.format_exc(),
                completed_fits=results,
                automatic_restart=False,
                forecast_scored=False,
            ),
        )
        raise


def supplied_checks(output):
    """No optimizer: re-evaluate accepted prepared points through runner wrappers."""
    prep = preparation()
    rows = []
    for target in DESIGN["unique_training_targets"]:
        _, fold, blocks = prep.load_prepared(DESIGN["references"][1]["path"], DESIGN["references"][1]["sha256"], target)
        points = json.loads(sources.checked(fold["objective_points"]))
        for width in (5, 6):
            objective, _ = objective_for(prep, blocks, width)
            for index, point in enumerate(points):
                value, gradient = objective(np.array(point["log_scales"][:width]))
                key = "five" if width == 5 else "six"
                require(
                    abs(value - point[key]["value"]) <= 1e-10
                    and np.max(abs(gradient - np.asarray(point[key]["gradient"]))) <= 1e-10,
                    "accepted supplied-point mismatch",
                )
                rows.append(
                    dict(
                        target=target,
                        width=width,
                        point=index,
                        value=value,
                        gradient=gradient,
                        value_absolute_difference=abs(value - point[key]["value"]),
                        gradient_max_difference=float(np.max(abs(gradient - np.asarray(point[key]["gradient"])))),
                    )
                )
    output = Path(output).resolve()
    require(output.is_relative_to(HERE), "isolated supplied-check output required")
    write(
        output,
        dict(
            schema="MomentumCalibrationExecutionSuppliedChecksV1",
            points=rows,
            optimized=False,
            sampled=False,
            forecast_scored=False,
            runtime=sources.runtime_identity(),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-supplied", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--execution-receipt", type=Path)
    parser.add_argument("--execution-receipt-sha256")
    args = parser.parse_args()
    require(bool(args.check_supplied) != bool(args.run), "select supplied-point checks or bound execution")
    if args.check_supplied:
        supplied_checks(args.check_supplied)
    else:
        require(
            args.execution_receipt and args.execution_receipt_sha256, "externally pinned execution receipt required"
        )
        run(dict(path=str(args.execution_receipt), sha256=args.execution_receipt_sha256), args.run)


if __name__ == "__main__":
    main()
