"""Source-bound exclusive forecast jobs and separate prelabel references.

Author tests are synthetic. Actual historical target runs require an externally
SHA-bound root receipt and are launched separately by root, never by this file's
import, tests or preparation step.
"""

import argparse
import json
import os
import time
import traceback
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import execution_sources as sources

HERE = sources.HERE


def utc():
    return datetime.now(timezone.utc).isoformat()


def plain(value):
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def write_json(path, value):
    encoded = json.dumps(plain(value), sort_keys=True, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x") as stream:
        stream.write(encoded)


def write_npz(path, values):
    import numpy as np

    arrays = {key: np.asarray(value) for key, value in values.items()}
    sources.require(all(not value.dtype.hasobject for value in arrays.values()), "pickle/object outputs forbidden")
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return dict(
        file=sources.pin(path), arrays={key: dict(shape=list(v.shape), dtype=v.dtype.str) for key, v in arrays.items()}
    )


def job_seed(cycle, arm, attempt):
    import numpy as np

    sources.require(
        cycle in (2018, 2020, 2022, 2024) and arm in ("FS5", "FS6", "LP5") and attempt in (0, 1),
        "selected job identity required",
    )
    return int(
        np.random.SeedSequence([2026102501, cycle, {"FS5": 1, "FS6": 2, "LP5": 3}[arm], attempt]).generate_state(
            1, dtype=np.uint64
        )[0]
    )


def job_settings(cycle, arm, attempt):
    return dict(
        cycle=cycle,
        arm=arm,
        attempt=attempt,
        chains=4,
        warmup=1000 if attempt == 0 else 2000,
        draws=4000 if attempt == 0 else 8000,
        seed=job_seed(cycle, arm, attempt),
    )


def expected_adapter_report(cycle):
    bindings, _ = sources.verify_sources()
    item = next(
        row
        for row in bindings["sources"]
        if row["path"].endswith("momentum-forecast-adapter-sol-v1/artifact-manifest.json")
    )
    manifest = sources.json_pin(item)
    root = Path(item["path"]).parent
    row = next(row for row in manifest["files"] if row["path"] == f"verified-v1/{cycle}-adapter.json")
    return sources.json_pin({**row, "path": str(root / row["path"])})


def validate_retry(previous, cycle, output, binding):
    sources.require(len(previous) == 3, "attempt1 requires all three original arm terminals")
    expected = expected_adapter_report(cycle)
    records = [sources.json_pin(row) for row in previous]
    sources.require({r["job"]["arm"] for r in records} == {"FS5", "FS6", "LP5"}, "matched complete arm trio required")
    trigger = False
    for row in records:
        arm = row["job"]["arm"]
        sources.require(
            row["schema"] == "MomentumForecastJobTerminalV1"
            and row["job"] == job_settings(cycle, arm, 0)
            and row["status"] == "complete_prelabel"
            and row["synthetic"] is False,
            "retry requires original exact completed prelabel attempt0, never exceptions/timeouts",
        )
        sources.require(Path(row["output"]).resolve() != Path(output).resolve(), "new retry output required")
        provenance = row["provenance"]
        previous_receipt = sources.json_pin(provenance["execution_receipt"])
        for key in (
            "execution_group_id",
            "execution_manifest_sha256",
            "plan_sha256",
            "source_bindings_sha256",
            "runtime_sha256",
            "independent_execution_review",
        ):
            sources.require(
                previous_receipt[key] == binding[key], "retry execution group/source/runtime/review mismatch: " + key
            )
        sources.require(
            previous_receipt["schema"] == "MomentumForecastRootExecutionReceiptV1"
            and previous_receipt["action"] == "job"
            and previous_receipt["job"] == row["job"]
            and Path(previous_receipt["output"]).resolve() == Path(row["output"]).resolve()
            and previous_receipt["run_id"] == provenance["run_id"],
            "original receipt/job identity mismatch",
        )
        selected = expected["variants"][arm]["source_pins"]
        sources.require(
            row["source_identity"] == provenance["source_identity"] == expected["source_identity"]
            and provenance["forecast_design_sha256"] == sources.DESIGN_SHA
            and provenance["data_source_pins"] == expected["source_pins"]
            and provenance["measurement_source_pins"] == selected,
            "matched bank/target/calibration/observation source identities required",
        )
        sources.require(
            row["prelabel_summary"] == row["artifacts"]["prelabel_summary"], "summary pin identity mismatch"
        )
        summary = sources.json_pin(row["prelabel_summary"])
        sources.require(
            summary["schema"] == "MomentumPollingPrelabelNumericalSummaryV1"
            and summary["source_identity"] == expected["source_identity"]
            and summary["synthetic"] is False
            and summary["numeric_target_labels"] is False
            and summary["scores"] is False,
            "exact prelabel numerical status required",
        )
        gates = summary["gates"]
        selected_triggers = dict(
            core_state=not gates["core_state"],
            bulk_mean_share=not gates["bulk_mean_share"],
            bulk_absolute_win_probability=not gates["bulk_absolute_win_probability"],
            bank_chain_consistency=not gates["bank_chain_consistency"],
        )
        sources.require(
            summary["permitted_retry_triggers"] == selected_triggers
            and summary["any_permitted_retry_trigger"] == any(selected_triggers.values()),
            "retry trigger consistency required",
        )
        trigger = trigger or summary["any_permitted_retry_trigger"]
    sources.require(trigger, "no selected numerical trigger; tail/display/elapsed/score retry forbidden")
    return records


def validate_receipt(path, expected_sha):
    receipt_pin = dict(path=str(Path(path).resolve()), sha256=expected_sha)
    value = sources.json_pin(receipt_pin)
    sources.require(
        value["schema"] == "MomentumForecastRootExecutionReceiptV1"
        and isinstance(value["run_id"], str)
        and bool(value["run_id"].strip()),
        "unique source-bound run_id required",
    )
    sources.require(
        isinstance(value["execution_group_id"], str) and bool(value["execution_group_id"].strip()),
        "matched execution group required",
    )
    sources.verify_execution_manifest(value["execution_manifest_sha256"])
    bindings, plan = sources.verify_sources()
    sources.require(
        value["plan_sha256"] == sources.PLAN_SHA
        and value["source_bindings_sha256"] == sources.BINDINGS_SHA
        and value["runtime_sha256"] == sources.RUNTIME_SHA,
        "execution plan/source/runtime mismatch",
    )
    review = sources.json_pin(value["independent_execution_review"])
    sources.require(
        review["manifest_sha256"] == value["execution_manifest_sha256"]
        and review["disposition"].startswith("accepted"),
        "exact independent execution review required",
    )
    sources.require(value["action"] in ("job", "references"), "separate selected prelabel action required")
    output = Path(value["output"]).resolve()
    sources.require(output.is_relative_to(HERE) and output != HERE, "new isolated execution output required")
    if value["action"] == "job":
        job = value["job"]
        sources.require(
            job == job_settings(job["cycle"], job["arm"], job["attempt"]), "exact selected job/budget/seed required"
        )
        if job["attempt"] == 1:
            validate_retry(value["attempt0_terminals"], job["cycle"], output, value)
        else:
            sources.require(not value.get("attempt0_terminals"), "attempt0 must not silently reuse prior outputs")
    else:
        sources.require(value["cycles"] == [2018, 2020, 2022, 2024], "prelabel references require exact four-fold set")
    return value, receipt_pin, bindings, plan


class ProgressObserver:
    """Preserve initialized observer prefixes, never a resumable sampler state."""

    def __init__(self, collector, output, *, every=100):
        self.collector, self.output, self.every = collector, Path(output), every
        self.records, self.completed_records, self.last = 0, 0, None
        self.started = time.monotonic()
        self.saved_counts = [0] * collector.chains
        self.receipts = []

    def __call__(self, record):
        self.collector(record)
        self.records += 1
        self.last = dict(
            phase=record["phase"],
            chain=int(record["chain"]),
            retained_draw=record.get("retained_draw"),
            retained_draws=record.get("retained_draws"),
        )
        if record["phase"] == "completed_sweep":
            self.completed_records += 1
            if self.completed_records % self.every == 0:
                self.partial("periodic_observer_evidence")
        if record["phase"] == "chain_final_rng":
            self.partial("authentic_chain_final_rng_emitted")

    def partial(self, reason):
        exported = self.collector.export_partial()
        sequence = len(list(self.output.glob("partial-*-receipt.json")))
        prefix = self.output / f"partial-{sequence:05d}"
        # Retain initialized deltas; avoid quadratic duplication of long prefixes.
        selected = {}
        counts = exported["metadata"]["completed_draws_per_chain"]
        for name, value in exported["arrays"].items():
            if not name.startswith("chain"):
                selected[name] = value
                continue
            head, field = name.split("__", 1)
            chain = int(head[5:])
            start, stop = self.saved_counts[chain], counts[chain]
            if field in {"bank_batch_weight_sums", "bank_batch_counts"}:
                selected[name] = value[:, start // 100 : (stop + 99) // 100] if stop > start else value[:, :0]
            elif field in {"forecast_chain_index", "bank_sampled_occupancy"}:
                selected[name] = value
            else:
                selected[name] = value[:, start:stop]
        arrays = write_npz(prefix.with_suffix(".npz"), selected)
        metadata = dict(exported["metadata"])
        metadata.update(
            reason=reason,
            delta_start_completed_records_per_chain=list(self.saved_counts),
            delta_stop_completed_records_per_chain=list(counts),
            bank_sampled_occupancy_is_cumulative=True,
            bank_batch_updates_replace_original_batch_positions=True,
            utc=utc(),
            observer_records_seen=self.records,
            completed_sweep_observer_records=self.completed_records,
            last_record=self.last,
            elapsed_seconds=time.monotonic() - self.started,
            original_full_predictive_draw_completion_not_certified=True,
            completed_sweep_callback_precedes_missing_race_rng_and_output_assignment=True,
            resumable_sampler_state=False,
        )
        write_json(prefix.with_suffix(".json"), metadata)
        receipt = dict(
            schema="MomentumForecastPartialObserverEvidenceV1",
            arrays=arrays,
            metadata=sources.pin(prefix.with_suffix(".json")),
            reason=reason,
        )
        write_json(prefix.parent / (prefix.name + "-receipt.json"), receipt)
        print(
            json.dumps(
                dict(
                    progress=metadata["last_record"],
                    records=self.records,
                    completed_observer_sweeps=self.completed_records,
                    partial=str(prefix),
                )
            ),
            flush=True,
        )
        value = sources.pin(prefix.parent / (prefix.name + "-receipt.json"))
        self.receipts.append(value)
        self.saved_counts = list(counts)
        return value


def serialize_original(result, output):
    import numpy as np

    arrays, metadata = {}, {}
    for key, value in result.items():
        if isinstance(value, np.ndarray):
            arrays[key] = value
        else:
            metadata[key] = value
    array_receipt = write_npz(output / "original-predictions.npz", arrays)
    write_json(output / "original-metadata.json", metadata)
    value = dict(
        schema="MomentumOriginalJointPredictionsV1",
        arrays=array_receipt,
        metadata=sources.pin(output / "original-metadata.json"),
        all_bank_races_preserved=True,
        forecast_chain_draw_axes_preserved=True,
    )
    write_json(output / "original-receipt.json", value)
    return sources.pin(output / "original-receipt.json")


def run_prepared_job(prior, likelihood, modules, job, output, *, provenance, synthetic=False, precreated_launch=None):
    """Shared persistence path; tests pass explicitly labeled supplied tiny inputs."""
    import numpy as np

    output = Path(output).resolve()
    sources.require(output.is_relative_to(HERE) and output != HERE, "isolated new execution output required")
    if synthetic:
        sources.require(
            provenance.get("synthetic") is True
            and job.get("synthetic_budget") is True
            and prior.metadata.get("synthetic") is True,
            "explicit synthetic supplied-input scope required",
        )
    else:
        sources.require(
            job == job_settings(job["cycle"], job["arm"], job["attempt"]) and prior.beta.shape == (16000, 6),
            "exact actual settings and complete prior required",
        )
        bound, _, _, _ = validate_receipt(
            provenance["execution_receipt"]["path"], provenance["execution_receipt"]["sha256"]
        )
        sources.require(
            bound["action"] == "job" and bound["job"] == job and Path(bound["output"]).resolve() == output,
            "actual prepared job must match external receipt",
        )
    if precreated_launch is None:
        output.mkdir(parents=True, exist_ok=False)
        launch = dict(
            schema="MomentumForecastJobLaunchV1",
            utc=utc(),
            pid=os.getpid(),
            ppid=os.getppid(),
            job=job,
            synthetic=synthetic,
            provenance=provenance,
            output=str(output),
            exec_handle="recorded separately by root after tool launch returns",
            scientific_scope="conditional empirical-Bayes aggregate D/R share; no count qualification",
            BLAS_thread_environment={
                name: os.environ.get(name)
                for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
            },
        )
        write_json(output / "launch.json", launch)
    else:
        launch = sources.json_pin(precreated_launch)
        sources.require(
            set(p.name for p in output.iterdir()) == {"launch.json"}
            and Path(precreated_launch["path"]).resolve() == output / "launch.json"
            and launch["job"] == job
            and launch["output"] == str(output)
            and launch["provenance"]["execution_receipt"] == provenance["execution_receipt"],
            "exclusive fresh authenticated input-preparation launch required",
        )
    observer = None
    artifacts = {"launch": sources.pin(output / "launch.json")}
    try:
        observed = np.unique(likelihood.race_indices)
        missing = np.setdiff1d(np.arange(len(prior.race_ids)), observed)
        pins = dict(modules["trace"].REQUIRED_SOURCE_PINS)
        pins.update(provenance["trace_source_pins"])
        collector = modules["trace"].RetainedTraceCollector(
            chains=job["chains"],
            draws=job["draws"],
            race_ids=prior.race_ids,
            observed_race_indices=observed,
            missing_race_indices=missing,
            posterior_source_chain=prior.posterior_chain,
            posterior_source_draw=prior.posterior_draw,
            source_identity=prior.source_identity,
            source_pins=pins,
            batch_length=100,
        )
        observer = ProgressObserver(collector, output)
        result = modules["sampler"].sample_poll_posterior(
            prior,
            likelihood,
            chains=job["chains"],
            warmup=job["warmup"],
            draws=job["draws"],
            seed=job["seed"],
            observer=observer,
        )
        sources.require(
            result["source_identity"] == prior.source_identity and tuple(result["race_ids"]) == prior.race_ids,
            "returned complete original bank identity mismatch",
        )
        sources.require(
            result["logits"].shape == (job["chains"], job["draws"], len(prior.race_ids)),
            "complete original joint prediction dimensions required",
        )
        artifacts["original"] = serialize_original(result, output)
        trace = collector.export()
        trace_saved = modules["trace"].save_trace(trace, output / "trace")
        artifacts["trace"] = dict(path=str(output / "trace/receipt.json"), sha256=trace_saved["receipt_sha256"])
        # Freeze original/trace bytes before deterministic numerical processing.
        write_json(
            output / "prediction-seal.json",
            dict(
                schema="MomentumPollingPredictionSealV1",
                job=job,
                source_identity=prior.source_identity,
                synthetic=synthetic,
                artifacts=artifacts,
                numeric_target_labels=False,
                scores=False,
            ),
        )
        from numerical import conditional_summary

        summary, arrays = conditional_summary(result, trace, modules)
        summary["synthetic"] = synthetic
        summary["source_identity"] = prior.source_identity
        artifacts["marginal_components"] = write_npz(output / "prelabel-components.npz", arrays)
        write_json(output / "prelabel-summary.json", summary)
        artifacts["prelabel_summary"] = sources.pin(output / "prelabel-summary.json")
        terminal = dict(
            schema="MomentumForecastJobTerminalV1",
            status="complete_prelabel",
            utc=utc(),
            job=job,
            output=str(output),
            synthetic=synthetic,
            provenance=provenance,
            artifacts=artifacts,
            prelabel_summary=artifacts["prelabel_summary"],
            source_identity=prior.source_identity,
            numerical_core_bulk_passed=all(summary["gates"].values()),
            partial_observer_receipts=list(observer.receipts),
            numeric_target_labels=False,
            scores=False,
        )
        write_json(output / "terminal.json", terminal)
        return terminal
    except BaseException as error:
        failure = dict(
            schema="MomentumForecastJobTerminalV1",
            status="failed_preserved_no_restart",
            utc=utc(),
            job=job,
            output=str(output),
            synthetic=synthetic,
            provenance=provenance,
            artifacts=artifacts,
            exception_type=type(error).__name__,
            exception=str(error),
            traceback=traceback.format_exc(),
            numeric_target_labels=False,
            scores=False,
            original_full_draws_not_certified_from_partial=True,
        )
        if observer is not None:
            try:
                failure["partial_observer"] = observer.partial("caught_exception_diagnostic_evidence")
            except BaseException as partial_error:
                failure["partial_export_failure"] = dict(type=type(partial_error).__name__, error=str(partial_error))
        failure["partial_observer_receipts"] = list(observer.receipts) if observer is not None else []
        write_json(output / "terminal.json", failure)
        raise


def _actual_job_inputs(receipt, receipt_pin, modules, precreated_launch):
    data = modules["adapter"].load_adapter(receipt["job"]["cycle"])
    sampler = modules["sampler"]
    adapter = modules["adapter"]
    root = sources.LAB / "momentum-observed-sampler-sol-v1"
    prior, likelihood, constructor = data.build_for_sampler(
        sampler,
        receipt["job"]["arm"],
        adapter.trusted_pin(root / "observed_sampler.py"),
        [
            adapter.trusted_pin(root / "snapshots/src/forecasting/polling_v1.py"),
            adapter.trusted_pin(root / "snapshots/src/forecasting/response_v5_reuse.py"),
        ],
    )
    measurement = data.measurements[receipt["job"]["arm"]]
    trace_pins = dict(
        target_plan=adapter.BINDINGS["files"][1]["sha256"],
        target_joint=measurement.source_pins["target_joint"]["sha256"],
        target_design=data.source_pins["target_fold_design"]["sha256"],
        bank_reference=data.source_pins["bank_reference"]["sha256"],
        calibration_result=measurement.source_pins["calibration"]["result"]["sha256"],
        adapter_manifest="1fed21e47d1ae377bdce69a486011fe6a8172ede93e656b23845ab0b3933062d",
        execution_receipt=receipt_pin["sha256"],
        numerical_clarification=sources.CLARIFICATION_SHA,
        execution_plan=sources.PLAN_SHA,
        execution_bindings=sources.BINDINGS_SHA,
        paired_tail_manifest="f35357f4615757be5e6a4e51a60c0b47f13c88cd8613c53ae0511b8abd542198",
        trace_manifest="8e93d2d78f9efcadb208350ecf03872fdf6e5f929d437834d33058ea74a74513",
    )
    for index, raw in enumerate(data.source_pins["raw_banks"]):
        trace_pins[f"original_bank_chain_{index}"] = raw["sha256"]
    _, fold, _ = adapter.TARGET.load_prepared(
        adapter.BINDINGS["files"][1]["path"],
        adapter.BINDINGS["files"][1]["sha256"],
        receipt["job"]["cycle"],
        measurement.operator,
    )
    common = tuple(prior.race_ids[j] for j in __import__("numpy").unique(likelihood.race_indices))
    sources.require(
        common == tuple(fold["common_polled_race_ids"]), "exact selected ordered common-polled mask required"
    )
    provenance = dict(
        run_id=receipt["run_id"],
        execution_receipt=receipt_pin,
        source_identity=data.source_identity,
        constructors=constructor,
        data_source_pins=data.source_pins,
        measurement_source_pins=measurement.source_pins,
        trace_source_pins=trace_pins,
        settings="selected design exactly; fixed MAP scales",
        forecast_design_sha256=sources.DESIGN_SHA,
        numerical_clarification_sha256=sources.CLARIFICATION_SHA,
        execution_group_id=receipt["execution_group_id"],
        attempt0_terminal_pins=receipt.get("attempt0_terminals", []),
    )
    if receipt["job"]["attempt"] == 1:
        validate_retry(receipt["attempt0_terminals"], receipt["job"]["cycle"], receipt["output"], receipt)
    return run_prepared_job(
        prior,
        likelihood,
        modules,
        receipt["job"],
        receipt["output"],
        provenance=provenance,
        precreated_launch=precreated_launch,
    )


def actual_job(receipt, receipt_pin, modules):
    checked, _, _, _ = validate_receipt(receipt_pin["path"], receipt_pin["sha256"])
    sources.require(checked == receipt and receipt["action"] == "job", "root job receipt identity required")
    output = Path(receipt["output"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "launch.json",
        dict(
            schema="MomentumForecastJobLaunchV1",
            utc=utc(),
            pid=os.getpid(),
            ppid=os.getppid(),
            job=receipt["job"],
            synthetic=False,
            output=str(output),
            stage="preparing_authenticated_inputs",
            provenance=dict(
                run_id=receipt["run_id"],
                execution_receipt=receipt_pin,
                execution_group_id=receipt["execution_group_id"],
                forecast_design_sha256=sources.DESIGN_SHA,
                numerical_clarification_sha256=sources.CLARIFICATION_SHA,
            ),
            exec_handle="recorded separately by root after launch returns",
            scientific_scope="conditional empirical-Bayes aggregate D/R share; no count qualification",
            BLAS_thread_environment={
                name: os.environ.get(name)
                for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
            },
        ),
    )
    launch_pin = sources.pin(output / "launch.json")
    try:
        return _actual_job_inputs(receipt, receipt_pin, modules, launch_pin)
    except BaseException as error:
        if not (output / "terminal.json").exists():
            write_json(
                output / "terminal.json",
                dict(
                    schema="MomentumForecastJobTerminalV1",
                    status="failed_preserved_no_restart",
                    utc=utc(),
                    job=receipt["job"],
                    output=str(output),
                    synthetic=False,
                    stage="input_preparation",
                    execution_receipt=receipt_pin,
                    artifacts=dict(launch=launch_pin),
                    exception_type=type(error).__name__,
                    exception=str(error),
                    traceback=traceback.format_exc(),
                    numeric_target_labels=False,
                    scores=False,
                ),
            )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--receipt-sha256", required=True)
    args = parser.parse_args()
    receipt, receipt_pin, _, _ = validate_receipt(args.receipt, args.receipt_sha256)
    modules = sources.load_dependencies()
    if receipt["action"] == "job":
        terminal = actual_job(receipt, receipt_pin, modules)
    else:
        from references import actual_references

        terminal = actual_references(receipt, receipt_pin, modules)
    print(json.dumps(dict(terminal=sources.pin(Path(receipt["output"]) / "terminal.json"), status=terminal["status"])))


if __name__ == "__main__":
    main()
