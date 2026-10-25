"""Invented temporary scoring artifacts; never actual jobs/outcomes or a Gibbs run."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import score_reader as reader
from scipy.special import expit, log_ndtr
from trace_diagnostics import REQUIRED_SOURCE_PINS, RetainedTraceCollector, save_trace

LAB = Path("/home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024")


def pin(path):
    return dict(path=str(path), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())


def js(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    return pin(path)


def nz(path, values):
    np.savez_compressed(path, **values)
    return dict(
        file=pin(path),
        arrays={key: dict(shape=list(value.shape), dtype=value.dtype.str) for key, value in values.items()},
    )


def fixture_trace(races, identity, c=4, d=1600, *, caller_pins=None):
    caller_pins = dict(synthetic_bank="a" * 64) if caller_pins is None else dict(caller_pins)
    constructor_pins = dict(REQUIRED_SOURCE_PINS)
    constructor_pins.update(caller_pins)
    k = 12
    source_chain = np.arange(k) // 3
    source_draw = np.arange(k) % 3
    collector = RetainedTraceCollector(
        chains=c,
        draws=d,
        race_ids=races,
        observed_race_indices=[0, 1],
        missing_race_indices=[2],
        posterior_source_chain=source_chain,
        posterior_source_draw=source_draw,
        source_identity=identity,
        source_pins=constructor_pins,
        batch_length=100,
    )
    means = np.empty((c, d, 3))
    raw = np.empty((c, d, 3))
    index = np.empty((c, d), int)
    for chain in range(c):
        row = chain
        lam = np.ones(2)
        for draw in range(d):
            x = 0.03 * np.sin((draw + 17 * chain) / 13)
            mean = np.array([-0.2 + x, 0.3 + x, 0.02 * np.cos(draw / 17)])
            variance = np.array([0.25, 0.36, 0.01])
            cycle = mean[-1]
            theta = mean[:2] + 0.01 * np.cos(draw / 11)
            post_row = (chain + draw) % k
            post_lam = np.array([1.1 + 0.1 * np.cos(draw / 7), 1.2 + 0.1 * np.sin(draw / 7)])
            base = dict(chain=chain, retained_draw=draw)
            collector(
                dict(
                    phase="gaussian_conditional",
                    **base,
                    conditioning_row_index=int(row),
                    conditioning_lambda=lam,
                    mean_theta_cycle=mean,
                    marginal_variances=variance,
                )
            )
            collector(
                dict(
                    phase="row_weights",
                    **base,
                    conditioning_row_index=int(row),
                    conditioning_lambda=lam,
                    theta=theta,
                    cycle=cycle,
                    normalized_row_weights=np.full(k, 1 / k),
                )
            )
            collector(
                dict(
                    phase="completed_sweep",
                    **base,
                    sampled_row_index=int(post_row),
                    lambda_after_gamma=post_lam,
                    cycle=cycle,
                    missing_location=np.array([0.1 + cycle]),
                    missing_scale=0.4,
                    observed_race_indices=np.array([0, 1]),
                    missing_race_indices=np.array([2]),
                )
            )
            means[chain, draw] = mean
            raw[chain, draw] = np.r_[theta, 0.1 + cycle]
            index[chain, draw] = post_row
            row = post_row
            lam = post_lam
        # Invented provenance schema fixture, not a claim to have sampled a chain.
        collector(
            dict(
                phase="chain_final_rng",
                chain=chain,
                retained_draws=d,
                state=dict(bit_generator="PCG64", state=dict(state=chain, inc=2 * chain + 1), has_uint32=0, uinteger=0),
            )
        )
    trace = collector.export()
    original = dict(
        logits=raw,
        shares=expit(raw),
        first_wins=raw > 0,
        posterior_index=index,
        posterior_chain=source_chain[index],
        posterior_draw=source_draw[index],
        shared_cycle=means[:, :, -1],
        coefficients=np.stack([0.01 * np.sin(np.arange(c * d).reshape(c, d) / 17 + j) for j in range(6)], axis=2),
        tau_cycle=np.full((c, d), 0.2),
        tau_race=np.full((c, d), 0.4),
        shared_cycle_z=means[:, :, -1] / 0.2,
    )
    return trace, original


def make_bundle(folder, *, cycle=2018, d=1600):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    races = tuple(f"{cycle}:SYN_{letter}:senate:regular" for letter in "ABC")
    common = races[:2]
    training = (f"{cycle - 2}:SYN_X:senate:regular", f"{cycle - 2}:SYN_Y:senate:regular")
    identity = "synthetic-bank-only"
    group = "SYNTHETIC-matched-file-test"
    accepted = LAB / "momentum-forecast-execution-sol-v1"
    review = LAB / "momentum-forecast-execution-review-sol-v1/receipt.json"
    execution_identity = dict(
        execution_group_id=group,
        execution_manifest_sha256=reader.ACCEPTED_EXECUTION_MANIFEST_SHA,
        plan_sha256=reader.ACCEPTED_PLAN_SHA,
        source_bindings_sha256=reader.ACCEPTED_BINDINGS_SHA,
        runtime_sha256=reader.ACCEPTED_RUNTIME_SHA,
        independent_execution_review=pin(review),
    )
    bank_pin = js(
        folder / "synthetic-bank-reference.json", dict(year=cycle, training_race_ids=training, target_race_ids=races)
    )
    caller_trace_pins = dict(synthetic_bank="a" * 64)
    trace, original = fixture_trace(races, identity, d=d, caller_pins=caller_trace_pins)
    jobs = {}
    terminals = {}
    for arm in reader.ARMS:
        out = folder / arm
        out.mkdir()
        job = dict(
            cycle=cycle, arm=arm, attempt=0, chains=4, warmup=0, draws=d, seed={"FS5": 11, "FS6": 12, "LP5": 13}[arm]
        )
        execution = js(
            out / "root-execution.json",
            dict(schema="MomentumForecastRootExecutionReceiptV1", **execution_identity, action="job", job=job),
        )
        provenance = dict(
            execution_receipt=execution,
            execution_group_id=group,
            forecast_design_sha256=reader.DESIGN_SHA,
            numerical_clarification_sha256=reader.CLARIFICATION_SHA,
            source_identity=identity,
            trace_source_pins=dict(caller_trace_pins),
            data_source_pins=dict(bank_reference=bank_pin),
        )
        original_arrays = nz(out / "original-predictions.npz", original)
        original_meta = js(out / "original-metadata.json", dict(race_ids=races, source_identity=identity))
        original_receipt = js(
            out / "original-receipt.json",
            dict(schema="MomentumOriginalJointPredictionsV1", arrays=original_arrays, metadata=original_meta),
        )
        saved = save_trace(trace, out / "trace")
        trace_pin = dict(path=str(out / "trace/receipt.json"), sha256=saved["receipt_sha256"])
        artifacts = dict(original=original_receipt, trace=trace_pin)
        seal_pin = js(
            out / "prediction-seal.json",
            dict(
                schema="MomentumPollingPredictionSealV1",
                job=job,
                source_identity=identity,
                synthetic=True,
                artifacts=dict(artifacts),
                numeric_target_labels=False,
                scores=False,
            ),
        )
        means = trace["arrays"]["gaussian_mean_theta_cycle"][..., :-1]
        sd = np.sqrt(trace["arrays"]["gaussian_variance_theta_cycle"][..., :-1])
        components = dict(
            component_expected_shares=expit(means),
            component_log_win=log_ndtr(means / sd),
            component_log_lose=log_ndtr(-means / sd),
            quantile_levels=np.array([0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975]),
            quantiles=np.tile([0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.9], (2, 1)),
            common_polled_race_indices=np.array([0, 1]),
            common_polled_race_ids=np.array(common),
        )
        artifacts["marginal_components"] = nz(out / "prelabel-components.npz", components)
        side = dict(
            coordinates=[
                dict(
                    coordinate=r,
                    status="measured",
                    structural_constant=False,
                    gates=dict(relative_mcse=True, chain_log_difference=True, absolute_mcse=True),
                )
                for r in common
            ]
        )
        gates = dict(
            core_state=True, bulk_mean_share=True, bulk_absolute_win_probability=True, bank_chain_consistency=True
        )
        summary = dict(
            schema="MomentumPollingPrelabelNumericalSummaryV1",
            source_identity=identity,
            common_polled_race_ids=common,
            win_side=side,
            lose_side=side,
            interval_display_qualified=True,
            inherited_bank_warning=False,
            gates=gates,
            permitted_retry_triggers={key: False for key in gates},
            any_permitted_retry_trigger=False,
            numeric_target_labels=False,
            scores=False,
        )
        summary_pin = js(out / "prelabel-summary.json", summary)
        artifacts["prelabel_summary"] = summary_pin
        terminal = dict(
            schema="MomentumForecastJobTerminalV1",
            status="complete_prelabel",
            job=job,
            synthetic=True,
            provenance=provenance,
            artifacts=artifacts,
            prelabel_summary=summary_pin,
            source_identity=identity,
            numerical_core_bulk_passed=True,
            numeric_target_labels=False,
            scores=False,
        )
        terminal_pin = js(out / "terminal.json", terminal)
        jobs[arm] = dict(terminal=terminal_pin, prediction_seal=seal_pin, numerical_status=summary_pin)
        terminals[arm] = terminal
    refs = folder / "references"
    refs.mkdir()
    ref_execution = js(
        refs / "root-execution.json",
        dict(schema="MomentumForecastRootExecutionReceiptV1", **execution_identity, action="references"),
    )
    joint = 0.5 + 0.2 * np.sin(np.arange(4 * d * 5 * 3).reshape(4, d * 5, 3) / 29)
    f0summary = dict(
        schema="MomentumArchivedF0PrelabelReferenceV1",
        source_identity=identity,
        common_polled_race_ids=common,
        competitive_race_ids=common,
        near_even_race_ids=[],
        win_side=side,
        lose_side=side,
    )
    f0summary_pin = js(refs / "archived-F0-summary.json", f0summary)
    f0 = nz(
        refs / "archived-F0-cdf.npz",
        dict(
            joint_empirical_shares=joint,
            sorted_empirical_shares=np.sort(joint.reshape(-1, 3), axis=0),
            race_ids=np.array(races),
        ),
    )
    po5 = nz(
        refs / "PO5-analytic.npz",
        dict(mean_logit=np.array([-0.1, 0.3, 0.2]), covariance=np.eye(3) * 0.25, race_ids=np.array(races)),
    )
    po5summary = js(
        refs / "PO5-summary.json",
        dict(
            schema="MomentumAnalyticPO5PrelabelReferenceV1",
            source_identity=identity,
            common_polled_race_ids=common,
            rows=[dict(race_id=r, expected_share=float(expit(mu))) for r, mu in zip(races, [-0.1, 0.3, 0.2])],
        ),
    )
    references = js(
        refs / "terminal.json",
        dict(
            schema="MomentumPrelabelReferencesTerminalV1",
            status="complete_prelabel_references",
            execution_receipt=ref_execution,
            artifacts=[
                dict(
                    cycle=cycle,
                    archived_F0_summary=f0summary_pin,
                    archived_F0_arrays=f0,
                    PO5_summary=po5summary,
                    PO5_arrays=po5,
                )
            ],
            numeric_target_labels=False,
            scores=False,
        ),
    )
    mask = dict(
        full_bank_race_ids=races,
        common_polled_race_ids=common,
        competitive_race_ids=common,
        near_even_race_ids=[],
        archived_F0_summary=f0summary_pin,
    )
    mask_pin = js(
        folder / "masks.json",
        dict(
            schema="MomentumFrozenScoreMasksV1",
            orientation="D/(D+R)",
            forecast_design_sha256=reader.DESIGN_SHA,
            folds={str(cycle): mask},
        ),
    )
    label_path = folder / "SYNTHETIC-outcomes.csv"
    with label_path.open("w") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["cycle", "race_id", "dem_votes", "rep_votes", "two_party_votes", "actual_dem_share"]
        )
        writer.writeheader()
        for rid, dem, rep in [
            (races[0], 65, 35),
            (races[1], 40, 60),
            (races[2], 50, 50),
            (training[0], 60, 40),
            (training[1], 30, 70),
        ]:
            writer.writerow(
                dict(
                    cycle=int(rid.split(":")[0]),
                    race_id=rid,
                    dem_votes=dem,
                    rep_votes=rep,
                    two_party_votes=dem + rep,
                    actual_dem_share=dem / (dem + rep),
                )
            )
        # Foreign/future rows have deliberately nonnumeric outcome text and must
        # be filtered by identity metadata before numeric parsing.
        writer.writerow(
            dict(
                cycle=2098,
                race_id="2098:FOREIGN:senate:regular",
                dem_votes="not-parsed",
                rep_votes="not-parsed",
                two_party_votes="not-parsed",
                actual_dem_share="not-parsed",
            )
        )
    admission = dict(
        schema="MomentumForecastScoringAdmissionV1",
        forecast_design_sha256=reader.DESIGN_SHA,
        numerical_clarification_sha256=reader.CLARIFICATION_SHA,
        synthetic=True,
        source_manifests=reader.REQUIRED_SOURCES,
        execution_identity=execution_identity,
        execution_manifest=pin(accepted / "artifact-manifest.json"),
        runtime_receipt=pin(accepted / "runtime.json"),
        selected_folds=[dict(cycle=cycle, attempt=0, training_race_ids=training, jobs=jobs)],
        references_terminal=references,
        mask_receipt=mask_pin,
        label_source=pin(label_path),
    )
    score_here = Path(__file__).resolve().parent
    own_files = [
        dict(path=name, sha256=hashlib.sha256((score_here / name).read_bytes()).hexdigest())
        for name in ["score.py", "score_math.py", "score_reader.py"]
    ]
    own_manifest = js(folder / "synthetic-scorer-manifest.json", dict(files=own_files))
    own_review = js(
        folder / "synthetic-scorer-review.json",
        dict(
            manifest_sha256=own_manifest["sha256"],
            disposition="accepted_synthetic_fixture_only",
            synthetic_fixture_only=True,
        ),
    )
    own_runtime = js(folder / "synthetic-scorer-runtime.json", reader.runtime_identity())
    admission.update(scoring_manifest=own_manifest, independent_scoring_review=own_review, scoring_runtime=own_runtime)
    return (
        js(folder / "admission.json", admission),
        admission,
        dict(
            terminals=terminals, trace=trace, original=original, races=races, training=training, label_path=label_path
        ),
    )
