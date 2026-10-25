# ruff: noqa: E402
"""Source-only/static inventories plus invented temporary predictions/outcomes."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "snapshots"))
sys.path.insert(0, str(HERE))
import numpy as np
import pytest
import score as driver
import score_reader as reader
from synthetic_files import js, make_bundle, nz, pin


@pytest.fixture
def packet(tmp_path):
    return make_bundle(tmp_path / "synthetic")


def test_positive_complete_gate_then_labels_and_exact_earlier_frequency(packet):
    external, admission, fixture = packet
    audit = []
    opened = []
    native = reader.checked_bytes

    def observed(pin_value):
        if pin_value["path"] == str(fixture["label_path"]):
            opened.append(pin_value["path"])
        return native(pin_value)

    with patch.object(reader, "checked_bytes", side_effect=observed):
        bundle = reader.verify_admission(external, audit=audit)
        assert opened == []
        labels = reader.read_labels_after_admission(bundle)
    assert len(opened) == 1
    assert labels["frequency_references"][2018]["probability"] == 0.5
    assert labels["frequency_references"][2018]["binary_training_count"] == 2
    assert len(labels["labels"]) == 5
    assert labels["labels"][fixture["races"][2]]["tie"] is True
    assert labels["audit"][-1] == "numeric_outcome_source_read_after_all_receipts"
    with pytest.raises(TypeError):
        bundle["folds"][2018]["mask"]["common_polled_race_ids"][0] = "foreign"
    with pytest.raises(ValueError):
        bundle["folds"][2018]["jobs"]["FS5"]["means"].setflags(write=True)


def test_forged_bundle_cannot_open_any_label_file():
    fake = dict(audit=["all_prediction_reference_mask_status_receipts_verified_before_numeric_label_read"])
    with patch.object(reader, "checked_bytes", side_effect=AssertionError("No file may open")):
        with pytest.raises(reader.ArtifactHardFailure, match="capability"):
            reader.read_labels_after_admission(fake)


def test_existing_output_rejected_before_label_or_prediction_io(packet, tmp_path):
    external, _, _ = packet
    output = tmp_path / "existing"
    output.mkdir()
    with patch.object(driver, "verify_admission", side_effect=AssertionError("Should not reach input loading")):
        with pytest.raises(reader.ArtifactHardFailure, match="new before label"):
            driver.run_scoring(external, output)


def test_output_failure_after_label_access_keeps_launch_and_failure_terminal(packet, tmp_path):
    external, _, _ = packet
    output = tmp_path / "failed"
    with patch.object(driver, "score_verified", side_effect=RuntimeError("invented supplied score failure")):
        with pytest.raises(RuntimeError, match="invented"):
            driver.run_scoring(external, output)
    assert (output / "launch.json").exists() and (output / "label-access.json").exists()
    terminal = json.loads((output / "terminal.json").read_text())
    assert terminal["status"] == "failed_preserved_no_repair_or_sampling"
    assert terminal["numeric_label_reader_returned"] is True and terminal["post_label_sampling"] is False


def test_full_synthetic_score_action_has_masks_ties_descriptives_and_no_fine_empirical_wis(packet, tmp_path):
    external, _, _ = packet
    output = tmp_path / "score-output"
    receipt = driver.run_scoring(external, output)
    report = json.loads((output / "scores.json").read_text())
    common = report["cycles"]["2018"]["masks"]["common"]
    assert common["race_ids"] == list(packet[2]["races"][:2])
    assert report["cycles"]["2018"]["masks"]["near_even"]["empty_mask_unavailable"]
    assert report["tiers"]["A"]["near_even"]["arms"]["FS5"]["crps"]["value"] is None
    assert report["tiers"]["A"]["near_even"]["primary_FS6_minus_FS5"]["crps"]["value"] is None
    for arm in ["FS5", "FS6", "LP5", "F0_archived_predictive", "PO5"]:
        row = common["arms"][arm]["rows"][0]
        assert "PIT" in row and "expected_margin" in row and "median_margin" in row
        assert "p_absolute_margin_le_02" in row and "p_absolute_margin_ge_10" in row
        assert len(row["margin_intervals"]) == 3
    assert common["arms"]["F0_archived_predictive"]["aggregates"]["wis"]["qualified"] is False
    assert common["arms"]["PO5"]["aggregates"]["wis"]["qualified"] is True
    assert receipt["post_label_sampling"] is False
    for item in receipt["artifacts"].values():
        assert pin(item["path"]) == item
    with np.load(output / "cycle-influences.npz", allow_pickle=False) as arrays:
        assert all(not arrays[name].dtype.hasobject for name in arrays.files)


def rebound_admission(external, admission):
    return js(external["path"], admission)


def label_guard(fixture):
    native = reader.checked_bytes
    opened = []

    def read(pin_value):
        if pin_value["path"] == str(fixture["label_path"]):
            opened.append(pin_value["path"])
        return native(pin_value)

    return read, opened


@pytest.mark.parametrize(
    "kind",
    [
        "omitted_runtime",
        "rebound_group",
        "wrong_review",
        "training_subset",
        "training_foreign",
        "orientation",
        "self_review",
        "self_code",
        "runtime_drift",
        "mixed_attempt",
    ],
)
def test_coherent_admission_rebindings_fail_before_any_label_open(packet, kind):
    external, admission, fixture = packet
    if kind == "omitted_runtime":
        admission["execution_identity"].pop("runtime_sha256")
    if kind == "rebound_group":
        admission["execution_identity"]["execution_group_id"] = "foreign-group"
    if kind == "wrong_review":
        admission["execution_identity"]["independent_execution_review"]["sha256"] = "0" * 64
    if kind == "training_subset":
        admission["selected_folds"][0]["training_race_ids"] = list(fixture["training"][:1])
    if kind == "training_foreign":
        admission["selected_folds"][0]["training_race_ids"] = [*fixture["training"], "2014:FOREIGN:senate:regular"]
    if kind == "mixed_attempt":
        admission["selected_folds"][0]["attempt"] = 1
    if kind == "orientation":
        masks = reader.json_pin(admission["mask_receipt"])
        masks["orientation"] = "R/(D+R)"
        admission["mask_receipt"] = js(admission["mask_receipt"]["path"], masks)
    if kind == "self_review":
        review = reader.json_pin(admission["independent_scoring_review"])
        review["manifest_sha256"] = "0" * 64
        admission["independent_scoring_review"] = js(admission["independent_scoring_review"]["path"], review)
    if kind == "self_code":
        manifest = reader.json_pin(admission["scoring_manifest"])
        manifest["files"][0]["sha256"] = "0" * 64
        admission["scoring_manifest"] = js(admission["scoring_manifest"]["path"], manifest)
    if kind == "runtime_drift":
        profile = reader.json_pin(admission["scoring_runtime"])
        profile["packages"]["scipy"] = "invented-incompatible"
        admission["scoring_runtime"] = js(admission["scoring_runtime"]["path"], profile)
    external = rebound_admission(external, admission)
    read, opened = label_guard(fixture)
    with patch.object(reader, "checked_bytes", side_effect=read):
        with pytest.raises(reader.ArtifactHardFailure):
            reader.verify_admission(external)
    assert opened == []


def update_summary(external, admission, fixture, modify, *, terminal_core=True):
    record = admission["selected_folds"][0]["jobs"]["FS6"]
    summary = reader.json_pin(record["numerical_status"])
    modify(summary)
    record["numerical_status"] = js(record["numerical_status"]["path"], summary)
    terminal = fixture["terminals"]["FS6"]
    terminal["prelabel_summary"] = record["numerical_status"]
    terminal["artifacts"]["prelabel_summary"] = record["numerical_status"]
    terminal["numerical_core_bulk_passed"] = terminal_core
    record["terminal"] = js(record["terminal"]["path"], terminal)
    return rebound_admission(external, admission)


def test_hidden_retry_trigger_and_triggered_initial_trio_cannot_be_success(packet):
    external, admission, fixture = packet

    def hidden(summary):
        summary["gates"]["core_state"] = False

    external = update_summary(external, admission, fixture, hidden, terminal_core=False)
    read, opened = label_guard(fixture)
    with patch.object(reader, "checked_bytes", side_effect=read):
        with pytest.raises(reader.ArtifactHardFailure, match="retry trigger inconsistency"):
            reader.verify_admission(external)
    assert opened == []

    def honest(summary):
        summary["permitted_retry_triggers"]["core_state"] = True
        summary["any_permitted_retry_trigger"] = True

    external = update_summary(external, admission, fixture, honest, terminal_core=False)
    with patch.object(reader, "checked_bytes", side_effect=read):
        with pytest.raises(reader.ArtifactHardFailure, match="Triggered initial"):
            reader.verify_admission(external)
    assert opened == []
    # Explicit immutable failed-experiment evidence permits descriptive math,
    # but orchestration must never turn it into a qualified selected trio.
    admission["selected_folds"][0]["frozen_failure_disposition"] = (
        "explicit_failed_unresolved_experiment_no_retry_or_qualification"
    )
    bundle = reader.verify_admission(rebound_admission(external, admission))
    assert bundle["folds"][2018]["frozen_failure_disposition"]


def test_genuine_failed_terminal_without_prediction_or_status_is_numerical_unresolved(packet):
    external, admission, fixture = packet
    record = admission["selected_folds"][0]["jobs"]["FS5"]
    terminal = fixture["terminals"]["FS5"]
    terminal["status"] = "failed_preserved_no_restart"
    terminal["exception"] = "invented failure before any prediction seal"
    admission["selected_folds"][0]["jobs"]["FS5"] = {"terminal": js(record["terminal"]["path"], terminal)}
    read, opened = label_guard(fixture)
    with patch.object(reader, "checked_bytes", side_effect=read):
        with pytest.raises(reader.NumericalEvidenceUnresolved) as caught:
            reader.verify_admission(rebound_admission(external, admission))
    assert opened == []
    assert list(caught.value.evidence["full_fixed_mask"]["common_polled_race_ids"]) == list(fixture["races"][:2])


def test_coherent_trace_original_components_truncation_rejected_by_declared_job(packet, tmp_path):
    from trace_diagnostics import save_trace

    external, admission, fixture = packet
    record = admission["selected_folds"][0]["jobs"]["FS5"]
    terminal = fixture["terminals"]["FS5"]
    trace = fixture["trace"]
    arrays = {k: v.copy() for k, v in trace["arrays"].items()}
    metadata = (
        reader.plain(trace["metadata"])
        if hasattr(reader, "plain")
        else json.loads(json.dumps(driver._plain(trace["metadata"])))
    )
    old, new = 1600, 1400
    for key, value in list(arrays.items()):
        if value.shape[:2] == (4, old):
            arrays[key] = value[:, :new].copy()
    arrays["bank_batch_weight_sums"] = arrays["bank_batch_weight_sums"][:, :14].copy()
    arrays["bank_batch_counts"] = arrays["bank_batch_counts"][:, :14].copy()
    arrays["bank_sampled_occupancy"] = np.stack(
        [np.bincount(chain, minlength=12) for chain in arrays["sampled_row_index"]]
    )
    metadata["draws"] = new
    saved = save_trace(dict(arrays=arrays, metadata=metadata), tmp_path / "coherent-short-trace")
    terminal["artifacts"]["trace"] = dict(
        path=str(tmp_path / "coherent-short-trace/receipt.json"), sha256=saved["receipt_sha256"]
    )
    original_receipt = reader.json_pin(terminal["artifacts"]["original"])
    original = reader.npz_record(original_receipt["arrays"])
    shortened = {k: v[:, :new].copy() if v.shape[:2] == (4, old) else v.copy() for k, v in original.items()}
    original_receipt["arrays"] = nz(original_receipt["arrays"]["file"]["path"], shortened)
    terminal["artifacts"]["original"] = js(terminal["artifacts"]["original"]["path"], original_receipt)
    components = reader.npz_record(terminal["artifacts"]["marginal_components"])
    terminal["artifacts"]["marginal_components"] = nz(
        terminal["artifacts"]["marginal_components"]["file"]["path"],
        {k: v[:, :new].copy() if v.shape[:2] == (4, old) else v.copy() for k, v in components.items()},
    )
    seal = reader.json_pin(record["prediction_seal"])
    seal["artifacts"]["original"] = terminal["artifacts"]["original"]
    seal["artifacts"]["trace"] = terminal["artifacts"]["trace"]
    record["prediction_seal"] = js(record["prediction_seal"]["path"], seal)
    record["terminal"] = js(record["terminal"]["path"], terminal)
    read, opened = label_guard(fixture)
    with patch.object(reader, "checked_bytes", side_effect=read):
        with pytest.raises(reader.ArtifactHardFailure, match="budget mismatch"):
            reader.verify_admission(rebound_admission(external, admission))
    assert opened == []


def test_prelabel_failure_cannot_gain_pure_math_qualification_and_display_is_separate():
    import copy

    from score_math import METRICS

    raw = dict(
        rows=[],
        aggregates={
            metric: dict(value=0.1, qualified=True, status="qualified_conditional_numerical") for metric in METRICS
        },
    )
    passed = driver.apply_prelabel_status(copy.deepcopy(raw), core_bulk_qualified=True)
    passed["prelabel_interval_display_qualified"] = False
    assert all(r["qualified"] for r in passed["aggregates"].values())
    failed = driver.apply_prelabel_status(copy.deepcopy(raw), core_bulk_qualified=False)
    assert all(not r["qualified"] and r["value"] == 0.1 for r in failed["aggregates"].values())
    frozen = driver.apply_prelabel_status(
        copy.deepcopy(raw), core_bulk_qualified=True, frozen_failure_disposition="invented-frozen-failure"
    )
    assert all(not r["qualified"] for r in frozen["aggregates"].values())


def test_larger_attempt_requires_exact_original_trio_before_any_label_io(packet):
    external, admission, fixture = packet
    terminal = dict(fixture["terminals"]["FS5"])
    terminal["job"] = {**terminal["job"], "attempt": 1, "warmup": 2000, "draws": 8000}
    terminal["output"] = "SYNTHETIC-larger-output"
    execution = {**admission["execution_identity"], "job": terminal["job"]}
    # Entire invented larger job metadata is coherent, but no original trio.
    with patch.object(reader, "_load_bound_retry_runner", side_effect=AssertionError("No validator import needed")):
        with pytest.raises(reader.ArtifactHardFailure, match="three original terminal"):
            reader.verify_retry_provenance(terminal, execution, admission["execution_manifest"])
    execution["attempt0_terminals"] = [dict(path="invented", sha256="0" * 64)] * 3
    terminal["provenance"] = {**terminal["provenance"], "attempt0_terminal_pins": []}
    with pytest.raises(reader.ArtifactHardFailure, match="pin mismatch"):
        reader.verify_retry_provenance(terminal, execution, admission["execution_manifest"])


def test_source_bound_runner_retry_validator_eligible_trio_and_invalid_trigger(tmp_path):
    from unittest.mock import patch

    from synthetic_files import LAB

    manifest = pin(LAB / "momentum-forecast-execution-sol-v1/artifact-manifest.json")
    runner = reader._load_bound_retry_runner(manifest)
    base = dict(
        execution_group_id="SYNTHETIC-only-metadata",
        execution_manifest_sha256=reader.ACCEPTED_EXECUTION_MANIFEST_SHA,
        plan_sha256=reader.ACCEPTED_PLAN_SHA,
        source_bindings_sha256=reader.ACCEPTED_BINDINGS_SHA,
        runtime_sha256=reader.ACCEPTED_RUNTIME_SHA,
        independent_execution_review=pin(LAB / "momentum-forecast-execution-review-sol-v1/receipt.json"),
    )
    expected = dict(
        source_identity="SYNTHETIC-bank-metadata",
        source_pins=dict(bank="synthetic-bank-pin"),
        variants={arm: dict(source_pins=dict(arm=arm, calibration="synthetic-calibration-pin")) for arm in reader.ARMS},
    )
    previous = []
    summaries = []
    for arm in reader.ARMS:
        job = runner.job_settings(2018, arm, 0)
        old_output = str(tmp_path / ("invented-initial-" + arm))
        run_id = "SYNTHETIC-" + arm
        root_receipt = js(
            tmp_path / (arm + "-root.json"),
            dict(
                schema="MomentumForecastRootExecutionReceiptV1",
                **base,
                action="job",
                job=job,
                output=old_output,
                run_id=run_id,
            ),
        )
        gates = dict(
            core_state=arm != "FS6",
            bulk_mean_share=True,
            bulk_absolute_win_probability=True,
            bank_chain_consistency=True,
        )
        summary = dict(
            schema="MomentumPollingPrelabelNumericalSummaryV1",
            source_identity=expected["source_identity"],
            synthetic=False,
            numeric_target_labels=False,
            scores=False,
            gates=gates,
            permitted_retry_triggers={key: not value for key, value in gates.items()},
            any_permitted_retry_trigger=not all(gates.values()),
        )
        summary_pin = js(tmp_path / (arm + "-summary.json"), summary)
        summaries.append((summary_pin, summary))
        provenance = dict(
            execution_receipt=root_receipt,
            run_id=run_id,
            source_identity=expected["source_identity"],
            forecast_design_sha256=reader.DESIGN_SHA,
            data_source_pins=expected["source_pins"],
            measurement_source_pins=expected["variants"][arm]["source_pins"],
        )
        old = dict(
            schema="MomentumForecastJobTerminalV1",
            job=job,
            status="complete_prelabel",
            synthetic=False,
            output=old_output,
            provenance=provenance,
            source_identity=expected["source_identity"],
            prelabel_summary=summary_pin,
            artifacts=dict(prelabel_summary=summary_pin),
        )
        previous.append(js(tmp_path / (arm + "-terminal.json"), old))
    execution = {**base, "attempt0_terminals": previous}
    terminal = dict(
        job=runner.job_settings(2018, "FS6", 1),
        output=str(tmp_path / "invented-larger"),
        source_identity=expected["source_identity"],
        provenance=dict(
            attempt0_terminal_pins=previous,
            data_source_pins=expected["source_pins"],
            measurement_source_pins=expected["variants"]["FS6"]["source_pins"],
        ),
    )
    with (
        patch.object(runner, "expected_adapter_report", return_value=expected),
        patch.object(reader, "_load_bound_retry_runner", return_value=runner),
    ):
        assert reader.verify_retry_provenance(terminal, execution, manifest) == previous
        # Rebind every original terminal consistently to all-passing statuses:
        # larger opportunity is now ineligible even though source/group match.
        corrected = []
        for old_pin, (summary_pin, summary) in zip(previous, summaries):
            summary["gates"] = {key: True for key in summary["gates"]}
            summary["permitted_retry_triggers"] = {key: False for key in summary["gates"]}
            summary["any_permitted_retry_trigger"] = False
            updated = js(summary_pin["path"], summary)
            old = reader.json_pin(old_pin)
            old["prelabel_summary"] = updated
            old["artifacts"]["prelabel_summary"] = updated
            corrected.append(js(old_pin["path"], old))
        execution["attempt0_terminals"] = corrected
        terminal["provenance"]["attempt0_terminal_pins"] = corrected
        with pytest.raises(reader.ArtifactHardFailure, match="no selected numerical trigger"):
            reader.verify_retry_provenance(terminal, execution, manifest)


def test_constructor_union_contract_caller_only_terminal_and_exact_stored_map(packet):
    from trace_diagnostics import REQUIRED_SOURCE_PINS, load_trace

    external, admission, fixture = packet
    terminal = fixture["terminals"]["FS5"]
    caller = terminal["provenance"]["trace_source_pins"]
    stored = fixture["trace"]["metadata"]["source_pins"]
    assert set(caller) == {"synthetic_bank"}
    assert set(stored) == set(caller) | set(REQUIRED_SOURCE_PINS)
    expected = reader.expected_trace_source_pins(caller)
    assert expected == dict(stored)
    # This is the old consumer contract that escaped the original full-map fixture.
    trace_pin = terminal["artifacts"]["trace"]
    with pytest.raises(ValueError, match="Trace source pins mismatch"):
        load_trace(
            Path(trace_pin["path"]).parent, expected_source_pins=caller, expected_receipt_sha256=trace_pin["sha256"]
        )
    bundle = reader.verify_admission(external)
    assert dict(bundle["folds"][2018]["jobs"]["FS5"]["trace"]["metadata"]["source_pins"]) == expected


def test_fixed_trace_pin_conflict_rejected_and_full_consistent_map_supported(packet):

    external, admission, fixture = packet
    caller = fixture["terminals"]["FS5"]["provenance"]["trace_source_pins"]
    expected = reader.expected_trace_source_pins(caller)
    assert reader.expected_trace_source_pins(expected) == expected
    conflicting = {**caller, "forecast_design": "0" * 64}
    with pytest.raises(reader.ArtifactHardFailure, match="conflicts with fixed"):
        reader.expected_trace_source_pins(conflicting)
    record = admission["selected_folds"][0]["jobs"]["FS5"]
    terminal = fixture["terminals"]["FS5"]
    terminal["provenance"]["trace_source_pins"] = conflicting
    record["terminal"] = js(record["terminal"]["path"], terminal)
    read, opened = label_guard(fixture)
    with patch.object(reader, "checked_bytes", side_effect=read):
        with pytest.raises(reader.ArtifactHardFailure, match="conflicts with fixed"):
            reader.verify_admission(rebound_admission(external, admission))
    assert opened == []


def test_unrecorded_caller_or_omitted_caller_pin_cannot_pass_as_subset(packet):
    external, admission, fixture = packet
    record = admission["selected_folds"][0]["jobs"]["FS5"]
    terminal = fixture["terminals"]["FS5"]
    original = dict(terminal["provenance"]["trace_source_pins"])
    read, opened = label_guard(fixture)
    for changed in [{**original, "unrecorded_job_pin": "c" * 64}, {}]:
        terminal["provenance"]["trace_source_pins"] = changed
        record["terminal"] = js(record["terminal"]["path"], terminal)
        with patch.object(reader, "checked_bytes", side_effect=read):
            with pytest.raises(ValueError, match="Trace source pins mismatch"):
                reader.verify_admission(rebound_admission(external, admission))
    assert opened == []
