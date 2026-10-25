"""Receipt-first scoring admission; target numeric label I/O occurs at one final boundary.

Author tests use synthetic files only. This module never discovers masks, chooses
attempts, fits models, samples, repairs artifacts or retries numeric computation.
"""

import csv
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from observed_sampler import _observed_copy
from scipy.special import expit, log_ndtr
from trace_diagnostics import REQUIRED_SOURCE_PINS, load_trace

DESIGN_SHA = "45707d4a933ded2f7d4bffcbaab0fb6fb9a4ab7b1d62ad4494a27422b7d75678"
CLARIFICATION_SHA = "0d6dd97153687df2583c0a086e92e8a42c22eb0c850ed0a0133acdcb87c14024"
ACTUAL_OUTCOME_SOURCE_SHA = "5f5cafa1afc53387331d4e75b3c65f92ed4cab3a91578dcb7cb0815357b029ad"
REQUIRED_SOURCES = dict(
    math_manifest="2bfa468bca013c7bfa7d6e80395ba7534947026ff35209fd2d51fed1b40f28c8",
    paired_manifest="f35357f4615757be5e6a4e51a60c0b47f13c88cd8613c53ae0511b8abd542198",
    paired_review="8868d1ff239e294201a96e385ad6ea10714e4e52194af9ffd0b6cb407e89ab19",
    trace_manifest="8e93d2d78f9efcadb208350ecf03872fdf6e5f929d437834d33058ea74a74513",
)
ACCEPTED_EXECUTION_MANIFEST_SHA = "e74bc838eaa859a55d9ebe1811d1bce1c2647ca8f428707192dc036bd844795d"
ACCEPTED_EXECUTION_REVIEW_SHA = "b736fe648823de95a34612f02b5794136d19835fa33539629e9e6f65ab4c2380"
ACCEPTED_PLAN_SHA = "103cd92c48bccdb388f25b0b58aa15efb938b5282824419c615dbf85ea24c755"
ACCEPTED_BINDINGS_SHA = "b87f10262822c57a4a733a130a5ccc73e858d312898e9c8d4ed802d11f94934a"
ACCEPTED_RUNTIME_SHA = "01ff4314a314fcb315719c0efb30062d9ee81878fa12ae428f7f1645f56cd354"
ARMS = ("FS5", "FS6", "LP5")
CYCLES = (2018, 2020, 2022, 2024)


class ArtifactHardFailure(ValueError):
    pass


class NumericalEvidenceUnresolved(RuntimeError):
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = _observed_copy(evidence)


_VERIFIED_TOKEN = object()


class VerifiedAdmission:
    """Immutable verified capability; ordinary audit strings cannot open labels."""

    __slots__ = ("_data", "_token")

    def __init__(self, data, token):
        require(token is _VERIFIED_TOKEN, "Verified admission capability required")
        object.__setattr__(self, "_data", _observed_copy(data))
        object.__setattr__(self, "_token", token)

    def __getitem__(self, key):
        return self._data[key]

    def __setattr__(self, key, value):
        raise AttributeError("Verified admission is immutable")


def require(ok, message):
    if not ok:
        raise ArtifactHardFailure(message)


def checked_bytes(pin):
    require(isinstance(pin, Mapping) and set(pin) == {"path", "sha256"}, "Exact external file pin required")
    content = Path(pin["path"]).read_bytes()
    require(hashlib.sha256(content).hexdigest() == pin["sha256"], "Artifact byte digest mismatch: " + pin["path"])
    return content


def json_pin(pin):
    return json.loads(checked_bytes(pin))


def npz_record(record):
    require(set(record) == {"file", "arrays"}, "Source-bound NPZ receipt required")
    with np.load(io.BytesIO(checked_bytes(record["file"])), allow_pickle=False) as archive:
        require(set(archive.files) == set(record["arrays"]), "NPZ array schema mismatch")
        values = {name: archive[name].copy() for name in archive.files}
    for name, value in values.items():
        spec = record["arrays"][name]
        require(
            not value.dtype.hasobject and list(value.shape) == spec["shape"] and value.dtype.str == spec["dtype"],
            "NPZ array dtype/shape mismatch: " + name,
        )
        if value.dtype.kind in "f":
            require(not np.isnan(value).any() and not np.isposinf(value).any(), "Nonfinite NPZ numeric values")
        value.setflags(write=False)
    return values


def runtime_identity():
    return dict(
        schema="MomentumForecastAdapterRuntimeV1",
        executable=os.path.abspath(sys.executable),
        prefix=sys.prefix,
        python=sys.version,
        packages={
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "pandas", "pydantic", "arviz", "h5py", "h5netcdf")
        },
    )


def verify_scoring_self(admission):
    """Externally bound own inventory/review/runtime before numeric forecast I/O."""
    here = Path(__file__).resolve().parent
    manifest_pin = admission["scoring_manifest"]
    manifest = json_pin(manifest_pin)
    if not admission["synthetic"]:
        require(
            Path(manifest_pin["path"]).resolve() == here / "artifact-manifest.json",
            "Actual scorer manifest must bind this exact source packet",
        )
    inventory = {item["path"]: item for item in manifest["files"]}
    require(
        {"score.py", "score_math.py", "score_reader.py"}.issubset(inventory), "Scorer entrypoint inventory incomplete"
    )
    for item in manifest["files"]:
        path = (here / item["path"]).resolve()
        require(path.is_relative_to(here), "Scorer source escaped packet")
        checked_bytes(dict(path=str(path), sha256=item["sha256"]))
    review = json_pin(admission["independent_scoring_review"])
    require(
        review["manifest_sha256"] == manifest_pin["sha256"] and review["disposition"].startswith("accepted"),
        "Exact independent scoring review required",
    )
    require(
        admission["synthetic"] or not review.get("synthetic_fixture_only", False),
        "Synthetic fixture review cannot authorize actual scoring",
    )
    expected_runtime = json_pin(admission["scoring_runtime"])
    require(expected_runtime == runtime_identity(), "Scoring interpreter/prefix/package drift")


def identities(value, expected):
    for key, reference in expected.items():
        require(value.get(key) == reference, "Execution/source/runtime identity mismatch: " + key)


def exact_ids(values, label):
    ids = tuple(values)
    require(
        len(set(ids)) == len(ids) and all(isinstance(x, str) and x for x in ids), "Unique named " + label + " required"
    )
    return ids


def side_flags(summary, side, names):
    coordinate = summary[side + "_side"]["coordinates"]
    require(tuple(row["coordinate"] for row in coordinate) == tuple(names), "Prelabel side race axis mismatch")
    flags = []
    for row in coordinate:
        gates = row.get("gates", {})
        known = row.get("status") == "structural" and row.get("structural_constant") is True
        measured = row.get("status") == "measured"
        flags.append(
            bool(
                (known or measured)
                and all(
                    key in gates and (gates[key] is True or (known and gates[key] is None))
                    for key in ["relative_mcse", "chain_log_difference"]
                )
            )
        )
    return flags


def _load_bound_retry_runner(manifest_pin):
    """Import only source-authenticated runner validation; never call a sampler."""
    root = Path(manifest_pin["path"]).resolve().parent
    require(manifest_pin["sha256"] == ACCEPTED_EXECUTION_MANIFEST_SHA, "Accepted retry-validator source required")
    manifest = json_pin(manifest_pin)
    files = {row["path"]: row for row in manifest["files"]}
    for name in ["runner.py", "execution_sources.py"]:
        checked_bytes(dict(path=str(root / name), sha256=files[name]["sha256"]))

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    previous = sys.modules.get("execution_sources")
    try:
        sources = load("execution_sources", root / "execution_sources.py")
        runner = load("_score_bound_prelabel_retry_validator", root / "runner.py")
        require(runner.sources is sources, "Bound retry-validator dependency mismatch")
        return runner
    finally:
        if previous is None:
            sys.modules.pop("execution_sources", None)
        else:
            sys.modules["execution_sources"] = previous


def verify_retry_provenance(terminal, execution, manifest_pin):
    """Exact prelabel original-trio eligibility, before outcome access."""
    job = terminal["job"]
    provenance = terminal["provenance"]
    if job["attempt"] == 0:
        require(
            not execution.get("attempt0_terminals") and not provenance.get("attempt0_terminal_pins"),
            "Initial attempt must not rebind a retry lineage",
        )
        return []
    previous = execution.get("attempt0_terminals")
    require(
        isinstance(previous, list) and len(previous) == 3, "Larger attempt requires exact three original terminal pins"
    )
    require(provenance.get("attempt0_terminal_pins") == previous, "Execution/provenance original-trio pin mismatch")
    runner = _load_bound_retry_runner(manifest_pin)
    try:
        records = runner.validate_retry(previous, job["cycle"], terminal["output"], execution)
    except (ValueError, KeyError, OSError) as error:
        raise ArtifactHardFailure("Original-trio retry eligibility failed: " + str(error)) from error
    original = next(row for row in records if row["job"]["arm"] == job["arm"])
    require(
        terminal["source_identity"] == original["source_identity"]
        and provenance["data_source_pins"] == original["provenance"]["data_source_pins"]
        and provenance["measurement_source_pins"] == original["provenance"]["measurement_source_pins"],
        "Larger attempt changed original bank/target/measurement/calibration identity",
    )
    return previous


def expected_trace_source_pins(caller_pins):
    """Reproduce the sealed runner's fixed+caller constructor contract exactly.

    The collector requires fixed pins; the runner supplies their union with its
    job-specific provenance map. Never infer expected pins from stored metadata.
    """
    require(isinstance(caller_pins, Mapping), "Caller trace-source pin mapping required")
    expected = dict(REQUIRED_SOURCE_PINS)
    for key, value in caller_pins.items():
        require(
            isinstance(key, str)
            and isinstance(value, str)
            and len(value) == 64
            and all(c in "0123456789abcdef" for c in value),
            "SHA256 caller trace-source pins required",
        )
        if key in REQUIRED_SOURCE_PINS:
            require(
                value == REQUIRED_SOURCE_PINS[key], "Caller trace-source pin conflicts with fixed source/design: " + key
            )
        expected[key] = value
    return expected


def _verify_job(record, fold, arm, mask, expected_identity, synthetic, execution_manifest):
    terminal = json_pin(record["terminal"])
    require(terminal.get("schema") == "MomentumForecastJobTerminalV1", "Known job terminal schema required")
    if terminal.get("status") != "complete_prelabel":
        raise NumericalEvidenceUnresolved(
            "Selected prediction computation incomplete; numeric labels remain closed",
            dict(
                cycle=fold["cycle"],
                arm=arm,
                terminal_pin=record["terminal"],
                numerical_status_pin=record.get("numerical_status"),
                status=terminal.get("status"),
                full_fixed_mask=mask,
            ),
        )
    require(
        set(record) == {"terminal", "prediction_seal", "numerical_status"},
        "Exact complete selected-job pin inventory required",
    )
    seal = json_pin(record["prediction_seal"])
    summary = json_pin(record["numerical_status"])
    require(seal.get("schema") == "MomentumPollingPredictionSealV1", "Exact prediction seal required")
    job = terminal["job"]
    require(job == seal["job"], "Terminal/prediction selected-job mismatch")
    require(
        job["cycle"] == fold["cycle"] and job["arm"] == arm and job["attempt"] == fold["attempt"],
        "Mixed fold/arm/selected attempts forbidden",
    )
    require(
        terminal["synthetic"] is synthetic and seal["synthetic"] is synthetic, "Synthetic/actual admission mismatch"
    )
    if not synthetic:
        require(
            job["chains"] == 4
            and job["draws"] == (4000 if fold["attempt"] == 0 else 8000)
            and job["warmup"] == (1000 if fold["attempt"] == 0 else 2000),
            "Selected sampler budget mismatch",
        )
        seed = int(
            np.random.SeedSequence(
                [2026102501, fold["cycle"], {"FS5": 1, "FS6": 2, "LP5": 3}[arm], fold["attempt"]]
            ).generate_state(1, dtype=np.uint64)[0]
        )
        require(job["seed"] == seed, "Selected independent arm stream mismatch")
    provenance = terminal["provenance"]
    execution = json_pin(provenance["execution_receipt"])
    require(
        execution.get("schema") == "MomentumForecastRootExecutionReceiptV1", "Known Root execution receipt required"
    )
    identities(execution, expected_identity)
    identities(
        provenance,
        {
            "execution_group_id": expected_identity["execution_group_id"],
            "forecast_design_sha256": DESIGN_SHA,
            "numerical_clarification_sha256": CLARIFICATION_SHA,
        },
    )
    require(execution["action"] == "job" and execution["job"] == job, "Execution receipt selected-job mismatch")
    retry_lineage = verify_retry_provenance(terminal, execution, execution_manifest)
    require(
        terminal["source_identity"] == seal["source_identity"] == provenance["source_identity"],
        "Fundamentals source identity mismatch",
    )
    bank = json_pin(provenance["data_source_pins"]["bank_reference"])
    require(
        tuple(bank["training_race_ids"]) == tuple(fold["training_race_ids"])
        and bank["year"] == fold["cycle"]
        and tuple(bank["target_race_ids"]) == tuple(mask["full_bank_race_ids"]),
        "Exact admitted bank training/target scope required",
    )
    require(
        terminal["prelabel_summary"] == record["numerical_status"]
        and terminal["artifacts"]["prelabel_summary"] == record["numerical_status"],
        "Externally expected numerical status pin required",
    )
    require(
        terminal.get("numeric_target_labels") is False
        and terminal.get("scores") is False
        and summary.get("numeric_target_labels") is False
        and summary.get("scores") is False,
        "Prediction/status must be prelabel",
    )
    for key in ["original", "trace"]:
        require(terminal["artifacts"][key] == seal["artifacts"][key], "Prediction seal/terminal artifact mismatch")
    original_receipt = json_pin(terminal["artifacts"]["original"])
    require(
        original_receipt.get("schema") == "MomentumOriginalJointPredictionsV1", "Original full joint receipt required"
    )
    original = npz_record(original_receipt["arrays"])
    original_meta = json_pin(original_receipt["metadata"])
    require(
        original_meta["source_identity"] == terminal["source_identity"], "Original/terminal source identity mismatch"
    )
    require(
        tuple(original_meta["race_ids"]) == tuple(mask["full_bank_race_ids"]), "Full-bank original event axis mismatch"
    )
    trace_pin = terminal["artifacts"]["trace"]
    checked_bytes(trace_pin)
    trace = load_trace(
        Path(trace_pin["path"]).parent,
        expected_source_pins=expected_trace_source_pins(provenance["trace_source_pins"]),
        expected_receipt_sha256=trace_pin["sha256"],
    )
    c, d, n = job["chains"], job["draws"], len(mask["full_bank_race_ids"])
    require(
        trace["metadata"]["chains"] == c and trace["metadata"]["draws"] == d,
        "Selected job/trace chain-draw budget mismatch",
    )
    require(
        original["logits"].shape == (c, d, n)
        and original["shares"].shape == (c, d, n)
        and original["posterior_index"].shape == (c, d),
        "Complete original prediction chain/draw/race dimensions required",
    )
    require(np.array_equal(original["shares"], expit(original["logits"])), "Original logit/share quantity mismatch")
    source_chain = trace["arrays"]["posterior_source_chain"]
    source_draw = trace["arrays"]["posterior_source_draw"]
    if not synthetic:
        require(
            np.array_equal(source_chain, np.repeat(np.arange(4), 4000))
            and np.array_equal(source_draw, np.tile(np.arange(4000), 4)),
            "Full original16k chain-major bank source map required",
        )
    index = original["posterior_index"]
    require(
        index.dtype.kind in "iu" and (index >= 0).all() and (index < len(source_chain)).all(),
        "Valid original categorical source rows required",
    )
    require(
        np.array_equal(original["posterior_chain"], source_chain[index])
        and np.array_equal(original["posterior_draw"], source_draw[index]),
        "Original sampled source-chain/draw identity mismatch",
    )
    component = npz_record(terminal["artifacts"]["marginal_components"])
    names = exact_ids(component["common_polled_race_ids"].tolist(), "common-polled race axis")
    require(
        names == tuple(mask["common_polled_race_ids"]) == tuple(summary["common_polled_race_ids"]),
        "Exact common fixed scoring event axis mismatch",
    )
    m = len(names)
    require(
        all(
            component[key].shape == (c, d, m)
            for key in ["component_expected_shares", "component_log_win", "component_log_lose"]
        ),
        "Complete component chain/draw/common dimensions required",
    )
    require(
        np.isfinite(component["component_expected_shares"]).all()
        and ((component["component_expected_shares"] >= 0) & (component["component_expected_shares"] <= 1)).all(),
        "Bounded conditional expected shares required",
    )
    require(
        np.array_equal(component["quantile_levels"], [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975])
        and component["quantiles"].shape == (m, 7)
        and np.isfinite(component["quantiles"]).all()
        and ((component["quantiles"] >= 0) & (component["quantiles"] <= 1)).all()
        and (np.diff(component["quantiles"], axis=1) >= 0).all(),
        "Exact bounded quantile axis required",
    )
    axes = np.asarray(component["common_polled_race_indices"])
    require(np.array_equal(axes, trace["arrays"]["observed_race_indices"]), "Conditional/trace observed axes mismatch")
    require(tuple(original_meta["race_ids"][j] for j in axes) == names, "Original/conditional event identity mismatch")
    require(
        np.array_equal(original["posterior_index"], trace["arrays"]["sampled_row_index"])
        and np.array_equal(original["shared_cycle"], trace["arrays"]["cycle"]),
        "Original/trace posterior draw identity mismatch",
    )
    means = trace["arrays"]["gaussian_mean_theta_cycle"][..., :-1]
    sds = np.sqrt(trace["arrays"]["gaussian_variance_theta_cycle"][..., :-1])
    require(
        np.array_equal(component["component_log_win"], log_ndtr(means / sds))
        and np.array_equal(component["component_log_lose"], log_ndtr(-means / sds)),
        "Conditional log-tail arrays/source parameters mismatch",
    )
    require(
        np.array_equal(original["logits"][..., axes], trace["arrays"]["observed_theta"]),
        "Original/trace observed theta mismatch",
    )
    require(summary["source_identity"] == terminal["source_identity"], "Numerical/fundamentals source mismatch")
    required_gates = ("core_state", "bulk_mean_share", "bulk_absolute_win_probability", "bank_chain_consistency")
    calculated_core_bulk = all(key in summary["gates"] and summary["gates"][key] is True for key in required_gates)
    require(
        terminal["numerical_core_bulk_passed"] is calculated_core_bulk, "Terminal/status numerical gate inconsistency"
    )
    retry_keys = dict(
        core_state="core_state",
        bulk_mean_share="bulk_mean_share",
        bulk_absolute_win_probability="bulk_absolute_win_probability",
        bank_chain_consistency="bank_chain_consistency",
    )
    require(
        set(summary["gates"]) == set(required_gates)
        and all(type(value) is bool for value in summary["gates"].values()),
        "Exact boolean prelabel gate inventory required",
    )
    expected_retry = {name: not summary["gates"][gate] for name, gate in retry_keys.items()}
    require(
        summary.get("permitted_retry_triggers") == expected_retry
        and type(summary.get("any_permitted_retry_trigger")) is bool
        and summary["any_permitted_retry_trigger"] == any(expected_retry.values()),
        "Numerical status/retry trigger inconsistency",
    )
    return dict(
        job=job,
        means=means,
        sds=sds,
        components=component,
        summary=summary,
        original=original,
        trace=trace,
        win_tail_qualified=side_flags(summary, "win", names),
        lose_tail_qualified=side_flags(summary, "lose", names),
        core_bulk_qualified=calculated_core_bulk,
        terminal_pin=record["terminal"],
        numerical_status_pin=record.get("numerical_status"),
        attempt0_terminal_pins=retry_lineage,
    )


def verify_admission(admission_pin, *, audit=None):
    """Every prediction/reference/status/mask and source pin before label I/O."""
    audit = [] if audit is None else audit
    admission = json_pin(admission_pin)
    require(admission.get("schema") == "MomentumForecastScoringAdmissionV1", "Root-selected scoring admission required")
    require(
        admission["forecast_design_sha256"] == DESIGN_SHA
        and admission["numerical_clarification_sha256"] == CLARIFICATION_SHA,
        "Selected design/clarification mismatch",
    )
    require(type(admission["synthetic"]) is bool, "Explicit synthetic/actual score admission required")
    require(admission["source_manifests"] == REQUIRED_SOURCES, "Accepted math/paired/trace source versions required")
    verify_scoring_self(admission)
    identity = admission["execution_identity"]
    required_identity = {
        "execution_group_id",
        "execution_manifest_sha256",
        "plan_sha256",
        "source_bindings_sha256",
        "runtime_sha256",
        "independent_execution_review",
    }
    require(
        set(identity) == required_identity and bool(identity.get("execution_group_id")),
        "Exact matched source/runtime/review execution identity required",
    )
    require(
        identity["execution_manifest_sha256"] == ACCEPTED_EXECUTION_MANIFEST_SHA
        and identity["plan_sha256"] == ACCEPTED_PLAN_SHA
        and identity["source_bindings_sha256"] == ACCEPTED_BINDINGS_SHA
        and identity["runtime_sha256"] == ACCEPTED_RUNTIME_SHA
        and identity["independent_execution_review"]["sha256"] == ACCEPTED_EXECUTION_REVIEW_SHA,
        "Exact accepted execution source/runtime/review binding required",
    )
    manifest_pin = admission["execution_manifest"]
    require(manifest_pin["sha256"] == identity["execution_manifest_sha256"], "Execution manifest external pin mismatch")
    execution_manifest = json_pin(manifest_pin)
    for item in execution_manifest["files"]:
        path = Path(item["path"])
        path = path if path.is_absolute() else Path(manifest_pin["path"]).parent / path
        require(item["sha256"] != ACTUAL_OUTCOME_SOURCE_SHA, "Outcome bytes may not appear in prelabel source manifest")
        checked_bytes(dict(path=str(path), sha256=item["sha256"]))
    require(
        admission["runtime_receipt"]["sha256"] == identity["runtime_sha256"], "Runtime receipt external pin mismatch"
    )
    require(
        json_pin(admission["runtime_receipt"]) == runtime_identity(),
        "Exact prediction/scoring numerical runtime required",
    )
    review = json_pin(identity["independent_execution_review"])
    require(
        review["manifest_sha256"] == identity["execution_manifest_sha256"]
        and review["disposition"].startswith("accepted"),
        "Exact independent execution acceptance required",
    )
    masks = json_pin(admission["mask_receipt"])
    require(
        masks.get("schema") == "MomentumFrozenScoreMasksV1"
        and masks["orientation"] == "D/(D+R)"
        and masks["forecast_design_sha256"] == DESIGN_SHA,
        "Frozen event masks and exact D orientation required",
    )
    folds = admission["selected_folds"]
    cycles = [f["cycle"] for f in folds]
    require(
        len(set(cycles)) == len(cycles) and all(c in CYCLES for c in cycles),
        "Unique selected evaluation cycles required",
    )
    if not admission["synthetic"]:
        require(set(cycles) == set(CYCLES), "All selected A/B prediction receipts required before numeric labels")
    require(set(masks["folds"]) == set(map(str, cycles)), "Exact mask/evaluation cycle identity required")
    references = json_pin(admission["references_terminal"])
    require(
        references.get("schema") == "MomentumPrelabelReferencesTerminalV1"
        and references.get("status") == "complete_prelabel_references"
        and references.get("numeric_target_labels") is False
        and references.get("scores") is False,
        "Complete prelabel reference receipt required",
    )
    ref_execution = json_pin(references["execution_receipt"])
    require(
        ref_execution.get("schema") == "MomentumForecastRootExecutionReceiptV1",
        "Known reference Root execution receipt required",
    )
    identities(ref_execution, identity)
    require(ref_execution["action"] == "references", "Reference execution action required")
    ref_artifacts = {row["cycle"]: row for row in references["artifacts"]}
    require(set(ref_artifacts) == set(cycles), "Exact reference/evaluation cycles required")
    verified = {}
    for fold in folds:
        cycle = fold["cycle"]
        require(
            fold["attempt"] in (0, 1) and set(fold["jobs"]) == set(ARMS), "Matched full polling trio selection required"
        )
        mask = masks["folds"][str(cycle)]
        full = exact_ids(mask["full_bank_race_ids"], "full-bank races")
        common = exact_ids(mask["common_polled_race_ids"], "common races")
        require(set(common).issubset(full), "Common mask outside full bank support")
        for key in ["competitive_race_ids", "near_even_race_ids"]:
            require(set(exact_ids(mask[key], key)).issubset(common), "Subset outside fixed common mask")
        training = exact_ids(fold["training_race_ids"], "earlier admitted training events")
        require(
            all(int(rid.split(":", 1)[0]) < cycle for rid in training),
            "Target/future event in frequency-reference training scope",
        )
        jobs = {
            arm: _verify_job(
                fold["jobs"][arm], fold, arm, mask, identity, admission["synthetic"], admission["execution_manifest"]
            )
            for arm in ARMS
        }
        if fold["attempt"] == 1:
            lineages = [job["attempt0_terminal_pins"] for job in jobs.values()]
            require(
                all(lineage == lineages[0] for lineage in lineages),
                "Larger trio must share identical original terminal lineage",
            )
        trigger = any(j["summary"].get("any_permitted_retry_trigger") is True for j in jobs.values())
        if fold["attempt"] == 0 and trigger:
            require(
                fold.get("frozen_failure_disposition")
                == "explicit_failed_unresolved_experiment_no_retry_or_qualification",
                "Triggered initial trio cannot be selected as successful evidence",
            )
        # Source equality is a matched comparison condition, separate from computation.
        require(
            len({j["trace"]["metadata"]["source_identity"] for j in jobs.values()}) == 1,
            "Polling arms use different historical banks",
        )
        ref = ref_artifacts[cycle]
        f0summary = json_pin(ref["archived_F0_summary"])
        po5summary = json_pin(ref["PO5_summary"])
        f0 = npz_record(ref["archived_F0_arrays"])
        po5 = npz_record(ref["PO5_arrays"])
        require(
            tuple(f0["race_ids"].tolist()) == full == tuple(po5["race_ids"].tolist()),
            "Reference/full-bank event axis mismatch",
        )
        require(
            tuple(f0summary["common_polled_race_ids"]) == common == tuple(po5summary["common_polled_race_ids"]),
            "Reference/common event axis mismatch",
        )
        require(
            tuple(f0summary["competitive_race_ids"]) == tuple(mask["competitive_race_ids"])
            and tuple(f0summary["near_even_race_ids"]) == tuple(mask["near_even_race_ids"]),
            "Mask drift from immutable archived-F0 prelabel decision",
        )
        require(mask["archived_F0_summary"] == ref["archived_F0_summary"], "Frozen mask/source reference mismatch")
        require(
            f0summary["source_identity"]
            == po5summary["source_identity"]
            == jobs["FS5"]["trace"]["metadata"]["source_identity"],
            "Reference/bank source identity mismatch",
        )
        require("joint_empirical_shares" in f0, "Original unsorted F0 replicates required for row-clustered CRPS")
        require(
            f0["joint_empirical_shares"].ndim == 3
            and f0["joint_empirical_shares"].shape[-1] == len(full)
            and np.isfinite(f0["joint_empirical_shares"]).all()
            and ((f0["joint_empirical_shares"] >= 0) & (f0["joint_empirical_shares"] <= 1)).all(),
            "Archived full event axes/bounded shares required before labels",
        )
        require(
            po5["mean_logit"].shape == (len(full),)
            and po5["covariance"].shape == (len(full), len(full))
            and np.isfinite(po5["mean_logit"]).all()
            and np.isfinite(po5["covariance"]).all()
            and np.allclose(po5["covariance"], po5["covariance"].T, rtol=1e-12, atol=1e-12)
            and (np.diag(po5["covariance"]) > 0).all(),
            "Analytic PO5 full event axes/covariance required before labels",
        )
        require(
            f0["joint_empirical_shares"].shape[:2] == (4, 20000)
            if not admission["synthetic"]
            else f0["joint_empirical_shares"].shape[0] == 4,
            "Original F0 source-chain/replicate dimensions required",
        )
        verified[cycle] = dict(
            mask=mask,
            training_race_ids=training,
            jobs=jobs,
            f0=f0,
            f0summary=f0summary,
            po5=po5,
            po5summary=po5summary,
            frozen_failure_disposition=fold.get("frozen_failure_disposition"),
            selected_attempt=fold["attempt"],
        )
    audit.append("all_prediction_reference_mask_status_receipts_verified_before_numeric_label_read")
    return VerifiedAdmission(dict(admission=admission, folds=verified, audit=audit), _VERIFIED_TOKEN)


def read_labels_after_admission(bundle):
    """Only numeric target/earlier-outcome reader, after complete verified gateway."""
    require(
        type(bundle) is VerifiedAdmission and bundle._token is _VERIFIED_TOKEN,
        "Verified immutable admission capability required before labels",
    )
    require(
        bundle["audit"]
        and bundle["audit"][-1] == "all_prediction_reference_mask_status_receipts_verified_before_numeric_label_read",
        "Receipt-first gate not satisfied",
    )
    admission = bundle["admission"]
    if not admission["synthetic"]:
        require(
            admission["label_source"]["sha256"] == ACTUAL_OUTCOME_SOURCE_SHA,
            "Exact admitted F0 outcome-source bytes required",
        )
    content = checked_bytes(admission["label_source"])
    audit = list(bundle["audit"]) + ["numeric_outcome_source_read_after_all_receipts"]
    needed = set()
    for fold in bundle["folds"].values():
        needed.update(fold["mask"]["full_bank_race_ids"])
        needed.update(fold["training_race_ids"])
    labels = {}
    for raw in csv.DictReader(io.StringIO(content.decode())):
        rid = raw.get("race_id")
        cycle = int(raw["cycle"])
        if rid not in needed:
            continue  # Filter identity/cycle metadata before numeric fields.
        require(rid not in labels and int(rid.split(":", 1)[0]) == cycle, "Outcome event duplication/cycle mismatch")
        dem = int(raw["dem_votes"])
        rep = int(raw["rep_votes"])
        require(dem >= 0 and rep >= 0 and dem + rep > 0, "Finite nonnegative ordinary D/R counts required")
        share = dem / (dem + rep)
        reported = float(raw["actual_dem_share"])
        require(np.isfinite(reported) and abs(reported - share) <= 1e-14, "Outcome share/count orientation mismatch")
        if raw.get("two_party_votes"):
            require(int(raw["two_party_votes"]) == dem + rep, "Outcome denominator mismatch")
        labels[rid] = dict(cycle=cycle, actual_dem_share=share, tie=dem == rep, dem_win=dem > rep)
    require(set(labels) == needed, "Missing exact target/training outcome identities")
    frequencies = {}
    for cycle, fold in bundle["folds"].items():
        ids = fold["training_race_ids"]
        eligible = [labels[rid] for rid in ids if not labels[rid]["tie"]]
        require(all(row["cycle"] < cycle for row in eligible), "Target/future numeric outcome in frequency reference")
        frequencies[cycle] = dict(
            probability=(sum(row["dem_win"] for row in eligible) + 0.5) / (len(eligible) + 1),
            earlier_training_count=len(ids),
            binary_training_count=len(eligible),
            training_ties_excluded=len(ids) - len(eligible),
            arithmetic_phase="after_all_prediction_and_status_receipts",
        )
    return _observed_copy(dict(labels=labels, frequency_references=frequencies, audit=audit))
