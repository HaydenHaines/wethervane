# ruff: noqa: E402
# Imports intentionally follow binding to the private source snapshots.
"""Authorized synthetic collector parity plus deterministic diagnostic fixtures."""

import copy
import json
import sys
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "snapshots"))
sys.path.insert(0, str(HERE))
import numpy as np
import observed_sampler as observed
import pytest
import trace_diagnostics as td
from src.forecasting import polling_basis_v1 as original
from src.forecasting.polling_v1 import GaussianPollLikelihood


def fixture(module, full=False):
    n, q, k = 3, 6, 12
    prior = module.EmpiricalBasisFundamentals(
        np.array([-0.2, 0.1, 0.3]),
        np.cos(np.arange(n * q).reshape(n, q) / 5),
        0.1 * np.sin(np.arange(k * q).reshape(k, q) / 4),
        0.13 + 0.02 * (np.arange(k) % 3),
        0.3 + 0.03 * (np.arange(k) % 4),
        ("SYNTHETIC:A", "SYNTHETIC:B", "SYNTHETIC:C"),
        "synthetic-only",
        tuple(f"b{j}" for j in range(q)),
        posterior_chain=np.arange(k) // 3,
        posterior_draw=np.arange(k) % 3,
    )
    index = np.array([0, 1, 0, 2] if full else [0, 1, 0])
    m = len(index)
    polls = GaussianPollLikelihood(
        index,
        0.1 * np.sin(np.arange(m)),
        0.03 * np.eye(m) + 0.002 * np.ones((m, m)),
        tuple(f"SYNTHETIC:{j}" for j in range(m)),
    )
    return prior, polls


def collector(prior, draws=10, full=False):
    return td.RetainedTraceCollector(
        chains=4,
        draws=draws,
        race_ids=prior.race_ids,
        observed_race_indices=[0, 1, 2] if full else [0, 1],
        missing_race_indices=[] if full else [2],
        posterior_source_chain=prior.posterior_chain,
        posterior_source_draw=prior.posterior_draw,
        source_identity=prior.source_identity,
        source_pins={**td.REQUIRED_SOURCE_PINS, "synthetic_bank": "a" * 64},
        batch_length=4,
    )


def synthetic(full=False):
    prior, polls = fixture(observed, full)
    sink = collector(prior, full=full)
    records = []

    def observer(record):
        sink(record)
        records.append(record)

    result = observed.sample_poll_posterior(prior, polls, chains=4, warmup=3, draws=10, seed=811, observer=observer)
    return prior, polls, sink, records, result


def eq(a, b):
    if isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    elif isinstance(a, dict) or hasattr(a, "items"):
        assert set(a) == set(b)
        for key in a:
            eq(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            eq(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("full", [False, True])
def test_original_outputs_rng_and_collector_alignment(full):
    native = np.random.default_rng

    def run(module, prior, polls, observer=None):
        generators = []

        def factory(seed):
            rng = native(seed)
            generators.append(rng)
            return rng

        with patch.object(np.random, "default_rng", factory):
            kwargs = {} if observer is None else {"observer": observer}
            result = module.sample_poll_posterior(prior, polls, chains=4, warmup=3, draws=10, seed=811, **kwargs)
        return result, [copy.deepcopy(r.bit_generator.state) for r in generators]

    p0, polls = fixture(original, full)
    p1, _ = fixture(observed, full)
    sink = collector(p1, full=full)
    baseline, s0 = run(original, p0, polls)
    result, s1 = run(observed, p1, polls, sink)
    eq(baseline, result)
    eq(s0, s1)
    exported = sink.export()
    a = exported["arrays"]
    meta = exported["metadata"]
    eq(s0, meta["final_rng_states"])
    np.testing.assert_array_equal(a["observed_theta"], result["logits"][:, :, a["observed_race_indices"]])
    np.testing.assert_array_equal(a["sampled_row_index"], result["posterior_index"])
    np.testing.assert_array_equal(a["conditioning_row_index"][:, 1:], a["sampled_row_index"][:, :-1])
    np.testing.assert_array_equal(a["conditioning_lambda"][:, 1:], a["lambda_after_gamma"][:, :-1])
    assert a["bank_batch_weight_sums"].shape == (4, 3, 12)
    np.testing.assert_array_equal(a["bank_batch_counts"], np.tile([4, 4, 2], (4, 1)))
    assert not any(v.shape == (4, 10, 12) for v in a.values())
    for value in a.values():
        assert not value.flags.writeable
        with pytest.raises(ValueError):
            value.setflags(write=True)
    with pytest.raises(TypeError):
        meta["draws"] = 20
    with pytest.raises(ValueError, match="complete"):
        sink({"phase": "gaussian_conditional"})


def test_npz_json_roundtrip_and_tamper_rejection(tmp_path):
    *_, sink, records, result = synthetic()  # prior,polls,sink,records,result
    exported = sink.export()
    path = tmp_path / "exclusive"
    receipt = td.save_trace(exported, path)
    loaded = td.load_trace(
        path,
        expected_source_pins=exported["metadata"]["source_pins"],
        expected_receipt_sha256=receipt["receipt_sha256"],
    )
    eq(exported, loaded)
    with np.load(path / "trace.npz", allow_pickle=False) as archive:
        assert all(archive[k].dtype.kind in "iuf" and not archive[k].dtype.hasobject for k in archive.files)
    with pytest.raises(FileExistsError):
        td.save_trace(exported, path)
    with pytest.raises(ValueError, match="source pins"):
        td.load_trace(path, expected_source_pins={}, expected_receipt_sha256=receipt["receipt_sha256"])
    (path / "trace.npz").write_bytes((path / "trace.npz").read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="digest"):
        td.load_trace(
            path,
            expected_source_pins=exported["metadata"]["source_pins"],
            expected_receipt_sha256=receipt["receipt_sha256"],
        )


@pytest.mark.parametrize(
    "kind",
    [
        "skip",
        "duplicate",
        "wrong_chain",
        "wrong_draw",
        "bad_variance",
        "nonfinite_mean",
        "weight_lambda",
        "weight_sum",
        "post_lambda",
        "axes",
        "cycle",
        "next_row",
        "missing_location",
    ],
)
def test_malformed_observer_streams_rejected(kind):
    prior, _, _, records, _ = synthetic()
    sink = collector(prior)
    record = dict(records[0])
    prefix = []
    if kind == "skip":
        record = dict(records[1])
    if kind == "duplicate":
        prefix = [records[0]]
        record = dict(records[0])
    if kind == "wrong_chain":
        record["chain"] = 1
    if kind == "wrong_draw":
        record["retained_draw"] = 1
    if kind == "bad_variance":
        record["marginal_variances"] = np.zeros(3)
    if kind == "nonfinite_mean":
        record["mean_theta_cycle"] = np.full(3, np.nan)
    if kind in ["weight_lambda", "weight_sum"]:
        prefix = [records[0]]
        record = dict(records[1])
        if kind == "weight_lambda":
            record["conditioning_lambda"] = record["conditioning_lambda"] + 1
        else:
            record["normalized_row_weights"] = np.ones(12)
    if kind in ["post_lambda", "axes", "cycle", "missing_location"]:
        prefix = records[:2]
        record = dict(records[2])
        if kind == "post_lambda":
            record["lambda_after_gamma"] = np.zeros(2)
        if kind == "axes":
            record["observed_race_indices"] = np.array([1, 0])
        if kind == "cycle":
            record["cycle"] += 1
        if kind == "missing_location":
            record["missing_location"] = np.array([np.inf])
    if kind == "next_row":
        prefix = records[:3]
        record = dict(records[3])
        record["conditioning_row_index"] = (records[2]["sampled_row_index"] + 1) % 12
    for item in prefix:
        sink(item)
    with pytest.raises((ValueError, TypeError)):
        sink(record)
    with pytest.raises(ValueError, match="Incomplete"):
        sink.export()


def test_constructor_and_serialized_schema_rejections(tmp_path):
    prior, _ = fixture(observed)
    sink = collector(prior)
    with pytest.raises(ValueError, match="Incomplete"):
        sink.export()
    with pytest.raises(ValueError, match="at least one observed"):
        td.RetainedTraceCollector(
            chains=4,
            draws=10,
            race_ids=prior.race_ids,
            observed_race_indices=[],
            missing_race_indices=[0, 1, 2],
            posterior_source_chain=prior.posterior_chain,
            posterior_source_draw=prior.posterior_draw,
            source_identity="synthetic",
            source_pins=td.REQUIRED_SOURCE_PINS,
        )
    _, _, sink, _, _ = synthetic()
    ex = sink.export()
    arrays = {k: v.copy() for k, v in ex["arrays"].items()}
    meta = td.plain(ex["metadata"])
    arrays["conditioning_row_index"][0, 1] = (arrays["sampled_row_index"][0, 0] + 1) % 12
    with pytest.raises(ValueError, match="phase alignment"):
        td.save_trace(dict(arrays=arrays, metadata=meta), tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
    arrays = {k: v.copy() for k, v in ex["arrays"].items()}
    arrays["cycle"] = np.array([[object()]], dtype=object)
    with pytest.raises(ValueError, match="non-pickle"):
        td.save_trace(dict(arrays=arrays, metadata=meta), tmp_path / "object")


def supplied():
    offsets = np.array([-0.03, -0.01, 0.01, 0.03])[:, None]
    return offsets + 0.1 * np.sin(2 * np.pi * np.arange(1600)[None, :] / 40)


def test_mcse_rhat_passes_and_independent_batch_variance_oracle():
    x = supplied()
    values = np.stack([x, np.zeros_like(x)], axis=2)
    result = td.function_diagnostics(values, ["signal", "constant"])
    independent = [np.var(x.mean(axis=1), ddof=1) / 4]
    for length in [50, 100, 200]:
        batch = x.reshape(4, 1600 // length, length).mean(axis=2).ravel()
        independent.append(np.var(batch, ddof=1) / len(batch))
    row = result["coordinates"][0]
    assert abs(row["mcse"] - np.sqrt(max(independent))) < 1e-15
    assert abs(row["mean_ess_estimate"] - np.var(x, ddof=1) / max(independent)) < 1e-10
    assert row["gates"]["rhat"] is False
    assert result["mean_ess_is_not_bulk_or_tail_ess"]
    constant = result["coordinates"][1]
    assert constant["status"] == "unresolved_constant" and constant["mcse"] is None
    declared = td.function_diagnostics(values, ["signal", "constant"], structural_constants=["constant"])[
        "coordinates"
    ][1]
    assert declared["status"] == "structural" and declared["mcse"] == 0
    with pytest.raises(ValueError):
        td.function_diagnostics(values, ["signal", "constant"], structural_constants=["signal"])
    with pytest.raises(ValueError):
        td.function_diagnostics(values[:, :100], ["signal", "constant"])


def test_tail_rescaling_absolute_relative_separation_and_raw_constants():
    base = np.log(np.array([0.1, 0.4, 0.7, 1.0])[:, None] * (1 + 0.01 * np.sin(np.arange(1600))[None, :]))
    # Keep ordinary probabilities <=1 while retaining deliberately different chain means.
    base -= 1
    rare = td.log_tail_diagnostics((base - 1000)[..., None], ["lose"])["coordinates"][0]
    ordinary = td.log_tail_diagnostics((base - 10)[..., None], ["lose"])["coordinates"][0]
    assert abs(rare["relative_mcse"] - ordinary["relative_mcse"]) < 1e-13
    assert abs(rare["log_probability"] - ordinary["log_probability"] + 990) < 2e-13
    assert rare["gates"]["absolute_mcse"] is True and rare["gates"]["relative_mcse"] is False
    assert rare["gates"]["chain_log_difference"] is False
    assert rare["absolute_mcse_underflows"] and rare["absolute_mcse"] is None
    raw = td.raw_indicator_diagnostics(np.ones((4, 1600, 1)), ["win"])
    assert raw["coordinates"][0]["mcse"] is None and raw["coordinates"][0]["status"] == "unresolved_constant"
    empty = td.log_tail_diagnostics(np.full((4, 1600, 1), -np.inf), ["lose"])["coordinates"][0]
    assert empty["status"] == "unresolved_all_negative_infinity" and empty["relative_mcse"] is None
    zeros = np.array(base - 10)
    zeros[0] = -np.inf
    result = td.log_tail_diagnostics(zeros[..., None], ["lose"])["coordinates"][0]
    assert result["chains_without_tail_support"] == [0] and result["gates"]["chain_log_difference"] is False
    with pytest.raises(ValueError):
        td.log_tail_diagnostics(np.ones((4, 1600, 1)), ["bad"])
    json.dumps(rare, allow_nan=False)
    json.dumps(result, allow_nan=False)


def test_bank_metrics_against_exact_weights_and_relabel_invariance():
    weights = np.array([0.5, 0.25, 0.125, 0.125])
    sums = np.tile(weights, (4, 2, 1)) * 100
    counts = np.full((4, 2), 100)
    source = np.array([0, 0, 1, 1])
    result = td.bank_diagnostics(sums, counts, source)
    np.testing.assert_array_equal(result["row_mass"], weights)
    assert result["max_row_mass"] == 0.5 and result["top10_mass"] == 1 and result["top100_mass"] == 1
    assert result["effective_bank_rows"] == 1 / np.sum(weights**2)
    assert result["rows_carrying_99_mass"] == 4
    np.testing.assert_array_equal(result["original_source_chain_mass"], [0.75, 0.25])
    assert result["bank_approximation_sensitive"] and all(result["gates"].values())
    permutation = [2, 0, 3, 1]
    renamed = td.bank_diagnostics(sums[:, :, permutation], counts, source[permutation])
    for key in [
        "max_row_mass",
        "effective_bank_rows",
        "entropy",
        "rows_carrying_99_mass",
        "original_source_chain_mass",
        "forecast_chain_tv",
    ]:
        assert result[key] == renamed[key]
    divergent = sums.copy()
    divergent[0] = np.tile([100, 0, 0, 0], (2, 1))
    failed = td.bank_diagnostics(divergent, counts, source)
    assert not failed["gates"]["forecast_chain_tv"] and failed["bank_approximation_sensitive"]
    with pytest.raises(ValueError):
        td.bank_diagnostics(sums * 2, counts, source)


def test_display_and_mean_gates_are_separate_and_selected_config_bound():
    values = (0.5 + supplied())[..., None]
    mean = td.mean_share_diagnostics(values, ["share"])
    display = td.interval_display_diagnostics(values, ["q50"], [0.006])
    assert mean["coordinates"][0]["gates"]["mean_mcse"] is False
    assert display["coordinates"][0]["gates"]["share_bracket_width"] is False
    assert display["scope"] == "interval_display_only_not_score_precision_or_sampling_retry"
    assert td.DiagnosticConfig().design_sha256 == td.ROOT_DESIGN_SHA256
    with pytest.raises(ValueError):
        td.DiagnosticConfig(design_sha256="a" * 64)
    with pytest.raises(ValueError):
        td.DiagnosticConfig(batch_lengths=(50,))


def test_selected_core_coordinate_assembly_on_supplied_arrays():
    from scipy.special import expit

    x = supplied()
    c, d = x.shape
    logits = np.stack([x, x + 0.1, x - 0.1], axis=2)
    cycle = 0.03 + x / 10
    row = np.tile(np.arange(d) % 12, (c, 1))
    coefficient_names = tuple(f"b{j}" for j in range(6))
    result = dict(
        race_ids=("A", "B", "C"),
        posterior_index=row,
        shared_cycle=cycle,
        logits=logits,
        shares=expit(logits),
        coefficients=np.stack([x + j * 0.1 for j in range(6)], axis=2),
        coefficient_names=coefficient_names,
        tau_cycle=0.2 + x / 100,
        tau_race=0.4 + x / 100,
        shared_cycle_z=cycle / (0.2 + x / 100),
    )
    lam = np.exp(np.stack([x, x + 0.2], axis=2))
    trace = dict(
        arrays=dict(
            observed_race_indices=np.array([0, 1]),
            sampled_row_index=row,
            cycle=cycle,
            observed_theta=logits[:, :, :2],
            lambda_after_gamma=lam,
        ),
        metadata=dict(race_ids=("A", "B", "C")),
    )
    report = td.core_state_diagnostics(result, trace)
    names = [r["coordinate"] for r in report["coordinates"]]
    assert len(names) == 16
    assert names[:4] == ["logit:A", "logit:B", "share:A", "share:B"]
    assert names[-2:] == ["log_post_gamma_lambda:A", "log_post_gamma_lambda:B"]
    assert abs(report["coordinates"][-1]["mean"] - np.log(lam[:, :, 1]).mean()) < 1e-15
    trace["arrays"]["cycle"] = cycle + 1
    with pytest.raises(ValueError, match="row-cycle"):
        td.core_state_diagnostics(result, trace)


def test_strict_source_binding_and_log_side_independence():
    prior, _ = fixture(observed)
    bad = dict(td.REQUIRED_SOURCE_PINS)
    bad["forecast_design"] = "0" * 64
    with pytest.raises(ValueError, match="pins mismatch"):
        td.RetainedTraceCollector(
            chains=4,
            draws=10,
            race_ids=prior.race_ids,
            observed_race_indices=[0, 1],
            missing_race_indices=[2],
            posterior_source_chain=prior.posterior_chain,
            posterior_source_draw=prior.posterior_draw,
            source_identity="synthetic",
            source_pins=bad,
        )
    with pytest.raises(TypeError):
        td.REQUIRED_SOURCE_PINS["forecast_design"] = "0" * 64
    for invalid in [np.nan, np.inf, 0.01]:
        with pytest.raises(ValueError):
            td.log_tail_diagnostics(np.full((4, 1600, 1), invalid), ["lose"])
    with pytest.raises(ValueError):
        td.DiagnosticConfig(batch_lengths=[50, 100])


def test_external_receipt_rejects_coherent_valid_numeric_rewrite(tmp_path):
    _, _, sink, _, _ = synthetic()
    exported = sink.export()
    path = tmp_path / "rewrite"
    receipt = td.save_trace(exported, path)
    arrays = {k: v.copy() for k, v in exported["arrays"].items()}
    # Another finite, positive and otherwise valid trace, not merely broken bytes.
    arrays["gaussian_mean_theta_cycle"] += 0.1
    np.savez_compressed(path / "trace.npz", **arrays)
    rewritten = json.loads((path / "receipt.json").read_text())
    rewritten["files"]["trace.npz"] = td.sha(path / "trace.npz")
    (path / "receipt.json").write_text(json.dumps(rewritten, sort_keys=True, indent=2) + "\n")
    assert td.sha(path / "receipt.json") != receipt["receipt_sha256"]
    with pytest.raises(ValueError, match="Externally bound receipt"):
        td.load_trace(
            path,
            expected_source_pins=exported["metadata"]["source_pins"],
            expected_receipt_sha256=receipt["receipt_sha256"],
        )
    # Without the independent pin, shape-valid alternate data would pass.
    altered = td.load_trace(
        path,
        expected_source_pins=exported["metadata"]["source_pins"],
        expected_receipt_sha256=td.sha(path / "receipt.json"),
    )
    assert not np.array_equal(
        altered["arrays"]["gaussian_mean_theta_cycle"], exported["arrays"]["gaussian_mean_theta_cycle"]
    )


def test_selected_threshold_overrides_cannot_claim_frozen_design():
    for kwargs in [
        dict(max_rhat=1.05),
        dict(min_mean_ess=1),
        dict(max_win_absolute_mcse=0.5),
        dict(bank_warning_min_effective_rows=1),
    ]:
        with pytest.raises(ValueError, match="threshold override"):
            td.DiagnosticConfig(**kwargs)


def test_unrepresentable_mean_ess_is_unresolved_not_infinite_certificate():
    values = supplied()[..., None]
    with patch.object(td, "correlated_mcse", return_value={"mcse": np.array([1e-200]), "rank_rhat": np.array([1.0])}):
        result = td.function_diagnostics(values, ["signal"])
    row = result["coordinates"][0]
    assert row["status"] == "unresolved_numeric_mean_ess"
    assert row["mean_ess_estimate"] is None and row["mcse"] is None
    assert row["gates"]["mean_ess"] is None
    json.dumps(result, allow_nan=False)


def test_partial_prefixes_every_phase_exclude_pending_bank_mass_and_uninitialized_arrays(tmp_path):
    prior, _, _, records, _ = synthetic()
    sink = collector(prior)
    completed = [0] * 4
    expected_weights = [[] for _ in range(4)]
    pending = {}
    final_states = 0
    for index in range(len(records) + 1):
        partial = sink.export_partial()
        a = partial["arrays"]
        meta = partial["metadata"]
        assert list(meta["completed_draws_per_chain"]) == completed
        assert meta["restartable"] is False
        assert len(meta["final_rng_states_emitted"]) == final_states
        for c, count in enumerate(completed):
            prefix = f"chain{c}__"
            assert a[prefix + "conditioning_lambda"].shape == (1, count, 2)
            assert a[prefix + "gaussian_mean_theta_cycle"].shape == (1, count, 3)
            np.testing.assert_array_equal(a[prefix + "retained_draw_index"], np.arange(count)[None, :])
            np.testing.assert_array_equal(a[prefix + "forecast_chain_index"], [c])
            assert a[prefix + "bank_sampled_occupancy"].sum() == count
            sums = a[prefix + "bank_batch_weight_sums"]
            expected = np.sum(expected_weights[c], axis=0) if count else np.zeros(12)
            np.testing.assert_allclose(sums.sum(axis=(0, 1)), expected, atol=3e-15, rtol=0)
            assert abs(sums.sum() - count) < 2e-14
            for name, value in a.items():
                assert np.isfinite(value).all() and not value.flags.writeable
        if index == 1:
            # Neither the strict full exporter nor full serializer accepts a prefix.
            with pytest.raises(ValueError):
                sink.export()
            with pytest.raises(ValueError, match="schema"):
                td.save_trace(partial, tmp_path / "not-full")
        if index == len(records):
            break
        record = records[index]
        c = record["chain"]
        if record["phase"] == "row_weights":
            pending[c] = record["normalized_row_weights"]
        if record["phase"] == "completed_sweep":
            completed[c] += 1
            expected_weights[c].append(pending.pop(c))
        if record["phase"] == "chain_final_rng":
            final_states += 1
        sink(record)
    full = sink.export()
    for c in range(4):
        np.testing.assert_array_equal(
            partial["arrays"][f"chain{c}__lambda_after_gamma"][0], full["arrays"]["lambda_after_gamma"][c]
        )
    assert partial["metadata"]["complete"] is True
    assert list(partial["metadata"]["final_rng_state_chain_indices"]) == [0, 1, 2, 3]


def test_completed_observer_prefix_is_not_completed_raw_prediction_or_rng_state():
    prior, polls = fixture(observed)
    sink = collector(prior)
    native = np.random.default_rng
    missing_rng_calls = []

    class Wrapper:
        def __init__(self, rng):
            self.rng = rng

        @property
        def bit_generator(self):
            return self.rng.bit_generator

        def __getattr__(self, name):
            return getattr(self.rng, name)

        def standard_t(self, *args, **kwargs):
            missing_rng_calls.append((args, kwargs))
            return self.rng.standard_t(*args, **kwargs)

    def callback(record):
        sink(record)
        if record["phase"] == "completed_sweep":
            raise RuntimeError("synthetic failure before missing-race RNG")

    with patch.object(np.random, "default_rng", lambda seed: Wrapper(native(seed))):
        with pytest.raises(RuntimeError, match="before missing-race"):
            observed.sample_poll_posterior(prior, polls, chains=4, warmup=3, draws=10, seed=811, observer=callback)
    assert missing_rng_calls == []
    partial = sink.export_partial()
    meta = partial["metadata"]
    assert list(meta["completed_draws_per_chain"]) == [1, 0, 0, 0]
    assert meta["observer_phase_prefix_only"] is True
    assert meta["raw_predictive_draw_completeness_claimed"] is False
    assert "may not have occurred" in meta["completed_draw_count_semantics"]
    assert not meta["final_rng_states_emitted"] and meta["restartable"] is False
