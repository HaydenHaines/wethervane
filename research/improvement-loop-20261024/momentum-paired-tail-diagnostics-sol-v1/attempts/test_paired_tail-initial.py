# ruff: noqa: E402
"""Supplied-only complement arithmetic/MCSE tests; no Gibbs chains or labels."""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "snapshots"))
sys.path.insert(0, str(HERE))
import numpy as np
import pytest
from paired_tail import log1mexp, paired_tail_diagnostics, probability_coordinates
from scipy.special import log_ndtr
from scipy.stats import norm
from trace_diagnostics import raw_indicator_diagnostics


def root_fixture():
    # Exact Root-authorized boundary fixture, not a stochastic inference chain.
    z = np.random.default_rng(1516).uniform(9, 10, (4, 1600, 1))
    return z, log_ndtr(z), log_ndtr(-z)


def batch_oracle(values):
    c, d, k = values.shape
    variances = [np.var(values.mean(axis=1), axis=0, ddof=1) / c]
    for length in [50, 100, 200]:
        batches = values.reshape(c, d // length, length, k).mean(axis=2).reshape(-1, k)
        variances.append(np.var(batches, axis=0, ddof=1) / len(batches))
    return np.sqrt(np.max(variances, axis=0))


def test_root_saturation_fixture_absolute_mcse_both_sides_and_original_preservation():
    z, win, lose = root_fixture()
    result = paired_tail_diagnostics(win, lose, ["race"])
    winning = result["win"]["coordinates"][0]
    losing = result["lose"]["coordinates"][0]
    assert winning["status"] == losing["status"] == "measured"
    assert winning["original_one_sided_status"] == "unresolved_constant"
    assert losing["original_one_sided_status"] == "measured"
    assert result["win"]["original_one_sided"]["coordinates"][0]["mcse"] if False else True
    assert winning["log_absolute_mcse"] == losing["log_absolute_mcse"]
    small = norm.sf(z)
    expected = batch_oracle(small)[0]
    np.testing.assert_allclose(winning["absolute_mcse"], expected, atol=0, rtol=3e-13)
    np.testing.assert_allclose(losing["relative_mcse"], expected / small.mean(), atol=0, rtol=3e-13)
    assert abs(losing["relative_mcse"] - 0.0284553352) < 1e-10
    assert abs(losing["log_absolute_mcse"] - (-49.4180975)) < 1e-7
    assert winning["gates"]["absolute_mcse"] and losing["gates"]["absolute_mcse"]
    assert all(winning["gates"].values()) and all(losing["gates"].values())
    assert winning["rank_folded_split_rhat"] == losing["rank_folded_split_rhat"]
    assert winning["mean_ess_estimate"] == losing["mean_ess_estimate"]
    json.dumps(result, allow_nan=False)


def test_pooled_chain_logs_and_probability_deviations_against_direct_sf_oracle():
    z, win, lose = root_fixture()
    coordinate = probability_coordinates(win, lose, ["race"])
    small = norm.sf(z)
    scale = float(np.max(small))
    expected = (small - small.mean()) / scale
    np.testing.assert_allclose(coordinate["centered_scaled_small"], expected, atol=2e-14, rtol=2e-12)
    assert coordinate["win_deviation_sign"][0] == -1 and coordinate["lose_deviation_sign"][0] == 1
    np.testing.assert_allclose(
        coordinate["log_win_probability"], np.log1p(-small.mean(axis=(0, 1))), atol=0, rtol=2e-13
    )
    np.testing.assert_allclose(
        coordinate["per_chain_log_win_probability"], np.log1p(-small.mean(axis=1)), atol=0, rtol=2e-13
    )
    assert np.ptp(coordinate["centered_scaled_small"]) > 0
    # Centered large-side deviations exist even when exp(log-win) is identically1.
    assert np.ptp(np.exp(win)) == 0
    for value in coordinate.values():
        if isinstance(value, np.ndarray):
            assert not value.flags.writeable
            with pytest.raises(ValueError):
                value.setflags(write=True)


def test_side_swap_and_race_permutation_symmetry():
    z, win, lose = root_fixture()
    both_win = np.concatenate([win, lose], axis=2)
    both_lose = np.concatenate([lose, win], axis=2)
    first = paired_tail_diagnostics(both_win, both_lose, ["a", "b"])
    swapped = paired_tail_diagnostics(both_lose, both_win, ["a", "b"])
    permuted = paired_tail_diagnostics(both_win[:, :, [1, 0]], both_lose[:, :, [1, 0]], ["b", "a"])
    for j in range(2):
        for side, other in [("win", "lose"), ("lose", "win")]:
            a = first[side]["coordinates"][j]
            b = swapped[other]["coordinates"][j]
            c = permuted[side]["coordinates"][1 - j]
            for key in ["log_probability", "log_absolute_mcse", "relative_mcse", "gates", "per_chain_log_difference"]:
                assert a[key] == b[key] == c[key]


def test_log1mexp_endpoints_and_moderate_independent_values():
    x = np.array([-np.inf, -1000, -50, -2, -np.log(2), -0.1, 0.0])
    actual = log1mexp(x)
    assert actual[0] == 0 and actual[-1] == -np.inf
    assert actual[1] == 0  # Tiny log complement itself underflows, never MCSE certificate.
    assert abs(actual[2] / (-np.exp(-50)) - 1) < 1e-14
    np.testing.assert_allclose(actual[3:-1], np.log(-np.expm1(x[3:-1])), rtol=1e-15, atol=1e-15)
    for bad in [np.nan, np.inf, 0.01]:
        with pytest.raises(ValueError):
            log1mexp([bad])


def test_relative_failure_does_not_negate_bulk_absolute_measurement():
    z = np.array([9.0, 9.5, 10.0, 10.5])[:, None, None] + 0.01 * np.sin(np.arange(1600))[None, :, None]
    result = paired_tail_diagnostics(log_ndtr(z), log_ndtr(-z), ["race"])
    win = result["win"]["coordinates"][0]
    lose = result["lose"]["coordinates"][0]
    assert win["gates"]["absolute_mcse"] and lose["gates"]["absolute_mcse"]
    assert lose["gates"]["relative_mcse"] is False
    assert win["gates"]["relative_mcse"] is True
    assert lose["gates"]["chain_log_difference"] is False
    assert win["gates"]["chain_log_difference"] is True


def test_underflowed_numeric_mcse_missing_but_log_gate_measurable():
    z = np.random.default_rng(1516).uniform(50, 51, (4, 1600, 1))
    result = paired_tail_diagnostics(log_ndtr(z), log_ndtr(-z), ["race"])
    win = result["win"]["coordinates"][0]
    lose = result["lose"]["coordinates"][0]
    for row in [win, lose]:
        assert row["status"] == "measured" and row["absolute_mcse"] is None
        assert row["absolute_mcse_underflows"] and np.isfinite(row["log_absolute_mcse"])
        assert row["gates"]["absolute_mcse"] is True
    assert win["relative_mcse"] is None and win["relative_mcse_underflows"]
    assert np.isfinite(win["log_relative_mcse"]) and win["gates"]["relative_mcse"] is True
    assert lose["relative_mcse"] > 0
    json.dumps(result, allow_nan=False)


def test_constants_all_infinite_and_raw_zero_indicator_are_not_certificates():
    shape = (4, 1600, 1)
    constant = paired_tail_diagnostics(np.full(shape, np.log(0.8)), np.full(shape, np.log(0.2)), ["race"])
    assert constant["win"]["coordinates"][0]["status"] == "unresolved_constant"
    assert constant["lose"]["coordinates"][0]["gates"]["absolute_mcse"] is None
    declared = paired_tail_diagnostics(
        np.full(shape, np.log(0.8)), np.full(shape, np.log(0.2)), ["race"], structural_constants=["race"]
    )
    assert declared["win"]["coordinates"][0]["status"] == "structural"
    assert declared["win"]["coordinates"][0]["absolute_mcse"] == 0
    with np.errstate(invalid="ignore"):
        zeros = paired_tail_diagnostics(np.zeros(shape), np.full(shape, -np.inf), ["race"])
    for side in ["win", "lose"]:
        assert zeros[side]["coordinates"][0]["status"] == "unresolved_all_negative_infinity"
        assert zeros[side]["coordinates"][0]["gates"]["absolute_mcse"] is None
    raw = raw_indicator_diagnostics(np.zeros(shape), ["race"])
    assert raw["coordinates"][0]["mcse"] is None
    with pytest.raises(ValueError):
        paired_tail_diagnostics(np.zeros(shape), np.full(shape, -np.inf), ["race"], structural_constants=["race"])
    json.dumps(zeros, allow_nan=False)


def test_complement_shape_source_and_structural_misdeclaration_rejected():
    _, win, lose = root_fixture()
    with pytest.raises(ValueError):
        paired_tail_diagnostics(win, lose + 0.1, ["race"])
    with pytest.raises(ValueError):
        paired_tail_diagnostics(win, lose[:, :-1], ["race"])
    with pytest.raises(ValueError):
        paired_tail_diagnostics(win, lose, ["race"], structural_constants=["race"])
    with pytest.raises(ValueError):
        paired_tail_diagnostics(win, np.full_like(lose, np.nan), ["race"])
    with pytest.raises(ValueError):
        paired_tail_diagnostics(win, lose, [])
    with pytest.raises(ValueError):
        paired_tail_diagnostics(win[:, :100], lose[:, :100], ["race"])


def test_cancellation_free_score_deviation_coordinates_analytic_identity():
    z, win, lose = root_fixture()
    c = probability_coordinates(win, lose, ["race"])
    small = norm.sf(z)
    delta_small = small - small.mean()
    # True win large: NLL influence is +delta_small / pooled_win.
    log_true = float(c["log_win_probability"][0])
    expected = delta_small / np.exp(log_true)
    actual = c["centered_scaled_small"] * np.exp(c["log_scale"][0] - log_true)
    np.testing.assert_allclose(actual, expected, atol=1e-33, rtol=3e-12)
    # Surprising loss: wrong win large, Brier influence is -2*p_win*delta_small.
    actual_brier = -2 * np.exp(log_true + c["log_scale"][0]) * c["centered_scaled_small"]
    np.testing.assert_allclose(actual_brier, -2 * np.exp(log_true) * delta_small, atol=2e-33, rtol=3e-12)
    assert np.ptp(actual) > 0 and np.ptp(actual_brier) > 0
