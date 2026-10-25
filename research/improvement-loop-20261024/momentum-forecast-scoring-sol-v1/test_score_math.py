# ruff: noqa: E402
"""Invented supplied values/labels only; no actual forecast or outcome I/O."""

import sys
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "snapshots"))
sys.path.insert(0, str(HERE))
import numpy as np
import pytest
import score_math as sm
from marginal import GaussianLogitMixture
from scipy.special import log_ndtr


def test_empirical_crps_and_centered_influence_against_brute_force_pairs():
    values = np.array([0.1, 0.2, 0.4, 0.8, 0.9, 0.9])
    for truth in [0, 0.37, 0.5, 1]:
        score, influence = sm.empirical_crps_and_influence(values, truth)
        pair = abs(values[:, None] - values[None, :])
        oracle = np.mean(abs(values - truth)) - 0.5 * pair.mean()
        expected = abs(values - truth) - pair.mean(axis=1)
        expected -= expected.mean()
        assert abs(score - oracle) < 2e-16
        np.testing.assert_allclose(influence, expected, atol=2e-16, rtol=0)


def test_gaussian_mixture_weight_derivatives_all_metrics():
    means = np.array([[-1.2, 0.3], [0.8, 1.4]])
    sd = np.array([[0.5, 0.6], [0.7, 0.8]])
    truth = 0.38
    row = sm.gaussian_race(means, sd, truth, win_tail_qualified=True, lose_tail_qualified=True)
    direction = np.array([0.1, -0.2, 0.05, 0.05])
    epsilon = 1e-4
    weights = np.full(4, 0.25)

    def points(weight):
        model = GaussianLogitMixture(means.ravel(), sd.ravel(), weight)
        logwin, loglose = model.log_win_lose()
        q = {q: model.quantile(q)[0] for q in sm.LEVELS}
        return dict(
            crps=model.crps(truth)[0],
            wis=sm.weighted_interval_score(truth, q),
            brier=np.exp(2 * logwin),
            negative_log=-loglose,
        )

    up, down = points(weights + epsilon * direction), points(weights - epsilon * direction)
    for metric in sm.METRICS:
        influence = row["influences"][metric]
        expected = direction @ influence.values.ravel() * np.exp(influence.log_scale)
        actual = (up[metric] - down[metric]) / (2 * epsilon)
        assert abs(actual - expected) < 2e-6


def test_near_one_binary_influences_do_not_round_to_constant():
    z = np.random.default_rng(1516).uniform(9, 10, (4, 1600))
    loser = np.exp(log_ndtr(-z))
    delta = loser - loser.mean()
    winning = sm.gaussian_race(z, np.ones_like(z), 0.7, win_tail_qualified=True, lose_tail_qualified=True)
    surprising = sm.gaussian_race(z, np.ones_like(z), 0.3, win_tail_qualified=True, lose_tail_qualified=True)
    for row, metric, expected in [
        (winning, "negative_log", delta / (1 - loser.mean())),
        (surprising, "brier", -2 * (1 - loser.mean()) * delta),
    ]:
        influence = row["influences"][metric]
        actual = influence.values * np.exp(influence.log_scale)
        np.testing.assert_allclose(actual, expected, atol=2e-33, rtol=3e-12)
        assert np.ptp(actual) > 0


def test_scaled_race_average_retains_covariance_and_log_scale():
    x = np.sin(np.arange(6400).reshape(4, 1600) / 7)
    a = sm.ScaledInfluence(x, -1000)
    b = sm.ScaledInfluence(-0.25 * x, -999)
    result = sm.average_influences([a, b])
    expected = (x - 0.25 * np.e * x) / 2
    expected -= expected.mean()
    np.testing.assert_allclose(result.values * np.exp(result.log_scale + 1000), expected, atol=5e-14, rtol=1e-13)
    opposite = sm.average_influences([sm.ScaledInfluence(x), sm.ScaledInfluence(-x)])
    assert np.ptp(opposite.values) == 0  # Not independent per-race error addition.


def test_po5_deterministic_wis_kink_requires_no_density_or_influence():
    with patch.object(
        GaussianLogitMixture, "density", side_effect=AssertionError("density is unnecessary for deterministic PO5")
    ):
        result = sm.score_gaussian_forecast(
            np.zeros((1, 1, 1)), np.ones((1, 1, 1)), [0.5], ["invented"], [True], [True], deterministic=True
        )
    assert result["aggregates"]["wis"]["qualified"] is True
    assert result["aggregates"]["wis"]["mcse"] == 0
    assert result["aggregates"]["wis"]["uncertainty_scope"] == "deterministic_gaussian_analytic"
    assert result["aggregates"]["brier"]["count"] == 0 and result["binary_tie_count"] == 1
    assert result["aggregates"]["crps"]["count"] == 1


def test_stochastic_wis_truth_inside_bisection_bracket_is_one_sided():
    row = sm.gaussian_race(
        np.zeros((4, 1600)), np.ones((4, 1600)), 0.5, win_tail_qualified=True, lose_tail_qualified=True
    )
    assert row["quantiles"][0.5] != 0.5  # Midpoint representation, not an exact equality check.
    assert row["quantile_brackets"][0.5]["lower"] <= 0.5 <= row["quantile_brackets"][0.5]["upper"]
    assert row["metrics"]["wis"]["status"] == "uncertainty_unresolved"
    assert 0.5 in row["wis_one_sided_quantile_levels"]
    assert row["metrics"]["brier"]["status"] == "not_applicable_tie"


def test_empty_mask_pair_and_equal_cycle_are_unavailable_not_zero_or_dropped():
    empty = sm.score_gaussian_forecast(np.empty((4, 1600, 0)), np.empty((4, 1600, 0)), [], [], [], [])
    assert empty["aggregates"]["crps"]["count"] == 0 and empty["aggregates"]["crps"]["value"] is None
    difference = sm.paired_difference(empty, empty, "crps", independent_streams=True)
    assert difference["value"] is None and difference["mcse"] is None
    nonempty = dict(aggregates=dict(crps=dict(value=0.2, qualified=True, mcse=0.001, log_mcse=np.log(0.001))))
    tier = sm.equal_cycle_summary({2018: nonempty, 2022: empty}, "crps", tier="A")
    assert tier["value"] is None and not tier["qualified"] and tier["cycles"] == [2018, 2022]


def test_log_score_never_uses_only_successful_races():
    base = 0.05 * np.sin(np.arange(6400).reshape(4, 1600) / 11)
    means = np.stack([base, base + 0.2], axis=2)
    sd = np.ones_like(means) * 0.6
    result = sm.score_gaussian_forecast(means, sd, [0.6, 0.65], ["a", "b"], [False, True], [True, True])
    log = result["aggregates"]["negative_log"]
    assert log["count"] == 2 and log["value"] is not None and not log["qualified"]
    assert log["unqualified_race_indices"] == [0]
    assert len(result["rows"]) == 2


def test_one_share_quadrature_failure_does_not_remove_binary_points():
    with patch.object(GaussianLogitMixture, "crps", side_effect=ArithmeticError("invented bounded failure")):
        row = sm.gaussian_race(
            np.array([[0.2, 0.3], [0.1, 0.4]]), np.ones((2, 2)), 0.7, win_tail_qualified=True, lose_tail_qualified=True
        )
    assert row["metrics"]["crps"]["status"] == "deterministic_integration_unresolved"
    assert row["metrics"]["crps"]["value"] is None
    assert row["metrics"]["brier"]["value"] is not None
    assert row["metrics"]["negative_log"]["status"] == "computed"
    assert row["metrics"]["wis"]["value"] is not None


def test_archived_empirical_row_cluster_and_wis_uncertainty_scope():
    values = 0.5 + 0.25 * np.sin(np.arange(4 * 1600 * 5).reshape(4, 1600, 5) / 19)
    result = sm.score_empirical_forecast(values[..., None], [0.61], ["invented"], [True], [True])
    point, influence = sm.empirical_crps_and_influence(values, 0.61)
    assert abs(result["rows"][0]["metrics"]["crps"]["value"] - point) < 1e-14
    row = sm.empirical_race(values, 0.61, win_tail_qualified=True, lose_tail_qualified=True)
    scaled = row["influences"]["crps"]
    np.testing.assert_allclose(
        scaled.values * np.exp(scaled.log_scale), sm.centered(influence.mean(axis=2)), atol=1e-14, rtol=0
    )
    assert result["aggregates"]["wis"]["status"] == "source_chain_dispersion_only"
    assert result["aggregates"]["wis"]["mcse"] is None and not result["aggregates"]["wis"]["qualified"]
    assert row["wis_source_chain_dispersion"]["certified_mcse"] is False


def test_zero_empirical_true_outcome_has_infinite_log_score_no_floor():
    values = np.full((4, 1600, 5, 1), 0.9)
    result = sm.score_empirical_forecast(values, [0.2], ["invented"], [True], [False])
    assert result["rows"][0]["metrics"]["negative_log"]["positive_infinity"]
    assert (
        result["aggregates"]["negative_log"]["value"] is None
        and result["aggregates"]["negative_log"]["positive_infinity"]
    )
    assert result["aggregates"]["negative_log"]["count"] == 1
    assert result["aggregates"]["brier"]["value"] == 1


def test_underflowed_qualified_difference_keeps_log_mcse_and_requires_stream_evidence():
    def report(value):
        return dict(
            mask_race_ids=["a"],
            aggregates=dict(
                brier=dict(
                    value=value,
                    qualified=True,
                    mcse=None,
                    log_mcse=-1000.0,
                    uncertainty_scope="conditional_fixed_bank_Gibbs_integration",
                )
            ),
        )

    a, b = report(0.0), report(0.0)
    unbound = sm.paired_difference(a, b, "brier")
    assert unbound["mcse"] is None and not unbound["independent_arm_streams"]
    bound = sm.paired_difference(a, b, "brier", independent_streams=True)
    assert bound["mcse"] is None and bound["mcse_underflows"]
    assert np.isfinite(bound["log_mcse"]) and bound["numerically_resolved"] is False


def test_half_reference_and_tie_policy():
    result = sm.winner_reference_scores(0.5, [0.2, 0.5, 0.8])
    assert result["binary_count"] == 2 and result["tie_count"] == 1
    assert result["brier"] == 0.25 and result["negative_log"] == np.log(2)
    assert result["share_distribution"] is None
    with pytest.raises(ValueError):
        sm.winner_reference_scores(0, [0.2])


def test_known_analytic_zero_variance_composes_without_conflating_unresolved_none():
    def report(value, deterministic=False):
        row = dict(
            value=value,
            qualified=True,
            mcse=0.0 if deterministic else 0.0001,
            log_mcse=None if deterministic else np.log(0.0001),
            status="deterministic_analytic_no_MCMC" if deterministic else "qualified_conditional_numerical",
            uncertainty_scope="deterministic_gaussian_analytic"
            if deterministic
            else "conditional_fixed_bank_Gibbs_integration",
        )
        return dict(mask_race_ids=["a"], aggregates=dict(crps=row))

    mixed = sm.paired_difference(report(0.2), report(0.1, True), "crps")
    assert mixed["mcse"] == pytest.approx(0.0001) and mixed["deterministic_variance_composition"]
    analytic = sm.equal_cycle_summary({2018: report(0.1, True), 2022: report(0.2, True)}, "crps", tier="A")
    assert analytic["value"] == pytest.approx(0.15) and analytic["mcse"] == 0
    assert analytic["sampling_variance_zero"] and analytic["deterministic_quadrature_error_separate"]


def test_nested_analytic_tier_zero_variance_remains_certified_in_composition():
    def report(value, mcse, analytic):
        row = dict(
            value=value,
            qualified=True,
            mcse=mcse,
            log_mcse=None if analytic or mcse == 0 else np.log(mcse),
            status="deterministic_analytic_no_MCMC" if analytic else "qualified_conditional_numerical",
            uncertainty_scope="deterministic_gaussian_analytic"
            if analytic
            else "conditional_fixed_bank_Gibbs_integration",
        )
        return dict(mask_race_ids=["invented"], aggregates=dict(crps=row))

    analytic = sm.equal_cycle_summary({2018: report(0.1, 0.0, True), 2022: report(0.2, 0.0, True)}, "crps", tier="A")
    stochastic = sm.equal_cycle_summary(
        {2018: report(0.2, 0.0001, False), 2022: report(0.3, 0.0001, False)}, "crps", tier="A"
    )
    a = dict(mask_race_ids=["A"], aggregates=dict(crps=analytic))
    b = dict(mask_race_ids=["A"], aggregates=dict(crps=stochastic))
    mixed = sm.paired_difference(b, a, "crps")
    assert mixed["mcse"] == pytest.approx(0.0001 / np.sqrt(2))
    both = sm.paired_difference(a, a, "crps")
    assert both["mcse"] == 0 and both["sampling_variance_zero"]
    # Arbitrary rounded zero in a stochastic report is not a deterministic certificate.
    bad = report(0.2, 0.0, False)
    bad["aggregates"]["crps"]["log_mcse"] = None
    unresolved = sm.paired_difference(bad, report(0.1, 0.0, True), "crps")
    assert unresolved["mcse"] is None and not unresolved["sampling_variance_zero"]


def test_equal_cycle_primary_difference_combines_four_arm_cycle_variances():
    def report(value, mcse):
        return dict(
            mask_race_ids=["a"],
            aggregates=dict(
                crps=dict(
                    value=value,
                    qualified=True,
                    mcse=mcse,
                    log_mcse=np.log(mcse),
                    status="qualified_conditional_numerical",
                    uncertainty_scope="conditional_fixed_bank_Gibbs_integration",
                )
            ),
        )

    first = sm.equal_cycle_summary({2018: report(0.2, 0.0001), 2022: report(0.4, 0.0002)}, "crps", tier="A")
    second = sm.equal_cycle_summary({2018: report(0.1, 0.0003), 2022: report(0.3, 0.0004)}, "crps", tier="A")
    delta = sm.paired_difference(
        dict(mask_race_ids=["A"], aggregates=dict(crps=first)),
        dict(mask_race_ids=["A"], aggregates=dict(crps=second)),
        "crps",
        independent_streams=True,
    )
    assert delta["mcse"] == pytest.approx(np.sqrt(0.0001**2 + 0.0002**2 + 0.0003**2 + 0.0004**2) / 2)
