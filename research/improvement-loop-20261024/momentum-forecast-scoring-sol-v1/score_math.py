"""Supplied forecast/label score arithmetic, never source loading or sampling."""

from dataclasses import dataclass

import numpy as np
from marginal import GaussianLogitMixture, weighted_interval_score
from paired_tail import probability_coordinates
from scipy.special import logsumexp
from trace_diagnostics import function_diagnostics

LEVELS = (0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)
METRICS = ("crps", "brier", "negative_log", "wis")
MAX_CYCLE_MCSE = dict(crps=0.0005, brier=0.0005, negative_log=0.005, wis=0.0005)
DESIGN_SHA256 = "45707d4a933ded2f7d4bffcbaab0fb6fb9a4ab7b1d62ad4494a27422b7d75678"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def shares(values, label):
    values = np.asarray(values, float)
    require(np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(), f"Finite bounded {label} required")
    return values


def centered(values):
    return values - values.mean(axis=(0, 1), keepdims=True)


@dataclass(frozen=True)
class ScaledInfluence:
    """Centered chain/draw influence = exp(log_scale)*values, with no zero floor."""

    values: np.ndarray
    log_scale: float = 0.0

    def __post_init__(self):
        value = np.asarray(self.values, float)
        require(
            value.ndim == 2 and np.isfinite(value).all() and np.isfinite(self.log_scale),
            "Finite scaled chain/draw influence required",
        )
        value = centered(value)
        magnitude = float(np.max(abs(value)))
        scale = float(self.log_scale)
        if magnitude:
            value = value / magnitude
            scale += np.log(magnitude)
        immutable = np.frombuffer(value.tobytes(), dtype=np.float64).reshape(value.shape)
        object.__setattr__(self, "values", immutable)
        object.__setattr__(self, "log_scale", scale)


def average_influences(influences):
    require(bool(influences), "Nonempty race influence list required")
    shape = influences[0].values.shape
    require(all(i.values.shape == shape for i in influences), "Common chain/draw coordinates required")
    scale = max(i.log_scale for i in influences)
    values = sum((np.exp(i.log_scale - scale) * i.values for i in influences), np.zeros(shape)) / len(influences)
    return ScaledInfluence(values, scale)


def numeric_log_measurement(log_value):
    if log_value is None:
        return None, False
    with np.errstate(under="ignore"):
        value = float(np.exp(log_value))
    return (None, True) if value == 0 else (value, False)


def finite_score(value, status="computed", reason=None):
    value = float(value)
    return dict(
        value=value if np.isfinite(value) else None,
        positive_infinity=bool(np.isposinf(value)),
        status=status,
        reason=reason,
    )


def empirical_quantiles(values):
    ordered = np.sort(np.asarray(values, float).reshape(-1))
    require(len(ordered) > 0, "Nonempty empirical distribution required")
    return {q: float(ordered[int(np.ceil(len(ordered) * q)) - 1]) for q in LEVELS}


def empirical_crps_and_influence(values, truth):
    """Exact empirical CRPS and centered influence in original supplied order."""
    values = shares(values, "empirical predictions")
    truth = float(shares(truth, "truth"))
    flat = values.reshape(-1)
    require(flat.size > 0, "Nonempty empirical predictions required")
    order = np.argsort(flat, kind="stable")
    ordered = flat[order]
    n = len(flat)
    prefix = np.cumsum(ordered) - ordered
    pair_sorted = ((2 * np.arange(n) - n) * ordered + ordered.sum() - 2 * prefix) / n
    pair = np.empty_like(flat)
    pair[order] = pair_sorted
    score = float(np.mean(abs(flat - truth)) - 0.5 * np.mean(pair))
    influence = (abs(flat - truth) - pair).reshape(values.shape)
    influence -= influence.mean()
    return score, influence


def gaussian_race(means, sds, truth, *, win_tail_qualified, lose_tail_qualified, deterministic=False):
    """Race scores with separately fallible numerical calculations and phases."""
    means = np.asarray(means, float)
    sds = np.asarray(sds, float)
    require(
        means.shape == sds.shape
        and means.ndim == 2
        and means.size > 0
        and np.isfinite(means).all()
        and np.isfinite(sds).all()
        and (sds > 0).all(),
        "Aligned finite chain/draw Gaussian parameters required",
    )
    require(
        isinstance(win_tail_qualified, (bool, np.bool_)) and isinstance(lose_tail_qualified, (bool, np.bool_)),
        "Explicit prelabel side flags required",
    )
    truth = float(shares(truth, "truth"))
    model = GaussianLogitMixture(means.ravel(), sds.ravel())
    lose_components, win_components = model.component_logtails(0.5)
    pair = probability_coordinates(
        win_components.reshape(means.shape)[..., None], lose_components.reshape(means.shape)[..., None], ["race"]
    )
    logwin = float(pair["log_win_probability"][0])
    loglose = float(pair["log_lose_probability"][0])
    tie = truth == 0.5
    winner = truth > 0.5
    result = dict(
        quantiles={},
        quantile_brackets={},
        win_probability=float(np.exp(logwin)),
        log_win=logwin,
        log_lose=loglose,
        tie=tie,
        metrics={},
        influences={},
        nonlinear_deltas={},
        wis_endpoint_or_density_unresolved=False,
    )
    # Binary scores never depend on success of a share quadrature.
    if tie:
        for metric in ["brier", "negative_log"]:
            result["metrics"][metric] = dict(
                value=None, positive_infinity=False, status="not_applicable_tie", reason="exact_binary_tie"
            )
    else:
        log_true = logwin if winner else loglose
        log_wrong = loglose if winner else logwin
        with np.errstate(under="ignore"):
            point_brier = float(np.exp(2 * log_wrong))
        result["metrics"]["brier"] = finite_score(point_brier)
        result["metrics"]["brier"].update(
            log_value=2 * log_wrong, point_underflows=bool(point_brier == 0 and np.isfinite(log_wrong))
        )
        qualified = bool(win_tail_qualified if winner else lose_tail_qualified)
        result["metrics"]["negative_log"] = finite_score(
            -log_true,
            "computed" if qualified else "prelabel_true_tail_unqualified",
            None if qualified else "true_outcome_side_failed_prelabel_gate",
        )
        result["metrics"]["negative_log"]["point_underflows"] = bool(log_true == 0 and np.isfinite(log_wrong))
        if pair["valid_small_side"][0]:
            delta = np.asarray(pair["centered_scaled_small"])[:, :, 0]
            prob_scale = float(pair["log_scale"][0])
            wrong_side = "lose" if winner else "win"
            true_side = "win" if winner else "lose"
            wrong_sign = int(pair[wrong_side + "_deviation_sign"][0])
            true_sign = -wrong_sign
            result["influences"]["brier"] = ScaledInfluence(wrong_sign * delta, np.log(2) + log_wrong + prob_scale)
            result["influences"]["negative_log"] = ScaledInfluence(-true_sign * delta, prob_scale - log_true)
            # Exact nonlinear Brier difference without subtracting near-one squares.
            chain_wrong = np.asarray(pair["per_chain_log_" + wrong_side + "_probability"])[:, 0]
            sum_logs = np.logaddexp(chain_wrong, log_wrong)
            sum_scale = float(np.max(sum_logs))
            chain_delta = delta.mean(axis=1)
            result["nonlinear_deltas"]["brier"] = (
                wrong_sign * chain_delta * np.exp(sum_logs - sum_scale),
                prob_scale + sum_scale,
            )
            if true_sign == 1:  # true side is the informative smaller side
                chain_true = np.asarray(pair["per_chain_log_" + true_side + "_probability"])[:, 0]
                result["nonlinear_deltas"]["negative_log"] = (-(chain_true - log_true), 0.0)
            else:
                ratio_scale_log = prob_scale - log_true
                # For sub-machine-epsilon ratios the analytic first-order value
                # differs from log1p(r)/r by less than floating relative precision.
                if ratio_scale_log < np.log(np.finfo(float).eps):
                    nonlinear = -true_sign * chain_delta
                else:
                    scale = np.exp(ratio_scale_log)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        nonlinear = -np.log1p(true_sign * chain_delta * scale) / scale
                result["nonlinear_deltas"]["negative_log"] = (nonlinear, ratio_scale_log)
        else:
            for metric in ["brier", "negative_log"]:
                result["metrics"][metric]["status"] = "tail_coordinate_unresolved"
                result["metrics"][metric]["reason"] = "informative_smaller_side_has_no_finite_log_values"
    quadrature = {}
    try:
        crps, receipt = model.crps(truth)
        result["metrics"]["crps"] = finite_score(crps)
        quadrature["crps"] = receipt
        if not deterministic:
            influence, receipt = model.crps_influence(truth)
            result["influences"]["crps"] = ScaledInfluence(influence.reshape(means.shape))
            quadrature["crps_influence"] = receipt
    except ArithmeticError as error:
        previous = result["metrics"].get("crps", finite_score(np.nan))
        previous.update(status="deterministic_integration_unresolved", reason=str(error))
        result["metrics"]["crps"] = previous
    quantiles = {}
    brackets = {}
    for q in LEVELS:
        quantiles[q], brackets[q] = model.quantile(q)
    result["quantiles"] = quantiles
    result["quantile_brackets"] = brackets
    result["metrics"]["wis"] = finite_score(weighted_interval_score(truth, quantiles))
    if not deterministic:
        density = np.array([model.density(quantiles[q]) for q in LEVELS])
        stable = bool(np.isfinite(density).all() and (density > 0).all())
        kinks = [q for q in LEVELS if brackets[q]["lower"] <= truth <= brackets[q]["upper"]]
        if stable and not kinks:
            influence = np.zeros(means.size)
            for j, q in enumerate(LEVELS):
                logcdf, _ = model.component_logtails(quantiles[q])
                q_i = -(np.exp(logcdf) - q) / density[j]
                influence += (float(truth < quantiles[q]) - q) * q_i / 3.5
            result["influences"]["wis"] = ScaledInfluence(influence.reshape(means.shape))
        else:
            result["wis_endpoint_or_density_unresolved"] = True
            result["metrics"]["wis"].update(
                status="uncertainty_unresolved",
                reason="one_sided_quantile_bracket_kink" if kinks else "unstable_quantile_density",
            )
            result["wis_one_sided_quantile_levels"] = kinks
    result["quadrature"] = quadrature
    return result


def gaussian_chain_scores(means, sds, truth):
    """Recompute per-chain nonlinear scores, with metric-specific failures."""
    output = {metric: [] for metric in METRICS}
    for c in range(len(means)):
        model = GaussianLogitMixture(means[c], sds[c])
        quantiles = {q: model.quantile(q)[0] for q in LEVELS}
        try:
            crps = model.crps(truth)[0]
        except ArithmeticError:
            crps = None
        output["crps"].append(crps)
        output["wis"].append(weighted_interval_score(truth, quantiles))
        logwin, loglose = model.log_win_lose()
        if truth == 0.5:
            output["brier"].append(None)
            output["negative_log"].append(None)
        else:
            log_true = logwin if truth > 0.5 else loglose
            log_wrong = loglose if truth > 0.5 else logwin
            output["brier"].append(np.exp(2 * log_wrong))
            output["negative_log"].append(-log_true)
    return output


def empirical_race(values, truth, *, win_tail_qualified, lose_tail_qualified):
    """Original source-chain,row,5-replicate axis is mandatory for F0 errors."""
    values = shares(values, "archived F0 predictions")
    require(
        values.ndim == 3 and values.shape[2] == 5 and values.shape[0] == 4,
        "Original4-source-chain/row/5-replicate F0 axes required",
    )
    truth = float(shares(truth, "truth"))
    tie = truth == 0.5
    quantiles = empirical_quantiles(values)
    p = float((values > 0.5).mean())
    crps, crps_i = empirical_crps_and_influence(values, truth)
    row_influence = crps_i.mean(axis=2)
    result = dict(
        quantiles=quantiles,
        win_probability=p,
        tie=tie,
        metrics={},
        influences={},
        uncertainty_scope=(
            "archived original source-row and predictive-replicate sampling; inherited bank dependence included"
        ),
    )
    result["metrics"]["crps"] = finite_score(crps)
    result["influences"]["crps"] = ScaledInfluence(row_influence)
    result["metrics"]["wis"] = finite_score(
        weighted_interval_score(truth, quantiles),
        "source_chain_dispersion_only",
        "no_smooth_empirical_density_or_certified_WIS_MCSE",
    )
    chain_wis = [weighted_interval_score(truth, empirical_quantiles(chain)) for chain in values]
    result["wis_source_chain_dispersion"] = dict(
        values=chain_wis, standard_deviation=float(np.std(chain_wis, ddof=1)), certified_mcse=False
    )
    if tie:
        for metric in ["brier", "negative_log"]:
            result["metrics"][metric] = dict(
                value=None, positive_infinity=False, status="not_applicable_tie", reason="exact_binary_tie"
            )
    else:
        z = float(truth > 0.5)
        row_p = (values > 0.5).mean(axis=2)
        result["metrics"]["brier"] = finite_score((p - z) ** 2)
        result["influences"]["brier"] = ScaledInfluence(2 * (p - z) * (row_p - p))
        true_probability = p if truth > 0.5 else 1 - p
        qualified = bool(win_tail_qualified if truth > 0.5 else lose_tail_qualified)
        if true_probability == 0:
            result["metrics"]["negative_log"] = finite_score(
                np.inf, "prelabel_true_tail_unqualified", "zero_empirical_true_outcome_probability"
            )
        else:
            result["metrics"]["negative_log"] = finite_score(
                -np.log(true_probability),
                "computed" if qualified else "prelabel_true_tail_unqualified",
                None if qualified else "true_outcome_side_failed_prelabel_gate",
            )
            result["influences"]["negative_log"] = ScaledInfluence(
                -(row_p - p) / p if truth > 0.5 else (row_p - p) / (1 - p)
            )
    return result


def metric_aggregate(
    rows,
    influences,
    chain_scores,
    metric,
    *,
    deterministic=False,
    uncertainty_scope="conditional_fixed_bank_Gibbs_integration",
):
    """Average races BEFORE MCSE; never drop numerically failed races."""
    selected = [j for j, row in enumerate(rows) if not (row["tie"] and metric in ("brier", "negative_log"))]
    if not selected:
        return dict(
            value=None,
            positive_infinity=False,
            status="empty_binary_or_race_mask",
            count=0,
            mcse=None,
            log_mcse=None,
            mcse_underflows=False,
            qualified=False,
            uncertainty_scope="deterministic_gaussian_analytic" if deterministic else uncertainty_scope,
            unqualified_race_indices=[],
            numerical_limit=MAX_CYCLE_MCSE[metric],
        ), None
    values = [rows[j]["metrics"][metric]["value"] for j in selected]
    has_infinity = any(rows[j]["metrics"][metric]["positive_infinity"] for j in selected)
    point = None if has_infinity or any(v is None for v in values) else float(np.mean(values))
    unavailable = [j for j in selected if rows[j]["metrics"][metric]["status"] != "computed"]
    output = dict(
        value=point,
        positive_infinity=has_infinity,
        count=len(selected),
        mcse=None,
        qualified=False,
        status="metric_unresolved" if unavailable else "computed",
        unqualified_race_indices=unavailable,
        uncertainty_scope="deterministic_gaussian_analytic" if deterministic else uncertainty_scope,
        numerical_limit=MAX_CYCLE_MCSE[metric],
    )
    if deterministic and not unavailable and point is not None:
        output.update(
            qualified=True,
            status="deterministic_analytic_no_MCMC",
            mcse=0.0,
            log_mcse=None,
            sampling_variance_zero=True,
            quadrature_error_separate=True,
        )
        return output, None
    if metric == "wis" and uncertainty_scope.startswith("archived"):
        output["status"] = "source_chain_dispersion_only"
        return output, None
    if point is None or unavailable or any(influences[j].get(metric) is None for j in selected):
        return output, None
    average = average_influences([influences[j][metric] for j in selected])
    diagnostic = function_diagnostics(average.values[..., None], [metric])
    check = diagnostic["coordinates"][0]
    output["influence_diagnostic"] = diagnostic
    if check["mcse"] is None:
        output["status"] = "uncertainty_unresolved"
        return output, average
    output["log_mcse"] = average.log_scale + np.log(check["mcse"])
    output["mcse"], output["mcse_underflows"] = numeric_log_measurement(output["log_mcse"])
    passed = bool(output["log_mcse"] <= np.log(MAX_CYCLE_MCSE[metric]))
    nonlinear = None
    if chain_scores is None and uncertainty_scope == "conditional_fixed_bank_Gibbs_integration":
        output["status"] = "nonlinear_check_unavailable"
        return output, average
    if chain_scores is not None:
        if any(value is None for j in selected for value in chain_scores[j][metric]):
            output["status"] = "nonlinear_check_unavailable"
            return output, average
        nonlinear = np.mean(np.stack([chain_scores[j][metric] for j in selected], axis=1), axis=1)
        if any(value is None for j in selected for value in chain_scores[j][metric]):
            output["status"] = "nonlinear_check_unavailable"
            return output, average
        nonlinear_scaled = np.zeros(average.values.shape[0])
        for j in selected:
            if metric in rows[j].get("nonlinear_deltas", {}):
                delta_values, delta_scale = rows[j]["nonlinear_deltas"][metric]
                with np.errstate(over="ignore", under="ignore"):
                    converted = np.asarray(delta_values) * np.exp(delta_scale - average.log_scale)
            else:
                difference = np.asarray(chain_scores[j][metric]) - rows[j]["metrics"][metric]["value"]
                converted = np.zeros_like(difference)
                nonzero = difference != 0
                with np.errstate(over="ignore", under="ignore"):
                    converted[nonzero] = np.sign(difference[nonzero]) * np.exp(
                        np.log(abs(difference[nonzero])) - average.log_scale
                    )
            nonlinear_scaled += converted / len(selected)
        residual = nonlinear_scaled - average.values.mean(axis=1)
        output["nonlinear_chain_check"] = dict(
            scores=nonlinear.tolist(),
            linearization_residual_scaled=[float(x) if np.isfinite(x) else None for x in residual],
            residual_log_scale=average.log_scale,
            max_scaled_residual=float(np.max(abs(residual))) if np.isfinite(residual).all() else None,
            scaled_limit=3 * check["mcse"],
            passed=bool(np.isfinite(residual).all() and np.max(abs(residual)) <= 3 * check["mcse"]),
        )
        passed = passed and output["nonlinear_chain_check"]["passed"]
    output.update(
        qualified=bool(passed), status="qualified_conditional_numerical" if passed else "uncertainty_unresolved"
    )
    return output, average


def score_gaussian_forecast(
    means, sds, truths, race_ids, win_tail_qualified, lose_tail_qualified, *, deterministic=False
):
    means = np.asarray(means, float)
    sds = np.asarray(sds, float)
    truths = shares(truths, "truths")
    race_ids = tuple(race_ids)
    require(
        means.shape == sds.shape
        and means.ndim == 3
        and means.shape[2] == len(race_ids) == len(truths)
        and len(set(race_ids)) == len(race_ids),
        "Exact aligned Gaussian/race/truth axes required",
    )
    require(
        len(win_tail_qualified) == len(lose_tail_qualified) == len(race_ids),
        "Prelabel per-side qualification axes required",
    )
    rows = []
    chain = []
    for j, race in enumerate(race_ids):
        try:
            row = gaussian_race(
                means[:, :, j],
                sds[:, :, j],
                truths[j],
                win_tail_qualified=win_tail_qualified[j],
                lose_tail_qualified=lose_tail_qualified[j],
                deterministic=deterministic,
            )
            chain.append(gaussian_chain_scores(means[:, :, j], sds[:, :, j], truths[j]) if not deterministic else None)
        except ArithmeticError as error:
            # Numerical unresolved is preserved; source/axis errors are never caught here.
            row = dict(
                tie=bool(truths[j] == 0.5),
                metrics={
                    metric: dict(
                        value=None,
                        positive_infinity=False,
                        status="deterministic_integration_unresolved",
                        reason=str(error),
                    )
                    for metric in METRICS
                },
                influences={},
            )
            chain.append(None)
        row["race_id"] = race
        row["actual_dem_share"] = float(truths[j])
        rows.append(row)
    influence = [row["influences"] for row in rows]
    aggregates = {}
    cycle_influence = {}
    for metric in METRICS:
        aggregates[metric], cycle_influence[metric] = metric_aggregate(
            rows,
            influence,
            None if deterministic or any(c is None for c in chain) else chain,
            metric,
            deterministic=deterministic,
        )
    for row in rows:
        row.pop("model", None)
        row.pop("influences", None)
        row.pop("nonlinear_deltas", None)
    return dict(
        rows=rows,
        aggregates=aggregates,
        binary_tie_count=int(np.sum(truths == 0.5)),
        mask_race_ids=list(race_ids),
        cycle_influences=cycle_influence,
    )


def score_empirical_forecast(values, truths, race_ids, win_tail_qualified, lose_tail_qualified):
    values = shares(values, "archived predictions")
    truths = shares(truths, "truths")
    race_ids = tuple(race_ids)
    require(
        values.ndim == 4
        and values.shape[2] == 5
        and values.shape[0] == 4
        and values.shape[3] == len(race_ids) == len(truths),
        "Exact4-source-chain/row/5-replicate/race F0 axes required",
    )
    rows = [
        empirical_race(
            values[:, :, :, j],
            truths[j],
            win_tail_qualified=win_tail_qualified[j],
            lose_tail_qualified=lose_tail_qualified[j],
        )
        for j in range(len(race_ids))
    ]
    influence = [row["influences"] for row in rows]
    aggregates = {}
    cycle_influence = {}
    for metric in METRICS:
        aggregates[metric], cycle_influence[metric] = metric_aggregate(
            rows, influence, None, metric, uncertainty_scope="archived_original_source_row_and_predictive_simulation"
        )
    for j, row in enumerate(rows):
        row["race_id"] = race_ids[j]
        row["actual_dem_share"] = float(truths[j])
        row.pop("influences", None)
    return dict(
        rows=rows,
        aggregates=aggregates,
        binary_tie_count=int(np.sum(truths == 0.5)),
        mask_race_ids=list(race_ids),
        cycle_influences=cycle_influence,
    )


def winner_reference_scores(probability, truths):
    require(np.isfinite(probability) and 0 < probability < 1, "Proper fixed winner-reference probability required")
    truths = shares(truths, "truths")
    eligible = truths != 0.5
    z = truths[eligible] > 0.5
    return dict(
        probability=float(probability),
        binary_count=int(eligible.sum()),
        tie_count=int((~eligible).sum()),
        brier=None if not len(z) else float(np.mean((probability - z) ** 2)),
        negative_log=None if not len(z) else float(np.mean(np.where(z, -np.log(probability), -np.log1p(-probability)))),
        share_distribution=None,
    )


def _sampling_log_variance(row):
    if not row.get("qualified"):
        return False, None
    if row.get("mcse") == 0 and (
        row.get("status") == "deterministic_analytic_no_MCMC"
        or (
            row.get("sampling_variance_zero") is True
            and row.get("uncertainty_scope") == "deterministic_gaussian_analytic"
        )
    ):
        return True, -np.inf
    log_mcse = row.get("log_mcse")
    if log_mcse is not None and np.isfinite(log_mcse):
        return True, 2 * log_mcse
    return False, None


def paired_difference(first, second, metric, *, independent_streams=False):
    a, b = first["aggregates"][metric], second["aggregates"][metric]
    require(first["mask_race_ids"] == second["mask_race_ids"], "Paired comparisons require exact fixed race masks")
    require(type(independent_streams) is bool, "Explicit independent-stream evidence flag required")
    value = None if a["value"] is None or b["value"] is None else a["value"] - b["value"]
    scopes = [x.get("uncertainty_scope") for x in [a, b]]
    conditional = "conditional_fixed_bank_Gibbs_integration"
    analytic = "deterministic_gaussian_analytic"
    independent = independent_streams and all(scope == conditional for scope in scopes)
    deterministic_mix = all(scope in (conditional, analytic) for scope in scopes) and analytic in scopes
    va, la = _sampling_log_variance(a)
    vb, lb = _sampling_log_variance(b)
    usable = (independent or deterministic_mix) and va and vb
    log_mcse = float(0.5 * np.logaddexp(la, lb)) if usable else None
    exact_zero = bool(usable and np.isneginf(log_mcse))
    if exact_zero:
        mcse, underflows = 0.0, False
        log_mcse = None
    else:
        mcse, underflows = numeric_log_measurement(log_mcse)
    log_three = None if log_mcse is None else np.log(3) + log_mcse
    resolved = (
        None
        if value is None or not usable
        else bool(value != 0)
        if exact_zero
        else bool(value != 0 and np.log(abs(value)) >= log_three)
    )
    return dict(
        value=value,
        mcse=mcse,
        log_mcse=log_mcse,
        mcse_underflows=underflows,
        sampling_variance_zero=exact_zero,
        three_mcse=None if mcse is None else 3 * mcse,
        log_three_mcse=log_three,
        numerically_resolved=resolved,
        independent_arm_streams=independent,
        deterministic_variance_composition=deterministic_mix,
        deterministic_quadrature_error_separate=True,
        inherited_bank_error_included=False,
    )


def equal_cycle_summary(cycle_reports, metric, *, tier):
    required = {"A": (2018, 2022), "B": (2020, 2024)}[tier]
    if not all(c in cycle_reports for c in required):
        return dict(
            status="missing_required_cycles",
            cycles=list(required),
            value=None,
            mcse=None,
            log_mcse=None,
            qualified=False,
            missing_cycles=[c for c in required if c not in cycle_reports],
        )
    rows = [cycle_reports[c]["aggregates"][metric] for c in required]
    value = None if any(r["value"] is None for r in rows) else float(np.mean([r["value"] for r in rows]))
    qualified = all(r["qualified"] for r in rows)
    measurements = [_sampling_log_variance(r) for r in rows]
    usable = qualified and all(valid for valid, _ in measurements)
    log_mcse = float(0.5 * logsumexp([logvar for _, logvar in measurements]) - np.log(2)) if usable else None
    zero = bool(usable and np.isneginf(log_mcse))
    if zero:
        mcse, underflows = 0.0, False
        log_mcse = None
    else:
        mcse, underflows = numeric_log_measurement(log_mcse)
    return dict(
        cycles=list(required),
        value=value,
        mcse=mcse,
        log_mcse=log_mcse,
        mcse_underflows=underflows,
        uncertainty_scope=rows[0].get("uncertainty_scope")
        if all(r.get("uncertainty_scope") == rows[0].get("uncertainty_scope") for r in rows)
        else "mixed",
        sampling_variance_zero=zero,
        qualified=qualified,
        status="deterministic_analytic_no_MCMC"
        if zero
        else "qualified_conditional_numerical"
        if qualified
        else "metric_unresolved",
        deterministic_quadrature_error_separate=True,
        election_generalization_confidence_interval=False,
    )
