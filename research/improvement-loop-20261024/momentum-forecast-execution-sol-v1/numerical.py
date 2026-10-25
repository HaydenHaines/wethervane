"""Prelabel Gaussian marginals and separate selected numerical flags."""

import numpy as np
from scipy.special import log_ndtr


def require(ok, message):
    if not ok:
        raise ValueError(message)


def all_measured_gates(value, keys=None):
    rows = value["coordinates"]
    if not rows:
        return False
    for row in rows:
        gates = row["gates"]
        required = set(gates) if keys is None else set(keys)
        if not required or not required.issubset(gates):
            return False
        if row["status"] not in ("measured", "structural"):
            return False
        structural = row["status"] == "structural" and row.get("structural_constant") is True
        if row["status"] == "structural" and not structural:
            return False
        if not all(gates[key] is True or (gates[key] is None and structural) for key in required):
            return False
    return True


def component_means(means, sds, math, *, chunk=65536):
    """Shared adaptive nodes over each vector chunk, never quadrature per draw."""
    require(means.shape == sds.shape and means.ndim == 3, "aligned chain/draw/race Gaussian arrays required")
    output = np.empty(means.size)
    m, s = means.reshape(-1), sds.reshape(-1)
    receipts = []
    for start in range(0, len(m), chunk):
        stop = min(start + chunk, len(m))
        value, receipt = math.GaussianLogitMixture(m[start:stop], s[start:stop]).component_mean_share()
        output[start:stop] = value
        receipts.append(dict(start=start, stop=stop, quadrature=receipt))
    return output.reshape(means.shape), receipts


def conditional_summary(result, trace, modules):
    diag, math = modules["trace"], modules["math"]
    arrays = trace["arrays"]
    observed = np.asarray(arrays["observed_race_indices"], int)
    names = [result["race_ids"][j] for j in observed]
    # Each component conditions on incoming row/lambda and integrates Gaussian state noise.
    # Equal-weight averaging of components integrates the retained row/lambda states.
    # Equal empirical component mass: no second multiplication by row weights.
    means = arrays["gaussian_mean_theta_cycle"][..., :-1]
    sds = np.sqrt(arrays["gaussian_variance_theta_cycle"][..., :-1])
    require(means.shape[-1] == len(names), "Gaussian/common-polled axis mismatch")
    expected, quad_receipts = component_means(means, sds, math)
    log_win, log_lose = log_ndtr(means / sds), log_ndtr(-means / sds)
    core = diag.core_state_diagnostics(result, trace)
    mean_gate = diag.mean_share_diagnostics(expected, names)
    paired = modules["paired"].paired_tail_diagnostics(log_win, log_lose, names)
    win, lose = paired["win"], paired["lose"]
    bank = diag.bank_diagnostics(
        arrays["bank_batch_weight_sums"], arrays["bank_batch_counts"], arrays["posterior_source_chain"]
    )
    levels = (0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)
    quantiles, interval_rows = [], []
    component_cdfs = []
    widths, interval_names = [], []
    for race, name in enumerate(names):
        mix = math.GaussianLogitMixture(means[..., race].reshape(-1), sds[..., race].reshape(-1))
        race_quantiles = []
        for q in levels:
            x, bisection = mix.quantile(q)
            logcdf, _ = mix.component_logtails(x)
            cdf = np.exp(logcdf).reshape(means.shape[:2])
            cdf_diag = diag.function_diagnostics(cdf[..., None], [f"{name}:q{q}"], max_mean_mcse=0.0025)
            mcse = cdf_diag["coordinates"][0]["mcse"]
            if mcse is None:
                bracket = [0.0, 1.0]
            else:
                lo, _ = mix.quantile(max(0.0, q - 3 * mcse))
                hi, _ = mix.quantile(min(1.0, q + 3 * mcse))
                bracket = [lo, hi]
            widths.append(bracket[1] - bracket[0])
            interval_names.append(f"{name}:q{q}")
            component_cdfs.append(cdf)
            race_quantiles.append(x)
            interval_rows.append(
                dict(
                    race_id=name,
                    probability=q,
                    quantile=x,
                    bisection=bisection,
                    three_mcse_probability_bracket=bracket,
                    bracket_method="selected frozen_local_pointwise_MCSE_at_nominal_quantile",
                    cdf_diagnostic=cdf_diag,
                )
            )
        quantiles.append(race_quantiles)
    interval = diag.interval_display_diagnostics(np.stack(component_cdfs, axis=2), interval_names, widths)
    raw_win = diag.raw_indicator_diagnostics(result["first_wins"][..., observed], names)
    core_ok = all_measured_gates(core)
    mean_ok = all_measured_gates(mean_gate, {"mean_mcse"})
    absolute_win_ok = all_measured_gates(win, {"absolute_mcse"}) and all_measured_gates(lose, {"absolute_mcse"})
    bank_ok = all(bank["gates"].values())
    retry = dict(
        core_state=not core_ok,
        bulk_mean_share=not mean_ok,
        bulk_absolute_win_probability=not absolute_win_ok,
        bank_chain_consistency=not bank_ok,
    )
    return dict(
        schema="MomentumPollingPrelabelNumericalSummaryV1",
        common_polled_race_ids=names,
        core_state=core,
        mean_share=mean_gate,
        win_side=win,
        lose_side=lose,
        tail_pairing=paired["pairing"],
        raw_win_indicators=raw_win,
        bank=bank,
        interval_display=interval,
        interval_details=interval_rows,
        interval_bracket_method=(
            "selected local frozen pointwise MCSE at nominal quantile; no envelope/confidence/calibration guarantee"
        ),
        quadrature=quad_receipts,
        gates=dict(
            core_state=core_ok,
            bulk_mean_share=mean_ok,
            bulk_absolute_win_probability=absolute_win_ok,
            bank_chain_consistency=bank_ok,
        ),
        permitted_retry_triggers=retry,
        any_permitted_retry_trigger=any(retry.values()),
        tail_qualification=dict(
            win=all_measured_gates(win, {"relative_mcse", "chain_log_difference"}),
            lose=all_measured_gates(lose, {"relative_mcse", "chain_log_difference"}),
        ),
        interval_display_qualified=all_measured_gates(interval, {"mean_mcse", "share_bracket_width"}),
        inherited_bank_warning=bank["bank_approximation_sensitive"],
        mixture="equal pre-state Gaussian components; no categorical reweighting",
        numeric_target_labels=False,
        scores=False,
    ), dict(
        component_expected_shares=expected,
        component_log_win=log_win,
        component_log_lose=log_lose,
        quantile_levels=np.asarray(levels),
        quantiles=np.asarray(quantiles),
        common_polled_race_indices=observed,
        common_polled_race_ids=np.asarray(names),
    )
