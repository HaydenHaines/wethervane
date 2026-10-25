"""Frozen F-only aggregate scores; target labels are supplied after fold sealing."""

import numpy as np

POLICY = dict(
    primary="empirical_share_CRPS_in_share_units",
    aggregation="equal_race_mean_within_cycle_then_equal_cycle_A_or_B_mean",
    intervals=[0.80, 0.95],
    quantiles="numpy_linear",
    winner="share_strictly_greater_than_half",
    log_probability_floor=1e-6,
    competitive="sealed_F0_win_probability_inclusive_0.1_to_0.9",
    near_even="sealed_F0_win_probability_inclusive_0.4_to_0.6",
    margin_bands=dict(narrow_absolute_max=0.02, large_directional_min=0.10),
    frequency_reference="mean_of_earlier_cycle_D_win_frequencies_equal_cycle_weight",
    numerical_error=(
        " "
        "max_batch50_batch100_original_posterior_rows_and_independent_chain_mean_variance; "
        "predictive_replicates_kept_contiguous; no race-iid election error"
    ),
    practical_CRPS_review_aid=dict(absolute=0.0005, relative=0.02),
    regression_guard=(
        "report every per-cycle and A/B loss; flag losses beyond two numerical MCSE; no automatic adoption"
    ),
    constant_win_precision="unresolved rare tail; zero delta MCSE is not precision evidence",
)


def crps_influence(values, truth):
    """Owner benchmark formula: empirical CRPS influence, original chain/draw order."""
    flat = values.reshape(-1, values.shape[-1])
    order = np.argsort(flat, axis=0, kind="stable")
    sorted_values = np.take_along_axis(flat, order, axis=0)
    n = len(flat)
    prefix = np.cumsum(sorted_values, axis=0) - sorted_values
    pair_sorted = ((2 * np.arange(n) - n)[:, None] * sorted_values + sorted_values.sum(axis=0) - 2 * prefix) / n
    pair = np.empty_like(flat)
    np.put_along_axis(pair, order, pair_sorted, axis=0)
    return (abs(flat - truth) - pair).reshape(values.shape)


def scores(values, truth, ids, weighted_crps):
    values = np.asarray(values, float)
    truth = np.asarray(truth, float)
    if values.ndim != 3 or truth.shape != (values.shape[-1],) or len(ids) != len(truth):
        raise ValueError("Aligned chain/draw/race predictions and explicit outcomes required")
    if (
        not np.isfinite(values).all()
        or not np.isfinite(truth).all()
        or (truth == 0.5).any()
        or ((values < 0) | (values > 1)).any()
        or ((truth <= 0) | (truth >= 1)).any()
    ):
        raise ValueError("Finite predictions and non-tied ordinary outcomes required")
    flat = values.reshape(-1, len(truth))
    prob = (values > 0.5).mean(axis=(0, 1))
    rows = []
    quantiles = np.quantile(flat, [0.025, 0.1, 0.5, 0.9, 0.975], axis=0, method="linear")
    for j, race in enumerate(ids):
        y = float(truth[j])
        win = y > 0.5
        p = float(prob[j])
        lo95, lo80, median, hi80, hi95 = quantiles[:, j]
        wis = 0.5 * abs(median - y)
        row = dict(
            race_id=race,
            actual_dem_share=y,
            mean_dem_share=float(flat[:, j].mean()),
            share_crps=weighted_crps(flat[:, j], np.ones(len(flat)), y),
            win_probability=p,
            win_brier=(p - int(win)) ** 2,
            win_log_score=-float(np.log(np.clip(p if win else 1 - p, 1e-6, 1))),
            margin_mae_pp=200 * abs(float(flat[:, j].mean()) - y),
            zero_empirical_true_outcome=int((p if win else 1 - p) == 0),
            winner_tail_precision="unresolved_constant_empirical_indicator"
            if p in (0, 1)
            else "estimated_by_chain_batch_mcse",
        )
        for level, alpha, lo, hi in ((80, 0.2, lo80, hi80), (95, 0.05, lo95, hi95)):
            width = hi - lo
            interval = width + 2 / alpha * (max(lo - y, 0) + max(y - hi, 0))
            wis += alpha / 2 * interval
            row.update(
                {
                    f"coverage{level}": bool(lo <= y <= hi),
                    f"width{level}": float(width),
                    f"interval_score{level}": float(interval),
                }
            )
        row.update(
            share_wis=float(wis / 2.5),
            expected_margin=float(2 * flat[:, j].mean() - 1),
            median_margin=float(2 * median - 1),
            narrow_margin_probability=float((abs(2 * flat[:, j] - 1) <= 0.02).mean()),
            large_D_margin_probability=float((2 * flat[:, j] - 1 >= 0.10).mean()),
            large_R_margin_probability=float((2 * flat[:, j] - 1 <= -0.10).mean()),
        )
        rows.append(row)
    return rows


def score_mcse(values, truth, correlated_mcse):
    probability = (values > 0.5).mean(axis=(0, 1))
    difference = (values > 0.5) - probability
    winner = (truth > 0.5).astype(float)
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    derivative = ((1 - winner) / (1 - clipped) - winner / clipped) * ((probability > 1e-6) & (probability < 1 - 1e-6))
    influence = np.stack(
        [
            crps_influence(values, truth).mean(axis=2),
            (2 * (probability - winner) * difference).mean(axis=2),
            (derivative * difference).mean(axis=2),
        ],
        axis=2,
    )
    checked = correlated_mcse(influence)
    return dict(zip(("share_crps", "win_brier", "win_log_score"), map(float, checked["mcse"]))), np.flatnonzero(
        (probability == 0) | (probability == 1)
    ).tolist()


def masks(ids, baseline):
    probability = (baseline > 0.5).mean(axis=(0, 1))
    return dict(
        all=list(ids),
        competitive=[race for race, p in zip(ids, probability) if 0.1 <= p <= 0.9],
        near_even=[race for race, p in zip(ids, probability) if 0.4 <= p <= 0.6],
    )


def summarize(rows):
    if not rows:
        return dict(status="unavailable_empty_mask", n=0)
    fields = (
        "share_crps",
        "share_wis",
        "win_brier",
        "win_log_score",
        "margin_mae_pp",
        "coverage80",
        "width80",
        "coverage95",
        "width95",
    )
    return dict(status="available", n=len(rows), **{key: float(np.mean([r[key] for r in rows])) for key in fields})
