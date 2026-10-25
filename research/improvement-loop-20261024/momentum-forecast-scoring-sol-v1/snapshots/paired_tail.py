"""Complementary tail MCSE from one noncancelling probability coordinate.

Supplied inputs only: no RNG, labels, scores, fitting, retries or target I/O.
Original one-sided and raw indicator policies remain independently visible.
"""

import numpy as np
from observed_sampler import _observed_copy
from scipy.special import logsumexp
from trace_diagnostics import DiagnosticConfig, function_diagnostics, log_tail_diagnostics

METHOD = "paired_complementary_tail_scaled_coordinate_v2"
CLARIFICATION_SHA256 = "0d6dd97153687df2583c0a086e92e8a42c22eb0c850ed0a0133acdcb87c14024"
COMPLEMENT_LOG_TOLERANCE = 1e-12
RECOVERABLE_LOG_TOLERANCE = 1e-12


def require(ok, message):
    if not ok:
        raise ValueError(message)


def log1mexp(log_probability):
    """Stable log(1-exp(x)), x<=0; endpoint -inf and0 handled explicitly."""
    x = np.asarray(log_probability, float)
    require(
        not np.isnan(x).any() and not np.isposinf(x).any() and (x <= 0).all(), "Nonpositive log probabilities required"
    )
    output = np.empty_like(x)
    far = x <= -np.log(2)
    with np.errstate(divide="ignore", under="ignore", invalid="ignore"):
        output[far] = np.log1p(-np.exp(x[far]))
        output[~far] = np.log(-np.expm1(x[~far]))
    return output


def probability_coordinates(log_win, log_lose, names):
    """Immutable scaled centered coordinate for diagnostics and score consumers.

    `win_delta = win_deviation_sign * exp(log_scale) * centered_scaled_small`,
    and loser delta is exactly its negative. Keep the scale in logs whenever
    the actual absolute deviations underflow. No per-side1-p subtraction.
    """
    win, lose = np.asarray(log_win, float), np.asarray(log_lose, float)
    names = tuple(names)
    require(
        win.shape == lose.shape
        and win.ndim == 3
        and len(names) > 0
        and win.shape[2] == len(names)
        and len(set(names)) == len(names),
        "Aligned named chain/draw/race complementary log traces required",
    )
    require(
        not np.isnan(win).any() and not np.isnan(lose).any() and (win <= 0).all() and (lose <= 0).all(),
        "Nonpositive log tails required",
    )
    # Total complementary mass alone cannot validate a tiny rare side. When
    # the dominant negative log is normal-representable, -expm1 recovers that
    # rare tail with useful relative precision. Zero/subnormal dominant logs
    # cannot provide that cross-validation and are explicitly flagged instead.
    component_large_log = np.maximum(win, lose)
    component_small_log = np.minimum(win, lose)
    recoverable = (component_large_log < 0) & (abs(component_large_log) >= np.finfo(float).tiny)
    recovered_error = np.zeros_like(win)
    if recoverable.any():
        recovered = log1mexp(component_large_log[recoverable])
        recovered_error[recoverable] = abs(component_small_log[recoverable] - recovered)
        require(
            np.isfinite(recovered_error[recoverable]).all()
            and (recovered_error[recoverable] <= RECOVERABLE_LOG_TOLERANCE).all(),
            "Recoverably inconsistent complementary rare log tails",
        )
    normalization = np.logaddexp(win, lose)
    require(
        np.isfinite(normalization).all() and (abs(normalization) <= COMPLEMENT_LOG_TOLERANCE).all(),
        "Complementary log-tail drift exceeds tolerance",
    )
    # Remove only shared complementary-sum floating roundoff, never a floor.
    win = win - normalization
    lose = lose - normalization
    size = win.shape[0] * win.shape[1]
    pooled_win = logsumexp(win, axis=(0, 1)) - np.log(size)
    pooled_lose = logsumexp(lose, axis=(0, 1)) - np.log(size)
    small_is_win = pooled_win <= pooled_lose
    small = np.where(small_is_win[None, None, :], win, lose)
    log_scale = small.max(axis=(0, 1))
    valid = np.isfinite(log_scale)
    scaled = np.zeros_like(small)
    scaled[:, :, valid] = np.exp(small[:, :, valid] - log_scale[valid])
    means = scaled.mean(axis=(0, 1))
    centered = scaled - means
    log_small = np.full(len(names), -np.inf)
    chain_log_small = np.full((win.shape[0], len(names)), -np.inf)
    for j in np.flatnonzero(valid):
        relative = small[:, :, j] - log_scale[j]
        log_small[j] = log_scale[j] + logsumexp(relative) - np.log(size)
        chain_log_small[:, j] = log_scale[j] + logsumexp(relative, axis=1) - np.log(win.shape[1])
    log_large = log1mexp(log_small)
    chain_log_large = log1mexp(chain_log_small)
    log_probability_win = np.where(small_is_win, log_small, log_large)
    log_probability_lose = np.where(small_is_win, log_large, log_small)
    chain_log_win = np.where(small_is_win[None, :], chain_log_small, chain_log_large)
    chain_log_lose = np.where(small_is_win[None, :], chain_log_large, chain_log_small)
    return _observed_copy(
        dict(
            names=names,
            small_is_win=small_is_win,
            log_scale=log_scale,
            valid_small_side=valid,
            scaled_small_probabilities=scaled,
            centered_scaled_small=centered,
            log_small_probability=log_small,
            log_large_probability=log_large,
            log_win_probability=log_probability_win,
            log_lose_probability=log_probability_lose,
            per_chain_log_small_probability=chain_log_small,
            per_chain_log_win_probability=chain_log_win,
            per_chain_log_lose_probability=chain_log_lose,
            win_deviation_sign=np.where(small_is_win, 1, -1),
            lose_deviation_sign=np.where(small_is_win, -1, 1),
            method=METHOD,
            recoverable_log_tolerance=RECOVERABLE_LOG_TOLERANCE,
            relative_cross_validation_recoverable_count=recoverable.sum(axis=(0, 1)),
            relative_cross_validation_unavailable_count=(~recoverable).sum(axis=(0, 1)),
            recoverable_max_log_error=recovered_error.max(axis=(0, 1)),
            relative_cross_validation_unavailable_reason=(
                "dominant log probability is zero or subnormal; rare relative precision not recoverable there"
            ),
            complement_log_tolerance=COMPLEMENT_LOG_TOLERANCE,
        )
    )


def _exp_measurement(log_value):
    if log_value is None:
        return None, False
    with np.errstate(under="ignore"):
        value = float(np.exp(log_value))
    return (None, True) if value == 0 else (value, False)


def paired_tail_diagnostics(log_win, log_lose, names, *, structural_constants=(), config=DiagnosticConfig()):
    """Old-compatible side summaries, using shared complementary absolute MCSE.

    Absolute bulk, relative-tail and chain-log gates remain separate. Constants
    and all-minus-infinity smaller sides never create an empirical zero-error
    certificate. Original one-sided statuses are retained without alteration.
    """
    coordinates = probability_coordinates(log_win, log_lose, names)
    names = tuple(names)
    structural = set(structural_constants)
    require(structural.issubset(names), "Unknown structural coordinate")
    for j, name in enumerate(names):
        if name in structural:
            require(
                coordinates["valid_small_side"][j], "All-minus-infinity side cannot be structural precision evidence"
            )
            require(
                np.ptp(np.asarray(log_win)[:, :, j]) == 0 and np.ptp(np.asarray(log_lose)[:, :, j]) == 0,
                "Structural complementary tails must actually be constant",
            )
    common = function_diagnostics(
        coordinates["scaled_small_probabilities"], names, structural_constants=structural_constants, config=config
    )
    original_win = log_tail_diagnostics(log_win, names, structural_constants=structural_constants, config=config)
    original_lose = log_tail_diagnostics(log_lose, names, structural_constants=structural_constants, config=config)
    sides = {
        side: dict(
            method=METHOD,
            no_probability_floor=True,
            coordinates=[],
            config=common["config"],
            original_one_sided=original,
        )
        for side, original in [("win", original_win), ("lose", original_lose)]
    }
    for j, name in enumerate(names):
        base = common["coordinates"][j]
        valid = bool(coordinates["valid_small_side"][j])
        known = base["status"] == "structural"
        measured = base["status"] == "measured"
        mcse = base["mcse"]
        log_absolute = None if mcse is None or mcse == 0 else float(coordinates["log_scale"][j] + np.log(mcse))
        absolute, absolute_underflows = _exp_measurement(log_absolute)
        if known:
            absolute = 0.0
        for side in ["win", "lose"]:
            log_probability = float(coordinates[f"log_{side}_probability"][j])
            chain_log = np.asarray(coordinates[f"per_chain_log_{side}_probability"])[:, j]
            side_is_small = bool(coordinates["small_is_win"][j]) == (side == "win")
            if valid and side_is_small:
                # Center before subtracting huge shared log scales.
                logp = np.asarray(log_win if side == "win" else log_lose)[:, :, j]
                offset = float(np.max(logp))
                centered_log = logp - offset
                differences = (
                    logsumexp(centered_log, axis=1)
                    - np.log(logp.shape[1])
                    - (logsumexp(centered_log) - np.log(logp.size))
                )
            else:
                with np.errstate(invalid="ignore"):
                    differences = chain_log - log_probability
            log_relative = None if log_absolute is None else log_absolute - log_probability
            relative, relative_underflows = _exp_measurement(log_relative)
            if known:
                relative = 0.0
            usable = valid and (known or measured)
            gates = dict(
                absolute_mcse=None
                if not usable
                else bool(known or log_absolute <= np.log(config.max_win_absolute_mcse)),
                relative_mcse=None
                if not usable
                else bool(known or log_relative <= np.log(config.max_tail_relative_mcse)),
                chain_log_difference=None
                if not usable
                else bool(
                    np.isfinite(differences).all() and np.max(abs(differences)) <= config.max_chain_logtail_difference
                ),
            )
            row = dict(
                coordinate=name,
                status=base["status"] if valid else "unresolved_all_negative_infinity",
                structural_constant=bool(valid and base["structural_constant"]),
                log_probability=log_probability if np.isfinite(log_probability) else None,
                absolute_mcse=absolute if usable else None,
                log_absolute_mcse=log_absolute,
                absolute_mcse_underflows=absolute_underflows,
                relative_mcse=relative if usable else None,
                log_relative_mcse=log_relative,
                relative_mcse_underflows=relative_underflows,
                per_chain_log_difference=[float(x) if np.isfinite(x) else None for x in differences],
                chains_without_tail_support=np.flatnonzero(~np.isfinite(chain_log)).tolist(),
                rank_folded_split_rhat=base["rank_folded_split_rhat"],
                mean_ess_estimate=base["mean_ess_estimate"],
                tail_function_diagnostic_gates=base["gates"],
                gates=gates,
                failures=[key for key, value in gates.items() if value is False],
                smaller_side="win" if coordinates["small_is_win"][j] else "lose",
                probability_deviation_sign=int(coordinates[f"{side}_deviation_sign"][j]),
                absolute_mcse_shared_by_complement=True,
                input_consistency=dict(
                    recoverable_component_count=int(coordinates["relative_cross_validation_recoverable_count"][j]),
                    relative_cross_validation_unavailable_count=int(
                        coordinates["relative_cross_validation_unavailable_count"][j]
                    ),
                    recoverable_max_log_error=float(coordinates["recoverable_max_log_error"][j])
                    if coordinates["relative_cross_validation_recoverable_count"][j]
                    else None,
                    recoverable_log_tolerance=RECOVERABLE_LOG_TOLERANCE,
                    unavailable_reason=coordinates["relative_cross_validation_unavailable_reason"],
                ),
                original_one_sided_status=sides[side]["original_one_sided"]["coordinates"][j]["status"],
            )
            sides[side]["coordinates"].append(row)
    return dict(
        win=sides["win"],
        lose=sides["lose"],
        pairing=dict(
            method=METHOD,
            complementary_log_tolerance=COMPLEMENT_LOG_TOLERANCE,
            smaller_side_per_coordinate=["win" if x else "lose" for x in coordinates["small_is_win"]],
            larger_centered_probability_deviations="exact negative of smaller-side deviations",
            clarification_sha256=CLARIFICATION_SHA256,
            no_new_statistical_threshold_or_probability_law=True,
            recoverable_log_tolerance=RECOVERABLE_LOG_TOLERANCE,
            no_raw_indicator_certificate=True,
        ),
    )
