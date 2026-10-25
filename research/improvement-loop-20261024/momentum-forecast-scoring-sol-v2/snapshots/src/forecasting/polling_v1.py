"""Standard aggregate poll update of the saved v5 fundamentals distribution.

This is a separately versioned aggregate comparison, not population_cut_v1.
Published poll estimates have a declared Gaussian working observation law.
The prior remains a finite mixture of authenticated historical posterior rows,
with one shared normal cycle effect and independent Student-t race effects.
No poll counts, missing crosstabs or election outcomes are constructed here.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky, solve_triangular
from scipy.special import expit

from src.forecasting.response_v5_reuse import correlated_mcse


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class EmpiricalFundamentals:
    offset: np.ndarray
    design: np.ndarray
    beta: np.ndarray
    tau_cycle: np.ndarray
    tau_race: np.ndarray
    race_ids: tuple[str, ...]
    source_identity: str
    race_df: float = 5.0

    def __post_init__(self):
        n, k = len(self.race_ids), len(self.beta)
        require(n > 0 and len(set(self.race_ids)) == n, "Unique nonempty race IDs required")
        require(bool(self.source_identity), "Historical posterior identity required")
        for name, shape in (
            ("offset", (n,)),
            ("design", (n, 2)),
            ("beta", (k, 2)),
            ("tau_cycle", (k,)),
            ("tau_race", (k,)),
        ):
            a = np.asarray(getattr(self, name), dtype=float).copy()
            require(a.shape == shape and np.isfinite(a).all(), f"Invalid {name}")
            a.setflags(write=False)
            object.__setattr__(self, name, a)
        require(k > 0 and (self.tau_cycle > 0).all() and (self.tau_race > 0).all(), "Positive scales required")
        require(np.isfinite(self.race_df) and self.race_df > 2, "Finite race df > 2 required")
        require(np.array_equal(self.design[:, 0], np.ones(n)), "Intercept and signed-incumbency design required")
        require(np.isin(self.design[:, 1], [-1, 0, 1]).all(), "Signed incumbency must be -1, 0 or 1")


@dataclass(frozen=True)
class GaussianPollLikelihood:
    """One Gaussian factor, including all specified observation dependence.

    `values` are conditional principal-contender log odds. `covariance` owns
    measurement errors and polling-date-to-election discrepancies, not the
    fundamentals' prior uncertainty. IDs name unique retained observations.
    Source selection/rounding/universe assumptions are owned by the adapter.
    """

    race_indices: np.ndarray
    values: np.ndarray
    covariance: np.ndarray
    observation_ids: tuple[str, ...]

    def __post_init__(self):
        m = len(self.observation_ids)
        require(len(set(self.observation_ids)) == m, "Duplicate poll observations")
        raw = np.asarray(self.race_indices)
        require(raw.shape == (m,) and np.isfinite(raw).all(), "Invalid poll race indices")
        require(np.equal(raw, np.floor(raw)).all() and (raw >= 0).all(), "Nonnegative integer race indices required")
        index = raw.astype(int).copy()
        y, r = np.asarray(self.values, float).copy(), np.asarray(self.covariance, float).copy()
        require(y.shape == (m,) and r.shape == (m, m), "Poll dimensions differ")
        require(np.isfinite(y).all() and np.isfinite(r).all(), "Nonfinite poll likelihood")
        require(np.allclose(r, r.T, atol=1e-12, rtol=1e-12), "Poll covariance must be symmetric")
        if m:
            cholesky(r, lower=True)  # A singular declared likelihood is rejected.
        for name, a in (("race_indices", index), ("values", y), ("covariance", r)):
            a.setflags(write=False)
            object.__setattr__(self, name, a)


def compress_likelihood(prior, polls):
    """Exact Gaussian sufficient statistics; marginalize unpolled race states."""
    require((polls.race_indices < len(prior.race_ids)).all(), "Poll references unknown race")
    observed, inverse = np.unique(polls.race_indices, return_inverse=True)
    operator = np.eye(len(observed))[inverse]
    if len(observed) == 0:
        return observed, np.zeros((0, 0)), np.zeros(0)
    factor = cho_factor(polls.covariance, lower=True)
    precision = operator.T @ cho_solve(factor, operator)
    linear = operator.T @ cho_solve(factor, polls.values)
    return observed, precision, linear


def conditional_state(precision, linear, location, tau_cycle, variance):
    """Mean and Cholesky of precision for joint (observed race logits, cycle)."""
    n = len(location)
    d = 1.0 / variance
    joint = np.zeros((n + 1, n + 1))
    joint[:n, :n] = precision + np.diag(d)
    joint[:n, n] = -d
    joint[n, :n] = -d
    joint[n, n] = d.sum() + 1.0 / tau_cycle**2
    rhs = np.r_[linear + d * location, -np.dot(d, location)]
    root = cholesky(joint, lower=True)
    return cho_solve((root, True), rhs), root


def conditional_row_log_weights(prior, observed, theta, cycle, mixture_precision):
    """O(K) categorical update using six sufficient statistics, without K×R."""
    u = theta - prior.offset[observed] - cycle
    s = prior.design[observed, 1]
    w = mixture_precision
    b0, b1 = prior.beta.T
    quadratic = (
        np.dot(w, u * u)
        - 2 * b0 * np.dot(w, u)
        - 2 * b1 * np.dot(w, u * s)
        + b0 * b0 * w.sum()
        + 2 * b0 * b1 * np.dot(w, s)
        + b1 * b1 * np.dot(w, s * s)
    )
    # The polynomial is an exact squared norm; tolerate only roundoff at zero.
    tolerance = 1e-10 * max(1.0, float(np.dot(w, u * u)))
    require(quadratic.min() >= -tolerance, "Unstable conditional squared norm")
    quadratic = np.maximum(quadratic, 0.0)
    return (
        -np.log(prior.tau_cycle)
        -len(observed) * np.log(prior.tau_race)
        -0.5 * (cycle / prior.tau_cycle) ** 2
        -0.5 * quadratic / prior.tau_race**2
    )


def _categorical(log_weights, rng):
    weights = np.exp(log_weights - np.max(log_weights))
    return int(np.searchsorted(np.cumsum(weights), rng.random() * weights.sum(), side="right"))


def sample_poll_posterior(prior, polls, *, chains=4, warmup=1000, draws=4000, seed=202610201):
    """Blocked Gibbs for the exact finite-mixture prior and Gaussian likelihood.

    Each iteration samples (theta,cycle), then the historical posterior row,
    then the independent Student-t mixture precisions. Unpolled race effects
    are integrated out during inference and generated afterward, conditional
    on the same posterior row and cycle. Returned draws retain chain order.
    Historical MCMC error remains in the empirical prior.
    """
    require(type(chains) is int and chains >= 2, "At least two chains required")
    require(type(warmup) is int and warmup >= 0, "Nonnegative warmup required")
    require(type(draws) is int and draws >= 10, "At least ten draws required")
    require(type(seed) is int and seed >= 0, "Nonnegative integer seed required")
    observed, precision, linear = compress_likelihood(prior, polls)
    n, k = len(prior.race_ids), len(prior.beta)
    missing = np.setdiff1d(np.arange(n), observed)
    logits = np.empty((chains, draws, n))
    cycles = np.empty((chains, draws))
    indices = np.empty((chains, draws), dtype=int)
    for chain, child_seed in enumerate(np.random.SeedSequence(seed).spawn(chains)):
        rng = np.random.default_rng(child_seed)
        if len(observed) == 0:
            rows = rng.integers(k, size=draws)
            c = rng.normal(size=draws) * prior.tau_cycle[rows]
            locations = prior.offset + prior.beta[rows] @ prior.design.T
            logits[chain] = (
                locations + c[:, None]
                + rng.standard_t(prior.race_df, size=(draws, n)) * prior.tau_race[rows, None]
            )
            cycles[chain], indices[chain] = c, rows
            continue
        row = int(rng.integers(k))
        lam = rng.gamma(prior.race_df / 2, 2 / prior.race_df, size=len(observed))
        for step in range(warmup + draws):
            location = prior.offset[observed] + prior.design[observed] @ prior.beta[row]
            mean, root = conditional_state(
                precision, linear, location, prior.tau_cycle[row], prior.tau_race[row] ** 2 / lam
            )
            state = mean + solve_triangular(root.T, rng.normal(size=len(observed) + 1), lower=False)
            theta, cycle = state[:-1], state[-1]
            row = _categorical(conditional_row_log_weights(prior, observed, theta, cycle, lam), rng)
            location = prior.offset[observed] + prior.design[observed] @ prior.beta[row]
            residual = (theta - location - cycle) / prior.tau_race[row]
            lam = rng.gamma((prior.race_df + 1) / 2, 2 / (prior.race_df + residual**2))
            if step >= warmup:
                j = step - warmup
                logits[chain, j, observed] = theta
                logits[chain, j, missing] = (
                    prior.offset[missing] + prior.design[missing] @ prior.beta[row] + cycle
                    + prior.tau_race[row] * rng.standard_t(prior.race_df, size=len(missing))
                )
                cycles[chain, j], indices[chain, j] = cycle, row
    return {
        "logits": logits,
        "shares": expit(logits),
        "first_wins": logits > 0,
        "shared_cycle": cycles,
        "posterior_index": indices,
        "beta": prior.beta[indices],
        "tau_cycle": prior.tau_cycle[indices],
        "tau_race": prior.tau_race[indices],
        "race_ids": prior.race_ids,
        "observed_race_indices": observed,
        "source_identity": prior.source_identity,
        "target_id": "national_senate_aggregate_poll_v1",
        "sampler": "empirical_mixture_student_t_blocked_gibbs_v1" if len(observed) else "direct_prior_v1",
    }


def posterior_diagnostics(result, *, batch_length=50):
    """Chain-aware diagnostics, including parameters hidden by predictive noise."""
    shares = result["shares"]
    p = np.concatenate(
        [
            shares,
            result["beta"],
            result["tau_cycle"][:, :, None],
            result["tau_race"][:, :, None],
            result["shared_cycle"][:, :, None],
        ],
        axis=2,
    )
    names = [f"share:{r}" for r in result["race_ids"]] + ["beta:office", "beta:incumbency", "tau_cycle", "tau_race", "cycle"]
    diag = correlated_mcse(p, batch_length=batch_length)
    win_diag = correlated_mcse(result["first_wins"].astype(float), batch_length=batch_length)
    variance = np.var(p.reshape(-1, p.shape[-1]), axis=0, ddof=1)
    ess = np.divide(variance, diag["mcse"] ** 2, out=np.full_like(variance, np.inf), where=diag["mcse"] > 0)
    return {
        "method": "split_rank_folded_rhat_and_multiscale_batch_chain_mcse",
        "coordinates": names,
        "rhat": diag["rank_rhat"].tolist(),
        "mean_mcse": diag["mcse"].tolist(),
        "mean_ess_estimate": [float(x) if np.isfinite(x) else None for x in ess],
        "win_probability_mcse": win_diag["mcse"].tolist(),
        "max_rhat": float(diag["rank_rhat"].max()),
        "max_share_mean_mcse": float(diag["mcse"][: shares.shape[-1]].max()),
        "max_win_probability_mcse": float(win_diag["mcse"].max()),
        "min_mean_ess_estimate_nonconstant": float(ess[np.isfinite(ess)].min()) if np.isfinite(ess).any() else None,
        "historical_prior_mcmc_error": "Inherited; additional forecast draws do not increase historical effective sample size",
    }
