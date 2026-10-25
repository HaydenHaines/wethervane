"""Deterministic marginal math; supplied inputs only, with no source/label I/O.

Gaussian components are conditional logit laws, not independent posterior draws.
Their caller owns chain coordinates, mixture weights and numerical admission.
"""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.integrate import IntegrationWarning, quad, quad_vec
from scipy.special import log_ndtr, logit, logsumexp


def require(ok, message):
    if not ok:
        raise ValueError(message)


def unit(value):
    value = float(value)
    require(np.isfinite(value) and 0 <= value <= 1, "Finite share/probability in [0, 1] required")
    return value


def frozen(value):
    array = np.ascontiguousarray(value, dtype=np.float64)
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


def integral(function, lower, upper, *, vector=False, max_error=1e-8):
    """Bounded deterministic refinement, independent of any model score.

    Error estimates are quadrature estimates, never Monte Carlo precision.
    """
    if lower == upper:
        return function(lower) * 0, dict(attempt=0, estimated_error=0.0)
    attempts = []
    previous = None
    for attempt, (absolute, relative, limit) in enumerate(((1e-9, 1e-8, 200), (1e-10, 1e-9, 400))):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", IntegrationWarning)
            if vector:
                value, error, info = quad_vec(
                    function,
                    lower,
                    upper,
                    epsabs=absolute,
                    epsrel=relative,
                    norm="max",
                    limit=limit,
                    workers=1,
                    cache_size=32_000_000,
                    full_output=True,
                )
                success = bool(info.success)
            else:
                value, error = quad(function, lower, upper, epsabs=absolute, epsrel=relative, limit=limit)
                success = True
        clean = not any(issubclass(item.category, IntegrationWarning) for item in caught)
        finite = bool(np.isfinite(value).all() and np.isfinite(error))
        agreement = None if previous is None else float(np.max(np.abs(np.asarray(value) - previous)))
        row = dict(
            attempt=attempt,
            estimated_error=float(error),
            successful=success,
            finite=finite,
            no_integration_warning=clean,
            refinement_difference=agreement,
        )
        attempts.append(row)
        if finite and clean and success and error <= max_error and (previous is None or agreement <= 2e-8):
            return value, {**row, "attempts": attempts}
        previous = np.asarray(value)
    raise ArithmeticError(f"Deterministic marginal integration unresolved: {attempts}")


@dataclass(frozen=True)
class GaussianLogitMixture:
    mean: np.ndarray
    sd: np.ndarray
    weights: np.ndarray | None = None

    def __post_init__(self):
        mean, sd = np.asarray(self.mean, float), np.asarray(self.sd, float)
        require(mean.ndim == 1 and mean.size > 0 and mean.shape == sd.shape, "Aligned nonempty vectors required")
        require(
            np.isfinite(mean).all() and np.isfinite(sd).all() and (sd > 0).all(),
            "Finite means and positive scales required",
        )
        weights = np.ones(len(mean)) if self.weights is None else np.asarray(self.weights, float)
        require(
            weights.shape == mean.shape and np.isfinite(weights).all() and (weights > 0).all(),
            "Positive finite aligned weights required",
        )
        total = weights.sum()
        require(np.isfinite(total) and total > 0, "Finite positive weight sum required")
        object.__setattr__(self, "mean", frozen(mean))
        object.__setattr__(self, "sd", frozen(sd))
        object.__setattr__(self, "weights", frozen(weights / total))

    def component_logtails(self, x):
        x = unit(x)
        if x == 0:
            return np.full_like(self.mean, -np.inf), np.zeros_like(self.mean)
        if x == 1:
            return np.zeros_like(self.mean), np.full_like(self.mean, -np.inf)
        z = (logit(x) - self.mean) / self.sd
        return log_ndtr(z), log_ndtr(-z)

    def logtails(self, x):
        lower, upper = self.component_logtails(x)
        logweights = np.log(self.weights)
        return float(logsumexp(logweights + lower)), float(logsumexp(logweights + upper))

    def cdf(self, x):
        return float(np.exp(self.logtails(x)[0]))

    def sf(self, x):
        return float(np.exp(self.logtails(x)[1]))

    def log_win_lose(self):
        lose, win = self.logtails(0.5)
        return win, lose

    def density(self, x):
        x = unit(x)
        if x in (0, 1):
            return 0.0
        z = (logit(x) - self.mean) / self.sd
        logdens = -0.5 * z**2 - 0.5 * np.log(2 * np.pi) - np.log(self.sd) - np.log(x) - np.log1p(-x)
        return float(np.exp(logsumexp(np.log(self.weights) + logdens)))

    def quantile(self, probability):
        probability = unit(probability)
        if probability in (0, 1):
            return probability, dict(lower=probability, upper=probability, width=0.0)
        lower, upper = 0.0, 1.0
        while upper - lower > 1e-10:
            middle = (lower + upper) / 2
            loglower, logupper = self.logtails(middle)
            below = loglower < np.log(probability) if probability <= 0.5 else logupper > np.log1p(-probability)
            if below:
                lower = middle
            else:
                upper = middle
        return (lower + upper) / 2, dict(
            lower=lower, upper=upper, width=upper - lower, cdf_lower=self.cdf(lower), cdf_upper=self.cdf(upper)
        )

    def mean_share(self):
        return integral(self.sf, 0.0, 1.0)

    def component_mean_share(self):
        return integral(lambda x: np.exp(self.component_logtails(x)[1]), 0.0, 1.0, vector=True)

    def crps(self, truth):
        truth = unit(truth)
        lower, receipt_lower = integral(lambda x: self.cdf(x) ** 2, 0.0, truth, max_error=5e-9)
        upper, receipt_upper = integral(lambda x: self.sf(x) ** 2, truth, 1.0, max_error=5e-9)
        error = receipt_lower["estimated_error"] + receipt_upper["estimated_error"]
        require(error <= 1e-8, "Summed CRPS integration error unresolved")
        return float(lower + upper), dict(lower=receipt_lower, upper=receipt_upper, estimated_error=error)

    def crps_influence(self, truth):
        truth = unit(truth)

        def low(x):
            components = np.exp(self.component_logtails(x)[0])
            average = self.weights @ components
            return 2 * average * (components - average)

        def high(x):
            components = np.exp(self.component_logtails(x)[1])
            average = self.weights @ components
            return 2 * average * (components - average)

        lower, rlo = integral(low, 0.0, truth, vector=True)
        upper, rhi = integral(high, truth, 1.0, vector=True)
        return lower + upper, dict(lower=rlo, upper=rhi)


def weighted_interval_score(truth, quantiles):
    """Standard central 50/80/95 WIS, K=3 and denominator 3.5."""
    truth = unit(truth)
    levels = (0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975)
    require(set(quantiles) == set(levels), "Exact seven WIS quantiles required")
    values = np.array([unit(quantiles[q]) for q in levels])
    require((np.diff(values) >= 0).all(), "Ordered quantiles required")
    score = 0.5 * abs(truth - quantiles[0.5])
    for alpha in (0.50, 0.20, 0.05):
        lo, hi = quantiles[alpha / 2], quantiles[1 - alpha / 2]
        interval = hi - lo + 2 / alpha * (max(lo - truth, 0) + max(truth - hi, 0))
        score += alpha / 2 * interval
    return float(score / 3.5)


def winner_scores(log_win, log_lose, winner):
    """No probability clipping; numerical qualification is the caller's job."""
    require(type(winner) is bool, "Explicit boolean winner required")
    require(np.isfinite(log_win) and np.isfinite(log_lose) and max(log_win, log_lose) <= 0, "Finite log tails required")
    require(abs(np.logaddexp(log_win, log_lose)) <= 1e-12, "Complementary log tails required")
    # For a D win use the losing tail, avoiding loss of precision in 1-p.
    error = np.exp(log_lose if winner else log_win)
    return dict(brier=float(error**2), negative_log_score=float(-log_win if winner else -log_lose))
