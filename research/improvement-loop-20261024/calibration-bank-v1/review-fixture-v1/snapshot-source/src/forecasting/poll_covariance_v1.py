"""Total-error covariance for published principal-contender polling estimates.

This working Gaussian law is on log odds. Its independent component includes
sampling, weighting, rounding and residual measurement error. Do NOT add a
second sampling/MOE term to an empirically fitted total-error component.
Brownian discrepancies are anchored to zero on Election Day and are separate
from persistent poll bias. Parameters are SDs; movement uses sqrt(30-day units).
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from src.forecasting.polling_v1 import GaussianPollLikelihood, require

COMPONENTS = ("independent", "cycle", "race", "house", "movement")


@dataclass(frozen=True)
class PollCoordinates:
    observation_ids: tuple[str, ...]
    race_ids: tuple[str, ...]
    cycles: tuple[str, ...]
    pollsters: tuple[str, ...]
    days_to_election: np.ndarray

    def __post_init__(self):
        n = len(self.observation_ids)
        require(n > 0 and len(set(self.observation_ids)) == n, "Unique nonempty observations required")
        for key in ("race_ids", "cycles", "pollsters"):
            value = tuple(getattr(self, key))
            require(len(value) == n and all(isinstance(x, str) and x for x in value), f"Invalid {key}")
            object.__setattr__(self, key, value)
        days = np.asarray(self.days_to_election, float).copy()
        require(days.shape == (n,) and np.isfinite(days).all() and (days >= 0).all(), "Invalid poll age")
        days.setflags(write=False)
        object.__setattr__(self, "days_to_election", days)

    def kernels(self):
        cycle = np.asarray(self.cycles)
        race = np.asarray(self.race_ids)
        house = np.asarray(self.pollsters)
        same_cycle = cycle[:, None] == cycle
        same_race = (race[:, None] == race) & same_cycle
        return np.array([
            np.eye(len(cycle)),
            same_cycle.astype(float),
            same_race.astype(float),
            ((house[:, None] == house) & same_cycle).astype(float),
            same_race * np.minimum.outer(self.days_to_election, self.days_to_election) / 30.0,
        ])


def error_covariance(coordinates, standard_deviations):
    scales = np.array([standard_deviations[key] for key in COMPONENTS], float)
    require(np.isfinite(scales).all() and (scales >= 0).all() and scales[0] > 0,
            "Positive independent scale and nonnegative other scales required")
    return np.einsum("k,kij->ij", scales**2, coordinates.kernels())


def build_poll_likelihood(race_ids, coordinates, shares, standard_deviations):
    shares = np.asarray(shares, float)
    require(shares.shape == (len(coordinates.observation_ids),) and np.isfinite(shares).all(),
            "Matching finite conditional shares required")
    require(((shares > 0) & (shares < 1)).all(), "Interior conditional shares required")
    mapping = {race: index for index, race in enumerate(race_ids)}
    require(len(mapping) == len(race_ids) and set(coordinates.race_ids) <= set(mapping),
            "Known unique target race IDs required")
    return GaussianPollLikelihood(
        np.array([mapping[race] for race in coordinates.race_ids]),
        np.log(shares) - np.log1p(-shares),
        error_covariance(coordinates, standard_deviations),
        coordinates.observation_ids,
    )


def gaussian_error_objective(log_scales, blocks, *, prior_medians, prior_log_sd):
    """Exact Gaussian likelihood + Gaussian regularizer on log SD coordinates.

    This is a penalized likelihood / log-scale MAP, not integrated uncertainty
    for covariance parameters. `blocks` may split independent election cycles.
    All kernels are supplied; no training outcome is read inside this function.
    """
    variances = np.exp(2 * np.asarray(log_scales))
    value, gradient = 0.0, np.zeros(len(COMPONENTS))
    for residual, kernels in blocks:
        covariance = np.einsum("k,kij->ij", variances, kernels)
        root = cho_factor(covariance, lower=True)
        alpha = cho_solve(root, residual)
        value += np.log(np.diag(root[0])).sum() + 0.5 * np.dot(residual, alpha)
        value += 0.5 * len(residual) * np.log(2 * np.pi)
        inverse = cho_solve(root, np.eye(len(residual)))
        derivative = inverse - np.outer(alpha, alpha)
        gradient += variances * np.einsum("ij,kij->k", derivative, kernels)
    standardized = (log_scales - np.log(prior_medians)) / prior_log_sd
    value += 0.5 * np.dot(standardized, standardized)
    gradient += standardized / prior_log_sd
    return float(value), gradient


def fit_error_scales(coordinates, residuals, *, prior_medians, prior_log_sd,
                     starts=(0.5, 1.0, 2.0)):
    """Fit only supplied historical residuals; callers own label/cutoff gates."""
    residuals = np.asarray(residuals, float)
    medians = np.array([prior_medians[key] for key in COMPONENTS], float)
    require(residuals.shape == (len(coordinates.observation_ids),) and np.isfinite(residuals).all(),
            "Finite aligned historical residuals required")
    require(np.isfinite(medians).all() and (medians > 0).all(), "Positive regularizer medians required")
    require(np.isfinite(prior_log_sd) and prior_log_sd > 0, "Positive log-SD regularization required")
    blocks = []
    cycle = np.asarray(coordinates.cycles)
    for name in sorted(set(cycle)):
        selected = np.flatnonzero(cycle == name)
        block = PollCoordinates(
            tuple(coordinates.observation_ids[i] for i in selected),
            tuple(coordinates.race_ids[i] for i in selected),
            tuple(coordinates.cycles[i] for i in selected),
            tuple(coordinates.pollsters[i] for i in selected),
            coordinates.days_to_election[selected],
        )
        blocks.append((residuals[selected], block.kernels()))
    objective = lambda value: gaussian_error_objective(
        value, blocks, prior_medians=medians, prior_log_sd=prior_log_sd
    )
    attempts = [
        minimize(objective, np.log(medians * start), jac=True, method="L-BFGS-B",
                 bounds=[(np.log(1e-5), np.log(2.0))] * len(COMPONENTS),
                 options={"ftol": 1e-11, "gtol": 1e-6, "maxiter": 300})
        for start in starts
    ]
    successful = [result for result in attempts if result.success and np.isfinite(result.fun)]
    require(bool(successful), "Historical error fit did not converge")
    best = min(successful, key=lambda result: result.fun)
    return {
        "schema": "TotalPollErrorScalesV1",
        "standard_deviations": dict(zip(COMPONENTS, np.exp(best.x).tolist())),
        "regularization": {"log_scale_medians": prior_medians, "log_scale_sd": prior_log_sd},
        "conditional_parameter_estimates": True,
        "objective": float(best.fun),
        "gradient_max": float(np.abs(best.jac).max()),
        "attempts": [
            {"success": bool(a.success), "message": str(a.message), "objective": float(a.fun),
             "iterations": int(a.nit), "gradient_max": float(np.abs(a.jac).max())}
            for a in attempts
        ],
        "training_cycles": sorted(set(coordinates.cycles)),
        "training_observations": len(residuals),
        "units": "principal-contender log odds; movement per sqrt(30 days)",
        "qualification": "Regularized historical working-error fit; covariance uncertainty not integrated",
    }
