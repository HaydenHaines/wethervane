"""Normalized Gaussian scale mixture for ONE complete poll vector.

One shared component index governs the entire vector. This is a MARGINAL
likelihood: conditioning on observations updates component weights. It is NOT a
calibration cut and does not preserve calibration weights in the posterior.
The caller owns posterior provenance, calibration-data reuse, and normalized
within-population joint inference. This module is not a full cut engine, sampler,
prior, posterior-bank constructor, or scientific activation adapter. Covariance
components must come from the unchanged provider; no kernel is implemented here.

Background: https://mc-stan.org/docs/stan-users-guide/posterior-prediction.html
Cut normalizers: Carmona and Nicholls, Semi-Modular Inference (2020):
https://proceedings.mlr.press/v108/carmona20a.html
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_solve
from scipy.special import logsumexp


def _immutable(value):
    """Bytes-backed arrays cannot be made writable again via setflags."""
    array = np.ascontiguousarray(value, dtype=np.float64)
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


@dataclass(frozen=True, init=False, eq=False)
class GaussianScaleMixture:
    covariances: np.ndarray
    weights: np.ndarray
    _precisions: np.ndarray
    _log_constants: np.ndarray

    def __init__(self, covariances, weights):
        covariances = np.array(covariances, dtype=np.float64, copy=True)
        weights = np.array(weights, dtype=np.float64, copy=True)
        if covariances.ndim != 3 or covariances.shape[0] == 0:
            raise ValueError("covariances must have shape (S,N,N), S,N positive")
        components, dimension, columns = covariances.shape
        if dimension == 0 or dimension != columns:
            raise ValueError("covariances must be square nonempty matrices")
        if weights.shape != (components,) or not np.all(np.isfinite(weights)):
            raise ValueError("one finite weight is required per component")
        if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-12):
            raise ValueError("weights must be nonnegative and normalized")
        if not np.all(np.isfinite(covariances)):
            raise ValueError("covariances must be finite")
        if not np.allclose(covariances, covariances.transpose(0, 2, 1), rtol=0, atol=1e-12):
            raise ValueError("covariances must be symmetric")
        # Remove only accepted floating-point asymmetry, without diagonal repair.
        covariances = (covariances + covariances.transpose(0, 2, 1)) / 2
        precisions, constants = [], []
        for covariance in covariances:
            try:
                factor = np.linalg.cholesky(covariance)
            except np.linalg.LinAlgError as error:
                raise ValueError("every covariance must be positive definite") from error
            precisions.append(cho_solve((factor, True), np.eye(dimension)))
            constants.append(-0.5 * (dimension * np.log(2 * np.pi) + 2 * np.log(np.diag(factor)).sum()))
        object.__setattr__(self, "covariances", _immutable(covariances))
        object.__setattr__(self, "weights", _immutable(weights / weights.sum()))
        object.__setattr__(self, "_precisions", _immutable(precisions))
        object.__setattr__(self, "_log_constants", _immutable(constants))

    def _terms(self, y, mean):
        y, mean = np.asarray(y, dtype=float), np.asarray(mean, dtype=float)
        shape = (self.covariances.shape[1],)
        if y.shape != shape or mean.shape != shape:
            raise ValueError("y and mean must each be one complete vector of length N")
        if not np.all(np.isfinite(y)) or not np.all(np.isfinite(mean)):
            raise ValueError("y and mean must be finite")
        residual = y - mean
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            terms = self._log_constants - 0.5 * np.einsum("i,sij,j->s", residual, self._precisions, residual)
            terms = terms + np.log(self.weights)
        normalizer = logsumexp(terms)
        if not np.isfinite(normalizer):
            raise ValueError("likelihood exceeds supported floating-point range")
        return residual, terms, normalizer

    def logpdf(self, y, mean):
        return float(self._terms(y, mean)[2])

    def responsibilities(self, y, mean):
        """P(shared component | y, mean); these generally differ from input weights."""
        _, terms, normalizer = self._terms(y, mean)
        return np.exp(terms - normalizer)

    def gradient_mean(self, y, mean):
        """Gradient of the joint-vector log density with respect to mean."""
        residual, terms, normalizer = self._terms(y, mean)
        responsibilities = np.exp(terms - normalizer)
        return np.einsum("s,sij,j->i", responsibilities, self._precisions, residual)
