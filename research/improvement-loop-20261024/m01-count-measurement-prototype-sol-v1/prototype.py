"""Supplied-state M01 mathematics only; no sampler or native provider.

One scale row governs the whole vector. Calibration input weights may update
inside each population conditional; protected population masses never update.
The existing bank is restricted to the authenticated six-poll TX2026 fixture.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

MANIFEST_SHA = "0fbec6366c2640f30a17c2fc576b87303d9f4065a632b1488b9a72311345fcb8"
CONSUMER_SHA = "f2f28207125b9474ddc92b22b75b814c962e415c3152df6a9170a15e4dd1fd70"
REVIEW_SHA = "3a07125bed8944a7c9582a9c50d484cec154d1a3f76fa9c716eea70bf9d43b54"
TARGET = "M01_marginal_bayes_complete_poll_vector_likelihood_v1"
CUTOFF = "2026-10-20T22:13:33Z"


def need(condition, message):
    if not condition:
        raise ValueError(message)


def immutable(value):
    value = np.ascontiguousarray(value, dtype=np.float64)
    return np.frombuffer(value.tobytes(), dtype=np.float64).reshape(value.shape)


def normalized_weights(value, size):
    w = np.asarray(value, dtype=np.float64)
    need(w.shape == (size,) and np.isfinite(w).all() and (w >= 0).all()
         and abs(float(w.sum()) - 1) <= 1e-12, "Nonnegative normalized weights required")
    # Preserve externally supplied masses exactly. Their floating-point sum may
    # differ from one within the admitted tolerance; do not change every row's
    # original weight to remove that rounding residual.
    return immutable(w)


def stable_logsumexp(terms):
    terms = np.asarray(terms, dtype=np.float64)
    need(terms.ndim == 1 and terms.size > 0 and not np.isnan(terms).any()
         and not np.isposinf(terms).any() and np.isfinite(terms).any(),
         "Unsupported log terms or floating-point range")
    maximum = float(terms.max())
    result = maximum + np.log(np.exp(terms - maximum).sum())
    need(np.isfinite(result), "Unsupported log likelihood range")
    return float(result)


@dataclass(frozen=True, init=False)
class SharedVectorMixture:
    """Frozen finite Gaussian mixture with a single component for the vector."""

    covariances: np.ndarray
    weights: np.ndarray
    precisions: np.ndarray
    log_constants: np.ndarray
    log_weights: np.ndarray

    def __init__(self, covariances, weights):
        c = np.asarray(covariances, dtype=np.float64)
        need(c.ndim == 3 and c.shape[0] > 0 and c.shape[1] > 0 and c.shape[1] == c.shape[2]
             and np.isfinite(c).all(), "Finite nonempty square covariance bank required")
        need(np.allclose(c, c.transpose(0, 2, 1), rtol=0, atol=1e-12), "Asymmetric covariance")
        c = (c + c.transpose(0, 2, 1)) / 2
        w = normalized_weights(weights, len(c))
        precisions, constants = [], []
        for matrix in c:
            try:
                lower = np.linalg.cholesky(matrix)
            except np.linalg.LinAlgError as error:
                raise ValueError("Every covariance must be positive definite; no jitter") from error
            precision = np.linalg.solve(lower.T, np.linalg.solve(lower, np.eye(len(matrix))))
            need(np.isfinite(precision).all(), "Covariance precision exceeds supported floating range")
            precisions.append(precision)
            constants.append(-0.5 * (len(matrix) * np.log(2 * np.pi)
                                     + 2 * np.log(np.diag(lower)).sum()))
        with np.errstate(divide="ignore"):
            logs = np.log(w)
        for key, value in dict(covariances=c, weights=w, precisions=precisions,
                               log_constants=constants, log_weights=logs).items():
            object.__setattr__(self, key, immutable(value))

    @property
    def dimension(self):
        return self.covariances.shape[1]

    def _vector(self, value):
        value = np.asarray(value, dtype=np.float64)
        need(value.shape == (self.dimension,) and np.isfinite(value).all(),
             "Complete finite vector required")
        return value

    def evaluate(self, y, mean):
        residual = self._vector(y) - self._vector(mean)
        with np.errstate(over="ignore", invalid="ignore"):
            quadratic = np.einsum("i,sij,j->s", residual, self.precisions, residual)
        need(np.isfinite(quadratic).all(), "Unsupported component quadratic range")
        terms = self.log_weights + self.log_constants - 0.5 * quadratic
        logp = stable_logsumexp(terms)
        responsibilities = np.exp(terms - logp)
        gradient = np.einsum("s,sij,j->i", responsibilities, self.precisions, residual)
        need(np.isfinite(gradient).all(), "Unsupported gradient range")
        return logp, gradient, responsibilities

    def symbolic_logpdf(self, y, mean):
        """Pure tensor expression; bank/weights/observations are constants."""
        import pytensor.tensor as pt
        from pytensor.raise_op import CheckAndRaise

        y = self._vector(y)
        need(mean.ndim == 1 and mean.dtype == "float64", "Float64 symbolic mean vector required")
        mean = pt.specify_shape(mean, (self.dimension,))
        mean = CheckAndRaise(ValueError, "Complete finite symbolic mean required")(
            mean, pt.all(~pt.isnan(mean) & ~pt.isinf(mean)))
        residual = pt.as_tensor_variable(y) - mean
        quadratic = pt.sum(residual[None, :, None] * pt.as_tensor_variable(self.precisions)
                           * residual[None, None, :], axis=(1, 2))
        # Check every component, including zero-weight rows, before exp/logsumexp:
        # a dead term must not introduce 0*infinity into its symbolic derivative.
        quadratic = CheckAndRaise(ValueError, "Unsupported symbolic component quadratic range")(
            quadratic, pt.all(~pt.isnan(quadratic) & ~pt.isinf(quadratic)))
        terms = pt.as_tensor_variable(self.log_weights + self.log_constants) - 0.5 * quadratic
        valid = (~pt.any(pt.isnan(terms)) & ~pt.any(pt.isinf(terms) & (terms > 0))
                 & pt.any(~pt.isinf(terms)))
        terms = CheckAndRaise(ValueError, "Unsupported symbolic log likelihood range")(terms, valid)
        maximum = pt.max(terms)
        return maximum + pt.log(pt.sum(pt.exp(terms - maximum)))


@dataclass(frozen=True)
class CountPairPort:
    """Tiny supplied count-rate interface, not an installed native adapter."""

    candidate_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    candidate_pairs: tuple[tuple[str, str], ...]

    def __post_init__(self):
        need(isinstance(self.candidate_ids, tuple) and len(self.candidate_ids) > 1
             and len(set(self.candidate_ids)) == len(self.candidate_ids)
             and all(isinstance(v, str) and v for v in self.candidate_ids), "Unique candidate IDs required")
        need(isinstance(self.observation_ids, tuple) and self.observation_ids
             and len(set(self.observation_ids)) == len(self.observation_ids)
             and all(isinstance(v, str) and v for v in self.observation_ids), "Unique observation owners required")
        need(isinstance(self.candidate_pairs, tuple) and len(self.candidate_pairs) == len(self.observation_ids)
             and all(isinstance(p, tuple) and len(p) == 2 and p[0] != p[1]
                     and set(p) <= set(self.candidate_ids) for p in self.candidate_pairs),
             "Distinct supported named pair required per observation")

    def mean_from_rates(self, rates):
        rates = np.asarray(rates, dtype=np.float64)
        need(rates.shape == (len(self.candidate_ids),) and np.isfinite(rates).all()
             and (rates > 0).all(), "Finite positive candidate rates required; no clipping")
        logs = np.log(rates)
        return np.array([logs[self.candidate_ids.index(a)] - logs[self.candidate_ids.index(b)]
                         for a, b in self.candidate_pairs])

    def symbolic_mean_from_log_rates(self, logs):
        import pytensor.tensor as pt

        need(logs.ndim == 1 and logs.dtype == "float64", "Float64 symbolic log rates required")
        logs = pt.specify_shape(logs, (len(self.candidate_ids),))
        return pt.stack([logs[self.candidate_ids.index(a)] - logs[self.candidate_ids.index(b)]
                         for a, b in self.candidate_pairs])

    def claim_observations(self, existing):
        """Local duplicate-factor guard; never mutates an external registry."""
        owned = frozenset(existing)
        need(not owned.intersection(self.observation_ids), "Duplicate observation factor")
        return owned.union(self.observation_ids)


def protected_parent_joint(log_kernel, parent_masses):
    """Finite reference: normalize downstream states separately for every U."""
    kernel = np.asarray(log_kernel, dtype=np.float64)
    need(kernel.ndim == 2 and all(n > 0 for n in kernel.shape), "Parent/state table required")
    masses = normalized_weights(parent_masses, kernel.shape[0])
    conditional = np.array([np.exp(row - stable_logsumexp(row)) for row in kernel])
    return masses[:, None] * conditional


def load_frozen_current_case(root, *, target_cycle=2026, cutoff=CUTOFF, target=TARGET):
    """Authenticate the old consumer before import; no raw historical labels.

    This delegates existing bank reconciliation to its pinned independent
    review and preserves its current-only supplied-state scope. No empirical
    native/model admission is granted by this helper.
    """
    root = Path(root).resolve()
    for name, expected in [("input-manifest.json", MANIFEST_SHA), ("consumer.py", CONSUMER_SHA)]:
        need(hashlib.sha256((root / name).read_bytes()).hexdigest() == expected,
             "Pinned consumer source/manifest mismatch before import")
    name = "m01_frozen_supplied_consumer_" + hashlib.sha256(str(root).encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(name, root / "consumer.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.verify_snapshots(root)
    bank = module.load_bank(root, expected_review_sha256=REVIEW_SHA, target_cycle=target_cycle,
                            forecast_cutoff_utc=cutoff, target_label=target)
    coordinates, y, fixed, error_covariance = module.actual_inputs(root)
    sys.path.insert(0, str(root))  # authenticated primitive import for the old consumer
    reference = module.make_mixture(bank, coordinates, error_covariance)
    return dict(bank=bank, coordinates=coordinates, y=immutable(y), fixed=immutable(fixed),
                component=json.loads((root / "inputs/poll-component.json").read_text()),
                mixture=SharedVectorMixture(reference.covariances, bank.weights),
                saved_reference=reference, manifest=json.loads((root / "input-manifest.json").read_text()))
