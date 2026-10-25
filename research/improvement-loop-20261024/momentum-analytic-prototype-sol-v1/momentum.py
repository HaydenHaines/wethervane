"""Supplied-coordinate six-scale Gaussian mathematics; no fitting interface.

Five byte-pinned incumbent error kernels plus a symmetric race/cycle slope.
Log-SD penalty has precisely the incumbent's coordinate/constant convention.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import source_guard

SOURCE_BINDINGS = source_guard.verify()

import numpy as np  # noqa: E402
from scipy.linalg import cho_factor, cho_solve  # noqa: E402

INCUMBENT = source_guard.incumbent()
COMPONENTS = INCUMBENT.COMPONENTS + ("persistent_linear",)
require = source_guard.require


def real_array(value, shape, label):
    original = np.asarray(value)
    require(np.isrealobj(original) and original.dtype.kind in "fiu", "real numeric " + label + " required")
    result = np.asarray(value, dtype=float)
    require(result.shape == shape and np.isfinite(result).all(), "finite aligned " + label + " required")
    # Immutable byte-backed arrays cannot be made writable through setflags.
    return np.frombuffer(result.tobytes(), dtype=float).reshape(shape)


def scalar(value, label):
    array = real_array(value, (), label)
    return float(array)


@dataclass(frozen=True)
class PollCoordinatesV1:
    observation_ids: tuple[str, ...]
    race_ids: tuple[str, ...]
    cycles: tuple[str, ...]
    pollsters: tuple[str, ...]
    days_to_election: np.ndarray

    def __post_init__(self):
        require(not isinstance(self.observation_ids, str), "observation sequence required")
        ids = tuple(self.observation_ids)
        n = len(ids)
        require(
            n > 0 and all(isinstance(v, str) and v.strip() for v in ids) and len(set(ids)) == n,
            "unique nonempty observations required",
        )
        for name in ("observation_ids", "race_ids", "cycles", "pollsters"):
            source = getattr(self, name)
            require(not isinstance(source, str), "named coordinate sequence required")
            values = tuple(source)
            require(
                len(values) == n and all(isinstance(v, str) and v.strip() for v in values),
                "nonempty aligned " + name + " required",
            )
            object.__setattr__(self, name, values)
        days = real_array(self.days_to_election, (n,), "poll ages")
        require((days >= 0).all(), "nonnegative days before election required")
        object.__setattr__(self, "days_to_election", days)

    def incumbent_coordinates(self):
        return INCUMBENT.PollCoordinates(
            self.observation_ids, self.race_ids, self.cycles, self.pollsters, self.days_to_election
        )

    def kernels(self):
        old = self.incumbent_coordinates().kernels()
        race, cycle = np.asarray(self.race_ids), np.asarray(self.cycles)
        x = self.days_to_election / 30
        slope = ((race[:, None] == race) & (cycle[:, None] == cycle)) * np.outer(x, x)
        require(np.isfinite(slope).all(), "finite slope kernel required")
        return np.concatenate((old, slope[None, :, :]), axis=0)


def scale_vector(scales):
    require(isinstance(scales, Mapping) and set(scales) == set(COMPONENTS), "exact six named scales required")
    values = real_array([scalar(scales[name], name + " scale") for name in COMPONENTS], (6,), "scales")
    require((values >= 0).all() and values[0] > 0, "positive independent and nonnegative other scales required")
    return values


def error_covariance(coordinates, standard_deviations):
    require(isinstance(coordinates, PollCoordinatesV1), "versioned coordinates required")
    scales = scale_vector(standard_deviations)
    base = INCUMBENT.error_covariance(
        coordinates.incumbent_coordinates(), dict(zip(INCUMBENT.COMPONENTS, scales[:5], strict=True))
    )
    covariance = base + scales[5] ** 2 * coordinates.kernels()[5]
    require(np.isfinite(covariance).all(), "finite covariance required")
    # No jitter, clipping, extra nominal-N term, or silent SPD fallback.
    np.linalg.cholesky(covariance)
    return covariance


def build_poll_likelihood(race_ids, coordinates, shares, standard_deviations):
    """Existing Gaussian factor type, under the experiment-only augmented law."""
    require(isinstance(coordinates, PollCoordinatesV1), "versioned coordinates required")
    require(not isinstance(race_ids, str), "target race sequence required")
    race_ids = tuple(race_ids)
    require(
        len(race_ids) > 0
        and all(isinstance(r, str) and r.strip() for r in race_ids)
        and len(set(race_ids)) == len(race_ids),
        "unique nonempty target race IDs required",
    )
    require(set(coordinates.race_ids) <= set(race_ids), "known target race IDs required")
    values = real_array(shares, (len(coordinates.observation_ids),), "conditional shares")
    require(((values > 0) & (values < 1)).all(), "interior conditional shares required")
    mapping = {race: i for i, race in enumerate(race_ids)}
    return INCUMBENT.GaussianPollLikelihood(
        np.array([mapping[r] for r in coordinates.race_ids]),
        np.log(values) - np.log1p(-values),
        error_covariance(coordinates, standard_deviations),
        coordinates.observation_ids,
    )


@dataclass(frozen=True)
class ResidualBlockV1:
    coordinates: PollCoordinatesV1
    residual: np.ndarray

    def __post_init__(self):
        require(isinstance(self.coordinates, PollCoordinatesV1), "versioned coordinates required")
        object.__setattr__(
            self, "residual", real_array(self.residual, (len(self.coordinates.observation_ids),), "supplied residuals")
        )


def validated_blocks(blocks):
    blocks = tuple(blocks)
    require(
        len(blocks) > 0 and all(isinstance(b, ResidualBlockV1) for b in blocks),
        "nonempty versioned residual blocks required",
    )
    seen_cycles, seen_ids = set(), set()
    for block in blocks:
        cycles, ids = set(block.coordinates.cycles), set(block.coordinates.observation_ids)
        require(not (cycles & seen_cycles), "independent blocks must have disjoint cycles")
        require(not (ids & seen_ids), "observation IDs must be globally unique")
        seen_cycles.update(cycles)
        seen_ids.update(ids)
    return blocks


def gaussian_error_objective(log_scales, blocks, *, prior_medians, prior_log_sd):
    """Normalized Gaussian NLL plus all six quadratic log-SD penalties.

    As in the incumbent, fixed Gaussian penalty normalizer is omitted. No
    natural-SD Jacobian is added. This evaluates supplied points only: no
    optimizer, posterior scale mixing, cutoff, admission or outcome loading.
    """
    logs = real_array(log_scales, (6,), "log scales")
    medians = real_array(prior_medians, (6,), "prior medians")
    sd = scalar(prior_log_sd, "prior log SD")
    require((medians > 0).all() and sd > 0, "positive log-scale regularizer required")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        variances = np.exp(2 * logs)
    require(np.isfinite(variances).all() and (variances > 0).all(), "finite strictly positive variances required")
    value, gradient = 0.0, np.zeros(6)
    for block in validated_blocks(blocks):
        kernels = block.coordinates.kernels()
        covariance = np.einsum("k,kij->ij", variances, kernels)
        require(np.isfinite(covariance).all(), "finite covariance required")
        root = cho_factor(covariance, lower=True)
        alpha = cho_solve(root, block.residual)
        value += np.log(np.diag(root[0])).sum() + 0.5 * np.dot(block.residual, alpha)
        value += 0.5 * len(alpha) * np.log(2 * np.pi)
        inverse = cho_solve(root, np.eye(len(alpha)))
        derivative = inverse - np.outer(alpha, alpha)
        gradient += variances * np.einsum("ij,kij->k", derivative, kernels)
    standardized = (logs - np.log(medians)) / sd
    value += 0.5 * np.dot(standardized, standardized)
    gradient += standardized / sd
    require(np.isfinite(value) and np.isfinite(gradient).all(), "finite supplied-point objective/gradient required")
    return float(value), gradient


def kernel_geometry(coordinates, scales):
    """Supplied-design diagnostics, not fitted-parameter correlations."""
    covariance = error_covariance(coordinates, scales)
    kernels = coordinates.kernels().reshape(6, -1)
    brownian, slope = kernels[4:6]
    denom = np.linalg.norm(brownian) * np.linalg.norm(slope)
    return dict(
        kernel_design_rank=int(np.linalg.matrix_rank(kernels)),
        brownian_slope_frobenius_cosine=None if denom == 0 else float(brownian @ slope / denom),
        covariance_min_eigenvalue=float(np.linalg.eigvalsh(covariance).min()),
        covariance_condition=float(np.linalg.cond(covariance)),
        interpretation="kernel geometry at supplied scales; not posterior parameter correlation",
    )


def synthetic_conditional_example(slope_sd):
    """Gaussian toy only: theta election logit + beta * days/30 + five-term error."""
    slope_sd = scalar(slope_sd, "synthetic slope SD")
    require(slope_sd > 0, "positive synthetic slope SD required")
    coordinates = PollCoordinatesV1(
        ("toy-old", "toy-middle", "toy-late"),
        ("toy-race",) * 3,
        ("toy-cycle",) * 3,
        ("toy-house",) * 3,
        [60.0, 30.0, 7.0],
    )
    scales = dict(zip(COMPONENTS, [0.12, 0.03, 0.04, 0.03, 0.04, slope_sd], strict=True))
    x = coordinates.days_to_election / 30
    design = np.column_stack((np.ones(3), x))
    prior = np.diag([0.35**2, slope_sd**2])
    # Slope variance is in the latent prior, so only the five incumbent terms
    # remain in conditional observation error. Do not count slope twice.
    residual_covariance = error_covariance(coordinates, {**scales, "persistent_linear": 0.0})
    total = residual_covariance + design @ prior @ design.T
    gain = np.linalg.solve(total, design @ prior).T
    posterior_covariance = prior - gain @ design @ prior
    y = np.array([-0.30, -0.15, -0.04])
    mean = gain @ y
    return dict(
        schema="ClearlySyntheticGaussianConditionalExampleV1",
        synthetic=True,
        source_data_used=False,
        actual_fundamentals_bank_used=False,
        poll_logits=y.tolist(),
        days_before_election=coordinates.days_to_election.tolist(),
        theta_prior_sd=0.35,
        slope_prior_sd=slope_sd,
        slope_units="logit per 30 days before election",
        posterior_mean_theta=float(mean[0]),
        posterior_mean_slope=float(mean[1]),
        mirrored_posterior_mean=(gain @ (-y)).tolist(),
        posterior_covariance=posterior_covariance.tolist(),
        flat_data_posterior_mean=(gain @ np.zeros(3)).tolist(),
        interpretation="negative slope means support rises toward election; mirrored data reverses all means",
    )
