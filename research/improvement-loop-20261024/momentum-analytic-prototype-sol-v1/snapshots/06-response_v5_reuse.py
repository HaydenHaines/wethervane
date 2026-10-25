"""Research prototype: raw joint-density reweighting of correlated HMC draws.

No production qualification is implied. Each source refit has a conditional
posterior; raw weights normalize WITHIN that refit. Equal outer source mass is
retained. Reused draws remain correlated and are never resampled into fake iid
posterior draws. All densities include historical AND current poll likelihoods.
"""

import hashlib
import json
from dataclasses import dataclass

import numpy as np
from scipy.special import expit, logsumexp
from scipy.stats import norm, rankdata


@dataclass(frozen=True)
class FlatCoordinates:
    names: tuple
    shapes: tuple

    def __post_init__(self):
        if not self.names or len(self.names) != len(self.shapes) or len(set(self.names)) != len(self.names):
            raise ValueError("Unique named coordinate blocks required")
        if any(any(not isinstance(n, int) or n < 1 for n in shape) for shape in self.shapes):
            raise ValueError("Positive block dimensions required; scalar shape is ()")

    @property
    def dimension(self):
        return int(sum(np.prod(shape, dtype=int) for shape in self.shapes))

    def unpack(self, point):
        point = np.asarray(point, float)
        if point.shape != (self.dimension,) or not np.isfinite(point).all():
            raise ValueError("Finite matching flat point required")
        result, first = {}, 0
        for name, shape in zip(self.names, self.shapes, strict=True):
            size = int(np.prod(shape, dtype=int))
            result[name] = point[first : first + size].reshape(shape)
            first += size
        return result


class PyMCJointEvaluatorV5:
    """Exact full transformed joint density plus target operator for a built graph.

    data_identity must identify unchanged historical labels/cutoff, baseline
    predictors, target roster/dates, poll observations and measurement covariance.
    Source identities are allowed to differ. Construction requires matched mode.
    """

    def __init__(self, model, *, data_identity):
        if model.v5_spec["mode"] != "matched_historical_community" or not data_identity:
            raise ValueError("Matched historical graph and explicit common-data identity required")
        self.model = model
        self.data_identity = data_identity
        self.source_identity = model.v5_spec["source_coordinate_signature"]
        initial = model.initial_point()
        self.coordinates = FlatCoordinates(
            tuple(v.name for v in model.value_vars), tuple(initial[v.name].shape for v in model.value_vars)
        )
        self._logp = model.compile_logp(jacobian=True)
        self.omitted = tuple(model.v5_spec.get("marginalized_target_race_indices", ()))
        self.sampled = tuple(model.v5_spec.get("sampled_target_race_indices", ()))
        self.auxiliary_dimension = len(self.omitted)
        self.auxiliary_df = model.v5_spec.get("race_df")
        if model.v5_spec.get("marginalize_unpolled", False):
            outputs = model.replace_rvs_by_values([model["target_mean_logit"], model["target_race"], model["tau_race"]])
            self._components = model.compile_fn(
                outputs, inputs=model.value_vars, on_unused_input="ignore", point_fn=True
            )
            self._predict = None
        else:
            election = model.replace_rvs_by_values([model["election_share"]])[0]
            self._predict = model.compile_fn(election, inputs=model.value_vars, on_unused_input="ignore", point_fn=True)

    def log_density(self, point):
        return float(self._logp(self.coordinates.unpack(point)))

    def predictions(self, point, auxiliary=None):
        if self._predict is not None:
            return np.asarray(self._predict(self.coordinates.unpack(point)), float)
        logits, race, scale = self._components(self.coordinates.unpack(point))
        logits = np.asarray(logits).copy()
        logits[list(self.sampled)] += race
        if self.omitted:
            auxiliary = np.asarray(auxiliary, float)
            if auxiliary.shape != (len(self.omitted),) or not np.isfinite(auxiliary).all():
                raise ValueError("Independent standardized Student-t predictive auxiliaries required")
            logits[list(self.omitted)] += scale * auxiliary
        return expit(logits)

    def diagnostic_predictions(self, point):
        """Retained base/scale diagnostics cannot be hidden by fresh IID race noise."""
        if self._predict is not None:
            return self.predictions(point)
        logits, _, scale = self._components(self.coordinates.unpack(point))
        return np.r_[expit(logits), scale / (1 + scale)]

    def pack_natural_posterior(self, posterior):
        """Map runner posterior__ arrays to EXACT model value-variable coordinates.

        Only identity and positive-log transforms used by v5 are supported;
        unknown transforms fail instead of silently changing the density measure.
        """
        pieces, shape = {}, None
        for rv in self.model.free_RVs:
            values = np.asarray(posterior[rv.name], float)
            if values.ndim < 2 or not np.isfinite(values).all():
                raise ValueError("Finite chain-by-draw natural posterior arrays required")
            if shape is not None and values.shape[:2] != shape:
                raise ValueError("Aligned chain identities and draw order required")
            shape = values.shape[:2]
            transform = self.model.rvs_to_transforms[rv]
            name = self.model.rvs_to_values[rv].name
            if transform is not None:
                if type(transform).__name__ != "LogTransform" or (values <= 0).any():
                    raise ValueError("Unsupported transform or invalid positive scale")
                values = np.log(values)
            pieces[name] = values
        arrays = []
        for name, block_shape in zip(self.coordinates.names, self.coordinates.shapes, strict=True):
            if pieces[name].shape != (*shape, *block_shape):
                raise ValueError("Natural posterior block shape differs from model")
            arrays.append(pieces[name].reshape(*shape, -1))
        return np.concatenate(arrays, axis=2)


class AffineTransport:
    """Fixed invertible point map; its complete Jacobian enters every raw weight.

    Orthogonal community rotations have |det|=1. General basis/normalizer maps
    are permitted only through this explicit map and full joint-density ratio;
    their priors generally do NOT cancel. No point-dependent map is supported.
    """

    def __init__(self, matrix, offset=None):
        matrix = np.asarray(matrix, float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.isfinite(matrix).all():
            raise ValueError("Finite square transport required")
        sign, logdet = np.linalg.slogdet(matrix)
        if sign == 0 or not np.isfinite(logdet):
            raise ValueError("Transport must be invertible")
        self.matrix, self.log_abs_det = matrix.copy(), float(logdet)
        self.offset = np.zeros(len(matrix)) if offset is None else np.asarray(offset, float).copy()
        if self.offset.shape != (len(matrix),) or not np.isfinite(self.offset).all():
            raise ValueError("Finite matching transport offset required")

    def __call__(self, points):
        return np.asarray(points) @ self.matrix.T + self.offset


def community_rotation(coordinates, rotation):
    """Apply ONE orthogonal basis map to every historical, endpoint and dated z.

    This maps standardized community coordinates, not aggregate race predictions.
    The rotation must be chosen using source/operator geometry, not outcomes.
    """
    rotation = np.asarray(rotation, float)
    if (
        rotation.ndim != 2
        or rotation.shape[0] != rotation.shape[1]
        or not np.isfinite(rotation).all()
        or not np.allclose(rotation.T @ rotation, np.eye(len(rotation)), atol=1e-10, rtol=0)
    ):
        raise ValueError("An orthogonal community rotation is required")
    matrix, start = np.eye(coordinates.dimension), 0
    names = {"historical_community_z", "community_endpoint_z", "community_innovation_z"}
    if not names.intersection(coordinates.names):
        raise ValueError("No community coordinates to align")
    for name, shape in zip(coordinates.names, coordinates.shapes, strict=True):
        size = int(np.prod(shape, dtype=int))
        if name in names:
            if not shape or shape[-1] != len(rotation):
                raise ValueError("Community rank changed; full refit required")
            matrix[start : start + size, start : start + size] = np.kron(np.eye(size // len(rotation)), rotation)
        start += size
    return AffineTransport(matrix)


def align_community_loadings(reference_loadings, target_loadings, row_weights):
    """Outcome-free weighted orthogonal Procrustes alignment in common row order.

    Supply prior-linearized FULL historical/target aggregate-logit sensitivities,
    divided by each source fit's own common normalizer. The result Q minimizes
    ||B_target Q-B_reference||_W. Differing normalizers are not silently treated
    as a pure rotation; remaining mismatch is reported and density ratios remain
    mandatory. This function does not select among maps using posterior weights.
    """
    a, b, weights = (np.asarray(value, float) for value in (reference_loadings, target_loadings, row_weights))
    if (
        a.ndim != 2
        or a.shape != b.shape
        or not a.shape[1]
        or weights.shape != (len(a),)
        or not np.isfinite(np.r_[a.ravel(), b.ravel(), weights]).all()
        or (weights < 0).any()
        or weights.sum() <= 0
    ):
        raise ValueError("Aligned finite loadings and nonnegative supported row weights required")
    weights = weights / weights.sum()
    u, singular, vt = np.linalg.svd(b.T @ (weights[:, None] * a))
    rotation = u @ vt
    mismatch = b @ rotation - a
    return rotation, {
        "weighted_squared_mismatch": float(np.sum(weights[:, None] * mismatch**2)),
        "cross_singular_values": singular.tolist(),
        "method": "One declared source-geometry Procrustes map; full target/prior/Jacobian ratio still required",
    }


def _rank_rhat(values):
    """Maximum split rank-normalized/folded Rhat for each influence coordinate."""
    chains, draws, dimensions = values.shape
    half = draws // 2
    split = np.concatenate([values[:, :half], values[:, -half:]], axis=0)
    result = []
    for j in range(dimensions):
        raw = split[:, :, j]
        if np.ptp(raw) == 0:
            result.append(1.0)
            continue
        rhats = []
        for x in (raw, np.abs(raw - np.median(raw))):
            if np.ptp(x) == 0:
                # Folding a balanced two-level coordinate can remove all
                # variation even when the original coordinate varies.
                rhats.append(1.0)
                continue
            if np.all(np.ptp(x, axis=1) == 0):
                # Different constant chains have zero within-chain variance;
                # avoid roundoff in variance calculations hiding that failure.
                rhats.append(np.inf)
                continue
            ranks = rankdata(x.ravel()).reshape(x.shape)
            transformed = norm.ppf((ranks - 0.375) / (x.size + 0.25))
            within = np.var(transformed, axis=1, ddof=1).mean()
            between = half * np.var(transformed.mean(axis=1), ddof=1)
            rhats.append(np.sqrt((half - 1) / half + between / (half * within)) if within > 0 else np.inf)
        result.append(max(rhats))
    return np.asarray(result)


def correlated_mcse(influence, *, batch_length=50):
    """Conservative finite-sample batch/chain checks, NOT a convergence proof.

    Use the maximum of batch-mean variance at L and 2L and between-independent-
    chain mean variance. No chain order is shuffled and no thinning is performed.
    Requires at least eight coarse batches per chain; strong longer dependence
    can still defeat finite-batch diagnostics and requires independent HMC checks.
    """
    values = np.asarray(influence, float)
    if (
        values.ndim != 3
        or values.shape[0] < 4
        or not np.isfinite(values).all()
        or not isinstance(batch_length, int)
        or batch_length < 2
        or values.shape[1] < 16 * batch_length
        or values.shape[1] % (2 * batch_length)
    ):
        raise ValueError("Four aligned chains and at least eight complete coarse batches required")
    c, n, k = values.shape
    variances = [np.var(values.mean(axis=1), axis=0, ddof=1) / c]
    for length in [batch_length, 2 * batch_length]:
        batch = values.reshape(c, n // length, length, k).mean(axis=2).reshape(-1, k)
        variances.append(np.var(batch, axis=0, ddof=1) / len(batch))
    return {
        "mcse": np.sqrt(np.max(variances, axis=0)),
        "rank_rhat": _rank_rhat(values),
        "variance_estimates": np.asarray(variances),
        "batch_lengths": [batch_length, 2 * batch_length],
    }


@dataclass(frozen=True)
class ReuseThresholds:
    # Prototype thresholds; require actual independent references before adoption.
    max_pareto_k: float = 0.5
    min_raw_weight_ess: float = 1000
    min_adjusted_weight_ess: float = 400
    min_mean_ess: float = 400
    max_mean_mcse: float = 0.0005
    max_relative_variance_mcse: float = 0.05
    max_cdf_mcse: float = 0.01
    max_influence_rhat: float = 1.01
    batch_length: int = 50

    def __post_init__(self):
        numbers = [
            self.max_pareto_k,
            self.min_raw_weight_ess,
            self.min_adjusted_weight_ess,
            self.min_mean_ess,
            self.max_mean_mcse,
            self.max_relative_variance_mcse,
            self.max_cdf_mcse,
            self.max_influence_rhat,
        ]
        if not np.isfinite(numbers).all() or min(numbers) <= 0 or self.max_influence_rhat < 1:
            raise ValueError("Finite positive diagnostic thresholds required")
        if not isinstance(self.batch_length, int) or self.batch_length < 2:
            raise ValueError("Positive integer batch length required")


def arviz_pareto_k(log_weights, reff):
    """Diagnostic only: discard the PSIS-smoothed weights, retaining raw ratios."""
    import arviz as az

    _, k = az.psislw(np.asarray(log_weights, float), reff=reff)
    return float(k)


def diagnose_raw_weights(log_weights, predictions, *, thresholds=None, pareto_k=None):
    """Return exact normalized weights and chain-aware influence diagnostics.

    pareto_k callback receives raw flattened log weights and estimated relative
    weight ESS. Smoothed PSIS weights MUST NOT replace these returned raw weights.
    A caller can use arviz.psislw(..., reff=reff)[1] only as that diagnostic.
    """
    settings = thresholds or ReuseThresholds()
    lw, pred = np.asarray(log_weights, float), np.asarray(predictions, float)
    if (
        lw.ndim != 2
        or pred.ndim != 3
        or pred.shape[:2] != lw.shape
        or not pred.shape[2]
        or not np.isfinite(lw).all()
        or not np.isfinite(pred).all()
        or ((pred < 0) | (pred > 1)).any()
    ):
        raise ValueError("Finite chain-aligned log weights and bounded race predictions required")
    weights = np.exp(lw - logsumexp(lw))
    total = lw.size
    relative = weights * total
    mean = np.einsum("cd,cdr->r", weights, pred)
    influence = relative[:, :, None] * (pred - mean)
    mean_info = correlated_mcse(influence, batch_length=settings.batch_length)
    weight_info = correlated_mcse(relative[:, :, None] - 1, batch_length=settings.batch_length)
    var_w = float(np.var(relative, ddof=1))
    inflation = max(1.0, total * float(weight_info["mcse"][0]) ** 2 / var_w) if var_w > 1e-20 else 1.0
    kish = float(1 / np.sum(weights**2))
    adjusted = kish / inflation
    if np.ptp(lw) == 0:
        k, k_pass = None, True
    elif pareto_k is None:
        k, k_pass = None, False
    else:
        k = float(pareto_k(lw.ravel(), 1 / inflation))
        k_pass = bool(np.isfinite(k) and k <= settings.max_pareto_k)
    variance = np.einsum("cd,cdr->r", weights, (pred - mean) ** 2)
    variance_influence = relative[:, :, None] * ((pred - mean) ** 2 - variance)
    variance_info = correlated_mcse(variance_influence, batch_length=settings.batch_length)
    relative_variance_mcse = np.divide(variance_info["mcse"], variance, out=np.zeros_like(variance), where=variance > 0)
    mean_ess = np.divide(
        variance, mean_info["mcse"] ** 2, out=np.full_like(variance, float(total)), where=mean_info["mcse"] > 0
    )
    mean_ess = np.minimum(mean_ess, total)
    # CDF influence at each raw-weighted 5th/95th empirical quantile. This
    # diagnoses tail probability error; it is NOT a quantile-value MCSE.
    cdf = []
    for race in range(pred.shape[2]):
        order = np.argsort(pred[:, :, race], axis=None, kind="stable")
        flat = pred[:, :, race].ravel()
        csum = np.cumsum(weights.ravel()[order])
        for probability in [0.05, 0.95]:
            index = min(np.searchsorted(csum, probability), len(order) - 1)
            event = (pred[:, :, race] <= flat[order[index]]).astype(float)
            center = float(np.sum(weights * event))
            cdf.append(relative * (event - center))
    tail_info = correlated_mcse(np.stack(cdf, axis=2), batch_length=settings.batch_length)
    max_rhat = float(max(np.max(info["rank_rhat"]) for info in [mean_info, weight_info, tail_info, variance_info]))
    gates = {
        "pareto_k": k_pass,
        "raw_weight_ess": kish >= settings.min_raw_weight_ess,
        "autocorrelation_adjusted_weight_ess": adjusted >= settings.min_adjusted_weight_ess,
        "mean_ess": bool(np.all(mean_ess >= settings.min_mean_ess)),
        "mean_mcse": bool(np.all(mean_info["mcse"] <= settings.max_mean_mcse)),
        "variance_mcse": bool(np.all(relative_variance_mcse <= settings.max_relative_variance_mcse)),
        "cdf_mcse": bool(np.all(tail_info["mcse"] <= settings.max_cdf_mcse)),
        "influence_rhat": max_rhat <= settings.max_influence_rhat,
    }
    return {
        "weights": weights,
        "mean": mean,
        "mean_influence": influence,
        "passed": bool(all(gates.values())),
        "diagnostics": {
            "gates": gates,
            "pareto_k": k,
            "raw_weight_ess_iid_upper_bound": kish,
            "weight_autocorrelation_inflation": inflation,
            "adjusted_weight_ess": adjusted,
            "mean_ess": mean_ess.tolist(),
            "mean_mcse": mean_info["mcse"].tolist(),
            "relative_variance_mcse": relative_variance_mcse.tolist(),
            "cdf_mcse": tail_info["mcse"].tolist(),
            "max_influence_rhat": max_rhat,
            "batch_lengths": mean_info["batch_lengths"],
            "chain_mass": weights.sum(axis=1).tolist(),
            "interpretation": (
                "Correlated raw self-normalized importance estimates; finite-sample biased, "
                "consistent under overlap and ergodicity; not iid draws or proven convergence"
            ),
            "required_on_failure": "Independent full conditional HMC refit; retain this failed attempt",
        },
    }


def reuse_joint_posterior(
    points,
    reference,
    target,
    *,
    reference_diagnostics_passed,
    transport=None,
    thresholds=None,
    pareto_k=None,
    augmentation_seed=None,
):
    """Evaluate complete joint ratio at mapped points; never silently reuse rank changes.

    reference/target expose coordinates, data_identity, log_density, predictions.
    Densities are in the SAME named unconstrained coordinates, including their
    log-transform Jacobians. Source-specific normalizers are inside each graph.
    """
    if reference_diagnostics_passed is not True:
        raise ValueError("Passed authenticated reference HMC diagnostics required")
    if reference.coordinates != target.coordinates or reference.data_identity != target.data_identity:
        raise ValueError("Latent layout or common-data identity changed; full refit required")
    values = np.asarray(points, float)
    if values.ndim != 3 or values.shape[2] != reference.coordinates.dimension or not np.isfinite(values).all():
        raise ValueError("Finite original chain-by-draw-by-coordinate reference points required")
    if transport is None:
        transport = AffineTransport(np.eye(values.shape[2]))
    if transport.matrix.shape != (values.shape[2], values.shape[2]):
        raise ValueError("Transport dimension mismatch")
    mapped = transport(values)
    auxiliary_dimension = getattr(target, "auxiliary_dimension", 0)
    auxiliary = None
    if auxiliary_dimension:
        if not isinstance(augmentation_seed, int):
            raise ValueError("Explicit predictive augmentation seed required for marginalized races")
        auxiliary = np.random.default_rng(augmentation_seed).standard_t(
            target.auxiliary_df, size=(*values.shape[:2], auxiliary_dimension)
        )
    weights, predictions, retained = [], [], []
    for index, (original, changed) in enumerate(
        zip(values.reshape(-1, values.shape[2]), mapped.reshape(-1, values.shape[2]), strict=True)
    ):
        weights.append(target.log_density(changed) - reference.log_density(original) + transport.log_abs_det)
        if auxiliary is None:
            predictions.append(target.predictions(changed))
        else:
            predictions.append(target.predictions(changed, auxiliary.reshape(-1, auxiliary_dimension)[index]))
        if hasattr(target, "diagnostic_predictions"):
            retained.append(target.diagnostic_predictions(changed))
    lw = np.asarray(weights).reshape(values.shape[:2])
    pred = np.asarray(predictions).reshape(*values.shape[:2], -1)
    result = diagnose_raw_weights(lw, pred, thresholds=thresholds, pareto_k=pareto_k)
    if retained:
        retained = np.asarray(retained).reshape(*values.shape[:2], -1)
        retained_result = diagnose_raw_weights(lw, retained, thresholds=thresholds, pareto_k=pareto_k)
        result["diagnostics"]["retained_component_checks"] = retained_result["diagnostics"]
        result["passed"] = bool(result["passed"] and retained_result["passed"])
    digest = hashlib.sha256(np.ascontiguousarray(values, dtype="<f8").tobytes())
    digest.update(
        json.dumps(
            [
                values.shape,
                reference.coordinates.names,
                reference.coordinates.shapes,
                reference.data_identity,
                getattr(reference, "source_identity", None),
            ],
            sort_keys=True,
        ).encode()
    )
    result.update(
        log_weights=lw,
        predictions=pred,
        mapped_points=mapped,
        prototype_only=True,
        reference_chain_identity=digest.hexdigest(),
    )
    if auxiliary is not None:
        result.update(
            predictive_auxiliaries=auxiliary,
            augmentation_seed=augmentation_seed,
            augmentation_df=float(target.auxiliary_df),
        )
    return result


def pool_reused_sources(results, races, *, expected_refits=16, batch_length=50):
    """Equal modular source mass; retain covariance of MC error from shared chains.

    This prototype handles reuse of ONE common reference chain set only. Mixed
    independent-HMC fallbacks need their own MC-error contributions, not fake
    alignment with reused chains, and are intentionally outside this function.
    """
    from src.forecasting.response_mixture import combine_response_draws

    if set(results) != set(range(expected_refits)) or any(r["passed"] is not True for r in results.values()):
        raise ValueError("Every planned refit must pass or receive full HMC fallback; no silent drops")
    ordered = [results[i] for i in range(expected_refits)]
    shapes = {r["mean_influence"].shape for r in ordered}
    identities = {r["reference_chain_identity"] for r in ordered}
    if len(shapes) != 1 or len(identities) != 1:
        raise ValueError("One common reference chain order/shape required")
    mixture = combine_response_draws(
        [r["predictions"].reshape(-1, len(races)) for r in ordered],
        races,
        expected_refits=expected_refits,
        weights_by_refit=[r["weights"].ravel() for r in ordered],
    )
    influence = np.mean([r["mean_influence"] for r in ordered], axis=0)
    info = correlated_mcse(influence, batch_length=batch_length)
    mixture["shared_reference_mean_mcse"] = info["mcse"]
    mixture["metadata"]["mc_error"] = (
        "Mean influence averaged before batch analysis; shared-reference covariance retained"
    )
    return mixture
