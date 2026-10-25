"""Exact finite empirical fundamentals mixture with a bounded generic linear basis.

This version leaves polling_v1 and its two-column historical contract unchanged.
Only row locations are generalized: cycle Normal and Student-t mixture laws and
Gaussian measurement ownership are the original aggregate comparison's laws.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import expit

from src.forecasting.polling_v1 import (
    GaussianPollLikelihood,
    _categorical,
    compress_likelihood,
    conditional_state,
    require,
)
from src.forecasting.response_v5_reuse import correlated_mcse

MAX_BASIS_DIMENSION = 32


def _freeze_metadata(value):
    if isinstance(value, np.generic):
        return _freeze_metadata(value.item())
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_metadata(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata(item) for item in value)
    require(isinstance(value, (str, int, float, bool, type(None))), "Metadata must contain immutable scalar values")
    require(not isinstance(value, float) or np.isfinite(value), "Finite metadata required")
    return value


def _copy_metadata(value):
    if isinstance(value, Mapping):
        return {key: _copy_metadata(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_copy_metadata(item) for item in value)
    return value


def _names(values, length, label):
    names = tuple(values)
    require(
        len(names) == length
        and len(set(names)) == length
        and all(isinstance(n, str) and n.strip() == n and bool(n) for n in names),
        f"Unique named {label} required",
    )
    return names


@dataclass(frozen=True)
class EmpiricalBasisFundamentals:
    offset: np.ndarray
    design: np.ndarray
    beta: np.ndarray
    tau_cycle: np.ndarray
    tau_race: np.ndarray
    race_ids: tuple[str, ...]
    source_identity: str
    basis_names: tuple[str, ...]
    coefficient_names: tuple[str, ...] = ()
    coefficients: np.ndarray | None = None
    posterior_chain: np.ndarray | None = None
    posterior_draw: np.ndarray | None = None
    explained_national_mean: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)
    race_df: float = 5.0
    _product_indices: tuple = field(init=False, repr=False, compare=False)
    _beta_products: np.ndarray = field(init=False, repr=False, compare=False)

    def metadata_copy(self):
        """Return serializable metadata without exposing the bound immutable object."""
        return _copy_metadata(self.metadata)

    def __post_init__(self):
        require(np.ndim(self.design) == 2 and np.ndim(self.beta) == 2, "Matrix basis and coefficients required")
        n, q = np.shape(self.design)
        k = len(self.beta)
        require(n > 0 and k > 0 and 1 <= q <= MAX_BASIS_DIMENSION, "Nonempty bounded basis required")
        object.__setattr__(self, "race_ids", _names(self.race_ids, n, "race IDs"))
        object.__setattr__(self, "basis_names", _names(self.basis_names, q, "basis columns"))
        require(
            isinstance(self.source_identity, str) and bool(self.source_identity.strip()),
            "Historical posterior source identity required",
        )
        require(np.isfinite(self.race_df) and self.race_df > 2, "Finite race df > 2 required")
        for name, shape in (
            ("offset", (n,)),
            ("design", (n, q)),
            ("beta", (k, q)),
            ("tau_cycle", (k,)),
            ("tau_race", (k,)),
        ):
            a = np.array(getattr(self, name), dtype=float, copy=True)
            require(a.shape == shape and np.isfinite(a).all(), f"Invalid {name}")
            a.setflags(write=False)
            object.__setattr__(self, name, a)
        require((self.tau_cycle > 0).all() and (self.tau_race > 0).all(), "Positive scales required")
        full = self.beta if self.coefficients is None else np.asarray(self.coefficients)
        require(full.ndim == 2 and len(full) == k, "Aligned full coefficient rows required")
        names = self.basis_names if self.coefficients is None and not self.coefficient_names else self.coefficient_names
        object.__setattr__(self, "coefficient_names", _names(names, full.shape[1], "full coefficients"))
        full = np.array(full, dtype=float, copy=True)
        require(np.isfinite(full).all(), "Finite full coefficients required")
        full.setflags(write=False)
        object.__setattr__(self, "coefficients", full)
        require(
            (self.posterior_chain is None) == (self.posterior_draw is None),
            "Paired original chain/draw identity required",
        )
        for name, default in (("posterior_chain", np.zeros(k, dtype=int)), ("posterior_draw", np.arange(k))):
            raw = default if getattr(self, name) is None else np.asarray(getattr(self, name))
            require(
                raw.shape == (k,)
                and np.isfinite(raw).all()
                and (raw >= 0).all()
                and np.equal(raw, np.floor(raw)).all(),
                "Nonnegative integer chain/draw IDs required",
            )
            a = raw.astype(int).copy()
            a.setflags(write=False)
            object.__setattr__(self, name, a)
        require(len(set(zip(self.posterior_chain, self.posterior_draw))) == k, "Duplicate posterior row identities")
        national = (
            np.zeros(k)
            if self.explained_national_mean is None
            else np.array(self.explained_national_mean, dtype=float, copy=True)
        )
        require(national.shape == (k,) and np.isfinite(national).all(), "Aligned explained national mean required")
        national.setflags(write=False)
        object.__setattr__(self, "explained_national_mean", national)
        object.__setattr__(self, "metadata", _freeze_metadata(dict(self.metadata)))
        j, ell = np.triu_indices(q)
        products = self.beta[:, j] * self.beta[:, ell]
        products.setflags(write=False)
        object.__setattr__(self, "_product_indices", (j, ell))
        object.__setattr__(self, "_beta_products", products)


def conditional_row_log_weights(prior, observed, theta, cycle, mixture_precision):
    """Exact O(m q² + K q²) update, without a K by m residual allocation."""
    observed, theta, w = np.asarray(observed), np.asarray(theta, float), np.asarray(mixture_precision, float)
    require(
        observed.ndim == 1
        and np.issubdtype(observed.dtype, np.integer)
        and len(set(observed.tolist())) == len(observed)
        and (observed >= 0).all()
        and (observed < len(prior.race_ids)).all(),
        "Unique valid observed race indices required",
    )
    require(
        theta.shape == w.shape == observed.shape
        and np.isfinite(theta).all()
        and np.isfinite(w).all()
        and (w > 0).all()
        and np.isfinite(cycle),
        "Valid conditional state and precisions required",
    )
    basis = prior.design[observed]
    u = theta - prior.offset[observed] - cycle
    square = np.dot(w, u * u)
    linear = basis.T @ (w * u)
    gram = basis.T @ (w[:, None] * basis)
    j, ell = prior._product_indices
    symmetric_gram = gram[j, ell] * np.where(j == ell, 1, 2)
    quadratic = square - 2 * (prior.beta @ linear) + prior._beta_products @ symmetric_gram
    # Bound cancellation using all summands, not just u², for nonzero basis means.
    magnitude = (
        abs(square) + 2 * (np.abs(prior.beta) @ np.abs(linear)) + np.abs(prior._beta_products) @ np.abs(symmetric_gram)
    )
    tolerance = 64 * np.finfo(float).eps * max(1, prior.design.shape[1] ** 2) * np.maximum(1, magnitude)
    require((quadratic >= -tolerance).all(), "Unstable conditional squared norm")
    quadratic = np.maximum(quadratic, 0)
    return (
        -np.log(prior.tau_cycle)
        - len(observed) * np.log(prior.tau_race)
        - 0.5 * (cycle / prior.tau_cycle) ** 2
        - 0.5 * quadratic / prior.tau_race**2
    )


def sample_poll_posterior(prior, polls, *, chains=4, warmup=1000, draws=4000, seed=202610201):
    """Blocked exact empirical-row Gibbs; unpolled Student-t states integrated out."""
    require(isinstance(prior, EmpiricalBasisFundamentals), "Generic empirical basis prior required")
    require(isinstance(polls, GaussianPollLikelihood), "Declared Gaussian poll likelihood required")
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
        if not len(observed):
            rows = rng.integers(k, size=draws)
            c = rng.normal(size=draws) * prior.tau_cycle[rows]
            logits[chain] = (
                prior.offset
                + prior.beta[rows] @ prior.design.T
                + c[:, None]
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
                    prior.offset[missing]
                    + prior.design[missing] @ prior.beta[row]
                    + cycle
                    + prior.tau_race[row] * rng.standard_t(prior.race_df, size=len(missing))
                )
                cycles[chain, j], indices[chain, j] = cycle, row
    return dict(
        logits=logits,
        shares=expit(logits),
        first_wins=logits > 0,
        shared_cycle=cycles,
        shared_cycle_z=cycles / prior.tau_cycle[indices],
        posterior_index=indices,
        beta=prior.beta[indices],
        basis_names=prior.basis_names,
        coefficients=prior.coefficients[indices],
        coefficient_names=prior.coefficient_names,
        posterior_chain=prior.posterior_chain[indices],
        posterior_draw=prior.posterior_draw[indices],
        explained_national_mean=prior.explained_national_mean[indices],
        tau_cycle=prior.tau_cycle[indices],
        tau_race=prior.tau_race[indices],
        race_ids=prior.race_ids,
        observed_race_indices=observed,
        source_identity=prior.source_identity,
        metadata=prior.metadata_copy(),
        implementation_id=prior.metadata.get("implementation_id"),
        target_id=prior.metadata.get("target_id", "generic_empirical_linear_mixture_v1"),
        output_contract_id=prior.metadata.get("output_contract_id", "empirical_linear_fundamentals_joint_v1"),
        sampler="empirical_basis_student_t_blocked_gibbs_v1" if len(observed) else "direct_empirical_basis_prior_v1",
    )


def posterior_diagnostics(result, *, batch_length=50):
    """Diagnose all original coefficients, scales and the residual cycle factor."""
    shares = result["shares"]
    p = np.concatenate(
        [
            shares,
            result["coefficients"],
            result["tau_cycle"][..., None],
            result["tau_race"][..., None],
            result["shared_cycle"][..., None],
            result["shared_cycle_z"][..., None],
        ],
        axis=2,
    )
    names = (
        [f"share:{r}" for r in result["race_ids"]]
        + [f"coefficient:{c}" for c in result["coefficient_names"]]
        + ["tau_cycle", "tau_race", "cycle", "cycle_z"]
    )
    diag = correlated_mcse(p, batch_length=batch_length)
    win = correlated_mcse(result["first_wins"].astype(float), batch_length=batch_length)
    variance = np.var(p.reshape(-1, p.shape[-1]), axis=0, ddof=1)
    ess = np.divide(variance, diag["mcse"] ** 2, out=np.full_like(variance, np.inf), where=diag["mcse"] > 0)
    return dict(
        method="split_rank_folded_rhat_and_multiscale_batch_chain_mcse",
        coordinates=names,
        rhat=diag["rank_rhat"].tolist(),
        mean_mcse=diag["mcse"].tolist(),
        mean_ess_estimate=[float(x) if np.isfinite(x) else None for x in ess],
        win_probability_mcse=win["mcse"].tolist(),
        max_rhat=float(diag["rank_rhat"].max()),
        max_share_mean_mcse=float(diag["mcse"][: shares.shape[-1]].max()),
        max_win_probability_mcse=float(win["mcse"].max()),
        min_mean_ess_estimate_nonconstant=float(ess[np.isfinite(ess)].min()) if np.isfinite(ess).any() else None,
        historical_prior_mcmc_error="Inherited; extra forecast draws do not increase historical effective sample size",
    )
