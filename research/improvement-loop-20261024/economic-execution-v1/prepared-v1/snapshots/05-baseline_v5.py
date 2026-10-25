"""Independent v5 hierarchical prior-only Senate baseline, on aggregate logits."""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.special import expit, logit
from scipy.stats import halfnorm, norm, t


@dataclass(frozen=True)
class BaselinePriorsV5:
    beta_sd: float = 0.5
    cycle_scale_prior_sd: float = 0.2
    race_scale_prior_sd: float = 0.5
    race_df: float = 5.0

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in asdict(self).values()) or self.race_df <= 2:
            raise ValueError("Positive finite prior scales and race_df>2 required")


@dataclass(frozen=True)
class BaselinePosteriorV5:
    samples: dict
    cutoff_cycle: int
    training_cycles: tuple
    priors: BaselinePriorsV5

    def __post_init__(self):
        if not self.training_cycles or any(c >= self.cutoff_cycle for c in self.training_cycles):
            raise ValueError("Posterior must bind strictly earlier training cycles")
        if not isinstance(self.priors, BaselinePriorsV5):
            raise ValueError("Posterior must bind its fitted prior specification")


def predictors_v5(inputs):
    """Validate metadata and return fixed historical presidential logit and design."""
    required = {
        "race_id",
        "cycle",
        "prior_presidential_cycle",
        "prior_presidential_dem_share",
        "dem_incumbent",
        "rep_incumbent",
    }
    if required - set(inputs) or inputs.empty or inputs.race_id.duplicated().any():
        raise ValueError("Unique nonempty race metadata required")
    if (
        inputs.cycle.isna().any()
        or inputs.prior_presidential_cycle.isna().any()
        or (inputs.prior_presidential_cycle >= inputs.cycle).any()
    ):
        raise ValueError("Presidential input must strictly precede target cycle")
    p = inputs.prior_presidential_dem_share.to_numpy(float)
    inc = inputs[["dem_incumbent", "rep_incumbent"]]
    if not np.isfinite(p).all() or ((p <= 0) | (p >= 1)).any():
        raise ValueError("Finite interior prior share required")
    if inc.isna().any().any() or not inc.isin([False, True, 0, 1]).all().all():
        raise ValueError("Verified incumbency required")
    a = inc.to_numpy(float)
    if (a.sum(axis=1) > 1).any():
        raise ValueError("Two incumbent nominees unsupported")
    return logit(p), np.column_stack([np.ones(len(inputs)), a[:, 0] - a[:, 1]])


def prepare_training_v5(inputs, outcomes, cutoff_cycle):
    """Filter before validation/join: target/future labels cannot affect training."""
    if "eligible" not in inputs or "cycle" not in outcomes:
        raise ValueError("Eligibility and dated outcomes required")
    earlier = inputs.loc[inputs.cycle.lt(cutoff_cycle) & inputs.eligible.eq(True)].copy()
    labels = outcomes.loc[outcomes.cycle.lt(cutoff_cycle), ["race_id", "cycle", "actual_dem_share"]].copy()
    if labels.race_id.duplicated().any():
        raise ValueError("Duplicate earlier outcomes")
    source = earlier.merge(labels, on=["race_id", "cycle"], how="left", validate="one_to_one").sort_values(
        ["cycle", "race_id"]
    )
    offset, design = predictors_v5(source)
    actual = source.actual_dem_share.to_numpy(float)
    if not np.isfinite(actual).all() or ((actual <= 0) | (actual >= 1)).any():
        raise ValueError("Every earlier eligible race needs an interior outcome")
    cycles, index = np.unique(source.cycle.to_numpy(int), return_inverse=True)
    if len(cycles) == 0 or max(cycles) >= cutoff_cycle:
        raise ValueError("Strict earlier training cycles required")
    return {
        "cutoff_cycle": int(cutoff_cycle),
        "race_ids": source.race_id.astype(str).tolist(),
        "cycles": cycles,
        "cycle_index": index,
        "offset": offset,
        "design": design,
        "y": logit(actual) - offset,
    }


def log_density_v5(training, beta, cycle_z, tau_cycle, tau_race, priors=None):
    """Density in natural positive scales and noncentered standard-normal cycles."""
    p = priors or BaselinePriorsV5()
    beta, z = np.asarray(beta, float), np.asarray(cycle_z, float)
    if beta.shape != (2,) or z.shape != (len(training["cycles"]),):
        raise ValueError("Incorrect parameter dimensions")
    if not np.isfinite(np.r_[beta, z, tau_cycle, tau_race]).all() or tau_cycle <= 0 or tau_race <= 0:
        return -np.inf
    location = training["design"] @ beta + tau_cycle * z[training["cycle_index"]]
    return float(
        norm.logpdf(beta, scale=p.beta_sd).sum()
        + norm.logpdf(z).sum()
        + halfnorm.logpdf(tau_cycle, scale=p.cycle_scale_prior_sd)
        + halfnorm.logpdf(tau_race, scale=p.race_scale_prior_sd)
        + t.logpdf(training["y"], df=p.race_df, loc=location, scale=tau_race).sum()
    )


def build_pymc_model_v5(training, priors=None):
    """Return model reusable by a future joint poll model; imports backend lazily."""
    import pymc as pm

    p = priors or BaselinePriorsV5()
    with pm.Model(
        coords={
            "coefficient": ["office", "signed_incumbency"],
            "source_cycle": training["cycles"],
            "source_race": training["race_ids"],
        }
    ) as model:
        beta = pm.Normal("beta", mu=0, sigma=p.beta_sd, dims="coefficient")
        tau_cycle = pm.HalfNormal("tau_cycle", sigma=p.cycle_scale_prior_sd)
        tau_race = pm.HalfNormal("tau_race", sigma=p.race_scale_prior_sd)
        z = pm.Normal("cycle_z", mu=0, sigma=1, dims="source_cycle")
        cycle = pm.Deterministic("cycle_effect", tau_cycle * z, dims="source_cycle")
        pm.StudentT(
            "historical_delta",
            nu=p.race_df,
            mu=pm.math.dot(training["design"], beta) + cycle[training["cycle_index"]],
            sigma=tau_race,
            observed=training["y"],
            dims="source_race",
        )
    return model


def predictive_draws_v5(inputs, posterior, *, cutoff_cycle=None, priors=None, seed=202610082):
    """One shared new-cycle draw per posterior draw; independent Student-t races.

    Requires one target cycle. Returned aggregate-logit components allow a future
    donor model to calibrate the exact aggregate rather than shift person logits.
    """
    if not isinstance(posterior, BaselinePosteriorV5):
        raise ValueError("Prediction requires a metadata-bound BaselinePosteriorV5")
    if cutoff_cycle is not None and cutoff_cycle != posterior.cutoff_cycle:
        raise ValueError("Requested cutoff differs from fitted posterior cutoff")
    if priors is not None and priors != posterior.priors:
        raise ValueError("Requested priors differ from fitted posterior priors")
    cutoff_cycle = posterior.cutoff_cycle
    p = posterior.priors
    posterior = posterior.samples
    if inputs.cycle.nunique() != 1 or (inputs.cycle < cutoff_cycle).any():
        raise ValueError("One target cycle at/after training cutoff required")
    offset, design = predictors_v5(inputs)
    beta = np.asarray(posterior["beta"], float)
    tc, tr = np.asarray(posterior["tau_cycle"], float), np.asarray(posterior["tau_race"], float)
    if beta.ndim != 2 or beta.shape[1] != 2 or tc.shape != (len(beta),) or tr.shape != tc.shape or len(beta) == 0:
        raise ValueError("Aligned posterior draws required")
    if not np.isfinite(np.r_[beta.ravel(), tc, tr]).all() or (tc <= 0).any() or (tr <= 0).any():
        raise ValueError("Finite posterior and positive scales required")
    rng = np.random.default_rng(seed)
    shared = rng.normal(size=len(beta)) * tc
    race = rng.standard_t(p.race_df, size=(len(beta), len(inputs))) * tr[:, None]
    location = offset[None, :] + beta @ design.T
    logits = location + shared[:, None] + race
    return {
        "shares": expit(logits),
        "logits": logits,
        "baseline_location": location,
        "shared_cycle": shared,
        "race_discrepancy": race,
        "race_ids": inputs.race_id.astype(str).tolist(),
    }
