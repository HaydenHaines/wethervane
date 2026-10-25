"""Replaceable, earlier-only aggregate Senate fundamentals (not population_cut_v1).

The four refreshed arms share the v5 cycle Normal / race Student-t5 law.
This module builds PyMC models; it neither fits nor selects an empirical arm.
"""

from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
from scipy.special import expit, logit
from scipy.stats import halfnorm, norm, t

from src.forecasting.baseline_v5 import predictors_v5

IMPLEMENTATION_NAMES = {
    "senate_history_refreshed_v1": ("office", "signed_incumbency"),
    "senate_presidential_slope_v1": ("office", "signed_incumbency", "presidential_slope"),
    "senate_relative_lean_v1": ("office", "signed_incumbency", "state_slope", "national_carryover"),
    "senate_relative_context_v1": (
        "office",
        "signed_incumbency",
        "state_slope",
        "national_carryover",
        "president_party",
        "midterm_interaction",
    ),
}


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def coefficient_names_v1(implementation_id):
    _require(implementation_id in IMPLEMENTATION_NAMES, "Unknown fundamentals implementation")
    return IMPLEMENTATION_NAMES[implementation_id]


@dataclass(frozen=True)
class FundamentalsPriorsV1:
    office_sd: float = 0.5
    incumbency_sd: float = 0.5
    presidential_slope_mean: float = 1.0
    presidential_slope_sd: float = 0.35
    national_carryover_sd: float = 0.5
    president_party_sd: float = 0.10
    midterm_interaction_sd: float = 0.15
    cycle_scale_prior_sd: float = 0.2
    race_scale_prior_sd: float = 0.5
    race_df: float = 5.0

    def __post_init__(self):
        _require(
            all(np.isfinite(v) and (v > 0 or n == "presidential_slope_mean") for n, v in vars(self).items()),
            "Finite positive prior scales required",
        )
        _require(self.race_df == 5, "Principal fundamentals law fixes race_df=5")

    def coefficient_parameters(self, implementation_id):
        parameters = {
            "office": (0, self.office_sd),
            "signed_incumbency": (0, self.incumbency_sd),
            "presidential_slope": (self.presidential_slope_mean, self.presidential_slope_sd),
            "state_slope": (self.presidential_slope_mean, self.presidential_slope_sd),
            "national_carryover": (0, self.national_carryover_sd),
            "president_party": (0, self.president_party_sd),
            "midterm_interaction": (0, self.midterm_interaction_sd),
        }
        return np.array([parameters[n] for n in coefficient_names_v1(implementation_id)]).T


def _dates_and_cycles(inputs):
    import pandas as pd

    for name in ("cycle", "prior_presidential_cycle"):
        values = inputs[name].to_numpy(float)
        _require(np.isfinite(values).all() and np.equal(values, np.floor(values)).all(), "Integer cycles required")
    expected_previous = ((inputs.cycle.to_numpy(int) - 1) // 4) * 4
    _require(
        np.array_equal(inputs.prior_presidential_cycle.to_numpy(int), expected_previous),
        "Most recent strictly earlier presidential election required",
    )
    # A partial date contract is rejected rather than silently discarding dates.
    date_names = ("election_date", "forecast_cutoff", "prior_presidential_election_date")
    if any(n in inputs for n in date_names):
        _require(all(n in inputs for n in date_names), "Complete election/cutoff/prior presidential dates required")
        dates = {n: pd.to_datetime(inputs[n], errors="raise", utc=True) for n in date_names}
        _require(all(v.notna().all() for v in dates.values()), "Nonmissing feature dates required")
        _require(
            (dates["prior_presidential_election_date"] < dates["forecast_cutoff"]).all()
            and (dates["forecast_cutoff"] <= dates["election_date"]).all(),
            "Strict prior presidential dates required",
        )
        _require(
            (dates["election_date"].dt.year.to_numpy() == inputs.cycle.to_numpy()).all()
            and (
                dates["prior_presidential_election_date"].dt.year.to_numpy()
                == inputs.prior_presidential_cycle.to_numpy()
            ).all(),
            "Feature date/cycle mismatch",
        )


def _national(inputs):
    name = "prior_national_presidential_dem_share"
    _require(name in inputs, "National prior presidential share required")
    p = inputs[name].to_numpy(float)
    _require(np.isfinite(p).all() and ((p > 0) & (p < 1)).all(), "Interior national presidential share required")
    # One presidential election has one national total, even across Senate cycles.
    for _, rows in inputs.groupby("prior_presidential_cycle"):
        _require(rows[name].nunique() == 1, "Inconsistent national presidential share")
    return logit(p)


def _context(inputs):
    _require(
        {"president_party_asof", "midterm"} <= set(inputs), "Dated presidential-party and midterm context required"
    )
    p = inputs.president_party_asof.to_numpy(float)
    m = inputs.midterm.to_numpy(float)
    _require(
        np.isin(p, [-1, 1]).all() and np.isin(m, [0, 1]).all(), "Verified signed party and binary midterm required"
    )
    for _, rows in inputs.groupby("cycle"):
        _require(
            rows.president_party_asof.nunique() == rows.midterm.nunique() == 1, "Context must be shared within cycle"
        )
    return p, m


def predictors_v1(inputs, implementation_id):
    """Return offset, named design; feature columns are selected by name, never order."""
    names = coefficient_names_v1(implementation_id)
    x, old_design = predictors_v5(inputs)
    _dates_and_cycles(inputs)
    values = {"office": old_design[:, 0], "signed_incumbency": old_design[:, 1]}
    offset = np.zeros(len(inputs))
    if implementation_id == "senate_history_refreshed_v1":
        offset = x
    elif implementation_id == "senate_presidential_slope_v1":
        values["presidential_slope"] = x
    else:
        national = _national(inputs)
        values.update(state_slope=x - national, national_carryover=national)
        if implementation_id == "senate_relative_context_v1":
            party, midterm = _context(inputs)
            values.update(president_party=party, midterm_interaction=party * midterm)
    return offset, np.column_stack([values[n] for n in names]), names


def prepare_training_v1(inputs, outcomes, cutoff_cycle, implementation_id):
    """Filter earlier eligible inputs and outcome rows before parsing labels/features."""
    _require(
        type(cutoff_cycle) is int
        and {"cycle", "eligible"} <= set(inputs)
        and {"race_id", "cycle", "actual_dem_share"} <= set(outcomes),
        "Dated eligible inputs and outcomes required",
    )
    earlier = inputs.loc[inputs.cycle.lt(cutoff_cycle) & inputs.eligible.eq(True)].copy()
    labels = outcomes.loc[outcomes.cycle.lt(cutoff_cycle), ["race_id", "cycle", "actual_dem_share"]].copy()
    _require(not labels.race_id.duplicated().any(), "Duplicate earlier outcome IDs")
    _require("actual_dem_share" not in earlier, "Outcome labels must remain separate from features")
    source = earlier.merge(labels, on=["race_id", "cycle"], how="left", validate="one_to_one").sort_values(
        ["cycle", "race_id"]
    )
    offset, design, names = predictors_v1(source, implementation_id)
    actual = source.actual_dem_share.to_numpy(float)
    _require(
        np.isfinite(actual).all() and ((actual > 0) & (actual < 1)).all(),
        "Every earlier eligible race needs an interior outcome",
    )
    cycles, index = np.unique(source.cycle.to_numpy(int), return_inverse=True)
    return dict(
        cutoff_cycle=cutoff_cycle,
        implementation_id=implementation_id,
        coefficient_names=names,
        race_ids=source.race_id.astype(str).tolist(),
        cycles=cycles,
        cycle_index=index,
        offset=offset,
        design=design,
        y=logit(actual) - offset,
    )


def log_density_v1(training, beta, cycle_z, tau_cycle, tau_race, priors=None):
    """Natural positive scales, noncentered cycle z; includes all normalization terms."""
    p = priors or FundamentalsPriorsV1()
    mu, sd = p.coefficient_parameters(training["implementation_id"])
    _require(
        tuple(training["coefficient_names"]) == coefficient_names_v1(training["implementation_id"]),
        "Coefficient names differ",
    )
    beta, z = np.asarray(beta, float), np.asarray(cycle_z, float)
    _require(beta.shape == mu.shape and z.shape == (len(training["cycles"]),), "Incorrect parameter dimensions")
    if not np.isfinite(np.r_[beta, z, tau_cycle, tau_race]).all() or tau_cycle <= 0 or tau_race <= 0:
        return -np.inf
    location = training["design"] @ beta + tau_cycle * z[training["cycle_index"]]
    return float(
        norm.logpdf(beta, loc=mu, scale=sd).sum()
        + norm.logpdf(z).sum()
        + halfnorm.logpdf(tau_cycle, scale=p.cycle_scale_prior_sd)
        + halfnorm.logpdf(tau_race, scale=p.race_scale_prior_sd)
        + t.logpdf(training["y"], p.race_df, loc=location, scale=tau_race).sum()
    )


def build_pymc_model_v1(training, priors=None):
    import pymc as pm

    p = priors or FundamentalsPriorsV1()
    names = coefficient_names_v1(training["implementation_id"])
    _require(tuple(training["coefficient_names"]) == names, "Coefficient names differ")
    mu, sd = p.coefficient_parameters(training["implementation_id"])
    with pm.Model(
        coords={"coefficient": list(names), "source_cycle": training["cycles"], "source_race": training["race_ids"]}
    ) as model:
        beta = pm.Normal("beta", mu=mu, sigma=sd, dims="coefficient")
        tc = pm.HalfNormal("tau_cycle", sigma=p.cycle_scale_prior_sd)
        tr = pm.HalfNormal("tau_race", sigma=p.race_scale_prior_sd)
        z = pm.Normal("cycle_z", mu=0, sigma=1, dims="source_cycle")
        cycle = pm.Deterministic("cycle_effect", tc * z, dims="source_cycle")
        pm.StudentT(
            "historical_delta",
            nu=p.race_df,
            mu=pm.math.dot(training["design"], beta) + cycle[training["cycle_index"]],
            sigma=tr,
            observed=training["y"],
            dims="source_race",
        )
    return model


@dataclass(frozen=True)
class FundamentalsPosteriorV1:
    samples: dict
    cutoff_cycle: int
    training_cycles: tuple
    implementation_id: str
    coefficient_names: tuple
    priors: FundamentalsPriorsV1 = field(default_factory=FundamentalsPriorsV1)
    source_identity: str = ""

    def __post_init__(self):
        names = coefficient_names_v1(self.implementation_id)
        _require(tuple(self.coefficient_names) == names, "Named posterior coefficients differ")
        _require(
            type(self.cutoff_cycle) is int
            and bool(self.training_cycles)
            and tuple(sorted(set(self.training_cycles))) == tuple(self.training_cycles)
            and all(type(c) in (int, np.int64, np.int32) and c < self.cutoff_cycle for c in self.training_cycles),
            "Unique sorted strict earlier training cycles and integer cutoff required",
        )
        _require(
            isinstance(self.priors, FundamentalsPriorsV1)
            and isinstance(self.source_identity, str)
            and bool(self.source_identity.strip()),
            "Bound priors and source identity required",
        )
        object.__setattr__(self, "coefficient_names", tuple(self.coefficient_names))
        object.__setattr__(self, "training_cycles", tuple(self.training_cycles))
        _require({"beta", "tau_cycle", "tau_race"} <= set(self.samples), "Posterior coefficients and scales required")
        samples = {n: np.asarray(v).copy() for n, v in self.samples.items()}
        beta = samples["beta"]
        _require(
            beta.ndim in (2, 3) and beta.shape[-1] == len(names) and beta.size > 0,
            "Aligned named posterior coefficients required",
        )
        shape = beta.shape[:-1]
        for n in ("tau_cycle", "tau_race"):
            _require(samples[n].shape == shape and (samples[n] > 0).all(), "Aligned positive posterior scales required")
        _require(
            ("posterior_chain" in samples) == ("posterior_draw" in samples),
            "Paired posterior chain/draw identity required",
        )
        if "posterior_chain" in samples:
            for name in ("posterior_chain", "posterior_draw"):
                a = samples[name]
                _require(
                    a.shape == shape and (a >= 0).all() and np.equal(a, np.floor(a)).all(),
                    "Aligned integer posterior chain/draw IDs required",
                )
            pairs = zip(samples["posterior_chain"].ravel(), samples["posterior_draw"].ravel())
            _require(len(set(pairs)) == int(np.prod(shape)), "Duplicate posterior chain/draw identity")
        for a in samples.values():
            _require(np.isfinite(a).all(), "Finite posterior required")
            a.setflags(write=False)
        object.__setattr__(self, "samples", MappingProxyType(samples))


def to_empirical_basis_v1(inputs, posterior):
    """Contract a single target to (1, relative state lean, incumbency), retaining full rows."""
    from src.forecasting.polling_basis_v1 import EmpiricalBasisFundamentals

    _require(isinstance(posterior, FundamentalsPosteriorV1), "Metadata-bound posterior required")
    _require(
        inputs.cycle.nunique() == 1 and (inputs.cycle >= posterior.cutoff_cycle).all(),
        "One target cycle at/after fitted cutoff required",
    )
    offset, design, names = predictors_v1(inputs, posterior.implementation_id)
    national = _national(inputs)
    _require(np.ptp(national) == 0, "One target national presidential environment required")
    x, old = predictors_v5(inputs)
    beta_raw = posterior.samples["beta"]
    coefficients = beta_raw.reshape(-1, len(names))
    coeff = {n: coefficients[:, j] for j, n in enumerate(names)}
    national_mean = np.zeros(len(coefficients))
    if posterior.implementation_id == "senate_history_refreshed_v1":
        slope = np.ones(len(coefficients))
        national_mean += national[0]
    elif posterior.implementation_id == "senate_presidential_slope_v1":
        slope = coeff["presidential_slope"]
        national_mean += slope * national[0]
    else:
        slope = coeff["state_slope"]
        national_mean += coeff["national_carryover"] * national[0]
        if posterior.implementation_id == "senate_relative_context_v1":
            party, midterm = _context(inputs)
            national_mean += coeff["president_party"] * party[0] + coeff["midterm_interaction"] * party[0] * midterm[0]
    basis = np.column_stack([np.ones(len(inputs)), x - national, old[:, 1]])
    effective = np.column_stack([coeff["office"] + national_mean, slope, coeff["signed_incumbency"]])
    _require(
        np.allclose(effective @ basis.T, offset + coefficients @ design.T, atol=1e-12, rtol=1e-12),
        "Target basis contraction failed",
    )
    if "posterior_chain" in posterior.samples:
        chain, draw = posterior.samples["posterior_chain"], posterior.samples["posterior_draw"]
    elif beta_raw.ndim == 3:
        chain, draw = np.indices(beta_raw.shape[:2])
    else:
        chain, draw = np.zeros(len(coefficients), dtype=int), np.arange(len(coefficients))
    return EmpiricalBasisFundamentals(
        offset=np.zeros(len(inputs)),
        design=basis,
        beta=effective,
        tau_cycle=posterior.samples["tau_cycle"].reshape(-1),
        tau_race=posterior.samples["tau_race"].reshape(-1),
        race_ids=tuple(inputs.race_id.astype(str)),
        source_identity=posterior.source_identity,
        basis_names=("intercept", "relative_state_lean", "signed_incumbency"),
        coefficient_names=names,
        coefficients=coefficients,
        posterior_chain=chain.reshape(-1),
        posterior_draw=draw.reshape(-1),
        explained_national_mean=national_mean,
        race_df=posterior.priors.race_df,
        metadata={
            "implementation_id": posterior.implementation_id,
            "target_id": "senate_aggregate_logit_empirical_mixture_v1",
            "output_contract_id": "empirical_linear_fundamentals_joint_v1",
            "cutoff_cycle": posterior.cutoff_cycle,
            "target_cycle": int(inputs.cycle.iloc[0]),
            "training_cycles": tuple(posterior.training_cycles),
            "priors": vars(posterior.priors).copy(),
        },
    )


def predictive_draws_v1(inputs, posterior, *, cutoff_cycle=None, priors=None, seed=202610202):
    _require(isinstance(posterior, FundamentalsPosteriorV1), "Metadata-bound posterior required")
    _require(
        cutoff_cycle is None or cutoff_cycle == posterior.cutoff_cycle,
        "Requested cutoff differs from fitted posterior cutoff",
    )
    _require(priors is None or priors == posterior.priors, "Requested priors differ from fitted posterior priors")
    prior = to_empirical_basis_v1(inputs, posterior)
    rng = np.random.default_rng(seed)
    z = rng.normal(size=len(prior.beta))
    cycle = z * prior.tau_cycle
    discrepancy = rng.standard_t(prior.race_df, size=(len(prior.beta), len(prior.race_ids))) * prior.tau_race[:, None]
    location = prior.offset + prior.beta @ prior.design.T
    logits = location + cycle[:, None] + discrepancy
    return dict(
        shares=expit(logits),
        logits=logits,
        baseline_location=location,
        shared_cycle=cycle,
        shared_cycle_z=z,
        explained_national_mean=prior.explained_national_mean,
        race_discrepancy=discrepancy,
        race_ids=prior.race_ids,
        coefficients=prior.coefficients,
        coefficient_names=prior.coefficient_names,
        posterior_chain=prior.posterior_chain,
        posterior_draw=prior.posterior_draw,
        source_identity=prior.source_identity,
        implementation_id=posterior.implementation_id,
        target_id=prior.metadata["target_id"],
        output_contract_id=prior.metadata["output_contract_id"],
        metadata=prior.metadata_copy(),
    )
