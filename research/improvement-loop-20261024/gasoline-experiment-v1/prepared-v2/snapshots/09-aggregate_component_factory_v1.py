"""Small jointly fitted aggregate factor factory; not the population/count stack.

Providers contribute named terms and uniquely owned priors, never independently
fitted prediction columns. Existing v1 engines, catalogs and artifacts are untouched.
Artifacts establish byte integrity, not scientific authenticity of caller metadata.
"""

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import AwareDatetime, Field, model_validator
from scipy.special import logit
from scipy.stats import halfnorm, norm, t

from src.forecasting.baseline_v5 import predictors_v5
from src.forecasting.fundamentals_v1 import _context, _dates_and_cycles, _national
from src.forecasting.polling_basis_v1 import EmpiricalBasisFundamentals, sample_poll_posterior
from src.forecasting.polling_v1 import GaussianPollLikelihood
from src.model_stack.artifacts import ArtifactRef, ArtifactStore, byte_digest, canonical, digest
from src.model_stack.contracts import (
    Component,
    Digest,
    Factor,
    Graph,
    InferenceBlock,
    Latent,
    Name,
    Record,
    Role,
    unique,
)

TARGET = "senate_aggregate_logit_empirical_mixture_v1"
CANONICAL_ORDER = (
    "structure.office",
    "candidate.incumbency",
    "structure.state_slope",
    "environment.carryover",
    "environment.party",
    "environment.midterm",
)
LEGACY_NAMES = dict(
    zip(
        ("office", "signed_incumbency", "state_slope", "national_carryover", "president_party", "midterm_interaction"),
        CANONICAL_ORDER,
        strict=True,
    )
) | {"presidential_slope": "structure.state_slope"}
FORBIDDEN = {"actual_dem_share", "dem_votes", "rep_votes", "winner", "winning_party", "winner_bloc"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_hash():
    return byte_digest(Path(__file__).read_bytes())


def implementation_fingerprint():
    return digest(
        {
            name: byte_digest(Path(__file__).with_name(name).read_bytes())
            for name in (
                "aggregate_component_factory_v1.py",
                "fundamentals_v1.py",
                "baseline_v5.py",
                "polling_basis_v1.py",
                "polling_v1.py",
            )
        }
    )


def _array(value, dtype=float):
    a = np.asarray(value, dtype=dtype)
    require(np.isfinite(a).all(), "Finite array required")
    # Immutable bytes backing prevents writeability from being re-enabled.
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def frame_digest(frame):
    """Exact JSON float representations; callers choose row order before hashing."""
    rows = frame.reindex(sorted(frame.columns), axis=1).astype(object)
    rows = rows.where(pd.notna(rows), None)
    return digest(rows.to_dict("records"))


class SourcePin(Record):
    path: Name
    sha256: Digest

    def read(self, root):
        root = Path(root).resolve()
        path = (root / self.path).resolve()
        require(path.is_relative_to(root) and path.is_file(), "Source outside repository or missing")
        value = path.read_bytes()
        require(byte_digest(value) == self.sha256, "Pinned source changed")
        return value


class ParameterSpec(Record):
    parameter_id: Name
    owner: Name
    mean: float
    sd: float = Field(gt=0)

    @model_validator(mode="after")
    def finite(self):
        require(np.isfinite([self.mean, self.sd]).all(), "Finite Normal prior required")
        return self


class ParameterRef(Record):
    parameter_id: Name


class MeanTerm(Record):
    term_id: Name
    owner: Name
    feature_id: Name
    coefficient: float | ParameterRef

    @model_validator(mode="after")
    def fixed(self):
        require(not isinstance(self.coefficient, float) or np.isfinite(self.coefficient), "Finite fixed coefficient")
        require(self.feature_id not in FORBIDDEN, "Outcome cannot be a mean feature")
        return self


class ProviderSpec(Record):
    instance_id: Name
    provider_id: Name
    source_sha256: Digest
    role: Role
    kind: Literal["mean", "cycle", "race", "poll"]
    parameters: tuple[ParameterSpec, ...] = ()
    terms: tuple[MeanTerm, ...] = ()
    kernel: Literal["linear", "normal_shared", "student_t5", "identity", "fixed_gaussian"]
    scale_prior_sd: float | None = None
    covariate_evidence_ids: tuple[Name, ...] = ()

    @model_validator(mode="after")
    def ownership(self):
        unique([p.parameter_id for p in self.parameters], "provider parameters")
        unique([t.term_id for t in self.terms], "provider terms")
        unique(self.covariate_evidence_ids, "covariate evidence")
        require(all(p.owner == self.instance_id for p in self.parameters), "Parameter owner differs")
        require(all(t.owner == self.instance_id for t in self.terms), "Term owner differs")
        allowed = {
            "mean": {"linear"},
            "cycle": {"normal_shared"},
            "race": {"student_t5"},
            "poll": {"identity", "fixed_gaussian"},
        }
        require(self.kernel in allowed[self.kind], "Unsupported kernel; residual-off is not implemented")
        if self.kind in {"cycle", "race"}:
            require(
                self.scale_prior_sd is not None and np.isfinite(self.scale_prior_sd) and self.scale_prior_sd > 0,
                "Positive scale prior required",
            )
        else:
            require(self.scale_prior_sd is None, "Unexpected scale prior")
        if self.kind != "mean":
            require(not self.parameters and not self.terms, "Only mean providers own linear coefficients")
        require(
            self.role
            == {"cycle": "environment", "race": "fundamentals", "poll": "measurement"}.get(self.kind, self.role),
            "Provider role differs from kernel",
        )
        require(self.kind != "mean" or self.role in {"fundamentals", "environment"}, "Aggregate mean role required")
        return self


class Recipe(Record):
    target_id: Literal["senate_aggregate_logit_empirical_mixture_v1"] = TARGET
    denominator: Literal["D_plus_R"] = "D_plus_R"
    universe: Literal["ordinary_DR_contests"] = "ordinary_DR_contests"
    providers: tuple[ProviderSpec, ...]

    @model_validator(mode="after")
    def references(self):
        unique([p.instance_id for p in self.providers], "provider instances")
        for kind in ("cycle", "race", "poll"):
            require(sum(p.kind == kind for p in self.providers) == 1, "Exactly one " + kind + " provider required")
        params = [q for p in self.providers for q in p.parameters]
        terms = [q for p in self.providers for q in p.terms]
        unique([p.parameter_id for p in params], "parameter ownership/prior")
        unique([t.term_id for t in terms], "term IDs")
        ids = {p.parameter_id for p in params}
        refs = {t.coefficient.parameter_id for t in terms if isinstance(t.coefficient, ParameterRef)}
        require(ids == refs and bool(ids), "Every parameter needs exactly one owner and a used reference")
        require(len(ids) <= 32, "Existing polling basis supports at most 32 parameters")
        # Detect duplicate free columns structurally, even on a tiny fixture whose
        # other columns happen to be collinear. Full sample rank is reported later.
        signatures = [
            tuple(
                sorted(
                    t.feature_id
                    for t in terms
                    if isinstance(t.coefficient, ParameterRef) and t.coefficient.parameter_id == key
                )
            )
            for key in sorted(ids)
        ]
        unique(signatures, "structurally identical free columns")
        return self

    def payload(self, *, training=False):
        value = self.model_dump(mode="json")
        value["providers"] = sorted(
            (p for p in value["providers"] if not training or p["kind"] != "poll"),
            key=lambda p: p["instance_id"],
        )
        for provider in value["providers"]:
            provider["parameters"].sort(key=lambda p: p["parameter_id"])
            provider["terms"].sort(key=lambda t: t["term_id"])
            provider["covariate_evidence_ids"].sort()
        return value

    @property
    def recipe_hash(self):
        return digest(self.payload())

    @property
    def training_recipe_hash(self):
        return digest(self.payload(training=True))

    @property
    def parameters(self):
        params = {q.parameter_id: q for p in self.providers for q in p.parameters}
        order = [n for n in CANONICAL_ORDER if n in params] + sorted(set(params) - set(CANONICAL_ORDER))
        return tuple(params[n] for n in order)

    def provider(self, kind):
        return next(p for p in self.providers if p.kind == kind)


def standard_providers(
    *,
    state_slope="learned",
    candidate="signed_incumbency",
    carryover="learned",
    context="party_midterm",
    polling="identity",
):
    """Provider options compose without a whole-forecast implementation enum."""
    require(state_slope in {"fixed_one", "learned"}, "Unknown state slope")
    require(candidate in {"off", "signed_incumbency"}, "Unknown candidate adjustment")
    require(carryover in {"off", "fixed_one", "reference_state_slope", "learned"}, "Unknown carryover")
    require(context in {"off", "party", "party_midterm"}, "Unknown context")
    require(polling in {"identity", "fixed_gaussian"}, "Unknown polling mode")
    specifications = []

    def mean(owner, provider_id, role, parameters, terms):
        specifications.append(
            ProviderSpec(
                instance_id=owner,
                provider_id=provider_id,
                source_sha256=source_hash(),
                role=role,
                kind="mean",
                kernel="linear",
                parameters=tuple(ParameterSpec(parameter_id=n, owner=owner, mean=m, sd=s) for n, m, s in parameters),
                terms=tuple(
                    MeanTerm(
                        term_id=owner + "." + str(j),
                        owner=owner,
                        feature_id=f,
                        coefficient=ParameterRef(parameter_id=c) if isinstance(c, str) else c,
                    )
                    for j, (f, c) in enumerate(terms)
                ),
            )
        )

    mean(
        "structure",
        "aggregate_structure_v1",
        "fundamentals",
        [("structure.office", 0, 0.5)] + ([("structure.state_slope", 1, 0.35)] if state_slope == "learned" else []),
        [("one", "structure.office"), ("relative_lean", "structure.state_slope" if state_slope == "learned" else 1.0)],
    )
    enabled = candidate != "off"
    mean(
        "candidate",
        "aggregate_candidate_incumbency_v1",
        "fundamentals",
        [("candidate.incumbency", 0, 0.5)] if enabled else [],
        [("incumbency", "candidate.incumbency")] if enabled else [],
    )
    mean(
        "carryover",
        "aggregate_national_carryover_v1",
        "environment",
        [("environment.carryover", 0, 0.5)] if carryover == "learned" else [],
        []
        if carryover == "off"
        else [
            (
                "national",
                {
                    "fixed_one": 1.0,
                    "reference_state_slope": "structure.state_slope",
                    "learned": "environment.carryover",
                }[carryover],
            )
        ],
    )
    mean(
        "context",
        "aggregate_presidential_context_v1",
        "environment",
        ([] if context == "off" else [("environment.party", 0, 0.1)])
        + ([("environment.midterm", 0, 0.15)] if context == "party_midterm" else []),
        ([] if context == "off" else [("party", "environment.party")])
        + ([("party_midterm", "environment.midterm")] if context == "party_midterm" else []),
    )
    for owner, provider, role, kind, kernel, scale in [
        ("cycle", "aggregate_cycle_normal_v1", "environment", "cycle", "normal_shared", 0.2),
        ("race", "aggregate_race_student_t5_v1", "fundamentals", "race", "student_t5", 0.5),
        ("poll", "aggregate_poll_gaussian_v1", "measurement", "poll", polling, None),
    ]:
        specifications.append(
            ProviderSpec(
                instance_id=owner,
                provider_id=provider,
                source_sha256=source_hash(),
                role=role,
                kind=kind,
                kernel=kernel,
                scale_prior_sd=scale,
            )
        )
    return tuple(specifications)


def legacy_recipe(implementation_id, *, polling="identity"):
    mappings = {
        "senate_history_refreshed_v1": ("fixed_one", "fixed_one", "off"),
        "senate_presidential_slope_v1": ("learned", "reference_state_slope", "off"),
        "senate_relative_lean_v1": ("learned", "learned", "off"),
        "senate_relative_context_v1": ("learned", "learned", "party_midterm"),
    }
    require(implementation_id in mappings, "No exact existing-law mapping")
    b, n, c = mappings[implementation_id]
    return Recipe(providers=standard_providers(state_slope=b, carryover=n, context=c, polling=polling))


class Admission(Record):
    target_id: Literal["senate_aggregate_logit_empirical_mixture_v1"] = TARGET
    denominator: Literal["D_plus_R"] = "D_plus_R"
    universe: Literal["ordinary_DR_contests"] = "ordinary_DR_contests"
    cutoff_utc: AwareDatetime
    cutoff_cycle: int
    support_id: Name
    feature_source_sha256: Digest
    outcome_source_sha256: Digest | None = None
    receipt_sha256: Digest
    roster_qualification: Name
    extra_feature_ids: tuple[Name, ...] = ()
    start_cycle: int | None = None
    allowed_cycles: tuple[int, ...] = ()

    @model_validator(mode="after")
    def timing(self):
        require(self.cutoff_utc.year == self.cutoff_cycle, "Cutoff cycle and instant differ")
        unique(self.extra_feature_ids, "extra features")
        require(not FORBIDDEN.intersection(self.extra_feature_ids), "Outcome feature prohibited")
        require(self.start_cycle is None or self.start_cycle < self.cutoff_cycle, "Empty historical window")
        unique(self.allowed_cycles, "allowed cycles")
        require(
            all(
                c < self.cutoff_cycle and (self.start_cycle is None or c >= self.start_cycle)
                for c in self.allowed_cycles
            ),
            "Allowed cycles outside earlier window",
        )
        return self


def _features(inputs, admission):
    require(len(inputs) > 0 and not FORBIDDEN.intersection(inputs.columns), "Nonempty outcome-free features required")
    require(not inputs.race_id.duplicated().any(), "Unique race IDs required")
    require(inputs.eligible.eq(True).all(), "Only admitted ordinary contests allowed")
    if "special" in inputs:
        require(inputs.special.eq(False).all(), "Special contests require a different target")
    for column, expected in (
        ("first_actual_party", "D"),
        ("second_actual_party", "R"),
        ("share_interpretation", "democratic_two_party"),
    ):
        if column in inputs:
            require(inputs[column].dropna().eq(expected).all(), "Principal-pair proxy incompatible with ordinary D/R")
    _dates_and_cycles(inputs)
    if "forecast_cutoff" in inputs:
        require(
            (pd.to_datetime(inputs.forecast_cutoff, utc=True) <= admission.cutoff_utc).all(),
            "Row feature cutoff is later than the admitted information instant",
        )
    if "source_cutoff_utc" in inputs:
        dates = pd.to_datetime(inputs.source_cutoff_utc.dropna(), utc=True)
        require((dates <= admission.cutoff_utc).all(), "Feature source unavailable at exact cutoff")
    x, old = predictors_v5(inputs)
    n = _national(inputs)
    values = {"one": np.ones(len(inputs)), "relative_lean": x - n, "national": n, "incumbency": old[:, 1]}
    if {"president_party_asof", "midterm"} <= set(inputs):
        p, m = _context(inputs)
        values.update(party=p, party_midterm=p * m)
    for feature in admission.extra_feature_ids:
        require(feature not in values and feature in inputs, "Missing or colliding extra feature")
        values[feature] = inputs[feature].to_numpy(float)
    require(all(np.isfinite(v).all() for v in values.values()), "Finite admitted features required")
    return values


def _design(recipe, inputs, admission):
    values = _features(inputs, admission)
    names = tuple(p.parameter_id for p in recipe.parameters)
    offset, design = np.zeros(len(inputs)), np.zeros((len(inputs), len(names)))
    for provider in sorted(recipe.providers, key=lambda p: p.instance_id):
        for term in sorted(provider.terms, key=lambda t: t.term_id):
            require(term.feature_id in values, "Unadmitted feature: " + term.feature_id)
            v = values[term.feature_id]
            if provider.role == "environment":
                require(
                    all(np.ptp(v[inputs.cycle.to_numpy() == c]) == 0 for c in inputs.cycle.unique()),
                    "Environment mean must be national within cycle; use a versioned adapter for other loadings",
                )
            if isinstance(term.coefficient, ParameterRef):
                design[:, names.index(term.coefficient.parameter_id)] += v
            else:
                offset += term.coefficient * v
    return _array(offset), _array(design)


@dataclass(frozen=True)
class CompiledTraining:
    recipe: Recipe
    admission: Admission
    race_ids: tuple
    cycles: np.ndarray
    cycle_index: np.ndarray
    offset: np.ndarray
    design: np.ndarray
    response: np.ndarray
    feature_rows_sha256: str
    outcome_rows_sha256: str
    implementation_sha256: str = field(default_factory=implementation_fingerprint)

    def __post_init__(self):
        for name in ("cycles", "cycle_index", "offset", "design", "response"):
            object.__setattr__(
                self, name, _array(getattr(self, name), int if name in {"cycles", "cycle_index"} else float)
            )
        n = len(self.race_ids)
        require(n > 0 and len(set(self.race_ids)) == n, "Unique training rows required")
        require(self.design.shape == (n, len(self.recipe.parameters)), "Training design dimensions differ")
        require(self.offset.shape == self.response.shape == self.cycle_index.shape == (n,), "Training arrays differ")
        require(
            np.array_equal(self.cycles, np.unique(self.cycles))
            and len(self.cycles) > 0
            and (self.cycles < self.admission.cutoff_cycle).all(),
            "Strict earlier cycles required",
        )
        require(
            (self.cycle_index >= 0).all() and (self.cycle_index < len(self.cycles)).all(), "Cycle index outside support"
        )

    @property
    def parameter_names(self):
        return tuple(p.parameter_id for p in self.recipe.parameters)

    def payload(self):
        return {
            "schema": "aggregate-compiled-training-v1",
            "recipe": self.recipe.payload(training=True),
            "admission": self.admission.model_dump(mode="json"),
            "race_ids": self.race_ids,
            "parameter_names": self.parameter_names,
            "implementation_sha256": self.implementation_sha256,
            **{n: getattr(self, n).tolist() for n in ("cycles", "cycle_index", "offset", "design", "response")},
            "feature_rows_sha256": self.feature_rows_sha256,
            "outcome_rows_sha256": self.outcome_rows_sha256,
        }

    @property
    def law_hash(self):
        return digest(self.payload())

    def graph(self):
        components, latents, factors = [], [], []
        for p in self.recipe.providers:
            if p.kind == "poll":
                continue
            components.append(
                Component(
                    component_id=p.instance_id,
                    role=p.role,
                    statistical_model_version=p.provider_id,
                    implementation_version="v1",
                    implementation_sha256=p.source_sha256,
                    capabilities=("prepare", "factors"),
                    inputs=(),
                    outputs=(),
                    limitations=("Aggregate logit component; no population transport or candidate counts",),
                )
            )
            for parameter in p.parameters:
                latents.append(Latent(state_id=parameter.parameter_id, owner=p.instance_id))
                factors.append(
                    Factor(
                        factor_id="prior:" + parameter.parameter_id,
                        owner=p.instance_id,
                        kind="prior",
                        latent_ids=(parameter.parameter_id,),
                        observation_ids=(),
                        sample_dependence_ids=(),
                    )
                )
        cycle, race = self.recipe.provider("cycle"), self.recipe.provider("race")
        for name, owner in (
            ("tau_cycle", cycle.instance_id),
            ("cycle_z", cycle.instance_id),
            ("tau_race", race.instance_id),
        ):
            latents.append(Latent(state_id=name, owner=owner))
            factors.append(
                Factor(
                    factor_id="prior:" + name,
                    owner=owner,
                    kind="prior",
                    latent_ids=(name,),
                    observation_ids=(),
                    sample_dependence_ids=(),
                )
            )
        factors.append(
            Factor(
                factor_id="historical-logit",
                owner=race.instance_id,
                kind="likelihood",
                latent_ids=(*self.parameter_names, "tau_cycle", "cycle_z", "tau_race"),
                observation_ids=tuple("outcome:" + r for r in self.race_ids),
                sample_dependence_ids=tuple("cycle:" + str(c) for c in self.cycles),
            )
        )
        return Graph(
            components=tuple(components),
            fixtures=(),
            bindings=(),
            latents=tuple(latents),
            factors=tuple(factors),
            blocks=(
                InferenceBlock(
                    block_id="aggregate-joint",
                    components=tuple(c.component_id for c in components),
                    target="joint_posterior",
                    target_definition_sha256=self.law_hash,
                    feedback_policy="joint_within_block",
                    limitations=("Separate ordinary D/R aggregate target",),
                ),
            ),
        )


def compile_training(recipe, inputs, outcomes, admission):
    recipe, admission = Recipe.model_validate(recipe), Admission.model_validate(admission)
    require(admission.outcome_source_sha256 is not None, "Historical outcome source identity required")
    require({"cycle", "eligible", "race_id"} <= set(inputs), "Dated eligible features required")
    require({"cycle", "race_id", "actual_dem_share"} <= set(outcomes), "Separate dated outcomes required")
    x = inputs.loc[inputs.cycle.lt(admission.cutoff_cycle) & inputs.eligible.eq(True)].copy()
    if admission.start_cycle is not None:
        x = x.loc[x.cycle.ge(admission.start_cycle)]
    if admission.allowed_cycles:
        x = x.loc[x.cycle.isin(admission.allowed_cycles)]
    x = x.sort_values(["cycle", "race_id"]).reset_index(drop=True)
    y = outcomes.loc[outcomes.cycle.lt(admission.cutoff_cycle), ["race_id", "cycle", "actual_dem_share"]].copy()
    if admission.start_cycle is not None:
        y = y.loc[y.cycle.ge(admission.start_cycle)]
    if admission.allowed_cycles:
        y = y.loc[y.cycle.isin(admission.allowed_cycles)]
    require(not y.race_id.duplicated().any(), "Duplicate earlier outcome IDs")
    joined = x[["race_id", "cycle"]].merge(y, on=["race_id", "cycle"], how="left", validate="one_to_one")
    values = joined.actual_dem_share.to_numpy(float)
    require(np.isfinite(values).all() and ((values > 0) & (values < 1)).all(), "Interior matched outcomes required")
    require(
        not set("outcome:" + r for r in x.race_id).intersection(
            e for p in recipe.providers for e in p.covariate_evidence_ids
        ),
        "Outcome reused as covariate evidence",
    )
    offset, design = _design(recipe, x, admission)
    cycles, index = np.unique(x.cycle.to_numpy(int), return_inverse=True)
    result = CompiledTraining(
        recipe,
        admission,
        tuple(x.race_id.astype(str)),
        cycles,
        index,
        offset,
        design,
        logit(values),
        frame_digest(x),
        frame_digest(joined),
    )
    result.graph()  # Reuse the existing joint-block and evidence-ownership validator.
    return result


def log_density(training, beta, cycle_z, tau_cycle, tau_race):
    beta, z = np.asarray(beta, float), np.asarray(cycle_z, float)
    require(
        beta.shape == (len(training.parameter_names),) and z.shape == training.cycles.shape, "Parameter shape differs"
    )
    if not np.isfinite(np.r_[beta, z, tau_cycle, tau_race]).all() or tau_cycle <= 0 or tau_race <= 0:
        return -np.inf
    params = training.recipe.parameters
    location = training.offset + training.design @ beta + tau_cycle * z[training.cycle_index]
    return float(
        norm.logpdf(beta, [p.mean for p in params], [p.sd for p in params]).sum()
        + norm.logpdf(z).sum()
        + halfnorm.logpdf(tau_cycle, scale=training.recipe.provider("cycle").scale_prior_sd)
        + halfnorm.logpdf(tau_race, scale=training.recipe.provider("race").scale_prior_sd)
        + t.logpdf(training.response, 5, loc=location, scale=tau_race).sum()
    )


def gradient(training, beta, cycle_z, tau_cycle, tau_race):
    """Natural-coordinate gradient order: beta, tau_cycle, tau_race, cycle_z."""
    require(np.isfinite(log_density(training, beta, cycle_z, tau_cycle, tau_race)), "Interior finite state required")
    beta, z = np.asarray(beta), np.asarray(cycle_z)
    residual = training.response - training.offset - training.design @ beta - tau_cycle * z[training.cycle_index]
    score = 6 * residual / (5 * tau_race**2 + residual**2)
    pars = training.recipe.parameters
    db = training.design.T @ score - (beta - [p.mean for p in pars]) / np.square([p.sd for p in pars])
    dz = tau_cycle * np.bincount(training.cycle_index, weights=score, minlength=len(z)) - z
    dc = np.dot(score, z[training.cycle_index]) - tau_cycle / training.recipe.provider("cycle").scale_prior_sd ** 2
    dr = np.sum(-1 / tau_race + 6 * residual**2 / (tau_race * (5 * tau_race**2 + residual**2)))
    dr -= tau_race / training.recipe.provider("race").scale_prior_sd ** 2
    return np.r_[db, dc, dr, dz]


def build_pymc_model(training):
    import pymc as pm

    params = training.recipe.parameters
    with pm.Model(
        coords={
            "coefficient": list(training.parameter_names),
            "source_cycle": training.cycles,
            "source_race": list(training.race_ids),
        }
    ) as model:
        beta = pm.Normal("beta", mu=[p.mean for p in params], sigma=[p.sd for p in params], dims="coefficient")
        tc = pm.HalfNormal("tau_cycle", sigma=training.recipe.provider("cycle").scale_prior_sd)
        tr = pm.HalfNormal("tau_race", sigma=training.recipe.provider("race").scale_prior_sd)
        z = pm.Normal("cycle_z", mu=0, sigma=1, dims="source_cycle")
        cycle = pm.Deterministic("cycle_effect", tc * z, dims="source_cycle")
        pm.StudentT(
            "historical_logit",
            nu=5,
            mu=training.offset + pm.math.dot(training.design, beta) + cycle[training.cycle_index],
            sigma=tr,
            observed=training.response,
            dims="source_race",
        )
    return model


class SamplerPlan(Record):
    chains: int = Field(default=4, ge=2)
    draws: int = Field(default=4000, ge=20)
    tune: int = Field(default=1500, ge=0)
    cores: int = Field(default=2, ge=1)
    target_accept: float = Field(default=0.97, gt=0, lt=1)
    seed: int = Field(ge=0)
    max_treedepth: int = Field(default=10, ge=1)
    max_rank_rhat: float = Field(default=1.01, ge=1)
    min_bulk_ess: float = Field(default=400, gt=0)
    min_tail_ess: float = Field(default=400, gt=0)

    @model_validator(mode="after")
    def finite(self):
        require(
            np.isfinite([self.target_accept, self.max_rank_rhat, self.min_bulk_ess, self.min_tail_ess]).all(),
            "Finite sampler gates required",
        )
        return self


def _diagnostics(idata, plan):
    import arviz as az

    names = ["beta", "tau_cycle", "tau_race", "cycle_z"]
    values = []
    for fn, method in ((az.rhat, "rank"), (az.ess, "bulk"), (az.ess, "tail")):
        result = fn(idata, var_names=names, method=method)
        values.append(np.concatenate([np.asarray(v).ravel() for v in result.data_vars.values()]))
    finite = all(np.isfinite(a).all() for a in values)
    rhat, bulk, tail = (
        (float(values[0].max()), float(values[1].min()), float(values[2].min())) if finite else (None,) * 3
    )
    diverging = int(np.asarray(idata.sample_stats.diverging).sum())
    depth = int((np.asarray(idata.sample_stats.tree_depth) >= plan.max_treedepth).sum())
    return {
        "finite": finite,
        "max_rank_rhat": rhat,
        "min_bulk_ess": bulk,
        "min_tail_ess": tail,
        "divergences": diverging,
        "max_depth_hits": depth,
        "passed": bool(
            finite
            and rhat <= plan.max_rank_rhat
            and bulk >= plan.min_bulk_ess
            and tail >= plan.min_tail_ess
            and diverging == depth == 0
        ),
    }


def _npz(arrays):
    stream = BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def _posterior_arrays(arrays, training):
    required = {"beta", "tau_cycle", "tau_race", "cycle_z", "cycle_effect"}
    require(set(arrays) == required, "One complete joint parameter bank required; no separate component columns")
    result = {n: _array(arrays[n]) for n in sorted(required)}
    beta = result["beta"]
    require(
        beta.ndim == 3 and beta.shape[-1] == len(training.parameter_names) and beta.shape[0] >= 2 and beta.shape[1] > 0,
        "Named chain/draw/coefficient bank required",
    )
    shape = beta.shape[:2]
    for name in ("tau_cycle", "tau_race"):
        require(result[name].shape == shape and (result[name] > 0).all(), "Aligned positive scale rows required")
    for name in ("cycle_z", "cycle_effect"):
        require(result[name].shape == (*shape, len(training.cycles)), "Aligned training cycle rows required")
    require(
        np.allclose(result["cycle_effect"], result["tau_cycle"][..., None] * result["cycle_z"], atol=1e-12, rtol=1e-12),
        "Shared cycle state inconsistent with its same-row scale",
    )
    return MappingProxyType(result)


def _store_joint(store, training, arrays, diagnostics, provenance, extra_files=None):
    arrays = _posterior_arrays(arrays, training)
    files = {
        "compiled.json": canonical(training.payload()),
        "graph.json": canonical(training.graph().model_dump(mode="json")),
        "posterior.npz": _npz(arrays),
        "diagnostics.json": canonical(diagnostics),
        "provenance.json": canonical(provenance),
    }
    require(not set(files).intersection(extra_files or {}), "Reserved artifact file")
    files.update(extra_files or {})
    return store.put(
        "aggregate-factory-joint-posterior-v1",
        {
            "law_hash": training.law_hash,
            "parameter_names": training.parameter_names,
            "posterior_sha256": byte_digest(files["posterior.npz"]),
            "status": "passed" if diagnostics["passed"] else "diagnostic_failure",
        },
        files,
    )


def fit_compiled(training, sampler_plan, store):
    """Fit one complete joint block and save all attempts; never runs on import.

    Returns an ArtifactRef, including on numerical failure. Prediction loading
    requires a passing posterior artifact. Callers freeze the input plan first.
    """
    import traceback

    plan = SamplerPlan.model_validate(sampler_plan)
    require(isinstance(store, ArtifactStore), "Existing exclusive ArtifactStore required")
    require(training.implementation_sha256 == implementation_fingerprint(), "Compiled implementation changed")
    request = store.put(
        "aggregate-factory-fit-request-v1",
        {"law_hash": training.law_hash},
        {"compiled.json": canonical(training.payload()), "sampler-plan.json": canonical(plan.model_dump(mode="json"))},
    )
    try:
        import pymc as pm

        with build_pymc_model(training):
            data = pm.sample(
                draws=plan.draws,
                tune=plan.tune,
                chains=plan.chains,
                cores=plan.cores,
                target_accept=plan.target_accept,
                random_seed=plan.seed,
                progressbar=False,
                init="jitter+adapt_diag",
                jitter_max_retries=0,
                nuts={"max_treedepth": plan.max_treedepth},
                discard_tuned_samples=False,
                idata_kwargs={"save_warmup": True},
            )
        arrays = {
            n: np.asarray(data.posterior[n]) for n in ("beta", "tau_cycle", "tau_race", "cycle_z", "cycle_effect")
        }
        require(
            arrays["beta"].shape == (plan.chains, plan.draws, len(training.parameter_names)),
            "Sampler layout differs from frozen plan",
        )
        diagnostics = _diagnostics(data, plan)
        raw, layout = {}, {}
        for group in data.groups():
            dataset = getattr(data, group)
            layout[group] = {}
            for kind, values in (("variable", dataset.data_vars), ("coordinate", dataset.coords)):
                for name, value in values.items():
                    key = group + "__" + kind + "__" + name
                    a = np.asarray(value)
                    require(a.dtype.kind != "O" or kind == "coordinate", "Unsupported nonnumeric inference variable")
                    raw[key] = a.astype(str) if a.dtype.kind == "O" else a
                    layout[group][key] = list(value.dims)
        require(training.implementation_sha256 == implementation_fingerprint(), "Code changed during fit")
        return _store_joint(
            store,
            training,
            arrays,
            diagnostics,
            {
                "origin": "joint_fit",
                "sampler_plan": plan.model_dump(mode="json"),
                "pymc_version": pm.__version__,
                "fit_request": request.model_dump(mode="json"),
            },
            {"raw-inference.npz": _npz(raw), "raw-layout.json": canonical(layout)},
        )
    except Exception as exc:
        return store.put(
            "aggregate-factory-failed-attempt-v1",
            {"status": "failed_error", "law_hash": training.law_hash},
            {
                "compiled.json": canonical(training.payload()),
                "sampler-plan.json": canonical(plan.model_dump(mode="json")),
                "fit-request.json": canonical(request.model_dump(mode="json")),
                "failure.txt": traceback.format_exc().encode(),
                "error.json": canonical({"type": type(exc).__name__, "message": str(exc)}),
            },
        )


@dataclass(frozen=True, init=False)
class JointPosterior:
    """Load one hash-verified complete artifact. No column-wise construction API."""

    artifact: ArtifactRef
    training: CompiledTraining
    samples: MappingProxyType
    posterior_sha256: str

    def __init__(self, store, artifact, training):
        import json

        require(artifact.kind == "aggregate-factory-joint-posterior-v1", "Complete joint posterior artifact required")
        metadata, files = store.read(artifact)
        require(
            metadata["status"] == "passed" and json.loads(files["diagnostics.json"])["passed"] is True,
            "Passing numerical diagnostics required",
        )
        require(
            metadata["law_hash"] == training.law_hash and files["compiled.json"] == canonical(training.payload()),
            "Posterior belongs to a different joint law/data/cutoff",
        )
        require(metadata["parameter_names"] == list(training.parameter_names), "Coefficient order differs")
        require(metadata["posterior_sha256"] == byte_digest(files["posterior.npz"]), "Joint bank fingerprint differs")
        with np.load(BytesIO(files["posterior.npz"]), allow_pickle=False) as raw:
            arrays = _posterior_arrays({n: raw[n] for n in raw.files}, training)
        object.__setattr__(self, "artifact", artifact)
        object.__setattr__(self, "training", training)
        object.__setattr__(self, "samples", arrays)
        object.__setattr__(self, "posterior_sha256", metadata["posterior_sha256"])


def predictive_basis(recipe, inputs, admission, posterior):
    """Generic full basis, retaining all coefficients and original whole-row IDs."""
    recipe, admission = Recipe.model_validate(recipe), Admission.model_validate(admission)
    require(isinstance(posterior, JointPosterior), "Verified complete JointPosterior required")
    training = posterior.training
    require(training.implementation_sha256 == implementation_fingerprint(), "Compiled implementation changed")
    require(recipe.training_recipe_hash == training.recipe.training_recipe_hash, "Cannot splice different recipes")
    require(
        admission.cutoff_utc >= training.admission.cutoff_utc,
        "Target information instant precedes the fitted admission cutoff",
    )
    require(
        inputs.cycle.nunique() == 1
        and inputs.cycle.iloc[0] == admission.cutoff_cycle
        and admission.cutoff_cycle >= training.admission.cutoff_cycle,
        "One target at/after fitted cutoff required",
    )
    require(admission.outcome_source_sha256 is None, "Target prediction admission must not contain outcome source")
    offset, design = _design(recipe, inputs, admission)
    names = training.parameter_names
    full = posterior.samples["beta"].reshape(-1, len(names))
    chain, draw = np.indices(posterior.samples["beta"].shape[:2])
    values = _features(inputs, admission)
    national = np.zeros(len(full))
    for provider in recipe.providers:
        if provider.kind != "mean" or provider.role != "environment":
            continue
        for term in provider.terms:
            feature = values[term.feature_id]
            require(np.ptp(feature) == 0, "Environment mean must be national in this interface")
            coefficient = (
                full[:, names.index(term.coefficient.parameter_id)]
                if isinstance(term.coefficient, ParameterRef)
                else term.coefficient
            )
            national += coefficient * feature[0]
    return EmpiricalBasisFundamentals(
        offset=offset,
        design=design,
        beta=full,
        tau_cycle=posterior.samples["tau_cycle"].ravel(),
        tau_race=posterior.samples["tau_race"].ravel(),
        race_ids=tuple(inputs.race_id.astype(str)),
        source_identity=posterior.artifact.artifact_id,
        basis_names=names,
        coefficient_names=names,
        coefficients=full,
        posterior_chain=chain.ravel(),
        posterior_draw=draw.ravel(),
        explained_national_mean=national,
        race_df=5,
        metadata={
            "target_id": TARGET,
            "output_contract_id": "empirical_linear_fundamentals_joint_v1",
            "recipe_hash": recipe.recipe_hash,
            "training_law_hash": training.law_hash,
            "joint_bank_sha256": posterior.posterior_sha256,
            "target_admission": admission.model_dump(mode="json"),
            "target_feature_rows_sha256": frame_digest(inputs),
            "target_cycle": admission.cutoff_cycle,
            "cutoff_cycle": training.admission.cutoff_cycle,
            "training_cycles": training.cycles.tolist(),
        },
    )


def poll_factor_digest(polls):
    return digest(
        {
            "race_indices": polls.race_indices.tolist(),
            "values": polls.values.tolist(),
            "covariance": polls.covariance.tolist(),
            "observation_ids": polls.observation_ids,
        }
    )


class PollBinding(Record):
    target_id: Literal["senate_aggregate_logit_empirical_mixture_v1"] = TARGET
    denominator: Literal["D_plus_R"] = "D_plus_R"
    universe: Literal["ordinary_DR_contests"] = "ordinary_DR_contests"
    race_ids: tuple[Name, ...]
    evidence_ids: tuple[Name, ...]
    cutoff_utc: AwareDatetime
    likelihood_sha256: Digest
    calibration_sha256: Digest
    source_batch_sha256: Digest

    @model_validator(mode="after")
    def unique_ids(self):
        unique(self.race_ids, "poll race order")
        unique(self.evidence_ids, "poll evidence")
        require(bool(self.race_ids) and bool(self.evidence_ids), "Nonempty poll binding required")
        return self


def condition_polls(recipe, inputs, admission, posterior, polls=None, *, poll_binding=None, **sampler_options):
    """Reuse the fixed Gaussian empirical-prior update with explicit evidence ownership."""
    recipe = Recipe.model_validate(recipe)
    admission = Admission.model_validate(admission)
    prior = predictive_basis(recipe, inputs, admission, posterior)
    if recipe.provider("poll").kernel == "identity":
        require(polls is None and poll_binding is None, "Poll-off cannot silently consume supplied poll data")
        polls = GaussianPollLikelihood(np.zeros(0, dtype=int), np.zeros(0), np.zeros((0, 0)), ())
    else:
        require(
            isinstance(polls, GaussianPollLikelihood) and len(polls.observation_ids) > 0,
            "Bound Gaussian polls required",
        )
        poll_binding = PollBinding.model_validate(poll_binding)
        require(poll_binding.evidence_ids == polls.observation_ids, "Poll evidence ledger differs")
        require(poll_binding.race_ids == prior.race_ids, "Poll race order/support differs")
        require(poll_binding.cutoff_utc <= admission.cutoff_utc, "Poll batch unavailable at target cutoff")
        require(poll_binding.likelihood_sha256 == poll_factor_digest(polls), "Poll factor changed")
        used = {e for p in recipe.providers for e in p.covariate_evidence_ids}
        used.update("outcome:" + r for r in posterior.training.race_ids)
        require(
            not used.intersection(poll_binding.evidence_ids), "Poll observation already used as prior/feature evidence"
        )
    result = sample_poll_posterior(prior, polls, **sampler_options)
    result["factory_recipe_hash"] = recipe.recipe_hash
    result["poll_binding"] = poll_binding.model_dump(mode="json") if poll_binding else None
    result["poll_factor_sha256"] = poll_factor_digest(polls)
    return result


def import_legacy_fit(training, repository_root, plan_pin, receipt_pin, store):
    """Certify one frozen v1 fit, then import its entire bank into a new artifact.

    Requires hashes supplied by the caller for the original plan and receipt.
    Source records/diagnostics are verified against those pins, not authenticated
    by a hash alone. No empirical fitting, target scoring or original-file writes.
    """
    import json

    from src.forecasting.fundamentals_v1 import (
        FundamentalsPriorsV1,
        log_density_v1,
        prepare_training_v1,
    )

    plan_pin, receipt_pin = SourcePin.model_validate(plan_pin), SourcePin.model_validate(receipt_pin)
    plan = json.loads(plan_pin.read(repository_root))
    receipt = json.loads(receipt_pin.read(repository_root))
    require(plan["schema"] == "fundamentals-fit-plan-v1" and plan["frozen"] is True, "Frozen legacy plan required")
    require(
        receipt["schema"] == "fundamentals-fit-receipt-v1" and receipt["status"] == "passed",
        "Passing legacy receipt required",
    )
    require(SourcePin.model_validate(receipt["plan"]) == plan_pin, "Receipt belongs to another plan")
    for pin in [*plan["sources"], *plan["inputs"].values(), plan["prior_predictive"]]:
        SourcePin.model_validate(pin).read(repository_root)
    original_sources = {p["path"]: p["sha256"] for p in plan["sources"]}
    for name in ("fundamentals_v1.py", "baseline_v5.py"):
        require(
            original_sources.get("src/forecasting/" + name) == byte_digest(Path(__file__).with_name(name).read_bytes()),
            "Unknown or changed legacy mathematical implementation",
        )
    for key in (
        "implementation_id",
        "coefficient_names",
        "cutoff_cycle",
        "training_cycles",
        "training_race_ids",
        "priors",
    ):
        require(receipt[key] == plan[key], "Legacy receipt/plan mismatch: " + key)
    arm = plan["implementation_id"]
    require(
        training.recipe.payload(training=True) == legacy_recipe(arm).payload(training=True),
        "No certified exact standard-recipe mapping; substantive provider change needs a joint refit",
    )
    admission = training.admission
    require(
        admission.cutoff_cycle == plan["cutoff_cycle"]
        and admission.start_cycle == plan["start_cycle"]
        and sorted(admission.allowed_cycles) == sorted(plan.get("allowed_cycles", [])),
        "Training window differs",
    )
    for field_name, input_name in (
        ("feature_source_sha256", "features"),
        ("outcome_source_sha256", "outcomes"),
        ("receipt_sha256", "panel_receipt"),
    ):
        require(getattr(admission, field_name) == plan["inputs"][input_name]["sha256"], "Training source differs")
    root = Path(repository_root).resolve()
    panel_pin = SourcePin.model_validate(plan["inputs"]["panel_receipt"])
    panel = json.loads(panel_pin.read(root))
    require(
        panel.get("schema") == "SenateFundamentalsPanelV1"
        and panel.get("strict_prior") is True
        and panel.get("status") == "qualified_retrospective_proxy",
        "Qualified panel receipt required",
    )
    panel_path = (root / panel_pin.path).resolve()
    admitted = {(panel_path.parent / name).resolve(): h for name, h in panel["output_sha256"].items()}
    for name in ("features", "outcomes"):
        pin = plan["inputs"][name]
        require(admitted.get((root / pin["path"]).resolve()) == pin["sha256"], "Legacy input outside admitted panel")
    x = pd.read_csv(BytesIO(SourcePin.model_validate(plan["inputs"]["features"]).read(root)))
    outcome_bytes = SourcePin.model_validate(plan["inputs"]["outcomes"]).read(root)
    # Parse only eligible earlier outcome records, exactly as the original runner.
    meta = pd.read_csv(BytesIO(outcome_bytes), usecols=["cycle"])
    keep = meta.cycle.ge(plan["start_cycle"]) & meta.cycle.lt(plan["cutoff_cycle"])
    if plan.get("allowed_cycles"):
        keep &= meta.cycle.isin(plan["allowed_cycles"])
    lines = set(np.flatnonzero(keep) + 1)
    y = pd.read_csv(
        BytesIO(outcome_bytes),
        usecols=["race_id", "cycle", "actual_dem_share"],
        skiprows=lambda line: line > 0 and line not in lines,
    )
    rebuilt = compile_training(training.recipe, x, y, admission)
    require(rebuilt.payload() == training.payload(), "Compiled rows/data/design differ from frozen admitted sources")
    x = x.loc[x.cycle.ge(plan["start_cycle"]) & x.cycle.lt(plan["cutoff_cycle"])]
    if plan.get("allowed_cycles"):
        x = x.loc[x.cycle.isin(plan["allowed_cycles"])]
    old = prepare_training_v1(x, y, plan["cutoff_cycle"], arm)
    require(
        old["race_ids"] == plan["training_race_ids"] == list(training.race_ids)
        and old["cycles"].tolist() == plan["training_cycles"] == training.cycles.tolist(),
        "Legacy row roster differs",
    )
    require(tuple(old["coefficient_names"]) == tuple(plan["coefficient_names"]), "Legacy coefficient names differ")
    mapped_names = tuple(LEGACY_NAMES[n] for n in old["coefficient_names"])
    require(mapped_names == training.parameter_names, "Coefficient permutation/mapping unsupported")
    priors = FundamentalsPriorsV1(**plan["priors"])
    means, sds = priors.coefficient_parameters(arm)
    require(
        np.array_equal(means, [p.mean for p in training.recipe.parameters])
        and np.array_equal(sds, [p.sd for p in training.recipe.parameters])
        and priors.cycle_scale_prior_sd == training.recipe.provider("cycle").scale_prior_sd
        and priors.race_scale_prior_sd == training.recipe.provider("race").scale_prior_sd
        and priors.race_df == 5,
        "Legacy prior law differs",
    )
    for left, right in (
        (old["offset"], training.offset),
        (old["design"], training.design),
        (old["y"] + old["offset"], training.response),
    ):
        require(np.allclose(left, right, atol=2e-14, rtol=2e-14), "Legacy affine law differs")
    require(np.array_equal(old["cycle_index"], training.cycle_index), "Legacy cycle assignment differs")
    rng = np.random.default_rng(610203)
    density_error, gradient_error = [], []
    for _ in range(4):
        b = means + rng.normal(size=len(means)) * sds
        z = rng.normal(size=len(training.cycles))
        tc, tr = rng.uniform(0.07, 0.4, 2)
        vector = np.r_[b, tc, tr, z]
        q = len(b)

        def legacy_value(v):
            return log_density_v1(old, v[:q], v[q + 2 :], v[q], v[q + 1], priors)

        density_error.append(abs(log_density(training, b, z, tc, tr) - legacy_value(vector)))
        numeric = np.zeros(len(vector))
        for j in range(len(vector)):
            plus, minus = vector.copy(), vector.copy()
            plus[j] += 1e-5
            minus[j] -= 1e-5
            numeric[j] = (legacy_value(plus) - legacy_value(minus)) / 2e-5
        analytical = gradient(training, b, z, tc, tr)
        gradient_error.append(float(np.max(np.abs(numeric - analytical))))
        require(np.allclose(numeric, analytical, atol=5e-5, rtol=2e-6), "Legacy gradient equivalence failed")
    require(max(density_error) < 1e-8, "Legacy density equivalence failed")
    diagnostics, gates = receipt["diagnostics"], plan["diagnostics"]
    require(
        diagnostics["passed"] is True
        and diagnostics["finite"] is True
        and diagnostics["max_rank_rhat"] <= gates["max_rank_rhat"]
        and diagnostics["min_bulk_ess"] >= gates["min_bulk_ess"]
        and diagnostics["min_tail_ess"] >= gates["min_tail_ess"]
        and diagnostics["divergences"] == diagnostics["max_depth_hits"] == 0,
        "Legacy diagnostics do not satisfy their pinned gates",
    )
    post_pin = SourcePin.model_validate(receipt["posterior"])
    post_bytes = post_pin.read(root)
    with np.load(BytesIO(post_bytes), allow_pickle=False) as raw:
        arrays = _posterior_arrays({n: raw[n] for n in raw.files}, training)
    require(
        arrays["beta"].shape[:2] == (plan["sampler"]["chains"], plan["sampler"]["draws"]),
        "Legacy posterior shape differs from plan",
    )
    require(training.implementation_sha256 == implementation_fingerprint(), "Factory changed during import")
    certificate = {
        "schema": "aggregate-factory-legacy-equivalence-v1",
        "status": "passed",
        "compiled_law_hash": training.law_hash,
        "original_plan": plan_pin.model_dump(mode="json"),
        "original_receipt": receipt_pin.model_dump(mode="json"),
        "original_posterior": post_pin.model_dump(mode="json"),
        "legacy_implementation_id": arm,
        "parameter_mapping": dict(zip(old["coefficient_names"], mapped_names)),
        "max_density_error": max(density_error),
        "max_gradient_error": max(gradient_error),
        "training_cutoff_kind": "strictly_earlier_election_cycles",
        "target_cutoff_note": "Exact target information instant is a new binding, not a legacy publication assertion",
        "original_posterior_bytes_preserved": True,
        "independent_component_banks_combined": False,
    }
    return _store_joint(
        store,
        training,
        arrays,
        diagnostics,
        {"origin": "certified_legacy_import"},
        {
            "equivalence.json": canonical(certificate),
            "original-posterior.npz": post_bytes,
            "original-plan.json": plan_pin.read(root),
            "original-receipt.json": receipt_pin.read(root),
        },
    )
