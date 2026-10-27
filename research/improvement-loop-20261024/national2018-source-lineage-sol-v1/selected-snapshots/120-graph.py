"""Typed joint graph assembly; never merges separately inferred posteriors."""

import hashlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
from scipy.linalg import helmert

from src.senate_count_stack_v1.demographics import CELL_IDS, design
from .inputs_v3 import COMMUNITY_IDS, validate_parent, validate_problem

VERSION = "senate_national_emission_mixed_assembly_graph_v1"
ROLES = (
    "population",
    "community",
    "fundamentals",
    "environment",
    "participation",
    "choice",
    "counts_rules",
    "measurement",
)
COMPILER_FIELDS = frozenset(
    {
        "version",
        "composition_identity",
        "scientific_target_id",
        "scientific_law_version",
        "baseline_equivalence_certified",
        "problem_identity",
        "parent_identity",
        "upstream_identity",
        "parent_id",
        "parent_mass",
        "event_ids",
        "required_free_rvs",
        "required_physical_diagnostics",
        "inference_target",
        "parent_mass_used_in_logp",
    }
)
SCHEDULE = (
    ("population", "exposure"),
    ("community", "composition"),
    ("environment", "residuals"),
    ("fundamentals", "mean"),
    ("participation", "behavior"),
    ("choice", "utility"),
    ("counts_rules", "allocation"),
    ("participation", "event"),
    ("counts_rules", "counts"),
    ("measurement", "observations"),
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({k: freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(v) for v in value)
    if isinstance(value, np.ndarray):
        return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
    return value


@dataclass(frozen=True)
class PortType:
    name: str
    unit: str
    universe: str
    axes: tuple[str, ...]
    time: str
    denominator: str = "not_a_ratio"
    contest_map: bool = False


E = ("event", "demographic", "community")
PORTS = {
    p.name: p
    for p in [
        PortType("exposure", "persons", "citizen_adult_exposure", ("event", "demographic"), "event_date"),
        PortType(
            "community_logits",
            "log_relative_weight",
            "citizen_adult_profile",
            ("state", "demographic", "community"),
            "first_community_date",
        ),
        PortType(
            "community_temporal",
            "log_relative_weight",
            "citizen_adult_profile",
            ("date", "community"),
            "named_community_dates",
        ),
        PortType("community", "probability", "citizen_adult_profile", E, "event_date", "cell_population"),
        PortType("cycle_effect", "logit_shift", "turnout_and_DR_choice", ("cycle", "behavior2"), "cycle"),
        PortType("race_effect", "logit_shift", "DR_choice", ("contest",), "final_contest_outcome"),
        PortType("structural_choice", "logit_shift", "DR_choice", (), "event_covariates", contest_map=True),
        PortType("incumbency", "candidate_utility", "named_candidates", (), "event_covariates"),
        PortType("turnout_logit", "logit", "standardized_citizen_adult_propensity", E, "event_date"),
        PortType(
            "propensity",
            "probability",
            "standardized_citizen_adult_propensity",
            E,
            "event_date",
            "cell_profile_exposure",
        ),
        PortType("cell_intensity", "expected_participants_before_G", "event_participants", E, "event_date"),
        PortType("event_gamma_shape", "gamma_shape", "event_frailty", (), "all_events"),
        PortType("response_logits", "logit", "contest_valid_candidate_response", (), "event_date", contest_map=True),
        PortType(
            "response", "probability", "contest_valid_candidate_response", (), "event_date", "event_participants", True
        ),
        PortType(
            "candidate_utilities",
            "candidate_utility",
            "named_candidates",
            ("demographic", "community", "candidate"),
            "final_contest_outcome",
            contest_map=True,
        ),
        PortType(
            "candidate_probability",
            "probability",
            "named_candidates",
            ("demographic", "community", "candidate"),
            "final_contest_outcome",
            "candidate_response",
            True,
        ),
        PortType("political_axes", "contrast", "named_candidates", ("candidate",), "contest_roster", contest_map=True),
        PortType(
            "category_probability",
            "probability",
            "candidate_and_noncandidate_options",
            ("demographic", "community", "category"),
            "final_contest_outcome",
            "event_participants",
            True,
        ),
        PortType(
            "base_category_rates",
            "expected_counts_before_G",
            "candidate_and_noncandidate_options",
            ("category",),
            "final_contest_outcome",
            contest_map=True,
        ),
        PortType("eventG", "positive_multiplier", "shared_event_frailty", ("event",), "event_date"),
        PortType("event_intensity", "expected_participants", "event_participants", ("event",), "event_date"),
        PortType(
            "category_rates",
            "expected_counts",
            "candidate_and_noncandidate_options",
            ("category",),
            "final_contest_outcome",
            contest_map=True,
        ),
    ]
}


@dataclass(frozen=True)
class Stage:
    name: str
    inputs: tuple[PortType, ...]
    outputs: tuple[PortType, ...]


@dataclass(frozen=True)
class ComponentSpec:
    implementation_id: str
    role: str
    stages: tuple[Stage, ...]
    parameter_patterns: tuple[str, ...]
    implementation_path: str
    implementation_sha256: str
    joint_inference_required: bool = True
    configuration_json: str = "{}"
    dependency_pins: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        config = json.loads(self.configuration_json)
        if not isinstance(config, dict):
            raise ValueError("Component scientific configuration must be a JSON object")
        object.__setattr__(self, "configuration_json", canonical(config).decode())
        pins = tuple(tuple(pin) for pin in self.dependency_pins)
        if any(len(pin) != 2 for pin in pins) or len({p[0] for p in pins}) != len(pins):
            raise ValueError("Unique declared dependency path/hash pairs required")
        object.__setattr__(self, "dependency_pins", tuple(sorted(pins)))

    @property
    def configuration(self):
        return json.loads(self.configuration_json)

    def record(self):
        from dataclasses import asdict

        return asdict(self)


@dataclass(frozen=True)
class Composition:
    selections: tuple[tuple[str, str], ...]
    standardized: bool = True
    person_coordinates: str = "hybrid_v1"

    def record(self):
        return dict(
            selections=self.selections, standardized=self.standardized, person_coordinates=self.person_coordinates
        )


class Registry:
    def __init__(self):
        self._providers = {}
        self._specs = {}

    def register(self, provider):
        spec = provider.spec
        if set(vars(provider)) != {"spec"}:
            raise ValueError(
                "Provider configuration must be source-declared; hidden mutable execution state is unsupported"
            )
        if not isinstance(spec, ComponentSpec) or spec.role not in ROLES or not spec.joint_inference_required:
            raise ValueError("Versioned joint component with a supported responsibility required")
        if spec.implementation_id in self._providers:
            raise ValueError("Duplicate component implementation")
        actual = Path(inspect.getfile(type(provider))).resolve()
        if (
            str(actual) != spec.implementation_path
            or hashlib.sha256(actual.read_bytes()).hexdigest() != spec.implementation_sha256
        ):
            raise ValueError("Provider must bind its actual implementation source")
        if len({s.name for s in spec.stages}) != len(spec.stages):
            raise ValueError("Duplicate build stage")
        for path, sha in spec.dependency_pins:
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
                raise ValueError("Declared component dependency changed")
        self._providers[spec.implementation_id] = provider
        self._specs[spec.implementation_id] = canonical(spec.record())
        return self

    def get(self, name):
        if name not in self._providers:
            raise ValueError("Unregistered component implementation")
        if canonical(self._providers[name].spec.record()) != self._specs[name]:
            raise ValueError("Registered component specification mutated")
        if set(vars(self._providers[name])) != {"spec"}:
            raise ValueError("Registered provider acquired hidden mutable state")
        for path, sha in self._providers[name].spec.dependency_pins:
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
                raise ValueError("Declared component dependency changed")
        return self._providers[name]

    def catalog(self):
        return tuple(self._providers[k].spec.record() for k in sorted(self._providers))


@dataclass(frozen=True)
class BuildContext:
    problem: object
    parent: object
    roles: object
    composition: Composition
    data: object
    protected: object
    states: tuple
    events: tuple
    contests: tuple
    event_index: object
    contest_map: object
    state_axis: np.ndarray
    cycles: tuple
    cycle_axis: np.ndarray
    dates: tuple
    date_index: object
    event_dates: np.ndarray
    source_names: tuple
    source_axis: object
    x: np.ndarray
    z: np.ndarray
    midterm: np.ndarray
    context: np.ndarray

    @classmethod
    def create(cls, problem, parent, roles, composition):
        problem = validate_problem(problem)
        parent = validate_parent(problem, parent)
        data, protected = freeze(problem.to_dict()), freeze(parent.to_dict())
        states, events, contests = data["states"], data["events"], data["contests"]
        cycles = tuple(sorted({e["cycle"] for e in events}))
        dates = tuple(sorted({e["date"] for e in events} | {s["community_date"] for s in data["surveys"]}))
        source_names = tuple(sorted({s["source_id"] for s in data["surveys"]}))
        return cls(
            problem,
            parent,
            roles,
            composition,
            data,
            protected,
            states,
            events,
            contests,
            freeze({e["id"]: i for i, e in enumerate(events)}),
            freeze({c["id"]: c for c in contests}),
            freeze(np.array([states.index(e["state"]) for e in events])),
            cycles,
            freeze(np.array([cycles.index(e["cycle"]) for e in events])),
            dates,
            freeze({d: i for i, d in enumerate(dates)}),
            freeze(np.array([dates.index(e["date"]) for e in events])),
            source_names,
            freeze({s: i for i, s in enumerate(source_names)}),
            freeze(design(data["reference"]["weights"], data["reference"]["id"]).matrix),
            freeze(helmert(12).T),
            freeze(np.array([e["midterm"] for e in events], dtype=float)),
            freeze(np.array([e["president_party"] * e["midterm"] for e in events], dtype=float)),
        )

    @property
    def coords(self):
        return dict(
            event=[e["id"] for e in self.events],
            state=self.states,
            cycle=self.cycles,
            demographic=CELL_IDS,
            community_profile=COMMUNITY_IDS,
        )

    def shape(self, port, contest_id=None):
        sizes = dict(
            event=len(self.events),
            state=len(self.states),
            cycle=len(self.cycles),
            contest=len(self.contests),
            date=len(self.dates),
            demographic=24,
            community=12,
            behavior2=2,
        )
        if contest_id is not None:
            c = self.contest_map[contest_id]
            sizes.update(
                candidate=len(c["candidates"]), category=len(c["candidates"]) + len(c["noncandidate_categories"])
            )
        return tuple(sizes[a] for a in port.axes)

    @property
    def evidence_tokens(self):
        tokens = {f"survey:{s['source_id']}:{r['record_id']}" for s in self.data["surveys"] for r in s["rows"]}
        tokens |= {f"official:{o['id']}" for o in self.data["official_counts"]}
        tokens |= {f"report:{o['id']}" for o in self.data["participation_reports"]}
        tokens |= {f"poll:{r['observation_id']}" for b in self.data["polls"] for r in b["rows"]}
        return tokens


class Scope:
    """One stage can access only declared symbolic inputs and claim factors once."""

    def __init__(self, context, stage, ports):
        self.context = context
        self.inputs = MappingProxyType({p.name: ports[p.name] for p in stage.inputs})
        self.factors, self.annotations, self.physical_diagnostics = {}, {}, []

    def factor(self, name, tokens=(), *, kind="observation"):
        if name in self.factors or kind not in ("observation", "coordinate_prior"):
            raise ValueError("Duplicate or unsupported factor claim")
        if kind == "observation" and not tokens:
            raise ValueError("Observation factors require source observation tokens")
        if kind == "coordinate_prior" and tokens:
            raise ValueError("A coordinate prior cannot consume observed data again")
        self.factors[name] = dict(kind=kind, tokens=list(tokens))


def _validate_composition(registry, composition):
    if type(composition) is not Composition or type(composition.standardized) is not bool:
        raise ValueError("Immutable composition configuration required")
    if composition.person_coordinates not in ("centered_v1", "hybrid_v1"):
        raise ValueError("Unsupported candidate coordinate configuration")
    selections = dict(composition.selections)
    if len(selections) != len(composition.selections) or set(selections) != set(ROLES):
        raise ValueError("Exactly one implementation per scientific responsibility required")
    providers = {role: registry.get(name) for role, name in selections.items()}
    available = {}
    for role in ROLES:
        provider = providers[role]
        if provider.spec.role != role:
            raise ValueError("Implementation assigned to incompatible responsibility")
        expected = {stage for owner, stage in SCHEDULE if owner == role}
        if {s.name for s in provider.spec.stages} != expected:
            raise ValueError("Incomplete or incompatible joint build stages")
        spec = provider.spec
        if hashlib.sha256(Path(spec.implementation_path).read_bytes()).hexdigest() != spec.implementation_sha256:
            raise ValueError("Registered implementation source changed")
    for role, name in SCHEDULE:
        stage = next(s for s in providers[role].spec.stages if s.name == name)
        for port in stage.inputs:
            if available.get(port.name) != port:
                raise ValueError("Missing or incompatible typed input port")
        for port in stage.outputs:
            if port.name in available or PORTS.get(port.name) != port:
                raise ValueError("Duplicate output or incompatible units/axes/temporal semantics")
            available[port.name] = port
    if set(available) != set(PORTS):
        raise ValueError("Incomplete physical joint output interface")
    return providers


def compile_model(registry, composition, problem, parent, *, roles=None):
    import re

    import pymc as pm
    import pytensor.tensor as pt

    providers = _validate_composition(registry, composition)
    ctx = BuildContext.create(problem, parent, roles, composition)
    ports, inventory, tokens, annotations, physical = {}, {}, set(), {}, []
    with pm.Model(coords=ctx.coords) as model:
        for role, name in SCHEDULE:
            provider = providers[role]
            stage = next(s for s in provider.spec.stages if s.name == name)
            before = {
                kind: {v.name for v in values}
                for kind, values in [
                    ("parameters", model.free_RVs),
                    ("deterministics", model.deterministics),
                    ("observed", model.observed_RVs),
                    ("potentials", model.potentials),
                ]
            }
            scope = Scope(ctx, stage, ports)
            output = provider.build(name, ctx, scope)
            registry.get(provider.spec.implementation_id)
            if set(output) != {p.name for p in stage.outputs}:
                raise ValueError("Provider omitted or invented an output port")
            for port in stage.outputs:
                value = output[port.name]
                if port.name == "exposure":
                    if not isinstance(value, np.ndarray) or not np.array_equal(
                        value, np.array(ctx.protected["exposure"])
                    ):
                        raise ValueError(
                            "Protected population adapter cannot transform or resample its supplied parent"
                        )
                if port.contest_map:
                    if set(value) != set(ctx.contest_map):
                        raise ValueError("Ragged contest port must preserve every named contest")
                    # specify_shape is a deterministic constraint, not a prior.
                    ports[port.name] = MappingProxyType(
                        {k: pt.specify_shape(pt.as_tensor_variable(v), ctx.shape(port, k)) for k, v in value.items()}
                    )
                else:
                    ports[port.name] = pt.specify_shape(pt.as_tensor_variable(value), ctx.shape(port))
            after = {
                kind: {v.name for v in values}
                for kind, values in [
                    ("parameters", model.free_RVs),
                    ("deterministics", model.deterministics),
                    ("observed", model.observed_RVs),
                    ("potentials", model.potentials),
                ]
            }
            new = {kind: sorted(after[kind] - before[kind]) for kind in before}
            for parameter in new["parameters"]:
                if not any(re.fullmatch(pattern, parameter) for pattern in provider.spec.parameter_patterns):
                    raise ValueError("Provider created a parameter outside declared ownership")
            if set(scope.factors) != set(new["observed"] + new["potentials"]):
                raise ValueError("Every numerical factor needs explicit observation or coordinate-prior ownership")
            for claim in scope.factors.values():
                local = claim["tokens"]
                if len(local) != len(set(local)) or tokens & set(local) or not set(local) <= ctx.evidence_tokens:
                    raise ValueError("Duplicate or unknown source observation consumed")
                tokens.update(local)
            if set(annotations) & set(scope.annotations):
                raise ValueError("Duplicate shared-coordinate annotation")
            if COMPILER_FIELDS & set(scope.annotations):
                raise ValueError("Component annotation cannot claim compiler-owned identity, mass or qualification")
            annotations.update(scope.annotations)
            physical.extend(scope.physical_diagnostics)
            inventory[f"{role}:{name}"] = dict(
                implementation=provider.spec.implementation_id,
                **new,
                factors=scope.factors,
                inputs=[p.name for p in stage.inputs],
                outputs=[p.name for p in stage.outputs],
            )
        if tokens != ctx.evidence_tokens:
            raise ValueError("Joint composition did not consume each admitted observation exactly once")
    from .api import source_inventory

    composition_record = dict(
        version=VERSION,
        configuration=composition.record(),
        implementations=[providers[r].spec.record() for r in ROLES],
        compiler_and_baseline_dependencies=source_inventory(),
    )
    composition_id = digest(composition_record)
    target = digest(
        dict(
            composition_id=composition_id,
            problem=ctx.problem.identity,
            parent=ctx.parent.identity,
            target="population_cut_v1",
        )
    )
    # A later source-bound equivalence certificate may expose the baseline's
    # existing target. Compatibility alone never grants that certificate.
    model.joint_contract = MappingProxyType(
        dict(
            version=VERSION,
            composition_identity=composition_id,
            scientific_target_id=target,
            scientific_law_version="modular_composition_unqualified_v1",
            baseline_equivalence_certified=False,
            problem_identity=ctx.problem.identity,
            parent_identity=ctx.parent.identity,
            upstream_identity=ctx.protected["upstream_identity"],
            parent_id=ctx.protected["parent_id"],
            parent_mass=ctx.protected["mass"],
            event_ids=tuple(e["id"] for e in ctx.events),
            required_free_rvs=tuple(v.name for v in model.free_RVs),
            required_physical_diagnostics=tuple(dict.fromkeys(physical)),
            inference_target="population_cut_v1",
            parent_mass_used_in_logp=False,
            **annotations,
        )
    )
    model.component_inventory = freeze(inventory)
    model.composition_record = freeze(composition_record)
    model.typed_ports = MappingProxyType(ports)
    return model
