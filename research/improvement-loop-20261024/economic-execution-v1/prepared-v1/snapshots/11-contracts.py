"""Executable boundaries supplementing config/redesign/contracts.yaml.

These contracts validate declared scientific meaning and dependency structure.
They do not establish that an observation is authentic, a population model is
calibrated, or separately supplied draws have the declared joint distribution.
"""

from datetime import date
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

Name = Annotated[str, Field(min_length=1, pattern=r"^\S(?:.*\S)?$")]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Role = Literal[
    "population", "community", "fundamentals", "environment", "measurement", "participation", "choice", "counts_rules"
]
Stage = Literal["prepare", "fit", "factors", "diagnose", "predict", "score", "publish"]
Target = Literal[
    "joint_posterior",
    "cut_posterior",
    "bootstrap_mixture",
    "conditional_law",
    "sensitivity_ensemble",
    "synthetic_fixture",
]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")


def unique(values, label):
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate {label}")


class Evidence(Record):
    source_id: Name
    content_sha256: Digest
    reference_start: date
    reference_end: date
    first_published_at: AwareDatetime | None
    version_published_at: AwareDatetime | None
    retrieved_at: AwareDatetime
    version: Name
    geography_version: Name
    universe: Name
    measurement_time: AwareDatetime | None
    measurement_time_status: Literal["known", "unknown", "not_applicable"]
    dependence_group: Name

    @model_validator(mode="after")
    def periods(self):
        if self.reference_end < self.reference_start:
            raise ValueError("Reversed source reference period")
        if (
            self.first_published_at is not None
            and self.version_published_at is not None
            and self.version_published_at < self.first_published_at
        ):
            raise ValueError("Version release precedes first publication")
        if (self.measurement_time_status == "known") != (self.measurement_time is not None):
            raise ValueError("Explicit measurement recency and unknown status required")
        for t in (self.first_published_at, self.version_published_at):
            if t is not None and t > self.retrieved_at:
                raise ValueError("Publication cannot follow this source retrieval")
        if self.measurement_time is not None:
            if self.measurement_time > self.retrieved_at or (
                self.version_published_at is not None and self.measurement_time > self.version_published_at
            ):
                raise ValueError("Measurement cannot follow its retrieved/published source version")
        return self

    def require_available(self, cutoff):
        Evidence.model_validate(self)
        if self.first_published_at is None or self.version_published_at is None:
            raise ValueError("Unknown release/version blocks as-of admission")
        if max(self.first_published_at, self.version_published_at) > cutoff:
            raise ValueError("Source version unavailable at cutoff")
        if self.reference_end > cutoff.date():
            raise ValueError("Observed reference period extends beyond cutoff")
        if self.measurement_time is not None and self.measurement_time > cutoff:
            raise ValueError("Measurement unavailable at cutoff")


class Geography(Record):
    support_id: Name
    version: Name
    atomic_area_ids: tuple[Name, ...] = Field(min_length=1)
    valid_from: date
    valid_through: date | None
    mapping_sha256: Digest
    uncertainty_group: Name

    @model_validator(mode="after")
    def validate_geography(self):
        unique(self.atomic_area_ids, "geographic atoms")
        if self.valid_through is not None and self.valid_through < self.valid_from:
            raise ValueError("Reversed boundary validity interval")
        return self

    def require_date(self, target):
        Geography.model_validate(self)
        if target < self.valid_from or (self.valid_through is not None and target > self.valid_through):
            raise ValueError("Geography invalid on target date")


class Quantity(Record):
    port: Name
    estimand: Name
    universe: Name
    unit: Name
    denominator: Name  # Use explicit 'not_a_ratio', never infer a denominator.
    support: Name
    basis: Name
    geography: Name
    reference_start: date
    reference_end: date
    conditioned_on: tuple[Name, ...]

    @model_validator(mode="after")
    def period(self):
        if self.reference_end < self.reference_start:
            raise ValueError("Reversed quantity period")
        unique(self.conditioned_on, "conditioning coordinates")
        return self

    def meaning(self):
        return self.model_dump(mode="json", exclude={"port"})


class Component(Record):
    component_id: Name
    role: Role
    contract_version: Literal["model-stack-v1"] = "model-stack-v1"
    statistical_model_version: Name
    implementation_version: Name
    implementation_sha256: Digest
    capabilities: tuple[Stage, ...] = Field(min_length=1)
    inputs: tuple[Quantity, ...]
    outputs: tuple[Quantity, ...]
    limitations: tuple[Name, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def identifiers(self):
        unique(self.capabilities, "stage capabilities")
        unique([q.port for q in self.inputs], "input ports")
        unique([q.port for q in self.outputs], "output ports")
        return self

    def require_stage(self, stage: str):
        if stage not in self.capabilities:
            raise ValueError(f"{self.component_id} does not implement {stage}")


class Binding(Record):
    producer: Name
    output_port: Name
    consumer: Name
    input_port: Name


class Fixture(Record):
    fixture_id: Name
    artifact_id: Digest
    output: Quantity
    target: Target


class Latent(Record):
    state_id: Name
    owner: Name


class Factor(Record):
    factor_id: Name
    owner: Name
    kind: Literal["prior", "transition", "likelihood", "constraint"]
    latent_ids: tuple[Name, ...]
    observation_ids: tuple[Name, ...]
    sample_dependence_ids: tuple[Name, ...]

    @model_validator(mode="after")
    def identities(self):
        for key in ("latent_ids", "observation_ids", "sample_dependence_ids"):
            unique(getattr(self, key), key)
        if self.kind != "likelihood" and (self.observation_ids or self.sample_dependence_ids):
            raise ValueError("Observed evidence belongs to likelihood factors")
        if self.kind == "likelihood" and not self.observation_ids:
            raise ValueError("Likelihood requires explicit observation identities")
        return self


class InferenceBlock(Record):
    block_id: Name
    components: tuple[Name, ...] = Field(min_length=1)
    target: Target
    target_definition_sha256: Digest
    feedback_policy: Literal["joint_within_block", "condition_on_frozen_upstream"]
    limitations: tuple[Name, ...] = Field(min_length=1)


class Graph(Record):
    schema_version: Literal["model-stack-graph-v1"] = "model-stack-graph-v1"
    components: tuple[Component, ...] = Field(min_length=1)
    fixtures: tuple[Fixture, ...] = ()
    bindings: tuple[Binding, ...]
    blocks: tuple[InferenceBlock, ...] = Field(min_length=1)
    latents: tuple[Latent, ...]
    factors: tuple[Factor, ...]

    @model_validator(mode="after")
    def topology(self):
        components = {c.component_id: c for c in self.components}
        unique([c.component_id for c in self.components], "component IDs")
        unique([f.fixture_id for f in self.fixtures], "fixture IDs")
        fixtures = {f.fixture_id: f for f in self.fixtures}
        if set(fixtures).intersection(components):
            raise ValueError("Fixture and component identities overlap")
        unique([b.block_id for b in self.blocks], "block IDs")
        assigned = [c for b in self.blocks for c in b.components]
        unique(assigned, "inference-block membership")
        if set(assigned) != set(components):
            raise ValueError("Every component belongs to exactly one inference block")
        for b in self.blocks:
            if b.target in ("joint_posterior", "cut_posterior") and not any(
                "factors" in components[c].capabilities for c in b.components
            ):
                raise ValueError("Posterior block needs factor providers")
        ports = set()
        edges = set()
        for link in self.bindings:
            if link.producer not in components.keys() | fixtures.keys() or link.consumer not in components:
                raise ValueError("Unknown binding component")
            outputs = (
                (fixtures[link.producer].output,) if link.producer in fixtures else components[link.producer].outputs
            )
            source = {q.port: q for q in outputs}.get(link.output_port)
            dest = {q.port: q for q in components[link.consumer].inputs}.get(link.input_port)
            if source is None or dest is None or source.meaning() != dest.meaning():
                raise ValueError("Port meaning differs; register an explicit adapter component")
            key = (link.consumer, link.input_port)
            if key in ports:
                raise ValueError("Input port has multiple producers")
            ports.add(key)
            edges.add((link.producer, link.consumer))
        if ports != {(c.component_id, q.port) for c in self.components for q in c.inputs}:
            raise ValueError("Every input port needs a producer, including fixture components")
        unique([s.state_id for s in self.latents], "latent IDs")
        owners = {s.state_id: s.owner for s in self.latents}
        if not set(owners.values()) <= set(components):
            raise ValueError("Unknown latent owner")
        unique([f.factor_id for f in self.factors], "factor IDs")
        observations, samples = [], []
        for f in self.factors:
            if f.owner not in components or "factors" not in components[f.owner].capabilities:
                raise ValueError("Factor must belong to a registered factor provider")
            if not set(f.latent_ids) <= set(owners):
                raise ValueError("Factor references unowned latent state")
            observations.extend(f.observation_ids)
            samples.extend(f.sample_dependence_ids)
            edges.update((owners[s], f.owner) for s in f.latent_ids if owners[s] != f.owner)
        unique(observations, "likelihood observation ownership")
        # Conservative v1: shared-sample observations must be one joint factor.
        unique(samples, "sample dependence factor ownership")
        block_of = {c: b.block_id for b in self.blocks for c in b.components}
        block_edges = {(block_of[a], block_of[b]) for a, b in edges if a in block_of and block_of[a] != block_of[b]}
        remaining = set(block_of.values())
        while remaining:
            ready = {b for b in remaining if not any(v == b and u in remaining for u, v in block_edges)}
            if not ready:
                raise ValueError("Feedback cycle crosses blocks; register one joint inference block")
            remaining -= ready
        return self

    def affected_components(self, changed: set[str]) -> tuple[str, ...]:
        """Invalidate a whole inference block and its port/latent descendants."""
        Graph.model_validate(self)
        known = {c.component_id for c in self.components}
        if not changed <= known | {f.fixture_id for f in self.fixtures}:
            raise ValueError("Unknown changed component")
        result = set(changed)
        owners = {s.state_id: s.owner for s in self.latents}
        edges = {(b.producer, b.consumer) for b in self.bindings}
        edges.update((owners[s], f.owner) for f in self.factors for s in f.latent_ids)
        while True:
            previous = result.copy()
            for block in self.blocks:
                if result.intersection(block.components):
                    result.update(block.components)
            result.update(b for a, b in edges if a in result)
            if result == previous:
                return tuple(sorted(result.intersection(known)))
