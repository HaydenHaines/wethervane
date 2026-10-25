"""Complete F0 parameter banks and existing five/six Gaussian measurement laws.

No Gibbs transitions, optimizer, forecast summaries or numeric target labels.
"""

import argparse
import inspect
import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import adapter_sources as sources

BINDINGS, COMPATIBILITY = sources.verify()
sys.dont_write_bytecode = True

import arviz as az  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
require = sources.require
TARGET = sources.load_guarded("verified_adapter_target", LAB / "momentum-target-preparation-sol-v1/prepare.py")
MODEL = sources.load_guarded("verified_adapter_momentum", LAB / "momentum-analytic-prototype-sol-v1/momentum.py")
OWNER = TARGET.OWNER
from src.forecasting import polling_basis_v1 as native_basis  # noqa: E402

VARIANTS = {"FS5": ("native_full_series", 5), "FS6": ("native_full_series", 6), "LP5": ("latest_per_race_pollster", 5)}
POSTERIOR_VARIABLES = {
    "beta": ("chain", "draw", "coefficient"),
    "tau_cycle": ("chain", "draw"),
    "tau_race": ("chain", "draw"),
    "cycle_z": ("chain", "draw", "source_cycle"),
    "cycle_effect": ("chain", "draw", "source_cycle"),
}


def trusted_pins():
    result = {str(Path(row["path"]).resolve()): row for row in COMPATIBILITY["inspected_pins"]}
    review_path = Path(BINDINGS["files"][7]["path"]).parent / "results.json"
    for path, sha in json.loads(review_path.read_bytes())["inspected_pins"].items():
        result[str(Path(path).resolve())] = dict(path=path, sha256=sha)
    for row in json.loads(sources.checked(BINDINGS["files"][13]))["files"]:
        result[str(Path(row["path"]).resolve())] = row
    for row in json.loads(sources.checked(BINDINGS["files"][8]))["source_pins"]:
        result[str(Path(row["path"]).resolve())] = row
    parent = Path(BINDINGS["files"][11]["path"]).parent
    for row in json.loads(sources.checked(BINDINGS["files"][11]))["files"]:
        path = str((parent / row["path"]).resolve())
        result[path] = {**row, "path": path}
    return result


TRUSTED_PINS = MappingProxyType({key: MappingProxyType(dict(value)) for key, value in trusted_pins().items()})


def trusted_pin(path):
    key = str(Path(path).resolve())
    require(key in TRUSTED_PINS, "undeclared adapter dependency: " + key)
    return TRUSTED_PINS[key]


def trusted_json(path):
    return json.loads(sources.checked(trusted_pin(path)))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return __import__("hashlib").sha256(value).hexdigest()


def freeze(value):
    if isinstance(value, np.ndarray):
        require(not value.dtype.hasobject, "object arrays are not an admitted adapter representation")
        return np.frombuffer(np.ascontiguousarray(value).tobytes(), dtype=value.dtype).reshape(value.shape)
    if isinstance(value, dict):
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    return value


def immutable_instance(instance):
    # Native constructors copy their arrays. Freeze those owned copies too.
    for name, value in vars(instance).items():
        if isinstance(value, (np.ndarray, tuple)):
            object.__setattr__(instance, name, freeze(value))
    return instance


def array_pin(array):
    array = np.asarray(array)
    return dict(shape=list(array.shape), dtype=str(array.dtype), sha256=digest(np.ascontiguousarray(array).tobytes()))


def read_joint_bank(bank, design, execution_plan):
    year = bank["year"]
    names = bank["coefficient_names"]
    require(
        bank["arm"] == "F0" and bank["joint_shape"] == [4, 4000, 6] and len(names) == 6 and len(set(names)) == 6,
        "complete reviewed six-column F0 bank required",
    )
    require(
        bank["target_race_ids"] == design["race_ids"] and bank["law_hash"] == design["law_hash"],
        "bank/design/event law mismatch",
    )
    require(design["coefficient_names"] == names, "ordered design/coefficient identity mismatch")
    require(
        digest(np.asarray(design["design"], float).tobytes()) == bank["target_design_sha256"]
        and digest(np.asarray(design["offset"], float).tobytes()) == bank["target_offset_sha256"],
        "exact admitted design bytes required",
    )
    require(all(c < year for c in bank["training_cycles"]), "future training-cycle metadata rejected")
    job = next(j for j in execution_plan["jobs"] if j["target"] == year and j["arm"] == "F0")
    require(
        job["source_binding"]["law_hash"] == bank["law_hash"] and job["source_binding"]["parameter_names"] == names,
        "economic raw-bank job identity mismatch",
    )
    directory = LAB / f"economic-execution-v1/attempt-v1/{year}/F0"
    terminal = trusted_json(directory / "terminal.json")
    require(
        terminal["numerical_passed"] is True and terminal["raw_chains"] == bank["raw_banks"],
        "accepted complete bank terminal mismatch",
    )
    values = {name: [] for name in POSTERIOR_VARIABLES}
    chain_pins = []
    for chain, bank_pin in enumerate(bank["raw_banks"]):
        sources.checked(bank_pin)
        receipt_path = directory / f"chain-{chain}-complete.json"
        receipt = trusted_json(receipt_path)
        require(
            receipt["raw_bank"] == bank_pin
            and receipt["chain"] == chain
            and receipt["job"] == [year, "F0"]
            and receipt["plan_sha256"] == BINDINGS["files"][8]["sha256"]
            and receipt["complete"] is True
            and receipt["seed"] == job["chain_seeds"][chain],
            "chain receipt/seed/job identity mismatch",
        )
        handle = az.from_netcdf(bank_pin["path"])
        try:
            post = handle.posterior
            require(
                set(post.data_vars) == set(POSTERIOR_VARIABLES), "exact complete posterior variable inventory required"
            )
            require(
                post.chain.values.tolist() == [chain]
                and post.draw.values.tolist() == list(range(4000))
                and handle.warmup_posterior.sizes["draw"] == 1500,
                "original complete chain/draw/warmup identity required",
            )
            require(
                post.coefficient.values.tolist() == names
                and post.source_cycle.values.tolist() == bank["training_cycles"],
                "ordered original coefficient/historical-cycle axes required",
            )
            for name, dims in POSTERIOR_VARIABLES.items():
                data = post[name]
                require(
                    data.dims == dims and data.sizes["chain"] == 1 and data.sizes["draw"] == 4000,
                    "full ordered posterior array axes required: " + name,
                )
                require(data.dtype == np.dtype("float64"), "original float64 posterior precision required")
                array = np.asarray(data, float)[0].copy()
                require(np.isfinite(array).all(), "finite complete original joint rows required")
                if name in ("tau_cycle", "tau_race"):
                    require((array > 0).all(), "positive original scales required")
                values[name].append(array)
        finally:
            handle.close()
        chain_pins.append(sources.pin(receipt_path))
    result = {name: freeze(np.concatenate(parts, axis=0)) for name, parts in values.items()}
    require(
        result["beta"].shape == (16000, 6) and result["tau_cycle"].shape == result["tau_race"].shape == (16000,),
        "all 16000 aligned parameter rows required",
    )
    require(
        np.allclose(result["cycle_effect"], result["tau_cycle"][:, None] * result["cycle_z"], atol=1e-14, rtol=1e-14),
        "historical cycle-effect parameter identity mismatch",
    )
    result["posterior_chain"] = freeze(np.repeat(np.arange(4, dtype=int), 4000))
    result["posterior_draw"] = freeze(np.tile(np.arange(4000, dtype=int), 4))
    return freeze(result), tuple(chain_pins)


def calibration_for(cycle, width):
    arm = "brownian_total_error_map_v1" if width == 5 else "brownian_plus_persistent_slope_map_v1"
    root = LAB / "momentum-calibration-execution-sol-v2/attempt-root-v2"
    fold_path = root / f"folds/{cycle}/{arm}.json"
    fold = trusted_json(fold_path)
    expected_target = 2022 if cycle == 2024 else cycle
    require(
        fold["target"] == cycle
        and fold["calibration_target"] == expected_target
        and fold["aliased"] == (cycle == 2024),
        "calibration fold/alias identity mismatch",
    )
    summary = json.loads(sources.checked(BINDINGS["files"][6]))
    accepted = next(r for r in summary["results"] if r["target"] == expected_target and r["arm"] == arm)
    require(accepted["terminal"] == fold["fit_terminal"], "accepted calibration terminal mismatch")
    terminal = json.loads(sources.checked(fold["fit_terminal"]))
    result = json.loads(sources.checked(terminal["result"]))
    require(
        terminal["passed"] is True and result["gate"]["passed"] is True and all(result["gate"]["checks"].values()),
        "all fixed calibration gates must pass",
    )
    require(
        result["metadata"]["components"] == list(MODEL.COMPONENTS[:width])
        and result["metadata"]["target"] == expected_target
        and result["metadata"]["input_identity_sha256"] == fold["input_identity_sha256"],
        "calibration component/input identity mismatch",
    )
    scales = np.asarray(result["fitted_scales"], float)
    require(
        scales.shape == (width,) and np.isfinite(scales).all() and (scales > 0).all(),
        "complete positive matched MAP scales required",
    )
    named = dict(zip(MODEL.COMPONENTS[:width], scales.tolist(), strict=True))
    if width == 5:
        named["persistent_linear"] = 0.0
    return freeze(named), freeze(
        dict(
            fold=sources.pin(fold_path),
            terminal=fold["fit_terminal"],
            result=terminal["result"],
            input_identity_sha256=fold["input_identity_sha256"],
            calibration_target=expected_target,
        )
    )


def class_pin(cls, expected_name, expected_pin):
    require(cls.__name__ == expected_name, "explicit native constructor name required")
    path = Path(inspect.getfile(cls)).resolve()
    require(path == Path(expected_pin["path"]).resolve(), "constructor module source identity mismatch")
    require(sources.same_pin(expected_pin, trusted_pin(path)), "constructor source not reviewed")
    sources.checked(expected_pin)
    require(
        getattr(sys.modules[cls.__module__], expected_name, None) is cls, "constructor module class identity mismatch"
    )
    return dict(class_name=cls.__name__, module=cls.__module__, source=expected_pin)


@dataclass(frozen=True)
class MeasurementInput:
    variant: str
    operator: str
    coordinates: object
    shares: np.ndarray
    likelihood_kwargs: object
    scales: object
    source_pins: object


@dataclass(frozen=True)
class F0MeasurementAdapter:
    cycle: int
    source_identity: str
    prior_kwargs: object
    original_parameter_rows: object
    uniform_prior_weights: np.ndarray
    measurements: object
    source_pins: object

    def build_prior(self, prior_class=native_basis.EmpiricalBasisFundamentals, constructor_pin=None):
        constructor_pin = constructor_pin or trusted_pin(OWNER / "src/forecasting/polling_basis_v1.py")
        binding = class_pin(prior_class, "EmpiricalBasisFundamentals", constructor_pin)
        instance = immutable_instance(prior_class(**dict(self.prior_kwargs)))
        require(
            instance.source_identity == self.source_identity and instance.beta.shape == (16000, 6),
            "constructed bank identity mismatch",
        )
        return instance, freeze(binding)

    def build_likelihood(self, variant, likelihood_class=None, constructor_pin=None):
        likelihood_class = likelihood_class or MODEL.INCUMBENT.GaussianPollLikelihood
        constructor_pin = constructor_pin or trusted_pin(OWNER / "src/forecasting/polling_v1.py")
        binding = class_pin(likelihood_class, "GaussianPollLikelihood", constructor_pin)
        require(variant in self.measurements, "selected measurement variant required")
        instance = immutable_instance(likelihood_class(**dict(self.measurements[variant].likelihood_kwargs)))
        return instance, freeze(binding)

    def build_for_sampler(self, sampler_module, variant, module_pin, dependency_pins):
        require(
            Path(sampler_module.__file__).resolve() == Path(module_pin["path"]).resolve(),
            "explicit sampler module identity required",
        )
        require(sources.same_pin(module_pin, trusted_pin(module_pin["path"])), "sampler source not reviewed")
        sources.checked(module_pin)
        prior_class = sampler_module.EmpiricalBasisFundamentals
        likelihood_class = sampler_module.GaussianPollLikelihood
        function = sampler_module.sample_poll_posterior
        require(
            Path(inspect.getfile(function)).resolve() == Path(module_pin["path"]).resolve(),
            "sampler function source mismatch",
        )
        require(
            function.__globals__.get("EmpiricalBasisFundamentals") is prior_class
            and function.__globals__.get("GaussianPollLikelihood") is likelihood_class,
            "sampler constructor globals disagree",
        )
        pins = {str(Path(item["path"]).resolve()): item for item in (module_pin, *dependency_pins)}
        for item in pins.values():
            sources.checked(item)
        prior_source = str(Path(inspect.getfile(prior_class)).resolve())
        like_source = str(Path(inspect.getfile(likelihood_class)).resolve())
        require(prior_source in pins and like_source in pins, "both concrete constructor source pins required")
        prior, prior_binding = self.build_prior(prior_class, pins[prior_source])
        likelihood, like_binding = self.build_likelihood(variant, likelihood_class, pins[like_source])
        require(
            isinstance(prior, function.__globals__["EmpiricalBasisFundamentals"])
            and isinstance(likelihood, function.__globals__["GaussianPollLikelihood"]),
            "sampler class identities do not match",
        )
        return (
            prior,
            likelihood,
            freeze(
                dict(
                    sampler_module=sampler_module.__name__,
                    sampler_source=module_pin,
                    prior=prior_binding,
                    likelihood=like_binding,
                    sampler_not_called=True,
                )
            ),
        )


def load_adapter(cycle):
    sources.verify()
    target_path = BINDINGS["files"][1]["path"]
    target_sha = BINDINGS["files"][1]["sha256"]
    _, fold, full = TARGET.load_prepared(target_path, target_sha, cycle, "native_full_series")
    design = json.loads(sources.checked(fold["target_design"]))
    bank = json.loads(sources.checked(fold["bank_reference"]))
    expected = next(b for b in COMPATIBILITY["banks"] if b["year"] == cycle and b["arm"] == "F0")
    require(all(bank[key] == expected[key] for key in expected), "exact accepted F0 bank record required")
    execution_plan = json.loads(sources.checked(BINDINGS["files"][8]))
    rows, chain_receipts = read_joint_bank(bank, design, execution_plan)
    identity = digest(
        canonical(
            dict(
                schema="ExactF0BankTargetSourceIdentityV1",
                law_hash=bank["law_hash"],
                raw_banks=bank["raw_banks"],
                coefficient_names=bank["coefficient_names"],
                training_cycles=bank["training_cycles"],
                target_race_ids=bank["target_race_ids"],
                target_design_sha256=bank["target_design_sha256"],
                target_offset_sha256=bank["target_offset_sha256"],
                target_plan_sha256=target_sha,
            )
        )
    )
    kwargs = freeze(
        dict(
            offset=np.asarray(design["offset"], float),
            design=np.asarray(design["design"], float),
            beta=rows["beta"],
            tau_cycle=rows["tau_cycle"],
            tau_race=rows["tau_race"],
            race_ids=tuple(bank["target_race_ids"]),
            source_identity=identity,
            basis_names=tuple(bank["coefficient_names"]),
            coefficient_names=tuple(bank["coefficient_names"]),
            coefficients=rows["beta"],
            posterior_chain=rows["posterior_chain"],
            posterior_draw=rows["posterior_draw"],
            race_df=5.0,
            metadata=dict(
                schema="ExactReviewedF0AdapterMetadataV1",
                law_hash=bank["law_hash"],
                target_cycle=cycle,
                cutoff=fold["cutoff_utc"],
                joint_rows=16000,
                initial_row_weight=1 / 16000,
                cycle_z_is_historical_only=True,
                empirical_update_target="marginal; not calibration cut",
            ),
        )
    )
    measurements = {}
    for variant, (operator, width) in VARIANTS.items():
        _, _, data = TARGET.load_prepared(target_path, target_sha, cycle, operator)
        coordinates = MODEL.PollCoordinatesV1(
            data.observation_ids, data.race_ids, data.cycles, data.pollster_keys, data.days_to_election
        )
        object.__setattr__(coordinates, "days_to_election", freeze(coordinates.days_to_election))
        scales, calibration = calibration_for(cycle, width)
        covariance = freeze(MODEL.error_covariance(coordinates, scales))
        like = freeze(
            dict(
                race_indices=np.asarray(data.race_indices, int),
                values=data.logits,
                covariance=covariance,
                observation_ids=data.observation_ids,
            )
        )
        measurements[variant] = MeasurementInput(
            variant,
            operator,
            coordinates,
            data.shares,
            like,
            scales,
            freeze(
                dict(
                    target_joint=fold["operators"][operator]["joint_input"],
                    calibration=calibration,
                    no_nominal_variance_added=True,
                    units="conditional D/R log odds; time=field-end days/30",
                )
            ),
        )
    require(
        measurements["FS5"].coordinates.observation_ids == measurements["FS6"].coordinates.observation_ids
        and np.array_equal(measurements["FS5"].shares, measurements["FS6"].shares),
        "matched FS5/FS6 observations required",
    )
    return F0MeasurementAdapter(
        cycle,
        identity,
        kwargs,
        rows,
        freeze(np.full(16000, 1 / 16000)),
        freeze(measurements),
        freeze(
            dict(
                target_plan=BINDINGS["files"][1],
                target_fold_design=fold["target_design"],
                bank_reference=fold["bank_reference"],
                raw_banks=bank["raw_banks"],
                chain_receipts=chain_receipts,
                law_hash=bank["law_hash"],
            )
        ),
    )


def read_archive_metadata(path):
    """NPY headers plus three identity vectors; never predictive numeric values."""
    headers = {}
    metadata = {}
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        expected = {
            key + ".npy"
            for key in ("shares", "logits", "new_cycle_z", "new_race_z", "posterior_draw", "replicate", "race_ids")
        }
        require(names == expected, "archived predictive array inventory mismatch")
        for name in sorted(names):
            with archive.open(name) as stream:
                version = np.lib.format.read_magic(stream)
                require(version in ((1, 0), (2, 0)), "unsupported archived NPY metadata version")
                shape, fortran, dtype = (
                    np.lib.format.read_array_header_1_0(stream)
                    if version == (1, 0)
                    else np.lib.format.read_array_header_2_0(stream)
                )
                require(not dtype.hasobject, "object-array archive rejected")
                headers[name[:-4]] = dict(shape=list(shape), fortran_order=fortran, dtype=str(dtype))
            if name[:-4] in {"race_ids", "posterior_draw", "replicate"}:
                with archive.open(name) as stream:
                    metadata[name[:-4]] = np.lib.format.read_array(stream, allow_pickle=False)
    return headers, metadata


def validate_archive_metadata(headers, metadata, race_ids):
    n = len(race_ids)
    expected_shapes = dict(
        shares=[4, 20000, n],
        logits=[4, 20000, n],
        new_cycle_z=[4, 20000],
        new_race_z=[4, 20000, n],
        posterior_draw=[20000],
        replicate=[20000],
        race_ids=[n],
    )
    require({k: v["shape"] for k, v in headers.items()} == expected_shapes, "archived predictive shape mismatch")
    require(
        metadata["race_ids"].tolist() == list(race_ids)
        and np.array_equal(metadata["posterior_draw"], np.repeat(np.arange(4000), 5))
        and np.array_equal(metadata["replicate"], np.tile(np.arange(5), 4000)),
        "archived event/parameter-row/replicate identity mismatch",
    )
    for name, header in headers.items():
        dtype = np.dtype(header["dtype"])
        expected_kind = "U" if name == "race_ids" else "i" if name in {"posterior_draw", "replicate"} else "f"
        require(dtype.kind == expected_kind and not header["fortran_order"], "archived dtype/order mismatch")
        if expected_kind != "U":
            require(dtype.itemsize == 8, "archived numeric precision mismatch")


def archive_reference(cycle, adapter):
    """Read byte hashes/NPY headers and identity arrays only, never probabilities."""
    root = LAB / f"economic-execution-v1/attempt-v1/{cycle}/F0"
    seal_path = root / "prediction-seal.json"
    seal = trusted_json(seal_path)
    require(
        seal["target"] == cycle
        and seal["arm"] == "F0"
        and seal["numerical_passed"] is True
        and seal["plan_sha256"] == BINDINGS["files"][8]["sha256"]
        and seal["source_law_hash"] == adapter.source_pins["law_hash"],
        "archived predictive source/law identity mismatch",
    )
    sources.checked(seal["predictions"])
    plan = json.loads(sources.checked(BINDINGS["files"][8]))
    launch = json.loads(sources.checked(BINDINGS["files"][9]))
    require(
        launch["plan_sha256"] == BINDINGS["files"][8]["sha256"] and plan["prediction"]["replicates_per_joint_row"] == 5,
        "archived predictive operator identity mismatch",
    )
    job = next(j for j in plan["jobs"] if j["target"] == cycle and j["arm"] == "F0")
    headers, metadata = read_archive_metadata(seal["predictions"]["path"])
    validate_archive_metadata(headers, metadata, adapter.prior_kwargs["race_ids"])
    design = json.loads(sources.checked(adapter.source_pins["target_fold_design"]))
    require(
        job["source_binding"]["law_hash"] == adapter.source_pins["law_hash"] and job["recipe"] == design["recipe"],
        "archived F0 recipe identity mismatch",
    )
    return dict(
        schema="ArchivedF0PredictiveReferenceIdentityV1",
        cycle=cycle,
        source_identity=adapter.source_identity,
        prediction_seal=sources.pin(seal_path),
        prediction_bytes=seal["predictions"],
        archive_headers=headers,
        identity_array_pins={k: array_pin(v) for k, v in metadata.items()},
        parameter_rows=16000,
        predictive_realizations=80000,
        replicates_per_parameter_row=5,
        representation="four original chains, 4000 posterior rows each, five saved future realizations per row",
        posterior_rows_replaced_by_predictive_arrays=False,
        probability_or_logit_values_read=False,
        numerical_probabilities_or_scores_summarized=False,
        execution_plan=BINDINGS["files"][8],
        prediction_seed=job["prediction_seed"],
        feature_design=adapter.source_pins["target_fold_design"],
        target_design_sha256=design["design_array_sha256"],
        target_offset_sha256=design["offset_array_sha256"],
        ordered_race_ids=list(adapter.prior_kwargs["race_ids"]),
        coefficient_names=list(adapter.prior_kwargs["coefficient_names"]),
        recipe_sha256=job["recipe_sha256"],
        generator_source=trusted_pin(LAB / "economic-execution-v1/runner.py"),
        forecast_design=BINDINGS["files"][12],
        identity_scope="sealed bytes, generator metadata/source, exact target law/features/axes; numeric predictive values not replayed",
    )


def report(output):
    sources.verify()
    output = Path(output).resolve()
    require(output.is_relative_to(HERE), "new isolated adapter report required")
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for cycle in (2018, 2020, 2022, 2024):
        data = load_adapter(cycle)
        prior, prior_binding = data.build_prior()
        require(
            np.array_equal(prior.posterior_chain, data.original_parameter_rows["posterior_chain"]),
            "native prior chain identity changed",
        )
        record = dict(
            cycle=cycle,
            prior_constructor_binding=prior_binding,
            source_identity=data.source_identity,
            source_pins=dict(data.source_pins),
            original_joint_array_pins={k: array_pin(v) for k, v in data.original_parameter_rows.items()},
            uniform_prior_weights=array_pin(data.uniform_prior_weights),
            target_design=array_pin(prior.design),
            target_offset=array_pin(prior.offset),
            variants={},
        )
        for variant, measurement in data.measurements.items():
            likelihood, binding = data.build_likelihood(variant)
            record["variants"][variant] = dict(
                operator=measurement.operator,
                rows=len(measurement.shares),
                scales=dict(measurement.scales),
                source_pins=dict(measurement.source_pins),
                observation_ids=list(measurement.coordinates.observation_ids),
                covariance=array_pin(likelihood.covariance),
                constructor_binding=dict(binding),
            )
        archived = archive_reference(cycle, data)
        write_json(output / f"{cycle}-adapter.json", record)
        write_json(output / f"{cycle}-archived-F0-reference.json", archived)
        records.append(
            dict(
                cycle=cycle,
                adapter=sources.pin(output / f"{cycle}-adapter.json"),
                archived_F0_reference=sources.pin(output / f"{cycle}-archived-F0-reference.json"),
            )
        )
    write_json(
        output / "receipt.json",
        dict(
            schema="MomentumForecastAdapterReceiptV1",
            folds=records,
            target_plan=BINDINGS["files"][1],
            complete_joint_rows=16000,
            bank_arrays_exported=False,
            Gibbs_transitions=False,
            optimizer=False,
            numeric_target_labels=False,
            forecast_summaries=False,
            scores=False,
            constructor_binding="explicit verified class/module identities required for private sampler use",
            forecast_design=BINDINGS["files"][12],
            PO5="Same FS5 likelihood only; independent N(0,6.25) final-logit comparator not computed",
        ),
    )
    print(json.dumps(sources.pin(output / "receipt.json")))


def jsonable(value):
    if isinstance(value, (dict, MappingProxyType)):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "verified-v1")
    report(parser.parse_args().output)
