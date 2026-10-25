"""Authenticate complete execution sources/runtime before scientific imports."""

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
# Exact accepted source/runtime/plan bindings.
BINDINGS_SHA = "b87f10262822c57a4a733a130a5ccc73e858d312898e9c8d4ed802d11f94934a"
PLAN_SHA = "103cd92c48bccdb388f25b0b58aa15efb938b5282824419c615dbf85ea24c755"
RUNTIME_SHA = "01ff4314a314fcb315719c0efb30062d9ee81878fa12ae428f7f1645f56cd354"
CLARIFICATION_SHA = "0d6dd97153687df2583c0a086e92e8a42c22eb0c850ed0a0133acdcb87c14024"
DESIGN_SHA = "45707d4a933ded2f7d4bffcbaab0fb6fb9a4ab7b1d62ad4494a27422b7d75678"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def checked(item):
    data = Path(item["path"]).read_bytes()
    require(hashlib.sha256(data).hexdigest() == item["sha256"], "source byte drift: " + item["path"])
    if "bytes" in item:
        require(len(data) == item["bytes"], "source length drift")
    return data


def json_pin(item):
    return json.loads(checked(item))


def inventory(item):
    value = json_pin(item)
    root = Path(item["path"]).parent
    for row in value["files"]:
        path = (root / row["path"]).resolve()
        require(path.is_relative_to(root), "manifest dependency escaped source packet")
        checked({**row, "path": str(path)})
    return value


def runtime():
    return dict(
        schema="MomentumForecastAdapterRuntimeV1",
        executable=os.path.abspath(sys.executable),
        prefix=sys.prefix,
        python=sys.version,
        packages={
            n: importlib.metadata.version(n)
            for n in ("numpy", "scipy", "pandas", "pydantic", "arviz", "h5py", "h5netcdf")
        },
    )


def verify_runtime():
    expected = json_pin(dict(path=str(HERE / "runtime.json"), sha256=RUNTIME_SHA))
    require(runtime() == expected, "exact execution interpreter/prefix/package versions required")
    return expected


def verify_sources():
    bindings = json_pin(dict(path=str(HERE / "source-bindings.json"), sha256=BINDINGS_SHA))
    require(bindings["status"] == "frozen_accepted_dependencies", "trace/adapter acceptance required before execution")
    for row in bindings["sources"]:
        checked(row)
        checked(dict(path=str(HERE / "snapshots" / row["snapshot"]), sha256=row["sha256"]))
        if row.get("inventory"):
            inventory(row)
    require(len(bindings["sources"]) == 13, "complete selected dependency inventory required")
    selected = bindings["sources"]
    require(
        selected[0]["sha256"] == DESIGN_SHA and selected[10]["sha256"] == CLARIFICATION_SHA,
        "exact scientific selection and numerical clarification required",
    )
    for review_index, producer_index, field, disposition in (
        (2, 1, "manifest_sha256", "accepted_read_only_complete_F0_and_H14_measurement_adapter"),
        (4, 3, "producer_manifest_sha256", "accepted_synthetic_observation_parity_only"),
        (9, 8, "producer_manifest_sha256", "accepted_synthetic_trace_collection_and_selected_diagnostics_only"),
        (
            12,
            11,
            "producer_manifest_sha256",
            "accepted_supplied_paired_tail_arithmetic_and_recoverable_input_validation",
        ),
    ):
        review = json_pin(selected[review_index])
        require(
            review["disposition"] == disposition and review[field] == selected[producer_index]["sha256"],
            "exact accepted producer/reviewer pair required",
        )
    marginal = json_pin(selected[7])
    require(
        marginal["verdict"] == "accepted_supplied_arithmetic_only"
        and marginal["reviewed_manifest_sha256"] == selected[6]["sha256"],
        "accepted exact marginal math required",
    )
    paired_review = json_pin(selected[12])
    require(paired_review["clarification_sha256"] == CLARIFICATION_SHA, "paired clarification mismatch")
    checked(paired_review["original_v1_failure_review"])
    verify_runtime()
    plan = json_pin(dict(path=str(HERE / "plan.json"), sha256=PLAN_SHA))
    require(plan["forecast_design"]["sha256"] == DESIGN_SHA, "selected forecast design mismatch")
    return bindings, plan


def module(name, path, replacements=None):
    # This function is called only after source inventory verification. Explicit
    # dependency identities are scoped and restored, never selected by sys.path.
    old = {key: sys.modules.get(key) for key in replacements or {}}
    for key, value in (replacements or {}).items():
        sys.modules[key] = value
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        value = importlib.util.module_from_spec(spec)
        sys.modules[name] = value
        spec.loader.exec_module(value)
        return value
    finally:
        for key, value in old.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


def load_dependencies():
    bindings, plan = verify_sources()
    # Adapter's own source guard authenticates all native inputs before imports.
    adapter_root = LAB / "momentum-forecast-adapter-sol-v1"
    guard = module("execution_adapter_sources", adapter_root / "adapter_sources.py")
    adapter = module("execution_adapter", adapter_root / "adapter.py", {"adapter_sources": guard})
    sampler_root = LAB / "momentum-observed-sampler-sol-v1"
    response = module("execution_response", sampler_root / "snapshots/src/forecasting/response_v5_reuse.py")
    replacements = {"src.forecasting.response_v5_reuse": response}
    polling = module("execution_polling", sampler_root / "snapshots/src/forecasting/polling_v1.py", replacements)
    replacements["src.forecasting.polling_v1"] = polling
    sampler = module("execution_observed_sampler", sampler_root / "observed_sampler.py", replacements)
    replacements["observed_sampler"] = sampler
    trace = module("execution_trace", LAB / "momentum-trace-diagnostics-sol-v1/trace_diagnostics.py", replacements)
    math = module("execution_marginal", LAB / "momentum-marginal-math-root-v2/marginal.py")
    replacements["trace_diagnostics"] = trace
    paired = module(
        "execution_paired_tail", LAB / "momentum-paired-tail-diagnostics-sol-v2/paired_tail.py", replacements
    )
    return dict(
        adapter=adapter,
        sampler=sampler,
        trace=trace,
        math=math,
        paired=paired,
        response=response,
        bindings=bindings,
        plan=plan,
    )


def verify_execution_manifest(expected):
    manifest = inventory(dict(path=str(HERE / "artifact-manifest.json"), sha256=expected))
    required = {
        "runner.py",
        "numerical.py",
        "references.py",
        "execution_sources.py",
        "test_execution.py",
        "source-bindings.json",
        "runtime.json",
        "plan.json",
    }
    require(required <= {row["path"] for row in manifest["files"]}, "complete execution code inventory required")
    return manifest
