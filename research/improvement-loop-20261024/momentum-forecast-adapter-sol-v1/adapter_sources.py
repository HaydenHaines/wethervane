"""Source/runtime checks and explicit scoped module dependencies; no model calls."""

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
BINDINGS_SHA = "ef76d6087a325f85d4b5c7599a7aa8a7c43ba4b03258ad2fbe9344ad9683433a"
RUNTIME_SHA = "01ff4314a314fcb315719c0efb30062d9ee81878fa12ae428f7f1645f56cd354"
LOCK = threading.RLock()


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def checked(item):
    data = Path(item["path"]).read_bytes()
    require(hashlib.sha256(data).hexdigest() == item["sha256"], "source drift: " + item["path"])
    if "bytes" in item:
        require(len(data) == item["bytes"], "source length drift")
    return data


def same_pin(a, b):
    return Path(a["path"]).resolve() == Path(b["path"]).resolve() and a["sha256"] == b["sha256"]


def listed(item):
    value = json.loads(checked(item))
    parent = Path(item["path"]).parent
    for row in value["files"]:
        path = (parent / row["path"]).resolve()
        require(path.is_relative_to(parent), "source escaped sealed packet")
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


def verify(directory=HERE):
    directory = Path(directory)
    bindings = json.loads(checked(dict(path=str(directory / "source-bindings.json"), sha256=BINDINGS_SHA)))
    require(len(bindings["files"]) == 14, "complete adapter source inventory required")
    for index, item in enumerate(bindings["files"]):
        checked(item)
        checked(
            dict(path=str(directory / "snapshots" / f"{index:02d}-{Path(item['path']).name}"), sha256=item["sha256"])
        )
    listed(bindings["files"][0])
    target_review = listed(bindings["files"][2])
    require(
        target_review["disposition"] == "accepted_H14_target_inputs_and_feature_axes_only"
        and target_review["plan_sha256"] == bindings["files"][1]["sha256"],
        "accepted exact target input packet required",
    )
    compatibility = json.loads(checked(bindings["files"][3]))
    for row in compatibility["inspected_pins"]:
        checked(row)
    listed(bindings["files"][4])
    listed(bindings["files"][5])
    review = listed(bindings["files"][7])
    require(
        review["disposition"] == "accepted_all_fixed_fit_gates_with_retained_failed_start"
        and review["summary_sha256"] == bindings["files"][6]["sha256"],
        "accepted calibration result identity required",
    )
    results = json.loads((Path(bindings["files"][7]["path"]).parent / "results.json").read_bytes())
    for path, digest in results["inspected_pins"].items():
        checked(dict(path=path, sha256=digest))
    direct = json.loads(checked(bindings["files"][13]))
    for row in direct["files"]:
        checked(row)
    plan = json.loads(checked(bindings["files"][8]))
    for row in plan["source_pins"]:
        checked(row)
    design = json.loads(checked(bindings["files"][12]))
    require(
        set(design["comparison"]["arms"]) == {"FS5", "FS6", "LP5", "F0_archived_predictive", "PO5"},
        "exact selected arms required",
    )
    for row in design["references"]:
        checked({**row, "path": str(directory.parent / row["path"])})
    # Authenticate original target/prototype dependencies before importing them.
    for index in (0, 5):
        parent = Path(bindings["files"][index]["path"]).parent
        original = json.loads((parent / "source-bindings.json").read_bytes())
        for row in original.get("files", original.get("sources", [])):
            checked(row)
    seal = json.loads(checked(bindings["files"][10]))
    require(seal["manifest_sha256"] == bindings["files"][11]["sha256"], "observed constructor source seal mismatch")
    listed(bindings["files"][11])
    expected = json.loads(checked(dict(path=str(directory / "runtime.json"), sha256=RUNTIME_SHA)))
    require(runtime() == expected, "exact bank-loading interpreter/runtime required")
    return bindings, compatibility


def load_bound_module(name, item, substitutions=None):
    """Bind dependency identities explicitly, restoring unrelated module entries."""
    checked(item)
    with LOCK:
        previous = {key: sys.modules.get(key) for key in substitutions or {}}
        for key, module in (substitutions or {}).items():
            sys.modules[key] = module
        try:
            spec = importlib.util.spec_from_file_location(name, item["path"])
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module
        finally:
            for key, value in previous.items():
                if value is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = value


def load_guarded(name, path):
    path = Path(path)
    guard = load_bound_module(name + "_source_guard", pin(path.parent / "source_guard.py"))
    return load_bound_module(name, pin(path), {"source_guard": guard})
