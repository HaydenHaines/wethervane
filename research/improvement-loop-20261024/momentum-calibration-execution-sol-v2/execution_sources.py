"""Source/runtime authentication. No numerical imports or fitting here."""

import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DESIGN_SHA = "dede87ca1843d8d98fd569effe3ddc81f84ccbb5111486ea3d1b19508b541f62"
BINDINGS_SHA = "2c1ac1d9c44ce8af3586e3ea2adab196cc33da9a1f6441934616814a2b6ec7c3"
RUNTIME_SHA = "f78840fc8096c201e315ec62f95cb6485d208695bcdf2a8089910859cdfa651a"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def checked(item):
    data = Path(item["path"]).read_bytes()
    require(hashlib.sha256(data).hexdigest() == item["sha256"], "fingerprint mismatch: " + item["path"])
    if "bytes" in item:
        require(len(data) == item["bytes"], "source size mismatch")
    return data


def same_pin(a, b):
    return Path(a["path"]).resolve() == Path(b["path"]).resolve() and a["sha256"] == b["sha256"]


def verify_manifest(path, expected):
    path = Path(path)
    manifest = json.loads(checked(dict(path=str(path), sha256=expected)))
    for item in manifest["files"]:
        source = (path.parent / item["path"]).resolve()
        require(source.is_relative_to(path.parent), "manifest source escaped packet")
        checked({**item, "path": str(source)})
    return manifest


def runtime_identity():
    return dict(
        schema="MomentumCalibrationNumericalRuntimeV1",
        python_executable=os.path.abspath(sys.executable),
        prefix=sys.prefix,
        python=sys.version,
        packages={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "pandas")},
    )


def verify_sources(directory=HERE):
    directory = Path(directory)
    bindings = json.loads(checked(dict(path=str(directory / "source-bindings.json"), sha256=BINDINGS_SHA)))
    require(len(bindings["files"]) == 8, "complete source inventory required")
    for i, item in enumerate(bindings["files"]):
        checked(item)
        checked(dict(path=str(directory / "snapshots" / f"{i:02d}-{Path(item['path']).name}"), sha256=item["sha256"]))
    design = json.loads(checked(bindings["files"][0]))
    require(bindings["files"][0]["sha256"] == DESIGN_SHA, "wrong execution design")
    require(
        all(same_pin(a, b) for a, b in zip(bindings["files"][1:6], design["references"], strict=True)),
        "design source refs differ",
    )
    verify_manifest(design["references"][0]["path"], design["references"][0]["sha256"])
    verify_manifest(design["references"][2]["path"], design["references"][2]["sha256"])
    for reference in (design["references"][3], bindings["files"][-2]):
        receipt = json.loads(checked(reference))
        parent = Path(reference["path"]).parent
        for item in receipt["files"]:
            checked({**item, "path": str(parent / item["path"])})
    review = json.loads(checked(bindings["files"][-2]))
    require(
        review["disposition"] == "accepted_earlier_only_calibration_preparation"
        and review["reviewed_manifest_sha256"] == design["references"][0]["sha256"]
        and review["reviewed_plan_sha256"] == design["references"][1]["sha256"],
        "accepted exact preparation required",
    )
    correction = json.loads(checked(bindings["files"][-1]))
    checked(
        {**correction["note"], "path": str(Path(bindings["files"][-1]["path"]).parent / correction["note"]["path"])}
    )
    # Authenticate original scientific dependencies before preparation imports.
    prep = Path(design["references"][0]["path"]).parent
    for item in json.loads((prep / "source-bindings.json").read_bytes())["files"]:
        checked(item)
    prototype = Path(design["references"][2]["path"]).parent
    original = json.loads((prototype / "source-bindings.json").read_bytes())
    checked(original["selection"])
    for item in original["sources"]:
        checked(item)
    runtime = json.loads(checked(dict(path=str(directory / "runtime.json"), sha256=RUNTIME_SHA)))
    require(runtime == runtime_identity(), "exact interpreter path/prefix and scientific runtime required")
    return design, bindings


def verify_execution(receipt_pin, output):
    design, bindings = verify_sources()
    receipt = json.loads(checked(receipt_pin))
    require(receipt["schema"] == "MomentumCalibrationExecutionReceiptV1", "wrong execution receipt schema")
    require(isinstance(receipt["run_id"], str) and bool(receipt["run_id"].strip()), "unique run_id required")
    require(same_pin(receipt["design"], bindings["files"][0]), "execution design differs")
    require(same_pin(receipt["preparation_acceptance"], bindings["files"][-2]), "exact preparation acceptance required")
    checked(receipt["preparation_acceptance"])
    require(
        same_pin(receipt["runtime"], dict(path=str(HERE / "runtime.json"), sha256=RUNTIME_SHA)),
        "execution runtime differs",
    )
    checked(receipt["runtime"])
    require(
        Path(receipt["runner_manifest"]["path"]).resolve() == HERE / "artifact-manifest.json",
        "wrong runner manifest path",
    )
    manifest = verify_manifest(receipt["runner_manifest"]["path"], receipt["runner_manifest"]["sha256"])
    require(
        manifest["execution_design_sha256"] == DESIGN_SHA and manifest["source_bindings_sha256"] == BINDINGS_SHA,
        "runner source/contract identity differs",
    )
    output = Path(output).resolve()
    require(
        output.is_relative_to(HERE) and output == Path(receipt["output_directory"]).resolve(),
        "bound exclusive output directory required",
    )
    require(
        all(
            os.environ.get(key) == "1"
            for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        ),
        "single-thread numerical runtime required",
    )
    return receipt, design
