"""Authenticate selected sources and reviewed dependencies before imports."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT_SHA = "f30f0a92920647276afa44f741515ff0cebfe5fe18417e273ae975d6133469e9"
BINDINGS_SHA = "b774844876026bda2544eb140d7bb07f8b1b617ef791ff1387dcd030c3e09d7f"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(item):
    data = Path(item["path"]).read_bytes()
    require(hashlib.sha256(data).hexdigest() == item["sha256"], "source fingerprint mismatch: " + item["path"])
    if "bytes" in item:
        require(len(data) == item["bytes"], "source length mismatch")
    return data


def verify(directory=HERE):
    directory = Path(directory)
    bindings = json.loads(checked(dict(path=str(directory / "source-bindings.json"), sha256=BINDINGS_SHA)))
    require(bindings["root_selection_sha256"] == ROOT_SHA and len(bindings["files"]) == 26, "wrong source inventory")
    for index, item in enumerate(bindings["files"]):
        checked(item)
        checked(
            dict(path=str(directory / "snapshots" / f"{index:02d}-{Path(item['path']).name}"), sha256=item["sha256"])
        )
    selection = json.loads(checked(bindings["files"][0]))
    require(bindings["files"][0]["sha256"] == ROOT_SHA, "wrong root selection")
    require(
        [{k: i[k] for k in ("path", "sha256")} for i in bindings["files"][1:5]] == selection["references"],
        "root reference identity changed",
    )
    prototype = Path(selection["references"][1]["path"]).parent
    for item in json.loads((prototype / "artifact-manifest.json").read_bytes())["files"]:
        checked({**item, "path": str(prototype / item["path"])})
    original = json.loads((prototype / "source-bindings.json").read_bytes())
    checked(original["selection"])
    for item in original["sources"]:
        checked(item)
    review_path = Path(selection["references"][2]["path"])
    review = json.loads(review_path.read_bytes())
    require(review["disposition"] == "accepted_supplied_coordinate_mathematics_only", "accepted mathematics required")
    for item in review["files"]:
        checked({**item, "path": str(review_path.parent / item["path"])})
    coverage = HERE.parent / "momentum-preflight-coverage-luna-v1"
    authority = json.loads(
        checked(dict(path=str(coverage / "authoritative-index-v2.json"), sha256=bindings["coverage_authority_sha256"]))
    )
    require(
        authority["authoritative_tables"]["complete_earlier_cycle_baseline_observation_id_ledger"]
        == "historical-calibration-baseline-id-ledger-v2.csv",
        "corrected complete coverage ledger required",
    )
    correction = json.loads(
        checked(
            dict(
                path=str(coverage / "artifact-correction-manifest-v2.json"),
                sha256=bindings["coverage_correction_sha256"],
            )
        )
    )
    for item in correction["files"]:
        path = (coverage / item["path"]).resolve()
        require(path.is_relative_to(coverage), "coverage path escaped packet")
        checked({**item, "path": str(path)})
    coverage_review_path = Path(bindings["files"][-1]["path"])
    coverage_review = json.loads(checked(bindings["files"][-1]))
    require(
        coverage_review["disposition"] == "accepted_versioned_corrections_for_no_fit_preparation"
        and coverage_review["reviewed_final_manifest_sha256"] == bindings["coverage_correction_sha256"],
        "accepted corrected coverage required",
    )
    for item in coverage_review["files"]:
        checked({**item, "path": str(coverage_review_path.parent / item["path"])})
    return selection, bindings
