"""Exact target-input source authentication; no scientific imports or labels."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SELECTION_SHA = "59dbe9ffa8098b1582eff2a6998452825428b2f7eb738a31f0e8ac66555f99f1"
BINDINGS_SHA = "c91871309b437fc5c24043b547ce1ad18c71f945b13143009c5ac1c3ecc769ef"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def checked(item):
    body = Path(item["path"]).read_bytes()
    require(hashlib.sha256(body).hexdigest() == item["sha256"], "source binding drift: " + item["path"])
    if "bytes" in item:
        require(len(body) == item["bytes"], "source byte length drift")
    return body


def listed_receipt(item):
    document = json.loads(checked(item))
    parent = Path(item["path"]).parent
    for row in document["files"]:
        path = (parent / row["path"]).resolve()
        require(path.is_relative_to(parent), "receipt path escaped packet")
        checked({**row, "path": str(path)})
    return document


def verify(directory=HERE):
    directory = Path(directory)
    bindings = json.loads(checked(dict(path=str(directory / "source-bindings.json"), sha256=BINDINGS_SHA)))
    require(
        bindings["root_selection_sha256"] == SELECTION_SHA and len(bindings["files"]) == 34, "wrong source inventory"
    )
    for index, item in enumerate(bindings["files"]):
        checked(item)
        checked(
            dict(path=str(directory / "snapshots" / f"{index:02d}-{Path(item['path']).name}"), sha256=item["sha256"])
        )
    selection = json.loads(checked(bindings["files"][0]))
    require(bindings["files"][0]["sha256"] == SELECTION_SHA, "wrong target selection")
    require(
        [{k: r[k] for k in ("path", "sha256")} for r in bindings["files"][1:8]] == selection["references"],
        "root reference identity drift",
    )
    listed_receipt(selection["references"][1])
    listed_receipt(selection["references"][2])
    compatibility = json.loads(checked(selection["references"][3]))
    for item in compatibility["inspected_pins"]:
        checked(item)
    listed_receipt(bindings["files"][9])
    review = listed_receipt(bindings["files"][8])
    require(
        review["disposition"] == "accepted_all_fixed_fit_gates_with_retained_failed_start"
        and review["summary_sha256"] == selection["references"][6]["sha256"],
        "accepted calibration result identity required",
    )
    summary = json.loads(checked(selection["references"][6]))
    require(summary["all_calibration_gates_passed"] is True, "accepted calibration gates required")
    for row in summary["results"]:
        terminal = json.loads(checked(row["terminal"]))
        checked(terminal["result"])
        require(terminal["passed"] is True, "calibration result drift")
    return selection, compatibility, bindings
