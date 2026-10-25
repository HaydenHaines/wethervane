"""Authenticate exact selected source bytes before importing scientific code."""

import hashlib
import importlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OWNER = Path("/home/hayden/projects/wethervane-redesign-20261006/integration")
SELECTION_SHA = "0b28ea26002ba2b46ad4862a4028e285a6a0de2dcfff03830376ed4ffdfc3a4f"
BINDINGS_SHA = "c3a38476aee69845065657595e89a68f664751e94d5b1d1aafc548f22621e1d3"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked(path, expected):
    body = Path(path).read_bytes()
    require(hashlib.sha256(body).hexdigest() == expected, "source fingerprint mismatch: " + str(path))
    return body


def verify(directory=HERE):
    directory = Path(directory)
    bindings = json.loads(checked(directory / "source-bindings.json", BINDINGS_SHA))
    require(bindings["schema"] == "MomentumAnalyticPrototypeSourceBindingsV1", "wrong source binding schema")
    require(bindings["selection"]["sha256"] == SELECTION_SHA, "wrong root selection")
    selection = json.loads(checked(bindings["selection"]["path"], SELECTION_SHA))
    require(
        bindings["sources"][:5] == selection["source_pins"] and len(bindings["sources"]) == 7,
        "complete selected and transitive source inventory required",
    )
    for index, item in enumerate([bindings["selection"]] + bindings["sources"]):
        checked(item["path"], item["sha256"])
        checked(directory / "snapshots" / f"{index:02d}-{Path(item['path']).name}", item["sha256"])
    return bindings


def incumbent():
    verify()
    if str(OWNER) not in sys.path:
        sys.path.insert(0, str(OWNER))
    module = importlib.import_module("src.forecasting.poll_covariance_v1")
    require(
        Path(module.__file__).resolve() == OWNER / "src/forecasting/poll_covariance_v1.py",
        "wrong incumbent module loaded",
    )
    return module
