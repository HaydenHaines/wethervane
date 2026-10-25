"""Pure-stdlib byte/contract delta audit; no numerical execution."""

import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
V1 = LAB / "momentum-calibration-execution-sol-v1"
V2 = LAB / "momentum-calibration-execution-sol-v2"
pins = {}


def body(path, expected=None):
    data = path.read_bytes()
    value = hashlib.sha256(data).hexdigest()
    if expected:
        assert value == expected, str(path)
    pins[str(path)] = value
    return data


for folder, expected in [
    (V1, "fcaccb33a483045e16d179ce5266370520db853f4c997cb3e0e5afcb4c991ef2"),
    (V2, "583ca465e8fea568843486542ad8d85832585661919aae974bcfb5c103e67160"),
]:
    manifest = json.loads(body(folder / "artifact-manifest.json", expected))
    for item in manifest["files"]:
        data = body(folder / item["path"], item["sha256"])
        assert len(data) == item["bytes"]
delta = json.loads(body(V2 / "byte-delta.json", "c44b928f27980610e2e9d74cd36d4c85c6b5e4743832dcf3c8e8340b927569dd"))
d1 = json.loads(
    body(
        LAB / "checkpoints/momentum-calibration-execution-design-v1.json",
        "5c13a5d7552fdfe893bb0548a9205688c9a921a5d9faf2fd90147d5ff1cf5234",
    )
)
d2 = json.loads(
    body(
        LAB / "checkpoints/momentum-calibration-execution-design-v2.json",
        "dede87ca1843d8d98fd569effe3ddc81f84ccbb5111486ea3d1b19508b541f62",
    )
)
changed = sorted(k for k in set(d1) | set(d2) if d1.get(k) != d2.get(k))
assert changed == sorted(
    [
        "optimizer",
        "output_directory",
        "predecessor",
        "schema",
        "refinement_reason",
        "recorded_utc",
        "execution_contract",
    ]
)
optimizer = json.loads(json.dumps(d1["optimizer"]))
optimizer["options"]["ftol"] = 1e-14
assert d2["optimizer"] == optimizer and d1["optimizer"]["options"]["ftol"] == 1e-11
for name in ["runner.py", "test_runner.py", "runtime.json", "supplied-real-point-checks-v1.json"]:
    assert body(V1 / name) == body(V2 / name)
old = body(V1 / "execution_sources.py").decode()
new = body(V2 / "execution_sources.py").decode()


def normalize(text):
    return re.sub(r'(?m)^(DESIGN_SHA|BINDINGS_SHA) = "[a-f0-9]{64}"', r'\1 = "NORMALIZED"', text)


assert normalize(old) == normalize(new)
b1 = json.loads(body(V1 / "source-bindings.json"))
b2 = json.loads(body(V2 / "source-bindings.json"))
assert b1["files"][1:] == b2["files"][1:] and len(b2["files"]) == 8
assert {k: v for k, v in b1.items() if k != "files"} == {k: v for k, v in b2.items() if k != "files"}
for i, item in enumerate(b2["files"]):
    body(Path(item["path"]), item["sha256"])
    body(V2 / "snapshots" / f"{i:02d}-{Path(item['path']).name}", item["sha256"])
    if i:
        assert body(V1 / "snapshots" / f"{i:02d}-{Path(item['path']).name}") == body(
            V2 / "snapshots" / f"{i:02d}-{Path(item['path']).name}"
        )
v1review = json.loads(
    body(
        LAB / "momentum-calibration-results-review-sol-v1/receipt.json",
        "76e224ced1ccd4eec2a6a591a628db86312e8a7023f9e0f01b3aba9dbc7c4a19",
    )
)
for item in v1review["files"]:
    body(LAB / "momentum-calibration-results-review-sol-v1" / item["path"], item["sha256"])
original = json.loads((LAB / "momentum-calibration-results-review-sol-v1/results.json").read_bytes())
for path, expected in original["inspected_pins"].items():
    body(Path(path), expected)
result = dict(
    v1_manifest_sha256=pins[str(V1 / "artifact-manifest.json")],
    v2_manifest_sha256=pins[str(V2 / "artifact-manifest.json")],
    byte_delta_sha256=pins[str(V2 / "byte-delta.json")],
    design_changed_fields=changed,
    optimizer_change_only_ftol=True,
    ftol_old=1e-11,
    ftol_new=1e-14,
    runner_test_runtime_supplied_points_byte_identical=True,
    source_guard_only_two_binding_constants=True,
    remaining_seven_refs_snapshots_identical=True,
    all_scientific_and_gate_fields_unchanged=True,
    original_failure_review_and_outputs_unchanged=True,
    inspected_pins=pins,
    optimizer=False,
    sampler=False,
    forecast_scores=False,
)
(HERE / "delta-results.json").write_text(json.dumps(result, indent=2) + "\n")
print(
    json.dumps(
        {
            k: result[k]
            for k in [
                "optimizer_change_only_ftol",
                "source_guard_only_two_binding_constants",
                "remaining_seven_refs_snapshots_identical",
                "all_scientific_and_gate_fields_unchanged",
                "original_failure_review_and_outputs_unchanged",
            ]
        }
    )
)
