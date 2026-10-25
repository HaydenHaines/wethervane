"""Read-only native checks and small synthetic adapter counterexamples.

The fixture uses inert placeholder pickle bytes. Nothing is unpickled or executed.
All writes stay under this review directory.
"""

import copy
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
INTEGRATION = Path("/home/hayden/projects/wethervane-redesign-20261006/integration")
PREFIX = Path("research/senate-stack-integration-20261020/partial-affine-completion-root-v1")
sys.path.insert(0, str(LAB))
from run_receipts import SCOPE_KEYS, inspect_completed_failure  # noqa: E402


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(obj):
    return (json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n").encode()


def source_json(name):
    return json.loads((INTEGRATION / PREFIX / name).read_bytes())


def put(root, name, raw):
    path = root / PREFIX / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return digest(raw)


def fixture(name, mutate=None, terminal_elsewhere=False):
    root = HERE / "fixtures" / name
    plan = source_json("plan.json")
    failure = source_json("attempt1/failure.json")
    fit = source_json("attempt1/fit/failure.json")
    result = source_json("attempt1/fit/parent-0-result.json")
    archive = source_json("attempt1/fit/parent-0-returned-inference.json")
    terminal = source_json("terminal-root-v1.json")
    put(root, "plan.json", (INTEGRATION / PREFIX / "plan.json").read_bytes())
    put(root, "downstream-admission.json", (INTEGRATION / PREFIX / "downstream-admission.json").read_bytes())
    put(root, "attempt1/failure.json", (INTEGRATION / PREFIX / "attempt1/failure.json").read_bytes())
    pickle_hash = put(root, "attempt1/fit/parent-0-returned-inference.pkl", b"INERT PLACEHOLDER: NEVER UNPICKLE\n")
    archive["files"] = {"parent-0-returned-inference.pkl": pickle_hash}
    result["returned_inference_archive"] = copy.deepcopy(archive)
    fit["complete_returned_inference_archive"] = copy.deepcopy(archive)
    if mutate:
        mutate(terminal, fit, result)
    fit["completed_parents"] = [copy.deepcopy(result)]
    hashes = {}
    names = {
        "fit/parent-0-returned-inference.json": encoded(archive),
        "fit/parent-0-result.json": encoded(result),
        "fit/failure.json": encoded(fit),
    }
    for name_, raw in names.items():
        hashes[name_] = put(root, "attempt1/" + name_, raw)
    for name_ in ["fit/parent-0-diagnostics.csv", "fit/parent-0-execution.json", "fit/parent-0-sampling-budget.json"]:
        hashes[name_] = put(root, "attempt1/" + name_, (INTEGRATION / PREFIX / "attempt1" / name_).read_bytes())
    for name_ in set(fit["observer_inventory"]["files"]) | set(fit["progress_files"]):
        put(root, "attempt1/fit/" + name_, (INTEGRATION / PREFIX / "attempt1/fit" / name_).read_bytes())
    hashes["fit/parent-0-returned-inference.pkl"] = pickle_hash
    hashes["failure.json"] = digest((INTEGRATION / PREFIX / "attempt1/failure.json").read_bytes())
    for pin in terminal["preserved_files"]:
        path = Path(pin["path"])
        pin["sha256"] = hashes[str(path.relative_to(PREFIX / "attempt1"))]
    terminal_name = "terminal-root-v1.json" if not terminal_elsewhere else "other-attempt/terminal-root-v1.json"
    terminal_hash = put(root, terminal_name, encoded(terminal))
    pin = {"path": str(PREFIX / terminal_name), "sha256": terminal_hash}
    try:
        report, inspected = inspect_completed_failure(
            root, pin, failure["plan"]["sha256"], {key: plan[key] for key in SCOPE_KEYS}
        )
    except Exception as exc:
        return {"accepted": False, "terminal_sha256": terminal_hash,
                "exception": type(exc).__name__, "reason": str(exc)}
    return {"accepted": True, "terminal_sha256": terminal_hash, "pins": len(inspected.pins),
            "qualification": report["qualification"], "numerical_fit": report["dimensions"]["numerical_fit"]}


def native():
    terminal_raw = (INTEGRATION / PREFIX / "terminal-root-v1.json").read_bytes()
    terminal = json.loads(terminal_raw)
    plan_raw = (INTEGRATION / PREFIX / "plan.json").read_bytes()
    plan = json.loads(plan_raw)
    result = source_json("attempt1/fit/parent-0-result.json")
    fit = source_json("attempt1/fit/failure.json")
    rows = list(csv.DictReader(io.StringIO((INTEGRATION / PREFIX / "attempt1/fit/parent-0-diagnostics.csv").read_text())))
    values = {key: [float(row[key]) for row in rows] for key in ["r_hat", "ess_bulk", "ess_tail"]}
    pinned = all(digest((INTEGRATION / pin["path"]).read_bytes()) == pin["sha256"] for pin in terminal["preserved_files"])
    return {
        "terminal_sha256": digest(terminal_raw),
        "plan_sha256": digest(plan_raw),
        "terminal_inventory_pins_match": pinned,
        "parent_count": len(plan["parent_ids"]),
        "parent_masses": plan["parent_masses"],
        "completed_result_count": len(fit["completed_parents"]),
        "fit_completed_result_matches_file": fit["completed_parents"] == [result],
        "csv_rows": len(rows),
        "csv_extrema": {"max_rhat": max(values["r_hat"]), "min_bulk_ess": min(values["ess_bulk"]),
                        "min_tail_ess": min(values["ess_tail"])},
        "terminal_exit_code": terminal["authoritative_exit_code"],
        "terminal_budget_exceeded": terminal["resource_budget_exceeded"],
    }


def main():
    checks = {"native": native(), "synthetic": {}}
    checks["synthetic"]["baseline_inert_archive"] = fixture("baseline")
    checks["synthetic"]["different_terminal_directory"] = fixture("terminal_elsewhere", terminal_elsewhere=True)
    checks["synthetic"]["contradictory_preservation_error"] = fixture(
        "preservation_error", lambda terminal, fit, result: fit.update(returned_inference_preservation_error="archive failed")
    )
    checks["synthetic"]["omitted_terminal_diagnostics"] = fixture(
        "omitted_diagnostics", lambda terminal, fit, result: terminal.update(diagnostics={})
    )
    checks["synthetic"]["result_coordinate_mismatch"] = fixture(
        "coordinate_mismatch", lambda terminal, fit, result: result.update(coordinate_version="synthetic_wrong_coordinate")
    )
    (HERE / "independent-checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps({"native_pins": checks["native"]["terminal_inventory_pins_match"],
                      "csv_rows": checks["native"]["csv_rows"],
                      "synthetic_acceptance": {k: v["accepted"] for k, v in checks["synthetic"].items()}}, indent=2))


if __name__ == "__main__":
    main()
