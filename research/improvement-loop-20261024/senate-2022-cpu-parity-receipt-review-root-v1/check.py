"""Review saved CPU parent-0 parity receipts without loading predictive arrays."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path("/home/hayden/projects/wethervane-redesign-20261006/integration")
BASE = ROOT / "research/senate-stack-integration-20261020/national-2022-gpu-preparation-root-v1"
CANDIDATE = BASE / "gpu-kernel-argument-remediation-v1"
RESULT = CANDIDATE / "cpu-jax-parent0-parity-v2/result.json"
RESULT_SHA = "109feacd03133852444239e9bc173481b41a92086208722230537a0a38f72d39"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    raw = RESULT.read_bytes()
    need(hashlib.sha256(raw).hexdigest() == RESULT_SHA, "parity receipt bytes differ")
    result = json.loads(raw)
    inputs = {
        "source_sha256": CANDIDATE / "cpu_jax_parent0_parity_v2.py",
        "CPU_reference_sha256": BASE / "parent0-native-reference-v2/result.json",
        "all_parent_structure_sha256": BASE / "all-parent-graph-structure-v1/result.json",
        "adapter_boundary_sha256": BASE / "candidate-boundary-validation-v2/result.json",
        "boundary_sha256": CANDIDATE / "cpu-jax-parent0-boundary-validation-v2.json",
        "original_plan_sha256": BASE.parent / "national-multicycle-execution-astra-v2/A2022/UNADMITTED-plan.json",
    }
    for key, path in inputs.items():
        need(sha(path) == result[key], f"receipt input pin differs: {key}")
    reference = json.loads(inputs["CPU_reference_sha256"].read_text())
    structure = json.loads(inputs["all_parent_structure_sha256"].read_text())
    boundary = json.loads(inputs["boundary_sha256"].read_text())
    need(result["schema"] == "CorrectedNationalParent0CPUJaxParityCandidateV1" and
         result["status"] == "PASS_parent0_native_CPU_JAX_parity" and result["stage"] == "complete" and
         result["GPU_used"] is False and result["fit_admitted"] is False and
         result["forecast_qualified"] is False and result["sampler_transitions"] == 0,
         "parity status or qualification flags differ")
    need(reference["status"] == "PASS_native_parent0_reference" and reference["states"] == 5 and
         reference["full_gradients"] == 5 and len(reference["physical_names"]) == 815,
         "native reference coverage differs")
    need(structure["status"] == "PASS_all_parent_structure_only" and structure["parent_indices"] == list(range(8)) and
         boundary["status"] == "PASS_synthetic_boundaries_only", "structure or boundary status differs")
    need(reference["original_plan_sha256"] == result["original_plan_sha256"] == structure["original_plan_sha256"],
         "plan identity differs across reference and structure")
    expected_boundary_paths = {str((CANDIDATE / name).resolve()) for name in
        ("cpu_jax_parent0_parity_v2.py", "test_cpu_jax_parent0_parity_v2.py")}
    need(set(boundary["source_pins"]) == expected_boundary_paths and
         all(sha(Path(path)) == digest for path, digest in boundary["source_pins"].items()),
         "boundary proof source pins differ")
    need(sha(CANDIDATE / "cpu-jax-parent0-parity-v2/cpu_jax_parent0_parity_v2.py") == result["source_sha256"],
         "saved runtime source differs")
    rows = result["checks"]
    need(result["points"] == 5 and result["complete_gradients_compared"] == 5 and
         result["physical_comparisons"] == 4075 and len(rows) == 5 and
         [row["point"] for row in rows] == list(range(5)), "point coverage differs")
    max_fraction = 0.0
    for row in rows:
        checks = row["checks"]
        need(row["passed"] is True and len(checks) == 817 and
             [check["variable"] for check in checks] == ["logp", "full_gradient", *reference["physical_names"]],
             "comparison order or coverage differs")
        for check in checks:
            value = check["max_tolerance_fraction"]
            need(check["pass_check"] is True and check["failures"] == 0 and check["nonfinite_count"] == 0 and
                 type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
                 "comparison record is not a finite passing result")
            max_fraction = max(max_fraction, value)
    need(sha(RESULT) == RESULT_SHA and all(sha(path) == result[key] for key, path in inputs.items()),
         "receipt or inputs changed during review")
    print(json.dumps({"schema": "National2022CPUParent0ParityReceiptReviewV1", "result_sha256": RESULT_SHA,
        "input_hashes_rechecked": len(inputs), "points": len(rows), "checks_per_point": 817,
        "reported_physical_comparisons": 4075, "max_tolerance_fraction": max_fraction,
        "raw_numeric_comparisons_independently_recomputed": False, "all_parent_cpu_parity_verified": False,
        "GPU_parity_verified": False, "fit_admitted": False, "forecast_qualified": False}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
