"""Bind four immutable independent raw reviews to one terminal G01 attempt."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
RUN = BASE / "gasoline-execution-v1/attempt-v1"
PLAN = BASE / "gasoline-execution-v1/prepared-v2/plan.json"
PLAN_SHA = "63f6de0e57949857a48d02e5084af8cc9d576b4ffa96985b13afc938fe55b5ab"
TERMINAL_SHA = "f9922ed230ce99c83f88d8663f7ec1f3b97634e0707e4dd62ac1f91fc4ac4ec9"
SUMMARY_SHA = "b72223532e63555332a3628ab4b9d4aeafd5a3f2d926f6111f126f7e8e673b22"
YEARS = (2018, 2020, 2022, 2024)
ARMS = ("H0", "F0", "G01")
PINS = {}


def pin(path, expected=None):
    path = Path(path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    PINS[str(path)] = actual
    if expected is not None:
        assert actual == expected, f"changed pin: {path}"
    return actual


def read(path, expected=None):
    pin(path, expected)
    return json.loads(Path(path).read_bytes())


def main():
    pin(Path(__file__))
    plan = read(PLAN, PLAN_SHA)
    assert len(plan["jobs"]) == 12
    launch = read(RUN / "launch.json")
    assert launch["plan_sha256"] == PLAN_SHA and launch["continuation_manifest"] is None
    assert launch["plan"] == {"path": str(PLAN), "sha256": PLAN_SHA}
    assert launch["runtime_certificate"] == plan["inherited_runtime_acceptance"]
    pin(launch["runtime_certificate"]["path"], launch["runtime_certificate"]["sha256"])
    terminal = read(RUN / "terminal.json", TERMINAL_SHA)
    assert terminal["plan_sha256"] == PLAN_SHA
    assert terminal["status"] == "completed_F_only_comparison"
    assert terminal["all_numerical_passed"] is True
    assert terminal["count_qualification"] is False and terminal["promotion_authorized"] is False
    assert terminal["summary"] == {"path": str(RUN / "summary.json"), "sha256": SUMMARY_SHA}
    summary = read(RUN / "summary.json", SUMMARY_SHA)
    assert summary["count_qualification"] is False
    assert summary["promotion_authorized"] is False and summary["prospective_evidence"] is False
    assert [(cycle["target"], cycle["tier"]) for cycle in summary["cycles"]] == [
        (2018, "A"), (2020, "B"), (2022, "A"), (2024, "B")
    ]
    old_manifest_path = HERE / "reviews-manifest-v2.json"
    old_manifest = read(old_manifest_path)
    assert [item["year"] for item in old_manifest["folds"]] == [2018, 2020, 2022]
    for item in old_manifest["folds"]:
        for key in ("checker", "receipt"):
            pin(item[key]["path"], item[key]["sha256"])
    predecessor = old_manifest["predecessor"]
    pin(predecessor["path"], predecessor["sha256"])
    fold_receipts = {}
    for year in YEARS:
        receipt_path = HERE / ("verification.json" if year == 2018 else f"verification-{year}.json")
        receipt = read(receipt_path)
        assert [fold["year"] for fold in receipt["reviewed_folds"]] == [year]
        assert receipt["numeric_target_labels_parsed"] is False
        assert receipt["sampler_or_predictive_simulation_invoked"] is False
        assert receipt["inspected_sha256"][str(PLAN)] == PLAN_SHA
        assert all(arm["numerical_passed"] for arm in receipt["reviewed_folds"][0]["arms"])
        assert all(arm["max_logit_error"] == arm["max_share_error"] == 0 for arm in receipt["reviewed_folds"][0]["arms"])
        fold_receipts[year] = {"path": str(receipt_path), "sha256": PINS[str(receipt_path)]}
    assert not read(HERE / "verification-2024.json")["deferred_outcome_source_pins"]
    assert not list(RUN.rglob("failure.json")), "A failure artifact requires explicit audit"
    jobs = {}
    for year in YEARS:
        folder = RUN / str(year)
        mask_path = folder / "masks.json"
        mask = read(mask_path)
        assert mask["race_ids"]["all"]
        assert len(mask["source_prediction_seals"]) == 3
        seals = []
        scores = []
        for arm in ARMS:
            job_path = folder / arm
            seal_path = job_path / "prediction-seal.json"
            seal = read(seal_path)
            job_terminal = read(job_path / "terminal.json")
            score_path = job_path / "scores.json"
            score = read(score_path)
            fit = read(job_path / "fit-diagnostics.json")
            predictive = read(job_path / "prediction-diagnostics.json")
            assert (seal["target"], seal["arm"], seal["plan_sha256"]) == (year, arm, PLAN_SHA)
            assert seal["numerical_passed"] is True
            assert job_terminal["status"] == "completed" and job_terminal["numerical_passed"] is True
            assert job_terminal["plan_sha256"] == PLAN_SHA
            assert len(job_terminal["raw_chains"]) == 4
            assert fit["passed"] is True and predictive["passed"] is True
            assert score["numerical_passed"] is True and score["forecast_qualified"] is False
            assert summary["cycles"][YEARS.index(year)]["arms"][arm]["numerical_passed"] is True
            assert summary["cycles"][YEARS.index(year)]["arms"][arm]["strata"] == score["strata"]
            assert score["rows"] and [r["race_id"] for r in score["rows"]] == mask["race_ids"]["all"]
            assert seal["predictions"]["sha256"] == pin(seal["predictions"]["path"])
            seals.append({"path": str(seal_path), "sha256": PINS[str(seal_path)]})
            scores.append(score_path)
            jobs[f"{year}:{arm}"] = {
                "terminal_sha256": PINS[str(job_path / "terminal.json")],
                "seal_sha256": PINS[str(seal_path)],
                "score_sha256": PINS[str(score_path)],
                "fit_passed": True,
                "predictive_passed": True,
            }
        assert mask["source_prediction_seals"] == seals
        assert max((folder / arm / "prediction-seal.json").stat().st_mtime_ns for arm in ARMS) <= mask_path.stat().st_mtime_ns
        assert mask_path.stat().st_mtime_ns <= min(path.stat().st_mtime_ns for path in scores)
    assert len(jobs) == len(plan["jobs"]) == 12
    assert set(jobs) == {f"{item['target']}:{item['arm']}" for item in plan["jobs"]}
    report = {
        "schema": "G01IndependentTerminalAuditSolV1",
        "utc": datetime.now(timezone.utc).isoformat(),
        "decision": "raw_construction_and_parameter_gates_verified_for_all_twelve_jobs",
        "frozen_plan_sha256": PLAN_SHA,
        "terminal_sha256": TERMINAL_SHA,
        "summary_sha256": SUMMARY_SHA,
        "predecessor_manifest_sha256": PINS[str(old_manifest_path)],
        "immutable_fold_receipts": fold_receipts,
        "jobs": jobs,
        "source_and_terminal_sha256": PINS,
        "failed_jobs": [],
        "count_qualification": False,
        "promotion_authorized": False,
        "limitations": [
            "Independent rank Rhat, bulk/tail ESS, divergence and depth recomputation; predictive ArviZ estimates only threshold-checked from stored values.",
            "Every saved predictive logit/share reconstructed exactly from raw posterior, frozen design and saved future noise.",
            "Target numeric outcomes were not parsed by this reviewer; score arithmetic is a separate Luna review.",
            "Filesystem seal/mask/score chronology is consistency evidence, not authenticated first access timing.",
            "Exposed four-cycle retrospective development evidence; no full-count, current-2026 or public-model qualification.",
        ],
    }
    with (HERE / "final-audit.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"jobs": len(jobs), "failed_jobs": [], "utc": report["utc"]}))


if __name__ == "__main__":
    main()
