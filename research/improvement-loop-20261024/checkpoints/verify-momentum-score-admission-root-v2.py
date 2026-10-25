"""Verify frozen actual predictions with the accepted v2 reader; never open labels."""

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "momentum-forecast-scoring-sol-v2"))
import score  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    destination = Path(args.output)
    assert not destination.exists(), "Preserve earlier verification receipts"
    admission = {"path": args.admission, "sha256": args.expected_sha256}
    bundle = score.verify_admission(admission)
    folds = bundle["folds"]
    receipt = {
        "schema": "MomentumActualScoringAdmissionVerificationV2",
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "passed_before_numeric_label_access",
        "admission": admission,
        "verification_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "verified_cycles": sorted(folds),
        "verified_jobs": sum(len(fold["jobs"]) for fold in folds.values()),
        "selected_attempts": {str(cycle): fold["selected_attempt"] for cycle, fold in folds.items()},
        "audit": list(bundle["audit"]),
        "numeric_label_reader_invoked": False,
        "forecast_sampling_or_retuning": False,
    }
    with destination.open("x") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": receipt["status"], "verified_jobs": receipt["verified_jobs"]}))


if __name__ == "__main__":
    main()
