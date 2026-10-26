"""Read-only exact event intersections; write only beside this script. No fits."""
import collections
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
pins = {}


def read(relative):
    path = LAB / relative
    raw = path.read_bytes()
    pins[relative] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    if path.suffix == ".csv":
        return list(csv.DictReader(raw.decode().splitlines()))
    return json.loads(raw) if path.suffix == ".json" else raw.decode()


compat = read("momentum-preflight-bank-compatibility-sol-v1/compatibility.json")
support = read("momentum-preflight-coverage-luna-v1/f0-training-race-support.csv")
read("momentum-calibration-preparation-sol-v1/README.md")
read("calibration-consumer-v1/snapshot-source/src/forecasting/poll_covariance_v1.py")
read("m01-count-measurement-prototype-sol-v1/prototype.py")
folds = []
ledger = []
for year in (2018, 2020, 2022, 2024):
    prefix = f"momentum-calibration-preparation-sol-v1/prepared-v1/{year}"
    polls = read(prefix + "/selected-training-polls.csv")
    labels = read(prefix + "/earlier-outcomes.csv")
    label_keys = {r["race_id"] for r in labels}
    counts = collections.Counter(r["race_id"] for r in polls)
    assert set(counts) <= label_keys
    assert all(int(r["cycle"]) < year for r in polls)
    used = {r["race_id"] for r in polls}
    banks = [b for b in compat["banks"] if b["year"] == year]
    by_arm = {}
    for bank in banks:
        keys = set(bank["training_race_ids"])
        intersection = used & keys
        by_arm[bank["arm"]] = {
            "training_events": len(keys), "shared_used_calibration_labels": len(intersection),
            "calibration_rows_using_shared_labels": sum(counts[k] for k in intersection),
            "calibration_only_labels": sorted(used - keys),
            "bank_target_label_intersection": sorted(used & set(bank["target_race_ids"])),
        }
        if bank["arm"] == "F0":
            assert keys == {r["race_id"] for r in support if int(r["target_cycle"]) == year}
            for key in sorted(used | keys):
                ledger.append({"target_cycle": year, "canonical_event_key": key,
                               "calibration_rows": counts[key], "in_f0_training": key in keys})
    folds.append({"target_cycle": year, "calibration_rows": len(polls),
                  "used_calibration_labels": len(used), "earlier_label_table_rows": len(labels),
                  "bank_intersections": by_arm,
                  "unknown_sample_identity_rows": sum(r.get("sample_overlap_status", "").startswith("unknown") for r in polls)})

coords = read("calibration-consumer-v1/inputs/target-coordinates.json")
days = coords["days_to_election"]
movement = [[min(x, y) / 30 for y in days] for x in days]
# Exact illustrative covariance-budget split. Binary fractions avoid roundoff.
split_checks = []
for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
    discrepancy = max(abs(fraction * v + (1 - fraction) * v - v)
                      for row in movement for v in row)
    assert discrepancy < 1e-15
    split_checks.append({"latent_fraction": fraction, "maximum_kernel_difference": discrepancy})

result = {"schema": "M01OwnershipFollowupSolV1", "pins": pins, "folds": folds,
          "conditional_same_brownian_kernel_split": split_checks,
          "scope": "Exact key intersections from pinned lab preparation/review artifacts; not source authentication, strict-as-of admission, or full native Problem inventory.",
          "fits": False, "owner_files_read_or_written": False}
# A normalized conditional counterexample: shared labels need not mean that
# their likelihood is multiplied twice. Invented probabilities, no real fit.
beta_likelihood = [0.2, 0.8]  # p(Y=1 | beta)
theta_likelihood = [0.7, 0.3]  # p(P=1 | Y=1, theta)
joint = [[0.25 * b * t for t in theta_likelihood] for b in beta_likelihood]
normalizer = sum(map(sum, joint))
joint = [[v / normalizer for v in row] for row in joint]
product = [[b / sum(beta_likelihood) * t / sum(theta_likelihood)
            for t in theta_likelihood] for b in beta_likelihood]
error = max(abs(joint[i][j] - product[i][j]) for i in range(2) for j in range(2))
assert error < 1e-15
result["invented_conditional_factor_fixture"] = {
    "joint_equals_product_max_error": error,
    "beta_posterior_once": 0.8,
    "beta_posterior_if_label_likelihood_squared": 0.8**2 / (0.2**2 + 0.8**2),
    "meaning": "p(Y|beta) p(P|Y,theta) uses the outcome density once; event intersection alone cannot reject this conditional law. Squaring p(Y|beta) is a different law.",
}
(HERE / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
with (HERE / "event-ownership.csv").open("w") as out:
    writer = csv.DictWriter(out, fieldnames=list(ledger[0]))
    writer.writeheader()
    writer.writerows(ledger)
print(json.dumps(folds, indent=2))
