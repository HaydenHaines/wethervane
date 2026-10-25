"""Independent source/ordered-selection/residual reconstruction; no fitting."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "momentum-calibration-preparation-sol-v1"
OWNER = Path("/home/hayden/projects/wethervane-redesign-20261006/integration")
PINS = {}


def check(path, expected=None):
    path = Path(path)
    body = path.read_bytes()
    value = hashlib.sha256(body).hexdigest()
    if expected:
        assert value == expected, str(path)
    PINS[str(path)] = value
    return body


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def main(expected):
    manifest = json.loads(check(SOURCE / "artifact-manifest.json", expected))
    for item in manifest["files"]:
        body = check(SOURCE / item["path"], item["sha256"])
        assert len(body) == item["bytes"]
    bindings = json.loads((SOURCE / "source-bindings.json").read_bytes())
    for i, item in enumerate(bindings["files"]):
        check(item["path"], item["sha256"])
        check(SOURCE / "snapshots" / f"{i:02d}-{Path(item['path']).name}", item["sha256"])
    plan = json.loads((SOURCE / "prepared-v1/plan.json").read_bytes())
    hist = OWNER / "research/polling-layer-implementation-20261020/historical-intake-sol/derived"
    poll_rows = rows(hist / "historical_poll_panel.csv")
    labels = rows(hist / "outcomes_allowed_cycles_SEPARATE.csv")
    coverage = rows(
        HERE.parent / "momentum-preflight-coverage-luna-v1/historical-calibration-baseline-id-ledger-v2.csv"
    )
    records = []
    errors = []
    for fold in plan["folds"]:
        target = fold["target"]
        earlier = [r for r in poll_rows if int(r["cycle"]) < target]
        truth = {r["race_id"]: r for r in labels if int(r["cycle"]) < target}
        latest = {}
        for r in earlier:
            age = float(r["days_to_election"])
            if not 0 <= age <= 100:
                continue
            key = (int(r["cycle"]), r["race_id"], r["pollster"].strip().casefold(), math.floor(age / 14))
            order = (r["end_date"], r["observation_id"])
            if key not in latest or order > (latest[key]["end_date"], latest[key]["observation_id"]):
                latest[key] = r
        selected = sorted(latest.values(), key=lambda r: (int(r["cycle"]), r["race_id"], r["observation_id"]))
        ids = [r["observation_id"] for r in selected]
        out = rows(Path(fold["selected"]["path"]))
        assert ids == [r["observation_id"] for r in out]
        assert len(ids) == {2018: 930, 2020: 1218, 2022: 1514, 2024: 1514}[target]
        ledger = [r for r in coverage if int(r["target_cycle"]) == target]
        assert set(r["observation_id"] for r in ledger) == set(r["observation_id"] for r in earlier)
        assert set(r["observation_id"] for r in ledger if r["selected_by_exact_fixed_bin_operator"] == "True") == set(
            ids
        )
        numeric = []
        for item in fold["blocks"]:
            block = json.loads(check(item["path"], item["sha256"]))
            cycle_set = set(map(int, block["cycles"]))
            assert len(cycle_set) == 1 and max(cycle_set) < target
            numeric.extend(
                zip(
                    block["observation_ids"],
                    block["race_ids"],
                    block["cycles"],
                    block["pollsters"],
                    block["days_to_election"],
                    block["residual"],
                    strict=True,
                )
            )
        assert [r[0] for r in numeric] == ids
        for source, numeric_row, outrow in zip(selected, numeric, out, strict=True):
            label = truth[source["race_id"]]
            q = float(source["dem_share"])
            t = float(label["actual_dem_share"])
            assert (
                abs(
                    q
                    - float(source["source_democratic_value"])
                    / (float(source["source_democratic_value"]) + float(source["source_republican_value"]))
                )
                < 1e-14
            )
            assert abs(t - float(label["dem_votes"]) / (float(label["dem_votes"]) + float(label["rep_votes"]))) < 1e-14
            expected_residual = math.log(q / (1 - q)) - math.log(t / (1 - t))
            errors.append(abs(expected_residual - numeric_row[5]))
            assert numeric_row[1:5] == (
                source["race_id"],
                str(source["cycle"]),
                source["pollster"].strip().casefold(),
                float(source["days_to_election"]),
            )
            assert abs(float(outrow["actual_dem_share"]) - t) < 1e-14
        pollmask = rows(Path(fold["poll_mask"]["path"]))
        labelmask = rows(Path(fold["label_mask"]["path"]))
        assert len(pollmask) == 1837 and len(labelmask) == 118
        assert set(r["observation_id"] for r in pollmask if r["selected"] == "True") == set(ids)
        assert set(r["observation_id"] for r in pollmask if r["earlier_cycle"] == "True") == set(
            r["observation_id"] for r in earlier
        )
        assert set(r["race_id"] for r in labelmask if r["earlier_label"] == "True") == set(truth)
        tx = sum(r["race_id"] == "2020:TX:senate:regular" for r in selected)
        assert tx == (31 if target > 2020 else 0)
        records.append(
            dict(
                target=target,
                rows=len(ids),
                earlier_source_rows=len(earlier),
                earlier_labels=len(truth),
                cycle_block_rows={str(c): sum(int(r["cycle"]) == c for r in selected) for c in fold["training_cycles"]},
                texas_2020_rows=tx,
                input_identity_sha256=fold["input_identity_sha256"],
            )
        )
    assert sum(r["earlier_source_rows"] for r in records) == 6395
    assert records[2]["input_identity_sha256"] == records[3]["input_identity_sha256"]
    assert max(errors) < 1e-12
    a = (SOURCE / "prepared-v1/2022/calibration-input.json").read_bytes()
    b = (SOURCE / "prepared-v1/2024/calibration-input.json").read_bytes()
    assert a == b
    result = dict(
        manifest_sha256=expected,
        artifact_count=len(manifest["files"]),
        source_snapshot_pairs_verified=len(bindings["files"]),
        inspected_sha256=PINS,
        folds=records,
        complete_observation_fold_records=6395,
        max_independent_residual_error=max(errors),
        exact_2022_2024_input_bytes=True,
        optimizer=False,
        sampler=False,
        forecast_scores=False,
    )
    (HERE / "reconstruction.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "artifact_count",
                    "source_snapshot_pairs_verified",
                    "complete_observation_fold_records",
                    "max_independent_residual_error",
                    "exact_2022_2024_input_bytes",
                ]
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-sha256", required=True)
    main(parser.parse_args().manifest_sha256)
