"""Independent actual-source selection/residual and synthetic boundary checks."""

import csv
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import prepare as p

HERE = Path(__file__).resolve().parent


def independent_ids(target):
    with (p.HIST / "derived/historical_poll_panel.csv").open() as stream:
        source = list(csv.DictReader(stream))
    latest = {}
    for row in source:
        cycle = int(row["cycle"])
        if cycle >= target:
            continue
        age = float(row["days_to_election"])
        if not 0 <= age <= 100:
            continue
        key = (cycle, row["race_id"], row["pollster"].strip().casefold(), int(age // 14))
        if key not in latest or (row["end_date"], row["observation_id"]) > (
            latest[key]["end_date"],
            latest[key]["observation_id"],
        ):
            latest[key] = row
    return sorted(latest.values(), key=lambda r: (int(r["cycle"]), r["race_id"], r["observation_id"]))


class PreparationTests(unittest.TestCase):
    def test_all_four_exact_selections_and_corrected_coverage(self):
        for target, count in ((2018, 930), (2020, 1218), (2022, 1514), (2024, 1514)):
            polls, _, selected, _, blocks = p.fold_inputs(target)
            expected = independent_ids(target)
            self.assertEqual(selected.observation_id.tolist(), [r["observation_id"] for r in expected])
            self.assertEqual(len(selected), count)
            self.assertEqual(sum(len(b.residual) for b in blocks), count)
            self.assertTrue(p.reconcile_coverage(target, polls, selected)["all_ids_agree"])
            self.assertEqual(int(selected.race_id.eq("2020:TX:senate:regular").sum()), 31 if target > 2020 else 0)
            self.assertTrue(selected.cycle.lt(target).all())

    def test_residual_arithmetic_and_all_source_masks_independently(self):
        polls, outcomes, selected, _, blocks = p.fold_inputs(2018)
        numeric = np.concatenate([b.residual for b in blocks])
        expected = [
            math.log(q / (1 - q)) - math.log(t / (1 - t))
            for q, t in zip(selected.dem_share, selected.actual_dem_share, strict=True)
        ]
        np.testing.assert_allclose(numeric, expected, atol=1e-14, rtol=0)
        poll_mask, label_mask = p.source_masks(2018, polls, outcomes, selected)
        self.assertEqual(len(poll_mask), 1837)
        self.assertEqual(sum(row["selected"] for row in poll_mask), 930)
        self.assertEqual(sum(row["earlier_cycle"] for row in poll_mask), 1192)
        self.assertTrue(all(not row["selected"] for row in poll_mask if row["source_cycle"] >= 2018))
        self.assertEqual(sum(row["earlier_label"] for row in label_mask), 62)
        self.assertEqual(len(label_mask), 118)
        with self.assertRaises(ValueError):
            blocks[0].residual.setflags(write=True)
        with self.assertRaises(ValueError):
            blocks[0].coordinates.days_to_election.setflags(write=True)

    def test_future_numeric_cells_not_parsed_and_unknown_cycle_rejected(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            path = Path(directory) / "fixture.csv"
            path.write_text("cycle,value\n2012,.2\n2016,.3\n2018,FUTURE_NOT_NUMERIC\n2020,FUTURE_NOT_NUMERIC\n")
            frame = p.read_earlier_csv(path, 2018)
            self.assertEqual(frame.cycle.tolist(), [2012, 2016])
            self.assertEqual(frame.value.to_numpy().dtype.kind, "f")
            path.write_text("cycle,value\n2012,.2\n2028,UNKNOWN_SOURCE\n")
            with self.assertRaisesRegex(ValueError, "unsupported source cycle"):
                p.read_earlier_csv(path, 2018)

    def test_2022_2024_exact_numerical_identity_and_original_MAP_rows(self):
        _, _, a, _, ab = p.fold_inputs(2022)
        _, _, b, _, bb = p.fold_inputs(2024)
        ai, ap = p.input_identity(a, ab)
        bi, bp = p.input_identity(b, bb)
        self.assertEqual(ai, bi)
        self.assertEqual(p.canonical(ap), p.canonical(bp))
        self.assertEqual(a.to_csv(index=False), b.to_csv(index=False))
        original = p.pd.read_csv(p.POLL / "calibration-v1/selected-training-polls.csv")
        self.assertEqual(a.observation_id.tolist(), original.observation_id.tolist())
        self.assertEqual(a.groupby("cycle").size().to_dict(), {2012: 467, 2016: 463, 2018: 288, 2020: 296})

    def test_both_joint_objective_gradients_at_actual_supplied_points(self):
        _, _, _, _, blocks = p.fold_inputs(2018)
        medians = np.array([p.SELECTION["regularizer"]["medians"][k] for k in p.math_model.COMPONENTS])
        point = np.log(medians)
        for width in (5, 6):
            if width == 5:
                data = [(b.residual, b.coordinates.kernels()[:5]) for b in blocks]

                def objective(x):
                    return p.math_model.INCUMBENT.gaussian_error_objective(
                        x, data, prior_medians=medians[:5], prior_log_sd=0.7
                    )
            else:

                def objective(x):
                    return p.math_model.gaussian_error_objective(x, blocks, prior_medians=medians, prior_log_sd=0.7)

            value, gradient = objective(point[:width])
            self.assertTrue(np.isfinite(value))
            for i in range(width):
                step = np.eye(width)[i] * 1e-6
                numerical = (objective(point[:width] + step)[0] - objective(point[:width] - step)[0]) / 2e-6
                self.assertAlmostEqual(gradient[i], numerical, delta=2e-6)

    def test_source_corruption_rejected_before_scientific_imports(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            root = Path(directory)
            for name in ("guard.py", "prepare.py", "source-bindings.json"):
                shutil.copyfile(HERE / name, root / name)
            path = root / "source-bindings.json"
            path.write_bytes(path.read_bytes() + b"changed")
            code = (
                "try:\n import prepare\nexcept ValueError:\n import sys\n assert 'numpy' not in sys.modules\n"
                " print('rejected before imports')\nelse:\n raise AssertionError('changed source accepted')\n"
            )
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=root,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("rejected before imports", result.stdout)

    def test_native_inclusive_age_bins_and_lexical_tie_break(self):
        ages = [0, 0, 13, 14, 27, 100, 101, -1]
        ids = ["zero-low", "zero-high", "day13", "day14", "day27", "day100", "day101", "negative"]
        # Give same-day zero rows a deliberate lexical ID tie; native keep-last
        # selects zero-low because 'low' sorts after 'high'.
        frame = p.pd.DataFrame(
            dict(
                cycle=[2012] * 8,
                race_id=["synthetic"] * 8,
                pollster=[" House "] * 8,
                days_to_election=ages,
                observation_id=ids,
                dem_share=[0.5] * 8,
                end_date=[str((p.pd.Timestamp("2012-11-06") - p.pd.Timedelta(days=a)).date()) for a in ages],
            )
        )
        outcomes = p.pd.DataFrame(dict(race_id=["synthetic"], actual_dem_share=[0.6]))
        selected = p.select_native(frame, outcomes)
        self.assertEqual(selected.observation_id.tolist(), ["day100", "day14", "zero-low"])
        self.assertEqual(selected.lead_bin.tolist(), [7, 1, 0])
        self.assertEqual(selected.pollster_key.tolist(), ["house"] * 3)

    def test_exclusive_full_preparation_and_authenticated_reload_without_fits(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            output = Path(directory) / "prepared"
            with (
                patch("scipy.optimize.minimize", side_effect=AssertionError("no optimization")),
                patch("numpy.random.default_rng", side_effect=AssertionError("no sampling")),
            ):
                p.prepare(output)
            plan_path = output / "plan.json"
            plan_sha = p.guard.sha(plan_path)
            plan, fold, blocks = p.load_prepared(plan_path, plan_sha, 2024)
            self.assertEqual(fold["rows"], 1514)
            self.assertEqual(plan["unique_numerical_calibrations"], 3)
            self.assertEqual(sum(len(b.residual) for b in blocks), 1514)
            with self.assertRaises(FileExistsError):
                p.prepare(output)
            data = Path(fold["blocks"][0]["path"])
            original = data.read_bytes()
            data.write_bytes(data.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                p.load_prepared(plan_path, plan_sha, 2024)
            changed = json.loads(original)
            changed["cycles"] = ["2024"] * len(changed["cycles"])
            data.write_bytes(p.canonical(changed))
            plan["folds"][-1]["blocks"][0] = p.pin(data)
            bad_plan = output / "rebound-future-plan.json"
            p.write(bad_plan, plan)
            with self.assertRaisesRegex(ValueError, "target/future"):
                p.load_prepared(bad_plan, p.guard.sha(bad_plan), 2024)


if __name__ == "__main__":
    unittest.main()
