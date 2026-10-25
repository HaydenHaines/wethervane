"""Independent H14 input/identity checks. No numeric target labels or forecasts."""

import contextlib
import csv
import io
import json
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
from scipy.special import logit

HERE = Path(__file__).resolve().parent


def rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


class TargetPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(dir=HERE)
        cls.root = Path(cls.temporary.name)
        cls.output = cls.root / "prepared"
        original = p.pd.read_csv
        cls.label_calls = []

        def guarded_csv(path, *args, **kwargs):
            if "outcomes" in str(path).lower():
                if kwargs.get("usecols") != ["cycle", "race_id"]:
                    raise AssertionError("numeric target outcome parser prohibited")
                cls.label_calls.append(str(path))
            return original(path, *args, **kwargs)

        with (
            patch.object(p.pd, "read_csv", side_effect=guarded_csv),
            patch.object(p.factory, "compile_training", side_effect=AssertionError("no label-based compile")),
            patch("scipy.optimize.minimize", side_effect=AssertionError("no optimization")),
            patch("numpy.random.default_rng", side_effect=AssertionError("no forecast generation")),
            patch("numpy.load", side_effect=AssertionError("no bank array loading/export")),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            p.prepare(cls.output)
        cls.plan = json.loads((cls.output / "plan.json").read_bytes())
        cls.plan_sha = p.guard.sha(cls.output / "plan.json")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_exact_ordered_preflight_ids_counts_and_common_masks(self):
        old = rows(p.COVERAGE / "operator-selection-rows.csv")
        for fold in self.plan["folds"]:
            cycle = fold["cycle"]
            expected = p.SELECTION["expected_observations"][str(cycle)]
            self.assertEqual(len(fold["common_polled_race_ids"]), expected["polled_races"])
            for operator in p.OPERATORS:
                _, _, data = p.load_prepared(self.output / "plan.json", self.plan_sha, cycle, operator)
                wanted = [
                    row["observation_id"]
                    for row in old
                    if row["source_panel"] == "538_target_archive"
                    and row["cycle"] == str(cycle)
                    and row["horizon_days"] == "14"
                    and row["extra_lag_days_beyond_native_proxy"] == "0"
                    and row["operator"] == p.OPERATORS[operator]
                    and row["selected"] == "True"
                ]
                self.assertEqual(list(data.observation_ids), wanted)
                self.assertEqual(len(data.shares), expected[operator])
                self.assertEqual(set(data.race_ids), set(fold["common_polled_race_ids"]))
                self.assertTrue(((data.days_to_election >= 14) & (data.days_to_election <= 100)).all())
                self.assertTrue(((data.shares > 0) & (data.shares < 1)).all())
                for value in (data.shares, data.logits, data.days_to_election, data.race_indices):
                    with self.assertRaises(ValueError):
                        value.setflags(write=True)

    def test_latest_winners_independent_of_native_implementation(self):
        for fold in self.plan["folds"]:
            full = rows(fold["operators"]["native_full_series"]["observations"]["path"])
            latest = rows(fold["operators"]["latest_per_race_pollster"]["observations"]["path"])
            winners = {}
            for row in full:
                key = (row["race_id"], row["pollster_key"])
                order = (row["end_date"], row["available_date"], row["sample_id"], row["observation_id"])
                if key not in winners or order > winners[key][0]:
                    winners[key] = (order, row["observation_id"])
            wanted = {value[1] for value in winners.values()}
            self.assertEqual(
                [r["observation_id"] for r in latest],
                [r["observation_id"] for r in full if r["observation_id"] in wanted],
            )
            self.assertEqual(len(latest), len(winners))

    def test_exact_F0_design_and_complete_bank_reference(self):
        for fold in self.plan["folds"]:
            cycle = fold["cycle"]
            frame, offset, design, _, _, bank = p.target_features(cycle)
            expected = next(b for b in p.COMPATIBILITY["banks"] if b["year"] == cycle and b["arm"] == "F0")
            self.assertEqual(p.digest(offset.tobytes()), expected["target_offset_sha256"])
            self.assertEqual(p.digest(design.tobytes()), expected["target_design_sha256"])
            national = logit(frame.prior_national_presidential_dem_share.to_numpy())
            state = logit(frame.prior_presidential_dem_share.to_numpy())
            incumbency = frame.dem_incumbent.to_numpy().astype(int) - frame.rep_incumbent.to_numpy().astype(int)
            manual = np.column_stack(
                (
                    np.ones(len(frame)),
                    incumbency,
                    state - national,
                    national,
                    frame.president_party_asof,
                    frame.president_party_asof * frame.midterm,
                )
            )
            np.testing.assert_allclose(design, manual, atol=1e-15, rtol=0)
            self.assertEqual(bank["joint_shape"], [4, 4000, 6])
            self.assertEqual(len(bank["raw_banks"]), 4)
            stored = json.loads(p.guard.checked(fold["bank_reference"]))
            self.assertEqual(stored["row_count"], 16000)
            self.assertFalse(stored["bank_exported"])
            self.assertEqual(stored["raw_banks"], expected["raw_banks"])

    def test_no_event_expansion_and_full_roster_remains_separate(self):
        excluded = []
        for fold in self.plan["folds"]:
            masks = rows(fold["full_and_polled_mask"]["path"])
            self.assertEqual(len(masks), len(fold["bank_race_ids"]))
            self.assertEqual(
                sum(r["in_common_polled_mask"] == "True" for r in masks), len(fold["common_polled_race_ids"])
            )
            excluded.extend(r["race_id"] for r in masks if r["in_current_poll_roster"] == "False")
        self.assertEqual(
            set(excluded),
            {
                "2018:MN:senate:regular",
                "2018:MS:senate:regular",
                "2022:CA:senate:regular",
                "2022:OK:senate:regular",
                "2024:CA:senate:regular",
            },
        )
        self.assertFalse(self.plan["proposed_event_expansions_admitted"])
        self.assertTrue(self.plan["CA2022_new_poll_event_unresolved"])

    def test_numeric_outcomes_never_parsed_and_metadata_projection_poison(self):
        self.assertEqual(len(self.label_calls), 4)
        for fold in self.plan["folds"]:
            metadata = rows(fold["outcome_identity_metadata"]["path"])
            self.assertEqual(set(metadata[0]), {"cycle", "race_id"})
            self.assertEqual({r["race_id"] for r in metadata}, set(fold["bank_race_ids"]))
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            root = Path(directory)
            (root / "labels").mkdir()
            (root / "labels/outcomes-SEPARATE.csv").write_text(
                "cycle,race_id,actual_dem_share,dem_votes,rep_votes\n2018,synthetic,DO_NOT_PARSE,BAD,BAD\n"
            )
            with patch.object(p, "PANEL", root):
                metadata, _ = p.outcome_identity_metadata(2018)
            self.assertEqual(metadata.to_dict("records"), [{"cycle": 2018, "race_id": "synthetic"}])

    def test_cutoff_before_question_priority_and_adults_retained(self):
        roster = p.pd.DataFrame(
            [
                dict(
                    cycle=2018,
                    race_id="2018:AZ:senate:regular",
                    source_race_id="96",
                    eligible=True,
                    dem_candidate_id="1",
                    rep_candidate_id="2",
                )
            ]
        )
        candidates = []
        configurations = [
            (9001, 11, "lv", "2018-10-23 00:01", "2018-10-20"),
            (9001, 12, "rv", "2018-10-22 23:59", "2018-10-20"),
            (9002, 21, "a", "2018-10-21", "2018-10-20"),
            (9003, 31, "rv", "2018-10-21", "2018-10-20"),
            (9003, 32, "lv", "2018-10-21", "2018-10-20"),
            (9004, 41, "lv", "2018-10-21", "2018-10-24"),
        ]
        for poll, question, population, created, end in configurations:
            for party, cid, name, pct in (("DEM", 1, "Synthetic D", 48.0), ("REP", 2, "Synthetic R", 49.0)):
                candidates.append(
                    dict(
                        poll_id=poll,
                        question_id=question,
                        cycle=2018,
                        state="Arizona",
                        party=party,
                        pct=pct,
                        candidate_id=cid,
                        candidate_name=name,
                        race_id=96,
                        stage="general",
                        population=population,
                        sample_size=500,
                        start_date="2018-10-18",
                        end_date=end,
                        created_at=created,
                        pollster="Synthetic house",
                        election_date="2018-11-06",
                        tracking=False,
                        hypothetical=False,
                        ranked_choice_reallocated=False,
                    )
                )
        raw = p.pd.DataFrame(candidates)
        for column in ("created_at", "start_date", "end_date"):
            raw[column] = p.pd.to_datetime(raw[column], format="mixed")
        selected, _, _ = p.native_at(raw, roster, 2018, p.pd.Timestamp("2018-10-23"))
        lookup = selected.set_index("poll_id")
        self.assertEqual(lookup.loc["9001", "question_id"], "12")
        self.assertEqual(lookup.loc["9001", "population"], "RV")
        self.assertEqual(lookup.loc["9002", "population"], "A")
        self.assertEqual(lookup.loc["9003", "population"], "LV")
        self.assertNotIn("9004", lookup.index)

    def test_raw_record_hashes_and_selected_question_lineage(self):
        lines = p.ARCHIVE.read_bytes().splitlines(keepends=True)
        for fold in self.plan["folds"]:
            provenance = rows(fold["raw_provenance"]["path"])
            for row in provenance:
                raw = b"".join(lines[int(row["physical_start_line"]) - 1 : int(row["physical_end_line"])])
                self.assertEqual(p.digest(raw), row["raw_record_sha256"])
            selected = rows(fold["operators"]["native_full_series"]["observations"]["path"])
            for row in selected:
                source = [
                    r for r in provenance if r["poll_id"] == row["poll_id"] and r["question_id"] == row["question_id"]
                ]
                d = {float(r["pct_raw"]) for r in source if r["party"] == "DEM"}
                g = {float(r["pct_raw"]) for r in source if r["party"] == "REP"}
                self.assertEqual(len(d), 1)
                self.assertEqual(len(g), 1)
                self.assertAlmostEqual(
                    float(row["dem_share"]), next(iter(d)) / (next(iter(d)) + next(iter(g))), places=15
                )

    def test_rebound_malformed_packet_rejected_and_outputs_exclusive(self):
        base = self.plan["folds"][0]
        operator = "native_full_series"
        original = json.loads(p.guard.checked(base["operators"][operator]["joint_input"]))
        changes = [
            lambda d: d.update(schema="wrong"),
            lambda d: d["pollster_keys"].pop(),
            lambda d: d["race_indices"].__setitem__(0, 999),
            lambda d: d["logits"].__setitem__(0, 99.0),
            lambda d: d["cycles"].__setitem__(0, "2020"),
            lambda d: d["shares"].__setitem__(0, 0.0),
            lambda d: d["observation_ids"].__setitem__(0, d["observation_ids"][1]),
        ]
        for index, change in enumerate(changes):
            with self.subTest(case=index):
                data = json.loads(json.dumps(original))
                change(data)
                packet = self.root / f"bad-polls-{index}.json"
                p.write(packet, data)
                plan = json.loads(json.dumps(self.plan))
                plan["folds"][0]["operators"][operator]["joint_input"] = p.guard.pin(packet)
                bad_plan = self.root / f"bad-plan-{index}.json"
                p.write(bad_plan, plan)
                with self.assertRaises(ValueError):
                    p.load_prepared(bad_plan, p.guard.sha(bad_plan), 2018, operator)
        before = (self.output / "plan.json").read_bytes()
        with self.assertRaises(FileExistsError):
            p.prepare(self.output)
        self.assertEqual(before, (self.output / "plan.json").read_bytes())

    def test_source_drift_rejected_before_scientific_imports(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            root = Path(directory)
            for name in ("prepare.py", "source_guard.py", "source-bindings.json"):
                shutil.copyfile(HERE / name, root / name)
            path = root / "source-bindings.json"
            path.write_bytes(path.read_bytes() + b"changed")
            code = (
                "try:\n import prepare\nexcept ValueError:\n import sys\n assert 'numpy' not in sys.modules\n"
                " assert 'pandas' not in sys.modules\n print('drift blocked')\nelse:\n"
                " raise AssertionError('accepted drift')\n"
            )
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=root,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("drift blocked", result.stdout)


if __name__ == "__main__":
    unittest.main()
