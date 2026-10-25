"""Independent source/export controls for the four reviewed identity repairs."""

import ast
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import openpyxl
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import identity  # noqa: E402

V1 = HERE.parent / "momentum-audit-v1"


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


class IdentityDeltaTests(unittest.TestCase):
    def test_changed_review_packet_rejected_before_any_outputs(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            fixture = Path(directory)
            (fixture / "receipt.json").write_bytes((identity.INTAKE / "receipt.json").read_bytes())
            (fixture / "REPORT.md").write_bytes((identity.INTAKE / "REPORT.md").read_bytes() + b"changed")
            with patch.object(identity, "INTAKE", fixture):
                with self.assertRaisesRegex(ValueError, "fingerprint"):
                    identity.authenticate()
            self.assertEqual({p.name for p in fixture.iterdir()}, {"receipt.json", "REPORT.md"})

    def test_complete_v1_inventory_and_source_packet_unchanged(self):
        inventory, _, _ = identity.authenticate()
        self.assertEqual(len(inventory["files"]), 12)
        self.assertEqual((V1 / "plan.json").read_bytes(), (HERE / "plan.json").read_bytes())

    def test_only_truth_join_changed_native_selection_and_all_statistics_ast(self):
        def protected_segment(path):
            module = ast.parse(path.read_text())
            main = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "main")
            start = next(
                i
                for i, node in enumerate(main.body)
                if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "raw_path" for t in node.targets)
            )
            end = next(
                i
                for i, node in enumerate(main.body[start:], start)
                if isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and isinstance(node.value.func.value, ast.Name)
                and node.value.func.value.id == "paths"
                and node.value.func.attr == "extend"
            )
            return [ast.dump(n, include_attributes=False) for n in main.body[start:end]]

        self.assertEqual(protected_segment(V1 / "audit.py"), protected_segment(HERE / "audit.py"))

    def test_90_restored_rows_only_four_races_and_every_prior_cell_unchanged(self):
        old = rows(V1 / "selected-polls.csv")
        new = rows(HERE / "selected-polls.csv")

        def key(row):
            return row["race_id"], row["sample_id"]

        before = {key(r): r for r in old}
        after = {key(r): r for r in new}
        self.assertTrue(set(before) <= set(after))
        for k, value in before.items():
            self.assertEqual(value, after[k])
        restored = [r for r in new if key(r) not in before]
        actual = {race: sum(r["race_id"] == race for r in restored) for race in {r["race_id"] for r in restored}}
        self.assertEqual(
            actual,
            {
                "2022:IN:senate:regular": 2,
                "2022:NC:senate:regular": 34,
                "2022:OR:senate:regular": 4,
                "2022:PA:senate:regular": 50,
            },
        )
        old_receipt = json.loads((V1 / "receipt.json").read_text())
        new_receipt = json.loads((HERE / "receipt.json").read_text())
        self.assertEqual(old_receipt["helper_audits"], new_receipt["helper_audits"])
        self.assertEqual(new_receipt["polls_before_truth_join"], 2050)
        self.assertEqual(new_receipt["selected_polls"], 2050)
        self.assertEqual(new_receipt["excluded_poll_rows_missing_truth"], 0)

    def test_raw_workbook_complete_categories_and_principal_vote_values(self):
        sheet = openpyxl.load_workbook(
            identity.INTAKE / "raw/federalelections2022.xlsx", read_only=True, data_only=True
        )["7. US Senate Results by State"]
        raw = list(sheet.iter_rows(values_only=True))
        source = next(
            i["path"]
            for i in json.loads((V1 / "receipt.json").read_text())["inputs"]
            if i["path"].endswith("all-events-candidate-votes.csv")
        )
        exported = rows(Path(source))
        expected = {
            "IN": (222, "S2IN00208", 704480, 9, 1860154),
            "NC": (423, "S2NC00505 ", 1905786, 6, 3773924),
            "OR": (526, "S4OR00156", 788991, 5, 1927949),
            "PA": (540, "S6PA00274", 2751012, 6, 5375885),
        }
        for state, (number, fec_id, count, n, total) in expected.items():
            row = raw[number - 1]
            self.assertEqual((row[1], row[3], row[4], row[15]), (state, "S", fec_id, count))
            eligible = [
                (i + 1, r)
                for i, r in enumerate(raw)
                if r[1] == state and r[3] == "S" and r[8] is not None and r[15] is not None
            ]
            self.assertEqual(len(eligible), n)
            self.assertEqual(sum(r[15] for _, r in eligible), total)
            csv_rows = [r for r in exported if r["race_id"] == f"2022:{state}:senate:regular"]
            self.assertEqual({int(r["source_excel_row"]) for r in csv_rows}, {i for i, _ in eligible})
            self.assertEqual(sum(float(r["general_votes"]) for r in csv_rows), total)

    def test_whitespace_only_keys_and_conflicting_ids_rejected(self):
        frame = pd.DataFrame(dict(race_id=["synthetic", "synthetic"], fec_candidate_id=["S1 ", " S1"]))
        with self.assertRaisesRegex(ValueError, "conflicting"):
            identity.normalized_votes(frame)
        frame.fec_candidate_id = [" S1 ", "S 2"]
        self.assertEqual(identity.normalized_votes(frame).fec_id_key.tolist(), ["S1", "S 2"])
        self.assertEqual(frame.fec_candidate_id.tolist(), [" S1 ", "S 2"])

    def test_reviewed_alias_does_not_accept_different_archive_identity(self):
        race = pd.DataFrame(
            [
                dict(
                    cycle=2022,
                    race_id="2022:PA:senate:regular",
                    dem_candidate_id="WRONG",
                    rep_candidate_id="r",
                    dem_candidate_name="John Fetterman",
                    rep_candidate_name="Other",
                    election_date="2022-11-08",
                )
            ]
        )
        candidates = pd.DataFrame(
            [
                dict(
                    race_id=race.iloc[0].race_id,
                    group_id="D",
                    fec_candidate_id="S6PA00274",
                    candidate_id="FEC:S6PA00274",
                    candidate_name="John K. Fetterman",
                ),
                dict(
                    race_id=race.iloc[0].race_id,
                    group_id="R",
                    fec_candidate_id="S2PA00638",
                    candidate_id="FEC:S2PA00638",
                    candidate_name="Other",
                ),
            ]
        )
        votes = pd.DataFrame(
            [
                dict(
                    race_id=race.iloc[0].race_id,
                    fec_candidate_id="S6PA00274",
                    general_votes=2751012,
                    source_excel_row=540,
                ),
                dict(
                    race_id=race.iloc[0].race_id,
                    fec_candidate_id="S2PA00638",
                    general_votes=2487260,
                    source_excel_row=546,
                ),
            ]
        )
        truth, joins = identity.build_truth(race, candidates, votes, str.lower, str)
        self.assertEqual(truth, [])
        self.assertFalse(joins[0]["accepted"])

    def test_delta_exports_match_race_and_summary_tables(self):
        deltas = rows(HERE / "race-deltas.csv")
        self.assertEqual(len(deltas), 113)
        self.assertEqual(sum(int(r["restored_poll_rows"]) for r in deltas), 90)
        changes = rows(HERE / "comparison-deltas.csv")
        self.assertFalse(any(r["_merge"] == "left_only" for r in changes))
        for row in changes:
            if row["_merge"] == "both":
                for field in ("trend_pp", "performance_pp", "actual_dem_share", "prior_dem_share", "late_dem_share"):
                    self.assertEqual(row[field + "_v1"], row[field + "_v2"])
        summary = rows(HERE / "summary-deltas.csv")
        primary = next(
            r
            for r in summary
            if r["variant"] == "field_end"
            and r["comparison"] == "all_pollsters"
            and r["threshold_pp"] == "1"
            and r["cycle"] == "all"
        )
        self.assertEqual(
            (primary["n_v1"], primary["n_v2"], primary["below_v2"], primary["within_v2"], primary["above_v2"]),
            ("12", "12", "4", "6", "2"),
        )
        midpoint = next(
            r
            for r in summary
            if r["variant"] == "midpoint"
            and r["comparison"] == "all_pollsters"
            and r["threshold_pp"] == "1"
            and r["cycle"] == "all"
        )
        self.assertEqual((midpoint["n_v1"], midpoint["n_v2"]), ("6", "7"))


if __name__ == "__main__":
    unittest.main()
