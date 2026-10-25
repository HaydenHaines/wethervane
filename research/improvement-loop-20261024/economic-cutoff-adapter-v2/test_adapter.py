"""Focused tests for the exact-midnight cutoff adapter; no network or fits."""

import hashlib
import json
import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import adapter


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v2_rows = json.loads((adapter.HERE / "rows.json").read_text())["rows"]
        cls.v1_rows = json.loads((adapter.V1 / "rows.json").read_text())["rows"]
        cls.v1_selection = json.loads((adapter.V1 / "selections.json").read_text())["selections"]
        cls.v2_selection = json.loads((adapter.HERE / "selections.json").read_text())["selections"]

    def selection(self, year, kind="headline", rows=None, cutoff=None, horizon=14):
        return adapter.v1.select(
            self.v2_rows if rows is None else rows,
            cutoff_utc=cutoff or adapter.cutoff_for(year),
            election_date=adapter.v1.election_day(year).isoformat(),
            horizon_days=horizon,
            series_id=adapter.SERIES[kind],
        )

    def test_exact_midnight_h14_all_cycles_and_ab_values(self):
        expected_ab = {2018: (2.3, 2.2), 2020: (1.4, 1.7), 2022: (8.2, 6.6), 2024: (2.4, 3.3)}
        self.assertEqual(len(self.v2_selection), 28)
        for item in self.v2_selection:
            cutoff = datetime.fromisoformat(item["cutoff_utc"])
            self.assertEqual((cutoff.hour, cutoff.minute, cutoff.second, cutoff.microsecond), (0, 0, 0, 0))
            self.assertEqual(cutoff.utcoffset().total_seconds(), 0)
            self.assertLessEqual(datetime.fromisoformat(item["row"]["release_utc"]), cutoff)
        for year, pair in expected_ab.items():
            values = tuple(next(s["row"]["value"] for s in self.v2_selection if s["cycle"] == year and s["kind"] == kind)
                           for kind in ("headline", "core"))
            self.assertEqual(values, pair)

    def test_new_august_2004_pdf_values_and_actual_release_timestamp(self):
        record = adapter.NEW_RELEASE
        source = json.loads((adapter.HERE / "acquisition.json").read_text())["source"]
        parsed = adapter.v1.parse_release(adapter.HERE / source["file"], record, source)
        self.assertEqual([(r["kind"], r["value"]) for r in parsed], [("headline", 2.7), ("core", 1.7)])
        self.assertEqual({r["release_utc"] for r in parsed}, {"2004-09-16T12:30:00+00:00"})
        self.assertEqual([r["reference_month"] for r in parsed], ["2004-08", "2004-08"])
        self.assertEqual([r["extraction"]["row_text"] for r in parsed], [
            "100.000        189.4     189.5       2.7        0.1          0.3          -0.1           0.1",
            "78.537       196.6       196.8         1.7          .1          .1            .1            .1",
        ])

    def test_2004_midnight_excludes_later_september_release_uses_august(self):
        current = self.selection(2004)
        self.assertEqual((current["row"]["reference_month"], current["row"]["value"]), ("2004-08", 2.7))
        self.assertEqual(self.selection(2004, "core")["row"]["value"], 1.7)
        self.assertEqual(adapter.v1.select(
            self.v1_rows,
            cutoff_utc="2004-10-19T00:00:00+00:00",
            election_date="2004-11-02",
            horizon_days=14,
            series_id=adapter.SERIES["headline"],
        )["status"], "unavailable")
        with_aug = [*self.v2_rows]
        seprecord = next(r for r in with_aug if r["reference_month"] == "2004-09" and r["kind"] == "headline")
        before = self.selection(2004, rows=with_aug, cutoff="2004-10-19T12:29:59+00:00")
        at_release = self.selection(2004, rows=with_aug, cutoff="2004-10-19T12:30:00+00:00")
        self.assertEqual(before["row"]["reference_month"], "2004-08")
        self.assertEqual(at_release["row"]["reference_month"], "2004-09")
        self.assertEqual(at_release["row"]["value"], seprecord["value"])

    def test_only_2004_selection_changes_from_v1(self):
        old = {(s["cycle"], s["kind"]): s for s in self.v1_selection}
        changed = []
        for current in self.v2_selection:
            prior = old[(current["cycle"], current["kind"])]
            a, b = prior.get("row"), current.get("row")
            sig_a = None if a is None else (a["reference_month"], a["value"], a["release_utc"])
            sig_b = None if b is None else (b["reference_month"], b["value"], b["release_utc"])
            if sig_a != sig_b:
                changed.append((current["cycle"], current["kind"], sig_a, sig_b))
        self.assertEqual([(year, kind) for year, kind, _, _ in changed], [(2004, "headline"), (2004, "core")])
        self.assertEqual(changed[0][2][:2], ("2004-09", 2.5))
        self.assertEqual(changed[0][3][:2], ("2004-08", 2.7))
        self.assertEqual(changed[1][2][:2], ("2004-09", 2.0))
        self.assertEqual(changed[1][3][:2], ("2004-08", 1.7))

    def test_v1_inputs_and_new_source_are_pinned_and_unchanged(self):
        manifest = json.loads((adapter.HERE / "inputs.json").read_text())
        for key in ("v1_parser", "v1_rows", "v1_selections", "v1_receipt"):
            item = manifest[key]
            self.assertEqual(hashlib.sha256((adapter.HERE / item["file"]).read_bytes()).hexdigest(), item["sha256"])
        receipt = json.loads((adapter.HERE / "receipt.json").read_text())
        for pin in receipt["files"]:
            self.assertEqual(hashlib.sha256((adapter.HERE / pin["file"]).read_bytes()).hexdigest(), pin["sha256"])

    def test_rebuild_rejects_changed_v1_inputs_before_writing_outputs(self):
        targets = (
            "cpi_vintages.py",
            "rows.json",
            "receipt.json",
            "raw/cpi_19981016.pdf",
        )
        for target in targets:
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temp:
                sandbox = Path(temp)
                copied_v1 = sandbox / "economic-vintages-v1"
                copied_v2 = sandbox / "economic-cutoff-adapter-v2"
                shutil.copytree(adapter.V1, copied_v1)
                shutil.copytree(adapter.HERE, copied_v2)
                changed = copied_v1 / target
                changed.write_bytes(changed.read_bytes() + b"\nchanged")
                output_before = {
                    name: hashlib.sha256((copied_v2 / name).read_bytes()).hexdigest()
                    for name in ("inputs.json", "rows.json", "selections.json", "receipt.json")
                }
                original_here, original_v1 = adapter.HERE, adapter.V1
                try:
                    adapter.HERE, adapter.V1 = copied_v2, copied_v1
                    with self.assertRaises(ValueError):
                        adapter.build()
                finally:
                    adapter.HERE, adapter.V1 = original_here, original_v1
                output_after = {
                    name: hashlib.sha256((copied_v2 / name).read_bytes()).hexdigest()
                    for name in output_before
                }
                self.assertEqual(output_after, output_before)

    def test_rebuild_refuses_modified_pinned_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            sandbox = Path(temp)
            copied_v2 = sandbox / "economic-cutoff-adapter-v2"
            shutil.copytree(adapter.HERE, copied_v2)
            manifest = json.loads((copied_v2 / "inputs.json").read_text())
            manifest["v1_rows"]["sha256"] = "0" * 64
            (copied_v2 / "inputs.json").write_text(json.dumps(manifest))
            output_before = hashlib.sha256((copied_v2 / "rows.json").read_bytes()).hexdigest()
            original_here = adapter.HERE
            try:
                adapter.HERE = copied_v2
                with self.assertRaises(ValueError):
                    adapter.build()
            finally:
                adapter.HERE = original_here
            self.assertEqual(hashlib.sha256((copied_v2 / "rows.json").read_bytes()).hexdigest(), output_before)


if __name__ == "__main__":
    unittest.main()
