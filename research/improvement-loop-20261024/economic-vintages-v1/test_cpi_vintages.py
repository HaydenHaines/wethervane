"""Offline real-release checks and independently constructed boundary failures."""

import hashlib
import json
import unittest
from copy import deepcopy
from datetime import timedelta

from cpi_vintages import HERE, SERIES, cutoff_for, election_day, parse_release, parse_table_cells, select, timestamp


class VintageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = json.loads((HERE / "rows.json").read_text())["rows"]
        cls.acquisition = json.loads((HERE / "acquisition.json").read_text())

    def selection(self, year, rows=None, **overrides):
        args = dict(
            cutoff_utc=cutoff_for(year),
            election_date=election_day(year).isoformat(),
            horizon_days=14,
            series_id=SERIES["headline"],
        )
        args.update(overrides)
        return select(self.rows if rows is None else rows, **args)

    def test_real_pdf_headline_core_values_and_all_even_cycles(self):
        expected = {
            1998: (1.5, 2.5),
            2000: (3.5, 2.6),
            2002: (1.5, 2.2),
            2004: (2.5, 2.0),
            2006: (2.1, 2.9),
            2008: (4.9, 2.5),
            2010: (1.1, 0.8),
            2012: (2.0, 2.0),
            2014: (1.7, 1.7),
            2016: (1.5, 2.2),
            2018: (2.3, 2.2),
            2020: (1.4, 1.7),
            2022: (8.2, 6.6),
            2024: (2.4, 3.3),
        }
        for year, pair in expected.items():
            for kind, value in zip(("headline", "core"), pair):
                selected = self.selection(year, series_id=SERIES[kind])
                self.assertEqual(selected["status"], "available")
                self.assertEqual(selected["row"]["value"], value)
                self.assertLessEqual(timestamp(selected["row"]["release_utc"]), timestamp(cutoff_for(year)))
        self.assertEqual(len(self.rows), 30)
        self.assertEqual(json.loads((HERE / "rows.json").read_text())["failures"], [])

    def test_2014_future_september_rejected_august_selected(self):
        selected = self.selection(2014)
        self.assertEqual(selected["row"]["reference_month"], "2014-08")
        self.assertEqual(self.selection(2014, reference_month="2014-09")["status"], "unavailable")
        self.assertEqual(
            self.selection(2014, cutoff_utc="2014-10-22T12:30:00+00:00", horizon_days=13)["row"]["reference_month"],
            "2014-09",
        )

    def test_2004_actual_same_day_release_timestamp(self):
        # H14 calendar day alone cannot decide admission before 08:30 EDT.
        self.assertEqual(self.selection(2004, cutoff_utc="2004-10-19T12:29:59+00:00")["status"], "unavailable")
        self.assertEqual(self.selection(2004, cutoff_utc="2004-10-19T12:30:00+00:00")["row"]["value"], 2.5)

    def test_bad_horizon_units_seasonality_and_unknown_layout(self):
        for changes in [
            dict(horizon_days=7),
            dict(units="fraction"),
            dict(seasonality="SA"),
            dict(cutoff_utc="2024-10-22"),
            dict(horizon_days=True),
            dict(series_id="CPIAUCSL"),
        ]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.selection(2024, **changes)
        row = deepcopy(
            next(row for row in self.rows if row["reference_month"] == "2024-09" and row["kind"] == "headline")
        )
        for changes in [
            dict(value=float("nan")),
            dict(value=True),
            dict(seasonality="SA"),
            dict(revision_status="current_revised_series"),
            dict(schema="unknown"),
        ]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.selection(2024, [dict(row, **changes)])

    def test_future_revision_not_backdated_and_conflict_rejected(self):
        original = deepcopy(
            next(row for row in self.rows if row["reference_month"] == "2024-09" and row["kind"] == "headline")
        )
        revision = dict(original, value=9.9, vintage_utc="2024-10-23T12:30:00+00:00")
        self.assertEqual(self.selection(2024, [original, revision])["row"]["value"], 2.4)
        self.assertEqual(self.selection(2024, [revision])["status"], "unavailable")
        with self.assertRaisesRegex(ValueError, "conflicting"):
            self.selection(2024, [original, dict(original, value=9.9)])
        self.assertEqual(self.selection(2024, [])["status"], "unavailable")
        self.assertEqual(
            self.selection(2024, [row for row in self.rows if row["reference_month"] == "2018-09"])["status"],
            "unavailable",
        )

    def test_exact_ninety_day_staleness_boundary(self):
        row = deepcopy(
            next(row for row in self.rows if row["reference_month"] == "2024-09" and row["kind"] == "headline")
        )
        boundary = timestamp(row["release_utc"]) + timedelta(days=90)
        election = (boundary + timedelta(days=14)).date().isoformat()
        args = dict(election_date=election, horizon_days=14, series_id=SERIES["headline"])
        self.assertEqual(select([row], cutoff_utc=boundary.isoformat(), **args)["status"], "available")
        self.assertEqual(
            select([row], cutoff_utc=(boundary + timedelta(seconds=1)).isoformat(), **args)["status"], "unavailable"
        )

    def test_complete_revision_marked_numeric_cells(self):
        # Synthetic older Table1 row: relative importance, two NSA indexes,
        # NSA year-over-year, monthly NSA, and three monthly SA cells.
        values = parse_table_cells("100.000 R172.8 173.7 R2.5 0.5 .2 -0.1 0.5r")
        self.assertEqual(values, [100.0, 172.8, 173.7, 2.5, 0.5, 0.2, -0.1, 0.5])
        self.assertEqual(values[3], 2.5)
        for token in ("X2.5", "R2.5junk", "foo172.8", "2.5.1"):
            with self.assertRaises(ValueError):
                parse_table_cells(token)

    def test_raw_digest_and_header_month_integrity(self):
        record = self.acquisition["records"][-1]
        source = next(item for item in record["attempts"] if item.get("file"))
        with self.assertRaisesRegex(ValueError, "digest"):
            parse_release(HERE / source["file"], record, dict(source, sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "reference month"):
            parse_release(HERE / source["file"], dict(record, reference_month="2024-08"), source)
        receipt = json.loads((HERE / "receipt.json").read_text())
        for pin in receipt["source_files"]:
            self.assertEqual(hashlib.sha256((HERE / pin["file"]).read_bytes()).hexdigest(), pin["sha256"])
        self.assertEqual(
            sum(
                item["status"] == "fetch_failed"
                for record in self.acquisition["records"]
                for item in record["attempts"]
            ),
            15,
        )


if __name__ == "__main__":
    unittest.main()
