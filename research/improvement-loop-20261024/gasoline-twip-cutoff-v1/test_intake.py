"""Offline archive/selector/contrast checks, without forecast fitting."""

import hashlib
import json
import unittest
from copy import deepcopy
from datetime import date
from decimal import Decimal

from intake import (
    DATE_EXCEPTION_2004,
    HERE,
    SERIES,
    Rows,
    availability_bound,
    combine,
    issue_date_exception,
    parse_html,
    parse_index,
)


class IntakeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = json.loads((HERE / "panel.json").read_text())["panel"]

    def row(self):
        return deepcopy(next(row for row in self.panel if row["cycle"] == 2024))

    def test_real_pre_cutoff_four_cycle_values_and_independent_log_arithmetic(self):
        self.assertEqual(len(self.panel), 12)
        self.assertTrue(all(row["status"] == "available" for row in self.panel))
        expected = {
            2018: ("2018-10-17", "2018-10-15", 2.879, "2018-07-16", 2.865),
            2020: ("2020-10-15", "2020-10-12", 2.167, "2020-07-13", 2.195),
            2022: ("2022-10-19", "2022-10-17", 3.871, "2022-07-18", 4.490),
            2024: ("2024-10-17", "2024-10-14", 3.171, "2024-07-15", 3.496),
        }
        for row in self.panel:
            if row["status"] != "available":
                continue
            current, lag = row["current"], row["lag13"]
            self.assertLess(current["issue_date"], row["cutoff_utc"][:10])
            if row["cycle"] in expected:
                self.assertEqual(
                    (
                        current["issue_date"],
                        current["observation_date"],
                        current["value"],
                        lag["observation_date"],
                        lag["value"],
                    ),
                    expected[row["cycle"]],
                )
            decimal_log = (Decimal(str(current["value"])) / Decimal(str(lag["value"]))).ln()
            self.assertAlmostEqual(row["log_change_13_weeks"], float(decimal_log), places=14)
            self.assertFalse(row["crosses_2018_method_break"])

    def test_actual_index_latest_issue_and_raw_hash_reparse(self):
        issues = parse_index((HERE / "raw/index.html").read_bytes())
        for row in self.panel:
            if row["status"] != "available":
                continue
            latest = max(day for day in issues if day.isoformat() < row["cutoff_utc"][:10])
            self.assertEqual(latest.isoformat(), row["current"]["issue_date"])
            for part in ("current", "lag13"):
                expected = row[part]
                source = expected["source"]
                actual = parse_html(
                    (HERE / source["file"]).read_bytes(),
                    source,
                    date_exception=issue_date_exception((HERE / "raw/index.html").read_bytes(), source),
                )
                self.assertEqual(actual, expected)
        for pin in json.loads((HERE / "receipt.json").read_text())["files"]:
            self.assertEqual(hashlib.sha256((HERE / pin["file"]).read_bytes()).hexdigest(), pin["sha256"])

    def test_2004_explicit_exception_selects_gasoline_not_heating_oil(self):
        row = next(row for row in self.panel if row["cycle"] == 2004)
        self.assertEqual(row["status"], "available")
        self.assertEqual(row["current"]["source_value"], "199.3")
        self.assertEqual(row["current"]["value"], 1.993)
        self.assertEqual(row["lag13"]["value"], 1.917)
        self.assertAlmostEqual(row["log_change_13_weeks"], 0.038879577167, places=12)
        self.assertEqual(row["current"]["source_header_issue_date"], "2003-10-14")
        self.assertEqual(row["current"]["issue_date"], "2004-10-14")
        self.assertFalse(row["current"]["original_posting_log_authenticated"])
        self.assertEqual(row["current"]["issue_date_exception"]["id"], DATE_EXCEPTION_2004["id"])
        data = (HERE / "raw/2004-10-14.html").read_bytes()
        parser = Rows()
        parser.feed(data.decode("utf-8", errors="replace"))
        gasoline = next(cells for cells in parser.rows if cells and cells[0] == "Gasoline")
        self.assertEqual(gasoline[1], "199.3")
        self.assertEqual(gasoline[4:6], ["Heating Oil", "189.9"])
        source = row["current"]["source"]
        with self.assertRaisesRegex(ValueError, "issue date disagrees"):
            parse_html(data, source)
        with self.assertRaisesRegex(ValueError, "index digest"):
            issue_date_exception(b"unbound index", source)
        with self.assertRaisesRegex(ValueError, "issue date disagrees"):
            parse_html(data, dict(source, url="https://www.eia.gov/other"), DATE_EXCEPTION_2004)

    def test_future_source_wrong_series_units_dates_and_price_rejected(self):
        row = self.row()
        for changes in [
            dict(series_id="current_history"),
            dict(units="cents_per_gallon"),
            dict(value=float("nan")),
            dict(value=True),
            dict(observation_date="2024-10-15"),
            dict(observation_date="2024-10-21"),
        ]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                combine(dict(row["current"], **changes), row["lag13"], row["cutoff_utc"])
        current = dict(
            row["current"],
            issue_date="2024-10-23",
            available_upper_bound_utc=availability_bound(date(2024, 10, 23)).isoformat(),
        )
        with self.assertRaisesRegex(ValueError, "post-cutoff"):
            combine(current, row["lag13"], row["cutoff_utc"])
        with self.assertRaisesRegex(ValueError, "bound"):
            combine(
                dict(row["current"], available_upper_bound_utc="2024-10-17T00:00:00+00:00"),
                row["lag13"],
                row["cutoff_utc"],
            )

    def test_synthetic_legacy_cents_and_unrecognized_units(self):
        template = (
            "<title>This Week In Petroleum</title><p>Released on October 17, 2002</p>"
            "<p>U.S. average regular gasoline</p><!-- RETAIL TABLE --><table>"
            "<tr><td>Retail Prices (UNIT Per Gallon)</td></tr><tr><td>10/14/02</td></tr>"
            "<tr><td>Gasoline</td><td>144.0</td></tr></table>"
        )
        for unit in ("Cents", "Euros"):
            data = template.replace("UNIT", unit).encode()
            source = dict(sha256=hashlib.sha256(data).hexdigest(), issue_date="2002-10-17")
            if unit == "Cents":
                row = parse_html(data, source)
                self.assertEqual(row["value"], 1.44)
                self.assertEqual(row["source_value"], "144.0")
                self.assertEqual(row["series_id"], SERIES)
            else:
                with self.assertRaisesRegex(ValueError, "units"):
                    parse_html(data, source)


if __name__ == "__main__":
    unittest.main()
