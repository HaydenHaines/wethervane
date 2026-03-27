"""Tests for scripts/poll_scrape_notify.py."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

# The notify helper lives in scripts/, not a proper package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from poll_scrape_notify import (
    _read_race_counts,
    build_diff_message,
    build_failure_message,
    snapshot,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_polls_csv(path: Path, rows: list[dict]) -> None:
    """Write a minimal polls CSV with just the columns we need."""
    fieldnames = ["race", "geography", "geo_level", "dem_share", "n_sample", "date", "pollster", "notes"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({f: row.get(f, "") for f in fieldnames})


def _poll_row(race: str, date: str = "2026-01-01", pollster: str = "Cygnal") -> dict:
    return {
        "race": race,
        "geography": race.split()[1],
        "geo_level": "state",
        "dem_share": "0.45",
        "n_sample": "800",
        "date": date,
        "pollster": pollster,
        "notes": "",
    }


# ---------------------------------------------------------------------------
# _read_race_counts
# ---------------------------------------------------------------------------

class TestReadRaceCounts:
    def test_missing_file_returns_empty(self, tmp_path):
        counts = _read_race_counts(tmp_path / "nonexistent.csv")
        assert counts == {}

    def test_counts_by_race(self, tmp_path):
        csv_path = tmp_path / "polls.csv"
        _write_polls_csv(csv_path, [
            _poll_row("2026 FL Governor"),
            _poll_row("2026 FL Governor"),
            _poll_row("2026 GA Senate"),
        ])
        counts = _read_race_counts(csv_path)
        assert counts == {"2026 FL Governor": 2, "2026 GA Senate": 1}

    def test_empty_csv_returns_empty(self, tmp_path):
        csv_path = tmp_path / "polls.csv"
        _write_polls_csv(csv_path, [])
        counts = _read_race_counts(csv_path)
        assert counts == {}

    def test_blank_race_field_ignored(self, tmp_path):
        csv_path = tmp_path / "polls.csv"
        # Row with blank race should be skipped
        _write_polls_csv(csv_path, [
            _poll_row("2026 FL Governor"),
            {"race": "", "geography": "", "geo_level": "", "dem_share": "", "n_sample": "", "date": "", "pollster": "", "notes": ""},
        ])
        counts = _read_race_counts(csv_path)
        assert counts == {"2026 FL Governor": 1}

    def test_whitespace_in_race_stripped(self, tmp_path):
        csv_path = tmp_path / "polls.csv"
        _write_polls_csv(csv_path, [
            {"race": "  2026 FL Governor  ", "geography": "FL", "geo_level": "state", "dem_share": "0.45", "n_sample": "800", "date": "2026-01-01", "pollster": "X", "notes": ""},
        ])
        counts = _read_race_counts(csv_path)
        assert "2026 FL Governor" in counts


# ---------------------------------------------------------------------------
# snapshot
# ---------------------------------------------------------------------------

class TestSnapshot:
    def test_returns_valid_json(self, tmp_path):
        csv_path = tmp_path / "polls.csv"
        _write_polls_csv(csv_path, [_poll_row("2026 FL Governor")])
        result = snapshot(csv_path)
        data = json.loads(result)  # must not raise
        assert data == {"2026 FL Governor": 1}

    def test_missing_file_gives_empty_dict(self, tmp_path):
        result = snapshot(tmp_path / "nofile.csv")
        assert json.loads(result) == {}


# ---------------------------------------------------------------------------
# build_diff_message
# ---------------------------------------------------------------------------

class TestBuildDiffMessage:
    def test_no_new_polls(self, tmp_path):
        csv_path = tmp_path / "polls.csv"
        _write_polls_csv(csv_path, [_poll_row("2026 FL Governor")])
        before = json.dumps({"2026 FL Governor": 1})
        msg = build_diff_message(before, csv_path)
        assert "No new polls" in msg
        assert "1 total" in msg

    def test_new_polls_reported(self, tmp_path):
        csv_path = tmp_path / "polls.csv"
        _write_polls_csv(csv_path, [
            _poll_row("2026 FL Governor"),
            _poll_row("2026 FL Governor"),
            _poll_row("2026 GA Senate"),
        ])
        before = json.dumps({"2026 FL Governor": 1})
        msg = build_diff_message(before, csv_path)
        assert "+2 new polls" in msg
        assert "3 total" in msg
        assert "2026 FL Governor (+1)" in msg
        assert "2026 GA Senate (+1)" in msg

    def test_multiple_new_races(self, tmp_path):
        csv_path = tmp_path / "polls.csv"
        _write_polls_csv(csv_path, [
            _poll_row("2026 FL Governor"),
            _poll_row("2026 GA Senate"),
            _poll_row("2026 MI Senate"),
        ])
        before = json.dumps({})
        msg = build_diff_message(before, csv_path)
        assert "+3 new polls" in msg
        assert "2026 FL Governor (+1)" in msg
        assert "2026 GA Senate (+1)" in msg
        assert "2026 MI Senate (+1)" in msg

    def test_invalid_before_json_handled_gracefully(self, tmp_path):
        csv_path = tmp_path / "polls.csv"
        _write_polls_csv(csv_path, [_poll_row("2026 FL Governor")])
        # Corrupted snapshot — should not raise, should treat as empty before
        msg = build_diff_message("{not valid json!", csv_path)
        assert "+1 new polls" in msg

    def test_empty_before_json_treats_all_as_new(self, tmp_path):
        csv_path = tmp_path / "polls.csv"
        _write_polls_csv(csv_path, [
            _poll_row("2026 FL Governor"),
            _poll_row("2026 FL Governor"),
        ])
        msg = build_diff_message("{}", csv_path)
        assert "+2 new polls" in msg

    def test_message_contains_telegram_prefix(self, tmp_path):
        csv_path = tmp_path / "polls.csv"
        _write_polls_csv(csv_path, [_poll_row("2026 FL Governor")])
        before = json.dumps({})
        msg = build_diff_message(before, csv_path)
        assert msg.startswith("low-priority status update:")

    def test_no_new_polls_message_contains_prefix(self, tmp_path):
        csv_path = tmp_path / "polls.csv"
        _write_polls_csv(csv_path, [_poll_row("2026 FL Governor")])
        before = json.dumps({"2026 FL Governor": 1})
        msg = build_diff_message(before, csv_path)
        assert msg.startswith("low-priority status update:")

    def test_polls_removed_counted_as_zero_new(self, tmp_path):
        """If rows were removed (unusual but possible), report 0 new — not negative."""
        csv_path = tmp_path / "polls.csv"
        _write_polls_csv(csv_path, [_poll_row("2026 FL Governor")])
        # before claimed 5, after has 1 — net is -4, should report "no new"
        before = json.dumps({"2026 FL Governor": 5})
        msg = build_diff_message(before, csv_path)
        assert "No new polls" in msg

    def test_missing_csv_after_scrape(self, tmp_path):
        """Missing output CSV should not crash — treat as 0 polls."""
        msg = build_diff_message("{}", tmp_path / "missing.csv")
        assert "No new polls" in msg or "0 total" in msg


# ---------------------------------------------------------------------------
# build_failure_message
# ---------------------------------------------------------------------------

class TestBuildFailureMessage:
    def test_contains_stage(self):
        msg = build_failure_message("DuckDB rebuild")
        assert "DuckDB rebuild" in msg

    def test_starts_with_urgent(self):
        msg = build_failure_message("poll scrape")
        assert msg.startswith("URGENT blocking issue:")

    def test_contains_log_path(self):
        msg = build_failure_message("API restart")
        assert "wethervane-poll-scrape.log" in msg

    def test_contains_meat_puppet(self):
        msg = build_failure_message("anything")
        assert "meat puppet" in msg
