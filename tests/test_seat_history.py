"""Tests for the GET /forecast/seat-history endpoint.

Uses temporary directories to simulate forecast snapshots — no real filesystem
state required.  The endpoint reads from SNAPSHOTS_DIR which is monkey-patched
via the module attribute.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from api.routers.forecast.seat_history import (
    _compute_seat_counts,
    get_seat_history,
    SNAPSHOTS_DIR,
)
import api.routers.forecast.seat_history as seat_history_module


# ---------------------------------------------------------------------------
# _compute_seat_counts unit tests
# ---------------------------------------------------------------------------


class TestComputeSeatCounts:
    """Unit tests for the seat projection helper."""

    def test_empty_predictions_falls_back_to_incumbents(self):
        """Empty predictions dict → every seat falls back to incumbent hold."""
        dem, gop = _compute_seat_counts({})
        # Incumbent holds should add up: total Senate = 100
        assert dem + gop == 100

    def test_all_dem_predictions(self):
        """All Senate predictions > 0.5 → Dems win all contested seats."""
        preds = {f"2026 {st} Senate": 0.99 for st in [
            "AL", "AK", "AR", "CO", "DE", "GA", "IA", "ID", "IL", "KS",
            "KY", "LA", "MA", "ME", "MI", "MN", "MS", "MT", "NC", "NE",
            "NH", "NJ", "NM", "OK", "OR", "RI", "SC", "SD", "TN", "TX",
            "VA", "WV", "WY",
        ]}
        dem, gop = _compute_seat_counts(preds)
        # Dems win all 33 contested + holdover seats
        assert dem > gop
        assert dem + gop == 100

    def test_all_gop_predictions(self):
        """All Senate predictions < 0.5 → GOP wins all contested seats."""
        preds = {f"2026 {st} Senate": 0.01 for st in [
            "AL", "AK", "AR", "CO", "DE", "GA", "IA", "ID", "IL", "KS",
            "KY", "LA", "MA", "ME", "MI", "MN", "MS", "MT", "NC", "NE",
            "NH", "NJ", "NM", "OK", "OR", "RI", "SC", "SD", "TN", "TX",
            "VA", "WV", "WY",
        ]}
        dem, gop = _compute_seat_counts(preds)
        assert gop > dem
        assert dem + gop == 100

    def test_seat_totals_always_sum_to_100(self):
        """D + R should always equal 100 seats regardless of prediction values."""
        preds = {
            "2026 FL Senate": 0.52,
            "2026 GA Senate": 0.48,
            "2026 NC Senate": 0.501,
        }
        dem, gop = _compute_seat_counts(preds)
        assert dem + gop == 100

    def test_ignores_non_senate_races(self):
        """Governor and baseline races in the predictions dict are ignored."""
        preds = {
            "2026 FL Governor": 0.99,
            "2026 TX Governor": 0.01,
            "baseline": 0.5,
        }
        dem_with_noise, gop_with_noise = _compute_seat_counts(preds)
        dem_clean, gop_clean = _compute_seat_counts({})
        # Only Senate races affect the count; non-Senate entries are invisible
        assert dem_with_noise == dem_clean
        assert gop_with_noise == gop_clean


# ---------------------------------------------------------------------------
# get_seat_history endpoint tests (monkeypatched SNAPSHOTS_DIR)
# ---------------------------------------------------------------------------


class TestGetSeatHistory:
    def test_returns_empty_when_no_snapshots_dir(self, monkeypatch, tmp_path):
        """Non-existent snapshots dir → empty list."""
        monkeypatch.setattr(seat_history_module, "SNAPSHOTS_DIR", tmp_path / "nonexistent")
        result = get_seat_history()
        assert result == []

    def test_returns_empty_when_dir_has_no_json_files(self, monkeypatch, tmp_path):
        """Empty directory → empty list."""
        monkeypatch.setattr(seat_history_module, "SNAPSHOTS_DIR", tmp_path)
        result = get_seat_history()
        assert result == []

    def test_returns_empty_for_non_json_files(self, monkeypatch, tmp_path):
        """Non-JSON files in the directory are silently ignored."""
        (tmp_path / "notes.txt").write_text("not json")
        monkeypatch.setattr(seat_history_module, "SNAPSHOTS_DIR", tmp_path)
        result = get_seat_history()
        assert result == []

    def test_single_snapshot_returns_one_entry(self, monkeypatch, tmp_path):
        """One snapshot file → one entry in chronological order."""
        snapshot = {
            "date": "2026-03-01",
            "predictions": {"2026 FL Senate": 0.55},
        }
        (tmp_path / "2026-03-01.json").write_text(json.dumps(snapshot))
        monkeypatch.setattr(seat_history_module, "SNAPSHOTS_DIR", tmp_path)
        result = get_seat_history()
        assert len(result) == 1
        assert result[0]["date"] == "2026-03-01"
        assert result[0]["dem_projected"] + result[0]["gop_projected"] == 100

    def test_multiple_snapshots_sorted_chronologically(self, monkeypatch, tmp_path):
        """Multiple snapshots are sorted oldest-first by filename."""
        for date in ["2026-04-01", "2026-03-15", "2026-03-29"]:
            snap = {"date": date, "predictions": {}}
            (tmp_path / f"{date}.json").write_text(json.dumps(snap))

        monkeypatch.setattr(seat_history_module, "SNAPSHOTS_DIR", tmp_path)
        result = get_seat_history()
        assert len(result) == 3
        dates = [r["date"] for r in result]
        assert dates == sorted(dates)

    def test_snapshot_missing_date_field_is_skipped(self, monkeypatch, tmp_path):
        """Snapshots without a 'date' field are silently skipped."""
        bad = {"predictions": {"2026 FL Senate": 0.5}}
        good = {"date": "2026-04-01", "predictions": {}}
        (tmp_path / "2026-03-01.json").write_text(json.dumps(bad))
        (tmp_path / "2026-04-01.json").write_text(json.dumps(good))
        monkeypatch.setattr(seat_history_module, "SNAPSHOTS_DIR", tmp_path)
        result = get_seat_history()
        assert len(result) == 1
        assert result[0]["date"] == "2026-04-01"

    def test_malformed_json_file_is_skipped(self, monkeypatch, tmp_path):
        """Files with invalid JSON are skipped without crashing."""
        (tmp_path / "2026-03-01.json").write_text("{bad json!!")
        (tmp_path / "2026-04-01.json").write_text(json.dumps({
            "date": "2026-04-01",
            "predictions": {},
        }))
        monkeypatch.setattr(seat_history_module, "SNAPSHOTS_DIR", tmp_path)
        result = get_seat_history()
        assert len(result) == 1

    def test_seat_counts_change_between_snapshots(self, monkeypatch, tmp_path):
        """Seat counts update as new poll data changes predictions."""
        # Snapshot 1: FL + GA Senate both lean R (GOP wins both)
        snap1 = {
            "date": "2026-03-01",
            "predictions": {
                "2026 FL Senate": 0.48,
                "2026 GA Senate": 0.47,
            },
        }
        # Snapshot 2: FL + GA Senate both flip Dem
        snap2 = {
            "date": "2026-04-01",
            "predictions": {
                "2026 FL Senate": 0.53,
                "2026 GA Senate": 0.54,
            },
        }
        (tmp_path / "2026-03-01.json").write_text(json.dumps(snap1))
        (tmp_path / "2026-04-01.json").write_text(json.dumps(snap2))

        monkeypatch.setattr(seat_history_module, "SNAPSHOTS_DIR", tmp_path)
        result = get_seat_history()

        assert len(result) == 2
        # After polling shift, Dem count should be higher in snapshot 2
        assert result[1]["dem_projected"] > result[0]["dem_projected"]
        assert result[1]["gop_projected"] < result[0]["gop_projected"]

    def test_entry_schema(self, monkeypatch, tmp_path):
        """Each entry has the expected keys with correct types."""
        snap = {
            "date": "2026-04-01",
            "predictions": {"2026 NC Senate": 0.61},
        }
        (tmp_path / "2026-04-01.json").write_text(json.dumps(snap))
        monkeypatch.setattr(seat_history_module, "SNAPSHOTS_DIR", tmp_path)
        result = get_seat_history()

        assert len(result) == 1
        entry = result[0]
        assert set(entry.keys()) == {"date", "dem_projected", "gop_projected"}
        assert isinstance(entry["date"], str)
        assert isinstance(entry["dem_projected"], int)
        assert isinstance(entry["gop_projected"], int)
