"""Tests for scripts/check_prediction_changes.py

The script:
- Loads the PREVIOUS prediction snapshot from data/forecast_snapshots/
- Loads the CURRENT predictions
- Compares race-level predictions (state_pred or equivalent)
- For any race where the prediction changed by >2pp, formats a Telegram message
- Calls ~/scripts/notify.sh with the formatted message (if any changes)
- Saves the current snapshot for next comparison
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

# Import from the module under test
import sys

# Add scripts dir to path so we can import the module
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from check_prediction_changes import (
    detect_changes,
    find_latest_snapshot,
    format_alert_message,
    load_snapshot,
    save_snapshot,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_predictions():
    """Sample race predictions in the format used by forecast_snapshots/."""
    return {
        "2026 AK Governor": 0.41885,
        "2026 AZ Governor": 0.5115,
        "2026 GA Senate": 0.5839,
        "2026 NC Senate": 0.6180,
        "2026 PA Governor": 0.5696,
    }


@pytest.fixture
def snapshot_dir(tmp_path):
    """Return a temporary snapshot directory."""
    d = tmp_path / "forecast_snapshots"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# load_snapshot / save_snapshot tests
# ---------------------------------------------------------------------------


class TestLoadSnapshot:
    def test_loads_valid_snapshot_file(self, tmp_path, sample_predictions):
        f = tmp_path / "2026-04-03.json"
        f.write_text(
            json.dumps({"date": "2026-04-03", "predictions": sample_predictions})
        )
        result = load_snapshot(f)
        assert result == sample_predictions

    def test_returns_none_for_missing_file(self, tmp_path):
        result = load_snapshot(tmp_path / "nonexistent.json")
        assert result is None

    def test_returns_none_for_invalid_json(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("not json {{{")
        result = load_snapshot(f)
        assert result is None

    def test_returns_none_when_predictions_key_missing(self, tmp_path):
        f = tmp_path / "no_predictions.json"
        f.write_text(json.dumps({"date": "2026-04-03", "other": "data"}))
        result = load_snapshot(f)
        assert result is None

    def test_handles_flat_dict_snapshot(self, tmp_path, sample_predictions):
        """Flat dict (no 'predictions' wrapper) is also valid — legacy format."""
        f = tmp_path / "flat.json"
        f.write_text(json.dumps(sample_predictions))
        result = load_snapshot(f)
        assert result == sample_predictions


class TestSaveSnapshot:
    def test_saves_predictions_to_json_file(self, tmp_path, sample_predictions):
        out = tmp_path / "snapshot.json"
        save_snapshot(sample_predictions, out)
        data = json.loads(out.read_text())
        assert "predictions" in data
        assert data["predictions"] == sample_predictions

    def test_includes_date_field(self, tmp_path, sample_predictions):
        out = tmp_path / "snapshot.json"
        save_snapshot(sample_predictions, out)
        data = json.loads(out.read_text())
        assert "date" in data

    def test_creates_parent_dirs(self, tmp_path, sample_predictions):
        out = tmp_path / "nested" / "dir" / "snapshot.json"
        save_snapshot(sample_predictions, out)
        assert out.exists()

    def test_overwrites_existing_file(self, tmp_path):
        out = tmp_path / "snapshot.json"
        out.write_text(json.dumps({"predictions": {"old_race": 0.5}}))
        new_preds = {"new_race": 0.6}
        save_snapshot(new_preds, out)
        data = json.loads(out.read_text())
        assert "new_race" in data["predictions"]
        assert "old_race" not in data["predictions"]


# ---------------------------------------------------------------------------
# find_latest_snapshot tests
# ---------------------------------------------------------------------------


class TestFindLatestSnapshot:
    def test_returns_none_for_empty_dir(self, snapshot_dir):
        result = find_latest_snapshot(snapshot_dir)
        assert result is None

    def test_returns_most_recent_json_file(self, snapshot_dir):
        (snapshot_dir / "2026-03-29.json").write_text("{}")
        (snapshot_dir / "2026-04-03.json").write_text("{}")
        (snapshot_dir / "2026-03-20.json").write_text("{}")
        result = find_latest_snapshot(snapshot_dir)
        assert result.name == "2026-04-03.json"

    def test_ignores_non_json_files(self, snapshot_dir):
        (snapshot_dir / "2026-04-03.json").write_text("{}")
        (snapshot_dir / "not-a-snapshot.txt").write_text("{}")
        result = find_latest_snapshot(snapshot_dir)
        assert result.name == "2026-04-03.json"

    def test_returns_none_when_no_json_files(self, snapshot_dir):
        (snapshot_dir / "file.txt").write_text("hello")
        result = find_latest_snapshot(snapshot_dir)
        assert result is None

    def test_single_file_returned(self, snapshot_dir):
        (snapshot_dir / "2026-03-29.json").write_text("{}")
        result = find_latest_snapshot(snapshot_dir)
        assert result.name == "2026-03-29.json"


# ---------------------------------------------------------------------------
# detect_changes tests
# ---------------------------------------------------------------------------


class TestDetectChanges:
    THRESHOLD = 0.02  # 2 percentage points

    def test_no_changes_returns_empty_list(self, sample_predictions):
        result = detect_changes(sample_predictions, sample_predictions, self.THRESHOLD)
        assert result == []

    def test_change_below_threshold_excluded(self):
        before = {"2026 AZ Governor": 0.5000}
        after = {"2026 AZ Governor": 0.5100}  # +1pp — below 2pp threshold
        result = detect_changes(before, after, self.THRESHOLD)
        assert result == []

    def test_change_above_threshold_included(self):
        before = {"2026 AZ Governor": 0.5000}
        after = {"2026 AZ Governor": 0.5250}  # +2.5pp — above 2pp threshold
        result = detect_changes(before, after, self.THRESHOLD)
        assert len(result) == 1
        change = result[0]
        assert change["race"] == "2026 AZ Governor"
        assert abs(change["before"] - 0.5000) < 1e-9
        assert abs(change["after"] - 0.5250) < 1e-9
        assert abs(change["delta"] - 0.0250) < 1e-9

    def test_exactly_at_threshold_included(self):
        before = {"2026 AZ Governor": 0.5000}
        after = {"2026 AZ Governor": 0.5200}  # exactly 2pp
        result = detect_changes(before, after, self.THRESHOLD)
        assert len(result) == 1

    def test_negative_change_detected(self):
        before = {"2026 NC Senate": 0.6180}
        after = {"2026 NC Senate": 0.5900}  # -2.8pp
        result = detect_changes(before, after, self.THRESHOLD)
        assert len(result) == 1
        assert result[0]["delta"] < 0

    def test_multiple_races_only_big_changes_returned(self):
        before = {
            "2026 AZ Governor": 0.5000,
            "2026 GA Senate": 0.5839,
            "2026 NC Senate": 0.6180,
        }
        after = {
            "2026 AZ Governor": 0.5250,  # +2.5pp — above threshold
            "2026 GA Senate": 0.5845,    # +0.06pp — below threshold
            "2026 NC Senate": 0.5900,    # -2.8pp — above threshold
        }
        result = detect_changes(before, after, self.THRESHOLD)
        races = {r["race"] for r in result}
        assert "2026 AZ Governor" in races
        assert "2026 NC Senate" in races
        assert "2026 GA Senate" not in races

    def test_sorted_by_descending_absolute_delta(self):
        before = {
            "2026 AZ Governor": 0.5000,
            "2026 NC Senate": 0.6180,
        }
        after = {
            "2026 AZ Governor": 0.5250,  # +2.5pp
            "2026 NC Senate": 0.5900,    # -2.8pp (larger)
        }
        result = detect_changes(before, after, self.THRESHOLD)
        assert result[0]["race"] == "2026 NC Senate"
        assert result[1]["race"] == "2026 AZ Governor"

    def test_handles_empty_before(self):
        after = {"2026 AZ Governor": 0.5250}
        result = detect_changes({}, after, self.THRESHOLD)
        # New race with no prior — no delta to compare, should not error
        # Implementation may choose to include or exclude; test stability
        assert isinstance(result, list)

    def test_handles_empty_after(self):
        before = {"2026 AZ Governor": 0.5000}
        result = detect_changes(before, {}, self.THRESHOLD)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# format_alert_message tests
# ---------------------------------------------------------------------------


class TestFormatAlertMessage:
    def test_single_positive_change(self):
        changes = [
            {"race": "2026 AZ Governor", "before": 0.50, "after": 0.53, "delta": 0.03}
        ]
        msg = format_alert_message(changes)
        assert "AZ Governor" in msg
        assert "50.0%" in msg
        assert "53.0%" in msg
        assert "+" in msg  # direction indicator

    def test_single_negative_change(self):
        changes = [
            {"race": "2026 NC Senate", "before": 0.618, "after": 0.590, "delta": -0.028}
        ]
        msg = format_alert_message(changes)
        assert "NC Senate" in msg
        assert "61.8%" in msg
        assert "59.0%" in msg

    def test_multiple_changes_all_included(self):
        changes = [
            {"race": "2026 AZ Governor", "before": 0.50, "after": 0.53, "delta": 0.03},
            {"race": "2026 NC Senate", "before": 0.618, "after": 0.590, "delta": -0.028},
        ]
        msg = format_alert_message(changes)
        assert "AZ Governor" in msg
        assert "NC Senate" in msg

    def test_message_is_string(self):
        changes = [
            {"race": "2026 AZ Governor", "before": 0.50, "after": 0.53, "delta": 0.03}
        ]
        msg = format_alert_message(changes)
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_includes_count_of_changes(self):
        changes = [
            {"race": "2026 AZ Governor", "before": 0.50, "after": 0.53, "delta": 0.03},
            {"race": "2026 NC Senate", "before": 0.618, "after": 0.590, "delta": -0.028},
        ]
        msg = format_alert_message(changes)
        assert "2" in msg  # count of races changed

    def test_message_has_telegram_prefix(self):
        changes = [
            {"race": "2026 AZ Governor", "before": 0.50, "after": 0.53, "delta": 0.03}
        ]
        msg = format_alert_message(changes)
        # Should start with a prefix label like "low-priority status update:" or similar
        assert ":" in msg[:60]

    def test_positive_delta_has_plus_sign(self):
        changes = [
            {"race": "2026 AZ Governor", "before": 0.50, "after": 0.53, "delta": 0.03}
        ]
        msg = format_alert_message(changes)
        assert "+3.0pp" in msg or "+3.0%" in msg or "+3.0" in msg

    def test_magnitude_shown_as_pp(self):
        changes = [
            {"race": "2026 NC Senate", "before": 0.618, "after": 0.590, "delta": -0.028}
        ]
        msg = format_alert_message(changes)
        # Should show magnitude in pp (percentage points), e.g. -2.8pp
        assert "2.8" in msg


# ---------------------------------------------------------------------------
# Integration / main flow tests
# ---------------------------------------------------------------------------


class TestMainFlow:
    """Tests for the high-level check_prediction_changes() orchestration function."""

    def test_sends_alert_when_changes_exceed_threshold(self, tmp_path):
        from check_prediction_changes import check_prediction_changes

        snapshot_dir = tmp_path / "forecast_snapshots"
        snapshot_dir.mkdir()

        # Create a previous snapshot
        prev_predictions = {
            "2026 AZ Governor": 0.5000,
            "2026 GA Senate": 0.5839,
        }
        prev_file = snapshot_dir / "2026-03-29.json"
        prev_file.write_text(
            json.dumps({"date": "2026-03-29", "predictions": prev_predictions})
        )

        # Current predictions have a big change in AZ Governor
        current_predictions = {
            "2026 AZ Governor": 0.5300,  # +3pp
            "2026 GA Senate": 0.5845,    # tiny change
        }

        with patch("check_prediction_changes.send_notification") as mock_notify:
            check_prediction_changes(
                current_predictions=current_predictions,
                snapshot_dir=snapshot_dir,
                threshold=0.02,
                notify=True,
            )
            mock_notify.assert_called_once()
            msg = mock_notify.call_args[0][0]
            assert "AZ Governor" in msg

    def test_no_alert_when_all_changes_small(self, tmp_path):
        from check_prediction_changes import check_prediction_changes

        snapshot_dir = tmp_path / "forecast_snapshots"
        snapshot_dir.mkdir()

        prev_predictions = {"2026 AZ Governor": 0.5000}
        prev_file = snapshot_dir / "2026-03-29.json"
        prev_file.write_text(
            json.dumps({"date": "2026-03-29", "predictions": prev_predictions})
        )

        current_predictions = {"2026 AZ Governor": 0.5050}  # +0.5pp only

        with patch("check_prediction_changes.send_notification") as mock_notify:
            check_prediction_changes(
                current_predictions=current_predictions,
                snapshot_dir=snapshot_dir,
                threshold=0.02,
                notify=True,
            )
            mock_notify.assert_not_called()

    def test_saves_new_snapshot_after_run(self, tmp_path):
        from check_prediction_changes import check_prediction_changes

        snapshot_dir = tmp_path / "forecast_snapshots"
        snapshot_dir.mkdir()

        prev_predictions = {"2026 AZ Governor": 0.5000}
        prev_file = snapshot_dir / "2026-03-29.json"
        prev_file.write_text(
            json.dumps({"date": "2026-03-29", "predictions": prev_predictions})
        )

        current_predictions = {"2026 AZ Governor": 0.5300}  # big change

        with patch("check_prediction_changes.send_notification"):
            check_prediction_changes(
                current_predictions=current_predictions,
                snapshot_dir=snapshot_dir,
                threshold=0.02,
                notify=True,
            )

        # A new snapshot file should exist, dated today
        snapshots = list(snapshot_dir.glob("*.json"))
        assert len(snapshots) >= 2  # original + new one
        # New snapshot should contain the current predictions
        # Find the newest file that isn't the original
        new_snapshots = [s for s in snapshots if s.name != "2026-03-29.json"]
        assert len(new_snapshots) == 1
        data = json.loads(new_snapshots[0].read_text())
        assert data["predictions"]["2026 AZ Governor"] == pytest.approx(0.5300)

    def test_no_previous_snapshot_runs_silently(self, tmp_path):
        """First run: no prior snapshot → just save current, no alert."""
        from check_prediction_changes import check_prediction_changes

        snapshot_dir = tmp_path / "forecast_snapshots"
        snapshot_dir.mkdir()

        current_predictions = {"2026 AZ Governor": 0.5300}

        with patch("check_prediction_changes.send_notification") as mock_notify:
            check_prediction_changes(
                current_predictions=current_predictions,
                snapshot_dir=snapshot_dir,
                threshold=0.02,
                notify=True,
            )
            mock_notify.assert_not_called()

        # Should still save the snapshot
        snapshots = list(snapshot_dir.glob("*.json"))
        assert len(snapshots) == 1

    def test_snapshot_dir_created_if_missing(self, tmp_path):
        from check_prediction_changes import check_prediction_changes

        snapshot_dir = tmp_path / "nonexistent" / "forecast_snapshots"
        current_predictions = {"2026 AZ Governor": 0.5300}

        with patch("check_prediction_changes.send_notification"):
            check_prediction_changes(
                current_predictions=current_predictions,
                snapshot_dir=snapshot_dir,
                threshold=0.02,
                notify=True,
            )

        assert snapshot_dir.exists()


# ---------------------------------------------------------------------------
# send_notification tests
# ---------------------------------------------------------------------------


class TestSendNotification:
    def test_calls_notify_sh_with_message(self):
        from check_prediction_changes import send_notification

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            send_notification("test message")
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert "notify.sh" in " ".join(cmd)
            assert "test message" in cmd

    def test_does_not_raise_on_script_failure(self):
        from check_prediction_changes import send_notification

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("notify.sh not found")
            # Should not raise — fail gracefully
            send_notification("test message")

    def test_notify_sh_path_is_absolute(self):
        from check_prediction_changes import NOTIFY_SCRIPT

        assert Path(NOTIFY_SCRIPT).is_absolute()


# ---------------------------------------------------------------------------
# Cron integration: temp-file format written by the cron script
# ---------------------------------------------------------------------------


class TestCronTempFileIntegration:
    """Verify that the JSON format the cron script writes is accepted by load_snapshot.

    The cron script serialises predictions as:
        {"predictions": {race: float, ...}}
    (no "date" key — written on-the-fly without enrichment).
    """

    def test_loads_predictions_only_wrapper(self, tmp_path):
        """load_snapshot handles {"predictions": {...}} without a date key."""
        predictions = {
            "2026 AZ Governor": 0.5115,
            "2026 GA Senate": 0.5839,
        }
        f = tmp_path / "cron_temp.json"
        f.write_text(json.dumps({"predictions": predictions}))
        result = load_snapshot(f)
        assert result == predictions

    def test_cron_flow_detects_alert_from_temp_file(self, tmp_path):
        """End-to-end: previous snapshot in dir + new temp file → alert sent for >2pp change."""
        from check_prediction_changes import check_prediction_changes

        snapshot_dir = tmp_path / "forecast_snapshots"
        snapshot_dir.mkdir()

        # Simulate the snapshot saved by last week's cron run
        prev_predictions = {
            "2026 AZ Governor": 0.5000,
            "2026 GA Senate": 0.5839,
        }
        prev_file = snapshot_dir / "2026-03-26.json"
        prev_file.write_text(
            json.dumps({"date": "2026-03-26", "predictions": prev_predictions})
        )

        # Simulate the temp file the cron script writes after DuckDB rebuild
        # ({"predictions": ...} format, no date key)
        current_predictions = {
            "2026 AZ Governor": 0.5350,  # +3.5pp — above threshold
            "2026 GA Senate": 0.5845,    # +0.06pp — below threshold
        }
        cron_temp = tmp_path / "wethervane-after-snap-XXXXXX.json"
        cron_temp.write_text(json.dumps({"predictions": current_predictions}))

        with patch("check_prediction_changes.send_notification") as mock_notify:
            changes = check_prediction_changes(
                current_predictions=load_snapshot(cron_temp),  # mimics --current-file
                snapshot_dir=snapshot_dir,
                threshold=0.02,
                notify=True,
            )

        assert len(changes) == 1
        assert changes[0]["race"] == "2026 AZ Governor"
        assert abs(changes[0]["delta"] - 0.0350) < 1e-6
        mock_notify.assert_called_once()
        alert_msg = mock_notify.call_args[0][0]
        assert "AZ Governor" in alert_msg

    def test_cron_flow_no_alert_when_predictions_stable(self, tmp_path):
        """When all races change by less than 2pp, no notification is sent."""
        from check_prediction_changes import check_prediction_changes

        snapshot_dir = tmp_path / "forecast_snapshots"
        snapshot_dir.mkdir()

        prev_predictions = {"2026 NC Senate": 0.6180}
        prev_file = snapshot_dir / "2026-03-26.json"
        prev_file.write_text(
            json.dumps({"date": "2026-03-26", "predictions": prev_predictions})
        )

        # +0.5pp — below the 2pp threshold
        current_predictions = {"2026 NC Senate": 0.6230}
        cron_temp = tmp_path / "temp.json"
        cron_temp.write_text(json.dumps({"predictions": current_predictions}))

        with patch("check_prediction_changes.send_notification") as mock_notify:
            changes = check_prediction_changes(
                current_predictions=load_snapshot(cron_temp),
                snapshot_dir=snapshot_dir,
                threshold=0.02,
                notify=True,
            )

        assert changes == []
        mock_notify.assert_not_called()

    def test_alert_message_shows_direction_correctly(self):
        """Alert message clearly shows D vs R direction via before/after values."""
        from check_prediction_changes import format_alert_message

        # Dem gain (positive delta → shifting toward D)
        changes_d = [
            {"race": "2026 AZ Governor", "before": 0.50, "after": 0.53, "delta": 0.03}
        ]
        msg_d = format_alert_message(changes_d)
        # Before < after → shifting toward D
        assert "50.0%" in msg_d
        assert "53.0%" in msg_d
        assert "+3.0pp" in msg_d

        # Dem loss (negative delta → shifting toward R)
        changes_r = [
            {"race": "2026 NC Senate", "before": 0.618, "after": 0.590, "delta": -0.028}
        ]
        msg_r = format_alert_message(changes_r)
        assert "61.8%" in msg_r
        assert "59.0%" in msg_r
        assert "-2.8pp" in msg_r
