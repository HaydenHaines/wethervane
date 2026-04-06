"""Tests for scripts/fetch_approval_rating.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Allow importing the script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from fetch_approval_rating import (
    APPROVE_NAME,
    DISAPPROVE_NAME,
    RCP_AVERAGE_TYPE,
    _extract_polls_array,
    _find_rcp_average,
    _parse_percentage,
    extract_approval_from_polls,
    fetch_approval_rating,
    load_snapshot,
    update_snapshot,
)


# ---------------------------------------------------------------------------
# Helpers for building minimal Next.js-style HTML fixtures
# ---------------------------------------------------------------------------


def _make_next_f_html(polls: list[dict]) -> str:
    """Wrap a polls list in the __next_f.push HTML scaffolding that RCP uses.

    RCP embeds data as::

        self.__next_f.push([1,"<escaped-json-with-polls-array>"]);

    We encode the polls into that structure so our tests exercise the real
    extraction code path, not a simplified bypass.
    """
    inner = json.dumps({"tableTitle": "Poll Data", "polls": polls})
    # json.dumps produces a plain string; we need to JSON-encode it *again* so
    # it becomes a valid escaped string literal inside the push() call.
    escaped = json.dumps(inner)[1:-1]  # strip surrounding quotes
    payload = f'self.__next_f.push([1,"{escaped}"]);'
    return f"<html><body><script>{payload}</script></body></html>"


def _rcp_average_entry(
    approve: str = "40.9",
    disapprove: str = "56.9",
    date: str = "3/12 - 4/2",
) -> dict:
    """Return a minimal RCP average poll entry."""
    return {
        "id": "8656",
        "type": RCP_AVERAGE_TYPE,
        "pollster": RCP_AVERAGE_TYPE,
        "date": date,
        "candidate": [
            {"name": APPROVE_NAME, "value": approve},
            {"name": DISAPPROVE_NAME, "value": disapprove},
        ],
        "spread": {"name": DISAPPROVE_NAME, "value": "+16.0"},
    }


def _individual_poll_entry(
    approve: str = "41.0",
    disapprove: str = "55.0",
    pollster: str = "Economist/YouGov",
    date: str = "3/27 - 3/30",
) -> dict:
    """Return a minimal individual poll entry."""
    return {
        "id": "999",
        "type": "poll_rcp_avg",
        "pollster": pollster,
        "date": date,
        "sampleSize": "1000 RV",
        "candidate": [
            {"name": APPROVE_NAME, "value": approve},
            {"name": DISAPPROVE_NAME, "value": disapprove},
        ],
    }


# ---------------------------------------------------------------------------
# _parse_percentage
# ---------------------------------------------------------------------------


class TestParsePercentage:
    def test_plain_number_string(self):
        assert _parse_percentage("40.9") == pytest.approx(40.9)

    def test_integer_string(self):
        assert _parse_percentage("42") == pytest.approx(42.0)

    def test_none_returns_none(self):
        assert _parse_percentage(None) is None

    def test_non_numeric_returns_none(self):
        assert _parse_percentage("N/A") is None
        assert _parse_percentage("") is None

    def test_whitespace_stripped(self):
        assert _parse_percentage("  56.9  ") == pytest.approx(56.9)


# ---------------------------------------------------------------------------
# _extract_polls_array
# ---------------------------------------------------------------------------


class TestExtractPollsArray:
    def test_extracts_polls_from_valid_html(self):
        polls = [_rcp_average_entry(), _individual_poll_entry()]
        html = _make_next_f_html(polls)
        result = _extract_polls_array(html)
        assert result is not None
        assert len(result) == 2

    def test_returns_none_when_no_next_f_push(self):
        html = "<html><body><p>no scripts</p></body></html>"
        assert _extract_polls_array(html) is None

    def test_returns_none_when_rcp_average_absent(self):
        # Script exists but contains only individual polls without rcp_average
        polls = [_individual_poll_entry()]
        inner = json.dumps({"tableTitle": "Poll Data", "polls": polls})
        escaped = json.dumps(inner)[1:-1]
        payload = f'self.__next_f.push([1,"{escaped}"]);'
        html = f"<html><body><script>{payload}</script></body></html>"
        assert _extract_polls_array(html) is None

    def test_returns_none_on_malformed_json(self):
        # Inject deliberate JSON corruption — the push call won't decode cleanly
        html = "<html><body><script>self.__next_f.push([1,\"rcp_average{BROKEN\"]);</script></body></html>"
        # Should not raise; should return None
        assert _extract_polls_array(html) is None

    def test_ignores_unrelated_scripts(self):
        """Scripts without rcp_average in them should be skipped."""
        unrelated = "<script>window.__data = {foo: 1};</script>"
        polls = [_rcp_average_entry()]
        html = unrelated + _make_next_f_html(polls)
        result = _extract_polls_array(html)
        assert result is not None
        assert len(result) == 1


# ---------------------------------------------------------------------------
# _find_rcp_average
# ---------------------------------------------------------------------------


class TestFindRcpAverage:
    def test_finds_average_entry(self):
        avg = _rcp_average_entry()
        polls = [avg, _individual_poll_entry()]
        assert _find_rcp_average(polls) is avg

    def test_returns_none_when_absent(self):
        polls = [_individual_poll_entry()]
        assert _find_rcp_average(polls) is None

    def test_returns_none_for_empty_list(self):
        assert _find_rcp_average([]) is None


# ---------------------------------------------------------------------------
# extract_approval_from_polls
# ---------------------------------------------------------------------------


class TestExtractApprovalFromPolls:
    def test_correct_values(self):
        polls = [_rcp_average_entry(approve="40.9", disapprove="56.9", date="3/12 - 4/2")]
        result = extract_approval_from_polls(polls)
        assert result is not None
        assert result["approve_pct"] == pytest.approx(40.9)
        assert result["disapprove_pct"] == pytest.approx(56.9)
        assert result["net_approval"] == pytest.approx(-16.0)
        assert result["date_range"] == "3/12 - 4/2"

    def test_net_approval_calculation(self):
        """Net approval = approve - disapprove; positive when approve > disapprove."""
        polls = [_rcp_average_entry(approve="55.0", disapprove="40.0")]
        result = extract_approval_from_polls(polls)
        assert result["net_approval"] == pytest.approx(15.0)

    def test_returns_none_when_no_average_entry(self):
        polls = [_individual_poll_entry()]
        assert extract_approval_from_polls(polls) is None

    def test_returns_none_when_candidate_values_missing(self):
        broken_entry = {
            "id": "1",
            "type": RCP_AVERAGE_TYPE,
            "pollster": RCP_AVERAGE_TYPE,
            "date": "3/1 - 3/5",
            "candidate": [],  # no candidates
        }
        assert extract_approval_from_polls([broken_entry]) is None

    def test_case_insensitive_candidate_name_matching(self):
        """Candidate names should be matched case-insensitively."""
        entry = {
            "id": "1",
            "type": RCP_AVERAGE_TYPE,
            "pollster": RCP_AVERAGE_TYPE,
            "date": "3/1 - 3/5",
            "candidate": [
                {"name": "approve", "value": "41.0"},  # lowercase
                {"name": "disapprove", "value": "55.0"},  # lowercase
            ],
        }
        result = extract_approval_from_polls([entry])
        assert result is not None
        assert result["net_approval"] == pytest.approx(-14.0)

    def test_ignores_individual_polls_for_calculation(self):
        """Only the rcp_average entry should be used; individual polls are irrelevant."""
        polls = [
            _rcp_average_entry(approve="40.9", disapprove="56.9"),
            _individual_poll_entry(approve="50.0", disapprove="45.0"),  # different numbers
        ]
        result = extract_approval_from_polls(polls)
        # Should come from the average, not the individual poll
        assert result["approve_pct"] == pytest.approx(40.9)
        assert result["disapprove_pct"] == pytest.approx(56.9)


# ---------------------------------------------------------------------------
# update_snapshot
# ---------------------------------------------------------------------------


class TestUpdateSnapshot:
    def test_writes_correct_value(self, tmp_path):
        """update_snapshot should write the net_approval to the snapshot file."""
        snapshot_file = tmp_path / "snapshot_2026.json"
        existing = {
            "cycle": 2026,
            "in_party": "D",
            "approval_net_oct": -12.0,
            "source_notes": {"approval": "Placeholder"},
        }
        snapshot_file.write_text(json.dumps(existing))

        approval_data = {
            "approve_pct": 40.9,
            "disapprove_pct": 56.9,
            "net_approval": -16.0,
            "date_range": "3/12 - 4/2",
        }

        with patch("fetch_approval_rating.SNAPSHOT_PATH", snapshot_file):
            result = update_snapshot(approval_data, dry_run=False)

        assert result["approval_net_oct"] == pytest.approx(-16.0)
        saved = json.loads(snapshot_file.read_text())
        assert saved["approval_net_oct"] == pytest.approx(-16.0)

    def test_dry_run_does_not_write(self, tmp_path):
        """Dry-run mode must not modify the snapshot file."""
        snapshot_file = tmp_path / "snapshot_2026.json"
        original = {"cycle": 2026, "in_party": "D", "approval_net_oct": -12.0, "source_notes": {}}
        snapshot_file.write_text(json.dumps(original))

        approval_data = {
            "approve_pct": 40.9,
            "disapprove_pct": 56.9,
            "net_approval": -16.0,
            "date_range": "3/12 - 4/2",
        }

        with patch("fetch_approval_rating.SNAPSHOT_PATH", snapshot_file):
            update_snapshot(approval_data, dry_run=True)

        # File must be unchanged
        saved = json.loads(snapshot_file.read_text())
        assert saved["approval_net_oct"] == pytest.approx(-12.0)

    def test_preserves_other_fields(self, tmp_path):
        """Fields unrelated to approval should survive the update."""
        snapshot_file = tmp_path / "snapshot_2026.json"
        existing = {
            "cycle": 2026,
            "in_party": "D",
            "approval_net_oct": -12.0,
            "gdp_q2_growth_pct": 2.3,
            "unemployment_oct": 4.1,
            "source_notes": {},
        }
        snapshot_file.write_text(json.dumps(existing))

        approval_data = {
            "approve_pct": 40.9,
            "disapprove_pct": 56.9,
            "net_approval": -16.0,
            "date_range": "3/12 - 4/2",
        }

        with patch("fetch_approval_rating.SNAPSHOT_PATH", snapshot_file):
            update_snapshot(approval_data, dry_run=False)

        saved = json.loads(snapshot_file.read_text())
        assert saved["gdp_q2_growth_pct"] == pytest.approx(2.3)
        assert saved["unemployment_oct"] == pytest.approx(4.1)
        assert saved["cycle"] == 2026
        assert saved["in_party"] == "D"

    def test_source_note_contains_provenance(self, tmp_path):
        """The source_notes.approval string should mention RCP, percentages, and date."""
        snapshot_file = tmp_path / "snapshot_2026.json"
        snapshot_file.write_text(json.dumps({"cycle": 2026, "approval_net_oct": -12.0, "source_notes": {}}))

        approval_data = {
            "approve_pct": 40.9,
            "disapprove_pct": 56.9,
            "net_approval": -16.0,
            "date_range": "3/12 - 4/2",
        }

        with patch("fetch_approval_rating.SNAPSHOT_PATH", snapshot_file):
            result = update_snapshot(approval_data, dry_run=False)

        note = result["source_notes"]["approval"]
        assert "RealClearPolling" in note
        assert "40.9" in note
        assert "56.9" in note
        assert "3/12 - 4/2" in note


# ---------------------------------------------------------------------------
# fetch_approval_rating (integration via mock HTML)
# ---------------------------------------------------------------------------


class TestFetchApprovalRating:
    def test_returns_parsed_data_on_valid_html(self):
        """fetch_approval_rating should parse a minimal valid HTML page."""
        polls = [_rcp_average_entry(approve="40.9", disapprove="56.9", date="3/12 - 4/2")]
        html = _make_next_f_html(polls)

        with patch("fetch_approval_rating.fetch_html", return_value=html):
            result = fetch_approval_rating("https://fake.url/")

        assert result is not None
        assert result["net_approval"] == pytest.approx(-16.0)
        assert result["approve_pct"] == pytest.approx(40.9)

    def test_returns_none_when_fetch_fails(self):
        """If the network request fails, fetch_approval_rating returns None."""
        with patch("fetch_approval_rating.fetch_html", return_value=None):
            result = fetch_approval_rating("https://fake.url/")
        assert result is None

    def test_returns_none_when_html_unparseable(self):
        """If the HTML has no parseable polls, returns None."""
        with patch("fetch_approval_rating.fetch_html", return_value="<html>empty</html>"):
            result = fetch_approval_rating("https://fake.url/")
        assert result is None
