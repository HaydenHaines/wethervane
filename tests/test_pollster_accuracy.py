"""Tests for scripts/pollster_accuracy.py and api/routers/pollsters.py.

Tests cover:
  - Race string parsing
  - Pollster name normalization
  - Backtest data loading
  - Poll CSV loading
  - Accuracy computation (RMSE, bias, ranking)
  - Graceful handling of mismatches
  - Output JSON format
  - API endpoint (unit)
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Make the scripts directory importable
_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import pollster_accuracy as pa

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_polls_csv(rows: list[dict], path: Path) -> Path:
    """Write poll dicts to a CSV file for testing."""
    fieldnames = ["race", "geography", "geo_level", "dem_share", "n_sample", "date", "pollster", "notes"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    return path


def _write_backtest_json(governor_metrics: list[dict], senate_metrics: list[dict], path: Path) -> Path:
    """Write a minimal backtest JSON file for testing."""
    data = {
        "description": "test backtest",
        "model_config": {},
        "excluded_2022_columns": [],
        "excluded_holdout_columns": [],
        "governor": {
            "raw_prior": {
                "county_r": 0.9,
                "county_rmse": 0.05,
                "n_counties": 100,
                "spotlight": [],
                "state_metrics": governor_metrics,
            },
            "type_mean_prior": {
                "county_r": 0.88,
                "county_rmse": 0.06,
                "n_counties": 100,
                "spotlight": [],
                "state_metrics": [],
            },
        },
        "senate": {
            "raw_prior": {
                "county_r": 0.91,
                "county_rmse": 0.04,
                "n_counties": 100,
                "spotlight": [],
                "state_metrics": senate_metrics,
            },
            "type_mean_prior": {
                "county_r": 0.89,
                "county_rmse": 0.05,
                "n_counties": 100,
                "spotlight": [],
                "state_metrics": [],
            },
        },
    }
    with path.open("w") as f:
        json.dump(data, f)
    return path


# ---------------------------------------------------------------------------
# parse_race_string
# ---------------------------------------------------------------------------


class TestParseRaceString:
    def test_governor_race(self):
        result = pa.parse_race_string("2022 GA Governor")
        assert result == ("2022", "GA", "governor")

    def test_senate_race(self):
        result = pa.parse_race_string("2022 FL Senate")
        assert result == ("2022", "FL", "senate")

    def test_different_state(self):
        result = pa.parse_race_string("2022 AL Senate")
        assert result == ("2022", "AL", "senate")

    def test_lowercase_race_type(self):
        """Case-insensitive matching for race type."""
        result = pa.parse_race_string("2022 GA governor")
        assert result is not None
        assert result[2] == "governor"

    def test_unknown_race_type_returns_none(self):
        """Unknown race type (e.g. President) returns None."""
        result = pa.parse_race_string("2022 FL President")
        assert result is None

    def test_malformed_string_returns_none(self):
        assert pa.parse_race_string("not a race") is None

    def test_empty_string_returns_none(self):
        assert pa.parse_race_string("") is None

    def test_strips_leading_whitespace(self):
        result = pa.parse_race_string("  2022 GA Senate")
        assert result == ("2022", "GA", "senate")


# ---------------------------------------------------------------------------
# normalize_pollster_name
# ---------------------------------------------------------------------------


class TestNormalizePollsterName:
    def test_known_alias_nyt(self):
        """NYT/Siena College canonical alias is applied."""
        result = pa.normalize_pollster_name("The New York Times/Siena College")
        assert result == "New York Times/Siena College"

    def test_known_alias_fabrizio(self):
        result = pa.normalize_pollster_name("Fabrizio Lee & Associates/Impact Research")
        assert result == "Fabrizio/Impact Research"

    def test_unknown_pollster_unchanged(self):
        """A pollster without an alias is returned as-is (whitespace normalized)."""
        result = pa.normalize_pollster_name("Emerson College")
        assert result == "Emerson College"

    def test_collapses_internal_whitespace(self):
        result = pa.normalize_pollster_name("Emerson  College")
        assert result == "Emerson College"

    def test_strips_outer_whitespace(self):
        result = pa.normalize_pollster_name("  Emerson College  ")
        assert result == "Emerson College"


# ---------------------------------------------------------------------------
# load_backtest_actuals
# ---------------------------------------------------------------------------


class TestLoadBacktestActuals:
    def test_loads_governor_actuals(self, tmp_path):
        gov = [{"state": "GA", "pred": 0.50, "actual": 0.46, "error_pp": 4.0, "n_counties": 159}]
        sen = []
        path = _write_backtest_json(gov, sen, tmp_path / "backtest.json")
        actuals = pa.load_backtest_actuals(path)
        assert ("governor", "GA") in actuals
        assert abs(actuals[("governor", "GA")] - 0.46) < 1e-9

    def test_loads_senate_actuals(self, tmp_path):
        gov = []
        sen = [{"state": "AZ", "pred": 0.50, "actual": 0.525, "error_pp": -2.5, "n_counties": 15}]
        path = _write_backtest_json(gov, sen, tmp_path / "backtest.json")
        actuals = pa.load_backtest_actuals(path)
        assert ("senate", "AZ") in actuals
        assert abs(actuals[("senate", "AZ")] - 0.525) < 1e-9

    def test_state_abbr_uppercased(self, tmp_path):
        """State abbreviations are uppercased regardless of source case."""
        gov = [{"state": "fl", "pred": 0.40, "actual": 0.40, "error_pp": 0.0, "n_counties": 67}]
        sen = []
        path = _write_backtest_json(gov, sen, tmp_path / "backtest.json")
        actuals = pa.load_backtest_actuals(path)
        assert ("governor", "FL") in actuals

    def test_empty_sections_return_empty(self, tmp_path):
        path = _write_backtest_json([], [], tmp_path / "backtest.json")
        actuals = pa.load_backtest_actuals(path)
        assert len(actuals) == 0


# ---------------------------------------------------------------------------
# load_polls
# ---------------------------------------------------------------------------


class TestLoadPolls:
    def test_basic_load(self, tmp_path):
        rows = [
            {"race": "2022 GA Senate", "geography": "GA", "geo_level": "state",
             "dem_share": "0.51", "n_sample": "1000", "date": "2022-10-15",
             "pollster": "Emerson College", "notes": ""},
        ]
        path = _write_polls_csv(rows, tmp_path / "polls.csv")
        polls = pa.load_polls(path)
        assert len(polls) == 1
        assert polls[0]["dem_share"] == pytest.approx(0.51)
        assert polls[0]["pollster"] == "Emerson College"

    def test_invalid_dem_share_skipped(self, tmp_path):
        rows = [
            {"race": "2022 GA Senate", "geography": "GA", "geo_level": "state",
             "dem_share": "bad", "n_sample": "1000", "date": "2022-10-15",
             "pollster": "Test", "notes": ""},
        ]
        path = _write_polls_csv(rows, tmp_path / "polls.csv")
        polls = pa.load_polls(path)
        assert len(polls) == 0

    def test_geography_uppercased(self, tmp_path):
        rows = [
            {"race": "2022 GA Senate", "geography": "ga", "geo_level": "state",
             "dem_share": "0.51", "n_sample": "500", "date": "2022-10-15",
             "pollster": "Test", "notes": ""},
        ]
        path = _write_polls_csv(rows, tmp_path / "polls.csv")
        polls = pa.load_polls(path)
        assert polls[0]["geography"] == "GA"


# ---------------------------------------------------------------------------
# compute_pollster_accuracy
# ---------------------------------------------------------------------------


class TestComputePollsterAccuracy:
    def _simple_actuals(self) -> dict:
        return {
            ("senate", "GA"): 0.50,
            ("governor", "FL"): 0.40,
        }

    def _make_polls(self, entries: list[tuple]) -> list[dict]:
        """entries = (race, dem_share, pollster)"""
        return [
            {"race": race, "geography": race.split()[1], "dem_share": ds,
             "n_sample": 500, "date": "2022-10-15", "pollster": pollster, "notes": ""}
            for race, ds, pollster in entries
        ]

    def test_single_poll_rmse_and_bias(self):
        """With one poll per pollster, RMSE equals |error| and bias equals error."""
        actuals = self._simple_actuals()
        # Poll says 0.52 for GA Senate (actual 0.50) -> error = +2pp
        polls = self._make_polls([("2022 GA Senate", 0.52, "TestPollster")])
        results = pa.compute_pollster_accuracy(polls, actuals)
        assert len(results) == 1
        r = results[0]
        assert r["pollster"] == "TestPollster"
        assert r["n_polls"] == 1
        assert abs(r["rmse_pp"] - 2.0) < 0.01
        assert abs(r["mean_error_pp"] - 2.0) < 0.01

    def test_bias_direction(self):
        """Over-predicting Dem share gives positive bias."""
        actuals = {("senate", "GA"): 0.50}
        polls = self._make_polls([("2022 GA Senate", 0.55, "OverEstimator")])
        results = pa.compute_pollster_accuracy(polls, actuals)
        assert results[0]["mean_error_pp"] > 0

    def test_negative_bias_direction(self):
        """Under-predicting Dem share gives negative bias."""
        actuals = {("senate", "GA"): 0.50}
        polls = self._make_polls([("2022 GA Senate", 0.45, "UnderEstimator")])
        results = pa.compute_pollster_accuracy(polls, actuals)
        assert results[0]["mean_error_pp"] < 0

    def test_ranked_by_rmse(self):
        """Results are sorted by RMSE ascending -- best first."""
        actuals = {("senate", "GA"): 0.50}
        polls = self._make_polls([
            ("2022 GA Senate", 0.51, "Good"),  # error 1pp
            ("2022 GA Senate", 0.55, "Bad"),   # error 5pp
        ])
        results = pa.compute_pollster_accuracy(polls, actuals)
        assert results[0]["pollster"] == "Good"
        assert results[1]["pollster"] == "Bad"
        assert results[0]["rank"] == 1
        assert results[1]["rank"] == 2

    def test_multi_poll_rmse_calculation(self):
        """RMSE is correctly computed across multiple polls for one pollster."""
        actuals = {("senate", "GA"): 0.50, ("governor", "FL"): 0.40}
        # Pollster polls GA Senate (error 2pp) and FL Governor (error 4pp)
        # RMSE = sqrt((4 + 16) / 2) = sqrt(10) ~= 3.162
        polls = self._make_polls([
            ("2022 GA Senate", 0.52, "TwoPollPollster"),  # error +2pp
            ("2022 FL Governor", 0.44, "TwoPollPollster"),  # error +4pp
        ])
        results = pa.compute_pollster_accuracy(polls, actuals)
        assert len(results) == 1
        expected_rmse = math.sqrt((4 + 16) / 2)
        assert abs(results[0]["rmse_pp"] - expected_rmse) < 0.01

    def test_unmatched_race_skipped_gracefully(self):
        """Polls for races not in backtest are silently skipped, no crash."""
        actuals = {("senate", "GA"): 0.50}
        # Include a poll for a state not in backtest (KS Governor)
        polls = [
            {"race": "2022 KS Governor", "geography": "KS", "dem_share": 0.45,
             "n_sample": 500, "date": "2022-10-15", "pollster": "SomePollster", "notes": ""},
            {"race": "2022 GA Senate", "geography": "GA", "dem_share": 0.51,
             "n_sample": 500, "date": "2022-10-15", "pollster": "SomePollster", "notes": ""},
        ]
        results = pa.compute_pollster_accuracy(polls, actuals)
        # Only the GA Senate poll is counted
        assert len(results) == 1
        assert results[0]["n_polls"] == 1

    def test_empty_polls_returns_empty(self):
        actuals = {("senate", "GA"): 0.50}
        results = pa.compute_pollster_accuracy([], actuals)
        assert results == []

    def test_empty_actuals_returns_empty(self):
        polls = self._make_polls([("2022 GA Senate", 0.51, "Pollster")])
        results = pa.compute_pollster_accuracy(polls, {})
        assert results == []

    def test_n_races_counts_distinct_races(self):
        """n_races counts distinct race strings, not distinct poll entries."""
        actuals = {("senate", "GA"): 0.50}
        # Two polls for same race
        polls = self._make_polls([
            ("2022 GA Senate", 0.51, "Pollster"),
            ("2022 GA Senate", 0.52, "Pollster"),
        ])
        results = pa.compute_pollster_accuracy(polls, actuals)
        assert results[0]["n_races"] == 1
        assert results[0]["n_polls"] == 2

    def test_pollster_name_normalized(self):
        """Normalized pollster names are deduplicated correctly."""
        actuals = {("senate", "GA"): 0.50}
        polls = [
            {"race": "2022 GA Senate", "geography": "GA", "dem_share": 0.51,
             "n_sample": 500, "date": "2022-10-15",
             "pollster": "The New York Times/Siena College", "notes": ""},
        ]
        results = pa.compute_pollster_accuracy(polls, actuals)
        assert results[0]["pollster"] == "New York Times/Siena College"


# ---------------------------------------------------------------------------
# save_results and output format
# ---------------------------------------------------------------------------


class TestSaveResults:
    def test_output_json_structure(self, tmp_path):
        results = [
            {"pollster": "Test", "n_polls": 1, "n_races": 1,
             "rmse_pp": 1.5, "mean_error_pp": 1.2, "rank": 1},
        ]
        output = tmp_path / "out.json"
        pa.save_results(results, output)
        assert output.exists()
        with output.open() as f:
            data = json.load(f)
        assert "description" in data
        assert "n_pollsters" in data
        assert "pollsters" in data
        assert data["n_pollsters"] == 1
        assert data["pollsters"][0]["pollster"] == "Test"

    def test_output_is_valid_json(self, tmp_path):
        """The file must parse cleanly as JSON without errors."""
        results = [
            {"pollster": "A", "n_polls": 3, "n_races": 2,
             "rmse_pp": 2.1, "mean_error_pp": -0.5, "rank": 1},
        ]
        output = tmp_path / "out.json"
        pa.save_results(results, output)
        with output.open() as f:
            data = json.load(f)
        assert isinstance(data["pollsters"], list)


# ---------------------------------------------------------------------------
# API endpoint (unit tests using httpx test client)
# ---------------------------------------------------------------------------


class TestPollsterAccuracyEndpoint:
    """Unit tests for GET /api/v1/pollsters/accuracy.

    We mock _load_accuracy_data to avoid filesystem dependencies.
    """

    @pytest.fixture(autouse=True)
    def _app(self):
        """Build a minimal FastAPI app with just the pollsters router for testing."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        import api.routers.pollsters as pollsters_mod

        app = FastAPI()
        app.include_router(pollsters_mod.router, prefix="/api/v1")
        self.client = TestClient(app)
        self.mod = pollsters_mod

    def _sample_raw(self, n: int = 2) -> dict:
        return {
            "description": "test description",
            "n_pollsters": n,
            "pollsters": [
                {
                    "pollster": f"Pollster{i}",
                    "rank": i,
                    "n_polls": 3,
                    "n_races": 2,
                    "rmse_pp": float(i),
                    "mean_error_pp": float(i) * 0.5,
                }
                for i in range(1, n + 1)
            ],
        }

    def test_returns_200_with_data(self):
        with patch.object(self.mod, "_load_accuracy_data", return_value=self._sample_raw()):
            resp = self.client.get("/api/v1/pollsters/accuracy")
        assert resp.status_code == 200

    def test_response_has_pollsters_list(self):
        with patch.object(self.mod, "_load_accuracy_data", return_value=self._sample_raw(2)):
            resp = self.client.get("/api/v1/pollsters/accuracy")
        data = resp.json()
        assert "pollsters" in data
        assert len(data["pollsters"]) == 2

    def test_response_fields_present(self):
        with patch.object(self.mod, "_load_accuracy_data", return_value=self._sample_raw(1)):
            resp = self.client.get("/api/v1/pollsters/accuracy")
        entry = resp.json()["pollsters"][0]
        assert "pollster" in entry
        assert "rank" in entry
        assert "n_polls" in entry
        assert "n_races" in entry
        assert "rmse_pp" in entry
        assert "mean_error_pp" in entry

    def test_response_description_present(self):
        with patch.object(self.mod, "_load_accuracy_data", return_value=self._sample_raw()):
            resp = self.client.get("/api/v1/pollsters/accuracy")
        data = resp.json()
        assert "description" in data
        assert len(data["description"]) > 0

    def test_503_when_file_missing(self):
        """Returns 503 if the JSON has not been generated yet."""
        from fastapi import HTTPException

        def _missing():
            raise HTTPException(status_code=503, detail="not generated")

        with patch.object(self.mod, "_load_accuracy_data", side_effect=_missing):
            resp = self.client.get("/api/v1/pollsters/accuracy")
        assert resp.status_code == 503

    def test_n_pollsters_matches_list_length(self):
        with patch.object(self.mod, "_load_accuracy_data", return_value=self._sample_raw(3)):
            resp = self.client.get("/api/v1/pollsters/accuracy")
        data = resp.json()
        assert data["n_pollsters"] == len(data["pollsters"])
