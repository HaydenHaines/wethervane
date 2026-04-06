"""Tests for scripts/fetch_fred_fundamentals.py."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

# The script is not a package, so we import its functions by path manipulation.
import importlib.util

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "fetch_fred_fundamentals.py"
_spec = importlib.util.spec_from_file_location("fetch_fred_fundamentals", _SCRIPT_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_load_api_key = _mod._load_api_key
_fetch_series = _mod._fetch_series
_latest_value = _mod._latest_value
_compute_yoy = _mod._compute_yoy
fetch_all = _mod.fetch_all
update_snapshot = _mod.update_snapshot
SERIES = _mod.SERIES


# ---------------------------------------------------------------------------
# _latest_value
# ---------------------------------------------------------------------------

class TestLatestValue:
    def test_returns_most_recent(self):
        obs = [{"date": "2026-03-01", "value": "4.3"}, {"date": "2026-02-01", "value": "4.1"}]
        val, date = _latest_value(obs)
        assert val == 4.3
        assert date == "2026-03-01"

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="No observations"):
            _latest_value([])

    def test_float_conversion(self):
        obs = [{"date": "2025-10-01", "value": "0.70"}]
        val, date = _latest_value(obs)
        assert val == 0.7
        assert isinstance(val, float)


# ---------------------------------------------------------------------------
# _compute_yoy
# ---------------------------------------------------------------------------

class TestComputeYoY:
    def _make_cpi_obs(self, current: float, prior: float) -> list[dict]:
        """Create 14 months of CPI observations with known current and prior values."""
        obs = []
        # Most recent (Feb 2026)
        obs.append({"date": "2026-02-01", "value": str(current)})
        # Fill months between
        for i in range(1, 12):
            month = 2 - i
            year = 2026
            if month <= 0:
                month += 12
                year -= 1
            obs.append({"date": f"{year}-{month:02d}-01", "value": str(current - i * 0.1)})
        # Prior year same month (Feb 2025)
        obs.append({"date": "2025-02-01", "value": str(prior)})
        obs.append({"date": "2025-01-01", "value": str(prior - 0.1)})
        return obs

    def test_basic_yoy(self):
        obs = self._make_cpi_obs(310.0, 300.0)
        yoy, current_date, prior_date = _compute_yoy(obs)
        expected = ((310.0 - 300.0) / 300.0) * 100
        assert abs(yoy - round(expected, 2)) < 0.01
        assert current_date == "2026-02-01"
        assert prior_date == "2025-02-01"

    def test_too_few_observations(self):
        obs = [{"date": f"2026-{i:02d}-01", "value": "300"} for i in range(1, 13)]
        with pytest.raises(ValueError, match="at least 13"):
            _compute_yoy(obs)

    def test_zero_inflation(self):
        obs = self._make_cpi_obs(300.0, 300.0)
        yoy, _, _ = _compute_yoy(obs)
        assert yoy == 0.0

    def test_negative_inflation(self):
        obs = self._make_cpi_obs(295.0, 300.0)
        yoy, _, _ = _compute_yoy(obs)
        assert yoy < 0


# ---------------------------------------------------------------------------
# update_snapshot
# ---------------------------------------------------------------------------

class TestUpdateSnapshot:
    def test_updates_values(self, tmp_path: Path):
        # Create initial snapshot
        snapshot_path = tmp_path / "snapshot.json"
        snapshot_path.write_text(json.dumps({
            "cycle": 2026, "in_party": "D", "approval_net_oct": -12.0,
            "gdp_q2_growth_pct": 1.8, "unemployment_oct": 4.1,
            "cpi_yoy_oct": 3.2, "consumer_sentiment": 68.5,
            "source_notes": {"last_updated": "2026-03-27"},
        }))

        results = {
            "gdp_q2_growth_pct": {"value": 0.7, "date": "2025-10-01",
                                   "description": "test", "series_id": "X"},
            "unemployment_oct": {"value": 4.3, "date": "2026-03-01",
                                  "description": "test", "series_id": "Y"},
        }

        with patch.object(_mod, "SNAPSHOT_PATH", snapshot_path):
            updated = update_snapshot(results, dry_run=False)

        assert updated["gdp_q2_growth_pct"] == 0.7
        assert updated["unemployment_oct"] == 4.3
        # Approval should be preserved
        assert updated["approval_net_oct"] == -12.0
        # CPI should be preserved (not in results)
        assert updated["cpi_yoy_oct"] == 3.2

        # Verify file was written
        written = json.loads(snapshot_path.read_text())
        assert written["gdp_q2_growth_pct"] == 0.7

    def test_dry_run_does_not_write(self, tmp_path: Path):
        snapshot_path = tmp_path / "snapshot.json"
        original = {"cycle": 2026, "in_party": "D", "gdp_q2_growth_pct": 1.8,
                     "source_notes": {}}
        snapshot_path.write_text(json.dumps(original))

        results = {"gdp_q2_growth_pct": {"value": 0.5, "date": "2025-10-01",
                                          "description": "test", "series_id": "X"}}

        with patch.object(_mod, "SNAPSHOT_PATH", snapshot_path):
            update_snapshot(results, dry_run=True)

        # File should still have original value
        written = json.loads(snapshot_path.read_text())
        assert written["gdp_q2_growth_pct"] == 1.8

    def test_creates_snapshot_if_missing(self, tmp_path: Path):
        snapshot_path = tmp_path / "new_snapshot.json"
        assert not snapshot_path.exists()

        results = {"unemployment_oct": {"value": 4.0, "date": "2026-01-01",
                                         "description": "test", "series_id": "Z"}}

        with patch.object(_mod, "SNAPSHOT_PATH", snapshot_path):
            updated = update_snapshot(results, dry_run=False)

        assert snapshot_path.exists()
        assert updated["unemployment_oct"] == 4.0
        assert updated["cycle"] == 2026


# ---------------------------------------------------------------------------
# SERIES config validation
# ---------------------------------------------------------------------------

class TestSeriesConfig:
    def test_all_series_have_required_keys(self):
        for name, spec in SERIES.items():
            assert "series_id" in spec, f"{name} missing series_id"
            assert "snapshot_key" in spec, f"{name} missing snapshot_key"
            assert "description" in spec, f"{name} missing description"

    def test_snapshot_keys_are_valid(self):
        valid_keys = {
            "gdp_q2_growth_pct", "unemployment_oct", "cpi_yoy_oct",
            "consumer_sentiment", "approval_net_oct",
        }
        for name, spec in SERIES.items():
            assert spec["snapshot_key"] in valid_keys, (
                f"{name} has unexpected snapshot_key: {spec['snapshot_key']}"
            )

    def test_cpi_has_yoy_compute(self):
        assert SERIES["cpi"]["compute"] == "yoy"


# ---------------------------------------------------------------------------
# _load_api_key
# ---------------------------------------------------------------------------

class TestLoadApiKey:
    def test_from_env_var(self):
        with patch.dict("os.environ", {"FRED_API_KEY": "test_key_123"}):
            assert _load_api_key() == "test_key_123"

    def test_from_env_file(self, tmp_path: Path):
        env_file = tmp_path / ".env"
        env_file.write_text("OTHER_VAR=foo\nFRED_API_KEY=file_key_456\n")

        with patch.dict("os.environ", {}, clear=True), \
             patch.object(_mod, "ENV_PATH", env_file):
            assert _load_api_key() == "file_key_456"

    def test_missing_raises(self, tmp_path: Path):
        env_file = tmp_path / ".env"
        env_file.write_text("OTHER_VAR=foo\n")

        with patch.dict("os.environ", {}, clear=True), \
             patch.object(_mod, "ENV_PATH", env_file):
            with pytest.raises(RuntimeError, match="FRED_API_KEY not found"):
                _load_api_key()
