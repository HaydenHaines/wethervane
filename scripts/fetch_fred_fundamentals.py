"""Fetch current economic indicators from the FRED API and update snapshot_2026.json.

Updates: GDP Q2 growth, unemployment rate, CPI year-over-year, and consumer sentiment.
Presidential approval is NOT available from FRED — must be updated manually.

Usage:
    uv run python scripts/fetch_fred_fundamentals.py [--dry-run]

Requires FRED_API_KEY in .env or environment.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = PROJECT_ROOT / "data" / "fundamentals" / "snapshot_2026.json"
ENV_PATH = PROJECT_ROOT / ".env"

# FRED API base URL
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# Series we fetch and how we interpret them
SERIES = {
    # Real GDP — quarterly, annualized growth rate.  We want the most recent
    # quarter's growth.  FRED series A191RL1Q225SBEA gives the percent change
    # directly (annualized rate).
    "gdp": {
        "series_id": "A191RL1Q225SBEA",
        "description": "Real GDP growth (annualized quarterly, %)",
        "snapshot_key": "gdp_q2_growth_pct",
    },
    # Unemployment rate — monthly, seasonally adjusted
    "unemployment": {
        "series_id": "UNRATE",
        "description": "Unemployment rate (%)",
        "snapshot_key": "unemployment_oct",
    },
    # CPI — monthly, seasonally adjusted.  We compute YoY from the last two
    # October observations (or the most recent 12-month window).
    "cpi": {
        "series_id": "CPIAUCSL",
        "description": "CPI-U (seasonally adjusted)",
        "snapshot_key": "cpi_yoy_oct",
        "compute": "yoy",  # Need to compute YoY from levels
    },
    # Consumer Sentiment — monthly
    "sentiment": {
        "series_id": "UMCSENT",
        "description": "University of Michigan Consumer Sentiment Index",
        "snapshot_key": "consumer_sentiment",
    },
}


def _load_api_key() -> str:
    """Load FRED_API_KEY from .env file or environment variable."""
    key = os.environ.get("FRED_API_KEY")
    if key:
        return key

    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if line.startswith("FRED_API_KEY="):
                return line.split("=", 1)[1].strip().strip("'\"")

    raise RuntimeError(
        "FRED_API_KEY not found. Set it in .env or as an environment variable. "
        "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
    )


def _fetch_series(
    series_id: str,
    api_key: str,
    limit: int = 24,
    sort_order: str = "desc",
) -> list[dict]:
    """Fetch recent observations for a FRED series.

    Returns a list of {"date": "YYYY-MM-DD", "value": "123.456"} dicts,
    most recent first.
    """
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "limit": limit,
        "sort_order": sort_order,
    }
    url = f"{FRED_BASE}?{urlencode(params)}"
    log.debug("Fetching %s from FRED", series_id)

    try:
        with urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except URLError as e:
        raise RuntimeError(f"FRED API request failed for {series_id}: {e}") from e

    observations = data.get("observations", [])
    # Filter out missing values (FRED uses "." for missing)
    return [
        obs for obs in observations
        if obs.get("value", ".") != "."
    ]


def _latest_value(observations: list[dict]) -> tuple[float, str]:
    """Extract the most recent value and its date from FRED observations."""
    if not observations:
        raise ValueError("No observations returned from FRED")
    latest = observations[0]
    return float(latest["value"]), latest["date"]


def _compute_yoy(observations: list[dict]) -> tuple[float, str, str]:
    """Compute year-over-year percent change from CPI levels.

    Returns (yoy_pct, current_date, prior_date).
    """
    if len(observations) < 13:
        raise ValueError(
            f"Need at least 13 CPI observations for YoY, got {len(observations)}"
        )
    # observations are sorted desc by date
    current = observations[0]
    current_val = float(current["value"])
    current_date = current["date"]

    # Find observation ~12 months prior
    target_month = datetime.strptime(current_date, "%Y-%m-%d")
    prior_year = target_month.year - 1
    prior_month_str = f"{prior_year}-{target_month.month:02d}"

    for obs in observations:
        if obs["date"].startswith(prior_month_str):
            prior_val = float(obs["value"])
            yoy = ((current_val - prior_val) / prior_val) * 100
            return round(yoy, 2), current_date, obs["date"]

    # Fallback: use observation closest to 12 months back
    prior_val = float(observations[12]["value"])
    yoy = ((current_val - prior_val) / prior_val) * 100
    return round(yoy, 2), current_date, observations[12]["date"]


def fetch_all(api_key: str) -> dict:
    """Fetch all fundamentals indicators from FRED.

    Returns a dict with snapshot_key -> {"value": float, "date": str, "description": str}.
    """
    results = {}

    for name, spec in SERIES.items():
        try:
            obs = _fetch_series(spec["series_id"], api_key)

            if spec.get("compute") == "yoy":
                yoy, current_date, prior_date = _compute_yoy(obs)
                results[spec["snapshot_key"]] = {
                    "value": yoy,
                    "date": current_date,
                    "description": f"{spec['description']} — YoY: {current_date} vs {prior_date}",
                    "series_id": spec["series_id"],
                }
            else:
                val, date_str = _latest_value(obs)
                results[spec["snapshot_key"]] = {
                    "value": round(val, 2),
                    "date": date_str,
                    "description": f"{spec['description']} — as of {date_str}",
                    "series_id": spec["series_id"],
                }

            log.info(
                "  %s: %.2f (%s)",
                spec["snapshot_key"],
                results[spec["snapshot_key"]]["value"],
                results[spec["snapshot_key"]]["date"],
            )
        except Exception as e:
            log.warning("Failed to fetch %s (%s): %s", name, spec["series_id"], e)

    return results


def update_snapshot(results: dict, dry_run: bool = False) -> dict:
    """Update snapshot_2026.json with fetched FRED values.

    Only updates fields that were successfully fetched. Preserves
    approval_net_oct (not from FRED) and in_party.

    Returns the updated snapshot dict.
    """
    if SNAPSHOT_PATH.exists():
        snapshot = json.loads(SNAPSHOT_PATH.read_text())
    else:
        snapshot = {
            "cycle": 2026,
            "in_party": "D",
            "approval_net_oct": -12.0,
            "source_notes": {},
        }

    source_notes = snapshot.get("source_notes", {})

    for key, info in results.items():
        old_val = snapshot.get(key)
        snapshot[key] = info["value"]
        # Map snapshot keys to the source_notes keys used in the JSON
        _NOTES_KEY_MAP = {
            "gdp_q2_growth_pct": "gdp",
            "unemployment_oct": "unemployment",
            "cpi_yoy_oct": "cpi",
            "consumer_sentiment": "consumer_sentiment",
        }
        notes_key = _NOTES_KEY_MAP.get(key, key)
        source_notes[notes_key] = (
            f"FRED {info['series_id']} — {info['description']}. "
            f"Fetched {datetime.now().strftime('%Y-%m-%d %H:%M')}."
        )

        change_str = ""
        if old_val is not None and old_val != info["value"]:
            change_str = f" (was {old_val})"
        print(f"  {key}: {info['value']}{change_str}")

    source_notes["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    snapshot["source_notes"] = source_notes

    if not dry_run:
        SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2) + "\n")
        print(f"\nWrote {SNAPSHOT_PATH}")
    else:
        print(f"\n[DRY RUN] Would write to {SNAPSHOT_PATH}")
        print(json.dumps(snapshot, indent=2))

    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch FRED economic indicators and update fundamentals snapshot."
    )
    parser.add_argument("--dry-run", action="store_true", help="Print but don't write")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(message)s"
    )

    api_key = _load_api_key()
    print("Fetching economic indicators from FRED...")

    results = fetch_all(api_key)
    if not results:
        print("No data fetched. Check API key and network.", file=sys.stderr)
        sys.exit(1)

    print(f"\nUpdating {SNAPSHOT_PATH}:")
    update_snapshot(results, dry_run=args.dry_run)

    # Remind about approval rating
    print(
        "\nNOTE: Presidential approval is NOT from FRED. "
        "Update approval_net_oct manually from RCP/538 average."
    )


if __name__ == "__main__":
    main()
