"""
Poll ingestion: load polls from data/polls/ CSV files into PollObservation objects.

CSV format (data/polls/polls_2026.csv):
  race,geography,geo_level,dem_share,n_sample,date,pollster,notes

Usage:
  from src.assembly.ingest_polls import load_polls
  polls = load_polls("2026")                         # all 2026 polls
  polls = load_polls("2026", race="2026 FL Senate")  # filter by race
  polls = load_polls("2026", geography="FL")          # filter by state
  polls = load_polls("2026", after="2026-02-01")      # filter by date
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Optional

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.propagation.propagate_polls import PollObservation  # noqa: E402

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _csv_path(cycle: str) -> Path:
    return PROJECT_ROOT / "data" / "polls" / f"polls_{cycle}.csv"


def _read_csv(cycle: str) -> list[dict]:
    path = _csv_path(cycle)
    if not path.exists():
        raise FileNotFoundError(
            f"Poll CSV not found: {path}\n"
            f"Expected file: data/polls/polls_{cycle}.csv relative to project root.\n"
            f"Create or download it, then re-run."
        )
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _row_to_observation(row: dict, row_num: int) -> Optional[PollObservation]:
    """Convert a CSV row dict to a PollObservation. Returns None if row is invalid."""
    race = row.get("race", "").strip()
    geography = row.get("geography", "").strip()
    geo_level = row.get("geo_level", "state").strip() or "state"
    date = row.get("date", "").strip()
    pollster = row.get("pollster", "").strip()

    # Required numeric fields
    raw_dem = row.get("dem_share", "").strip()
    raw_n = row.get("n_sample", "").strip()

    if not raw_dem:
        log.warning("Row %d skipped: missing dem_share (race=%r)", row_num, race)
        return None
    if not raw_n:
        log.warning("Row %d skipped: missing n_sample (race=%r)", row_num, race)
        return None

    try:
        dem_share = float(raw_dem)
    except ValueError:
        log.warning("Row %d skipped: dem_share=%r is not a float (race=%r)", row_num, raw_dem, race)
        return None

    try:
        n_sample = int(float(raw_n))
    except ValueError:
        log.warning("Row %d skipped: n_sample=%r is not an integer (race=%r)", row_num, raw_n, race)
        return None

    if not (0.0 < dem_share < 1.0):
        log.warning(
            "Row %d skipped: dem_share=%.4f out of range (0, 1) (race=%r)",
            row_num, dem_share, race,
        )
        return None

    if n_sample <= 0:
        log.warning("Row %d skipped: n_sample=%d must be positive (race=%r)", row_num, n_sample, race)
        return None

    return PollObservation(
        geography=geography,
        dem_share=dem_share,
        n_sample=n_sample,
        race=race,
        date=date,
        pollster=pollster,
        geo_level=geo_level,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_polls(
    cycle: str,
    race: str | None = None,
    geography: str | None = None,
    after: str | None = None,
    before: str | None = None,
    geo_level: str | None = None,
) -> list[PollObservation]:
    """Load and filter poll observations from data/polls/polls_{cycle}.csv.

    Parameters
    ----------
    cycle:
        Election cycle year string, e.g. "2026". Determines which CSV to read.
    race:
        Optional substring filter on race name (case-insensitive).
    geography:
        Optional exact match filter on geography (e.g. "FL", "GA").
    after:
        Optional ISO date string (YYYY-MM-DD). Only polls on or after this date.
    before:
        Optional ISO date string (YYYY-MM-DD). Only polls strictly before this date.
    geo_level:
        Optional filter on geo_level (e.g. "state", "county").

    Returns
    -------
    List of PollObservation objects sorted by date ascending.
    """
    rows = _read_csv(cycle)
    log.info("Loaded %d rows from polls_%s.csv", len(rows), cycle)

    observations: list[PollObservation] = []
    skipped = 0

    for i, row in enumerate(rows, start=2):  # row 1 is header
        obs = _row_to_observation(row, i)
        if obs is None:
            skipped += 1
            continue

        # Apply filters
        if race is not None and race.lower() not in obs.race.lower():
            continue
        if geography is not None and obs.geography != geography:
            continue
        if geo_level is not None and obs.geo_level != geo_level:
            continue
        if after is not None and obs.date < after:
            continue
        if before is not None and obs.date >= before:
            continue

        observations.append(obs)

    observations.sort(key=lambda o: o.date)

    log.info(
        "Returning %d poll observations for cycle %s (skipped %d invalid rows)",
        len(observations), cycle, skipped,
    )
    return observations


def list_races(cycle: str) -> list[str]:
    """Return sorted list of unique race names from polls_{cycle}.csv."""
    rows = _read_csv(cycle)
    races = sorted({row.get("race", "").strip() for row in rows if row.get("race", "").strip()})
    return races


def polls_summary(cycle: str) -> None:
    """Print a summary table: race → poll count, date range, dem_share mean ± std."""
    polls = load_polls(cycle)

    if not polls:
        print(f"No polls found for cycle {cycle}.")
        return

    # Group by race
    by_race: dict[str, list[PollObservation]] = {}
    for p in polls:
        by_race.setdefault(p.race, []).append(p)

    col_race = max(len(r) for r in by_race) + 2
    header = (
        f"{'Race':<{col_race}}  "
        f"{'N':>5}  "
        f"{'Date Range':<23}  "
        f"{'Dem Share (mean ± std)'}"
    )
    print(f"\n=== Poll Summary: {cycle} cycle ===")
    print(header)
    print("-" * len(header))

    for race in sorted(by_race):
        group = by_race[race]
        dates = [p.date for p in group if p.date]
        shares = [p.dem_share for p in group]

        date_range = f"{min(dates)} – {max(dates)}" if dates else "unknown"
        mean_share = mean(shares)
        std_share = stdev(shares) if len(shares) > 1 else 0.0

        print(
            f"{race:<{col_race}}  "
            f"{len(group):>5}  "
            f"{date_range:<23}  "
            f"{mean_share:.3f} ± {std_share:.3f}"
        )

    print()


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    polls_summary("2026")
