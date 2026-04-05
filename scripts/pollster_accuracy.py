"""
Pollster accuracy analysis for 2022 elections.

Cross-references pollster predictions from polls_2022.csv against actual
outcomes in backtest_2022_results.json, then ranks pollsters by accuracy.

The backtest file provides state-level actual dem_share (already aggregated
from county predictions). Polls are also at the state level, so no aggregation
step is needed — we match each poll directly to its corresponding state+race
actual result.

Output: data/experiments/pollster_accuracy.json

Usage:
    uv run python scripts/pollster_accuracy.py
"""

from __future__ import annotations

import csv
import json
import logging
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLLS_CSV = PROJECT_ROOT / "data" / "polls" / "polls_2022.csv"
BACKTEST_JSON = PROJECT_ROOT / "data" / "experiments" / "backtest_2022_results.json"
OUTPUT_JSON = PROJECT_ROOT / "data" / "experiments" / "pollster_accuracy.json"


# ---------------------------------------------------------------------------
# Race parsing
# ---------------------------------------------------------------------------

# Maps the race-type word in the poll race string to the backtest dict key.
# e.g. "2022 GA Governor" -> race_type="Governor" -> "governor"
_RACE_TYPE_MAP = {
    "governor": "governor",
    "senate": "senate",
}

# Regex to extract year, state abbreviation, and race type from a race string.
# Handles "2022 FL Senate", "2022 GA Governor", etc.
_RACE_RE = re.compile(r"^(\d{4})\s+([A-Z]{2})\s+(\w+)", re.IGNORECASE)


def parse_race_string(race: str) -> tuple[str, str, str] | None:
    """Parse a race string into (year, state_abbr, race_type_key).

    Returns None if the race string doesn't match the expected format.
    Example: "2022 GA Governor" -> ("2022", "GA", "governor")
    """
    m = _RACE_RE.match(race.strip())
    if not m:
        return None
    year, state, race_word = m.group(1), m.group(2).upper(), m.group(3).lower()
    race_type_key = _RACE_TYPE_MAP.get(race_word)
    if race_type_key is None:
        return None
    return year, state, race_type_key


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_backtest_actuals(path: Path) -> dict[tuple[str, str], float]:
    """Load backtest state-level actual dem_share by race type.

    Returns a dict mapping (race_type_key, state_abbr) -> actual_dem_share.
    Example: {("governor", "GA"): 0.4615, ("senate", "GA"): 0.5049, ...}

    Uses the raw_prior section (not type_mean_prior) because raw_prior has
    broader state coverage with county-aggregated actuals.
    """
    with open(path) as f:
        data = json.load(f)

    actuals: dict[tuple[str, str], float] = {}
    for race_type_key in ("governor", "senate"):
        section = data.get(race_type_key, {})
        raw = section.get("raw_prior", {})
        state_metrics = raw.get("state_metrics", [])
        for entry in state_metrics:
            state = entry["state"].upper()
            actual = entry["actual"]
            actuals[(race_type_key, state)] = float(actual)

    return actuals


def load_polls(path: Path) -> list[dict]:
    """Load polls from CSV.

    Returns a list of dicts with fields:
        race, geography, dem_share (float), n_sample (int), date, pollster, notes
    """
    polls = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                dem_share = float(row["dem_share"])
            except (ValueError, KeyError):
                log.warning("Skipping row with invalid dem_share: %s", row)
                continue
            try:
                n_sample = int(row.get("n_sample", 0) or 0)
            except ValueError:
                n_sample = 0

            polls.append({
                "race": row.get("race", "").strip(),
                "geography": row.get("geography", "").strip().upper(),
                "dem_share": dem_share,
                "n_sample": n_sample,
                "date": row.get("date", "").strip(),
                "pollster": row.get("pollster", "").strip(),
                "notes": row.get("notes", "").strip(),
            })
    return polls


# ---------------------------------------------------------------------------
# Pollster name normalization
# ---------------------------------------------------------------------------

# Known pollster name variants to normalize to a canonical form.
# Each entry maps variant -> canonical_name.
_POLLSTER_ALIASES: dict[str, str] = {
    "The New York Times/Siena College": "New York Times/Siena College",
    "Fabrizio Lee & Associates/Impact Research": "Fabrizio/Impact Research",
}


def normalize_pollster_name(name: str) -> str:
    """Normalize a pollster name to a canonical form.

    Strips leading/trailing whitespace, collapses interior whitespace,
    and applies known alias mappings.
    """
    # Collapse interior whitespace and strip
    normalized = " ".join(name.split())
    # Apply alias mapping
    return _POLLSTER_ALIASES.get(normalized, normalized)


# ---------------------------------------------------------------------------
# Core accuracy calculation
# ---------------------------------------------------------------------------


def compute_pollster_accuracy(
    polls: list[dict],
    actuals: dict[tuple[str, str], float],
) -> list[dict]:
    """Compute per-pollster accuracy metrics by comparing poll predictions to actuals.

    For each poll, we find the actual dem_share for that race+state from the
    backtest, then compute the signed error (poll_dem_share - actual_dem_share).
    We report error in percentage points (multiply by 100) to match the
    backtest's own error_pp convention.

    Pollsters with polls in races not covered by the backtest are skipped
    gracefully — only polls that can be matched to a backtest actual are used.

    Returns a list of dicts sorted by RMSE (ascending = most accurate first):
        pollster, n_polls, n_races, rmse_pp, mean_error_pp (bias), rank
    """
    # Accumulate errors per pollster
    # pollster -> list of signed errors in percentage points
    pollster_errors: dict[str, list[float]] = defaultdict(list)
    # Track which races each pollster covered for context
    pollster_races: dict[str, set[str]] = defaultdict(set)

    skipped = 0
    matched = 0

    for poll in polls:
        parsed = parse_race_string(poll["race"])
        if parsed is None:
            log.debug("Cannot parse race string: %s", poll["race"])
            skipped += 1
            continue

        _year, state, race_type_key = parsed
        actual = actuals.get((race_type_key, state))
        if actual is None:
            # Backtest doesn't cover this state/race — not an error, just skip
            log.debug(
                "No backtest actual for (%s, %s) — poll from %s skipped",
                race_type_key, state, poll["pollster"]
            )
            skipped += 1
            continue

        pollster = normalize_pollster_name(poll["pollster"])
        if not pollster:
            skipped += 1
            continue

        # Signed error in percentage points: positive = poll over-predicts Dem
        error_pp = (poll["dem_share"] - actual) * 100.0
        pollster_errors[pollster].append(error_pp)
        pollster_races[pollster].add(poll["race"])
        matched += 1

    log.info("Matched %d polls, skipped %d", matched, skipped)

    if not pollster_errors:
        log.warning("No polls matched to backtest actuals — output will be empty")
        return []

    # Build per-pollster metrics
    results = []
    for pollster, errors in pollster_errors.items():
        n = len(errors)
        mean_err = sum(errors) / n
        rmse = math.sqrt(sum(e * e for e in errors) / n)
        results.append({
            "pollster": pollster,
            "n_polls": n,
            "n_races": len(pollster_races[pollster]),
            "rmse_pp": round(rmse, 3),
            "mean_error_pp": round(mean_err, 3),
        })

    # Sort by RMSE ascending (most accurate first); break ties by n_polls descending
    results.sort(key=lambda x: (x["rmse_pp"], -x["n_polls"]))

    # Add rank (1 = most accurate)
    for i, row in enumerate(results, start=1):
        row["rank"] = i

    return results


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------


def print_accuracy_table(results: list[dict]) -> None:
    """Print a formatted ranked table of pollster accuracy."""
    if not results:
        print("No results to display.")
        return

    header = f"{'Rank':>4}  {'Pollster':<50}  {'n_polls':>7}  {'RMSE (pp)':>9}  {'Bias (pp)':>9}"
    print()
    print(header)
    print("-" * len(header))
    for row in results:
        bias_str = f"{row['mean_error_pp']:+.2f}"
        print(
            f"  {row['rank']:>2}  "
            f"{row['pollster']:<50}  "
            f"{row['n_polls']:>7}  "
            f"{row['rmse_pp']:>9.2f}  "
            f"{bias_str:>9}"
        )
    print()


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def save_results(results: list[dict], path: Path) -> None:
    """Save pollster accuracy results to JSON.

    The JSON has a top-level metadata block and a 'pollsters' array sorted by rank.
    """
    output = {
        "description": (
            "Pollster accuracy for 2022 elections. "
            "RMSE and bias are in percentage points. "
            "Positive mean_error_pp means the pollster systematically over-predicted "
            "the Democratic share (polling Dem-leaning). "
            "Sorted by RMSE ascending (rank 1 = most accurate)."
        ),
        "source_polls": str(POLLS_CSV.relative_to(PROJECT_ROOT)),
        "source_backtest": str(BACKTEST_JSON.relative_to(PROJECT_ROOT)),
        "n_pollsters": len(results),
        "pollsters": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Saved %d pollster records to %s", len(results), path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(
    polls_path: Path = POLLS_CSV,
    backtest_path: Path = BACKTEST_JSON,
    output_path: Path = OUTPUT_JSON,
) -> list[dict]:
    """Run the full pollster accuracy analysis and write output JSON.

    Exposed as a function (not just __main__) so tests can call it directly.
    Returns the list of pollster accuracy dicts.
    """
    polls = load_polls(polls_path)
    actuals = load_backtest_actuals(backtest_path)
    results = compute_pollster_accuracy(polls, actuals)
    print_accuracy_table(results)
    save_results(results, output_path)
    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    run()
