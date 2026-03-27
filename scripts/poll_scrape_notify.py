"""
Poll scrape notification helper.

Compares polls_2026.csv before and after a scrape run and returns a summary
suitable for a Telegram notification.

Usage (standalone):
    uv run python scripts/poll_scrape_notify.py snapshot   # save pre-scrape counts
    uv run python scripts/poll_scrape_notify.py diff        # print diff message

Usage (from bash wrapper):
    PRE=$(uv run python scripts/poll_scrape_notify.py snapshot)
    ... run scraper ...
    MSG=$(uv run python scripts/poll_scrape_notify.py diff "$PRE")
    ~/scripts/notify.sh "$MSG"
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
POLLS_CSV = PROJECT_ROOT / "data" / "polls" / "polls_2026.csv"


def _read_race_counts(csv_path: Path) -> dict[str, int]:
    """Return {race: poll_count} for every race in the CSV.

    Returns an empty dict if the file does not exist.
    """
    if not csv_path.exists():
        return {}
    counts: Counter[str] = Counter()
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            race = row.get("race", "").strip()
            if race:
                counts[race] += 1
    return dict(counts)


def snapshot(csv_path: Path = POLLS_CSV) -> str:
    """Return a JSON string encoding the current per-race poll counts."""
    counts = _read_race_counts(csv_path)
    return json.dumps(counts)


def build_diff_message(before_json: str, csv_path: Path = POLLS_CSV) -> str:
    """
    Compare pre-scrape counts (before_json) against current CSV and return a
    human-readable notification string.

    Returns a non-empty string in all cases (success or failure to parse).
    """
    try:
        before: dict[str, int] = json.loads(before_json)
    except (json.JSONDecodeError, ValueError):
        before = {}

    after = _read_race_counts(csv_path)

    total_before = sum(before.values())
    total_after = sum(after.values())
    new_total = total_after - total_before

    # Identify races with new polls
    new_by_race: dict[str, int] = {}
    for race, count in after.items():
        delta = count - before.get(race, 0)
        if delta > 0:
            new_by_race[race] = delta

    if new_total <= 0:
        return (
            f"low-priority status update: Weekly poll scrape complete, meat puppet. "
            f"No new polls ({total_after} total). API restarted."
        )

    # Build race breakdown (sorted for determinism)
    race_lines = ", ".join(
        f"{race} (+{delta})" for race, delta in sorted(new_by_race.items())
    )
    return (
        f"low-priority status update: Weekly poll scrape complete, meat puppet. "
        f"+{new_total} new polls ({total_after} total). "
        f"Races updated: {race_lines}. API restarted."
    )


def build_failure_message(stage: str) -> str:
    """Return a failure notification string for a given pipeline stage."""
    return (
        f"URGENT blocking issue: Weekly poll scrape FAILED at {stage}, meat puppet. "
        "Check ~/workspace/wethervane-poll-scrape.log"
    )


def _cmd_snapshot(csv_path: Path = POLLS_CSV) -> None:
    """Print JSON snapshot to stdout (used by bash wrapper)."""
    print(snapshot(csv_path))


def _cmd_diff(args: list[str], csv_path: Path = POLLS_CSV) -> None:
    """Print diff message to stdout (used by bash wrapper)."""
    before_json = args[0] if args else "{}"
    print(build_diff_message(before_json, csv_path))


def main(argv: list[str] | None = None) -> None:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)

    cmd = args[0]
    rest = args[1:]

    if cmd == "snapshot":
        _cmd_snapshot()
    elif cmd == "diff":
        _cmd_diff(rest)
    elif cmd == "failure":
        stage = rest[0] if rest else "unknown stage"
        print(build_failure_message(stage))
    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
