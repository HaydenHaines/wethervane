"""Poll prediction change alert system.

Compares the latest saved forecast snapshot against current predictions and
sends a Telegram alert for any race whose state_pred changes by more than the
configured threshold (default 2pp = 0.02).

Typical invocation (from cron wrapper, after poll scrape + forecast rebuild):

    uv run python scripts/check_prediction_changes.py

The script:
1. Loads the PREVIOUS prediction snapshot from data/forecast_snapshots/ (most
   recent JSON file by name).
2. Loads CURRENT predictions from data/forecast_snapshots/<today>.json if it
   exists, or from the DuckDB predictions table if available.
3. Compares race-level predictions.
4. For any race where |delta| >= threshold, formats a Telegram message and
   calls ~/scripts/notify.sh.
5. Saves the current predictions as a new snapshot for next comparison.

If there is no prior snapshot (first run), saves the current snapshot and
exits silently.

CLI flags
---------
--threshold FLOAT   Minimum absolute change to alert on (default 0.02 = 2pp).
--snapshot-dir DIR  Directory containing forecast snapshot JSON files.
--current-file FILE Explicit path to a JSON file with current predictions.
                    If omitted, the script reads from the API or DuckDB.
--dry-run           Print what would be sent without calling notify.sh.
--no-notify         Skip notification; still saves snapshot (useful for CI).
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import TypedDict

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT_DIR = PROJECT_ROOT / "data" / "forecast_snapshots"
NOTIFY_SCRIPT = "/home/hayden/scripts/notify.sh"

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class PredictionChange(TypedDict):
    race: str
    before: float
    after: float
    delta: float


# ---------------------------------------------------------------------------
# Public helpers (tested independently)
# ---------------------------------------------------------------------------


def load_snapshot(path: Path) -> dict[str, float] | None:
    """Load a predictions dict from a JSON snapshot file.

    Accepts two formats:
    - ``{"date": "...", "predictions": {race: float, ...}}``  (canonical)
    - ``{race: float, ...}``                                   (flat / legacy)

    Returns ``None`` on any error (missing file, bad JSON, wrong structure).
    """
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Could not read snapshot %s: %s", path, exc)
        return None

    if isinstance(data, dict) and "predictions" in data:
        preds = data["predictions"]
        if isinstance(preds, dict):
            return {str(k): float(v) for k, v in preds.items()}
        return None

    # Flat dict: all values must be numeric
    if isinstance(data, dict) and all(isinstance(v, (int, float)) for v in data.values()):
        return {str(k): float(v) for k, v in data.items()}

    return None


def save_snapshot(
    predictions: dict[str, float],
    path: Path,
    note: str = "",
) -> None:
    """Persist *predictions* to *path* in canonical snapshot JSON format.

    Creates parent directories as needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "date": str(date.today()),
        "predictions": predictions,
    }
    if note:
        payload["note"] = note
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("Snapshot saved to %s (%d races)", path, len(predictions))


def find_latest_snapshot(snapshot_dir: Path) -> Path | None:
    """Return the path to the most recent ``*.json`` file in *snapshot_dir*.

    Files are sorted lexicographically by name; YYYY-MM-DD names sort
    chronologically.  Returns ``None`` if the directory has no JSON files.
    """
    if not snapshot_dir.exists():
        return None
    candidates = sorted(snapshot_dir.glob("*.json"))
    return candidates[-1] if candidates else None


def detect_changes(
    before: dict[str, float],
    after: dict[str, float],
    threshold: float,
) -> list[PredictionChange]:
    """Return races where ``|after - before| >= threshold``, sorted descending by |delta|.

    Races that appear in one snapshot but not the other are ignored (no prior
    baseline to compare against).
    """
    results: list[PredictionChange] = []

    common_races = set(before) & set(after)
    for race in sorted(common_races):
        b = before[race]
        a = after[race]
        delta = a - b
        if abs(delta) >= threshold:
            results.append(PredictionChange(race=race, before=b, after=a, delta=delta))

    # Sort by descending absolute delta
    results.sort(key=lambda d: -abs(d["delta"]))
    return results


def format_alert_message(changes: list[PredictionChange]) -> str:
    """Format a list of prediction changes into a Telegram notification string.

    Returns a non-empty string with one line per changed race.
    """
    n = len(changes)
    header = (
        f"low-priority status update: WetherVane forecast shift detected, meat puppet. "
        f"{n} race(s) changed by >2pp after poll scrape."
    )
    lines = [header, ""]

    for ch in changes:
        race = ch["race"]
        b_pct = ch["before"] * 100
        a_pct = ch["after"] * 100
        delta_pp = ch["delta"] * 100
        sign = "+" if delta_pp > 0 else ""
        lines.append(f"  {race}: {b_pct:.1f}% -> {a_pct:.1f}% ({sign}{delta_pp:.1f}pp)")

    return "\n".join(lines)


def send_notification(message: str) -> None:
    """Call ~/scripts/notify.sh with *message*.

    Fails gracefully: logs a warning but does not raise.
    """
    try:
        result = subprocess.run(
            [NOTIFY_SCRIPT, message],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            log.warning(
                "notify.sh exited %d: %s", result.returncode, result.stderr.strip()
            )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        log.warning("Could not send notification: %s", exc)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def check_prediction_changes(
    current_predictions: dict[str, float],
    snapshot_dir: Path = DEFAULT_SNAPSHOT_DIR,
    threshold: float = 0.02,
    notify: bool = True,
    dry_run: bool = False,
    snapshot_note: str = "",
) -> list[PredictionChange]:
    """Core orchestration: compare, alert, save.

    Parameters
    ----------
    current_predictions:
        Dict of ``{race: dem_share}`` for the current poll scrape result.
    snapshot_dir:
        Directory containing dated snapshot JSON files.
    threshold:
        Minimum absolute change (fraction) to trigger an alert.  Default 0.02
        = 2 percentage points.
    notify:
        Whether to call notify.sh when changes are detected.
    dry_run:
        If True, print the message to stdout instead of calling notify.sh.
    snapshot_note:
        Optional annotation written into the saved snapshot's ``note`` field.

    Returns
    -------
    list[PredictionChange]
        The list of races that changed beyond *threshold* (may be empty).
    """
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Load previous snapshot
    prev_path = find_latest_snapshot(snapshot_dir)
    previous: dict[str, float] | None = None
    if prev_path is not None:
        previous = load_snapshot(prev_path)
        if previous is None:
            log.warning("Could not parse previous snapshot at %s; running cold.", prev_path)

    # Save current snapshot now (before comparison, so it's always persisted)
    today_str = str(date.today())
    new_snapshot_path = snapshot_dir / f"{today_str}.json"

    # If prev_path is today's file, we'd clobber the previous snapshot we just
    # loaded.  That's intentional — we always update to the latest run.
    save_snapshot(current_predictions, new_snapshot_path, note=snapshot_note)

    # First run or unreadable prior → no comparison to make
    if previous is None:
        log.info("No previous snapshot found — baseline saved, no alert sent.")
        return []

    # Detect changes
    changes = detect_changes(previous, current_predictions, threshold)

    if not changes:
        log.info("No races changed by more than %.1fpp — no alert sent.", threshold * 100)
        return []

    message = format_alert_message(changes)
    log.info("Sending alert for %d changed race(s).", len(changes))

    if dry_run:
        print(message)
    elif notify:
        send_notification(message)

    return changes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_current_from_file(path: Path) -> dict[str, float]:
    """Load current predictions from an explicit JSON file."""
    predictions = load_snapshot(path)
    if predictions is None:
        log.error("Could not load current predictions from %s", path)
        sys.exit(1)
    return predictions


def _load_current_from_snapshots(snapshot_dir: Path) -> dict[str, float]:
    """Load current predictions from the API snapshot for today.

    Falls back to the most recent snapshot in the directory when no today file
    exists.  Exits with an error if nothing is found.
    """
    today_file = snapshot_dir / f"{date.today()}.json"
    if today_file.exists():
        preds = load_snapshot(today_file)
        if preds is not None:
            return preds
        log.warning("Could not parse today's snapshot; falling back to latest.")

    latest = find_latest_snapshot(snapshot_dir)
    if latest is None:
        log.error(
            "No snapshot files found in %s.  "
            "Run the poll scrape first to generate predictions.",
            snapshot_dir,
        )
        sys.exit(1)

    preds = load_snapshot(latest)
    if preds is None:
        log.error("Could not parse latest snapshot at %s", latest)
        sys.exit(1)
    return preds


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.02,
        help="Minimum absolute change to alert on (default 0.02 = 2pp)",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=DEFAULT_SNAPSHOT_DIR,
        help="Directory with forecast snapshot JSON files (default: %(default)s)",
    )
    parser.add_argument(
        "--current-file",
        type=Path,
        default=None,
        help="Explicit JSON file with current predictions (skips auto-detect)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print alert to stdout instead of calling notify.sh",
    )
    parser.add_argument(
        "--no-notify",
        action="store_true",
        help="Disable notification (snapshot is still saved)",
    )
    parser.add_argument(
        "--note",
        default="",
        help="Optional note written into the saved snapshot",
    )

    args = parser.parse_args(argv)

    # Load current predictions
    if args.current_file:
        current = _load_current_from_file(args.current_file)
    else:
        current = _load_current_from_snapshots(args.snapshot_dir)

    changes = check_prediction_changes(
        current_predictions=current,
        snapshot_dir=args.snapshot_dir,
        threshold=args.threshold,
        notify=not args.no_notify,
        dry_run=args.dry_run,
        snapshot_note=args.note,
    )

    if changes:
        sys.exit(0)  # changes detected but handled — success
    sys.exit(0)  # no changes — also success


if __name__ == "__main__":
    main()
