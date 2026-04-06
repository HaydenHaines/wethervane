"""Fetch the presidential approval rating from RealClearPolling and update
snapshot_2026.json.

RCP pages are Next.js apps that embed data in ``self.__next_f.push()`` script
tags (same technique as ``scrape_2026_polls.py``).  This script locates the
RCP Average row in the approval-rating poll table, reads the Approve and
Disapprove percentages, and writes the net approval (Approve% - Disapprove%)
to ``data/fundamentals/snapshot_2026.json``.

Usage::

    uv run python scripts/fetch_approval_rating.py [--dry-run]

"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = PROJECT_ROOT / "data" / "fundamentals" / "snapshot_2026.json"

RCP_APPROVAL_URL = (
    "https://www.realclearpolling.com/polls/approval/donald-trump/approval-rating"
)

# RCP embeds the poll data in a candidate array with these names.
APPROVE_NAME = "Approve"
DISAPPROVE_NAME = "Disapprove"

# The type field that identifies the RCP composite average row.
RCP_AVERAGE_TYPE = "rcp_average"

# HTTP request timeout in seconds.
_HTTP_TIMEOUT = 30

# User-Agent that looks like a browser so RCP doesn't block us.
_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


# ---------------------------------------------------------------------------
# HTML fetch
# ---------------------------------------------------------------------------


def fetch_html(url: str) -> str | None:
    """Download a URL and return the response body as a string.

    Returns None on network or HTTP error; the caller decides how to handle it.
    """
    req = Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except URLError as exc:
        log.warning("Network error fetching %s: %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# JSON extraction (mirrors _extract_rcp_polls_json in scrape_2026_polls.py)
# ---------------------------------------------------------------------------


def _extract_polls_array(html: str) -> list[dict] | None:
    """Extract the ``polls`` JSON array from a RCP Next.js page.

    RCP pages embed their data in ``self.__next_f.push([1, "<escaped>"])``
    script tags.  We find the script tag that contains ``rcp_average`` (which
    identifies approval-rating pages), decode the escaped JSON payload, then
    bracket-match out the ``"polls"`` array so we don't have to parse the full
    megabyte-scale JSON tree.

    Returns None if the data cannot be found or parsed so the caller can bail
    cleanly instead of raising.
    """
    # Find all inline script contents.
    scripts = re.findall(r"<script[^>]*>(.*?)</script>", html, re.DOTALL)

    for script in scripts:
        # Only look at scripts that contain the RCP average marker.
        if RCP_AVERAGE_TYPE not in script:
            continue

        # Script follows the pattern: self.__next_f.push([1,"..."]);
        m = re.search(
            r'self\.__next_f\.push\(\[(\d+),"(.*)"\]\)',
            script,
            re.DOTALL,
        )
        if not m:
            log.debug("Script has rcp_average but didn't match __next_f pattern")
            continue

        # The second argument is a JSON-encoded string — decode it once.
        try:
            decoded = json.loads('"' + m.group(2) + '"')
        except json.JSONDecodeError as exc:
            log.debug("JSON decode failed for script payload: %s", exc)
            continue

        # Locate the ``"polls":`` key and bracket-match its array value.
        # We cannot simply regex-extract the array because it contains nested
        # objects with arbitrary content.  Note: json.dumps may or may not
        # insert a space after the colon ("polls": [ vs "polls":[), so we
        # search for the key without the bracket, then scan forward to "[".
        polls_marker = '"polls":'
        marker_idx = decoded.find(polls_marker)
        if marker_idx < 0:
            log.debug("'polls' key not found in decoded payload")
            continue

        # Skip past the colon and any optional whitespace to land on "[".
        array_start = marker_idx + len(polls_marker)
        while array_start < len(decoded) and decoded[array_start] != "[":
            array_start += 1
        if array_start >= len(decoded):
            log.debug("'polls' key found but no opening bracket")
            continue
        bracket_depth = 0
        array_end = array_start

        for pos, ch in enumerate(decoded[array_start:], array_start):
            if ch == "[":
                bracket_depth += 1
            elif ch == "]":
                bracket_depth -= 1
                if bracket_depth == 0:
                    array_end = pos + 1
                    break

        if array_end == array_start:
            log.debug("Could not bracket-match polls array end")
            continue

        try:
            return json.loads(decoded[array_start:array_end])
        except json.JSONDecodeError as exc:
            log.debug("JSON decode failed for polls array: %s", exc)
            continue

    return None


# ---------------------------------------------------------------------------
# Approval extraction
# ---------------------------------------------------------------------------


def _find_rcp_average(polls: list[dict]) -> dict | None:
    """Return the RCP Average composite entry from the polls list, or None."""
    for entry in polls:
        if entry.get("type") == RCP_AVERAGE_TYPE:
            return entry
    return None


def _parse_percentage(value: str | None) -> float | None:
    """Parse a percentage string like ``"40.9"`` to a float, or None on failure."""
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def extract_approval_from_polls(polls: list[dict]) -> dict | None:
    """Extract approve%, disapprove%, net approval, and date from the polls list.

    Looks for the ``rcp_average`` type entry, then reads the Approve and
    Disapprove candidate values from its ``candidate`` array.

    Returns a dict with keys::

        {
            "approve_pct": 40.9,
            "disapprove_pct": 56.9,
            "net_approval": -16.0,
            "date_range": "3/12 - 4/2",   # as shown on the RCP page
        }

    or None if the data cannot be parsed.
    """
    average_entry = _find_rcp_average(polls)
    if average_entry is None:
        log.warning("No rcp_average entry found in polls list")
        return None

    candidates = average_entry.get("candidate", [])
    approve_pct: float | None = None
    disapprove_pct: float | None = None

    for cand in candidates:
        name = (cand.get("name") or "").strip()
        value = cand.get("value")
        if name.lower() == APPROVE_NAME.lower():
            approve_pct = _parse_percentage(value)
        elif name.lower() == DISAPPROVE_NAME.lower():
            disapprove_pct = _parse_percentage(value)

    if approve_pct is None or disapprove_pct is None:
        log.warning(
            "Could not parse approve/disapprove from RCP average entry: %s",
            average_entry.get("candidate"),
        )
        return None

    net_approval = round(approve_pct - disapprove_pct, 1)
    date_range = average_entry.get("date", "")

    log.info(
        "RCP Average — Approve: %.1f%%  Disapprove: %.1f%%  Net: %.1f",
        approve_pct,
        disapprove_pct,
        net_approval,
    )

    return {
        "approve_pct": approve_pct,
        "disapprove_pct": disapprove_pct,
        "net_approval": net_approval,
        "date_range": date_range,
    }


# ---------------------------------------------------------------------------
# Snapshot update
# ---------------------------------------------------------------------------


def load_snapshot() -> dict:
    """Load snapshot_2026.json, returning a default skeleton if absent."""
    if SNAPSHOT_PATH.exists():
        return json.loads(SNAPSHOT_PATH.read_text())
    log.warning("Snapshot not found at %s; starting from defaults", SNAPSHOT_PATH)
    return {
        "cycle": 2026,
        "in_party": "D",
        "approval_net_oct": -12.0,
        "source_notes": {},
    }


def update_snapshot(approval_data: dict, dry_run: bool = False) -> dict:
    """Write the new net approval into snapshot_2026.json.

    Only updates ``approval_net_oct`` and ``source_notes.approval``.  All other
    fields are left unchanged.  Returns the updated snapshot dict.
    """
    snapshot = load_snapshot()

    old_value = snapshot.get("approval_net_oct")
    new_value = approval_data["net_approval"]
    snapshot["approval_net_oct"] = new_value

    fetched_at = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    source_notes = snapshot.get("source_notes", {})
    source_notes["approval"] = (
        f"RealClearPolling average — "
        f"Approve {approval_data['approve_pct']:.1f}%, "
        f"Disapprove {approval_data['disapprove_pct']:.1f}%, "
        f"Net {new_value:+.1f}pp. "
        f"Date range on page: {approval_data['date_range']}. "
        f"Fetched {fetched_at}."
    )
    source_notes["last_updated"] = fetched_at
    snapshot["source_notes"] = source_notes

    change_str = ""
    if old_value is not None and old_value != new_value:
        change_str = f" (was {old_value})"
    print(f"  approval_net_oct: {new_value}{change_str}")

    if not dry_run:
        SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2) + "\n")
        print(f"\nWrote {SNAPSHOT_PATH}")
    else:
        print(f"\n[DRY RUN] Would write to {SNAPSHOT_PATH}")
        print(json.dumps(snapshot, indent=2))

    return snapshot


# ---------------------------------------------------------------------------
# Public entry point (also useful for testing)
# ---------------------------------------------------------------------------


def fetch_approval_rating(url: str = RCP_APPROVAL_URL) -> dict | None:
    """Fetch the RCP approval rating and return the parsed data dict.

    Returns None if fetching or parsing fails; logs a warning in that case.
    This function does NOT touch the snapshot — call ``update_snapshot()``
    separately.
    """
    log.info("Fetching approval rating from %s", url)
    html = fetch_html(url)
    if html is None:
        log.error("Failed to download RCP approval page")
        return None

    polls = _extract_polls_array(html)
    if polls is None:
        log.error("Could not extract polls array from RCP page HTML")
        return None

    log.info("Extracted %d poll entries (including RCP average)", len(polls))
    return extract_approval_from_polls(polls)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch the current presidential approval rating from RCP "
            "and update data/fundamentals/snapshot_2026.json."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without modifying the file.",
    )
    parser.add_argument(
        "--url",
        default=RCP_APPROVAL_URL,
        help="Override the RCP URL (useful for testing against a cached file).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print(f"Fetching presidential approval rating from RCP...")
    approval_data = fetch_approval_rating(url=args.url)

    if approval_data is None:
        print(
            "ERROR: could not scrape approval rating. "
            "Snapshot was NOT modified.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\nUpdating {SNAPSHOT_PATH}:")
    update_snapshot(approval_data, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
