"""Fetch FiveThirtyEight data repository for historical poll analysis.

Downloads the 538 data repo and checking-our-work data from GitHub into
data/raw/fivethirtyeight/. These files are gitignored (large, ~887MB total).

538 shut down in 2024. Their data repo covers polls through the 2022 cycle.
No 2024 poll data exists in any 538 source. For 2024 backtesting, alternative
sources (e.g. RealClearPolitics, 270toWin) would need a separate ingestion
pipeline.

Usage:
    python scripts/fetch_538_data.py
    python scripts/fetch_538_data.py --skip-data-repo   # Only checking-our-work
    python scripts/fetch_538_data.py --skip-cow          # Only data repo
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_538_DIR = PROJECT_ROOT / "data" / "raw" / "fivethirtyeight"

# GitHub repository URLs for the two 538 data sources.
DATA_REPO_URL = "https://github.com/fivethirtyeight/data.git"
COW_REPO_URL = "https://github.com/fivethirtyeight/checking-our-work-data.git"

DATA_REPO_DIR = RAW_538_DIR / "data-repo"
COW_DIR = RAW_538_DIR / "checking-our-work-data"


def _clone_or_pull(url: str, target: Path) -> None:
    """Clone a git repo if it doesn't exist, or pull if it does."""
    if target.exists() and (target / ".git").exists():
        log.info("Updating existing repo: %s", target)
        subprocess.run(
            ["git", "-C", str(target), "pull", "--ff-only"],
            check=True,
        )
    elif target.exists():
        log.warning(
            "Directory exists but is not a git repo: %s — skipping. "
            "Delete it manually and re-run to re-clone.",
            target,
        )
    else:
        log.info("Cloning %s → %s", url, target)
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(target)],
            check=True,
        )


def _verify_key_files() -> None:
    """Check that the expected key files exist after download."""
    expected_files = [
        DATA_REPO_DIR / "pollster-ratings" / "raw_polls.csv",
        DATA_REPO_DIR / "pollster-ratings" / "pollster-ratings-combined.csv",
        COW_DIR / "presidential_elections.csv",
        COW_DIR / "us_senate_elections.csv",
        COW_DIR / "governors_elections.csv",
    ]
    missing = [f for f in expected_files if not f.exists()]
    if missing:
        log.error("Missing expected files after download:")
        for f in missing:
            log.error("  %s", f)
        sys.exit(1)
    else:
        log.info("All expected files present.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch FiveThirtyEight data for historical poll analysis"
    )
    parser.add_argument(
        "--skip-data-repo",
        action="store_true",
        help="Skip the main data repo (pollster ratings, raw polls)",
    )
    parser.add_argument(
        "--skip-cow",
        action="store_true",
        help="Skip the checking-our-work data (election forecasts)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    RAW_538_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_data_repo:
        _clone_or_pull(DATA_REPO_URL, DATA_REPO_DIR)

    if not args.skip_cow:
        _clone_or_pull(COW_REPO_URL, COW_DIR)

    _verify_key_files()
    log.info("538 data fetch complete. Run convert_538_polls.py to generate poll CSVs.")


if __name__ == "__main__":
    main()
