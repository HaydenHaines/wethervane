"""Data ingestion for the sabermetrics pipeline.

Downloads and links data from VoteView, thelawmakers.org (LES),
FEC bulk files, CES/CCES, Congress.gov API, and congressional
speech corpora. Builds the bioguide <-> icpsr <-> fec_id crosswalk.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import requests

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

# Base URL for the unitedstates/congress-legislators GitHub repository.
# These YAML files are the gold-standard crosswalk between bioguide, icpsr,
# fec_id, and ~10 other ID systems, maintained by the GovTrack community.
_CONGRESS_LEGISLATORS_BASE_URL = "https://raw.githubusercontent.com/unitedstates/congress-legislators/main/"

_CONGRESS_LEGISLATORS_FILES = [
    "legislators-current.yaml",
    "legislators-historical.yaml",
]


def download_congress_legislators(
    output_dir: str | Path = "data/raw/congress-legislators",
) -> None:
    """Download congress-legislators YAML files from GitHub.

    Downloads legislators-current.yaml and legislators-historical.yaml
    from the unitedstates/congress-legislators repository. These files
    contain bioguide/icpsr/fec/govtrack crosswalk for all historical
    legislators from 1789 to present.

    Parameters
    ----------
    output_dir : str | Path
        Directory to save YAML files (created if it doesn't exist).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename in _CONGRESS_LEGISLATORS_FILES:
        url = _CONGRESS_LEGISLATORS_BASE_URL + filename
        dest = output_dir / filename
        logger.info("Downloading %s → %s", url, dest)
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        dest.write_bytes(response.content)
        logger.info("  Saved %d bytes", len(response.content))


def build_id_crosswalk(
    output_path: str | Path = "data/sabermetrics/id_crosswalk.parquet",
    legislators_dir: str | Path = "data/raw/congress-legislators",
    download_if_missing: bool = True,
) -> "pd.DataFrame":
    """Build bioguide <-> icpsr <-> fec_id crosswalk from congress-legislators YAML.

    Primary source: unitedstates/congress-legislators YAML files
    (github.com/unitedstates/congress-legislators). Contains mappings
    between bioguide, icpsr, fec, govtrack, opensecrets, votesmart,
    and other ID systems.

    Parameters
    ----------
    output_path : str | Path
        Where to write the parquet output.
    legislators_dir : str | Path
        Directory containing the YAML files (downloaded if missing).
    download_if_missing : bool
        If True, download YAML files when not found on disk.

    Returns
    -------
    pd.DataFrame
        Columns: bioguide_id, icpsr_id, fec_candidate_ids, govtrack_id,
        name_full, name_last, name_first, party, state, term_types,
        term_start_years.
    """
    import pandas as pd

    legislators_dir = Path(legislators_dir)
    output_path = Path(output_path)

    # Download YAML files if not present
    missing = [f for f in _CONGRESS_LEGISLATORS_FILES if not (legislators_dir / f).exists()]
    if missing:
        if download_if_missing:
            logger.info("Downloading missing congress-legislators files: %s", missing)
            download_congress_legislators(legislators_dir)
        else:
            raise FileNotFoundError(
                f"Missing files in {legislators_dir}: {missing}. Pass download_if_missing=True to auto-download."
            )

    # Load and parse legislators
    from src.sabermetrics.registry import load_congress_legislators

    legislators = load_congress_legislators(legislators_dir)

    # Flatten to tabular format
    rows = []
    for leg in legislators:
        rows.append(
            {
                "bioguide_id": leg["bioguide_id"],
                "name_full": leg["name_full"],
                "name_last": leg["name_last"],
                "name_first": leg["name_first"],
                # Parties and states are lists (multi-career); store as pipe-joined strings
                "party": "|".join(leg["party_codes"]),
                "state": "|".join(leg["states"]),
                "term_types": "|".join(leg["term_types"]),
                "term_start_years": "|".join(str(y) for y in leg["term_years"]),
            }
        )

    df = pd.DataFrame(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("Crosswalk written to %s: %d legislators", output_path, len(df))
    return df


def download_voteview(output_dir: str = "data/raw/voteview") -> None:
    """Download HSall_members.csv, HSall_votes.csv, HSall_rollcalls.csv from voteview.com/data."""
    raise NotImplementedError


def download_les(output_dir: str = "data/raw/les") -> None:
    """Download Legislative Effectiveness Score data from thelawmakers.org/data-download."""
    raise NotImplementedError


def download_fec_bulk(
    cycles: list[str],
    output_dir: str = "data/raw/fec",
) -> None:
    """Download FEC bulk files (candidates, committees, individual contributions, disbursements).

    Parameters
    ----------
    cycles : list[str]
        Election cycles to download, e.g. ["2020", "2022", "2024"].
    """
    raise NotImplementedError


def download_ces_cumulative(output_dir: str = "data/raw/ces") -> None:
    """Download CES cumulative file from Harvard Dataverse (DOI: 10.7910/DVN/II2DB6)."""
    raise NotImplementedError


def fetch_congress_api(
    congress: int,
    output_dir: str = "data/raw/congress_api",
) -> None:
    """Fetch bills, co-sponsorships, votes, and committees from Congress.gov API.

    Parameters
    ----------
    congress : int
        Congress number (e.g., 118 for the 118th Congress).
    """
    raise NotImplementedError


def download_congressional_speeches(output_dir: str = "data/raw/congressional_record") -> None:
    """Download Gentzkow-Shapiro-Taddy dataset + ConSpeak corpus."""
    raise NotImplementedError


def build_politician_roster(
    crosswalk: "pd.DataFrame",
    voteview_members: "pd.DataFrame",
    fec_candidates: "pd.DataFrame",
) -> "pd.DataFrame":
    """Build master politician roster with all IDs and office history.

    Returns
    -------
    pd.DataFrame
        One row per politician with all IDs, party, state, and
        chronological list of offices held.
    """
    raise NotImplementedError
