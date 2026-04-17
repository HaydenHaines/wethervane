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
    file_types: list[str] | None = None,
) -> dict[str, Path]:
    """Download FEC bulk summary files needed for campaign finance stats.

    Downloads two files per cycle:
    - weball{YY}.zip: all-candidates summary (receipts, disbursements, etc.)
    - cm{YY}.zip: committee master (links committee ID to candidate ID)

    These are the lightest FEC bulk files — a few MB each — compared to the
    full individual-contribution files which are gigabytes. They contain
    everything needed to compute SDR, FER, and burn rate.

    Parameters
    ----------
    cycles : list[str]
        Election cycles to download, e.g. ["2022", "2024"].
    output_dir : str
        Directory where zip files will be saved (created if absent).
    file_types : list[str] | None
        Which file types to download: "weball" and/or "cm".
        Defaults to both.

    Returns
    -------
    dict[str, Path]
        Mapping of "{file_type}{YY}" -> Path for each downloaded file.
        E.g. {"weball22": Path(".../weball22.zip"), "cm22": Path(..)}.

    Notes
    -----
    FEC bulk download base URL:
        https://www.fec.gov/files/bulk-downloads/{YYYY}/
    All-candidates file description:
        https://www.fec.gov/campaign-finance-data/all-candidates-file-description/
    Committee master description:
        https://www.fec.gov/campaign-finance-data/committee-master-file-description/
    """
    if file_types is None:
        file_types = ["weball", "cm"]

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # FEC uses two-digit cycle suffixes: "2022" → "22"
    _FEC_BASE_URL = "https://www.fec.gov/files/bulk-downloads"

    downloaded: dict[str, Path] = {}

    for cycle in cycles:
        # Validate cycle is a 4-digit year string
        if not (len(cycle) == 4 and cycle.isdigit()):
            raise ValueError(f"Cycle must be a 4-digit year string, got {cycle!r}")
        cycle_suffix = cycle[-2:]  # "2022" → "22"

        for file_type in file_types:
            filename = f"{file_type}{cycle_suffix}.zip"
            url = f"{_FEC_BASE_URL}/{cycle}/{filename}"
            dest = output_dir_path / filename

            if dest.exists():
                logger.info("FEC bulk file already exists, skipping: %s", dest)
                downloaded[f"{file_type}{cycle_suffix}"] = dest
                continue

            logger.info("Downloading %s → %s", url, dest)
            response = requests.get(url, timeout=120, stream=True)
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                raise RuntimeError(
                    f"Failed to download FEC bulk file {url}: HTTP {response.status_code}"
                ) from exc

            # Stream to disk to avoid loading multi-GB files into memory
            total_bytes = 0
            with open(dest, "wb") as fh:
                for chunk in response.iter_content(chunk_size=65536):
                    fh.write(chunk)
                    total_bytes += len(chunk)

            logger.info("  Saved %s (%.1f MB)", dest, total_bytes / 1_048_576)
            downloaded[f"{file_type}{cycle_suffix}"] = dest

    return downloaded


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
