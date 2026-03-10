"""Data ingestion for the sabermetrics pipeline.

Downloads and links data from VoteView, thelawmakers.org (LES),
FEC bulk files, CES/CCES, Congress.gov API, and congressional
speech corpora. Builds the bioguide <-> icpsr <-> fec_id crosswalk.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


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


def build_id_crosswalk(output_path: str = "data/sabermetrics/id_crosswalk.parquet") -> "pd.DataFrame":
    """Build bioguide <-> icpsr <-> fec_id crosswalk.

    Primary source: unitedstates/congress-legislators YAML files
    (github.com/unitedstates/congress-legislators). Contains mappings
    between bioguide, icpsr, fec, govtrack, opensecrets, votesmart,
    and other ID systems.

    Returns
    -------
    pd.DataFrame
        Columns: bioguide_id, icpsr_id, fec_candidate_id, govtrack_id,
        name, party, state, chamber.
    """
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
