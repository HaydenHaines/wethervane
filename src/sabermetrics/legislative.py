"""Legislative performance stats.

Covers: LES import, amendment success rate, co-sponsorship network
centrality, bipartisan deviation rate, strategic defection/loyalty
scoring (Snyder-Groseclose methodology).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def import_les(les_path: str) -> "pd.DataFrame":
    """Import Volden-Wiseman Legislative Effectiveness Scores.

    Source: thelawmakers.org, 93rd-118th Congress, House and Senate.
    Includes LES 1.0 and LES 2.0 (with text incorporation credit).

    Returns
    -------
    pd.DataFrame
        Columns: icpsr_id, congress, chamber, les_1, les_2,
        bills_introduced, bills_enacted, pct_substantive,
        pct_significant, committee_chair.
    """
    raise NotImplementedError


def compute_amendment_success_rate(
    amendments: "pd.DataFrame",
    bill_significance: "pd.DataFrame",
) -> "pd.DataFrame":
    """Compute Amendment Success Rate (ASR) per legislator per Congress.

    Weighted by bill significance (commemorative=1, substantive=5,
    significant=10), following the LES weighting scheme.

    Parameters
    ----------
    amendments : pd.DataFrame
        Columns: sponsor_bioguide, amendment_id, bill_id,
        status (adopted/rejected/withdrawn).
    bill_significance : pd.DataFrame
        Columns: bill_id, significance (commemorative/substantive/significant).

    Returns
    -------
    pd.DataFrame
        Columns: bioguide_id, congress, asr_raw, asr_weighted,
        amendments_offered, amendments_adopted.
    """
    raise NotImplementedError


def build_cosponsorship_network(
    cosponsorship_data: "pd.DataFrame",
    congress: int,
) -> tuple:
    """Build co-sponsorship network for a given Congress.

    Nodes = legislators. Edges = co-sponsorship relationships,
    weighted by count.

    Parameters
    ----------
    cosponsorship_data : pd.DataFrame
        Columns: bill_id, sponsor_bioguide, cosponsor_bioguide.
    congress : int
        Congress number.

    Returns
    -------
    tuple
        (graph object, centrality_df) where centrality_df has columns:
        bioguide_id, pagerank, betweenness, eigenvector,
        cross_party_cosponsorship_rate.
    """
    raise NotImplementedError


def compute_party_loyalty(
    member_votes: "pd.DataFrame",
    rollcall_metadata: "pd.DataFrame",
) -> "pd.DataFrame":
    """Compute Bipartisan Deviation Rate (BDR) and Strategic Defection/Loyalty (SDL).

    BDR = party_defections / total_party_line_votes

    SDL uses Snyder-Groseclose (2000) methodology: partition roll calls
    into close votes (margin < 65-35) and lopsided votes (> 65-35).
    SDL = defection_rate_close / defection_rate_lopsided.
    SDL > 1 = principled dissenter. SDL < 1 = strategic brand-builder.

    Parameters
    ----------
    member_votes : pd.DataFrame
        Per-member per-vote records from VoteView.
        Columns: icpsr_id, rollnumber, cast_code.
    rollcall_metadata : pd.DataFrame
        Roll call metadata from VoteView.
        Columns: rollnumber, congress, chamber, yea_count, nay_count.

    Returns
    -------
    pd.DataFrame
        Columns: icpsr_id, congress, bdr, sdl,
        defections_close, defections_lopsided,
        total_close_votes, total_lopsided_votes.
    """
    raise NotImplementedError


def compute_responsiveness_index(
    nokken_poole_history: "pd.DataFrame",
    district_opinion_history: "pd.DataFrame",
) -> "pd.DataFrame":
    """Compute Responsiveness Index (RI).

    RI = correlation(delta_Nokken_Poole, delta_district_opinion)
    across Congresses for the same legislator.

    Parameters
    ----------
    nokken_poole_history : pd.DataFrame
        Columns: icpsr_id, congress, nokken_poole_dim1.
    district_opinion_history : pd.DataFrame
        Columns: district_id, congress, median_opinion (from CES MRP).

    Returns
    -------
    pd.DataFrame
        Columns: icpsr_id, ri, n_congresses, delta_nominate_trajectory,
        delta_opinion_trajectory.
    """
    raise NotImplementedError
