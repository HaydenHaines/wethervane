"""Data assembly pipeline.

Fetches, cleans, and joins county-level data from multiple sources into a
unified dataset for FL+GA+AL.

Sources assembled:
    - Election returns (MEDSL county presidential + downballot)
    - Demographics (ACS 5-year estimates via Census API)
    - Religion (RCMS / U.S. Religion Census via ARDA)
    - Migration (IRS county-to-county flows)
    - Commuting (LEHD LODES origin-destination)
    - Social networks (Facebook SCI county-to-county)
    - Economic (BLS QCEW industry composition)

Output: A set of Parquet files with consistent FIPS indexing:
    - county_features.parquet  (N_counties x P feature matrix)
    - county_elections.parquet  (N_counties x T_elections result matrix)
    - county_edges.parquet     (edge list with source, target, weight, layer)

See docs/DATA_SOURCES.md for source evaluation and priority.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd


def fetch_election_returns(states: list[str], years: list[int]) -> "pd.DataFrame":
    """Download and standardize MEDSL county-level election returns.

    Parameters
    ----------
    states : list[str]
        Two-letter state abbreviations (e.g., ["FL", "GA", "AL"]).
    years : list[int]
        Election years to retrieve (e.g., [2008, 2012, 2016, 2020]).

    Returns
    -------
    pd.DataFrame
        Columns: fips, year, office, party, votes, total_votes, two_party_share.
    """
    raise NotImplementedError


def fetch_acs_demographics(states: list[str], year: int, tables: list[str] | None = None) -> "pd.DataFrame":
    """Download ACS 5-year estimates for specified tables and states.

    Parameters
    ----------
    states : list[str]
        Two-letter state abbreviations.
    year : int
        End year of the 5-year ACS window.
    tables : list[str] | None
        ACS table IDs (e.g., ["B01001", "B03002"]). If None, use default set.

    Returns
    -------
    pd.DataFrame
        County-level demographic features with FIPS index.
    """
    raise NotImplementedError


def fetch_rcms_religion(year: int = 2020) -> "pd.DataFrame":
    """Download RCMS / U.S. Religion Census denominational adherence rates.

    Parameters
    ----------
    year : int
        Census year (2000, 2010, or 2020).

    Returns
    -------
    pd.DataFrame
        County-level adherence rates per denomination, FIPS-indexed.
    """
    raise NotImplementedError


def fetch_irs_migration(states: list[str], years: list[int]) -> "pd.DataFrame":
    """Download IRS county-to-county migration flow data.

    Parameters
    ----------
    states : list[str]
        Two-letter state abbreviations.
    years : list[int]
        Filing years to retrieve.

    Returns
    -------
    pd.DataFrame
        Edge list: origin_fips, dest_fips, n_returns, agi.
    """
    raise NotImplementedError


def fetch_lodes_commuting(states: list[str], year: int = 2020) -> "pd.DataFrame":
    """Download LEHD LODES origin-destination employment data.

    Parameters
    ----------
    states : list[str]
        Two-letter state abbreviations (lowercase for LODES API).
    year : int
        Reference year.

    Returns
    -------
    pd.DataFrame
        Edge list: home_county_fips, work_county_fips, n_workers.
    """
    raise NotImplementedError


def fetch_sci(states: list[str] | None = None) -> "pd.DataFrame":
    """Download Facebook Social Connectedness Index county pairs.

    Parameters
    ----------
    states : list[str] | None
        If provided, filter to pairs where at least one county is in
        the specified states. If None, return all pairs.

    Returns
    -------
    pd.DataFrame
        Edge list: fips_1, fips_2, sci_weight.
    """
    raise NotImplementedError


def fetch_qcew_industry(states: list[str], year: int) -> "pd.DataFrame":
    """Download BLS QCEW industry composition by county.

    Parameters
    ----------
    states : list[str]
        Two-letter state abbreviations.
    year : int
        Reference year.

    Returns
    -------
    pd.DataFrame
        County x NAICS-sector employment shares, FIPS-indexed.
    """
    raise NotImplementedError


def build_feature_matrix(
    demographics: "pd.DataFrame",
    religion: "pd.DataFrame",
    industry: "pd.DataFrame",
) -> "pd.DataFrame":
    """Join and normalize demographic, religious, and economic features.

    Applies log transforms, standardization, and imputation as needed.

    Parameters
    ----------
    demographics : pd.DataFrame
        ACS demographic features.
    religion : pd.DataFrame
        RCMS denominational adherence rates.
    industry : pd.DataFrame
        QCEW industry employment shares.

    Returns
    -------
    pd.DataFrame
        Unified county x feature matrix, FIPS-indexed, standardized.
    """
    raise NotImplementedError


def build_network_layers(
    commuting: "pd.DataFrame",
    migration: "pd.DataFrame",
    sci: "pd.DataFrame",
) -> "pd.DataFrame":
    """Combine commuting, migration, and SCI into a multi-layer edge list.

    Parameters
    ----------
    commuting : pd.DataFrame
        LODES commuting flows.
    migration : pd.DataFrame
        IRS migration flows.
    sci : pd.DataFrame
        Facebook SCI weights.

    Returns
    -------
    pd.DataFrame
        Unified edge list: source_fips, target_fips, weight, layer.
    """
    raise NotImplementedError


def save_assembled_data(
    features: "pd.DataFrame",
    elections: "pd.DataFrame",
    edges: "pd.DataFrame",
    output_dir: "Path",
) -> None:
    """Write assembled datasets to Parquet files.

    Parameters
    ----------
    features : pd.DataFrame
        County feature matrix.
    elections : pd.DataFrame
        County election returns.
    edges : pd.DataFrame
        Multi-layer network edge list.
    output_dir : Path
        Directory to write output files.
    """
    raise NotImplementedError
