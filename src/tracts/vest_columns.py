"""Map VEST shapefile column names to standardized vote columns.

VEST (Voting and Election Science Team) shapefiles use a cryptic column naming
convention:

    G{YY}{RACE}{PARTY}{CANDIDATE}

Where:
    YY   = two-digit year (16, 18, 20)
    RACE = office code (PRE=president, GOV=governor, USS=US Senate, etc.)
    PARTY = single letter (R=Republican, D=Democrat, L=Libertarian, G=Green,
            O=Other/write-in, I=Independent, C=Constitution, S=third party)
    CANDIDATE = abbreviated last name (3+ chars, e.g., TRU=Trump, CLI=Clinton)

Examples:
    G16PRERTRU  = 2016 president Republican Trump
    G16PREDCLI  = 2016 president Democrat Clinton
    G18GOVRDES  = 2018 governor Republican DeSantis
    G20PREDBID  = 2020 president Democrat Biden

Special cases:
    - GA 2020 uses non-standard prefixes for some races (C20, S20, R21)
    - AL includes GEOID{YY} columns that start with 'G' but are not vote columns
    - Write-in columns end in 'OWRI' (or 'OWR2', 'OWR3', etc.)

Reference: https://github.com/vest-org/vest-data
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import geopandas as gpd
import pandas as pd

log = logging.getLogger(__name__)

# ── Race code mapping ─────────────────────────────────────────────────────────

RACE_CODES = {
    "PRE": "president",
    "GOV": "governor",
    "USS": "us_senate",
    "ATG": "attorney_general",
    "AGR": "agriculture",
    "SOS": "secretary_of_state",
    "TRE": "treasurer",
    "AUD": "auditor",
    "INS": "insurance",
    "CFO": "cfo",
    "LTG": "lt_governor",
    "LAB": "labor",
    "SPI": "superintendent",
    "PSC": "public_service_commission",
    "SSC": "state_supreme_court",
    "SAC": "state_appeals_court",
    "SCC": "state_circuit_court",
}

# Party code → standardized party name
PARTY_CODES = {
    "R": "rep",
    "D": "dem",
    "L": "lib",
    "G": "grn",
    "O": "oth",
    "I": "ind",
    "C": "con",
    "S": "oth",
}

# ── Column pattern ────────────────────────────────────────────────────────────

# Standard VEST: G{YY}{RACE}{PARTY}{CANDIDATE}
# Race code is 3 chars, party is 1 char, candidate is 3+ chars
VEST_PATTERN = re.compile(
    r"^G(\d{2})([A-Z]{3})([RDLGOICS])([A-Z]{3,})$"
)

# GA 2020 uses non-standard prefixes for some races:
#   C20 = county-certified (nearly identical to G20, ~0.01% difference)
#   S20 = special election (e.g., S20USS = special US Senate)
#   R21 = runoff election (e.g., R21USS = January 2021 Senate runoff)
# These are intentionally NOT matched by the standard pattern to avoid
# double-counting. If a caller needs runoff/special data specifically,
# they should use find_vote_columns() with explicit column selection.
GA_ALT_PATTERN = re.compile(
    r"^([CSR])(\d{2})([A-Z]{3})([RDLGOICS])([A-Z]{3,})$"
)


# ── Precinct ID detection ─────────────────────────────────────────────────────

# State-specific precinct ID column names (discovered from actual VEST data)
_PRECINCT_ID_COLS = {
    # FL: COUNTY + PRECINCT or PCT_STD
    "FL": ["COUNTY", "PRECINCT", "PCT_STD"],
    # GA: PRECINCT_I (precinct ID), PRECINCT_N (precinct name), COUNTY
    "GA": ["PRECINCT_I", "PRECINCT_N", "COUNTY", "CTYSOSID"],
    # AL: GEOID{YY} (full VTD GEOID), VTDST{YY}, NAME{YY}
    "AL": ["GEOID16", "GEOID18", "GEOID20", "VTDST16", "VTDST18", "VTDST20",
           "NAME16", "NAME18", "NAME20"],
}


def _find_precinct_id(gdf: gpd.GeoDataFrame, state: str) -> str:
    """Find the best precinct ID column for the given state."""
    state_upper = state.upper()

    # Try state-specific columns first
    if state_upper in _PRECINCT_ID_COLS:
        for col in _PRECINCT_ID_COLS[state_upper]:
            if col in gdf.columns:
                return col

    # Fallback: look for common ID column patterns
    for col in gdf.columns:
        col_upper = col.upper()
        if col_upper.startswith("GEOID") or col_upper == "PRECINCT_I":
            return col

    # Last resort: use index
    return ""


# ── Core extraction ───────────────────────────────────────────────────────────


def find_vote_columns(
    gdf: gpd.GeoDataFrame,
    year: int,
    race: str = "president",
    include_alt_prefixes: bool = False,
) -> dict[str, list[str]]:
    """Find VEST vote columns matching a given year and race.

    Parameters
    ----------
    gdf : GeoDataFrame
        VEST shapefile loaded with geopandas.
    year : int
        Election year (e.g., 2016, 2018, 2020).
    race : str
        Race name (e.g., "president", "governor", "us_senate").
    include_alt_prefixes : bool
        If True, also match GA non-standard prefixes (C20, S20, R21).
        Default False to avoid double-counting with standard G-columns.

    Returns
    -------
    dict mapping party code ("dem", "rep", "lib", etc.) to list of matching
    column names. Multiple columns per party can exist (e.g., write-ins counted
    separately).
    """
    yy = str(year)[-2:]

    # Reverse lookup: race name -> race code(s)
    target_race_codes = [
        code for code, name in RACE_CODES.items() if name == race
    ]
    if not target_race_codes:
        raise ValueError(
            f"Unknown race '{race}'. Known races: "
            f"{sorted(set(RACE_CODES.values()))}"
        )

    result: dict[str, list[str]] = {}

    for col in gdf.columns:
        # Skip non-vote columns
        m = VEST_PATTERN.match(col)
        if m:
            col_yy, col_race, col_party, _candidate = m.groups()
        elif include_alt_prefixes:
            # Try GA alternate pattern (C20/S20/R21)
            m_alt = GA_ALT_PATTERN.match(col)
            if m_alt:
                _prefix, col_yy, col_race, col_party, _candidate = m_alt.groups()
            else:
                continue
        else:
            continue

        if col_yy != yy:
            continue
        if col_race not in target_race_codes:
            continue

        party = PARTY_CODES.get(col_party, "oth")
        result.setdefault(party, []).append(col)

    return result


def extract_vest_votes(
    gdf: gpd.GeoDataFrame,
    state: str,
    year: int,
    race: str = "president",
) -> gpd.GeoDataFrame:
    """Extract standardized vote columns from VEST shapefile.

    Parameters
    ----------
    gdf : GeoDataFrame
        VEST shapefile loaded with geopandas.
    state : str
        State abbreviation (e.g., "FL", "GA", "AL").
    year : int
        Election year (e.g., 2016, 2018, 2020).
    race : str
        Race name (e.g., "president", "governor"). Default: "president".

    Returns
    -------
    GeoDataFrame with columns:
        geometry    -- precinct polygon
        votes_dem   -- total Democratic votes (summed across all D columns)
        votes_rep   -- total Republican votes (summed across all R columns)
        votes_total -- total votes across all parties
        precinct_id -- best available precinct identifier
    """
    party_cols = find_vote_columns(gdf, year, race)

    if not party_cols:
        raise ValueError(
            f"No VEST columns found for {state} {year} {race}. "
            f"Available G-columns: {[c for c in gdf.columns if c.startswith('G')]}"
        )

    # Sum Democratic votes
    dem_cols = party_cols.get("dem", [])
    if dem_cols:
        votes_dem = gdf[dem_cols].sum(axis=1)
    else:
        log.warning("No Democratic columns found for %s %d %s", state, year, race)
        votes_dem = pd.Series(0, index=gdf.index)

    # Sum Republican votes
    rep_cols = party_cols.get("rep", [])
    if rep_cols:
        votes_rep = gdf[rep_cols].sum(axis=1)
    else:
        log.warning("No Republican columns found for %s %d %s", state, year, race)
        votes_rep = pd.Series(0, index=gdf.index)

    # Sum all votes across all parties
    all_vote_cols = [col for cols in party_cols.values() for col in cols]
    votes_total = gdf[all_vote_cols].sum(axis=1)

    # Find precinct ID
    id_col = _find_precinct_id(gdf, state)
    if id_col:
        precinct_id = gdf[id_col].astype(str)
    else:
        precinct_id = pd.Series(gdf.index.astype(str), index=gdf.index)

    log.info(
        "%s %d %s: D=[%s] R=[%s] total_cols=%d precincts=%d",
        state,
        year,
        race,
        ", ".join(dem_cols),
        ", ".join(rep_cols),
        len(all_vote_cols),
        len(gdf),
    )

    return gpd.GeoDataFrame(
        {
            "precinct_id": precinct_id,
            "votes_dem": votes_dem,
            "votes_rep": votes_rep,
            "votes_total": votes_total,
        },
        geometry=gdf.geometry,
        crs=gdf.crs,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    """Quick smoke test: extract votes from all available VEST files."""
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    project_root = Path(__file__).parents[2]
    vest_dir = project_root / "data" / "raw" / "vest"

    configs = [
        ("FL", 2016, "president"),
        ("GA", 2016, "president"),
        ("AL", 2016, "president"),
        ("FL", 2018, "governor"),
        ("GA", 2018, "governor"),
        ("AL", 2018, "governor"),
        ("FL", 2020, "president"),
        ("GA", 2020, "president"),
        ("AL", 2020, "president"),
    ]

    for state, year, race in configs:
        fname = f"{state.lower()}_{year}.zip"
        fpath = vest_dir / fname
        if not fpath.exists():
            log.warning("Skipping %s (not found)", fpath)
            continue

        gdf = gpd.read_file(fpath)

        # Show discovered columns
        party_cols = find_vote_columns(gdf, year, race)
        print(f"\n{state} {year} {race}:")
        for party, cols in sorted(party_cols.items()):
            print(f"  {party}: {cols}")

        # Extract standardized votes
        result = extract_vest_votes(gdf, state, year, race)
        print(
            f"  => {len(result)} precincts, "
            f"D={result['votes_dem'].sum():,.0f}, "
            f"R={result['votes_rep'].sum():,.0f}, "
            f"total={result['votes_total'].sum():,.0f}"
        )


if __name__ == "__main__":
    main()
