"""Aggregate DRA block-level election data to census tract level.

Dave's Redistricting provides election results at census block level (GEOID = 15 digits).
Census tract GEOID = first 11 digits of block GEOID. Simple groupby sum aggregation.

This replaces the areal interpolation pipeline for 2008-2024 election data.

Input:  data/raw/dra-block-data/{state}/v*/election_data_block_{state}.v*.csv
Output: data/tracts/tract_votes_dra.parquet (all states, all races, long format)

Usage:
    python -m src.tracts.aggregate_blocks_to_tracts
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DRA_DIR = PROJECT_ROOT / "data" / "raw" / "dra-block-data"
OUTPUT_DIR = PROJECT_ROOT / "data" / "tracts"

# Map DRA column prefixes to standardized race/year
# Format: E_{year}_{race}_{party} → (year, race)
RACE_COLUMNS = {
    "E_08_PRES": (2008, "president"),
    "E_12_PRES": (2012, "president"),
    "E_16_PRES": (2016, "president"),
    "E_20_PRES": (2020, "president"),
    "E_24_PRES": (2024, "president"),
    "E_16_SEN": (2016, "senate"),
    "E_18_SEN": (2018, "senate"),
    "E_22_SEN": (2022, "senate"),
    "E_24_SEN": (2024, "senate"),
    "E_18_GOV": (2018, "governor"),
    "E_22_GOV": (2022, "governor"),
    "E_18_AG": (2018, "ag"),
    "E_22_AG": (2022, "ag"),
    "E_22_CONG": (2022, "congress"),
}

STATE_FIPS = {
    "AL": "01", "FL": "12", "GA": "13",
    # National expansion: add all states here
    "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08", "CT": "09",
    "DE": "10", "DC": "11", "HI": "15", "ID": "16", "IL": "17", "IN": "18",
    "IA": "19", "KS": "20", "KY": "21", "LA": "22", "ME": "23", "MD": "24",
    "MA": "25", "MI": "26", "MN": "27", "MS": "28", "MO": "29", "MT": "30",
    "NE": "31", "NV": "32", "NH": "33", "NJ": "34", "NM": "35", "NY": "36",
    "NC": "37", "ND": "38", "OH": "39", "OK": "40", "OR": "41", "PA": "42",
    "RI": "44", "SC": "45", "SD": "46", "TN": "47", "TX": "48", "UT": "49",
    "VT": "50", "VA": "51", "WA": "53", "WV": "54", "WI": "55", "WY": "56",
}


def find_dra_file(state: str) -> Path | None:
    """Find the DRA block data CSV for a state."""
    state_dir = DRA_DIR / state
    if not state_dir.exists():
        return None
    csvs = list(state_dir.rglob("election_data_block_*.csv"))
    return csvs[0] if csvs else None


def aggregate_state(state: str) -> pd.DataFrame:
    """Aggregate block-level votes to tract level for one state.

    Returns DataFrame with columns:
        tract_geoid, state, year, race, votes_dem, votes_rep, votes_total, dem_share
    """
    csv_path = find_dra_file(state)
    if csv_path is None:
        log.warning("No DRA data for %s", state)
        return pd.DataFrame()

    df = pd.read_csv(csv_path, dtype={"GEOID": str})
    df["tract_geoid"] = df["GEOID"].str[:11]
    log.info("  %s: %d blocks → %d tracts", state, len(df), df["tract_geoid"].nunique())

    records = []
    available_cols = set(df.columns)

    for prefix, (year, race) in RACE_COLUMNS.items():
        dem_col = f"{prefix}_Dem"
        rep_col = f"{prefix}_Rep"
        total_col = f"{prefix}_Total"

        if dem_col not in available_cols or total_col not in available_cols:
            continue

        tract_agg = df.groupby("tract_geoid").agg(
            votes_dem=(dem_col, "sum"),
            votes_rep=(rep_col, "sum"),
            votes_total=(total_col, "sum"),
        ).reset_index()

        tract_agg["state"] = state
        tract_agg["year"] = year
        tract_agg["race"] = race
        tract_agg["dem_share"] = np.where(
            tract_agg["votes_total"] > 0,
            tract_agg["votes_dem"] / tract_agg["votes_total"],
            np.nan,
        )
        records.append(tract_agg)

    if not records:
        return pd.DataFrame()

    return pd.concat(records, ignore_index=True)


def aggregate_all(states: list[str] | None = None) -> pd.DataFrame:
    """Aggregate all available states to tract level."""
    if states is None:
        # Default: FL, GA, AL
        states = ["AL", "FL", "GA"]

    all_results = []
    for state in states:
        result = aggregate_state(state)
        if not result.empty:
            all_results.append(result)

    if not all_results:
        raise RuntimeError("No DRA data found for any state")

    combined = pd.concat(all_results, ignore_index=True)
    return combined


def main() -> None:
    """Aggregate DRA block data to tracts and save."""
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate DRA blocks to tracts")
    parser.add_argument(
        "--states",
        nargs="+",
        default=["AL", "FL", "GA"],
        help="States to process (default: AL FL GA)",
    )
    parser.add_argument(
        "--all-states",
        action="store_true",
        help="Process all available states",
    )
    args = parser.parse_args()

    states = list(STATE_FIPS.keys()) if args.all_states else args.states

    log.info("Aggregating DRA block data to tract level for %d states", len(states))
    result = aggregate_all(states)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "tract_votes_dra.parquet"
    result.to_parquet(output_path, index=False)

    # Summary
    log.info("Saved %d tract-race rows to %s", len(result), output_path)
    summary = result.groupby(["state", "race"]).agg(
        n_tracts=("tract_geoid", "nunique"),
        years=("year", lambda x: sorted(x.unique().tolist())),
    ).reset_index()
    print("\n=== Tract Vote Summary ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
