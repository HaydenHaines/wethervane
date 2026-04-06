"""
DRA Block-Level Election Data Ingestion Pipeline (Phase T.1)

Reads raw DRA block-level election data for all 51 state directories and
aggregates to census tracts by summing numeric columns within each
11-digit tract GEOID prefix.

Data layout:
  data/raw/dra-block-data/
    {STATE}/
      v06/  OR  v07/   (never both)
        election_data_block_{state}.v0N.csv

Output:
  data/assembled/tract_elections.parquet  — long format
    tract_geoid | year | race_type | total_votes | dem_votes | rep_votes | dem_share

Block GEOIDs are 15 digits; tract GEOIDs = GEOID[:11].
COMP (composite) columns are skipped — they are not real elections.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[2]
RAW_DRA_DIR = PROJECT_ROOT / "data" / "raw" / "dra-block-data"
OUTPUT_DIR = PROJECT_ROOT / "data" / "assembled"

# Pattern matches E_{YY}_{RACE}_{Metric} — rejecting COMP composites via negative
# lookahead on the race segment containing a hyphen (e.g. E_16-20_COMP_Total)
_ELECTION_COL_RE = re.compile(
    r"^E_(\d{2})_([A-Z_]+)_(Total|Dem|Rep)$"
)

# Races that contain a hyphen in their year token are always COMP columns:
# e.g. E_16-20_COMP_Total. We catch them by the hyphen in the full column name.
_COMP_COL_RE = re.compile(r"^E_\d{2}-\d{2}_COMP_")

# Mapping from 2-digit year to 4-digit year
def _parse_year(yy: str) -> int:
    """Convert 2-digit year string to 4-digit int (08→2008, 24→2024)."""
    y = int(yy)
    return 2000 + y


def _is_election_col(col: str) -> bool:
    """Return True if col is a real election column (not COMP, not GEOID)."""
    if _COMP_COL_RE.match(col):
        return False
    return bool(_ELECTION_COL_RE.match(col))


def _detect_version(state_dir: Path) -> str:
    """Return 'v06' or 'v07' for the state directory."""
    if (state_dir / "v06").is_dir():
        return "v06"
    if (state_dir / "v07").is_dir():
        return "v07"
    raise FileNotFoundError(f"No v06 or v07 subdirectory found in {state_dir}")


def _find_csv(version_dir: Path) -> Path:
    """Find the single election CSV inside a version directory."""
    csvs = list(version_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV found in {version_dir}")
    if len(csvs) > 1:
        log.warning("Multiple CSVs in %s; using first: %s", version_dir, csvs[0].name)
    return csvs[0]


def _ingest_state(state_dir: Path) -> pd.DataFrame | None:
    """
    Read one state's block CSV and aggregate to tract level.

    Returns a DataFrame indexed by tract_geoid with one column per election
    metric (e.g. E_08_PRES_Total).  Returns None if the directory should be
    skipped (e.g. 2020_Geography).
    """
    state = state_dir.name
    version = _detect_version(state_dir)
    csv_path = _find_csv(state_dir / version)

    log.info("Reading %s (%s) from %s", state, version, csv_path.name)

    df = pd.read_csv(csv_path, dtype={"GEOID": str}, low_memory=False)

    # Ensure GEOID is string-padded to 15 chars
    df["GEOID"] = df["GEOID"].str.zfill(15)

    # Extract tract GEOID (first 11 chars)
    df["tract_geoid"] = df["GEOID"].str[:11]

    # Identify election columns (drop COMP and non-matching)
    election_cols = [c for c in df.columns if _is_election_col(c)]

    if not election_cols:
        log.warning("No election columns found for %s — skipping", state)
        return None

    # Sum blocks → tracts
    tract_df = (
        df.groupby("tract_geoid")[election_cols]
        .sum(numeric_only=True)
        .reset_index()
    )

    tract_df["state"] = state
    return tract_df


def ingest_all_states(data_dir: str | Path | None = None) -> pd.DataFrame:
    """
    Main entry point — iterate all state directories and aggregate blocks to tracts.

    Parameters
    ----------
    data_dir:
        Path to the dra-block-data directory.  Defaults to
        ``data/raw/dra-block-data/`` relative to the project root.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with tract_geoid as index.  Columns are election
        metrics like ``E_08_PRES_Total``, plus a ``state`` column.
    """
    dra_dir = Path(data_dir) if data_dir is not None else RAW_DRA_DIR

    state_frames: list[pd.DataFrame] = []
    skipped = []

    for state_dir in sorted(dra_dir.iterdir()):
        if not state_dir.is_dir():
            continue
        if state_dir.name == "2020_Geography":
            log.debug("Skipping 2020_Geography")
            continue

        try:
            frame = _ingest_state(state_dir)
            if frame is not None:
                state_frames.append(frame)
        except FileNotFoundError as exc:
            log.error("Skipping %s: %s", state_dir.name, exc)
            skipped.append(state_dir.name)

    if not state_frames:
        raise RuntimeError("No state data loaded — check data_dir path")

    combined = pd.concat(state_frames, ignore_index=True)
    combined = combined.set_index("tract_geoid")

    log.info(
        "Loaded %d states (%d skipped), %d unique tracts, %d total rows",
        len(state_frames),
        len(skipped),
        combined.index.nunique(),
        len(combined),
    )
    return combined


def extract_election_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert wide block-aggregated DataFrame to long-format election results.

    Parameters
    ----------
    df:
        Wide DataFrame returned by ``ingest_all_states`` (tract_geoid as index).

    Returns
    -------
    pd.DataFrame
        Long format with columns:
        ``tract_geoid | year | race_type | total_votes | dem_votes | rep_votes | dem_share``
    """
    df_reset = df.reset_index()

    # Gather all (yy, race) pairs from columns
    parsed: dict[tuple[str, str], dict] = {}
    for col in df_reset.columns:
        m = _ELECTION_COL_RE.match(col)
        if m is None:
            continue
        yy, race, metric = m.group(1), m.group(2), m.group(3)
        key = (yy, race)
        parsed.setdefault(key, {})[metric] = col

    rows = []
    for (yy, race), metric_cols in parsed.items():
        total_col = metric_cols.get("Total")
        dem_col = metric_cols.get("Dem")
        rep_col = metric_cols.get("Rep")

        if not all([total_col, dem_col, rep_col]):
            log.debug("Skipping incomplete race %s_%s (missing metric column)", yy, race)
            continue

        sub = df_reset[["tract_geoid", total_col, dem_col, rep_col]].copy()
        sub.columns = ["tract_geoid", "total_votes", "dem_votes", "rep_votes"]
        sub["year"] = _parse_year(yy)
        sub["race_type"] = race
        rows.append(sub)

    if not rows:
        return pd.DataFrame(
            columns=["tract_geoid", "year", "race_type", "total_votes",
                     "dem_votes", "rep_votes", "dem_share"]
        )

    long_df = pd.concat(rows, ignore_index=True)

    # Filter zero-vote tracts
    long_df = long_df[long_df["total_votes"] > 0].copy()

    # Two-party dem share (dem / (dem + rep)); guard against zero denominator
    two_party = long_df["dem_votes"] + long_df["rep_votes"]
    long_df["dem_share"] = long_df["dem_votes"] / two_party.where(two_party > 0)

    # Reorder columns
    long_df = long_df[
        ["tract_geoid", "year", "race_type", "total_votes", "dem_votes", "rep_votes", "dem_share"]
    ]

    return long_df.reset_index(drop=True)


def save_tract_elections(
    output_path: str | Path | None = None,
    data_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Full pipeline: ingest → extract → save parquet.

    Parameters
    ----------
    output_path:
        Destination parquet file.  Defaults to
        ``data/assembled/tract_elections.parquet``.
    data_dir:
        Source directory for DRA block CSVs.

    Returns
    -------
    pd.DataFrame
        The long-format election results that were saved.
    """
    out = Path(output_path) if output_path is not None else OUTPUT_DIR / "tract_elections.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)

    wide = ingest_all_states(data_dir)
    long_df = extract_election_results(wide)

    long_df.to_parquet(out, index=False)

    n_tracts = long_df["tract_geoid"].nunique()
    n_states = wide["state"].nunique()
    n_elections = long_df.groupby(["year", "race_type"]).ngroups
    n_rows = len(long_df)

    log.info("Saved %s", out)
    log.info("  Unique tracts   : %d", n_tracts)
    log.info("  States          : %d", n_states)
    log.info("  Election races  : %d (year × race_type)", n_elections)
    log.info("  Total rows      : %d", n_rows)

    return long_df


if __name__ == "__main__":
    save_tract_elections()
