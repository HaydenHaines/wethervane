"""
Stage: Build national tract-level electoral shift vectors (Phase T.2).

This is the tract-primary replacement for build_shift_vectors.py (which covered
FL/GA/AL only). This module covers all 51 states and 83K+ tracts.

Shift dimensions produced:
  Presidential (raw, cross-state signal preserved):
    pres_shift_08_12, pres_shift_12_16, pres_shift_16_20, pres_shift_20_24

  Off-cycle (state-centered, candidate effects removed as proxy):
    gov_shift_{YY}_{YY}_centered  — one column per unique year-pair nationally
    sen_shift_{YY}_{YY}_centered  — one column per unique year-pair nationally

Design constraints:
  - Presidential shifts are raw: they carry cross-state structural signal.
  - Off-cycle shifts are state-centered: removes state-level candidate effects
    so cross-state clustering can find communities that move together, not just
    "this was a good D year in this state."
  - Population filter: tracts with <500 total votes in either election year are
    excluded (noise floor; ~3,300 tracts dropped per the design spec).
  - Off-cycle pairing: consecutive elections within 6 years, within each state.
    States with multiple valid pairs contribute one shift per pair.
  - Senate consolidation: SEN, SEN_SPEC, SEN_ROFF, SEN_SPECROFF all treated as
    "senate" for pairing. Within a state-year, the election with the most total
    votes is used (drops small special elections when a full election also ran).
  - NaN for missing races: a tract in a state without GOV 2018→2022 gets NaN
    for gov_shift_18_22_centered. Do not fill with 0 (0 is a real shift value).

MODELING NOTE: State-centering is a proxy for candidate effect removal. A proper
decomposition (district baseline + national environment + candidate draw) is future
work. When implemented, this centering step should be replaced. See design spec:
docs/superpowers/specs/2026-03-27-tract-primary-behavior-layer-design.md

Input:
  data/assembled/tract_elections.parquet

Output:
  data/shifts/tract_shifts_national.parquet
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parents[2]
ASSEMBLED_DIR = PROJECT_ROOT / "data" / "assembled"
SHIFTS_DIR = PROJECT_ROOT / "data" / "shifts"

INPUT_PATH = ASSEMBLED_DIR / "tract_elections.parquet"
OUTPUT_PATH = SHIFTS_DIR / "tract_shifts_national.parquet"

# Minimum total votes to include a tract in a shift pair.
# Tracts below this in either election year are excluded (noise floor).
MIN_VOTES = 500

# Presidential election years and shift pairs to compute.
PRES_YEARS = [2008, 2012, 2016, 2020, 2024]
PRES_PAIRS = [(2008, 2012), (2012, 2016), (2016, 2020), (2020, 2024)]

# Senate race types to consolidate into a single "senate" category.
SENATE_RACE_TYPES = frozenset(["SEN", "SEN_SPEC", "SEN_ROFF", "SEN_SPECROFF"])

# Maximum years between election pairs for off-cycle pairing.
MAX_PAIR_GAP_YEARS = 6


# ── Helper: state FIPS from tract GEOID ───────────────────────────────────────


def state_fips_from_tract(tract_geoid: pd.Series) -> pd.Series:
    """Extract 2-digit state FIPS code from 11-character tract GEOID."""
    return tract_geoid.str[:2]


# ── Helper: year-label string ─────────────────────────────────────────────────


def year_label(early: int, late: int) -> str:
    """Convert (2018, 2022) → '18_22'."""
    return f"{early % 100:02d}_{late % 100:02d}"


# ── Presidential shifts ────────────────────────────────────────────────────────


def compute_presidential_shifts(elections: pd.DataFrame) -> pd.DataFrame:
    """Compute 4 presidential shift dimensions from tract election data.

    Shift = later_dem_share - earlier_dem_share (raw, not state-centered).
    Presidential shifts are cross-state structural signal — centering would
    remove the very variation we want to cluster on.

    Tracts with < MIN_VOTES in either election year of a pair are excluded
    from that pair's shift (set to NaN for that pair's column).

    Parameters
    ----------
    elections:
        Long-format tract election DataFrame with columns:
        tract_geoid, year, race_type, total_votes, dem_share.

    Returns
    -------
    DataFrame indexed by tract_geoid with 4 presidential shift columns:
    pres_shift_08_12, pres_shift_12_16, pres_shift_16_20, pres_shift_20_24.
    """
    pres = elections[elections["race_type"] == "PRES"].copy()

    # Build a wide-format frame: one row per tract, columns for each year.
    # Each tract may appear in multiple presidential years.
    pres_wide = pres.pivot_table(
        index="tract_geoid",
        columns="year",
        values=["dem_share", "total_votes"],
        aggfunc="first",  # Each tract should have at most one PRES row per year
    )
    # Flatten MultiIndex columns: ('dem_share', 2008) → 'dem_share_2008'
    pres_wide.columns = [f"{col[0]}_{col[1]}" for col in pres_wide.columns]

    result = pd.DataFrame(index=pres_wide.index)

    for early_year, late_year in PRES_PAIRS:
        col_label = year_label(early_year, late_year)
        col_name = f"pres_shift_{col_label}"

        early_share = f"dem_share_{early_year}"
        late_share = f"dem_share_{late_year}"
        early_votes = f"total_votes_{early_year}"
        late_votes = f"total_votes_{late_year}"

        # Both years must be present for a tract to get a shift value.
        if not all(
            col in pres_wide.columns
            for col in [early_share, late_share, early_votes, late_votes]
        ):
            log.warning(
                "compute_presidential_shifts: missing columns for %s→%s; skipping",
                early_year,
                late_year,
            )
            result[col_name] = np.nan
            continue

        # Population filter: exclude tracts with < MIN_VOTES in either year.
        sufficient_votes = (pres_wide[early_votes] >= MIN_VOTES) & (
            pres_wide[late_votes] >= MIN_VOTES
        )

        shift = pres_wide[late_share] - pres_wide[early_share]
        result[col_name] = np.where(sufficient_votes, shift, np.nan)

        n_valid = sufficient_votes.sum()
        n_total = (~(pres_wide[early_share].isna() | pres_wide[late_share].isna())).sum()
        log.info(
            "pres_shift_%s: %d tracts with sufficient votes (of %d with both years)",
            col_label,
            n_valid,
            n_total,
        )

    result.index.name = "tract_geoid"
    return result.reset_index()


# ── Off-cycle shift helpers ────────────────────────────────────────────────────


def _consolidate_senate(elections: pd.DataFrame) -> pd.DataFrame:
    """Relabel all Senate race types to 'SEN' for pairing purposes.

    Within each (state, year), keep the election with the most total votes
    across all tracts — this drops small special elections when a full regular
    election also ran in the same year.

    Parameters
    ----------
    elections : DataFrame with tract_geoid, year, race_type, total_votes, dem_share.

    Returns
    -------
    Same schema as input, with race_type replaced by 'SEN' for all senate types,
    and at most one row per (tract_geoid, year) for senate elections.
    """
    is_senate = elections["race_type"].isin(SENATE_RACE_TYPES)
    senate = elections[is_senate].copy()
    senate["race_type"] = "SEN"
    senate["state_fips"] = state_fips_from_tract(senate["tract_geoid"])

    # The race_type column was already relabeled to 'SEN', so we need the
    # original to disambiguate. Store it before relabeling.
    senate_orig = elections[is_senate].copy()
    senate_orig["state_fips"] = state_fips_from_tract(senate_orig["tract_geoid"])

    state_year_orig_votes = (
        senate_orig.groupby(["state_fips", "year", "race_type"])["total_votes"]
        .sum()
        .reset_index()
        .rename(columns={"total_votes": "state_total"})
    )
    # Per state-year, find the original race_type with the most votes.
    state_year_orig_votes = state_year_orig_votes.sort_values(
        "state_total", ascending=False
    )
    best_type = state_year_orig_votes.drop_duplicates(
        subset=["state_fips", "year"], keep="first"
    )[["state_fips", "year", "race_type"]].rename(
        columns={"race_type": "best_race_type"}
    )

    # Keep only tracts from the best race_type per state-year.
    senate_filtered = senate_orig.merge(best_type, on=["state_fips", "year"])
    senate_filtered = senate_filtered[
        senate_filtered["race_type"] == senate_filtered["best_race_type"]
    ].copy()
    senate_filtered["race_type"] = "SEN"
    senate_filtered = senate_filtered.drop(
        columns=["state_fips", "best_race_type"]
    )

    return senate_filtered


def _find_consecutive_pairs(years: list[int]) -> list[tuple[int, int]]:
    """Return all consecutive pairs in a sorted year list with gap <= MAX_PAIR_GAP_YEARS."""
    years = sorted(years)
    return [
        (years[i], years[i + 1])
        for i in range(len(years) - 1)
        if years[i + 1] - years[i] <= MAX_PAIR_GAP_YEARS
    ]


def _compute_single_offcycle_pair(
    elections_subset: pd.DataFrame,
    early_year: int,
    late_year: int,
    state_fips: str,
    col_name: str,
) -> pd.DataFrame:
    """Compute state-centered shift for a single off-cycle election pair.

    Parameters
    ----------
    elections_subset:
        Election rows filtered to the correct race_type.
    early_year, late_year:
        The two election years to diff.
    state_fips:
        The 2-digit state FIPS code (used for state-centering assertion only).
    col_name:
        Output column name.

    Returns
    -------
    DataFrame with (tract_geoid, col_name) for tracts with sufficient votes.
    """
    early = elections_subset[elections_subset["year"] == early_year][
        ["tract_geoid", "dem_share", "total_votes"]
    ].rename(columns={"dem_share": "early_share", "total_votes": "early_votes"})
    late = elections_subset[elections_subset["year"] == late_year][
        ["tract_geoid", "dem_share", "total_votes"]
    ].rename(columns={"dem_share": "late_share", "total_votes": "late_votes"})

    merged = early.merge(late, on="tract_geoid", how="inner")

    # Population filter
    sufficient = (merged["early_votes"] >= MIN_VOTES) & (merged["late_votes"] >= MIN_VOTES)
    merged = merged[sufficient].copy()

    if len(merged) == 0:
        return pd.DataFrame(columns=["tract_geoid", col_name])

    merged[col_name] = merged["late_share"] - merged["early_share"]
    return merged[["tract_geoid", col_name]]


# ── Off-cycle shifts ───────────────────────────────────────────────────────────


def compute_offcycle_shifts(elections: pd.DataFrame) -> pd.DataFrame:
    """Compute state-centered off-cycle shift dimensions from tract election data.

    Processes governor and senate elections separately. For each state:
    1. Find all valid consecutive pairs (gap <= MAX_PAIR_GAP_YEARS).
    2. Compute dem_share shift for each pair.
    3. State-center the shift: subtract the state mean so cross-state clustering
       finds within-state variation, not candidate-quality effects.

    Output columns follow the pattern:
      gov_shift_{YY}_{YY}_centered  (e.g. gov_shift_18_22_centered)
      sen_shift_{YY}_{YY}_centered  (e.g. sen_shift_16_22_centered)

    A national column is created for each unique year-pair observed across any
    state. Tracts in states without that pair receive NaN.

    Parameters
    ----------
    elections : Long-format tract election DataFrame.

    Returns
    -------
    DataFrame with tract_geoid and one column per unique off-cycle shift pair.
    """
    # Consolidate senate types to 'SEN'.
    senate_rows = _consolidate_senate(elections)
    gov_rows = elections[elections["race_type"] == "GOV"].copy()

    # We'll collect per-(race_category, state, pair) shift frames, then
    # state-center each, then join them into a wide national frame.
    shift_frames: list[pd.DataFrame] = []

    for race_label, race_df in [("gov", gov_rows), ("sen", senate_rows)]:
        race_df = race_df.copy()
        race_df["state_fips"] = state_fips_from_tract(race_df["tract_geoid"])

        # Map of (pair) → list of per-state shift DataFrames (for state-centering).
        # We must collect all states for a given pair before we can center globally.
        pair_state_shifts: dict[tuple[int, int], list[pd.DataFrame]] = {}

        for state_fips, state_rows in race_df.groupby("state_fips"):
            years_in_state = sorted(state_rows["year"].unique().tolist())
            pairs = _find_consecutive_pairs(years_in_state)

            for early_year, late_year in pairs:
                pair_key = (early_year, late_year)
                lbl = year_label(early_year, late_year)
                col_name = f"{race_label}_shift_{lbl}_centered"

                pair_shifts = _compute_single_offcycle_pair(
                    state_rows, early_year, late_year, state_fips, col_name
                )

                if len(pair_shifts) == 0:
                    continue

                pair_shifts["_state_fips"] = state_fips

                if pair_key not in pair_state_shifts:
                    pair_state_shifts[pair_key] = []
                pair_state_shifts[pair_key].append(pair_shifts)

        # State-center each pair's shifts, then concatenate into a national series.
        for pair_key, state_dfs in pair_state_shifts.items():
            early_year, late_year = pair_key
            lbl = year_label(early_year, late_year)
            col_name = f"{race_label}_shift_{lbl}_centered"

            combined = pd.concat(state_dfs, ignore_index=True)

            # State-center: subtract each state's mean shift from its tract shifts.
            # This removes the state-level candidate/structural effect so that
            # cross-state clustering finds within-state variation (who moves more
            # vs less relative to their neighbors), not state-wide swings.
            state_means = combined.groupby("_state_fips")[col_name].transform("mean")
            combined[col_name] = combined[col_name] - state_means
            combined = combined.drop(columns=["_state_fips"])

            n_states = len(state_dfs)
            n_tracts = len(combined)
            log.info(
                "%s: %d tracts across %d states (state-centered)",
                col_name,
                n_tracts,
                n_states,
            )
            shift_frames.append(combined)

    if not shift_frames:
        log.warning("compute_offcycle_shifts: no valid shift pairs found")
        return pd.DataFrame(columns=["tract_geoid"])

    # Join all shift columns onto the full tract spine.
    # Start with all unique tract_geoids observed in any off-cycle election.
    offcycle_types = frozenset(["GOV"]) | SENATE_RACE_TYPES
    all_tracts = elections[
        elections["race_type"].isin(offcycle_types)
    ]["tract_geoid"].drop_duplicates()
    spine = all_tracts.to_frame()

    for frame in shift_frames:
        col_name = [c for c in frame.columns if c != "tract_geoid"][0]
        spine = spine.merge(frame[["tract_geoid", col_name]], on="tract_geoid", how="left")

    return spine


# ── Main orchestrator ─────────────────────────────────────────────────────────


def build_tract_shift_matrix(
    elections: pd.DataFrame,
    output_path: str | Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """Assemble presidential and off-cycle shift vectors into a national matrix.

    Parameters
    ----------
    elections:
        Long-format tract election DataFrame from data/assembled/tract_elections.parquet.
    output_path:
        Where to write the output parquet.

    Returns
    -------
    Wide DataFrame indexed by tract_geoid with:
      - 4 presidential shift columns (raw, not state-centered)
      - Variable number of off-cycle shift columns (state-centered, NaN where no data)
    """
    log.info(
        "build_tract_shift_matrix: %d rows, %d tracts, %d states",
        len(elections),
        elections["tract_geoid"].nunique(),
        elections["tract_geoid"].str[:2].nunique(),
    )

    # ── Presidential shifts ────────────────────────────────────────────────────
    log.info("Computing presidential shifts...")
    pres_shifts = compute_presidential_shifts(elections)
    log.info(
        "Presidential shifts: %d tracts, %d columns",
        len(pres_shifts),
        len([c for c in pres_shifts.columns if c != "tract_geoid"]),
    )

    # ── Off-cycle shifts ───────────────────────────────────────────────────────
    log.info("Computing off-cycle shifts...")
    offcycle_shifts = compute_offcycle_shifts(elections)
    offcycle_cols = [c for c in offcycle_shifts.columns if c != "tract_geoid"]
    log.info(
        "Off-cycle shifts: %d tracts, %d columns",
        len(offcycle_shifts),
        len(offcycle_cols),
    )

    # ── Merge on presidential spine ────────────────────────────────────────────
    # Use the presidential shifts as the spine — all tracts with any presidential
    # data anchor the matrix. Off-cycle shifts join on tract_geoid; NaN where missing.
    result = pres_shifts.merge(offcycle_shifts, on="tract_geoid", how="outer")
    result = result.set_index("tract_geoid")

    # ── Summary logging ────────────────────────────────────────────────────────
    total_tracts = len(result)
    total_cols = len(result.columns)
    pres_cols = [c for c in result.columns if c.startswith("pres_shift_")]
    oc_cols = [c for c in result.columns if not c.startswith("pres_shift_")]

    log.info("=" * 60)
    log.info("SHIFT MATRIX SUMMARY")
    log.info("  Total tracts: %d", total_tracts)
    log.info("  Total columns: %d", total_cols)
    log.info("  Presidential columns: %s", pres_cols)
    log.info("  Off-cycle columns (%d):", len(oc_cols))
    for col in sorted(oc_cols):
        n_valid = result[col].notna().sum()
        nan_pct = result[col].isna().mean() * 100
        log.info("    %-45s  %6d valid  (%5.1f%% NaN)", col, n_valid, nan_pct)
    log.info("=" * 60)

    # ── Save ───────────────────────────────────────────────────────────────────
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.reset_index().to_parquet(output_path, index=False)
        log.info("Saved → %s", output_path)

    return result


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    """Load tract election data, compute shifts, write output."""
    log.info("Loading %s...", INPUT_PATH)
    elections = pd.read_parquet(INPUT_PATH)
    log.info("Loaded: %d rows, %d tracts", len(elections), elections["tract_geoid"].nunique())

    result = build_tract_shift_matrix(elections, OUTPUT_PATH)

    log.info(
        "Done. %d tracts × %d shift columns → %s",
        len(result),
        len(result.columns),
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
