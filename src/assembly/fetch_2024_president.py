"""
Fetch 2024 presidential election results at county level for FL, GA, and AL.

Source: MIT Election Data + Science Lab (MEDSL) 2024-elections-official GitHub repo.
  https://github.com/MEDSL/2024-elections-official/tree/main/individual_states

Data is precinct-level CSV with county_fips attached — we aggregate to county.
Two-party only: DEMOCRAT and REPUBLICAN votes are kept; third-party candidates
(e.g. RFK Jr.) are excluded via party_simplified filter.

Filter logic:
  - office contains "PRESIDENT" but NOT "VICE PRESIDENT"
  - stage == "gen"
  - writein != TRUE
  - party_simplified in {"DEMOCRAT", "REPUBLICAN"}

Mode dedup: MEDSL sometimes includes both mode-specific rows (ABSENTEE, ELECTION DAY)
and a TOTAL row. We use TOTAL rows only if present; otherwise sum all modes.

Output:
  data/assembled/medsl_county_2024_president.parquet
  Columns: county_fips, state_abbr, pres_dem_2024, pres_rep_2024,
           pres_total_2024, pres_dem_share_2024
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "medsl" / "2024"
OUTPUT_DIR = PROJECT_ROOT / "data" / "assembled"

# MEDSL 2024-elections-official GitHub raw download URLs
MEDSL_BASE = (
    "https://github.com/MEDSL/2024-elections-official/raw/main/individual_states"
)

STATES = {
    "FL": ("fl24.zip", "12"),
    "GA": ("ga24.zip", "13"),
    "AL": ("al24.zip", "01"),
}

STATE_FIPS = {"01": "AL", "12": "FL", "13": "GA"}


def download_file(url: str, dest: Path, desc: str) -> None:
    """Download with progress bar. No-ops if cached."""
    if dest.exists():
        log.info("  Cached: %s", dest.name)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("  Downloading %s ...", desc)
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as fh, tqdm(total=total, unit="B", unit_scale=True, desc=desc) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            fh.write(chunk)
            bar.update(len(chunk))


def load_medsl_csv(zip_path: Path) -> pd.DataFrame:
    """Unzip and read the MEDSL precinct CSV from the zip file.

    2024 repo uses files without .csv extension (e.g. 'fl24' not 'fl24.csv').
    Fall back to first non-directory entry if no .csv file found.
    """
    with zipfile.ZipFile(zip_path) as zf:
        csv_files = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_files:
            # 2024 format: single entry with no extension
            csv_files = [n for n in zf.namelist() if not zf.getinfo(n).is_dir()]
        if not csv_files:
            raise FileNotFoundError(f"No data file in {zip_path}")
        log.info("  Reading %s from %s", csv_files[0], zip_path.name)
        with zf.open(csv_files[0]) as f:
            df = pd.read_csv(f, low_memory=False)
    return df


def extract_president_county(df: pd.DataFrame, state_abbr: str) -> pd.DataFrame:
    """
    Filter to presidential general election, dedup modes, aggregate to county level.

    Excludes VICE PRESIDENT rows (which may match a naive "PRESIDENT" substring check).
    Keeps only DEMOCRAT and REPUBLICAN party_simplified — excludes RFK Jr. and other
    third-party candidates so the output is a clean two-party share.

    Returns DataFrame with county_fips, state_abbr, and vote columns.
    """
    # Filter to presidential general (exclude VICE PRESIDENT)
    pres = df[
        (df["office"].str.upper().str.contains("PRESIDENT", na=False)) &
        (~df["office"].str.upper().str.contains("VICE", na=False)) &
        (df["stage"].str.lower() == "gen") &
        (df["writein"].fillna(False).astype(str).str.upper() != "TRUE")
    ].copy()

    if len(pres) == 0:
        log.warning("  No PRESIDENT/gen rows found for %s", state_abbr)
        return pd.DataFrame()

    log.info("  %s: %d presidential precinct rows before mode dedup", state_abbr, len(pres))

    # Mode dedup: prefer TOTAL rows if they exist
    if "mode" in pres.columns:
        modes = pres["mode"].str.upper().unique()
        log.info("  Modes present: %s", list(modes))
        if "TOTAL" in modes:
            pres = pres[pres["mode"].str.upper() == "TOTAL"]
            log.info("  Using TOTAL mode rows only (%d rows)", len(pres))
        else:
            log.info("  No TOTAL mode — summing all modes")

    # Two-party filter: keep only DEMOCRAT and REPUBLICAN
    pres = pres[
        pres["party_simplified"].str.upper().isin(["DEMOCRAT", "REPUBLICAN"])
    ].copy()

    if len(pres) == 0:
        log.warning("  No DEMOCRAT/REPUBLICAN rows after party filter for %s", state_abbr)
        return pd.DataFrame()

    # Ensure county_fips is zero-padded 5-digit string
    # MEDSL sometimes reads county_fips as float (e.g. 12001.0) — strip the .0 first
    pres["county_fips"] = (
        pres["county_fips"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(5)
    )

    # Aggregate votes by county + party
    county_party = (
        pres.groupby(["county_fips", "party_simplified"])["votes"]
        .sum()
        .reset_index()
    )

    dem = county_party[county_party["party_simplified"].str.upper() == "DEMOCRAT"]
    rep = county_party[county_party["party_simplified"].str.upper() == "REPUBLICAN"]

    dem = dem[["county_fips", "votes"]].rename(columns={"votes": "pres_dem_2024"})
    rep = rep[["county_fips", "votes"]].rename(columns={"votes": "pres_rep_2024"})

    result = dem.merge(rep, on="county_fips", how="outer").fillna(0)
    result["pres_total_2024"] = result["pres_dem_2024"] + result["pres_rep_2024"]
    denom = result["pres_dem_2024"] + result["pres_rep_2024"]
    result["pres_dem_share_2024"] = result["pres_dem_2024"] / denom.replace(0, float("nan"))
    result["state_abbr"] = state_abbr

    log.info(
        "  %s 2024 president: %d counties, total dem=%s rep=%s",
        state_abbr,
        len(result),
        f"{result['pres_dem_2024'].sum():,.0f}",
        f"{result['pres_rep_2024'].sum():,.0f}",
    )
    return result


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
    for state_abbr, (filename, fips) in STATES.items():
        url = f"{MEDSL_BASE}/{filename}"
        dest = RAW_DIR / filename

        log.info("=== %s 2024 ===", state_abbr)
        try:
            download_file(url, dest, f"MEDSL {state_abbr} 2024")
        except requests.HTTPError as e:
            log.warning("  Could not download %s: %s — skipping", state_abbr, e)
            continue

        df = load_medsl_csv(dest)
        county_df = extract_president_county(df, state_abbr)
        if len(county_df) > 0:
            frames.append(county_df)

    if not frames:
        log.error("No data loaded — check download errors above")
        return

    combined = pd.concat(frames, ignore_index=True)
    out_path = OUTPUT_DIR / "medsl_county_2024_president.parquet"
    combined.to_parquet(out_path, index=False)

    total_dem = combined["pres_dem_2024"].sum()
    total_rep = combined["pres_rep_2024"].sum()
    overall_dem_share = total_dem / (total_dem + total_rep)

    log.info(
        "Saved → %s | %d counties | overall dem share: %.1f%%",
        out_path, len(combined), overall_dem_share * 100,
    )

    # Summary by state
    print("\n── 2024 Presidential results by state ──────────────────────")
    print(f"  {'State':<6}  {'Counties':>8}  {'Dem votes':>12}  {'Rep votes':>12}  {'Dem share':>10}")
    print("  " + "-" * 55)
    for state_abbr in combined["state_abbr"].unique():
        s = combined[combined["state_abbr"] == state_abbr]
        d = s["pres_dem_2024"].sum()
        r = s["pres_rep_2024"].sum()
        print(f"  {state_abbr:<6}  {len(s):>8}  {d:>12,.0f}  {r:>12,.0f}  {d/(d+r):.1%}")

    # Sanity check against known 2024 state results
    ACTUAL_2024_STATE = {"FL": 0.4248, "GA": 0.4910, "AL": 0.3502}
    print("\n── Sanity check vs. known 2024 state results ───────────────")
    print(f"  {'State':<6}  {'Computed':>10}  {'Known':>10}  {'Diff':>8}")
    print("  " + "-" * 40)
    for state_abbr in combined["state_abbr"].unique():
        s = combined[combined["state_abbr"] == state_abbr]
        d = s["pres_dem_2024"].sum()
        r = s["pres_rep_2024"].sum()
        computed = d / (d + r)
        known = ACTUAL_2024_STATE.get(state_abbr, float("nan"))
        diff = computed - known
        print(f"  {state_abbr:<6}  {computed:.1%}  {known:.1%}  {diff:+.1%}")


if __name__ == "__main__":
    main()
