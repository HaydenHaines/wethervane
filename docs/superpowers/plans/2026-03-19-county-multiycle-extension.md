# County Multi-Cycle Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the county-level shift-based community discovery pipeline from 3 election cycles (2016–2024) to 12 cycles (2000–2024) by fetching MEDSL county presidential returns and Algara & Amlani governor returns, rebuilding shift vectors with all available pairs, and re-running clustering + holdout validation.

**Architecture:** Two new data fetchers write to `data/raw/` and `data/assembled/`. A new multi-year shift builder produces `data/shifts/county_shifts_multiyear.parquet` with named columns for every consecutive election pair. The existing county clustering and holdout scripts are updated to consume the richer vectors, with 2020→2024 retained as the held-out test. Training shifts grow from 6 dims to ~30 dims.

**Tech Stack:** Python, pandas, requests, geopandas (existing), libpysal (existing), scikit-learn (existing), kneed (existing). No new dependencies needed.

**Spec:** Research findings from agent `af5012d2ca7454877` (2026-03-19). Key sources:
- MEDSL County Presidential 2000–2024: `doi:10.7910/DVN/VOQCHQ`
- Algara & Amlani county governor/Senate 1872–2020: `doi:10.7910/DVN/DGUMFI`

---

## File Map

### New files

| File | Responsibility |
|------|---------------|
| `src/assembly/fetch_medsl_county_presidential.py` | Download MEDSL county presidential 2000–2024 from Harvard Dataverse → `data/assembled/medsl_county_presidential_{year}.parquet` for each election year |
| `src/assembly/fetch_algara_amlani.py` | Download Algara & Amlani county governor dataset from Harvard Dataverse → `data/raw/algara_amlani/county_governor_raw.parquet` + `data/assembled/algara_county_governor_{year}.parquet` per cycle |
| `src/assembly/build_county_shifts_multiyear.py` | Combine all election-year parquets into multi-year county shift vectors → `data/shifts/county_shifts_multiyear.parquet` |
| `src/validation/validate_county_holdout_multiyear.py` | Holdout validation using multi-year shifts; print comparison table vs. 3-cycle baseline |
| `tests/test_fetch_medsl_presidential.py` | Unit tests for the MEDSL presidential fetcher (mock HTTP, verify dedup, two-party filter, county aggregation) |
| `tests/test_fetch_algara_amlani.py` | Unit tests for the Algara fetcher (mock HTTP, verify governor filtering, county output) |
| `tests/test_county_shifts_multiyear.py` | Unit tests for multi-year shift builder (shift math, AL structural zeros, column naming, shape) |
| `tests/test_county_holdout_multiyear.py` | Unit tests for holdout splitter and community-level accuracy function |

### Modified files

| File | Change |
|------|--------|
| `src/discovery/run_county_clustering.py` | Accept optional `--shifts-file` arg; default to multi-year file when it exists |
| `src/validation/validate_county_holdout.py` | No code change — runs as-is against the new shifts file for direct comparison |
| `CLAUDE.md` | Add new data sources to Key Decisions Log; update Commands section |
| `docs/DATA_SOURCES.md` | Add MEDSL county presidential and Algara & Amlani entries |

---

## Uncontested Race Register

Before writing code, document known structural zeros so tests can assert them:

| State | Office | Year | Issue |
|-------|--------|------|-------|
| AL | Governor | 2018 | Uncontested — set midterm shift dims to 0.0 |
| AL | Governor | 2014 | Check: Kay Ivey won with ~63%, contested. OK. |
| GA | Governor | 2002 | Sonny Perdue narrowly defeated Roy Barnes — contested. OK. |

Only AL 2018 governor is a confirmed structural zero in our data range. All others are contested races.

---

## Task 0: Documentation first

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/DATA_SOURCES.md`

- [ ] **Step 1: Update CLAUDE.md Key Decisions Log**

Add this row to the decisions table:

```markdown
| 2026-03-19 | County-level model extended to 2000–2024 via MEDSL + Algara/Amlani | County-level holdout validation showed r=0.93–0.98 (compared to r=-0.14 at tract level due to data resolution). County level is now the primary model engine. Adds 4 presidential cycles (2000–2012) and 4 governor midterm cycles (2002–2014). Sources: MEDSL doi:10.7910/DVN/VOQCHQ (presidential county 2000–2024, free); Algara & Amlani doi:10.7910/DVN/DGUMFI (governor/Senate county 1872–2020, free). |
```

Also update the Architecture section to note county as primary and tract as future refinement layer.

- [ ] **Step 2: Update docs/DATA_SOURCES.md**

Add a "County-Level Historical Returns" section with:
- MEDSL County Presidential Returns 2000–2024: Harvard Dataverse doi:10.7910/DVN/VOQCHQ, free, all US counties, columns: year/state_po/county_fips/candidatevotes/totalvotes/party_simplified
- Algara & Amlani county electoral dataset 1872–2020: Harvard Dataverse doi:10.7910/DVN/DGUMFI, free (academic replication data), 2-party vote shares for president/governor/Senate at county level

- [ ] **Step 3: Commit docs**

```bash
git add CLAUDE.md docs/DATA_SOURCES.md
git commit -m "docs: record county-level pivot and new data sources (MEDSL presidential + Algara/Amlani)"
```

---

## Task 1: Fetch MEDSL county presidential returns 2000–2024

**Files:**
- Create: `src/assembly/fetch_medsl_county_presidential.py`
- Test: `tests/test_fetch_medsl_presidential.py`

### Context

The MEDSL county presidential dataset (doi:10.7910/DVN/VOQCHQ) is available via the Harvard Dataverse API. The dataset contains a single CSV file with rows for every candidate in every county for 2000–2024.

Dataverse file listing API:
```
GET https://dataverse.harvard.edu/api/datasets/:persistentId/versions/:latest/files?persistentId=doi:10.7910/DVN/VOQCHQ
```
Returns JSON with a `data` array. Find the file with `dataFile.filename` ending in `.csv` (look for the main data file, not codebook/README).

Download a specific file:
```
GET https://dataverse.harvard.edu/api/access/datafile/{id}
```

Expected CSV columns (typical MEDSL county format):
`year, state, state_po, county_name, county_fips, office, candidate, party_simplified, candidatevotes, totalvotes, version`

Filter logic (same as existing 2024 fetcher pattern):
- `office` == "PRESIDENT"
- `party_simplified` in {"DEMOCRAT", "REPUBLICAN"}
- Keep only D and R rows; compute `dem_share = dem_votes / (dem_votes + rep_votes)` (two-party)
- States filter: `state_po` in {"FL", "GA", "AL"}

Output: one parquet per year in `data/assembled/`:
- `medsl_county_presidential_2000.parquet` → columns: `county_fips`, `state_abbr`, `pres_dem_{year}`, `pres_rep_{year}`, `pres_total_{year}`, `pres_dem_share_{year}`

Cache the raw CSV to `data/raw/medsl/county_presidential_2000_2024.csv` (no re-download if exists).

- [ ] **Step 1: Write failing tests**

```python
# tests/test_fetch_medsl_presidential.py
"""Tests for MEDSL county presidential fetcher.

Uses synthetic DataFrames to verify filtering, two-party share,
per-year output shape, and county_fips zero-padding.
"""
from __future__ import annotations
import pandas as pd
import pytest
from src.assembly.fetch_medsl_county_presidential import (
    filter_presidential_rows,
    aggregate_county_year,
    STATES,
)


@pytest.fixture
def raw_df():
    return pd.DataFrame({
        "year":            [2020, 2020, 2020, 2020, 2020],
        "state_po":        ["FL",  "FL",  "FL",  "TX",  "FL"],
        "county_fips":     ["12001","12001","12001","48001","12003"],
        "office":          ["PRESIDENT","PRESIDENT","PRESIDENT","PRESIDENT","PRESIDENT"],
        "party_simplified":["DEMOCRAT","REPUBLICAN","LIBERTARIAN","DEMOCRAT","DEMOCRAT"],
        "candidatevotes":  [100, 80, 5, 200, 50],
        "totalvotes":      [185, 185, 185, 200, 50],
    })


def test_filter_drops_non_presidential(raw_df):
    raw_df.loc[0, "office"] = "SENATE"
    result = filter_presidential_rows(raw_df)
    assert all(result["office"] == "PRESIDENT")


def test_filter_drops_third_party(raw_df):
    result = filter_presidential_rows(raw_df)
    assert set(result["party_simplified"]) == {"DEMOCRAT", "REPUBLICAN"}


def test_filter_drops_other_states(raw_df):
    result = filter_presidential_rows(raw_df)
    assert set(result["state_po"]).issubset(set(STATES.values()))


def test_aggregate_county_year_dem_share(raw_df):
    filtered = filter_presidential_rows(raw_df)
    result = aggregate_county_year(filtered, 2020)
    fl_row = result[result["county_fips"] == "12001"].iloc[0]
    # dem_share = 100 / (100+80)
    assert abs(fl_row["pres_dem_share_2020"] - 100/180) < 1e-6


def test_aggregate_county_year_columns(raw_df):
    filtered = filter_presidential_rows(raw_df)
    result = aggregate_county_year(filtered, 2020)
    expected = {"county_fips","state_abbr",
                "pres_dem_2020","pres_rep_2020",
                "pres_total_2020","pres_dem_share_2020"}
    assert set(result.columns) == expected


def test_county_fips_zero_padded(raw_df):
    # Simulate numeric county_fips
    raw_df["county_fips"] = raw_df["county_fips"].astype(int)
    filtered = filter_presidential_rows(raw_df)
    result = aggregate_county_year(filtered, 2020)
    assert all(result["county_fips"].str.len() == 5)
```

- [ ] **Step 2: Run tests, confirm they fail**

```bash
uv run pytest tests/test_fetch_medsl_presidential.py -v
```
Expected: ImportError or AttributeError (module not written yet)

- [ ] **Step 3: Write `src/assembly/fetch_medsl_county_presidential.py`**

```python
"""Fetch MEDSL county-level presidential returns 2000–2024.

Source: MIT Election Data + Science Lab, Harvard Dataverse
  doi:10.7910/DVN/VOQCHQ

Downloads a single unified CSV covering all US presidential elections
2000–2024 at the county level. Filters to FL, GA, AL. Computes
two-party dem share per county per year.

Output (one parquet per election year, data/assembled/):
  medsl_county_presidential_{year}.parquet
  Columns: county_fips, state_abbr, pres_dem_{year}, pres_rep_{year},
           pres_total_{year}, pres_dem_share_{year}

Cache: data/raw/medsl/county_presidential_2000_2024.csv
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "medsl"
ASSEMBLED_DIR = PROJECT_ROOT / "data" / "assembled"
CACHE_PATH = RAW_DIR / "county_presidential_2000_2024.csv"

DATAVERSE_DOI = "doi:10.7910/DVN/VOQCHQ"
DATAVERSE_API = "https://dataverse.harvard.edu/api"

# FL=12, GA=13, AL=01
STATES: dict[str, str] = {"FL": "12", "GA": "13", "AL": "01"}
STATE_ABBR = {v: k for k, v in STATES.items()}

PRES_YEARS = [2000, 2004, 2008, 2012, 2016, 2020, 2024]


def _dataverse_download(doi: str, cache_path: Path) -> Path:
    """Download primary data file from a Harvard Dataverse DOI."""
    if cache_path.exists():
        log.info("Using cached file: %s", cache_path)
        return cache_path

    # List files in dataset
    list_url = f"{DATAVERSE_API}/datasets/:persistentId/versions/:latest/files"
    resp = requests.get(list_url, params={"persistentId": doi}, timeout=30)
    resp.raise_for_status()
    files = resp.json()["data"]

    # Find the main CSV (skip README/codebook)
    csv_files = [
        f for f in files
        if f["dataFile"]["filename"].lower().endswith(".csv")
        and "readme" not in f["dataFile"]["filename"].lower()
        and "codebook" not in f["dataFile"]["filename"].lower()
    ]
    if not csv_files:
        raise FileNotFoundError(f"No CSV data file found in {doi}")

    # Pick the largest CSV (most likely to be the main data file)
    main_file = max(csv_files, key=lambda f: f["dataFile"].get("filesize", 0))
    file_id = main_file["dataFile"]["id"]
    filename = main_file["dataFile"]["filename"]
    log.info("Downloading %s (id=%s)...", filename, file_id)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    dl_url = f"{DATAVERSE_API}/access/datafile/{file_id}"
    with requests.get(dl_url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(cache_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)

    log.info("Saved → %s", cache_path)
    return cache_path


def filter_presidential_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only D/R presidential rows for FL, GA, AL."""
    mask = (
        (df["office"].str.upper().str.contains("PRESIDENT"))
        & (~df["office"].str.upper().str.contains("VICE"))
        & (df["party_simplified"].isin({"DEMOCRAT", "REPUBLICAN"}))
        & (df["state_po"].isin(set(STATES.values())))
    )
    return df[mask].copy()


def aggregate_county_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Aggregate filtered rows to one row per county for a given year.

    Returns county_fips, state_abbr, pres_dem_{year}, pres_rep_{year},
    pres_total_{year}, pres_dem_share_{year}.
    """
    yr = df[df["year"] == year].copy()
    yr["county_fips"] = yr["county_fips"].astype(str).str.zfill(5)

    dem = (
        yr[yr["party_simplified"] == "DEMOCRAT"]
        .groupby("county_fips")["candidatevotes"]
        .sum()
        .rename(f"pres_dem_{year}")
    )
    rep = (
        yr[yr["party_simplified"] == "REPUBLICAN"]
        .groupby("county_fips")["candidatevotes"]
        .sum()
        .rename(f"pres_rep_{year}")
    )
    result = pd.concat([dem, rep], axis=1).reset_index()
    result[f"pres_total_{year}"] = result[f"pres_dem_{year}"] + result[f"pres_rep_{year}"]
    result[f"pres_dem_share_{year}"] = (
        result[f"pres_dem_{year}"] / result[f"pres_total_{year}"]
    )
    result["state_abbr"] = result["county_fips"].str[:2].map(STATE_ABBR)
    return result[["county_fips", "state_abbr",
                   f"pres_dem_{year}", f"pres_rep_{year}",
                   f"pres_total_{year}", f"pres_dem_share_{year}"]]


def main() -> None:
    csv_path = _dataverse_download(DATAVERSE_DOI, CACHE_PATH)
    log.info("Loading CSV...")
    df = pd.read_csv(csv_path, low_memory=False)
    log.info("Raw rows: %d", len(df))

    df_filtered = filter_presidential_rows(df)
    log.info("After filter (FL+GA+AL D/R pres only): %d rows", len(df_filtered))

    ASSEMBLED_DIR.mkdir(parents=True, exist_ok=True)
    years_in_data = sorted(df_filtered["year"].unique())
    log.info("Years in data: %s", years_in_data)

    for year in PRES_YEARS:
        if year not in years_in_data:
            log.warning("Year %d not in dataset — skipping", year)
            continue
        agg = aggregate_county_year(df_filtered, year)
        out = ASSEMBLED_DIR / f"medsl_county_presidential_{year}.parquet"
        agg.to_parquet(out, index=False)
        log.info("  %d → %s (%d counties)", year, out.name, len(agg))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests, confirm they pass**

```bash
uv run pytest tests/test_fetch_medsl_presidential.py -v
```
Expected: all 5 tests PASS

- [ ] **Step 5: Run the fetcher**

```bash
uv run python src/assembly/fetch_medsl_county_presidential.py
```
Expected output:
```
Downloading countypres_2000-2024.csv (id=XXXXX)...
Saved → data/raw/medsl/county_presidential_2000_2024.csv
Raw rows: ~700000
After filter: ~25000 rows
Years in data: [2000, 2004, 2008, 2012, 2016, 2020, 2024]
  2000 → medsl_county_presidential_2000.parquet (293 counties)
  2004 → ...
```

Verify: `ls data/assembled/medsl_county_presidential_*.parquet` shows 7 files.

- [ ] **Step 6: Spot check**

```bash
uv run python -c "
import pandas as pd
df = pd.read_parquet('data/assembled/medsl_county_presidential_2000.parquet')
print(df[df['county_fips']=='12086'])  # Miami-Dade should be heavily Dem in 2000
print(df[df['county_fips']=='13067'])  # Cobb County GA
print(f'Total counties: {len(df)}')
"
```
Expected: 293 counties, Miami-Dade dem_share ~0.63 in 2000.

- [ ] **Step 7: Commit**

```bash
git add src/assembly/fetch_medsl_county_presidential.py tests/test_fetch_medsl_presidential.py data/raw/medsl/.gitkeep
git commit -m "feat: fetch MEDSL county presidential returns 2000–2024 (Harvard Dataverse)"
```

---

## Task 2: Fetch Algara & Amlani county governor returns 2000–2020

**Files:**
- Create: `src/assembly/fetch_algara_amlani.py`
- Test: `tests/test_fetch_algara_amlani.py`

### Context

The Algara & Amlani dataset (doi:10.7910/DVN/DGUMFI) is academic replication data published on Harvard Dataverse. It contains county-level electoral returns for president, governor, and Senate from 1872–2020.

The dataset may have columns like:
`fips`, `year`, `office`, `dem_votes`, `rep_votes`, `total_votes` (or similar — exact column names need to be discovered from the actual file).

Governor cycles we need for FL, GA, AL: 2002, 2006, 2010, 2014, 2018.
- **AL 2018**: Uncontested — output 0.0 for all shift dims (structural zero).
- **2022**: NOT in this dataset (ends at 2020). Use existing `medsl_county_2022_governor.parquet`.

Output per election year: `data/assembled/algara_county_governor_{year}.parquet`
Columns: `county_fips`, `state_abbr`, `gov_dem_{year}`, `gov_rep_{year}`, `gov_total_{year}`, `gov_dem_share_{year}`

- [ ] **Step 1: Discover dataset schema**

Before writing tests, download and inspect the dataset:

```bash
uv run python -c "
import requests, json
r = requests.get(
    'https://dataverse.harvard.edu/api/datasets/:persistentId/versions/:latest/files',
    params={'persistentId': 'doi:10.7910/DVN/DGUMFI'},
    timeout=30
)
files = r.json()['data']
for f in files:
    print(f['dataFile']['filename'], f['dataFile'].get('filesize', 0))
"
```

Download the main CSV/RData file and inspect columns:
```bash
uv run python -c "
import pandas as pd
# adjust filename/id based on discovery above
df = pd.read_csv('data/raw/algara_amlani/raw_data.csv', nrows=5)
print(df.columns.tolist())
print(df.head())
"
```

**Update the fetcher implementation below based on actual column names discovered.**

- [ ] **Step 2: Write failing tests**

```python
# tests/test_fetch_algara_amlani.py
"""Tests for Algara & Amlani county governor fetcher.

Uses synthetic DataFrames matching the actual dataset schema
discovered in Step 1.
"""
from __future__ import annotations
import pandas as pd
import pytest
from src.assembly.fetch_algara_amlani import (
    filter_governor_rows,
    aggregate_county_year,
    STATES,
    GOV_YEARS,
)


@pytest.fixture
def raw_df():
    """Synthetic Algara/Amlani rows for governor races."""
    # IMPORTANT: adjust column names to match actual dataset schema
    return pd.DataFrame({
        "year":        [2006, 2006, 2006, 2006, 2010],
        "state_fips":  ["12",  "12",  "13",  "99",  "12"],
        "county_fips": ["12001","12003","13001","99001","12001"],
        "office":      ["governor","governor","governor","governor","governor"],
        "dem_votes":   [1000, 500, 800, 200, 1100],
        "rep_votes":   [800,  600, 700, 300, 900],
        "total_votes": [1800, 1100, 1500, 500, 2000],
    })


def test_filter_governor_only(raw_df):
    raw_df.loc[0, "office"] = "president"
    result = filter_governor_rows(raw_df)
    assert all(result["office"].str.lower().str.contains("gov"))


def test_filter_target_states_only(raw_df):
    result = filter_governor_rows(raw_df)
    state_fips = result["county_fips"].str[:2]
    assert set(state_fips).issubset({"01", "12", "13"})


def test_aggregate_dem_share(raw_df):
    filtered = filter_governor_rows(raw_df)
    result = aggregate_county_year(filtered, 2006)
    fl_row = result[result["county_fips"] == "12001"].iloc[0]
    assert abs(fl_row["gov_dem_share_2006"] - 1000/1800) < 1e-6


def test_aggregate_columns(raw_df):
    filtered = filter_governor_rows(raw_df)
    result = aggregate_county_year(filtered, 2006)
    expected = {"county_fips", "state_abbr",
                "gov_dem_2006", "gov_rep_2006",
                "gov_total_2006", "gov_dem_share_2006"}
    assert set(result.columns) == expected


def test_gov_years_coverage():
    # We need 2002, 2006, 2010, 2014, 2018 from this dataset
    for yr in [2002, 2006, 2010, 2014, 2018]:
        assert yr in GOV_YEARS
```

- [ ] **Step 3: Run tests, confirm they fail**

```bash
uv run pytest tests/test_fetch_algara_amlani.py -v
```

- [ ] **Step 4: Write `src/assembly/fetch_algara_amlani.py`**

Adapt the template below to the actual column names discovered in Step 1:

```python
"""Fetch Algara & Amlani county-level governor returns 2000–2020.

Source: Algara & Amlani County Electoral Dataset 1872–2020
  Harvard Dataverse doi:10.7910/DVN/DGUMFI

Downloads the main data file, filters to governor races for
FL, GA, AL, and writes one parquet per election year.

Output (data/assembled/):
  algara_county_governor_{year}.parquet
  Columns: county_fips, state_abbr, gov_dem_{year}, gov_rep_{year},
           gov_total_{year}, gov_dem_share_{year}

Note: AL 2018 gubernatorial was uncontested — rows will have
near-zero Dem votes. The shift builder handles this as a
structural zero.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "algara_amlani"
ASSEMBLED_DIR = PROJECT_ROOT / "data" / "assembled"

DATAVERSE_DOI = "doi:10.7910/DVN/DGUMFI"
DATAVERSE_API = "https://dataverse.harvard.edu/api"

STATES: dict[str, str] = {"FL": "12", "GA": "13", "AL": "01"}
STATE_ABBR = {v: k for k, v in STATES.items()}

# Governor election years for FL, GA, AL in our range
GOV_YEARS = [2002, 2006, 2010, 2014, 2018]


def _dataverse_download(doi: str, raw_dir: Path) -> Path:
    """Download main data file from Harvard Dataverse DOI."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    list_url = f"{DATAVERSE_API}/datasets/:persistentId/versions/:latest/files"
    resp = requests.get(list_url, params={"persistentId": doi}, timeout=30)
    resp.raise_for_status()
    files = resp.json()["data"]

    # Find main data file (CSV or tab-delimited)
    data_files = [
        f for f in files
        if any(f["dataFile"]["filename"].lower().endswith(ext)
               for ext in [".csv", ".tab", ".tsv"])
        and "readme" not in f["dataFile"]["filename"].lower()
        and "codebook" not in f["dataFile"]["filename"].lower()
    ]
    if not data_files:
        raise FileNotFoundError(f"No data file found in {doi}. Files: {[f['dataFile']['filename'] for f in files]}")

    main_file = max(data_files, key=lambda f: f["dataFile"].get("filesize", 0))
    file_id = main_file["dataFile"]["id"]
    filename = main_file["dataFile"]["filename"]
    cache_path = raw_dir / filename

    if cache_path.exists():
        log.info("Using cached: %s", cache_path)
        return cache_path

    log.info("Downloading %s (id=%s)...", filename, file_id)
    dl_url = f"{DATAVERSE_API}/access/datafile/{file_id}"
    with requests.get(dl_url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(cache_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
    log.info("Saved → %s", cache_path)
    return cache_path


def filter_governor_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only governor rows for FL, GA, AL.

    NOTE: Column names depend on actual dataset schema — adjust if needed.
    Expected columns: year, county_fips (or fips), office,
    dem_votes (or similar), rep_votes, total_votes.
    """
    # Normalize county_fips
    fips_col = "county_fips" if "county_fips" in df.columns else "fips"
    df = df.copy()
    df["county_fips"] = df[fips_col].astype(str).str.zfill(5)

    mask = (
        df["office"].str.lower().str.contains("gov")
        & df["county_fips"].str[:2].isin(set(STATES.values()))
    )
    return df[mask].copy()


def aggregate_county_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Aggregate governor rows to one row per county for a given year."""
    yr = df[df["year"] == year].copy()

    # Flexible column name handling
    dem_col = next(c for c in yr.columns if "dem" in c.lower() and "vote" in c.lower())
    rep_col = next(c for c in yr.columns if "rep" in c.lower() and "vote" in c.lower())

    result = yr.groupby("county_fips").agg(
        gov_dem=(dem_col, "sum"),
        gov_rep=(rep_col, "sum"),
    ).reset_index()

    result[f"gov_total_{year}"] = result["gov_dem"] + result["gov_rep"]
    result[f"gov_dem_share_{year}"] = result["gov_dem"] / result[f"gov_total_{year}"]
    result = result.rename(columns={
        "gov_dem": f"gov_dem_{year}",
        "gov_rep": f"gov_rep_{year}",
    })
    result["state_abbr"] = result["county_fips"].str[:2].map(STATE_ABBR)
    return result[["county_fips", "state_abbr",
                   f"gov_dem_{year}", f"gov_rep_{year}",
                   f"gov_total_{year}", f"gov_dem_share_{year}"]]


def main() -> None:
    raw_path = _dataverse_download(DATAVERSE_DOI, RAW_DIR)

    log.info("Loading data file...")
    sep = "\t" if raw_path.suffix in {".tab", ".tsv"} else ","
    df = pd.read_csv(raw_path, sep=sep, low_memory=False)
    log.info("Columns: %s", df.columns.tolist())
    log.info("Raw rows: %d", len(df))

    df_filtered = filter_governor_rows(df)
    log.info("After filter (FL+GA+AL governor): %d rows", len(df_filtered))

    ASSEMBLED_DIR.mkdir(parents=True, exist_ok=True)
    years_in_data = sorted(df_filtered["year"].unique())
    log.info("Governor years in data: %s", years_in_data)

    for year in GOV_YEARS:
        if year not in years_in_data:
            log.warning("Year %d not in dataset — skipping", year)
            continue
        agg = aggregate_county_year(df_filtered, year)
        out = ASSEMBLED_DIR / f"algara_county_governor_{year}.parquet"
        agg.to_parquet(out, index=False)
        log.info("  %d → %s (%d counties)", year, out.name, len(agg))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests, confirm they pass**

```bash
uv run pytest tests/test_fetch_algara_amlani.py -v
```

- [ ] **Step 6: Run the fetcher**

```bash
uv run python src/assembly/fetch_algara_amlani.py
```

If column names don't match the fixture in the tests, adjust `filter_governor_rows` and `aggregate_county_year` to match the actual schema, then re-run tests.

Expected: 5 parquet files in `data/assembled/algara_county_governor_{2002,2006,2010,2014,2018}.parquet`, each with ~200–293 counties (some small states may have fewer governor elections).

- [ ] **Step 7: Spot check**

```bash
uv run python -c "
import pandas as pd
for yr in [2002, 2006, 2010, 2014, 2018]:
    df = pd.read_parquet(f'data/assembled/algara_county_governor_{yr}.parquet')
    al = df[df['county_fips'].str.startswith('01')]
    fl = df[df['county_fips'].str.startswith('12')]
    ga = df[df['county_fips'].str.startswith('13')]
    print(f'{yr}: AL={len(al)} FL={len(fl)} GA={len(ga)} counties')
    if yr == 2018:
        print('  AL 2018 dem_share (expect near 0 — uncontested):',
              al[f'gov_dem_share_{yr}'].mean().round(3))
"
```

- [ ] **Step 8: Commit**

```bash
git add src/assembly/fetch_algara_amlani.py tests/test_fetch_algara_amlani.py
git commit -m "feat: fetch Algara & Amlani county governor returns 2002–2018 (Harvard Dataverse)"
```

---

## Task 3: Build multi-year county shift vectors

**Files:**
- Create: `src/assembly/build_county_shifts_multiyear.py`
- Test: `tests/test_county_shifts_multiyear.py`

### Design

The output is `data/shifts/county_shifts_multiyear.parquet` with columns:

**Training dimensions (all pre-2024 consecutive pairs):**
```
pres_d_shift_00_04, pres_r_shift_00_04, pres_turnout_shift_00_04
pres_d_shift_04_08, pres_r_shift_04_08, pres_turnout_shift_04_08
pres_d_shift_08_12, pres_r_shift_08_12, pres_turnout_shift_08_12
pres_d_shift_12_16, pres_r_shift_12_16, pres_turnout_shift_12_16
pres_d_shift_16_20, pres_r_shift_16_20, pres_turnout_shift_16_20
gov_d_shift_02_06,  gov_r_shift_02_06,  gov_turnout_shift_02_06
gov_d_shift_06_10,  gov_r_shift_06_10,  gov_turnout_shift_06_10
gov_d_shift_10_14,  gov_r_shift_10_14,  gov_turnout_shift_10_14
gov_d_shift_14_18,  gov_r_shift_14_18,  gov_turnout_shift_14_18
gov_d_shift_18_22,  gov_r_shift_18_22,  gov_turnout_shift_18_22
```

**Holdout dimensions (always separated, not used in clustering):**
```
pres_d_shift_20_24, pres_r_shift_20_24, pres_turnout_shift_20_24
```

Total: 30 training dims + 3 holdout dims = 33 columns + county_fips.

**Shift math (same as tract-level pipeline):**
- D-shift = later_dem_share - earlier_dem_share
- R-shift = (1 - later_dem_share) - (1 - earlier_dem_share)
- Turnout-shift = (later_total - earlier_total) / earlier_total

**Special cases:**
- AL 2018 governor was uncontested → `gov_d_shift_14_18` and `gov_d_shift_18_22` for AL counties: set all three dimensions to 0.0
- If an election year parquet is missing entirely for a pair, the three shift columns for that pair are filled with 0.0 for all counties (logged as a warning)

**County spine**: Use MEDSL 2024 county list (293 counties in FL+GA+AL) as the authoritative spine.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_county_shifts_multiyear.py
"""Tests for multi-year county shift vector builder."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from src.assembly.build_county_shifts_multiyear import (
    compute_pres_shift,
    compute_gov_shift,
    build_multiyear_shifts,
    TRAINING_SHIFT_COLS,
    HOLDOUT_SHIFT_COLS,
    AL_FIPS_PREFIX,
)


@pytest.fixture
def early_pres():
    return pd.DataFrame({
        "county_fips": ["12001", "13001", "01001"],
        "pres_dem_share_2016": [0.60, 0.55, 0.35],
        "pres_total_2016": [100000, 80000, 40000],
    })


@pytest.fixture
def late_pres():
    return pd.DataFrame({
        "county_fips": ["12001", "13001", "01001"],
        "pres_dem_share_2020": [0.65, 0.52, 0.33],
        "pres_total_2020": [110000, 82000, 41000],
    })


@pytest.fixture
def early_gov():
    return pd.DataFrame({
        "county_fips": ["12001", "13001", "01001"],
        "gov_dem_share_2014": [0.50, 0.45, 0.30],
        "gov_total_2014": [90000, 70000, 35000],
    })


@pytest.fixture
def late_gov():
    return pd.DataFrame({
        "county_fips": ["12001", "13001", "01001"],
        "gov_dem_share_2018": [0.49, 0.50, 0.05],   # AL near-zero: uncontested
        "gov_total_2018": [95000, 72000, 36000],
    })


def test_pres_d_shift_math(early_pres, late_pres):
    result = compute_pres_shift(early_pres, late_pres, "16", "20")
    fl_row = result[result["county_fips"] == "12001"].iloc[0]
    assert abs(fl_row["pres_d_shift_16_20"] - 0.05) < 1e-6


def test_pres_r_shift_is_negative_d(early_pres, late_pres):
    result = compute_pres_shift(early_pres, late_pres, "16", "20")
    fl_row = result[result["county_fips"] == "12001"].iloc[0]
    assert abs(fl_row["pres_r_shift_16_20"] + fl_row["pres_d_shift_16_20"]) < 1e-10


def test_pres_turnout_shift(early_pres, late_pres):
    result = compute_pres_shift(early_pres, late_pres, "16", "20")
    fl_row = result[result["county_fips"] == "12001"].iloc[0]
    expected = (110000 - 100000) / 100000
    assert abs(fl_row["pres_turnout_shift_16_20"] - expected) < 1e-9


def test_al_gov_shift_zeroed_when_uncontested(early_gov, late_gov):
    """AL 2018 governor was uncontested — all three shift dims must be 0.0."""
    result = compute_gov_shift(early_gov, late_gov, "14", "18")
    al_row = result[result["county_fips"] == "01001"].iloc[0]
    assert al_row["gov_d_shift_14_18"] == 0.0
    assert al_row["gov_r_shift_14_18"] == 0.0
    assert al_row["gov_turnout_shift_14_18"] == 0.0


def test_output_column_count():
    assert len(TRAINING_SHIFT_COLS) == 30
    assert len(HOLDOUT_SHIFT_COLS) == 3


def test_build_multiyear_spine(early_pres, late_pres, early_gov, late_gov, tmp_path):
    """build_multiyear_shifts returns all counties on the spine."""
    spine = pd.DataFrame({"county_fips": ["12001", "13001", "01001"]})
    # With only one pres pair and one gov pair provided, rest filled with 0
    pres_pairs = [("16", "20", early_pres, late_pres)]
    gov_pairs = [("14", "18", early_gov, late_gov)]
    result = build_multiyear_shifts(spine, pres_pairs, gov_pairs)
    assert len(result) == 3
    assert "county_fips" in result.columns
    all_shift_cols = TRAINING_SHIFT_COLS + HOLDOUT_SHIFT_COLS
    for col in all_shift_cols:
        assert col in result.columns, f"Missing column: {col}"
```

- [ ] **Step 2: Run tests, confirm they fail**

```bash
uv run pytest tests/test_county_shifts_multiyear.py -v
```

- [ ] **Step 3: Write `src/assembly/build_county_shifts_multiyear.py`**

```python
"""Build multi-year county shift vectors (33 dimensions).

Training (30 dims):
  5 consecutive presidential pairs: 2000→2004, 2004→2008, 2008→2012,
                                    2012→2016, 2016→2020
  5 consecutive governor pairs:     2002→2006, 2006→2010, 2010→2014,
                                    2014→2018, 2018→2022

Holdout (3 dims — never used during clustering):
  Presidential 2020→2024

Special cases:
  AL 2018 governor was uncontested → zero all three shift dims for AL
  for both the 2014→2018 and 2018→2022 gov pairs.

Output:
  data/shifts/county_shifts_multiyear.parquet
  Columns: county_fips + 30 training + 3 holdout shift dims
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSEMBLED_DIR = PROJECT_ROOT / "data" / "assembled"
SHIFTS_DIR = PROJECT_ROOT / "data" / "shifts"

AL_FIPS_PREFIX = "01"

# ── Column name constants ─────────────────────────────────────────────────────

PRES_PAIRS = [("00", "04"), ("04", "08"), ("08", "12"), ("12", "16"), ("16", "20")]
GOV_PAIRS  = [("02", "06"), ("06", "10"), ("10", "14"), ("14", "18"), ("18", "22")]
HOLDOUT_PAIRS = [("20", "24")]

TRAINING_SHIFT_COLS: list[str] = []
for a, b in PRES_PAIRS:
    TRAINING_SHIFT_COLS += [f"pres_d_shift_{a}_{b}", f"pres_r_shift_{a}_{b}", f"pres_turnout_shift_{a}_{b}"]
for a, b in GOV_PAIRS:
    TRAINING_SHIFT_COLS += [f"gov_d_shift_{a}_{b}", f"gov_r_shift_{a}_{b}", f"gov_turnout_shift_{a}_{b}"]

HOLDOUT_SHIFT_COLS: list[str] = []
for a, b in HOLDOUT_PAIRS:
    HOLDOUT_SHIFT_COLS += [f"pres_d_shift_{a}_{b}", f"pres_r_shift_{a}_{b}", f"pres_turnout_shift_{a}_{b}"]


# ── Core computation ──────────────────────────────────────────────────────────

def _dem_share_col(df: pd.DataFrame) -> str:
    return next(c for c in df.columns if "dem_share" in c)

def _total_col(df: pd.DataFrame) -> str:
    return next(c for c in df.columns if "_total_" in c)

def _zero_al(df: pd.DataFrame, shift_cols: list[str]) -> pd.DataFrame:
    """Set shift dimensions to 0.0 for Alabama counties."""
    al_mask = df["county_fips"].str.startswith(AL_FIPS_PREFIX)
    df.loc[al_mask, shift_cols] = 0.0
    return df


def compute_pres_shift(
    early: pd.DataFrame, late: pd.DataFrame, a: str, b: str
) -> pd.DataFrame:
    """Compute D/R/turnout shifts between two presidential election DataFrames."""
    early_dem = _dem_share_col(early)
    early_tot = _total_col(early)
    late_dem = _dem_share_col(late)
    late_tot = _total_col(late)

    merged = early[["county_fips", early_dem, early_tot]].merge(
        late[["county_fips", late_dem, late_tot]], on="county_fips", how="inner"
    )
    d_shift = merged[late_dem] - merged[early_dem]
    r_shift = -(d_shift)
    early_total = merged[early_tot].replace(0, float("nan"))
    t_shift = (merged[late_tot] - merged[early_tot]) / early_total

    result = pd.DataFrame({
        "county_fips": merged["county_fips"],
        f"pres_d_shift_{a}_{b}": d_shift.values,
        f"pres_r_shift_{a}_{b}": r_shift.values,
        f"pres_turnout_shift_{a}_{b}": t_shift.values,
    })
    return result


def compute_gov_shift(
    early: pd.DataFrame, late: pd.DataFrame, a: str, b: str
) -> pd.DataFrame:
    """Compute D/R/turnout shifts between two governor election DataFrames.

    AL counties are zeroed if the later year was an uncontested race.
    Currently: AL 2018 is always zeroed (structural zero).
    """
    early_dem = _dem_share_col(early)
    early_tot = _total_col(early)
    late_dem = _dem_share_col(late)
    late_tot = _total_col(late)

    merged = early[["county_fips", early_dem, early_tot]].merge(
        late[["county_fips", late_dem, late_tot]], on="county_fips", how="inner"
    )
    d_shift = merged[late_dem] - merged[early_dem]
    r_shift = -(d_shift)
    early_total = merged[early_tot].replace(0, float("nan"))
    t_shift = (merged[late_tot] - merged[early_tot]) / early_total

    result = pd.DataFrame({
        "county_fips": merged["county_fips"],
        f"gov_d_shift_{a}_{b}": d_shift.values,
        f"gov_r_shift_{a}_{b}": r_shift.values,
        f"gov_turnout_shift_{a}_{b}": t_shift.values,
    })

    # AL 2018 governor was uncontested — zero shifts involving 2018
    if b == "18" or a == "18":
        result = _zero_al(result, [f"gov_d_shift_{a}_{b}", f"gov_r_shift_{a}_{b}",
                                    f"gov_turnout_shift_{a}_{b}"])
    return result


def build_multiyear_shifts(
    spine: pd.DataFrame,
    pres_pairs: list,
    gov_pairs: list,
) -> pd.DataFrame:
    """Join all shift pairs onto the county spine.

    pres_pairs: list of (a_str, b_str, early_df, late_df)
    gov_pairs: list of (a_str, b_str, early_df, late_df)

    Missing pairs produce zero-filled columns (logged as warning).
    """
    result = spine[["county_fips"]].copy()

    all_cols = TRAINING_SHIFT_COLS + HOLDOUT_SHIFT_COLS
    for col in all_cols:
        result[col] = 0.0

    for a, b, early, late in pres_pairs:
        shifts = compute_pres_shift(early, late, a, b)
        cols = [f"pres_d_shift_{a}_{b}", f"pres_r_shift_{a}_{b}", f"pres_turnout_shift_{a}_{b}"]
        result = result.merge(shifts[["county_fips"] + cols], on="county_fips", how="left", suffixes=("_old",""))
        for col in cols:
            if col + "_old" in result.columns:
                result[col] = result[col].fillna(result[col + "_old"])
                result = result.drop(columns=[col + "_old"])
            result[col] = result[col].fillna(0.0)

    for a, b, early, late in gov_pairs:
        shifts = compute_gov_shift(early, late, a, b)
        cols = [f"gov_d_shift_{a}_{b}", f"gov_r_shift_{a}_{b}", f"gov_turnout_shift_{a}_{b}"]
        result = result.merge(shifts[["county_fips"] + cols], on="county_fips", how="left", suffixes=("_old",""))
        for col in cols:
            if col + "_old" in result.columns:
                result[col] = result[col].fillna(result[col + "_old"])
                result = result.drop(columns=[col + "_old"])
            result[col] = result[col].fillna(0.0)

    return result[["county_fips"] + TRAINING_SHIFT_COLS + HOLDOUT_SHIFT_COLS]


def _load(filename: str) -> pd.DataFrame | None:
    path = ASSEMBLED_DIR / filename
    if not path.exists():
        log.warning("Missing: %s — will zero-fill this pair", path.name)
        return None
    return pd.read_parquet(path)


def main() -> None:
    """Load all election parquets and build multi-year shifts."""
    # Spine: the 293 counties from the 2024 dataset
    spine_df = pd.read_parquet(ASSEMBLED_DIR / "medsl_county_2024_president.parquet")
    spine = spine_df[["county_fips"]].copy()
    log.info("County spine: %d counties", len(spine))

    # ── Presidential pairs ────────────────────────────────────────────────────
    pres_file = {
        "00": "medsl_county_presidential_2000.parquet",
        "04": "medsl_county_presidential_2004.parquet",
        "08": "medsl_county_presidential_2008.parquet",
        "12": "medsl_county_presidential_2012.parquet",
        "16": "medsl_county_presidential_2016.parquet",
        "20": "medsl_county_presidential_2020.parquet",
        "24": "medsl_county_2024_president.parquet",
    }
    pres_dfs: dict[str, pd.DataFrame | None] = {k: _load(v) for k, v in pres_file.items()}

    pres_pairs = []
    for a, b in PRES_PAIRS + HOLDOUT_PAIRS:
        early, late = pres_dfs.get(a), pres_dfs.get(b)
        if early is not None and late is not None:
            pres_pairs.append((a, b, early, late))
        else:
            log.warning("Skipping pres pair %s→%s (data missing)", a, b)

    # ── Governor pairs ────────────────────────────────────────────────────────
    gov_file = {
        "02": "algara_county_governor_2002.parquet",
        "06": "algara_county_governor_2006.parquet",
        "10": "algara_county_governor_2010.parquet",
        "14": "algara_county_governor_2014.parquet",
        "18": "algara_county_governor_2018.parquet",
        "22": "medsl_county_2022_governor.parquet",
    }
    gov_dfs: dict[str, pd.DataFrame | None] = {k: _load(v) for k, v in gov_file.items()}

    gov_pairs = []
    for a, b in GOV_PAIRS:
        early, late = gov_dfs.get(a), gov_dfs.get(b)
        if early is not None and late is not None:
            gov_pairs.append((a, b, early, late))
        else:
            log.warning("Skipping gov pair %s→%s (data missing)", a, b)

    # ── Build ─────────────────────────────────────────────────────────────────
    log.info("Building multi-year shifts: %d pres pairs, %d gov pairs",
             len(pres_pairs), len(gov_pairs))
    shifts = build_multiyear_shifts(spine, pres_pairs, gov_pairs)

    SHIFTS_DIR.mkdir(parents=True, exist_ok=True)
    out = SHIFTS_DIR / "county_shifts_multiyear.parquet"
    shifts.to_parquet(out, index=False)
    log.info("Saved → %s | %d counties | %d training dims + %d holdout dims",
             out, len(shifts), len(TRAINING_SHIFT_COLS), len(HOLDOUT_SHIFT_COLS))

    zero_cols = [c for c in TRAINING_SHIFT_COLS if shifts[c].abs().max() < 1e-10]
    if zero_cols:
        log.warning("These training columns are all-zero (missing data): %s", zero_cols)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests, confirm they pass**

```bash
uv run pytest tests/test_county_shifts_multiyear.py -v
```

- [ ] **Step 5: Run the builder**

```bash
uv run python src/assembly/build_county_shifts_multiyear.py
```

Expected: `data/shifts/county_shifts_multiyear.parquet` with 293 counties × 33 columns. Any missing data pairs logged as warnings (zero-filled).

- [ ] **Step 6: Spot check**

```bash
uv run python -c "
import pandas as pd
df = pd.read_parquet('data/shifts/county_shifts_multiyear.parquet')
print('Shape:', df.shape)
print('Columns:', df.columns.tolist())
print()
# Check AL gov structural zeros
al = df[df['county_fips'].str.startswith('01')]
print('AL gov_d_shift_14_18 (should all be 0.0):', al['gov_d_shift_14_18'].unique())
print()
# Check Miami-Dade has large negative pres_d_shift_20_24 (Hispanic realignment)
miami = df[df['county_fips']=='12086']
print('Miami-Dade pres_d_shift_20_24 (expect ~-0.10 to -0.15):', miami['pres_d_shift_20_24'].values)
# Check zero training cols
zero_cols = [c for c in df.columns if c != 'county_fips' and df[c].abs().max() < 1e-10]
print('All-zero cols (missing data):', zero_cols)
"
```

- [ ] **Step 7: Commit**

```bash
git add src/assembly/build_county_shifts_multiyear.py tests/test_county_shifts_multiyear.py
git commit -m "feat: multi-year county shift vectors (30 training + 3 holdout dims, 2000-2024)"
```

---

## Task 4: Holdout validation with multi-year shifts

**Files:**
- Create: `src/validation/validate_county_holdout_multiyear.py`
- Test: `tests/test_county_holdout_multiyear.py`

### Design

This script:
1. Loads `county_shifts_multiyear.parquet` and `county_adjacency.npz`
2. Splits into training (first 30 cols) and holdout (last 3 cols)
3. Normalises training with StandardScaler
4. Runs constrained Ward clustering at k=5, 7, 10, 15, 20
5. For each k, computes community-level D-shift means for training and holdout
6. Computes Pearson r and MAE
7. Prints a comparison table vs. the 3-cycle baseline (r=0.93–0.98 from agent run)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_county_holdout_multiyear.py
"""Tests for multi-year county holdout validation."""
from __future__ import annotations
import numpy as np
import pytest
from src.validation.validate_county_holdout_multiyear import (
    split_training_holdout,
    community_correlation,
)

N_TRAINING = 30
N_HOLDOUT = 3


def test_split_training_shape():
    shifts = np.random.rand(293, N_TRAINING + N_HOLDOUT)
    train, holdout = split_training_holdout(shifts, N_TRAINING)
    assert train.shape == (293, N_TRAINING)
    assert holdout.shape == (293, N_HOLDOUT)


def test_split_holdout_is_last_cols():
    shifts = np.arange(293 * 33).reshape(293, 33).astype(float)
    train, holdout = split_training_holdout(shifts, 30)
    np.testing.assert_array_equal(holdout, shifts[:, 30:])


def test_community_correlation_perfect():
    labels = np.array([0, 0, 1, 1, 2, 2])
    train_means = np.array([[0.1], [0.1], [0.5], [0.5], [-0.1], [-0.1]])
    holdout_means = np.array([[0.2], [0.2], [0.6], [0.6], [0.0], [0.0]])
    # Community means: train=[0.1, 0.5, -0.1], holdout=[0.2, 0.6, 0.0] — perfect r
    comm_train = np.array([train_means[labels == k].mean(axis=0) for k in [0,1,2]])
    comm_holdout = np.array([holdout_means[labels == k].mean(axis=0) for k in [0,1,2]])
    r, mae = community_correlation(comm_train, comm_holdout)
    assert r > 0.99


def test_community_correlation_range():
    labels = np.array([0, 0, 1, 1])
    train = np.random.rand(4, 30)
    holdout = np.random.rand(4, 3)
    comm_train = np.array([train[labels == k].mean(axis=0) for k in [0,1]])
    comm_holdout = np.array([holdout[labels == k].mean(axis=0) for k in [0,1]])
    r, mae = community_correlation(comm_train, comm_holdout)
    assert -1.0 <= r <= 1.0
    assert mae >= 0
```

- [ ] **Step 2: Run tests, confirm fail**

```bash
uv run pytest tests/test_county_holdout_multiyear.py -v
```

- [ ] **Step 3: Write `src/validation/validate_county_holdout_multiyear.py`**

```python
"""Multi-year county holdout validation.

Trains community discovery on 30-dimensional pre-2024 shifts,
tests against 2020→2024 (3 holdout dims). Reports Pearson r and
MAE at multiple k values. Compares against 3-cycle baseline.

Usage:
  uv run python -m src.validation.validate_county_holdout_multiyear
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHIFTS_PATH = PROJECT_ROOT / "data" / "shifts" / "county_shifts_multiyear.parquet"
ADJACENCY_DIR = PROJECT_ROOT / "data" / "communities"

# Baseline results from 3-cycle run (county_shifts.parquet, 6 training dims)
BASELINE = {5: 0.983, 7: 0.964, 10: 0.941, 15: 0.934, 20: 0.932}

N_TRAINING_COLS = 30  # first 30 cols after county_fips are training
K_VALUES = [5, 7, 10, 15, 20]


def split_training_holdout(
    shifts: np.ndarray,
    n_training: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split shift matrix into training and holdout portions."""
    return shifts[:, :n_training], shifts[:, n_training:]


def community_correlation(
    training_means: np.ndarray,
    holdout_means: np.ndarray,
) -> tuple[float, float]:
    """Pearson r and MAE on first (D-shift) column of community means."""
    from scipy.stats import pearsonr
    train_d = training_means[:, 0]
    holdout_d = holdout_means[:, 0]
    r, _ = pearsonr(train_d, holdout_d)
    mae = float(np.mean(np.abs(train_d - holdout_d)))
    return float(r), mae


def main() -> None:
    import pandas as pd
    from scipy.sparse import load_npz
    from sklearn.cluster._agglomerative import _hc_cut
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler

    log.info("Loading multi-year county shifts from %s", SHIFTS_PATH)
    df = pd.read_parquet(SHIFTS_PATH)
    fips_list = df["county_fips"].tolist()
    shift_cols = [c for c in df.columns if c != "county_fips"]
    all_shifts = df[shift_cols].values.astype(float)

    # Align to adjacency ordering
    geoids_path = ADJACENCY_DIR / "county_adjacency.fips.txt"
    geoids = geoids_path.read_text().splitlines()
    fips_indexed = df.set_index("county_fips")
    aligned = fips_indexed.reindex(geoids)
    n_missing = aligned[shift_cols[0]].isna().sum()
    if n_missing:
        log.info("Filling %d counties with missing data using column means", n_missing)
        aligned[shift_cols] = aligned[shift_cols].fillna(aligned[shift_cols].mean())
    all_shifts = aligned[shift_cols].values.astype(float)

    train, holdout = split_training_holdout(all_shifts, N_TRAINING_COLS)
    log.info("Training dims: %d | Holdout dims: %d | Counties: %d",
             train.shape[1], holdout.shape[1], train.shape[0])

    scaler = StandardScaler()
    train_norm = scaler.fit_transform(train)

    adjacency_path = ADJACENCY_DIR / "county_adjacency.npz"
    W = load_npz(str(adjacency_path))

    # Build full tree once
    log.info("Building full Ward dendrogram...")
    model = AgglomerativeClustering(
        linkage="ward", connectivity=W, n_clusters=1, compute_distances=True
    )
    model.fit(train_norm)

    print("\n" + "=" * 70)
    print("Multi-Year County Holdout Validation — 30 training dims vs 3-cycle baseline")
    print("Train: pres 2000–2020 (5 pairs) + gov 2002–2022 (5 pairs)")
    print("Holdout: pres 2020→2024")
    print("=" * 70)
    print(f"{'k':>4}  {'r (multiyear)':>16}  {'r (3-cycle)':>12}  {'delta':>7}  {'MAE':>8}")
    print("-" * 70)

    for k in K_VALUES:
        labels = _hc_cut(k, model.children_, len(geoids))
        unique_labels = np.unique(labels)
        train_means = np.array([train[labels == lbl].mean(axis=0) for lbl in unique_labels])
        holdout_means = np.array([holdout[labels == lbl].mean(axis=0) for lbl in unique_labels])
        r, mae = community_correlation(train_means, holdout_means)
        baseline_r = BASELINE.get(k, float("nan"))
        delta = r - baseline_r
        print(f"{k:>4}  {r:>16.4f}  {baseline_r:>12.4f}  {delta:>+7.4f}  {mae:>8.4f}")

    print()
    log.info("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
uv run pytest tests/test_county_holdout_multiyear.py -v
```

- [ ] **Step 5: Run the validation**

```bash
uv run python -m src.validation.validate_county_holdout_multiyear
```

Expected output format:
```
======================================================================
Multi-Year County Holdout Validation — 30 training dims vs 3-cycle baseline
Train: pres 2000–2020 (5 pairs) + gov 2002–2022 (5 pairs)
Holdout: pres 2020→2024
======================================================================
   k   r (multiyear)   r (3-cycle)    delta       MAE
----------------------------------------------------------------------
   5          0.XXXX        0.9830   +0.XXXX    0.XXXX
   7          0.XXXX        0.9640   +0.XXXX    0.XXXX
  10          0.XXXX        0.9408   +0.XXXX    0.XXXX
...
```

**If delta is consistently positive:** multi-year signals improve holdout prediction.
**If delta is near zero:** 3 cycles already captured most of the signal.
**If delta is negative:** investigate — possible overfitting or data quality issue in older cycles.

- [ ] **Step 6: Commit**

```bash
git add src/validation/validate_county_holdout_multiyear.py tests/test_county_holdout_multiyear.py
git commit -m "feat: multi-year county holdout validation vs 3-cycle baseline"
```

---

## Task 5: Final documentation update

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update CLAUDE.md Commands section**

Add to the Commands block:
```bash
# Multi-year county data pipeline
python src/assembly/fetch_medsl_county_presidential.py   # MEDSL county pres 2000–2024
python src/assembly/fetch_algara_amlani.py               # Algara governor 2002–2018
python src/assembly/build_county_shifts_multiyear.py     # 30-dim county shift vectors
python -m src.validation.validate_county_holdout_multiyear  # Compare to 3-cycle baseline
```

- [ ] **Step 2: Update Key Decisions Log with validation results**

Add a row with the actual r values from Task 4 Step 5:
```
| 2026-03-19 | Multi-year county holdout r=X.XX (k=10) | Compared to 3-cycle baseline r=0.94; delta=+X.XX. Confirms/does not confirm that additional historical cycles improve community structure. |
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update with multi-year validation results and commands"
```
