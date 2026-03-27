# National Forecast Expansion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand forecasting from 4 races (FL/GA/AL) to all ~83 Senate+Governor races nationally, so every defined race gets a forecast page even without polls.

**Architecture:** The pipeline is already race-agnostic — types are national (J=100, 3,154 counties, all 50 states). The main change is: (1) create a race registry that defines all 2026 races, (2) make the prediction loop iterate over the registry instead of poll groups, (3) races without polls get baseline predictions (model prior + generic ballot shift). API and frontend are already fully dynamic.

**Tech Stack:** Python, DuckDB, FastAPI, Next.js (no new dependencies)

**Spec:** `docs/superpowers/specs/2026-03-27-poll-calibration-national-forecast-design.md`

**Branch:** `feat/national-expansion`

---

## File Structure

```
Create: data/races/races_2026.csv              — Race registry (all Senate + Governor)
Create: src/assembly/define_races.py            — Load/validate race registry
Create: tests/assembly/test_define_races.py     — Tests for race registry
Modify: src/prediction/predict_2026_types.py    — Iterate over registry, not poll groups
Modify: src/db/build_database.py                — Add races table to DuckDB
Modify: data/polls/README.md                    — Document race label format
Create: tests/prediction/test_national_expansion.py — Integration tests
```

---

### Task 1: Create Race Registry

**Files:**
- Create: `data/races/races_2026.csv`
- Create: `src/assembly/define_races.py`
- Create: `tests/assembly/test_define_races.py`

The race registry is the single source of truth for what races exist. The prediction pipeline reads this, not the polls CSV.

- [ ] **Step 1: Write failing test for race loader**

```python
# tests/assembly/test_define_races.py
import pytest
from src.assembly.define_races import load_races, Race


def test_load_races_returns_list_of_race():
    races = load_races(2026)
    assert len(races) > 0
    assert all(isinstance(r, Race) for r in races)


def test_race_has_required_fields():
    races = load_races(2026)
    r = races[0]
    assert hasattr(r, "race_id")
    assert hasattr(r, "race_type")
    assert hasattr(r, "state")
    assert hasattr(r, "year")


def test_race_id_format():
    """Race IDs must match the label convention: 'YYYY ST RaceType'."""
    races = load_races(2026)
    for r in races:
        parts = r.race_id.split(" ")
        assert len(parts) == 3, f"Bad race_id format: {r.race_id}"
        assert parts[0] == "2026"
        assert len(parts[1]) == 2 and parts[1].isupper()
        assert parts[2] in ("Senate", "Governor")


def test_no_duplicate_race_ids():
    races = load_races(2026)
    ids = [r.race_id for r in races]
    assert len(ids) == len(set(ids)), f"Duplicate race IDs: {set(x for x in ids if ids.count(x) > 1)}"


def test_senate_count():
    """2026 has ~33-34 Senate races."""
    races = load_races(2026)
    senate = [r for r in races if r.race_type == "senate"]
    assert 30 <= len(senate) <= 40


def test_governor_count():
    """2026 has ~36 Governor races."""
    races = load_races(2026)
    gov = [r for r in races if r.race_type == "governor"]
    assert 30 <= len(gov) <= 40


def test_all_states_represented():
    """Most states should appear at least once."""
    races = load_races(2026)
    states = {r.state for r in races}
    assert len(states) >= 45  # Some states may have no race in 2026
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/assembly/test_define_races.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.assembly.define_races'`

- [ ] **Step 3: Create the race registry CSV**

Research the actual 2026 Senate and Governor races. Create `data/races/races_2026.csv`:

```csv
race_id,race_type,state,year
2026 AL Senate,senate,AL,2026
2026 AL Governor,governor,AL,2026
2026 AK Senate,senate,AK,2026
2026 AZ Senate,senate,AZ,2026
2026 AZ Governor,governor,AZ,2026
2026 AR Senate,senate,AR,2026
2026 AR Governor,governor,AR,2026
2026 CA Senate,senate,CA,2026
2026 CA Governor,governor,CA,2026
2026 CO Senate,senate,CO,2026
2026 CO Governor,governor,CO,2026
2026 CT Governor,governor,CT,2026
2026 DE Senate,senate,DE,2026
2026 FL Senate,senate,FL,2026
2026 FL Governor,governor,FL,2026
2026 GA Senate,senate,GA,2026
2026 GA Governor,governor,GA,2026
2026 HI Governor,governor,HI,2026
2026 IA Senate,senate,IA,2026
2026 IA Governor,governor,IA,2026
2026 ID Governor,governor,ID,2026
2026 IL Senate,senate,IL,2026
2026 IL Governor,governor,IL,2026
2026 IN Senate,senate,IN,2026
2026 KS Senate,senate,KS,2026
2026 KS Governor,governor,KS,2026
2026 KY Senate,senate,KY,2026
2026 LA Senate,senate,LA,2026
2026 MA Senate,senate,MA,2026
2026 MA Governor,governor,MA,2026
2026 MD Governor,governor,MD,2026
2026 ME Senate,senate,ME,2026
2026 ME Governor,governor,ME,2026
2026 MI Senate,senate,MI,2026
2026 MI Governor,governor,MI,2026
2026 MN Senate,senate,MN,2026
2026 MN Governor,governor,MN,2026
2026 MS Senate,senate,MS,2026
2026 MO Senate,senate,MO,2026
2026 MT Senate,senate,MT,2026
2026 NE Senate,senate,NE,2026
2026 NE Governor,governor,NE,2026
2026 NH Governor,governor,NH,2026
2026 NJ Senate,senate,NJ,2026
2026 NJ Governor,governor,NJ,2026
2026 NM Senate,senate,NM,2026
2026 NM Governor,governor,NM,2026
2026 NV Governor,governor,NV,2026
2026 NY Senate,senate,NY,2026
2026 NY Governor,governor,NY,2026
2026 NC Senate,senate,NC,2026
2026 OH Senate,senate,OH,2026
2026 OH Governor,governor,OH,2026
2026 OK Senate,senate,OK,2026
2026 OK Governor,governor,OK,2026
2026 OR Senate,senate,OR,2026
2026 OR Governor,governor,OR,2026
2026 PA Senate,senate,PA,2026
2026 PA Governor,governor,PA,2026
2026 RI Senate,senate,RI,2026
2026 RI Governor,governor,RI,2026
2026 SC Senate,senate,SC,2026
2026 SC Governor,governor,SC,2026
2026 SD Senate,senate,SD,2026
2026 SD Governor,governor,SD,2026
2026 TN Senate,senate,TN,2026
2026 TN Governor,governor,TN,2026
2026 TX Senate,senate,TX,2026
2026 TX Governor,governor,TX,2026
2026 UT Senate,senate,UT,2026
2026 VA Senate,senate,VA,2026
2026 VA Governor,governor,VA,2026
2026 VT Senate,senate,VT,2026
2026 VT Governor,governor,VT,2026
2026 WA Senate,senate,WA,2026
2026 WI Senate,senate,WI,2026
2026 WI Governor,governor,WI,2026
2026 WV Senate,senate,WV,2026
2026 WY Senate,senate,WY,2026
2026 WY Governor,governor,WY,2026
```

**IMPORTANT:** Verify the actual 2026 Senate class and Governor races via web search before finalizing. The list above is approximate — Class II Senate seats (33 regular + possible specials) and gubernatorial races vary. The agent MUST confirm which seats are actually up in 2026.

- [ ] **Step 4: Write the race loader module**

```python
# src/assembly/define_races.py
"""Load and validate the race registry for a given cycle."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "races"

VALID_RACE_TYPES = {"senate", "governor", "house"}


@dataclass(frozen=True)
class Race:
    race_id: str       # "2026 FL Senate"
    race_type: str     # "senate" | "governor" | "house"
    state: str         # "FL"
    year: int          # 2026
    district: int | None = None  # Only for house races


def load_races(cycle: int) -> list[Race]:
    """Load all races for a given election cycle from the registry CSV."""
    path = _DATA_DIR / f"races_{cycle}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No race registry found at {path}")
    df = pd.read_csv(path, dtype={"year": int})
    races = []
    for _, row in df.iterrows():
        race_type = row["race_type"].lower()
        if race_type not in VALID_RACE_TYPES:
            raise ValueError(f"Invalid race_type '{race_type}' in {row['race_id']}")
        races.append(Race(
            race_id=row["race_id"],
            race_type=race_type,
            state=row["state"],
            year=int(row["year"]),
            district=int(row["district"]) if "district" in row and pd.notna(row.get("district")) else None,
        ))
    return races


def races_for_state(cycle: int, state: str) -> list[Race]:
    """Return all races in a given state for a cycle."""
    return [r for r in load_races(cycle) if r.state == state]


def race_ids(cycle: int) -> list[str]:
    """Return all race_id strings for a cycle."""
    return [r.race_id for r in load_races(cycle)]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/assembly/test_define_races.py -v`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add data/races/races_2026.csv src/assembly/define_races.py tests/assembly/test_define_races.py
git commit -m "feat: add race registry for 2026 Senate + Governor races"
```

---

### Task 2: Modify Prediction Pipeline to Use Race Registry

**Files:**
- Modify: `src/prediction/predict_2026_types.py` (the main prediction loop, ~lines 469-496)
- Create: `tests/prediction/test_national_expansion.py`

Currently the pipeline does `for race, group in poll_agg.groupby("race")` — so only polled races get predictions. We need it to iterate over the race registry, using polls when available and baseline when not.

- [ ] **Step 1: Write failing integration test**

```python
# tests/prediction/test_national_expansion.py
"""Test that the prediction pipeline produces forecasts for all registered races."""
import pytest
import pandas as pd
from pathlib import Path

from src.assembly.define_races import load_races


@pytest.fixture
def predictions_path():
    return Path("data/predictions/county_predictions_2026_types.parquet")


def test_all_registered_races_have_predictions(predictions_path):
    """Every race in the registry must appear in predictions output."""
    if not predictions_path.exists():
        pytest.skip("Predictions not generated yet")
    preds = pd.read_parquet(predictions_path)
    pred_races = set(preds["race"].unique())
    registry_races = {r.race_id for r in load_races(2026)}
    missing = registry_races - pred_races
    assert len(missing) == 0, f"Races in registry but not predicted: {missing}"


def test_unpolled_race_gets_baseline_prediction(predictions_path):
    """A race with no polls should still get a prediction (baseline + GB shift)."""
    if not predictions_path.exists():
        pytest.skip("Predictions not generated yet")
    preds = pd.read_parquet(predictions_path)
    # Find a race that has no polls in the current CSV
    polls = pd.read_csv("data/polls/polls_2026.csv")
    polled_races = set(polls["race"].unique())
    registry_races = {r.race_id for r in load_races(2026)}
    unpolled = registry_races - polled_races
    if not unpolled:
        pytest.skip("All races have polls")
    sample_race = next(iter(unpolled))
    race_preds = preds[preds["race"] == sample_race]
    assert len(race_preds) > 0, f"Unpolled race '{sample_race}' has no predictions"
    assert race_preds["pred_dem_share"].notna().all(), "Predictions should not be NaN"


def test_polled_race_differs_from_baseline(predictions_path):
    """A race with polls should produce different predictions than baseline."""
    if not predictions_path.exists():
        pytest.skip("Predictions not generated yet")
    preds = pd.read_parquet(predictions_path)
    baseline = preds[preds["race"] == "baseline"].set_index("county_fips")["pred_dem_share"]
    # Use a well-polled race
    polls = pd.read_csv("data/polls/polls_2026.csv")
    polled_races = [r for r in polls["race"].unique() if not r.startswith("2026 Generic")]
    if not polled_races:
        pytest.skip("No polled races")
    race = polled_races[0]
    race_preds = preds[preds["race"] == race].set_index("county_fips")["pred_dem_share"]
    # At least some counties should differ from baseline
    common = baseline.index.intersection(race_preds.index)
    diff = (race_preds.loc[common] - baseline.loc[common]).abs()
    assert diff.max() > 0.001, f"Polled race '{race}' is identical to baseline"


def test_prediction_count_matches_counties_times_races(predictions_path):
    """Total prediction rows = n_counties * (n_races + 1 baseline)."""
    if not predictions_path.exists():
        pytest.skip("Predictions not generated yet")
    preds = pd.read_parquet(predictions_path)
    n_races = len(load_races(2026)) + 1  # +1 for baseline
    n_counties = preds["county_fips"].nunique()
    expected = n_counties * n_races
    actual = len(preds)
    assert actual == expected, f"Expected {expected} rows ({n_counties} counties × {n_races} races), got {actual}"
```

- [ ] **Step 2: Run test to confirm it fails**

Run: `uv run pytest tests/prediction/test_national_expansion.py -v`
Expected: FAIL (predictions don't include all registry races yet)

- [ ] **Step 3: Modify the prediction loop in predict_2026_types.py**

Find the main `run()` function (around line 363). The current loop at ~line 469 is:

```python
for race, race_group in poll_agg.groupby("race"):
    if race.startswith("2026 Generic Ballot"):
        continue
    ...
```

Replace with a loop over the race registry:

```python
# --- REPLACE the poll-driven race loop with registry-driven loop ---

from src.assembly.define_races import load_races

# Build poll lookup: race_id -> list of poll tuples
poll_lookup: dict[str, list[tuple]] = {}
if poll_agg is not None and len(poll_agg) > 0:
    for race, race_group in poll_agg.groupby("race"):
        if race.startswith("2026 Generic Ballot"):
            continue
        race_polls = [
            (float(row["dem_share"]), int(row["n_sample"]), str(row["state"]))
            for _, row in race_group.iterrows()
            if row.get("geo_level", "state") == "state"
        ]
        if race_polls:
            poll_lookup[race] = race_polls

# Iterate over ALL registered races
registry = load_races(2026)
for race_def in registry:
    race = race_def.race_id
    race_polls = poll_lookup.get(race, None)
    result = predict_race(
        race=race,
        type_scores=type_scores,
        type_covariance=type_covariance,
        type_priors=type_priors,
        county_fips=county_fips,
        polls=race_polls if race_polls else None,
        states=states,
        county_names=county_names,
        county_priors=county_priors,
        generic_ballot_shift=gb_info.shift,
    )
    result["race"] = race
    all_predictions.append(result)
```

This preserves all existing behavior for polled races but adds baseline predictions for every registered race.

- [ ] **Step 4: Run the prediction pipeline**

Run: `cd /home/hayden/projects/wethervane && uv run python -m src.prediction.predict_2026_types`
Expected: Completes without error, output file has rows for all registered races

- [ ] **Step 5: Run integration tests**

Run: `uv run pytest tests/prediction/test_national_expansion.py -v`
Expected: All pass

- [ ] **Step 6: Run full test suite to check for regressions**

Run: `uv run pytest tests/ -q --tb=short`
Expected: No new failures

- [ ] **Step 7: Commit**

```bash
git add src/prediction/predict_2026_types.py tests/prediction/test_national_expansion.py
git commit -m "feat: prediction pipeline iterates over race registry, all races get forecasts"
```

---

### Task 3: Add Races Table to DuckDB

**Files:**
- Modify: `src/db/build_database.py`

The DuckDB should store the race registry so the API can query race metadata (race_type, state) without parsing slugs.

- [ ] **Step 1: Write failing test**

```python
# tests/db/test_races_table.py
import pytest
import duckdb
from pathlib import Path


DB_PATH = Path("data/wethervane.duckdb")


@pytest.fixture
def db():
    if not DB_PATH.exists():
        pytest.skip("DuckDB not built")
    con = duckdb.connect(str(DB_PATH), read_only=True)
    yield con
    con.close()


def test_races_table_exists(db):
    tables = [row[0] for row in db.execute("SHOW TABLES").fetchall()]
    assert "races" in tables


def test_races_table_has_all_registered_races(db):
    from src.assembly.define_races import load_races
    registry = load_races(2026)
    db_races = db.execute("SELECT race_id FROM races").fetchall()
    db_ids = {row[0] for row in db_races}
    registry_ids = {r.race_id for r in registry}
    assert registry_ids == db_ids


def test_races_table_schema(db):
    cols = db.execute("DESCRIBE races").fetchall()
    col_names = {row[0] for row in cols}
    assert {"race_id", "race_type", "state", "year"} <= col_names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/db/test_races_table.py -v`
Expected: FAIL (races table doesn't exist)

- [ ] **Step 3: Add races table to build_database.py**

Find the database build function. Add after existing table creation:

```python
# --- Add to build_database.py ---

from src.assembly.define_races import load_races

def _build_races_table(con: duckdb.DuckDBPyConnection) -> None:
    """Create and populate the races table from the registry."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS races (
            race_id    VARCHAR PRIMARY KEY,
            race_type  VARCHAR NOT NULL,
            state      VARCHAR NOT NULL,
            year       INTEGER NOT NULL,
            district   INTEGER
        )
    """)
    con.execute("DELETE FROM races")  # Clear for rebuild
    races = load_races(2026)
    for r in races:
        con.execute(
            "INSERT INTO races VALUES (?, ?, ?, ?, ?)",
            [r.race_id, r.race_type, r.state, r.year, r.district],
        )
```

Call `_build_races_table(con)` from the main build function alongside other table builds.

- [ ] **Step 4: Rebuild DuckDB**

Run: `cd /home/hayden/projects/wethervane && uv run python src/db/build_database.py --reset`

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/db/test_races_table.py -v`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add src/db/build_database.py tests/db/test_races_table.py
git commit -m "feat: add races table to DuckDB from registry"
```

---

### Task 4: Extend API to Use Races Table

**Files:**
- Modify: `api/routers/forecast.py`

Currently the API parses race metadata from slug strings. With the races table, we can query directly.

- [ ] **Step 1: Add endpoint for race metadata**

```python
# Add to api/routers/forecast.py

@router.get("/forecast/race-metadata")
async def get_race_metadata():
    """Return metadata for all defined races."""
    con = _get_connection()
    try:
        rows = con.execute("""
            SELECT r.race_id, r.race_type, r.state, r.year,
                   COUNT(p.county_fips) > 0 AS has_predictions,
                   (SELECT COUNT(*) FROM polls pl WHERE pl.race = r.race_id) AS n_polls
            FROM races r
            LEFT JOIN predictions p ON r.race_id = p.race AND p.version_id = ?
            GROUP BY r.race_id, r.race_type, r.state, r.year
            ORDER BY r.state, r.race_type
        """, [_current_version_id()]).fetchall()
        return [
            {
                "race_id": row[0],
                "slug": race_to_slug(row[0]),
                "race_type": row[1],
                "state": row[2],
                "year": row[3],
                "has_predictions": bool(row[4]),
                "n_polls": row[5],
            }
            for row in rows
        ]
    finally:
        con.close()
```

- [ ] **Step 2: Update /forecast/race/{slug} to use races table for metadata**

In the existing `get_race_detail()` handler, replace slug parsing with a DB lookup:

```python
# Instead of parsing slug manually:
con = _get_connection()
try:
    race_label = slug_to_race(slug)
    row = con.execute(
        "SELECT race_type, state, year FROM races WHERE race_id = ?",
        [race_label],
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Race not found: {slug}")
    race_type, state_abbr, year = row[0], row[1], row[2]
finally:
    con.close()
```

- [ ] **Step 3: Run existing forecast API tests**

Run: `uv run pytest tests/ -k "forecast" -v`
Expected: All existing tests pass

- [ ] **Step 4: Commit**

```bash
git add api/routers/forecast.py
git commit -m "feat: API uses races table for metadata, add race-metadata endpoint"
```

---

### Task 5: Rebuild Pipeline and Deploy

**Files:** None new — this is an execution task.

- [ ] **Step 1: Verify race registry is accurate**

Web search to confirm the actual 2026 Senate (Class II) and Governor races. Update `data/races/races_2026.csv` if any races are wrong.

- [ ] **Step 2: Run full prediction pipeline**

```bash
cd /home/hayden/projects/wethervane
uv run python -m src.prediction.predict_2026_types
```

Expected: Predictions for ~83 races + baseline. Check output row count:
```bash
uv run python -c "import pandas as pd; df = pd.read_parquet('data/predictions/county_predictions_2026_types.parquet'); print(f'Rows: {len(df)}, Races: {df.race.nunique()}')"
```

- [ ] **Step 3: Rebuild DuckDB**

```bash
uv run python src/db/build_database.py --reset
```

- [ ] **Step 4: Run full test suite**

```bash
uv run pytest tests/ -q --tb=short
```

Expected: All tests pass (no regressions)

- [ ] **Step 5: Restart services and verify**

```bash
sudo systemctl restart wethervane-api.service
# Verify API serves new races:
curl -s http://localhost:8002/api/v1/forecast/races | python3 -m json.tool | head -20
curl -s http://localhost:8002/api/v1/forecast/race/2026-tx-senate | python3 -m json.tool | head -10
```

- [ ] **Step 6: Rebuild frontend**

```bash
cd /home/hayden/projects/wethervane/web
npm run build
cp -r public/ .next/standalone/public/
cp -r .next/static/ .next/standalone/.next/static/
sudo systemctl restart wethervane-frontend.service
```

- [ ] **Step 7: Verify site**

Visit `https://wethervane.hhaines.duckdns.org/forecast/2026-tx-senate` — should show a forecast page.

- [ ] **Step 8: Commit and push**

```bash
cd /home/hayden/projects/wethervane
git add -A
git commit -m "feat: national expansion — all Senate + Governor races forecasted"
git push
```

---

### Task 6: Update Polls README and Documentation

**Files:**
- Modify: `data/polls/README.md`
- Modify: `docs/ROADMAP.md`

- [ ] **Step 1: Update polls README with race label convention**

Add to `data/polls/README.md`:

```markdown
## Race Label Convention

Race labels MUST follow the format: `YYYY ST RaceType`

- `YYYY` = election year (e.g., 2026)
- `ST` = two-letter state abbreviation (uppercase)
- `RaceType` = `Senate`, `Governor`, or `House` (capitalized)

Examples:
- `2026 FL Senate`
- `2026 TX Governor`
- `2026 Generic Ballot` (special: national, geo_level=national)

Race labels must match entries in `data/races/races_YYYY.csv`. Polls with unrecognized
race labels will be loaded but will not match to a registered race.
```

- [ ] **Step 2: Update ROADMAP.md to mark national expansion done**

- [ ] **Step 3: Commit**

```bash
git add data/polls/README.md docs/ROADMAP.md
git commit -m "docs: update polls README and roadmap for national expansion"
```

---

## Validation Checklist

After all tasks complete:

- [ ] `data/races/races_2026.csv` has ~70-83 races (actual count depends on 2026 cycle)
- [ ] `load_races(2026)` returns all races with valid format
- [ ] Prediction pipeline generates forecasts for every registered race
- [ ] Unpolled races get baseline predictions (model prior + generic ballot)
- [ ] Polled races (FL/GA/AL) produce different predictions than baseline
- [ ] DuckDB has `races` table with all registered races
- [ ] API `/forecast/races` returns all ~83 race labels
- [ ] API `/forecast/race/{slug}` works for any registered race
- [ ] Frontend generates pages for all races via sitemap
- [ ] Full test suite passes
- [ ] Services restarted and site loads correctly
