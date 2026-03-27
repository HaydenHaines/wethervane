# Historical Calibration & Model Evaluation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a backtesting harness that replays the full pipeline (type discovery → poll integration → forecast) against historical election cycles (2012-2024), compares forecasts to actual results, and produces quantitative evaluation metrics. Secondarily, learn optimal λ and μ from the backtest.

**Architecture:** For each historical cycle k, the harness: (1) builds type model using only elections *before* cycle k, (2) loads polls for cycle k from DuckDB, (3) runs the forecast engine (θ_national + δ_race), (4) compares to actual county-level results. This is leave-one-cycle-out cross-validation. The evaluation framework produces per-cycle, per-race-type, and per-type accuracy metrics.

**Tech Stack:** Python, numpy, DuckDB, pandas. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-03-27-poll-calibration-national-forecast-design.md` (Section 3)

**Depends on:** Plan B (forecast engine must exist)

**Branch:** `feat/historical-calibration`

---

## File Structure

```
Create: src/calibration/ingest_historical_polls.py   — Convert 538 polls into DuckDB
Create: src/calibration/ingest_historical_results.py  — Load election results into DuckDB
Create: src/calibration/backtest.py                   — Leave-one-cycle-out harness
Create: src/calibration/evaluate.py                   — Metrics: RMSE, r, bias, calibration
Create: src/calibration/replay_pipeline.py            — Full pipeline replay per cycle
Create: tests/calibration/test_ingest_historical.py
Create: tests/calibration/test_backtest.py
Create: tests/calibration/test_evaluate.py
Modify: src/db/build_database.py                      — Add historical_polls, historical_results tables
```

---

### Task 1: Historical Polls into DuckDB

**Files:**
- Create: `src/calibration/ingest_historical_polls.py`
- Create: `tests/calibration/test_ingest_historical.py`
- Modify: `src/db/build_database.py`

Convert 538 raw_polls data (2012-2024) into a queryable DuckDB table.

- [ ] **Step 1: Write failing test**

```python
# tests/calibration/test_ingest_historical.py
import pytest
from pathlib import Path

from src.calibration.ingest_historical_polls import (
    convert_538_cycle,
    HistoricalPoll,
)


def test_convert_538_cycle_returns_polls():
    """538 data for 2020 should produce hundreds of polls."""
    raw_dir = Path("data/raw/fivethirtyeight/data-repo/state-of-the-polls-2024")
    polls_file = raw_dir / "2020_polls.csv"
    if not polls_file.exists():
        pytest.skip("538 data not available")
    polls = convert_538_cycle(2020, raw_dir)
    assert len(polls) > 100
    assert all(isinstance(p, HistoricalPoll) for p in polls)


def test_historical_poll_fields():
    raw_dir = Path("data/raw/fivethirtyeight/data-repo/state-of-the-polls-2024")
    polls_file = raw_dir / "2020_polls.csv"
    if not polls_file.exists():
        pytest.skip("538 data not available")
    polls = convert_538_cycle(2020, raw_dir)
    p = polls[0]
    assert hasattr(p, "cycle")
    assert hasattr(p, "race_type")
    assert hasattr(p, "state")
    assert hasattr(p, "dem_share")
    assert hasattr(p, "n_sample") or hasattr(p, "pollster")
    assert 0 < p.dem_share < 1


def test_race_types_include_all():
    """Should have presidential, senate, governor polls."""
    raw_dir = Path("data/raw/fivethirtyeight/data-repo/state-of-the-polls-2024")
    if not (raw_dir / "2020_polls.csv").exists():
        pytest.skip("538 data not available")
    polls = convert_538_cycle(2020, raw_dir)
    types = {p.race_type for p in polls}
    # At minimum should have presidential and senate
    assert "president" in types or "senate" in types
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/calibration/test_ingest_historical.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Explore 538 data format**

Read the first 10 lines of `data/raw/fivethirtyeight/data-repo/state-of-the-polls-2024/2020_polls.csv` to understand the column structure. Map 538 columns to our schema.

- [ ] **Step 4: Implement convert_538_cycle**

```python
# src/calibration/ingest_historical_polls.py
"""Convert 538 historical poll data into standardized format for DuckDB."""

from dataclasses import dataclass
from pathlib import Path
import pandas as pd


@dataclass
class HistoricalPoll:
    cycle: int
    race_id: str        # "2020 GA Senate" format
    race_type: str      # "president", "senate", "governor", "house"
    state: str          # 2-letter abbreviation
    district: int | None
    pollster: str
    dem_share: float
    n_sample: int
    date: str           # ISO format
    grade: str | None
    source: str         # "538"


# Map 538 race type codes to our types
_538_TYPE_MAP = {
    "Pres-G": "president",
    "Sen-G": "senate",
    "Gov-G": "governor",
    "House-G": "house",
}


def convert_538_cycle(cycle: int, raw_dir: Path) -> list[HistoricalPoll]:
    """Convert a single 538 cycle's polls to HistoricalPoll objects.

    Reads {cycle}_polls.csv from the raw 538 data directory.
    Extracts pollster, state, race type, and computes two-party Dem share.
    """
    polls_file = raw_dir / f"{cycle}_polls.csv"
    if not polls_file.exists():
        return []

    df = pd.read_csv(polls_file)
    polls = []

    # 538 format varies by year — adapt based on available columns
    # Common columns: pollster_name, state, start_date, end_date,
    # has_prez?, has_senate?, has_house?, 2024_pollster_rating
    # Need to cross-reference with races file for actual poll results

    # Implementation depends on exact 538 format — agent should read the CSV
    # and adapt. The key output is HistoricalPoll objects with dem_share computed
    # from candidate percentages.

    # [Agent: read the actual CSV columns and implement accordingly]

    return polls


def ingest_all_cycles(raw_dir: Path, db_path: Path) -> int:
    """Convert and load all available 538 cycles into DuckDB.

    Returns total number of polls ingested.
    """
    import duckdb

    con = duckdb.connect(str(db_path))
    con.execute("""
        CREATE TABLE IF NOT EXISTS historical_polls (
            cycle       INTEGER NOT NULL,
            race_id     VARCHAR NOT NULL,
            race_type   VARCHAR NOT NULL,
            state       VARCHAR,
            district    INTEGER,
            pollster    VARCHAR,
            dem_share   DOUBLE NOT NULL,
            n_sample    INTEGER,
            date        DATE,
            grade       VARCHAR,
            source      VARCHAR DEFAULT '538'
        )
    """)
    con.execute("DELETE FROM historical_polls WHERE source = '538'")

    total = 0
    for cycle in [2012, 2016, 2020, 2024]:
        polls = convert_538_cycle(cycle, raw_dir)
        for p in polls:
            con.execute(
                "INSERT INTO historical_polls VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                [p.cycle, p.race_id, p.race_type, p.state, p.district,
                 p.pollster, p.dem_share, p.n_sample, p.date, p.grade, p.source],
            )
        total += len(polls)

    con.close()
    return total
```

**NOTE:** The 538 CSV format varies by year. The implementing agent MUST read the actual CSV header and adapt the parsing. The dataclass and DuckDB schema are fixed; the parsing logic must be discovered from the data.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/calibration/test_ingest_historical.py -v`

- [ ] **Step 6: Commit**

```bash
git add src/calibration/ tests/calibration/
git commit -m "feat: ingest 538 historical polls into DuckDB"
```

---

### Task 2: Historical Election Results into DuckDB

**Files:**
- Create: `src/calibration/ingest_historical_results.py`
- Modify: `src/db/build_database.py`

Load actual county-level election results for backtesting.

- [ ] **Step 1: Write failing test**

```python
# tests/calibration/test_ingest_results.py
import pytest
from src.calibration.ingest_historical_results import load_presidential_results


def test_load_presidential_results_2020():
    results = load_presidential_results(2020)
    assert len(results) > 3000  # ~3,154 counties
    r = results[0]
    assert hasattr(r, "county_fips")
    assert hasattr(r, "dem_share")
    assert 0 <= r.dem_share <= 1
```

- [ ] **Step 2: Implement result loaders**

Load from existing MEDSL data (presidential), Algara (governor). Skeleton for Senate (MEDSL Harvard Dataverse — may need download).

```python
# src/calibration/ingest_historical_results.py
"""Load historical election results at county level for backtesting."""

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass
class CountyResult:
    cycle: int
    race_type: str    # "president", "senate", "governor"
    state: str
    county_fips: str
    dem_share: float
    dem_votes: int
    rep_votes: int
    total_votes: int


def load_presidential_results(cycle: int) -> list[CountyResult]:
    """Load county-level presidential results from MEDSL data."""
    # MEDSL files are at data/raw/medsl/ or data/communities/
    # Agent should discover exact path and column names
    # Key: compute dem_share = dem_votes / total_votes
    pass


def load_governor_results(cycle: int) -> list[CountyResult]:
    """Load county-level governor results from Algara/Amlani data."""
    pass


def load_all_results_to_db(db_path: Path) -> int:
    """Load all available results into DuckDB historical_results table."""
    import duckdb
    con = duckdb.connect(str(db_path))
    con.execute("""
        CREATE TABLE IF NOT EXISTS historical_results (
            cycle       INTEGER NOT NULL,
            race_type   VARCHAR NOT NULL,
            state       VARCHAR NOT NULL,
            county_fips VARCHAR NOT NULL,
            dem_share   DOUBLE NOT NULL,
            dem_votes   INTEGER,
            rep_votes   INTEGER,
            total_votes INTEGER,
            PRIMARY KEY (cycle, race_type, county_fips)
        )
    """)
    # [Agent: implement loading from existing data files]
    con.close()
    return 0
```

**NOTE:** The implementing agent must discover the exact file paths and column names for MEDSL presidential and Algara governor data. These files exist on disk — read them to determine format.

- [ ] **Step 3: Run tests and commit**

---

### Task 3: Pipeline Replay (Full Backtest Cycle)

**Files:**
- Create: `src/calibration/replay_pipeline.py`

For a given historical cycle, replay the entire pipeline: build types from pre-cycle data, run forecast engine with that cycle's polls, compare to actual results.

- [ ] **Step 1: Write failing test**

```python
# tests/calibration/test_backtest.py
import pytest
import numpy as np
from src.calibration.replay_pipeline import replay_cycle, CycleResult


def test_replay_cycle_2020():
    """Replay 2020 presidential cycle and get evaluation metrics."""
    result = replay_cycle(
        target_cycle=2020,
        race_type="president",
        lam=1.0,
        mu=1.0,
    )
    assert isinstance(result, CycleResult)
    assert result.rmse > 0
    assert result.rmse < 0.20  # Should be within 20pp at worst
    assert -1 <= result.correlation <= 1
    assert result.n_counties > 3000
    assert len(result.county_predictions) == result.n_counties
```

- [ ] **Step 2: Implement replay_cycle**

```python
# src/calibration/replay_pipeline.py
"""Replay the full forecast pipeline against a historical election cycle.

For cycle k:
1. Load type model (types were fit on all elections, but priors use only pre-k data)
2. Load polls for cycle k from DuckDB
3. Run forecast engine: θ_prior → θ_national → δ_race
4. Compare county predictions to actual results
5. Return evaluation metrics

This is the core of the model evaluation harness. If we can consistently
get close to the right answer election after election, the model is predictive.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from pathlib import Path


@dataclass
class CycleResult:
    """Evaluation result for one cycle."""
    target_cycle: int
    race_type: str
    lam: float
    mu: float
    rmse: float                    # Vote-weighted county RMSE
    correlation: float             # Pearson r (predicted vs actual)
    mean_bias: float               # Mean(predicted - actual), + = overestimates D
    n_counties: int
    n_polls: int
    county_predictions: np.ndarray  # (n_counties,) predicted dem_share
    county_actuals: np.ndarray      # (n_counties,) actual dem_share
    state_results: dict[str, dict]  # state -> {pred, actual, error}
    type_errors: np.ndarray         # (J,) per-type mean error


def replay_cycle(
    target_cycle: int,
    race_type: str = "president",
    lam: float = 1.0,
    mu: float = 1.0,
    db_path: Path | None = None,
) -> CycleResult:
    """Replay forecast pipeline for a single historical cycle.

    Steps:
    1. Load county type scores (J=100, already fitted on all data)
    2. Build county priors from elections BEFORE target_cycle
       (e.g., for 2020: use 2000-2018 as training data for Ridge priors)
    3. Load polls for target_cycle from historical_polls table
    4. Run forecast engine
    5. Load actual results from historical_results table
    6. Compute evaluation metrics
    """
    from src.prediction.forecast_engine import (
        compute_theta_prior, run_forecast, build_W_state,
    )

    if db_path is None:
        db_path = Path("data/wethervane.duckdb")

    import duckdb
    con = duckdb.connect(str(db_path), read_only=True)

    # Step 1: Load type scores (national model)
    ta_df = pd.read_parquet("data/communities/type_assignments.parquet")
    county_fips = ta_df["county_fips"].astype(str).str.zfill(5).tolist()
    score_cols = sorted([c for c in ta_df.columns if c.endswith("_score")])
    type_scores = ta_df[score_cols].values
    J = type_scores.shape[1]

    # Step 2: Build county priors from pre-cycle elections
    # Use the most recent presidential result before target_cycle as prior
    prior_cycle = target_cycle - 4  # Previous presidential cycle
    prior_results = con.execute("""
        SELECT county_fips, dem_share
        FROM historical_results
        WHERE cycle = ? AND race_type = 'president'
    """, [prior_cycle]).fetchdf()

    if len(prior_results) == 0:
        con.close()
        raise ValueError(f"No presidential results for prior cycle {prior_cycle}")

    # Align priors with type_assignments order
    prior_map = dict(zip(
        prior_results["county_fips"].astype(str).str.zfill(5),
        prior_results["dem_share"],
    ))
    county_priors = np.array([prior_map.get(f, 0.5) for f in county_fips])

    # Extract states from FIPS
    states = [_fips_to_state(f) for f in county_fips]

    # Placeholder votes (equal weight if no data)
    county_votes = np.ones(len(county_fips))

    # Step 3: Load polls for target cycle
    polls_df = con.execute("""
        SELECT race_id, state, dem_share, n_sample
        FROM historical_polls
        WHERE cycle = ? AND race_type = ?
    """, [target_cycle, race_type]).fetchdf()

    polls_by_race: dict[str, list[dict]] = {}
    for _, row in polls_df.iterrows():
        race_id = row["race_id"]
        if race_id not in polls_by_race:
            polls_by_race[race_id] = []
        polls_by_race[race_id].append({
            "dem_share": float(row["dem_share"]),
            "n_sample": int(row["n_sample"]) if pd.notna(row["n_sample"]) else 600,
            "state": str(row["state"]),
        })

    races = list(polls_by_race.keys()) if polls_by_race else ["baseline"]

    # Step 4: Run forecast engine
    results = run_forecast(
        type_scores=type_scores,
        county_priors=county_priors,
        states=states,
        county_votes=county_votes,
        polls_by_race=polls_by_race,
        races=races,
        lam=lam,
        mu=mu,
    )

    # Use the first race result (for presidential, there's typically one)
    if races:
        fr = results[races[0]]
        predictions = fr.county_preds_local
    else:
        predictions = county_priors  # Fallback

    # Step 5: Load actual results
    actual_df = con.execute("""
        SELECT county_fips, dem_share
        FROM historical_results
        WHERE cycle = ? AND race_type = ?
    """, [target_cycle, race_type]).fetchdf()
    con.close()

    actual_map = dict(zip(
        actual_df["county_fips"].astype(str).str.zfill(5),
        actual_df["dem_share"],
    ))
    actuals = np.array([actual_map.get(f, np.nan) for f in county_fips])

    # Step 6: Compute metrics (only where we have both pred and actual)
    valid = ~np.isnan(actuals)
    pred_valid = predictions[valid]
    actual_valid = actuals[valid]

    errors = pred_valid - actual_valid
    rmse = np.sqrt(np.mean(errors ** 2))
    correlation = np.corrcoef(pred_valid, actual_valid)[0, 1] if len(pred_valid) > 1 else 0.0
    mean_bias = np.mean(errors)

    # Per-type errors
    type_errors = np.zeros(J)
    for j in range(J):
        mask = valid & (type_scores[:, j] > 0.1)  # Counties with ≥10% membership
        if mask.any():
            type_errors[j] = np.mean(predictions[mask] - actuals[mask])

    # State-level results
    state_results = {}
    for state in set(states):
        s_mask = np.array([s == state for s in states]) & valid
        if s_mask.any():
            s_pred = np.mean(pred_valid[s_mask[valid]])
            s_actual = np.mean(actual_valid[s_mask[valid]])
            state_results[state] = {
                "pred": float(s_pred),
                "actual": float(s_actual),
                "error": float(s_pred - s_actual),
            }

    return CycleResult(
        target_cycle=target_cycle,
        race_type=race_type,
        lam=lam,
        mu=mu,
        rmse=rmse,
        correlation=correlation,
        mean_bias=mean_bias,
        n_counties=int(valid.sum()),
        n_polls=len(polls_df),
        county_predictions=predictions,
        county_actuals=actuals,
        state_results=state_results,
        type_errors=type_errors,
    )


def _fips_to_state(fips: str) -> str:
    """Convert 5-digit FIPS to state abbreviation."""
    # Import from config
    from src.core.config import get_state_fips
    fips_map = get_state_fips()
    state_fips = fips[:2]
    # Reverse lookup
    for abbr, code in fips_map.items():
        if str(code).zfill(2) == state_fips:
            return abbr
    return "XX"
```

- [ ] **Step 3: Run tests**

- [ ] **Step 4: Commit**

```bash
git add src/calibration/replay_pipeline.py tests/calibration/test_backtest.py
git commit -m "feat: replay_cycle — full pipeline replay for historical backtesting"
```

---

### Task 4: Evaluation Metrics Module

**Files:**
- Create: `src/calibration/evaluate.py`
- Create: `tests/calibration/test_evaluate.py`

Standardized evaluation metrics computed from CycleResults.

- [ ] **Step 1: Write tests**

```python
# tests/calibration/test_evaluate.py
import numpy as np
import pytest
from src.calibration.evaluate import (
    compute_rmse, compute_calibration, compute_state_accuracy,
    EvaluationReport, build_report,
)


def test_compute_rmse():
    pred = np.array([0.50, 0.60, 0.40])
    actual = np.array([0.52, 0.58, 0.42])
    rmse = compute_rmse(pred, actual)
    assert 0.01 < rmse < 0.03


def test_compute_calibration():
    """Calibration: do 90% CIs contain 90% of actuals?"""
    np.random.seed(42)
    n = 1000
    actuals = np.random.rand(n) * 0.6 + 0.2
    preds = actuals + np.random.randn(n) * 0.03
    ci_lower = preds - 0.06
    ci_upper = preds + 0.06
    coverage = compute_calibration(actuals, ci_lower, ci_upper)
    assert 0.85 < coverage < 0.99  # Should be ~95% for 2σ


def test_compute_state_accuracy():
    """State accuracy: did we call the right winner?"""
    state_results = {
        "FL": {"pred": 0.48, "actual": 0.47},  # Both R — correct
        "GA": {"pred": 0.51, "actual": 0.50},  # Both D — correct (barely)
        "PA": {"pred": 0.52, "actual": 0.49},  # Called D, was R — wrong
    }
    accuracy = compute_state_accuracy(state_results)
    assert accuracy == pytest.approx(2 / 3)
```

- [ ] **Step 2: Implement**

```python
# src/calibration/evaluate.py
"""Evaluation metrics for forecast backtesting.

Metrics:
- RMSE (county-level, vote-weighted)
- Correlation (predicted vs actual)
- Mean bias (systematic D/R lean)
- Calibration (CI coverage)
- State winner accuracy (binary correct/incorrect)
- Per-type error (which community types are hardest to predict)
"""

from dataclasses import dataclass
import numpy as np


def compute_rmse(
    predictions: np.ndarray,
    actuals: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    errors = predictions - actuals
    if weights is not None:
        return float(np.sqrt(np.average(errors ** 2, weights=weights)))
    return float(np.sqrt(np.mean(errors ** 2)))


def compute_calibration(
    actuals: np.ndarray,
    ci_lower: np.ndarray,
    ci_upper: np.ndarray,
) -> float:
    """Fraction of actuals within confidence interval."""
    within = (actuals >= ci_lower) & (actuals <= ci_upper)
    return float(within.mean())


def compute_state_accuracy(state_results: dict[str, dict]) -> float:
    """Fraction of states where winner was called correctly."""
    correct = 0
    total = 0
    for state, res in state_results.items():
        pred_d_wins = res["pred"] > 0.5
        actual_d_wins = res["actual"] > 0.5
        if pred_d_wins == actual_d_wins:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0


@dataclass
class EvaluationReport:
    """Summary report across multiple cycles."""
    cycles: list[int]
    mean_rmse: float
    mean_correlation: float
    mean_bias: float
    mean_state_accuracy: float
    per_cycle: dict[int, dict]  # cycle -> {rmse, r, bias, state_accuracy}
    per_type_rmse: np.ndarray   # (J,) — mean RMSE by type across cycles


def build_report(cycle_results: list) -> EvaluationReport:
    """Build evaluation report from list of CycleResult objects."""
    cycles = [cr.target_cycle for cr in cycle_results]
    per_cycle = {}
    type_errors_all = []

    for cr in cycle_results:
        state_acc = compute_state_accuracy(cr.state_results)
        per_cycle[cr.target_cycle] = {
            "rmse": cr.rmse,
            "r": cr.correlation,
            "bias": cr.mean_bias,
            "state_accuracy": state_acc,
            "n_polls": cr.n_polls,
            "n_counties": cr.n_counties,
        }
        type_errors_all.append(np.abs(cr.type_errors))

    return EvaluationReport(
        cycles=cycles,
        mean_rmse=float(np.mean([pc["rmse"] for pc in per_cycle.values()])),
        mean_correlation=float(np.mean([pc["r"] for pc in per_cycle.values()])),
        mean_bias=float(np.mean([pc["bias"] for pc in per_cycle.values()])),
        mean_state_accuracy=float(np.mean([pc["state_accuracy"] for pc in per_cycle.values()])),
        per_cycle=per_cycle,
        per_type_rmse=np.mean(type_errors_all, axis=0) if type_errors_all else np.array([]),
    )
```

- [ ] **Step 3: Run tests and commit**

---

### Task 5: λ/μ Grid Search

**Files:**
- Create: `src/calibration/optimize_params.py`

Sweep (λ, μ) grid across all held-out cycles. Find optimal values.

- [ ] **Step 1: Implement grid search**

```python
# src/calibration/optimize_params.py
"""Grid search for optimal λ (prior trust) and μ (candidate effect shrinkage)."""

import numpy as np
import json
from pathlib import Path
from itertools import product

from src.calibration.replay_pipeline import replay_cycle
from src.calibration.evaluate import build_report


def grid_search(
    cycles: list[int] = [2012, 2016, 2020, 2024],
    race_type: str = "president",
    lam_values: list[float] = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    mu_values: list[float] = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    output_path: Path | None = None,
) -> dict:
    """Run leave-one-cycle-out CV across (λ, μ) grid.

    Returns dict with:
    - best_lam, best_mu: optimal params
    - best_rmse: RMSE at optimum
    - grid: full results for all (λ, μ) combos
    """
    grid_results = []

    for lam, mu in product(lam_values, mu_values):
        cycle_results = []
        for cycle in cycles:
            try:
                cr = replay_cycle(
                    target_cycle=cycle,
                    race_type=race_type,
                    lam=lam,
                    mu=mu,
                )
                cycle_results.append(cr)
            except Exception as e:
                print(f"  Cycle {cycle} failed: {e}")
                continue

        if cycle_results:
            report = build_report(cycle_results)
            grid_results.append({
                "lam": lam,
                "mu": mu,
                "mean_rmse": report.mean_rmse,
                "mean_r": report.mean_correlation,
                "mean_bias": report.mean_bias,
                "mean_state_accuracy": report.mean_state_accuracy,
                "n_cycles": len(cycle_results),
            })

    # Find optimum
    best = min(grid_results, key=lambda x: x["mean_rmse"])

    result = {
        "best_lam": best["lam"],
        "best_mu": best["mu"],
        "best_rmse": best["mean_rmse"],
        "best_r": best["mean_r"],
        "best_state_accuracy": best["mean_state_accuracy"],
        "grid": grid_results,
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

    return result
```

- [ ] **Step 2: Write test**

```python
def test_grid_search_finds_optimum():
    """Grid search should return valid optimal params."""
    # This is a slow integration test — skip in CI
    result = grid_search(
        cycles=[2020],  # Single cycle for speed
        lam_values=[0.1, 1.0, 10.0],
        mu_values=[0.1, 1.0],
    )
    assert "best_lam" in result
    assert "best_mu" in result
    assert result["best_rmse"] > 0
    assert result["best_rmse"] < 0.20
```

- [ ] **Step 3: Commit**

---

### Task 6: Generate Backtest Report

**Files:**
- Create: script to run full backtest and output `docs/research/backtest-report.md`

- [ ] **Step 1: Run full backtest**

```bash
cd /home/hayden/projects/wethervane
uv run python -c "
from src.calibration.optimize_params import grid_search
from pathlib import Path
result = grid_search(
    cycles=[2012, 2016, 2020, 2024],
    race_type='president',
    output_path=Path('data/calibration/optimal_params.json'),
)
print(f'Best λ={result[\"best_lam\"]}, μ={result[\"best_mu\"]}')
print(f'Best RMSE={result[\"best_rmse\"]:.4f}, r={result[\"best_r\"]:.3f}')
print(f'State accuracy={result[\"best_state_accuracy\"]:.1%}')
"
```

- [ ] **Step 2: Write backtest report**

Generate `docs/research/backtest-report.md` with:
- Per-cycle RMSE, correlation, bias, state accuracy
- Optimal (λ, μ) and how sensitive results are to these params
- Worst-predicted states per cycle (systematic errors)
- Per-type error breakdown (which community types are hardest)
- Comparison: calibrated vs uncalibrated RMSE

- [ ] **Step 3: Commit**

```bash
git add data/calibration/ docs/research/backtest-report.md
git commit -m "feat: historical calibration — backtest report with optimal λ/μ"
```

---

## Validation Checklist

- [ ] 538 polls loaded into DuckDB for 2012, 2016, 2020, 2024
- [ ] Presidential county results loaded for 2012-2024
- [ ] Governor county results loaded for available cycles
- [ ] replay_cycle(2020) produces CycleResult with RMSE < 0.10
- [ ] Grid search converges to reasonable (λ, μ)
- [ ] Calibrated RMSE beats uncalibrated baseline
- [ ] Per-cycle metrics show consistent accuracy (no one cycle wildly off)
- [ ] State winner accuracy > 80% across cycles
- [ ] Backtest report written with actionable insights
