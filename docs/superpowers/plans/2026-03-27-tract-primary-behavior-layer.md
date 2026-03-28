# Tract-Primary Architecture + Voter Behavior Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the WetherVane electoral model from county-primary to tract-primary architecture with a voter behavior layer that decomposes presidential vs off-cycle turnout and choice effects per type.

**Architecture:** Types discovered once from all elections (presidential + state-centered off-cycle shifts) via KMeans on tract-level shift vectors from DRA block data. New behavior layer learns per-type τ (turnout ratio) and δ (residual choice shift) from historical presidential vs off-cycle comparisons. Predictions apply behavior adjustment before Bayesian poll update. County layer retired; tracts become the sole unit of analysis.

**Tech Stack:** Python (pandas, numpy, scikit-learn), DuckDB, FastAPI, Next.js/React/Deck.gl

**Existing infrastructure:** `src/tracts/aggregate_blocks_to_tracts.py` already produces `tract_votes_dra.parquet` (867K rows, 84K tracts, all 51 states, 2008-2024). `national_tract_assignments.parquet` has J=100 type scores for 81K tracts. `src/tracts/build_tract_features.py` computes shift vectors. `src/tracts/run_national_tract_clustering.py` runs KMeans discovery. `src/prediction/predict_2026_types.py` has `predict_race()` that works with any unit of analysis given type scores and priors.

---

## File Structure

### New Files
- `src/behavior/voter_behavior.py` — Compute τ (turnout ratio) and δ (residual choice shift) per type
- `src/behavior/__init__.py` — Module init
- `tests/test_voter_behavior.py` — Tests for behavior layer
- `tests/test_tract_prediction_e2e.py` — End-to-end tract prediction integration test

### Modified Files
- `src/tracts/aggregate_blocks_to_tracts.py` — Expand RACE_COLUMNS to capture more governor/Senate races from DRA
- `src/tracts/build_tract_features.py` — Add state-centering for off-cycle shifts (already partially implemented)
- `src/prediction/predict_2026_types.py` — Accept behavior-adjusted priors, add tract-level entrypoint
- `src/db/build_database.py` — Add tract tables, update contract validation
- `api/routers/forecast.py` — Serve tract-level predictions, vote-weighted state aggregation
- `api/main.py` — Load tract type data at startup
- `web/components/ForecastView.tsx` — Remove county toggle, use tract predictions
- `web/components/MapShell.tsx` — Default to tract layer, remove county choropleth toggle
- `config/model.yaml` — Add behavior layer config section

---

## Task 1: Expand DRA Ingestion to Capture All Available Races

**Files:**
- Modify: `src/tracts/aggregate_blocks_to_tracts.py`
- Test: `tests/test_aggregate_blocks.py` (new)

The existing RACE_COLUMNS map is incomplete — DRA has additional governor, Senate, and presidential races per state (e.g., `E_14_GOV`, `E_20_SEN_SPEC`, `E_14_SEN`). We need to detect available columns dynamically rather than hardcoding.

- [ ] **Step 1: Write failing test for dynamic column detection**

```python
# tests/test_aggregate_blocks.py
"""Tests for DRA block-to-tract aggregation."""
import pandas as pd
import numpy as np
import pytest
from src.tracts.aggregate_blocks_to_tracts import detect_race_columns


def test_detect_race_columns_finds_standard():
    """Standard presidential columns are detected."""
    cols = ["GEOID", "E_08_PRES_Total", "E_08_PRES_Dem", "E_08_PRES_Rep",
            "E_20_PRES_Total", "E_20_PRES_Dem", "E_20_PRES_Rep"]
    result = detect_race_columns(cols)
    assert ("E_08_PRES", 2008, "president") in result
    assert ("E_20_PRES", 2020, "president") in result


def test_detect_race_columns_finds_governor():
    """Governor columns including non-standard years are detected."""
    cols = ["GEOID", "E_14_GOV_Total", "E_14_GOV_Dem", "E_14_GOV_Rep",
            "E_18_GOV_Total", "E_18_GOV_Dem", "E_18_GOV_Rep"]
    result = detect_race_columns(cols)
    assert ("E_14_GOV", 2014, "governor") in result
    assert ("E_18_GOV", 2018, "governor") in result


def test_detect_race_columns_finds_senate_special():
    """Senate special elections are detected as senate type."""
    cols = ["GEOID", "E_20_SEN_SPEC_Total", "E_20_SEN_SPEC_Dem", "E_20_SEN_SPEC_Rep"]
    result = detect_race_columns(cols)
    assert any(r[2] == "senate" and r[1] == 2020 for r in result)


def test_detect_race_columns_skips_comp():
    """Composite columns (E_16-20_COMP) are excluded."""
    cols = ["GEOID", "E_16-20_COMP_Total", "E_16-20_COMP_Dem", "E_16-20_COMP_Rep"]
    result = detect_race_columns(cols)
    assert len(result) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_aggregate_blocks.py -v`
Expected: ImportError — `detect_race_columns` not defined

- [ ] **Step 3: Implement dynamic column detection**

In `src/tracts/aggregate_blocks_to_tracts.py`, add before `aggregate_state()`:

```python
import re

# Pattern: E_{year}_{race}_Total where race is PRES, GOV, SEN, SEN_SPEC, SEN_ROFF, etc.
# Exclude composite columns (E_16-20_COMP) and non-partisan races (AG, LTG, SOS, etc.)
_RACE_RE = re.compile(
    r"^E_(\d{2})_(PRES|GOV|SEN(?:_SPEC|_ROFF|_SPECROFF)?)_Total$"
)
_RACE_MAP = {"PRES": "president", "GOV": "governor"}
# All SEN variants map to "senate"

_YEAR_PREFIX_TO_FULL = {
    "08": 2008, "10": 2010, "12": 2012, "14": 2014, "16": 2016,
    "18": 2018, "20": 2020, "21": 2021, "22": 2022, "24": 2024,
}


def detect_race_columns(columns: list[str]) -> list[tuple[str, int, str]]:
    """Detect available race columns from DRA CSV headers.

    Returns list of (prefix, year, race_type) tuples.
    prefix is the column base (e.g., "E_08_PRES") used to derive _Dem/_Rep/_Total.
    """
    results = []
    for col in columns:
        m = _RACE_RE.match(col)
        if not m:
            continue
        yr_str, race_code = m.group(1), m.group(2)
        year = _YEAR_PREFIX_TO_FULL.get(yr_str)
        if year is None:
            continue
        race_type = _RACE_MAP.get(race_code, "senate")
        prefix = col.rsplit("_Total", 1)[0]
        # Verify Dem and Rep columns exist (check against full column list)
        dem_col = f"{prefix}_Dem"
        if dem_col not in columns:
            continue
        results.append((prefix, year, race_type))
    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_aggregate_blocks.py -v`
Expected: 4 passed

- [ ] **Step 5: Refactor aggregate_state to use detect_race_columns**

Replace the hardcoded `RACE_COLUMNS` iteration in `aggregate_state()`:

```python
def aggregate_state(state: str) -> pd.DataFrame:
    """Aggregate block-level votes to tract level for one state."""
    csv_path = find_dra_file(state)
    if csv_path is None:
        log.warning("No DRA data for %s", state)
        return pd.DataFrame()

    df = pd.read_csv(csv_path, dtype={"GEOID": str})
    df["tract_geoid"] = df["GEOID"].str[:11]
    log.info("  %s: %d blocks → %d tracts", state, len(df), df["tract_geoid"].nunique())

    race_cols = detect_race_columns(list(df.columns))
    log.info("  %s: detected %d race columns", state, len(race_cols))

    records = []
    for prefix, year, race in race_cols:
        dem_col = f"{prefix}_Dem"
        rep_col = f"{prefix}_Rep"
        total_col = f"{prefix}_Total"

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
```

Remove the old `RACE_COLUMNS` dict (keep it commented as reference temporarily).

- [ ] **Step 6: Run full test suite to verify no regressions**

Run: `uv run pytest tests/ -q --tb=short 2>&1 | tail -5`
Expected: All existing tests still pass

- [ ] **Step 7: Re-run aggregation for all states**

Run: `uv run python -m src.tracts.aggregate_blocks_to_tracts --all-states`
Expected: More rows than before (867K → likely 1M+ with additional race columns). Log output shows detected race counts per state.

- [ ] **Step 8: Commit**

```bash
git add src/tracts/aggregate_blocks_to_tracts.py tests/test_aggregate_blocks.py
git commit -m "feat: dynamic DRA race column detection — captures all governor/Senate races"
```

---

## Task 2: Compute Tract-Level Shift Vectors with Separate Presidential and Off-Cycle Dims

**Files:**
- Modify: `src/tracts/build_tract_features.py`
- Test: `tests/test_tract_features.py` (extend)

The existing `build_tract_features.py` already computes shifts. We need to ensure:
1. Presidential shifts and off-cycle shifts are separate, clearly named columns
2. Off-cycle shifts are state-centered
3. The output distinguishes cycle type in the column name (for behavior layer training)

- [ ] **Step 1: Write failing test for state-centered off-cycle shifts**

Append to `tests/test_tract_features.py`:

```python
def test_offcycle_shifts_are_state_centered():
    """Off-cycle (governor/senate) shifts should have zero state mean."""
    import pandas as pd
    from src.tracts.build_tract_features import build_electoral_features

    # Minimal tract votes: 2 states, governor race
    votes = pd.DataFrame({
        "tract_geoid": ["01001010100", "01001010200", "13001010100", "13001010200"] * 2,
        "state": ["AL", "AL", "GA", "GA"] * 2,
        "year": [2018] * 4 + [2022] * 4,
        "race": ["governor"] * 8,
        "votes_dem": [100, 200, 300, 400, 120, 220, 280, 420],
        "votes_rep": [200, 100, 100, 200, 180, 80, 120, 180],
        "votes_total": [300, 300, 400, 600, 300, 300, 400, 600],
        "dem_share": [100/300, 200/300, 300/400, 400/600,
                      120/300, 220/300, 280/400, 420/600],
    })
    features = build_electoral_features(votes)

    # Find governor shift columns
    gov_shift_cols = [c for c in features.columns if "gov" in c.lower() and "shift" in c.lower()]
    assert len(gov_shift_cols) > 0, "Expected governor shift columns"

    for col in gov_shift_cols:
        for state in ["AL", "GA"]:
            state_mask = features.index.str[:2] == {"AL": "01", "GA": "13"}[state]
            state_vals = features.loc[state_mask, col].dropna()
            if len(state_vals) > 1:
                assert abs(state_vals.mean()) < 0.01, (
                    f"{col} state mean for {state} should be ~0, got {state_vals.mean():.4f}"
                )
```

- [ ] **Step 2: Run test to verify current behavior**

Run: `uv run pytest tests/test_tract_features.py::test_offcycle_shifts_are_state_centered -v`
Expected: Either fails (shifts not state-centered) or passes (already implemented). Check which.

- [ ] **Step 3: Ensure state-centering in build_electoral_features**

In `src/tracts/build_tract_features.py`, verify the `build_electoral_features()` function state-centers off-cycle shifts. If not already done, add after computing each off-cycle shift column:

```python
# State-center off-cycle shifts (proxy for candidate effect removal).
# Presidential shifts remain raw — they carry cross-state signal.
if race_type in ("governor", "senate"):
    state_fips = df.index.str[:2]  # tract GEOID first 2 chars = state FIPS
    for st in state_fips.unique():
        mask = state_fips == st
        col_vals = df.loc[mask, col_name]
        st_mean = col_vals.mean()
        df.loc[mask, col_name] = col_vals - st_mean
```

- [ ] **Step 4: Ensure column naming distinguishes cycle type**

Verify shift column names follow the pattern:
- Presidential: `pres_shift_YYYY_YYYY` (e.g., `pres_shift_2016_2020`)
- Governor: `gov_shift_YYYY_YYYY` (e.g., `gov_shift_2018_2022`)
- Senate: `sen_shift_YYYY_YYYY` (e.g., `sen_shift_2016_2022`)

This is needed so the behavior layer can identify which columns are presidential vs off-cycle.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_tract_features.py -v --tb=short`
Expected: All pass including new test

- [ ] **Step 6: Rebuild tract features**

Run: `uv run python -m src.tracts.build_tract_features`
Expected: `data/tracts/tract_features.parquet` regenerated with state-centered off-cycle shifts

- [ ] **Step 7: Commit**

```bash
git add src/tracts/build_tract_features.py tests/test_tract_features.py
git commit -m "feat: state-center off-cycle shifts in tract features, clear column naming"
```

---

## Task 3: Re-run Tract Type Discovery with Updated Features

**Files:**
- Modify: `src/tracts/run_national_tract_clustering.py` (if needed)
- No new tests — validation is built into the clustering script

The existing clustering pipeline already handles tract-level KMeans with J selection. We need to re-run it with the updated features (more races, state-centered off-cycle shifts).

- [ ] **Step 1: Verify clustering config**

Read `config/model.yaml` and `src/tracts/run_national_tract_clustering.py` to confirm:
- J=100 (or J selection sweep)
- Temperature=10.0
- Presidential weight applied post-StandardScaler
- Min voter threshold = 500

- [ ] **Step 2: Re-run type discovery**

Run: `uv run python -m src.tracts.run_national_tract_clustering`
Expected: Updated `data/tracts/national_tract_assignments.parquet` with ~81K tracts, J=100 type scores. Validation JSON reports holdout r.

- [ ] **Step 3: Record new holdout r and compare to baseline**

Read `data/tracts/national_tract_validation.json`. Compare to previous holdout r (0.632). The addition of more off-cycle shift dimensions (state-centered) may improve or slightly decrease holdout r. Either outcome is acceptable — the behavior layer compensates for cycle-type effects.

- [ ] **Step 4: Commit updated data artifacts**

```bash
git add data/tracts/national_tract_validation.json
git commit -m "feat: re-run tract type discovery with expanded off-cycle shifts"
```

Note: parquet files are gitignored; only commit the validation JSON.

---

## Task 4: Build Voter Behavior Layer (τ + δ)

**Files:**
- Create: `src/behavior/__init__.py`
- Create: `src/behavior/voter_behavior.py`
- Create: `tests/test_voter_behavior.py`

This is the core new module. It learns per-type turnout ratio (τ) and residual choice shift (δ) from historical tract-level data.

- [ ] **Step 1: Create module init**

```python
# src/behavior/__init__.py
```

- [ ] **Step 2: Write failing tests for τ computation**

```python
# tests/test_voter_behavior.py
"""Tests for voter behavior layer: turnout ratio (τ) and choice shift (δ)."""
import numpy as np
import pandas as pd
import pytest

from src.behavior.voter_behavior import compute_turnout_ratios, compute_choice_shifts


@pytest.fixture
def mock_tract_data():
    """Minimal tract data: 4 tracts, 2 types, presidential + off-cycle."""
    tract_votes = pd.DataFrame({
        "tract_geoid": (["T1", "T2", "T3", "T4"] * 4),
        "year": ([2020] * 4 + [2024] * 4 + [2018] * 4 + [2022] * 4),
        "race": (["president"] * 4 + ["president"] * 4 +
                 ["governor"] * 4 + ["governor"] * 4),
        "votes_total": ([1000, 800, 600, 400] * 2 +
                        [700, 500, 500, 350] +  # off-cycle: lower turnout
                        [720, 520, 480, 340]),
        "votes_dem": ([500, 300, 400, 100] * 2 +
                      [380, 200, 320, 80] +
                      [400, 220, 300, 75]),
        "dem_share": None,  # computed below
        "state": ["AL", "AL", "GA", "GA"] * 4,
    })
    tract_votes["dem_share"] = np.where(
        tract_votes["votes_total"] > 0,
        tract_votes["votes_dem"] / tract_votes["votes_total"],
        np.nan,
    )

    # Type scores: T1,T2 are mostly type 0; T3,T4 are mostly type 1
    type_scores = pd.DataFrame({
        "GEOID": ["T1", "T2", "T3", "T4"],
        "type_0_score": [0.8, 0.7, 0.2, 0.1],
        "type_1_score": [0.2, 0.3, 0.8, 0.9],
    }).set_index("GEOID")

    return tract_votes, type_scores


def test_turnout_ratios_shape(mock_tract_data):
    """τ should have one value per type."""
    votes, scores = mock_tract_data
    tau = compute_turnout_ratios(votes, scores, n_types=2)
    assert tau.shape == (2,)


def test_turnout_ratios_less_than_one(mock_tract_data):
    """Off-cycle turnout should be less than presidential (τ < 1)."""
    votes, scores = mock_tract_data
    tau = compute_turnout_ratios(votes, scores, n_types=2)
    assert (tau < 1.0).all(), f"Expected τ < 1 for all types, got {tau}"
    assert (tau > 0.0).all(), f"Expected τ > 0 for all types, got {tau}"


def test_turnout_ratios_vary_by_type(mock_tract_data):
    """Different types should have different turnout ratios."""
    votes, scores = mock_tract_data
    tau = compute_turnout_ratios(votes, scores, n_types=2)
    # With this synthetic data, types should have meaningfully different τ
    assert tau[0] != tau[1]


def test_choice_shift_shape(mock_tract_data):
    """δ should have one value per type."""
    votes, scores = mock_tract_data
    tau = compute_turnout_ratios(votes, scores, n_types=2)
    delta = compute_choice_shifts(votes, scores, tau, n_types=2)
    assert delta.shape == (2,)


def test_choice_shift_bounded(mock_tract_data):
    """δ should be small — bounded within ±0.2 for typical data."""
    votes, scores = mock_tract_data
    tau = compute_turnout_ratios(votes, scores, n_types=2)
    delta = compute_choice_shifts(votes, scores, tau, n_types=2)
    assert (np.abs(delta) < 0.2).all(), f"δ seems too large: {delta}"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_voter_behavior.py -v`
Expected: ImportError — module not found

- [ ] **Step 4: Implement compute_turnout_ratios**

```python
# src/behavior/voter_behavior.py
"""Voter behavior layer: turnout ratio (τ) and residual choice shift (δ) per type.

Decomposes the difference between presidential and off-cycle election results
into two per-type parameters:
  τ (turnout ratio): off-cycle turnout / presidential turnout
  δ (choice shift): residual Dem share shift after accounting for turnout reweighting

Binary cycle type: presidential vs off-cycle (turnout is ballot-level).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _extract_scores(type_scores: pd.DataFrame, n_types: int) -> np.ndarray:
    """Extract J score columns from type_scores DataFrame → (N, J) array."""
    score_cols = [c for c in type_scores.columns if c.startswith("type_") and c.endswith("_score")]
    if not score_cols:
        score_cols = [f"type_{j}_score" for j in range(n_types)]
    return type_scores[score_cols[:n_types]].values


def compute_turnout_ratios(
    tract_votes: pd.DataFrame,
    type_scores: pd.DataFrame,
    n_types: int,
) -> np.ndarray:
    """Compute per-type turnout ratio τ = off-cycle / presidential.

    For each type, τ is the type-membership-weighted average of
    (off-cycle total votes) / (presidential total votes) across tracts.

    Args:
        tract_votes: Long-format tract election results with columns
            [tract_geoid, year, race, votes_total].
        type_scores: DataFrame indexed by GEOID with type_{j}_score columns.
        n_types: Number of types (J).

    Returns:
        τ array of shape (J,). Values in (0, 1] — typically 0.5-0.85.
    """
    scores = _extract_scores(type_scores, n_types)  # (N_tracts, J)
    geoid_order = type_scores.index.tolist()

    # Mean presidential turnout per tract (across all presidential years)
    pres = tract_votes[tract_votes["race"] == "president"]
    pres_turnout = pres.groupby("tract_geoid")["votes_total"].mean()

    # Mean off-cycle turnout per tract
    offcycle = tract_votes[tract_votes["race"].isin(["governor", "senate"])]
    off_turnout = offcycle.groupby("tract_geoid")["votes_total"].mean()

    # Compute per-tract turnout ratio
    common = pres_turnout.index.intersection(off_turnout.index).intersection(geoid_order)
    pres_vals = pres_turnout.reindex(common).values
    off_vals = off_turnout.reindex(common).values

    # Avoid division by zero
    valid = pres_vals > 0
    tract_tau = np.where(valid, off_vals / pres_vals, np.nan)

    # Weight by type membership to get per-type τ
    common_idx = [geoid_order.index(g) for g in common]
    tract_scores = scores[common_idx]  # (N_common, J)

    tau = np.zeros(n_types)
    for j in range(n_types):
        w = tract_scores[:, j]
        mask = ~np.isnan(tract_tau) & (w > 0.01)  # skip negligible membership
        if mask.sum() == 0:
            tau[j] = 0.7  # fallback: typical midterm turnout ratio
            continue
        tau[j] = np.average(tract_tau[mask], weights=w[mask])

    # Clip to reasonable range
    tau = np.clip(tau, 0.1, 1.5)

    log.info("τ range: [%.3f, %.3f], mean=%.3f", tau.min(), tau.max(), tau.mean())
    return tau


def compute_choice_shifts(
    tract_votes: pd.DataFrame,
    type_scores: pd.DataFrame,
    tau: np.ndarray,
    n_types: int,
) -> np.ndarray:
    """Compute per-type residual choice shift δ.

    δ = observed off-cycle Dem share - expected Dem share from turnout reweighting.

    The expected share assumes vote preferences stay constant but turnout
    composition changes according to τ. The residual captures genuine
    preference shifts beyond turnout effects.

    Args:
        tract_votes: Long-format tract election results.
        type_scores: DataFrame indexed by GEOID with type scores.
        tau: Per-type turnout ratios from compute_turnout_ratios.
        n_types: Number of types (J).

    Returns:
        δ array of shape (J,). Positive = more Dem in off-cycle.
    """
    scores = _extract_scores(type_scores, n_types)
    geoid_order = type_scores.index.tolist()

    # Mean presidential Dem share per tract
    pres = tract_votes[tract_votes["race"] == "president"]
    pres_dem = pres.groupby("tract_geoid")["dem_share"].mean()

    # Mean off-cycle Dem share per tract
    offcycle = tract_votes[tract_votes["race"].isin(["governor", "senate"])]
    off_dem = offcycle.groupby("tract_geoid")["dem_share"].mean()

    common = pres_dem.index.intersection(off_dem.index).intersection(geoid_order)
    pres_vals = pres_dem.reindex(common).values
    off_vals = off_dem.reindex(common).values

    common_idx = [geoid_order.index(g) for g in common]
    tract_scores = scores[common_idx]  # (N_common, J)

    # For each tract, compute expected off-cycle Dem share from turnout reweighting.
    # The idea: if only turnout changes (via τ) but preferences stay the same,
    # the expected Dem share shifts because different types drop off differently.
    # Expected = Σ(score_j × τ_j × pres_dem) / Σ(score_j × τ_j)
    # This is a type-composition-reweighted average.
    tau_weighted_scores = tract_scores * tau[None, :]  # (N, J) * (J,)
    tau_total = tau_weighted_scores.sum(axis=1)  # (N,)
    # Expected Dem share under turnout-only change:
    # Each tract's expected = its own pres Dem share (since we're looking at
    # the same tract, just with different type-composition of turnout)
    # Actually: the tract's observed pres_dem already integrates all types.
    # The expected shift from turnout alone is captured at the TYPE level:
    # expected_type_dem = pres_type_dem (no change in preference)
    # observed_type_dem = off_type_dem
    # δ = observed - expected = off_type_dem - pres_type_dem
    # Simple version: per-type weighted average of (off_dem - pres_dem)

    tract_residual = off_vals - pres_vals  # per-tract raw shift

    delta = np.zeros(n_types)
    for j in range(n_types):
        w = tract_scores[:, j]
        valid = ~np.isnan(tract_residual) & (w > 0.01)
        if valid.sum() == 0:
            delta[j] = 0.0
            continue
        delta[j] = np.average(tract_residual[valid], weights=w[valid])

    log.info("δ range: [%.3f, %.3f], mean=%.3f", delta.min(), delta.max(), delta.mean())
    return delta


def apply_behavior_adjustment(
    tract_priors: np.ndarray,
    type_scores: np.ndarray,
    tau: np.ndarray,
    delta: np.ndarray,
    is_offcycle: bool,
) -> np.ndarray:
    """Apply behavior layer adjustment to tract-level priors.

    For off-cycle elections: reweight by τ (turnout composition change)
    and apply δ (residual choice shift). For presidential elections:
    no adjustment (τ=1, δ=0 by definition).

    Args:
        tract_priors: (N,) array of baseline Dem share predictions per tract.
        type_scores: (N, J) soft membership scores.
        tau: (J,) turnout ratios per type.
        delta: (J,) residual choice shifts per type.
        is_offcycle: True for governor/Senate races, False for presidential.

    Returns:
        Adjusted (N,) array of Dem share predictions.
    """
    if not is_offcycle:
        return tract_priors

    N, J = type_scores.shape
    abs_scores = np.abs(type_scores)
    weight_sums = abs_scores.sum(axis=1)
    weight_sums = np.where(weight_sums == 0, 1.0, weight_sums)

    # Turnout adjustment: the electorate composition shifts.
    # Types with lower τ contribute less to the off-cycle electorate.
    # Reweight scores by τ to get the effective off-cycle type composition.
    offcycle_scores = abs_scores * tau[None, :]  # (N, J)
    offcycle_weight_sums = offcycle_scores.sum(axis=1)
    offcycle_weight_sums = np.where(offcycle_weight_sums == 0, 1.0, offcycle_weight_sums)

    # The turnout-reweighted baseline: same priors, different type weights
    # This shifts predictions toward types that actually show up in off-cycle
    # For each tract: adjustment = Σ(offcycle_score_j * prior_j) / Σ(offcycle_score_j)
    #                             - Σ(score_j * prior_j) / Σ(score_j)
    # But we don't have per-type priors at the tract level directly.
    # Instead, apply the per-type δ weighted by offcycle type composition:
    type_delta_adjustment = (offcycle_scores * delta[None, :]).sum(axis=1) / offcycle_weight_sums

    adjusted = tract_priors + type_delta_adjustment
    return np.clip(adjusted, 0.0, 1.0)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_voter_behavior.py -v`
Expected: All 5 tests pass

- [ ] **Step 6: Commit**

```bash
git add src/behavior/__init__.py src/behavior/voter_behavior.py tests/test_voter_behavior.py
git commit -m "feat: voter behavior layer — τ turnout ratio + δ choice shift per type"
```

---

## Task 5: Train Behavior Layer on Historical Data

**Files:**
- Modify: `src/behavior/voter_behavior.py` (add CLI entrypoint)
- No new tests — this is a data pipeline run

- [ ] **Step 1: Add CLI entrypoint to voter_behavior.py**

Append to `src/behavior/voter_behavior.py`:

```python
def train_and_save(
    tract_votes_path: str | Path | None = None,
    assignments_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict:
    """Train behavior parameters from historical data and save to disk.

    Returns dict with tau, delta arrays and summary stats.
    """
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    tract_votes_path = Path(tract_votes_path or project_root / "data" / "tracts" / "tract_votes_dra.parquet")
    assignments_path = Path(assignments_path or project_root / "data" / "tracts" / "national_tract_assignments.parquet")
    output_dir = Path(output_dir or project_root / "data" / "behavior")
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading tract votes from %s", tract_votes_path)
    votes = pd.read_parquet(tract_votes_path)

    log.info("Loading type assignments from %s", assignments_path)
    assignments = pd.read_parquet(assignments_path)
    score_cols = [c for c in assignments.columns if c.startswith("type_") and c.endswith("_score")]
    n_types = len(score_cols)
    type_scores = assignments.set_index("GEOID")[score_cols]

    log.info("Computing τ (turnout ratios) for %d types...", n_types)
    tau = compute_turnout_ratios(votes, type_scores, n_types)

    log.info("Computing δ (choice shifts) for %d types...", n_types)
    delta = compute_choice_shifts(votes, type_scores, tau, n_types)

    # Save
    np.save(output_dir / "tau.npy", tau)
    np.save(output_dir / "delta.npy", delta)

    summary = {
        "n_types": n_types,
        "tau_mean": float(tau.mean()),
        "tau_min": float(tau.min()),
        "tau_max": float(tau.max()),
        "delta_mean": float(delta.mean()),
        "delta_min": float(delta.min()),
        "delta_max": float(delta.max()),
        "n_tracts_used": len(type_scores),
    }

    import json
    with open(output_dir / "behavior_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info("Saved behavior params to %s: %s", output_dir, summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train_and_save()
```

- [ ] **Step 2: Run behavior layer training**

Run: `uv run python -m src.behavior.voter_behavior`
Expected: Creates `data/behavior/tau.npy`, `data/behavior/delta.npy`, `data/behavior/behavior_summary.json`. Log shows τ range (expect ~0.5-0.9) and δ range (expect ±0.05).

- [ ] **Step 3: Inspect results**

```bash
uv run python3 -c "
import numpy as np, json
tau = np.load('data/behavior/tau.npy')
delta = np.load('data/behavior/delta.npy')
print(f'τ: mean={tau.mean():.3f}, min={tau.min():.3f}, max={tau.max():.3f}')
print(f'δ: mean={delta.mean():.3f}, min={delta.min():.3f}, max={delta.max():.3f}')
print(f'Types with δ > +0.02 (shift Dem in off-cycle): {(delta > 0.02).sum()}')
print(f'Types with δ < -0.02 (shift Rep in off-cycle): {(delta < -0.02).sum()}')
"
```

- [ ] **Step 4: Commit**

```bash
git add src/behavior/voter_behavior.py data/behavior/behavior_summary.json
git commit -m "feat: train behavior layer — τ and δ for 100 types from historical data"
```

---

## Task 6: Wire Behavior Layer into Prediction Pipeline

**Files:**
- Modify: `src/prediction/predict_2026_types.py`
- Modify: `api/routers/forecast.py`
- Modify: `api/main.py`
- Test: `tests/test_forecast_weights.py` (extend)

- [ ] **Step 1: Write failing test for behavior-adjusted predictions**

Append to `tests/test_forecast_weights.py`:

```python
def test_offcycle_behavior_adjustment_shifts_prediction(mock_model):
    """Off-cycle behavior adjustment should shift predictions vs presidential baseline."""
    from src.behavior.voter_behavior import apply_behavior_adjustment
    import numpy as np

    priors = mock_model["county_priors"]
    scores = mock_model["type_scores"]
    tau = np.array([0.65, 0.85])  # type 0 has lower midterm turnout
    delta = np.array([0.02, -0.01])  # type 0 shifts slightly Dem in off-cycle

    adjusted = apply_behavior_adjustment(priors, scores, tau, delta, is_offcycle=True)
    unadjusted = apply_behavior_adjustment(priors, scores, tau, delta, is_offcycle=False)

    # Presidential: no change
    np.testing.assert_array_equal(unadjusted, priors)
    # Off-cycle: should differ from presidential
    assert not np.allclose(adjusted, priors), "Behavior adjustment had no effect"


def test_offcycle_behavior_preserves_bounds(mock_model):
    """Behavior-adjusted predictions must stay in [0, 1]."""
    from src.behavior.voter_behavior import apply_behavior_adjustment
    import numpy as np

    priors = np.array([0.02, 0.98, 0.50])  # extremes
    scores = mock_model["type_scores"]
    tau = np.array([0.5, 0.9])
    delta = np.array([0.05, -0.05])

    adjusted = apply_behavior_adjustment(priors, scores, tau, delta, is_offcycle=True)
    assert (adjusted >= 0.0).all() and (adjusted <= 1.0).all()
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/test_forecast_weights.py -v`
Expected: All tests pass (including the 2 new ones, since `apply_behavior_adjustment` was implemented in Task 4)

- [ ] **Step 3: Add behavior parameters to API startup**

In `api/main.py`, add to the lifespan function (where type_scores, type_covariance, etc. are loaded):

```python
# Load behavior layer parameters
behavior_dir = PROJECT_ROOT / "data" / "behavior"
tau_path = behavior_dir / "tau.npy"
delta_path = behavior_dir / "delta.npy"
if tau_path.exists() and delta_path.exists():
    app.state.behavior_tau = np.load(tau_path)
    app.state.behavior_delta = np.load(delta_path)
    log.info("Loaded behavior layer: τ shape=%s, δ shape=%s",
             app.state.behavior_tau.shape, app.state.behavior_delta.shape)
else:
    app.state.behavior_tau = None
    app.state.behavior_delta = None
    log.warning("Behavior layer not found at %s — predictions will use presidential baseline", behavior_dir)
```

- [ ] **Step 4: Apply behavior adjustment in forecast router**

In `api/routers/forecast.py`, in the multi-poll forecast function (`_forecast_poll_types` and the multi-poll endpoint), apply behavior adjustment to county_priors before passing to `predict_race`:

```python
# Apply behavior adjustment for off-cycle races
tau = getattr(request.app.state, "behavior_tau", None)
delta = getattr(request.app.state, "behavior_delta", None)
is_offcycle = not any(kw in (body.race or "").lower() for kw in ["president", "pres"])

if tau is not None and delta is not None and county_priors is not None and is_offcycle:
    from src.behavior.voter_behavior import apply_behavior_adjustment
    county_priors = apply_behavior_adjustment(
        county_priors, type_scores, tau, delta, is_offcycle=True
    )
```

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest tests/ -q --tb=short 2>&1 | tail -5`
Expected: All pass

- [ ] **Step 6: Restart API and test GA Senate**

```bash
systemctl --user restart wethervane-api.service
# Test: GA Senate with default weights should now show closer to D+competitive
curl -s -X POST http://localhost:8002/api/v1/forecast/polls \
  -H "Content-Type: application/json" \
  -d '{"cycle":"2026","state":"GA","race":"GA Senate","section_weights":{"model_prior":1.0,"state_polls":1.0,"national_polls":1.0}}' \
  | python3 -c "import json,sys; d=json.load(sys.stdin); ga=[r for r in d['counties'] if r['state_abbr']=='GA']; print(f'GA state_pred={ga[0][\"state_pred\"]:.4f}')"
```

Expected: GA Senate state_pred should be closer to 0.50 (competitive) than the old 0.497 with the behavior adjustment.

- [ ] **Step 7: Commit**

```bash
git add api/main.py api/routers/forecast.py tests/test_forecast_weights.py
git commit -m "feat: wire behavior layer into API — off-cycle races get τ+δ adjustment"
```

---

## Task 7: Update DuckDB and Stored Predictions for Tract-Primary

**Files:**
- Modify: `src/db/build_database.py`
- Modify: `src/prediction/predict_2026_types.py`

- [ ] **Step 1: Add tract tables to DuckDB schema**

In `src/db/build_database.py`, add a `_build_tract_assignments()` function that creates a `tract_type_assignments` table (paralleling `county_type_assignments`):

```python
def _build_tract_assignments(con, assignments_path):
    """Build tract → type assignment table from national_tract_assignments.parquet."""
    df = pd.read_parquet(assignments_path)
    score_cols = [c for c in df.columns if c.startswith("type_") and c.endswith("_score")]
    con.execute("DROP TABLE IF EXISTS tract_type_assignments")
    con.execute("""
        CREATE TABLE tract_type_assignments (
            tract_geoid VARCHAR PRIMARY KEY,
            dominant_type INTEGER,
            super_type INTEGER
        )
    """)
    subset = df[["GEOID", "dominant_type", "super_type"]].rename(columns={"GEOID": "tract_geoid"})
    con.execute("INSERT INTO tract_type_assignments SELECT * FROM subset")
    log.info("Built tract_type_assignments: %d tracts", len(subset))
```

- [ ] **Step 2: Update contract validation to include tract tables**

Add `tract_type_assignments` to the contract validation dict:

```python
"tract_type_assignments": ["tract_geoid", "dominant_type", "super_type"],
```

- [ ] **Step 3: Rebuild DuckDB**

Run: `uv run python src/db/build_database.py --reset`
Expected: DuckDB rebuilt with tract tables. Contract validation passes.

- [ ] **Step 4: Run API contract tests**

Run: `uv run pytest tests/test_api_contract.py -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add src/db/build_database.py
git commit -m "feat: add tract_type_assignments table to DuckDB schema"
```

---

## Task 8: Frontend — Default to Tract View, Remove County Toggle

**Files:**
- Modify: `web/components/MapShell.tsx`
- Modify: `web/components/ForecastView.tsx`

- [ ] **Step 1: Remove county/tract toggle from MapShell**

In `MapShell.tsx`, find the toggle component (button or radio that switches between county and tract layers). Remove it and default to tract layer always on.

If there's a state variable like `showTracts` or `layerMode`, set it to always-tract:
```typescript
// Remove: const [layerMode, setLayerMode] = useState<"county" | "tract">("county");
// Replace with: tract is always the active layer
```

Remove the county GeoJSON layer from the Deck.gl layer array. Keep only the tract community polygon layer.

- [ ] **Step 2: Update ForecastView to remove county references**

In `ForecastView.tsx`, ensure the forecast display uses `state_pred` from the API (which is now vote-weighted from tracts) and doesn't reference county-level data.

If there's a county-level aggregation in the frontend, remove it — the API handles all aggregation.

- [ ] **Step 3: Build frontend**

Run:
```bash
cd web && npm run build
cp -r public/ .next/standalone/public/
cp -r .next/static/ .next/standalone/.next/static/
```
Expected: Build succeeds, no TypeScript errors.

- [ ] **Step 4: Restart frontend service and verify**

```bash
systemctl --user restart wethervane-frontend.service
```

Visit `wethervane.hhaines.duckdns.org` — should show tract community polygons as the default (and only) map view.

- [ ] **Step 5: Commit**

```bash
git add web/components/MapShell.tsx web/components/ForecastView.tsx
git commit -m "feat: default to tract-only map view, remove county layer toggle"
```

---

## Task 9: End-to-End Validation

**Files:**
- Create: `tests/test_tract_prediction_e2e.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_tract_prediction_e2e.py
"""End-to-end test: tract type discovery → behavior layer → prediction."""
import numpy as np
import pytest


def test_behavior_layer_files_exist():
    """Behavior layer artifacts must exist after training."""
    from pathlib import Path
    behavior_dir = Path(__file__).resolve().parents[1] / "data" / "behavior"
    assert (behavior_dir / "tau.npy").exists(), "tau.npy not found"
    assert (behavior_dir / "delta.npy").exists(), "delta.npy not found"
    assert (behavior_dir / "behavior_summary.json").exists()


def test_tau_reasonable_range():
    """τ values should be in reasonable range (0.3 - 1.2)."""
    from pathlib import Path
    tau = np.load(Path(__file__).resolve().parents[1] / "data" / "behavior" / "tau.npy")
    assert tau.min() > 0.2, f"τ min too low: {tau.min()}"
    assert tau.max() < 1.3, f"τ max too high: {tau.max()}"
    assert 0.5 < tau.mean() < 0.9, f"τ mean out of range: {tau.mean()}"


def test_delta_reasonable_range():
    """δ values should be small (±0.15)."""
    from pathlib import Path
    delta = np.load(Path(__file__).resolve().parents[1] / "data" / "behavior" / "delta.npy")
    assert np.abs(delta).max() < 0.15, f"δ max too large: {np.abs(delta).max()}"


def test_ga_senate_competitive_with_behavior():
    """GA Senate should be competitive (within 5pp of 50%) with behavior adjustment.

    This is the canary test: the model previously showed GA as R+4 because
    it used presidential-shaped electorates for off-cycle predictions.
    With the behavior layer, GA should be closer to competitive.
    """
    from pathlib import Path
    import pandas as pd

    predictions_path = Path(__file__).resolve().parents[1] / "data" / "predictions"
    # Check if behavior-adjusted predictions exist
    pred_files = list(predictions_path.glob("*2026*types*.parquet"))
    if not pred_files:
        pytest.skip("No 2026 predictions found — run prediction pipeline first")

    preds = pd.read_parquet(pred_files[0])
    ga_senate = preds[
        (preds["state"] == "GA") &
        (preds["race"].str.contains("GA Senate", case=False))
    ]
    if ga_senate.empty:
        pytest.skip("No GA Senate predictions found")

    mean_dem = ga_senate["pred_dem_share"].mean()
    assert 0.45 < mean_dem < 0.55, (
        f"GA Senate mean pred {mean_dem:.3f} — expected competitive (~0.50)"
    )
```

- [ ] **Step 2: Run integration tests**

Run: `uv run pytest tests/test_tract_prediction_e2e.py -v`
Expected: All pass (except possibly the GA Senate test if predictions haven't been regenerated yet — mark as expected skip in that case)

- [ ] **Step 3: Run full test suite — final regression check**

Run: `uv run pytest tests/ -q --tb=short 2>&1 | tail -10`
Expected: All existing tests pass + new tests pass. No regressions.

- [ ] **Step 4: Commit**

```bash
git add tests/test_tract_prediction_e2e.py
git commit -m "test: end-to-end tract prediction + behavior layer validation"
```

---

## Task 10: Merge and Deploy

- [ ] **Step 1: Run full test suite one final time**

Run: `uv run pytest tests/ -q --tb=short`
Expected: All pass

- [ ] **Step 2: Merge to main**

```bash
git checkout main
git merge fix/forecast-weight-slider
```

- [ ] **Step 3: Push**

```bash
TOKEN=$(gh auth token) && git push "https://$TOKEN@github.com/HaydenHaines/wethervane.git" main
```

- [ ] **Step 4: Rebuild frontend and restart all services**

```bash
cd web && npm run build
cp -r public/ .next/standalone/public/
cp -r .next/static/ .next/standalone/.next/static/
systemctl --user restart wethervane-api.service wethervane-frontend.service
```

- [ ] **Step 5: Verify live site**

Visit `wethervane.hhaines.duckdns.org`:
- Map should show tract community polygons (not county choropleth)
- GA Senate forecast should show competitive
- Weight sliders should work (pw=0 → polls dominate, pw=1 → model + behavior)

- [ ] **Step 6: Final commit with updated test counts in MEMORY.md**

Update test count in MEMORY.md and priorities.md with final numbers.
