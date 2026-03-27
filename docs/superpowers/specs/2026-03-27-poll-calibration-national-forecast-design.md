# Poll Calibration & National Forecast Redesign

**Date:** 2026-03-27
**Session:** S243
**Status:** Design approved, pending implementation plan

---

## Problem Statement

The current poll integration pipeline is hand-designed and uncalibrated. It uses state-mean W vectors (losing geographic nuance), has broken house effect correction, no historical validation, and only covers 4 races in FL/GA/AL. The forecast needs to expand to all ~506 2026 races (Senate, Governor, House) and the poll integration needs to be learned from historical data rather than assumed.

## Design Overview

Three pillars:

1. **Hierarchical decomposition** — Decompose poll signal into national citizen sentiment (θ_national) and per-race candidate effects (δ_race)
2. **National forecast expansion** — Every 2026 Senate, Governor, and House race gets a forecast
3. **Historical calibration** — Learn model parameters (λ, μ) from 2012-2024 backtesting

## Core Principles

**θ is the fundamental inference target.** Type means θ are what the model estimates. Counties, states, and districts are downstream products of θ, not primary objects of inference. Polls are observations of W·θ — a poll tells us about the type composition of the polled geography.

**Voters move slowly; polls move quickly.** The base model captures structural voter state — where communities sit politically, informed by a decade of elections and deep demographic signal. This changes slowly. Polls capture momentum and pull — who appears to be moving right now. After an election cycle, the base tract model updates to reflect who actually moved.

**Polls are reductive; we dissect them into usable parts.** Polls consistently under- or overestimate movement because they sample imperfectly and compress complex type structure into a single number. The θ_national + δ_race decomposition is how we extract the usable signal: what is the national political environment doing to each type (θ_national), and what is the specific candidate matchup doing beyond that (δ_race)? The calibration system learns how much to trust each component from historical data where we know the answer.

---

## 1. Data Model

### Type-Level Objects

```
θ_prior[j]     — Model's pre-poll estimate for type j's Dem lean this cycle.
                  Source: Ridge+HGB ensemble on historical shifts + demographics.
                  Computed as vote-weighted mean of county priors by type membership:
                  θ_prior[j] = Σ_c W[c,j] · county_prior[c] / Σ_c W[c,j]

θ_national[j]  — Citizen sentiment estimate for type j.
                  Learned from ALL polls across ALL races this cycle.
                  Represents: "what would type j do in a generic D-vs-R matchup
                  given the current national political environment?"

δ_race[j]      — Candidate effect for a specific race on type j.
                  Learned from that race's polls as residual from θ_national.
                  Represents: "how does this specific matchup deviate from
                  generic sentiment for type j?"

θ_forecast[j]  — Final type-level forecast.
                  National environment mode: θ_national[j]
                  Local polling mode: θ_national[j] + δ_race[j]
```

### Geographic Predictions (Downstream)

```
county_pred[c]   = Σ_j W[c,j] · θ_forecast[j]
district_pred[d] = Σ_j W_district[d,j] · θ_forecast[j]
state_pred[s]    = Σ_{c∈s} votes[c] · county_pred[c] / Σ_{c∈s} votes[c]
```

### Poll Observations

Each poll p provides:
- `y_p` — observed Dem share (house-effect corrected, quality-weighted)
- `W_p[j]` — type composition of the polled geography (state or district)
- `σ_p` — poll noise (from sample size + quality weighting)
- `race_p` — which race the poll is for

Model: `y_p ≈ W_p · (θ_national + δ_race_p)`

### User-Facing Toggle

- **"National Environment"** mode: `θ_forecast = θ_national` (no race-specific δ)
- **"Local Polling"** mode: `θ_forecast = θ_national + δ_race` (for races with polls)

Every race gets a national environment forecast. Only polled races additionally get a local polling forecast.

---

## 2. Estimation Procedure

### Step 1: Compute θ_prior (No Polls)

Use existing Ridge+HGB ensemble county priors. Convert to type-level:

```
θ_prior[j] = Σ_c W[c,j] · county_prior[c] / Σ_c W[c,j]
```

### Step 2: Estimate θ_national (All Polls, All Races)

Pool every poll across every race this cycle. Solve via regularized regression:

```
minimize Σ_p (1/σ_p²) · (y_p - W_p · θ_national)² + λ · ||θ_national - θ_prior||²
```

- Regularization pulls θ_national toward model prior
- Few polls → θ_national ≈ θ_prior
- Many polls → data dominates
- λ learned from historical backtesting
- Closed-form weighted Ridge solution

### Step 3: Estimate δ_race (Race-Specific Polls)

For each race with polls, compute residuals from θ_national:

```
r_p = y_p - W_p · θ_national
```

Solve for δ_race:

```
minimize Σ_{p∈race} (1/σ_p²) · (r_p - W_p · δ_race)² + μ · ||δ_race||²
```

- μ regularizes toward zero (candidate effects assumed small unless data says otherwise)
- 1 poll → small δ; 20 polls → larger, data-driven δ
- μ learned from historical backtesting

### Crosstab-Informed W Vectors

When polls include demographic crosstabs (age, race, education breakdowns), the model's type estimates θ_national can be used to *back-infer* what types the poll was sampling. The logic:

1. The model produces θ_national[j] for each type
2. Each type has a known demographic profile (from census data)
3. A poll's crosstab tells us its demographic composition
4. We can compute: "given this poll's demographic mix, what types were overrepresented/underrepresented in the sample?"
5. This produces a poll-specific W_p that's more accurate than the state-mean W

This is an iterative refinement: initial θ_national is estimated with state-mean W, then W is refined using θ_national + crosstab data, then θ_national is re-estimated with improved W. Convergence is expected in 2-3 iterations.

### Step 4: Produce Forecasts

For each race:
1. Compute θ_forecast (national or national + δ depending on mode)
2. county_pred[c] = W[c] · θ_forecast
3. Aggregate to state/district level (vote-weighted)
4. Confidence intervals from type-level prediction error (learned from backtest)

---

## 3. Historical Calibration

### Purpose

Learn λ (model-vs-polls trust) and μ (candidate effect shrinkage) from cycles where ground truth is known.

### Training Data

| Cycle | Polls (538) | Race types |
|-------|-------------|------------|
| 2012 | ~1,838 | Presidential, Senate, Governor, House |
| 2016 | ~2,677 | Presidential, Senate, Governor, House |
| 2020 | ~2,663 | Presidential, Senate, Governor, House |
| 2022 | ~151 (our CSV) | Senate, Governor |
| 2024 | ~1,701 | Presidential, Senate, Governor, House |

### Procedure: Leave-One-Cycle-Out Cross-Validation

For each held-out cycle k:
1. Compute θ_prior[j] using only elections before cycle k
2. Run estimation procedure (Steps 2-4) on cycle k's polls with candidate (λ, μ)
3. Convert θ_forecast → county predictions → compare to actual county results
4. Score: vote-weighted county RMSE

Sweep (λ, μ) grid. Pick pair minimizing mean RMSE across held-out cycles.

### Additional Outputs

- **Systematic poll bias per cycle**: Residual between fitted θ_national and actual θ. Stored as reference (cannot correct in real-time).
- **W vector accuracy**: Back-solve for implied W_true from actual results. Validates or improves W construction.
- **Type-level prediction error**: Which types predict well/poorly? Feeds into confidence intervals.
- **Cycle-specific vs pooled**: Start cycle-specific (λ_k, μ_k). If close to pooled, use pooled for stability.

### Output Artifacts

- `data/calibration/backtest_results.parquet` — per-county, per-cycle predictions vs actuals
- `data/calibration/optimal_params.json` — learned λ, μ (pooled and per-cycle)
- `data/calibration/w_validation.parquet` — inferred vs actual W vectors per state per cycle
- `docs/research/backtest-report.md` — human-readable analysis

---

## 4. National Expansion

### Race Inventory (2026)

- ~34 Senate races
- ~36 Governor races
- 435 House races
- 1 generic ballot
- **Total: ~506 races**

### Race Definition

Single source of truth: `data/races/races_2026.csv`

```csv
race_id,race_type,state,district,year
2026-fl-senate,senate,FL,,2026
2026-fl-governor,governor,FL,,2026
2026-fl-13,house,FL,13,2026
```

Prediction pipeline discovers races from this file. Every defined race gets a forecast.

### W Vector Construction by Race Type

| Race type | Geography | W construction |
|-----------|-----------|----------------|
| Senate | State | Vote-weighted mean of county type memberships in state |
| Governor | State | Same as Senate |
| House | District | Overlap-weighted mean of county type memberships via county-district crosswalk |
| Generic ballot | National | Vote-weighted mean across all counties |

### Data Pipeline Requirements

**Historical polls into DuckDB:**

```sql
CREATE TABLE historical_polls (
    cycle INTEGER,
    race_id TEXT,
    race_type TEXT,        -- 'president', 'senate', 'governor', 'house'
    state TEXT,
    district INTEGER,      -- NULL for statewide races
    pollster TEXT,
    dem_share FLOAT,
    n_sample INTEGER,
    date DATE,
    grade TEXT,             -- pollster quality grade
    source TEXT             -- '538', 'scraper', 'manual'
);
```

538 raw_polls.csv converted and loaded for 2012-2024. Current 2026 polls also loaded.

**Historical election results in DuckDB:**

```sql
CREATE TABLE historical_results (
    cycle INTEGER,
    race_type TEXT,
    state TEXT,
    district INTEGER,      -- NULL for statewide
    county_fips TEXT,
    dem_votes INTEGER,
    rep_votes INTEGER,
    total_votes INTEGER,
    dem_share FLOAT
);
```

Sources: MEDSL (presidential), Algara (governor), MEDSL Senate loader (to build), DRA block data (House via crosswalk).

### Election Results Needed

| Data | Status | Source |
|------|--------|--------|
| Presidential county 2000-2024 | Have it | MEDSL |
| Governor county 2002-2022 | Have it | Algara/Amlani |
| Senate county 2000-2024 | Need loader | MEDSL Harvard Dataverse |
| House district 2012-2024 | Need crosswalk + aggregation | DRA block data |

---

## 5. Frontend UX

### Forecast Toggle

On each race forecast page (`/forecast/[slug]`), a segmented control:

```
[ National Environment | Local Polling ]
```

**National Environment** (default for unpolled races):
- Shows forecast from θ_national only
- Label: "Based on national political environment — no race-specific polling applied"

**Local Polling** (default for polled races):
- Shows forecast from θ_national + δ_race
- Label: "Based on national environment + N polls for this race"

### Behavior

**Both modes always visible for polled races.** User can toggle and compare.

For unpolled races, "Local Polling" is grayed out: "No polls available for this race."

**What changes on toggle:**
- State/district-level prediction number + confidence interval
- County-level map coloring
- Win probability bar
- Type breakdown table (shows θ_national vs θ_national + δ per type)

**What does NOT change:**
- Model prior (shown separately as reference)
- Poll list (always shown if polls exist)
- Methodology explanation

**Comparison annotation:** Show delta between modes: "Local polling shifts this race D+2.3 from national environment." Communicates candidate effect magnitude at a glance.

---

## 6. Module Architecture

### New Modules

```
src/calibration/
    backtest.py              — Leave-one-cycle-out calibration loop
    convert_538_historical.py — Load 538 polls into DuckDB historical_polls table
    load_historical_results.py — Load election results into DuckDB historical_results table

src/prediction/
    national_environment.py  — θ_national estimation (Step 2)
    candidate_effects.py     — δ_race estimation (Step 3)
    forecast_engine.py       — Orchestrates: θ_prior → θ_national → δ_race → county/state/district preds

src/assembly/
    build_district_crosswalk.py — County-to-district mapping (in progress, House agent)
    build_district_types.py     — District-level type compositions (in progress, House agent)
    fetch_medsl_senate.py       — Senate county results from MEDSL
    define_races.py             — Load/validate races_2026.csv

api/routers/forecast.py        — Extended with toggle support (mode=national|local query param)
```

### Modified Modules

```
src/prediction/predict_2026_types.py — Refactored to call forecast_engine.py
src/assembly/ingest_polls.py         — Extended for House race geography
src/propagation/poll_weighting.py    — House effect correction fixed (load 538 bias data)
src/db/build_database.py             — New tables: historical_polls, historical_results, races
```

### Data Files

```
data/races/races_2026.csv              — Race definitions (single source of truth)
data/calibration/backtest_results.parquet
data/calibration/optimal_params.json
data/calibration/w_validation.parquet
```

---

## 7. Implementation Priority

1. **National expansion** — Define all races, make prediction pipeline dynamic, extend API/frontend
2. **National environment model** — θ_national estimation from pooled polls, δ_race from residuals, frontend toggle
3. **Historical calibration** — 538 poll ingestion to DuckDB, election results loading, backtest loop, learn λ/μ
4. **House races** — District crosswalk, district type compositions, House poll ingestion (data pipeline in parallel)

Calibration backtest runs in parallel with pillars 1-2 as an independent workstream.

---

## 8. Success Criteria

- Every 2026 Senate and Governor race has a forecast page with predictions
- House races have forecast pages once district crosswalk is complete
- Backtested RMSE with calibrated (λ, μ) beats current uncalibrated pipeline RMSE
- User can toggle national/local on polled race pages and see the difference
- θ_national produces reasonable forecasts for unpolled races (sanity-checked against fundamentals)
- W vector validation shows improvement over state-mean W
