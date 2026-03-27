# Forecast Phase 4: Rich Poll Ingestion
**Date:** 2026-03-26
**Status:** Design — not yet implemented
**Prerequisite:** Phases 1–3 complete (bugs fixed, scaffold live, poll integration working)
**Spec reference:** `docs/superpowers/specs/2026-03-26-forecast-tab-design.md` (Phase 4 section)

---

## Problem Statement

Every poll currently enters the Bayesian update as a single scalar observation:
`(dem_share, n_sample, state)`.

The W row for that poll is derived from the **population-weighted mean of type scores across all counties in that state**. This is correct in expectation, but it ignores information that crosstab-bearing polls explicitly provide: *which demographic groups were sampled and in what proportions*.

A poll that interviewed 65% college-educated respondents (vs the state's 35% baseline) contains information that is concentrated in types with high `pct_bachelors_plus`. Its W row should weight those types more heavily than a random draw from the state would. Failing to use this information means:

1. **Over-dispersion:** the poll's signal is spread evenly across all state types rather than focused where the data actually points.
2. **Lost resolution:** polls with non-representative demographic sampling look identical to random samples. A poll oversampling seniors hits different types than a poll oversampling college grads, even if both report 48% Dem — but today they produce the same W row.
3. **No crosstab signal:** a poll that reports *both* the topline (48%) and a crosstab (college-educated subgroup: 62%; non-college: 38%) is providing two independent observations, each with its own W row. Today both are discarded.

Phase 4 fixes this by (a) ingesting and storing crosstab data, (b) constructing poll-specific W vectors that reflect each poll's actual demographic composition, and (c) supporting sub-topline crosstab observations as additional data points in the Bayesian update.

---

## Research: What Crosstab Data Actually Contains

### What Pollsters Report

High-quality polls (Quinnipiac, NYT/Siena, Marist, Emerson, CNN) routinely publish cross-tabulated breakdowns. Standard crosstab dimensions:

| Dimension | Typical Groups | Pollster Coverage |
|-----------|---------------|-------------------|
| **Education** | College grad / Non-college (often split by race too: white college vs. white non-college) | ~70% of quality polls |
| **Age** | 18–29, 30–44, 45–64, 65+ | ~80% of quality polls |
| **Gender** | Male / Female | ~90% of quality polls |
| **Race/ethnicity** | White, Black, Hispanic, Asian/Other | ~60% of quality polls |
| **Urbanicity** | Urban / Suburban / Rural | ~40% of quality polls |
| **Party ID** | Dem / Ind / Rep | ~90%, but endogenous — should not be used for W construction |
| **Region** | Often used for state-level polls covering a Senate race | ~30% of quality polls |

**What pollsters do NOT report:** crosstabs on all combinations. A poll of N=800 might have n=40 Black respondents — too small for stable estimates. Only the demographic breakdowns where the pollster's subgroup N is large enough (typically N≥100) appear in published tables.

### Current 538 Data

The FiveThirtyEight `raw_polls.csv` at `data/raw/fivethirtyeight/data-repo/pollster-ratings/raw_polls.csv` contains only poll-level summaries: `pollster, race, location, samplesize, cand1_pct, cand2_pct`. **No crosstab fields.** Crosstab data is not included in the 538/Silver Bulletin archival dataset.

For historical crosstab data, sources are:
- Individual pollster PDF/HTML crosstab tables (parseable but not archived in bulk)
- Roper Center archive (paid subscription — out of scope for free-data-only constraint)
- CNN/exit polls (structured, limited to presidential)
- ANES (academic surveys, highly detailed but not real-time)

**Implication:** For the initial Phase 4 implementation, crosstab data will be entered manually or scraped on demand, not sourced from a bulk download. The infrastructure must handle both cases: polls with crosstabs and polls without (the current simple format).

### What Crosstab Fields to Store

Based on what pollsters actually report and what maps cleanly to our type demographic features (from `type_profiles.parquet`), the priority demographic dimensions are:

| Crosstab Dimension | CSV column name | Maps to type feature | Coverage |
|-------------------|-----------------|---------------------|---------|
| Education: college grad | `educ_college` | `pct_bachelors_plus` | High |
| Education: non-college | `educ_noncollege` | `1 - pct_bachelors_plus` | High |
| Age: 65+ | `age_65plus` | `median_age` proxy | High |
| Race: White | `race_white` | `pct_white_nh` | High |
| Race: Black | `race_black` | `pct_black` | High |
| Race: Hispanic | `race_hispanic` | `pct_hispanic` | Medium |
| Urbanicity: Urban | `urb_urban` | `log_pop_density` | Medium |
| Urbanicity: Rural | `urb_rural` | `log_pop_density` inverse | Medium |
| Gender: Female | `gender_female` | none direct (future) | Medium |
| Religion: Evangelical | `rel_evangelical` | `evangelical_share` | Low today, growing |

---

## Type–Demographic Bridge

### The Key Mapping

Type profiles (`data/communities/type_profiles.parquet`) provide population-weighted mean values for each demographic feature across counties assigned to each type. The columns available per type include:

```
pct_white_nh, pct_black, pct_asian, pct_hispanic,
pct_bachelors_plus, median_age, log_pop_density,
pct_wfh, pct_owner_occupied,
evangelical_share, mainline_share, catholic_share,
pct_management, net_migration_rate, ...
```

Each of these maps to one or more crosstab dimensions. The mapping is not always perfect (age 65+ doesn't map directly to median_age), but a coarse approximation is far better than the current uniform assignment.

### Mapping Table (Production Subset)

| Crosstab dimension | Direction | Type profile feature | Notes |
|-------------------|-----------|---------------------|-------|
| `pct_sample_college` | higher → more college-heavy types | `pct_bachelors_plus` | Direct, high quality |
| `pct_sample_white` | higher → more white-heavy types | `pct_white_nh` | Direct |
| `pct_sample_black` | higher → more Black-heavy types | `pct_black` | Direct |
| `pct_sample_hispanic` | higher → more Hispanic-heavy types | `pct_hispanic` | Direct |
| `pct_sample_urban` | higher → higher density types | `log_pop_density` | Monotone mapping |
| `pct_sample_rural` | higher → lower density types | `log_pop_density` inverse | |
| `pct_sample_senior` | higher → older types | `median_age` | Approximate; mean ≠ share 65+ |
| `pct_sample_evangelical`| higher → more evangelical types | `evangelical_share` | Direct |

The baseline composition for a geography is the population-weighted mean across that geography's counties (same as the current W row construction). A crosstab-adjusted W row up-weights types whose demographic profile matches the poll's sample composition and down-weights types where it doesn't.

---

## Schema Design

### New DuckDB Tables

The existing `polls` table already has `poll_id` as the primary key. Phase 4 extends the polling domain with a populated `poll_crosstabs` table (the table already exists as a stub in `src/db/domains/polling.py` but is never populated).

#### `poll_crosstabs` (already stubbed, extend its semantics)

Current DDL (already in `polling.py`):
```sql
CREATE TABLE IF NOT EXISTS poll_crosstabs (
    poll_id           VARCHAR NOT NULL,
    demographic_group VARCHAR NOT NULL,   -- e.g. "education"
    group_value       VARCHAR NOT NULL,   -- e.g. "college"
    dem_share         FLOAT,
    n_sample          INTEGER
);
```

**This schema is already correct.** The table is EAV-style (entity-attribute-value), which is appropriate because:
- Polls have variable crosstab dimensions. Not every poll has an education break; not every poll reports urbanicity.
- New dimensions can be added without a schema migration.
- The `demographic_group` / `group_value` pair identifies the cell uniquely per poll.

**Extend with one additional column:**

```sql
ALTER TABLE poll_crosstabs ADD COLUMN pct_of_sample FLOAT;
-- Fraction of respondents in this group (e.g. 0.35 = 35% college grads).
-- NULL if the pollster only reports dem_share for the group, not composition.
-- Required for W vector construction; optional for pure crosstab recording.
```

**Full extended schema:**

```sql
CREATE TABLE IF NOT EXISTS poll_crosstabs (
    poll_id           VARCHAR  NOT NULL REFERENCES polls(poll_id),
    demographic_group VARCHAR  NOT NULL,  -- e.g. "education", "race", "age", "urbanicity"
    group_value       VARCHAR  NOT NULL,  -- e.g. "college", "white", "senior", "urban"
    dem_share         FLOAT,              -- dem two-party share in this group (0–1)
    n_sample          INTEGER,            -- respondents in this group (may be NULL if not reported)
    pct_of_sample     FLOAT,              -- fraction of total poll sample in this group
    PRIMARY KEY (poll_id, demographic_group, group_value)
);
```

**Notes on design choices:**
- `pct_of_sample` is the crucial column for W adjustment. Without it, we know how *this demographic group voted* but not how *over- or under-represented it was* in the poll sample. Both pieces of information are useful; `pct_of_sample` is what drives W construction.
- The column is nullable: many crosstab publications report only `dem_share` per group (e.g., "college voters went 62/38"). If pct_of_sample is NULL, the crosstab row is recorded (for future use as a direct sub-observation) but cannot adjust the aggregate W.
- The `PRIMARY KEY` constraint ensures no duplicate dimension-group pairs per poll.

### Input CSV Format Extensions

Current format (remains backward-compatible):
```
race,geography,geo_level,dem_share,n_sample,date,pollster,notes
```

New format (adds optional crosstab columns):
```
race,geography,geo_level,dem_share,n_sample,date,pollster,notes,xt_education_college,xt_education_noncollege,xt_race_white,xt_race_black,xt_race_hispanic,xt_age_senior,xt_urbanicity_urban,xt_urbanicity_rural
```

Each `xt_<group>_<value>` column encodes the poll's composition as a fraction (0–1). These map directly to `pct_of_sample` in the `poll_crosstabs` table.

**Convention:** prefix `xt_` = crosstab composition column. `dem_share` in the `poll_crosstabs` table comes from a separate column if the pollster reports sub-group vote shares. To keep the CSV manageable, sub-group vote shares are secondary — composition fractions are the priority.

**Backward compatibility:** All `xt_*` columns are optional. A poll row without any `xt_*` columns is ingested exactly as before and gets no `poll_crosstabs` rows.

---

## W Vector Construction Algorithm

### Current Behavior (Baseline)

In `predict_2026_types.py`, `predict_race()`:

```python
state_mask = np.array([s == poll_state for s in states])
state_scores = type_scores[state_mask]          # shape (n_state_counties, J)
W_row = np.abs(state_scores).mean(axis=0)       # simple mean across state counties
W_row = W_row / W_row.sum()                     # normalize to sum to 1
```

This is the **population-unweighted** mean of county type scores within the state. It represents "if you sampled a random county in this state, what type composition would you expect?" — but it doesn't account for (a) population weighting across counties, or (b) demographic composition of the actual poll sample.

### Phase 4 Target Behavior

When a poll has `poll_crosstabs` rows with `pct_of_sample` values, construct a demographically-adjusted W row:

#### Algorithm: Crosstab-Adjusted W Construction

**Step 1: Compute state baseline type composition**

For state `s`, the baseline W vector is the population-weighted mean of county type scores:

```
W_base[j] = Σ_c (population[c] * |score[c,j]|) / Σ_c population[c]
           for counties c in state s
```

This is a population-weighted version of the current mean. (Note: the current code does an unweighted mean; population weighting is a separate improvement bundled into this step.)

**Step 2: Compute type demographic affinity per crosstab dimension**

For each crosstab dimension (e.g., "education / college"), compute how strongly each type is associated with that demographic feature. This uses the type profiles:

```python
def type_affinity_for_crosstab(
    type_profiles: pd.DataFrame,  # shape (J, n_features)
    demographic_feature: str,      # e.g. "pct_bachelors_plus"
    state_baseline: float,         # state-level mean of this feature
) -> np.ndarray:                   # shape (J,), type affinity scores
    """
    Returns, for each type, how much its demographic feature value deviates
    from the state mean. Positive = this type is richer in this demographic
    than the state average. Negative = below average.
    """
    type_values = type_profiles[demographic_feature].values  # shape (J,)
    return type_values - state_baseline  # deviation from state mean
```

**Step 3: Compute sample composition deviation per crosstab dimension**

For each crosstab dimension present in `poll_crosstabs`, compute the deviation of the poll's sample composition from the state's population composition:

```
delta[dim] = pct_of_sample_in_poll[dim] - pct_of_population_in_state[dim]
```

If `pct_of_sample = 0.55` (55% college-educated) and the state is 35% college-educated, then `delta = +0.20`. The poll oversampled college-educated voters by 20 percentage points.

**Step 4: Adjust W row based on composition deviations**

Combine baseline W with demographic adjustments:

```
W_adjusted[j] = W_base[j] + α * Σ_dim (delta[dim] * affinity[j, dim] / max_affinity[dim])
```

Where:
- `α` is an adjustment strength hyperparameter (recommend: 0.3, meaning adjustments are capped at 30% of baseline weight)
- `affinity[j, dim]` is the type-level deviation from state mean for dimension `dim`
- `max_affinity[dim]` normalizes so that a +1.0 delta produces at most a unit shift

After adjustment, re-normalize W to sum to 1 (W_adjusted must remain a valid probability vector):

```
W_adjusted = clip(W_adjusted, min=0)
W_adjusted = W_adjusted / W_adjusted.sum()
```

**Step 5: Handle polls with no crosstab data**

If `poll_crosstabs` has no rows with `pct_of_sample` for this poll, fall back to the population-weighted state baseline (`W_base`). This is slightly better than the current implementation even without crosstab data.

#### Concrete Example

Suppose J=3 types for simplicity. State GA has:
- Type A: "Black Urban" — pct_bachelors_plus = 0.32, pct_black = 0.68
- Type B: "Rural Evangelical" — pct_bachelors_plus = 0.12, pct_black = 0.04
- Type C: "College Suburban" — pct_bachelors_plus = 0.58, pct_black = 0.12

State GA population baseline: pct_bachelors_plus = 0.31, pct_black = 0.31.

W_base = [0.4, 0.3, 0.3] (population-weighted type composition of GA counties).

Poll has crosstab: `pct_of_sample` for education/college = 0.52 (vs state 0.31, delta = +0.21).

Affinity for college dimension:
- Type A: 0.32 - 0.31 = +0.01 (near-average)
- Type B: 0.12 - 0.31 = -0.19 (well below average — rural types)
- Type C: 0.58 - 0.31 = +0.27 (well above average — college suburban)

Adjustment (α=0.3, max_affinity = 0.27):
- W_adj[A] += 0.3 * 0.21 * (0.01/0.27) ≈ +0.002
- W_adj[B] += 0.3 * 0.21 * (-0.19/0.27) ≈ -0.044
- W_adj[C] += 0.3 * 0.21 * (0.27/0.27) ≈ +0.063

W_adjusted (before normalization): [0.402, 0.256, 0.363]
After clipping to ≥0 and normalizing: [0.394, 0.251, 0.355]

Result: the college-oversampled poll appropriately weights "College Suburban" types more and "Rural Evangelical" types less. The Bayesian update via this W row will pull those types' θ estimates more than a generic state W would.

#### Using Crosstab Sub-Group Vote Shares as Separate Observations

When `dem_share` is present in `poll_crosstabs` (not just `pct_of_sample`), each crosstab cell is an **additional direct observation** of a sub-population. This can be incorporated as a separate W row in the stacked Bayesian update:

For crosstab cell (education=college, dem_share=0.64, n=280 in a poll of N=800):
- Construct `W_college_row` = type profile weighted by `pct_bachelors_plus` (soft assignment of the "college respondents in this geography" to types)
- Add `(W_college_row, y=0.64, sigma=sqrt(0.64*0.36/280))` as a row in the W matrix

This is mathematically clean because each sub-group observation is an independent draw from a different sub-population with its own type composition. The key constraint: the top-line observation and its sub-group observations are **not independent** (the top-line is a linear combination of the sub-group shares weighted by `pct_of_sample`). For initial implementation, include only one of: either the topline or the sub-group observations, not both for the same poll. The sub-group observations are more informative when their sample sizes are large enough (n ≥ 100 per group recommended).

**Phase 4a (initial):** Use crosstabs only to adjust the topline W row (Steps 1–5 above). No separate sub-group observations.

**Phase 4b (follow-on):** Add sub-group observations as separate W rows for the Bayesian update, replacing the topline observation if sub-group coverage is complete.

---

## Implementation Plan

### Step 0: Pre-work (no code changes)

**0a. Gather crosstab data for 2 pilot polls** (manual)
- Find a published Quinnipiac or NYT/Siena poll for a 2026 race with full crosstab tables
- Record: total N, and for each dimension (education, race, urbanicity) the fraction of respondents in each category
- Add `xt_*` columns to `data/polls/polls_2026.csv` for those 2 polls
- This drives the ingestion code and validates the CSV format before writing any parsing logic

### Step 1: Extend CSV parsing for crosstab columns

**File:** `src/db/domains/polling.py`

**Changes:**
1. Add `pct_of_sample FLOAT` column to `poll_crosstabs` DDL (or `ALTER TABLE` for existing DBs)
2. In `ingest()`, parse `xt_<group>_<value>` columns from the CSV. For each non-empty `xt_*` column:
   - Split on `_`: `xt_education_college` → `demographic_group="education"`, `group_value="college"`
   - Insert row into `poll_crosstabs`: `(poll_id, "education", "college", NULL, NULL, 0.55)`
3. Leave `dem_share` and `n_sample` in `poll_crosstabs` as NULL unless a matching sub-group vote share column is also present (see Phase 4b)
4. Update `create_tables()` to include `pct_of_sample` in the DDL

**File:** `data/polls/polls_2026.csv`
- Add `xt_*` columns for the pilot polls from Step 0

**Tests:**
- `tests/test_polling_domain.py` (new): test `xt_*` column parsing with a fixture CSV
- Verify backward compatibility: CSV without `xt_*` columns still ingests cleanly

**Dependencies:** None (schema extension is backward-compatible)

### Step 2: Build the type-demographic affinity index

**File:** `src/propagation/crosstab_w_builder.py` (new file)

**Purpose:** Given type profiles and a state's demographic baseline, compute type affinity vectors for each supported crosstab dimension. This is a pure function that produces an immutable affinity table loaded once at API startup.

```python
CROSSTAB_DIMENSION_MAP: dict[str, str] = {
    # crosstab dimension key → type_profiles column name
    "education_college":    "pct_bachelors_plus",
    "education_noncollege": None,       # derived: 1 - pct_bachelors_plus
    "race_white":           "pct_white_nh",
    "race_black":           "pct_black",
    "race_hispanic":        "pct_hispanic",
    "race_asian":           "pct_asian",
    "urbanicity_urban":     "log_pop_density",    # higher = more urban
    "urbanicity_rural":     "log_pop_density",    # higher value = LESS rural (inverse)
    "age_senior":           "median_age",         # proxy only
    "religion_evangelical": "evangelical_share",
}

def build_affinity_index(
    type_profiles: pd.DataFrame,   # shape (J, n_features)
    state_type_weights: pd.DataFrame,  # shape (n_states, J+1) with 'state_abbr' column
    county_demographics: pd.DataFrame, # shape (N, n_features + 'state_abbr' + 'county_fips')
) -> dict[str, np.ndarray]:
    """
    Returns a dict mapping dimension_key → affinity vector of shape (J,).

    Each affinity vector is the type's demographic feature value
    (population-weighted mean over type-member counties) minus the
    national population-weighted mean. Indexed by type_id.
    """
    ...
```

**Also in this file:** `construct_w_row()` function implementing Steps 1–5 from the algorithm section above. This is the public API for Phase 4:

```python
def construct_w_row(
    poll_crosstabs: list[dict],        # rows from DuckDB poll_crosstabs for one poll
    state_baseline_w: np.ndarray,      # shape (J,), current population-weighted W
    affinity_index: dict[str, np.ndarray],  # from build_affinity_index
    state_demographic_means: dict[str, float],  # feature → state mean
    adjustment_strength: float = 0.3,  # α
) -> np.ndarray:
    """
    Returns adjusted W row (J,) for a poll that has crosstab composition data.
    Falls back to state_baseline_w if no usable crosstab rows.
    """
    ...
```

**Tests:**
- `tests/test_crosstab_w_builder.py` (new): unit tests for affinity and W construction
- Test that W sums to 1.0 after adjustment
- Test that a college-oversampled poll increases weight on college-type
- Test fallback to state_baseline when no crosstab data
- Test with J=100 (production scale) for numerical stability

**Dependencies:** Step 1 (crosstab data must be in DuckDB); `type_profiles.parquet`

### Step 3: Wire crosstab W into the API forecast pipeline

**File:** `api/main.py` — lifespan

At startup, load the affinity index into `app.state`:

```python
app.state.crosstab_affinity = build_affinity_index(
    type_profiles=...,
    state_type_weights=...,
    county_demographics=...,
)
app.state.crosstab_state_means = ...  # state-level demographic means
```

**File:** `api/routers/forecast.py` — `update_forecast_with_multi_polls`

Modify the poll-building section to use `construct_w_row()` when crosstab data is available:

```python
# Current: polls are (dem_share, n_sample, state_abbr)
# Phase 4: polls are (dem_share, n_sample, state_abbr, w_row_override=None)
# If poll has crosstabs in DuckDB, compute w_row via construct_w_row() and pass it
```

**File:** `src/prediction/predict_2026_types.py` — `predict_race()`

Extend the `polls` parameter to support per-poll W row overrides:

```python
# Before:
polls: list[tuple[float, int, str]] | None = None
# After (backward-compatible — third element is still state_abbr, optional fourth is W override):
polls: list[tuple[float, int, str, np.ndarray | None]] | None = None
```

Inside the W-building loop:
```python
for dem_share, n, poll_state, *rest in polls:
    w_override = rest[0] if rest else None
    if w_override is not None:
        W_row = w_override / w_override.sum()  # use precomputed crosstab-adjusted W
    else:
        # existing state-based W construction
        ...
```

**Tests:**
- Extend `tests/test_predict_race.py` to verify W override propagates correctly
- Verify that API endpoint uses crosstab W when poll has crosstabs in DuckDB
- Verify API gracefully falls back to state W when crosstabs are absent

**Dependencies:** Steps 1 and 2

### Step 4: Extend the ingestion pipeline for crosstab CSV format

**File:** `src/assembly/ingest_polls.py`

Currently reads from CSV with the 8-column format. Extend to:
1. Detect and skip `xt_*` columns when constructing `PollObservation` objects (they don't belong in `PollObservation`, which remains lean)
2. Optionally: provide a `load_polls_with_crosstabs()` function that returns `(polls, crosstab_data)` for callers that need composition data before DuckDB is available

**File:** `data/polls/README.md`

Document the new `xt_*` column convention, with a worked example using an education crosstab.

**Dependencies:** Step 1

### Step 5: Rebuild DuckDB database with extended schema

The `build_database.py --reset` command should:
1. Create `poll_crosstabs` with the extended DDL (including `pct_of_sample`)
2. Re-ingest `polls_2026.csv` (now with `xt_*` columns for pilot polls)
3. Validate contract: `poll_crosstabs` table exists with correct columns

**File:** `src/db/build_database.py`

- Add `poll_crosstabs` to the `validate_contract()` required tables check
- Bump DB rebuild instruction in CLAUDE.md / README

### Step 6: Integration test and validation

**File:** `tests/test_api_contract.py`
- Add assertion that `poll_crosstabs` is present in schema
- Add test: poll with `xt_education_college=0.55` produces a different W row than a poll without crosstabs, all else equal

**Manual validation:**
- Take the 2 pilot polls from Step 0
- Compare `W_base` vs `W_adjusted` for each
- Verify the adjustment direction is correct (college-oversampled poll → higher weight on college-type types)
- Compare final county-level predictions: with and without crosstab adjustment
- Verify the delta is small but directionally sensible (not a large swing — these are adjustments to W, not to the topline)

**Calibration of `adjustment_strength` (α):**
- Backtesting: hold out one election cycle; fit α to minimize RMSE on county predictions across several historical polls with known crosstabs
- Initial value: 0.3 (conservative; keeps adjustments within ±30% of baseline W)
- α is a named constant in `crosstab_w_builder.py`, not hardcoded inline

---

## API Changes Needed

### No new endpoints required for Phase 4a

The existing `POST /forecast/polls` endpoint handles everything. The W vector construction is internal — the client doesn't need to know whether a poll used crosstab adjustment.

### Optional: `GET /polls/{poll_id}/crosstabs`

For UI transparency (showing users "this poll has demographic crosstab data"), expose the crosstab rows:

```
GET /polls/{poll_id}/crosstabs
→ [
    {"demographic_group": "education", "group_value": "college",
     "pct_of_sample": 0.55, "dem_share": null, "n_sample": null},
    ...
  ]
```

This endpoint is low-priority but useful for the Forecast tab's "data panel" — a tooltip showing "Quinnipiac 5/12: education crosstab available."

### `GET /polls` response extension

Add a boolean `has_crosstabs` field to `PollRow`:

```python
class PollRow(BaseModel):
    ...
    has_crosstabs: bool = False  # True if poll_crosstabs has rows for this poll
```

This lets the frontend show a visual indicator (e.g., a small demographic icon) next to polls that have crosstab data.

---

## Frontend Changes Needed

Phase 4 frontend changes are minimal. The forecast calculation itself is backend-only. The only user-facing change is indicating which polls have crosstab data.

**Forecast Tab — Data Panel:**
- Poll row in the section list shows a small "D" badge (demographic) if `has_crosstabs=True`
- Tooltip: "Demographic crosstab data available — W vector constructed from poll sample composition"
- No configuration needed; crosstab adjustment is automatic when data is present

**No map changes.** The output is still `pred_dem_share` per county — the frontend doesn't need to know about W vectors.

---

## Open Questions

| ID | Question | Notes |
|----|----------|-------|
| P4-001 | What is the correct `adjustment_strength` α? | Conservative default: 0.3. Requires historical backtest once pilot polls are identified. |
| P4-002 | Should sub-group vote shares (Phase 4b) be in the same CSV or separate? | Separate file (`polls_crosstabs_2026.csv`) is cleaner but requires a second ingestion step. Single file with `xt_dem_<group>_<value>` columns is more compact but verbose. |
| P4-003 | How to handle a poll with education crosstabs but no race crosstabs? | Apply education adjustment only; race dimension contributes zero adjustment. This is already handled by the algorithm. |
| P4-004 | Population weighting for W_base: use county population from ACS 2022? | Yes — county_demographics table already has `pop_total`. Use as population weight in W_base construction. Currently unweighted, which is a separate (existing) issue. |
| P4-005 | Should the affinity index be state-specific or national? | State-specific is more accurate (deviation from state mean, not national mean). Build state-level affinity vectors. |
| P4-006 | How to handle MRP-style crosstabs? | Some high-quality polls publish model-based (MRP) crosstabs rather than raw. These are more informative but also more model-dependent. For Phase 4, treat all crosstab-reported pct_of_sample values uniformly; defer MRP-vs-raw distinction. |
| P4-007 | Should `pct_of_sample` crosstab categories be required to sum to 1 within each dimension? | Yes — add validation in ingestion. `xt_education_college + xt_education_noncollege` should ≈ 1.0 (allow ±0.02 for rounding). |

---

## Estimated Scope

| Step | Description | Size | Complexity | Notes |
|------|-------------|------|------------|-------|
| 0 | Manual pilot data collection | 1–2 hours | Low | Blocking: needed before code can be tested |
| 1 | CSV parsing + DB schema extension | ~100 lines | Low | Extend existing `polling.py`; additive change |
| 2 | `crosstab_w_builder.py` (new file) | ~200 lines | Medium | Core logic; needs careful testing |
| 3 | Wire into forecast API | ~80 lines | Medium | Backward-compat critical; test thoroughly |
| 4 | Ingestion pipeline extension | ~50 lines | Low | Mostly documentation |
| 5 | DB rebuild + contract validation | ~30 lines | Low | Mechanical |
| 6 | Integration tests + calibration | ~150 lines tests | Medium | α calibration requires historical data |

**Total:** ~600 lines of code + tests; 2–3 days of focused work after Step 0 data collection.

The critical path is **Step 0 → Step 2** (data → algorithm). Steps 3–6 are mechanical once Step 2 is validated.

---

## Known Risks

1. **Sparse pilot data.** Real crosstab data requires manual curation. If Step 0 fails to find 2 usable polls, Steps 1–6 can still proceed using synthetic crosstab fixtures, but calibration (Step 6) requires real data before going live.

2. **Affinity map instability for small types.** With J=100 types, some types have few counties (n_counties=5–10). Their demographic means are noisier. The affinity index may produce erratic adjustments for small types. Mitigation: floor the adjustment to zero when type n_counties < 20 (a named constant).

3. **W normalization after adjustment can lose type resolution.** If the crosstab adjustment drives some W_row values negative (types very different from the poll's demographics), clipping to zero and renormalizing re-distributes those weights. With a conservative α=0.3, this is unlikely to be a significant issue.

4. **The "not independent" problem for Phase 4b.** Including both topline and sub-group observations as separate Bayesian update rows technically violates the independence assumption because they share respondents. For Phase 4a (topline only, adjusted W), this is not an issue. Flag clearly in code comments before implementing Phase 4b.

---

## File Change Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `src/db/domains/polling.py` | Modify | Add `pct_of_sample` to `poll_crosstabs` DDL; parse `xt_*` columns |
| `src/propagation/crosstab_w_builder.py` | New | Affinity index + `construct_w_row()` |
| `src/prediction/predict_2026_types.py` | Modify | Support per-poll W row override in `predict_race()` |
| `api/routers/forecast.py` | Modify | Load crosstab W rows when available in DuckDB |
| `api/main.py` | Modify | Load affinity index into `app.state` at startup |
| `api/models.py` | Modify | Add `has_crosstabs: bool` to `PollRow` |
| `data/polls/polls_2026.csv` | Modify | Add `xt_*` columns for 2 pilot polls |
| `data/polls/README.md` | Modify | Document `xt_*` column convention |
| `tests/test_crosstab_w_builder.py` | New | Unit tests for W construction algorithm |
| `tests/test_polling_domain.py` | Modify | Add `xt_*` parsing tests |
| `tests/test_api_contract.py` | Modify | Add `poll_crosstabs` schema assertion |
| `src/db/build_database.py` | Modify | Add `poll_crosstabs` to contract validation |
