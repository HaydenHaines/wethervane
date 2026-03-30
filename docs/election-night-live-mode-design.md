# Election Night Live Mode Design

**Status:** Design document for future implementation (November 2026)
**Scope:** Real-time election result ingestion, live prediction accuracy tracking, and race call support
**Priority:** Non-critical for launch; post-election feature

---

## 1. Core Concept

**Live Mode** transforms WetherVane from a pre-election prediction tool into a real-time election night companion that:

- **Ingests actual results** as they report from states (county-level vote tallies)
- **Tracks prediction accuracy** in real-time — did our model call this right? By how much?
- **Supports race calls** — when posterior probability crosses a threshold (95%), declare a winner
- **Updates the map** to show both reported votes and prediction confidence intervals

Flows linearly: pre-election (model predictions only) → poll-open (predictions + early vote tallies) → results-in (predictions + live results + accuracy) → called (posterior p(D wins) > threshold).

---

## 2. Data Sources

### Primary: Manual Entry (Recommended)

Why: AP Data API ($500+/month), NYT API (non-commercial only). Manual entry is lower cost and sufficient for a one-person MVP.

- **Input**: Web form + CSV upload
  - Form: county FIPS, race (e.g., "FL_Senate"), reported_dem_votes, reported_gop_votes, reported_turnout, timestamp, pct_reporting
  - CSV: batch ingest for multi-county updates (state election officials publish county result CSVs)
  - Validation: FIPS must exist in database; turnout must be plausible (state-level aggregate check)

- **Storage**: DuckDB table `live_results` (immutable append-only)
  - Columns: county_fips, race, reported_dem_votes, reported_gop_votes, reported_turnout, pct_reporting, ingested_at, source
  - No updates — only new rows. Historical accuracy computed from snapshot deltas.

### Secondary: AP Data API (Post-MVP)

If budget allows:
- Real-time county result feed
- Automatic ingestion on a 5min poll loop
- Requires API authentication in `api/main.py` startup

---

## 3. Key UI Components

### Live Map + Result Overlay (Left Panel)

- **Background**: Same choropleth county/tract map (colored by model super-type)
- **Overlay bars**: For each county, a horizontal bar showing:
  - Model prediction (pin position, e.g., 43% Dem)
  - Reported result (color fill, e.g., 45% Dem reported)
  - Confidence interval (whisker, e.g., 38–48% 90% CI)
- **Hover**: Show county name, prediction, reported result, pct reporting, error

### Race Cards (Right Panel, Live Tab)

For each tracked race (e.g., "FL Senate"):
- **Header**: Race name + timestamp of latest report
- **State-level summary**:
  - Current state prediction (model mean + 90% CI)
  - Reported state result (weighted by votes reported)
  - Posterior probability D wins (updated via Bayesian update)
  - Call status (undecided / leaning D / leaning R / called)
- **County breakdown**:
  - Top 5 counties (by votes cast, not percentage)
  - Each row: county name | model pred | reported result | error | pct reporting

### Prediction Accuracy Tracker (Bottom Panel)

- **Metric 1**: Across all races, how many calls were correct?
  - E.g., "12/15 races called correctly (80%)"
  - Disaggregated by race type (Senate vs Governor)
- **Metric 2**: Average error in non-called races
  - E.g., "Mean |actual - predicted| = 1.2 pp"
- **Metric 3**: Calibration: "Did 95% CI actually contain 95% of outcomes?"
  - Bin predictions by confidence level (50–60%, 60–70%, …, 90–100%)
  - Check if actual outcomes fall within predicted CI at claimed rate

---

## 4. API Design

### New Endpoints

```
POST /live/ingest
  Input: { county_fips: str, race: str, dem_votes: int, gop_votes: int,
           turnout: int, pct_reporting: float, timestamp: ISO8601 }
  Output: { ingested_id: str, validation_status: "ok" | "warning" | "error", message: str }

GET /live/results?race=FL_Senate&state=FL
  Output: list[{ county_fips, county_name, reported_dem_share, reported_gop_share,
                 reported_turnout, pct_reporting, ingested_at }]

GET /live/accuracy?race=FL_Senate
  Output: { called: bool, correct: bool, pred_dem_share: float, actual_dem_share: float,
            error_pp: float, posterior_p_dem: float, call_threshold: float }

GET /live/race-calls
  Output: list[{ race, called: bool, winner: "D" | "R" | null, p_dem_win: float,
                 timestamp: ISO8601 }]

POST /live/batch-ingest
  Input: CSV file with rows [county_fips, race, dem_votes, gop_votes, turnout, timestamp]
  Output: { ingested_count: int, errors: list[str] }
```

### Validation & Idempotency

- **Idempotent ingestion**: Each (county_fips, race, timestamp) tuple is unique. Re-posting the same result returns same ingested_id without duplicate rows.
- **Plausibility checks**:
  - Turnout: must be 20% ≤ turnout ≤ 80% (state-level aggregate must be within ACS/historical bounds)
  - Vote totals: dem_votes + gop_votes ≤ turnout × eligible_voters (sanity check)
  - Monotonicity: reported votes only increase over time for each (county, race) pair

---

## 5. Data Model

### New Tables (DuckDB)

```sql
CREATE TABLE live_results (
    ingested_id       VARCHAR PRIMARY KEY,
    county_fips       VARCHAR NOT NULL,
    race              VARCHAR NOT NULL,
    reported_dem_votes   INTEGER,
    reported_gop_votes   INTEGER,
    reported_turnout     INTEGER,
    pct_reporting        FLOAT,
    ingested_at          TIMESTAMP NOT NULL,
    source               VARCHAR,  -- 'manual', 'ap_api', 'csv_upload'
    UNIQUE (county_fips, race, ingested_at)
);

CREATE TABLE live_race_calls (
    race              VARCHAR PRIMARY KEY,
    called_at         TIMESTAMP,
    winner            VARCHAR,  -- 'D', 'R', null
    p_dem_win         FLOAT,     -- posterior prob at call time
    notes             VARCHAR,
    FOREIGN KEY (race) REFERENCES races(race)
);

CREATE TABLE live_accuracy_log (
    race              VARCHAR NOT NULL,
    county_fips       VARCHAR,
    pred_dem_share    FLOAT,
    actual_dem_share  FLOAT,
    error_pp          FLOAT,
    posterior_p_dem   FLOAT,
    pct_reporting     FLOAT,
    logged_at         TIMESTAMP,
    FOREIGN KEY (race) REFERENCES races(race)
);
```

### Computed Views (Materialized on Refresh)

- `live_state_results`: Latest reported result aggregated to state level (weighted by county votes)
- `live_posterior`: Bayesian posterior after each new county report (via Ridge priors + observed likelihood)
- `live_call_eligible`: Races where p_dem_win > call_threshold (95% default)

---

## 6. Implementation Phases

### Phase 1: Core Data Pipeline (Week 1)

1. **Create `live_results` table** in DuckDB
2. **Write `POST /live/ingest` endpoint** with validation
3. **Manual CSV upload form** in frontend (simple file input → API)
4. **CSV parser** (`src/ingestion/parse_election_csv.py`) to validate and batch-insert
5. **Test**: Ingest 2024 actual results as a dry run; compare to predictions made in 2024

**Output**: Can manually upload county results; API stores them correctly.

### Phase 2: Accuracy Computation (Week 2)

1. **Bayesian posterior updater** (`src/live/bayesian_update.py`)
   - Input: county-level live result + prior (from predictions table)
   - Output: updated posterior p(D wins | results so far)
   - Method: Gaussian update on logits (same as poll propagation, but with vote share → proportion conversion)

2. **`GET /live/accuracy` endpoint** — compute error for each race in real-time

3. **Accuracy aggregator** (`src/live/accuracy_metrics.py`)
   - Calibration curve (did 90% CI contain 90% of outcomes?)
   - Mean error by race type

**Output**: Can see how well predictions performed as results flow in.

### Phase 3: Live UI (Week 3)

1. **Live tab in race detail page** (`web/app/forecast/[slug]/live.tsx`)
   - Race card with state-level summary (pred | reported | posterior p_dem)
   - County table (top counties by votes, with errors)

2. **Live map overlay** (`web/components/LiveMapOverlay.tsx`)
   - Horizontal bars (prediction | reported | CI whisker)
   - Hover tooltip with county name, error, pct reporting

3. **Accuracy tracker** (bottom of Live tab)
   - Call accuracy (X/15 correct)
   - Average error

4. **Manual entry form** (modal, triggered from Live tab)
   - County FIPS input (with autocomplete)
   - Vote counts + turnout
   - Submit → POST /live/ingest

**Output**: Can see live results, accuracy, and manually enter county data on election night.

### Phase 4: Race Calls & Notifications (Week 4)

1. **Call logic** (`src/live/call_races.py`)
   - When p(D wins) > threshold for a race, record call (race, winner, timestamp)
   - Store in `live_race_calls` table

2. **`POST /live/call` endpoint** — manually or automatically trigger race call
   - Validates that posterior p > threshold
   - Notifies subscribers (webhook, email, Telegram)

3. **Frontend badge** — show "CALLED" with winner on race card
   - Gray if still undecided, green/red if called

4. **Telegram notification** (integration via `~/scripts/notify.sh`)
   - Message format: "CALLED: FL Senate — Democrat wins (p=0.98)"

**Output**: Race calls with confidence, notifications on election night.

---

## 7. Manual Data Entry UX

**Simplest flow for election night (no AP API):**

1. Open WetherVane at 8pm on election night
2. Click "Report Results" button in Live tab
3. Modal appears:
   - Dropdown: "Select county" (autocomplete on FIPS or name)
   - Dropdown: "Select race" (FL_Senate, GA_Senate, etc.)
   - Text inputs: dem_votes, gop_votes, turnout, pct_reporting
   - Submit button
4. API validates, stores result, recomputes posterior, updates map & race card in real-time
5. State officials release CSV → copy-paste into "Batch Upload" button
   - CSV parsed, all counties updated in one request

**Cost**: ~100 lines of frontend + 50 lines of backend validation = 2–3 hours.

---

## 8. Post-MVP Enhancements

- **AP Data API integration**: Real-time polling (5min intervals) with automatic ingestion
- **State-level aggregates**: Match state election official websites for cross-validation
- **Forecast accuracy deep-dive**: "Which demographic types did we miss? Why?"
- **Swing shift analysis**: "This county swung 7 points vs 2024 — anomaly or pattern?"
- **Next-cycle feedback**: Export accuracy metrics to train next election cycle's priors

---

## 9. Success Criteria

By November 5, 2026, election night:

- [ ] Can manually enter county results without errors
- [ ] Live map updates in real-time (< 2sec latency)
- [ ] Posterior probabilities sensible (p moves toward 0 or 1 as results flow in)
- [ ] Accuracy metrics computed correctly (checked against 2024 actuals)
- [ ] All 15 races trackable simultaneously
- [ ] No database locking (concurrent writes safe)
- [ ] Notifications fire on race calls

---

## 10. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Manual entry errors | Validation layer + state-level aggregate sanity check |
| Database locks during ingestion | Read-only schema for frontend; live_results append-only, non-blocking |
| Posterior drift from early returns | Use inverse-variance weighting; down-weight early partial counts |
| Missed calls (false negatives) | Set threshold conservatively (95%); audit calibration post-election |
| Wrong calls (false positives) | Audit: was prior well-calibrated? Were demographic shifts anticipated? |

---

## 11. File Structure (Post-Implementation)

```
wethervane/
├── src/live/
│   ├── __init__.py
│   ├── bayesian_update.py        # Posterior update logic
│   ├── accuracy_metrics.py        # Calibration, error computation
│   ├── call_races.py              # Call logic + thresholds
│   └── notification.py            # Telegram/webhook integration
├── src/ingestion/
│   └── parse_election_csv.py      # CSV parsing & validation
├── api/routers/
│   └── live.py                    # /live/ingest, /live/accuracy, /live/calls
├── web/
│   ├── app/forecast/[slug]/
│   │   └── live.tsx               # Live tab UI
│   └── components/
│       ├── LiveMapOverlay.tsx      # Map result overlay
│       └── LiveResultForm.tsx      # Manual data entry modal
├── tests/
│   ├── test_live_ingest.py        # API validation tests
│   ├── test_bayesian_update.py    # Posterior computation tests
│   └── test_accuracy_metrics.py   # Calibration tests
└── docs/
    └── election-night-operations.md # Playbook for election night
```

---

## 12. Example: Election Night Walkthrough

**8:00 PM ET (polls close in FL)**

- WetherVane shows 15 races: all "undecided"
- Pre-election predictions visible: FL Senate 43.2% D (38–48% CI)

**8:15 PM ET**

- Early vote (20%) reported in Broward County: 58% D
- User clicks "Report Results" → enters data manually
- API updates posterior p(FL D wins) from 0.38 → 0.51
- Map updates: Broward bar shows 58% reported vs 43% predicted

**9:00 PM ET**

- 60% of votes reported statewide
- User bulk-uploads CSV with 30+ counties
- Posterior p(FL D wins) jumps to 0.72
- Race card highlights "FL Senate" as "leaning D"

**10:30 PM ET**

- 95% reported, p(FL D wins) = 0.98
- Telegram notification: "CALLED: FL Senate — Democrat wins"
- Frontend shows green badge "CALLED" with final result

**11:45 PM ET**

- Post-election dashboard: 14/15 races called correctly (93%)
- Calibration chart: "90% CIs contained actual result 91% of the time — well-calibrated"
- Average error: 1.1 percentage points

---

**Author's note:** This design prioritizes simplicity and manual control. A one-person team can execute it in 4 weeks, focusing on manual data entry for MVP and deferring expensive API integrations to post-election feedback.
