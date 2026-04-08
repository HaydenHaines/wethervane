# Model Verification Plan

**Created**: 2026-04-08, S529
**Purpose**: Automated, iterative model verification so Hayden doesn't have to spot-check predictions manually.

## Problem

The type-compression bug (#139) went undetected because predictions *looked* reasonable at a glance (near 50/50 for unpolled races) and no automated checks existed to flag that ~20 Senate races were predicting near-tossup regardless of actual partisan lean. Manual spot-checking caught it only when Hayden happened to look at NJ.

## Design Principles

1. **Fail the build, not the deploy.** Sanity checks run inside `build_database.py` and block bad predictions from reaching the DuckDB that the API serves.
2. **Test behavior, not implementation.** Checks assert properties of predictions (spread, direction, correlation), not internal model math.
3. **Layered severity.** Tier 1 (hard constraints) blocks builds. Tier 2 (soft constraints) warns but doesn't block. Tier 3 (regression tracking) logs metrics for trend monitoring.
4. **No manual intervention required.** Every check runs automatically during the prediction → build → deploy pipeline.

## Three-Tier Verification Architecture

### Tier 1: Build-Time Gate (blocks bad builds)

These run inside `build_database.py` after predictions are ingested. If any fail, the build exits with status 1 and the DuckDB is not updated.

| Check | Threshold | Rationale |
|-------|-----------|-----------|
| **Prediction spread** | std > 0.05 across states | Type-compression produces std ~0.02 |
| **Safe D states above floor** | MA, IL, RI, DE > 0.53 | These states haven't gone R in decades |
| **Safe R states below ceiling** | WY, WV, OK, ID < 0.40 | These states haven't gone D in decades |
| **Both D and R predictions exist** | At least 1 state > 0.55 AND 1 < 0.45 | Model should predict both directions |
| **NJ canary** | NJ Senate > 0.50 | NJ is D+16; predicting R is a structural bug |
| **Cross-state correlation** | r > 0.70 with 2024 presidential | Predictions should track known partisan lean |
| **Non-degenerate predictions** | No state with pred exactly 0.50 ± 0.001 | Exactly 0.50 means the model didn't run |
| **County prediction count** | > 2,500 counties per race | Missing counties = data pipeline failure |

**Implementation**: `src/db/validate.py::validate_predictions()` — called from `build()` after `_ingest_data()` and before `_report_summary()`.

### Tier 2: pytest Sanity Suite (run before push)

These are the existing 13 tests in `tests/test_model_sanity.py`, plus expansions:

| Test Class | Tests | Status |
|-----------|-------|--------|
| `TestSafeSeats` | 8 safe D/R states | ✅ Implemented (S528) |
| `TestPredictionSpread` | std floor + both-sides check | ✅ Implemented (S528) |
| `TestNJRegression` | NJ > 0.50 + margin > 2pp | ✅ Implemented (S528) |
| `TestCrossStateCorrelation` | r > 0.80 with 2024 pres | ✅ Implemented (S528) |
| `TestGovernorSanity` | Safe gov predictions (new) | ❌ To implement |
| `TestSenateSeatCount` | 33 or 34 Senate races predicted | ❌ To implement |
| `TestGovernorRaceCount` | 36 governor races predicted | ❌ To implement |
| `TestPolledRaceStability` | Polled races within 5pp of poll average | ❌ To implement |
| `TestUnpolledRaceDirection` | Unpolled races match Ridge prior direction | ❌ To implement |

These tests run via `uv run pytest tests/test_model_sanity.py` and are enforced by the pre-push hook.

### Tier 3: Metric Tracking (logged, not blocking)

After each prediction regeneration, log key metrics to `data/predictions/metrics_log.jsonl`:

```json
{
  "timestamp": "2026-04-08T09:55:00",
  "session": "S528",
  "senate_pred_std": 0.087,
  "senate_pred_mean": 0.478,
  "dem_seats": 48,
  "gop_seats": 52,
  "nj_pred": 0.531,
  "ga_pred": 0.513,
  "cross_state_r": 0.92,
  "n_polls": 204,
  "n_races_polled": 27,
  "n_races_total": 33
}
```

This enables detecting slow drift: if NJ goes from D+6 to D+4 to D+2 over three regenerations, the trend is visible even though no single regeneration triggers a Tier 1 failure.

**Implementation**: Append to JSONL after `predict_2026_types.py` completes. No blocking behavior.

## Pipeline Integration Points

```
predict_2026_types.py
  └── writes county_predictions_2026_types.parquet
  └── appends to metrics_log.jsonl (Tier 3)

build_database.py
  ├── create_schema()
  ├── ingest_data()          ← predictions loaded into DuckDB
  ├── validate_predictions() ← NEW: Tier 1 checks (blocks on failure)
  ├── validate_integrity()   ← existing contract validation
  └── report_summary()

pre-push hook
  └── pytest tests/test_model_sanity.py  ← Tier 2

cron (Sun+Wed 4:07am)
  └── scrape polls → predict → build → deploy
  └── entire pipeline runs; Tier 1 blocks bad deploys
```

## What This Catches

| Bug Type | Example | Caught By |
|----------|---------|-----------|
| Type compression | All unpolled races → 0.49 | Tier 1: spread + safe seats |
| Broken prior loading | Wrong J loaded, priors all 0.5 | Tier 1: spread + non-degenerate |
| Poll pipeline failure | No polls loaded, all predictions = baseline | Tier 2: polled race stability |
| Data pipeline failure | Missing counties or races | Tier 1: county count + race count |
| Sign error in shift | D shift applied as R | Tier 1: safe seats + correlation |
| Stale DuckDB | J=130 data loaded for J=100 model | Tier 1: county count mismatch |
| Slow drift | NJ trending R over weeks | Tier 3: metrics log |

## Implementation Priority

1. **Now (S529)**: Wire Tier 1 checks into `build_database.py`. This is the highest-value change — it prevents bad predictions from going live.
2. **Next session**: Expand Tier 2 pytest suite (governor sanity, race counts, polled race stability).
3. **When convenient**: Add Tier 3 metrics logging to `predict_2026_types.py`.

## Maintenance

- When adding new states to safe-seat lists, verify with at least 3 election cycles of data.
- When tuning thresholds, err on the side of loose (avoid false positives that block valid builds).
- When the model changes fundamentally (e.g., tract-primary migration), review all thresholds.
- The NJ canary test (#140) should be updated once NJ midterm historical analysis is complete.
