# Republican Overestimation Analysis — S238

**Date:** 2026-03-27
**Requested by:** Hayden
**Status:** Research complete, no code changes — analysis only

---

## Executive Summary

The apparent Republican overestimation in 2026 WetherVane forecasts has **three distinct causes** with different severities. The most visible problem — the model showing D=35% for FL Senate when polls show D=45% — is primarily a **display bug** (county-unweighted mean vs population-weighted mean), not a modeling error. Once you account for population weighting, the model is largely tracking polls correctly in Florida. However, two genuine structural issues do exist and require action.

**The three issues, ranked by impact:**

| # | Issue | Severity | Fix Complexity |
|---|-------|----------|----------------|
| 1 | Display bug: state_pred is county-unweighted | High (visual gap ~10-15pp) | Simple |
| 2 | Stale type priors (J=20 file used for J=100 model) | High (corrupts Bayesian update) | Simple |
| 3 | Midterm baseline gap (model anchored to 2024 pres) | Medium (2-5pp structural gap) | Medium-Complex |

---

## Issue 1: The Display Bug (Primary Cause of Apparent R Overestimation)

### What Users See

The API `state_pred` field and the forecast page display the **county-unweighted mean** `AVG(pred_dem_share)` across all counties in a state. This reads as dramatically Republican-lean because:

- There are ~2,691 R-leaning counties vs ~415 D-leaning counties nationally (out of 3,106)
- But Democratic counties contain the majority of the population (urban areas)
- Miami-Dade County (D+70%, 1.4M votes) counts the same as rural Jefferson County (R+80%, 3K votes) in an unweighted average

### Magnitude

| Race | Displayed (unweighted) | Actual (pop-weighted) | Display Error |
|------|------------------------|----------------------|---------------|
| 2026 FL Senate | 35.0% | 46.0% | +11.0pp |
| 2026 GA Senate | 45.3% | 60.0% | +14.6pp |
| 2026 AL Governor | 30.1% | 32.7% | +2.6pp |

### Comparison to Polls

When using the correct population-weighted measure:

| Race | Model (pop-wt) | Poll Avg | Gap |
|------|----------------|----------|-----|
| 2026 FL Governor | 46.7% | 45.7% | -1.0pp (model slightly R) |
| 2026 FL Senate | 46.0% | 45.1% | -0.9pp (model slightly R) |
| 2026 GA Senate | 60.0% | 53.1% | -6.9pp (model too D) |
| 2026 AL Governor | 32.7% | 39.1% | +6.4pp (model too R) |

**The Florida races are tracking polls within 1pp when properly measured.** The "Republican overestimation" Hayden observed is the display bug in action.

### Root Cause in Code

From `api/routers/forecast.py` (line ~162):
```sql
SELECT AVG(p.pred_dem_share) AS state_pred, COUNT(*) AS n_counties
FROM predictions p JOIN counties c ON p.county_fips = c.county_fips
WHERE p.version_id = ? AND p.race = ? AND c.state_abbr = ?
```

This computes an unweighted county average. The fix requires either:
- Storing 2024 vote totals in the counties table for use as population weights
- Or computing a turnout-weighted `state_pred` during the `predict_race` step using 2024 vote counts

---

## Issue 2: Stale Type Priors (Corrupts Bayesian Update)

### What Happened

`data/communities/type_priors.parquet` was created on **March 21** when the model had J=20 types. The type assignments were rebuilt on **March 25** with J=100 types. The type priors file was never updated.

**Result:** The production pipeline loads type priors that:
1. Have entries for types 0-19 with **wrong values** (J=20 model type IDs don't correspond to J=100 type IDs)
2. Have **no entries** for types 20-99 — these all default to `0.45`

### Magnitude

| | Stored (stale) | Actual (from ridge model) | Error |
|-|----------------|---------------------------|-------|
| National mean type prior | 0.431 | 0.336 | +0.095 (too high) |
| Types with >20pp error | 7 of 20 stored, many unchecked | — | large |
| Types 20-99 default | 0.450 for all | 0.071 to 0.638 (varies) | up to ±37pp |

**Selected bad priors:**
- Type 0: stored 0.113, actual 0.512 — error = -0.399 (stored is too R when type is swing/D)
- Type 7: stored 0.616, actual 0.194 — error = +0.422 (stored is too D when type is very R)
- Types 22, 24, 26, 28: all stored 0.450, actual 0.179/0.164/0.248/0.220 (stored too D by +20-30pp)

### How This Affects Predictions

The type priors serve as the Bayesian update anchor. When a poll says D=45.1% for FL Senate:
- The Bayesian update computes: `type_shift = type_posterior - type_prior`
- `type_prior` here is the **stored stale value**, not actual 2024 results
- For types with badly wrong priors, the implied shift magnitude and direction are wrong
- The shifts propagate via county type scores to final predictions

**Empirical test — FL Senate with a D=45.1% poll:**
- With stored (stale) priors: pop-weighted = 45.2% (approximately matches polls, by coincidence)
- With correct priors computed from ridge model: pop-weighted = 53.5% (too D by +8.4pp)

The stale priors are accidentally producing better aggregate results in some states because the errors partially cancel. This cancellation is fragile — it will break for different poll values and states.

### Fix

Recompute `type_priors.parquet` to match the J=100 production model. The correct prior for type `j` is the score-weighted mean of ridge county priors across all counties:

```python
correct_type_priors[j] = sum(abs_scores[:, j] * ridge_pred) / sum(abs_scores[:, j])
```

This should be added as a step after `train_ensemble_model.py` runs. Currently there is no script that does this.

---

## Issue 3: Presidential Baseline for Midterm Elections

### The Problem

The county priors (`ridge_county_priors.parquet`) are predictions of **2024 presidential Dem share**, trained with `target = pres_dem_share_2024`. When used as baselines for 2026 midterm predictions:

- The 2026 electorate will differ from 2024 (lower turnout, different composition)
- Midterms typically favor the out-party; with Trump as president, Dems usually do better than in the prior presidential
- The presidential weighting (pw=8.0) means the 2024 Trump surge is baked into the type structure

### Magnitude of Midterm vs Presidential Gap

Comparing population-weighted 2024 presidential actuals to current 2026 polling:

| State | 2024 Pres (pop-wt) | 2026 Poll Avg | Expected D Improvement |
|-------|---------------------|---------------|----------------------|
| FL | 43.4% | 45.1% (Senate) | +1.7pp |
| GA | 48.9% | 53.1% (Senate) | +4.2pp |
| AL | 34.6% | 39.1% (Governor) | +4.5pp |

The model's Bayesian update with current polls partially corrects for this, but:
- Polls are sparse early in the cycle (4 FL polls, 15 GA polls as of March 2026)
- With sparse polls, the presidential baseline dominates the prior
- The Bayesian update "pulls" toward polls but doesn't fully close the gap

### Why pw=8.0 Makes This Worse

The presidential weight of 8.0 means the 2024 presidential shifts account for the majority of the KMeans clustering variance. Types are essentially defined by how they voted in 2024. When Trump overperformed in 2024 relative to historical trends (e.g., Black and Hispanic voters), the types absorbed that signal.

For 2026 midterms:
- Types anchored to Trump 2024 performance may regress toward their historical mean
- The model has no mechanism to represent "2026 backlash vs 2024 turnout model"
- A national generic ballot shift is the right correction, but it's not implemented

---

## Root Cause Summary (Hayden's Hypothesis)

Hayden suspected: *presidential weighting (pw=8.0) + Trump's strong 2024 showing → Republican overestimation in 2026.*

This is **partially correct** but the mechanism is more nuanced:

1. **pw=8.0 is NOT the direct culprit** for the displayed R bias — the display bug (unweighted mean) is responsible for the ~10pp visual gap.

2. **pw=8.0 DOES create a structural baseline problem**: the type structure is anchored to 2024 presidential results, and there's no correction for midterm environment. This creates a ~2-5pp residual R lean in the prior (before polls correct it).

3. **The stale type priors** (March 21 vs March 25) are actively corrupting the Bayesian update, making poll signal propagation unreliable.

4. **The real picture (population-weighted, with polls):**
   - FL Senate: model 46.0% vs polls 45.1% — essentially correct
   - GA Senate: model 60.0% vs polls 53.1% — model too D by 6.9pp (possibly type prior issue)
   - AL Governor: model 32.7% vs polls 39.1% — model too R by 6.4pp (midterm gap + sparse polls)

---

## Proposed Solutions

### Option A: Fix the Display Bug (Immediate, Simple)

**Change `state_pred` calculation to be population-weighted using 2024 vote totals.**

Implementation:
1. Add `total_votes_2024` to the `counties` table in DuckDB (from `medsl_county_presidential_2024.parquet`)
2. Change the API's `state_pred` SQL to use `SUM(pred * votes) / SUM(votes)` instead of `AVG(pred)`
3. Update the frontend to display population-weighted predictions

**Estimated effort:** 2-3 hours
**Impact:** Eliminates the 10-15pp visual R bias in displayed predictions

### Option B: Rebuild Type Priors (Immediate, Simple)

**Recompute `type_priors.parquet` to match the J=100 production model.**

Implementation:
1. Add a script `src/prediction/compute_type_priors.py` that computes score-weighted mean of ridge county priors per type
2. Run it after `train_ensemble_model.py` to keep them in sync
3. Save output to `data/communities/type_priors.parquet`

**Estimated effort:** 1-2 hours
**Impact:** Fixes corrupted Bayesian update; predictions will change (quantification needed before merging)

**Warning:** When correct priors are used, the pop-weighted FL prediction moves from 45.2% to 53.5% — further from polls. This suggests the Bayesian update math may need tuning after fixing the priors.

### Option C: National Generic Ballot Adjustment (Medium, Moderate)

**Apply a national D/R shift to all county baselines based on generic ballot polling.**

Implementation:
1. Track national generic ballot polls (separate from race-specific polls)
2. Compute expected national environment shift relative to 2024 presidential baseline
3. Apply as a flat shift to all county priors before running race-specific polls through the Bayesian update

Formula: `adjusted_county_prior = ridge_county_prior + national_gb_shift`

Where `national_gb_shift = mean(generic_ballot_polls) - 2024_presidential_national_dem_share`

**Estimated effort:** 4-6 hours
**Impact:** Corrects the ~2-4pp midterm structural gap; provides a principled national environment anchor

### Option D: Race-Type Prior (Medium, Better Than C)

**Compute separate priors per race type (Senate/Governor vs Presidential).**

Historical data shows Senate/Governor races differ from presidential by 2-6pp in swing states. Computing type priors from historical Senate/Governor results (not just presidential) would give a better midterm baseline.

Implementation:
1. For each type, compute the historical mean dem share for Senate/Governor elections separately from presidential
2. Use race-specific priors when generating Senate vs Governor forecasts

**Estimated effort:** 8-12 hours
**Impact:** Most principled fix; addresses both the midterm gap and the presidential weighting issue

### Option E: Separate Midterm Prediction Pipeline (Complex)

**Re-cluster types using Senate/Governor shift vectors weighted differently than presidential.**

The current pw=8.0 for presidential means 2024 Trump performance dominates type structure. A midterm-specific pipeline would use:
- Governor/Senate shifts weighted equally to presidential (pw=1.0 or 2.0)
- Or a completely separate KMeans run on Senate/Governor shifts only

**Estimated effort:** 20-40 hours
**Impact:** Most thorough fix but high risk of breaking the existing model's strong holdout metrics

---

## Recommendation

**Implement in this order:**

1. **Option B first (type priors rebuild)** — This is a correctness bug. The Bayesian update is working against wrong anchors. Fix it before anything else. But note: fixing it will change predictions and may make some predictions worse in the short term (until Option C is also applied).

2. **Option A second (display fix)** — The displayed state_pred values are misleading users. This is straightforward and has immediate UX impact.

3. **Option C third (generic ballot anchor)** — Once the prior is correct and display is fixed, the remaining gap is the midterm structural bias. Adding a national generic ballot adjustment will close this.

4. **Defer Option D/E** — These require careful evaluation to avoid degrading the model's strong holdout metrics (LOO r = 0.671). Re-clustering is a multi-session project.

---

## Implementation Notes

### Type Priors Rebuild Script

The script should:
```python
# For each type j:
# correct_prior[j] = sum(abs_scores[:, j] * ridge_county_prior) / sum(abs_scores[:, j])
# Save to data/communities/type_priors.parquet
# Update ridge_meta.json to include type priors hash
```

After the fix, re-run `predict_2026_types.py` and compare predictions to polls. Expect:
- FL Senate to move from 46.0% → somewhere between 45% and 55% (needs calibration)
- The Bayesian update may need `prior_weight` adjustment after correct priors are in place

### Population-Weighted state_pred

The `counties` table should be augmented with `total_votes_2024`. This field already exists in `data/assembled/medsl_county_presidential_2024.parquet` as `pres_total_2024`. Adding it to the DuckDB counties table is a one-line change to `src/db/build_database.py`.

### Generic Ballot Data

National generic ballot polls for 2026 are available from:
- FiveThirtyEight/Silver Bulletin house averages
- RealClearPolitics generic ballot tracker
- The `data/polls/polls_2026.csv` could accept a special race `"2026 Generic Ballot"` with geography `"national"`

---

## Files Affected

| File | Issue | Action |
|------|-------|--------|
| `data/communities/type_priors.parquet` | Issue 2 | Rebuild from J=100 ridge model |
| `src/prediction/compute_type_priors.py` | Issue 2 | Create new script |
| `src/db/build_database.py` | Issue 1 | Add `total_votes_2024` to counties |
| `api/routers/forecast.py` | Issue 1 | Fix `state_pred` SQL to population-weight |
| `src/prediction/predict_2026_types.py` | Issue 3 | Add optional generic ballot adjustment |
| `data/polls/polls_2026.csv` | Issue 3 | Add national generic ballot rows |
