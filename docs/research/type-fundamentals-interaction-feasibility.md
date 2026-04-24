---
title: Type×Fundamentals Interaction Feasibility
date: 2026-04-23
status: Blocked on data
priority-item: "#88 remaining"
related: S339 (type-fundamentals NEGATIVE), phase5-fundamentals-design, governor-backtest-2022-S492
---

# Type×Fundamentals Interaction Feasibility

## Question

Can we add Type×fundamentals interaction terms to the governor Ridge or
county-priors Ridge so that fundamentals hit types differently (e.g., unemployment
shifts manufacturing types more than knowledge-worker types), instead of a flat
national shift?

## Context

- #88 first pass added **state-level** QCEW employment/wage as additional features.
  Governor backtest r improved 0.745 → 0.754 (+0.010) as linear main effects.
  See `src/prediction/state_economics.py`.
- S339 documented type×national-fundamentals interactions as **net-negative** at
  the county level — confirmed in
  `scripts/experiments/exp3_feature_interactions.py`
  (type×demographic interactions hurt LOO r for presidential prediction at
  N≈3,100). That result is for presidential cycles; this note is about the
  **midterm/cycle-specific** interaction for governor/Senate forecasting.
- The phase-5 fundamentals spec (`docs/superpowers/specs/2026-03-27-phase5-fundamentals-design.md`)
  explicitly identifies type-level fundamentals as the target design, with the
  F-004 open question: "should manufacturing types be hit differently from
  agricultural types during an inflation shock?" — flagged as a Phase 2
  research item pending type-level economic characterization.

## The N=3 Constraint

The backtest harness (`src/validation/backtest_harness.py`) supports:

| Race type | Midterm cycles usable |
|-----------|-----------------------|
| Governor  | 2010, 2018, 2022 — **N=3** |
| Senate    | 2010, 2014, 2018, 2022 — N=4 cycles × ~33 seats/cycle |
| President | 2008, 2012, 2016, 2020 — not midterms |

Governor is the cleanest midterm signal because every state has one every 4
years (albeit staggered), but only 3 cycles of observations exist in the harness.

### Why N=3 is insufficient for interactions

- **National fundamentals vary by cycle only.** With 3 cycles, the fundamentals
  vector takes 3 distinct values. An interaction term Type×Fund (for a single
  fundamentals scalar) can therefore fit at most a rank-3 signal across types:
  the interaction coefficient for each type is identified only from how that
  type's residual varies across 3 points.
- **With J=100 types and k=5 fundamentals features**, the full interaction
  block is 500 coefficients. Even with 50 states × 3 cycles = 150 state-cycle
  observations and heavy Ridge regularization, the effective rank of the
  interaction signal is capped at k·(cycles−1) = 10 — far below 500. The
  regression will just shrink 490 coefficients to zero.
- Type×demographic interactions failed at county-level presidential (N≈3,100,
  no cycle variation) for a different reason (overfitting vs. structural
  rank). The midterm case is rank-limited, not overfit-limited.

## What would unblock this

**Either** expand the midterm observation count, **or** give the fundamentals
themselves state-level variation so the effective interaction rank is not
capped by `cycles − 1`.

### Option A — Backfill QCEW to 2008-2019 (state-level fundamentals variation)

QCEW on disk (`data/raw/qcew_county.parquet`) covers only 2020-2023:

```
Shape: (104439, 7)
Years: [2020, 2021, 2022, 2023]
```

BLS publishes QCEW county data back to 1975
(<https://www.bls.gov/cew/downloadable-data-files.htm>). Pulling 2008-2019
(adding ~12 more years × ~3,200 counties) would give **per-state** employment
and wage growth for the 2010, 2014, 2018 midterm cycles. Then the
"fundamentals" vector per state-cycle has 50 × 3 = 150 distinct values — the
rank limit disappears and the interaction is identified in the usual Ridge
sense.

Estimated work: 1-2 sessions (BLS has bulk CSVs by year; follow the same
build pipeline used in `scripts/fetch_bea_state_data.py`).

Risk: QCEW industry classification changed with the 2012 NAICS revision. The
NAICS 2012 → 2017 change is minor but 2008 data uses NAICS 2007 for some
series. Build should version-tag industry codes per year.

### Option B — Add more midterm cycles via VEST/DRA pre-2010

DRA has no 2010 or most 2014 state data (CLAUDE.md gotcha, GH#95 research).
VEST coverage is better pre-2016 but has the crosswalk distortion issues
documented in GH#95 findings. Adding 1994/1998/2002/2006 midterms via the
Algara dataset (already used for pre-2000 governor) gets us to N=7 midterms
but without tract-level outcomes — only state-level. That's still useful for
state-level fundamentals interactions (50 states × 7 cycles = 350 obs).

Limitation: national fundamentals go back to 1974
(`data/raw/fundamentals/midterm_history.csv` has 13 cycles) but only include
5 variables. Type scores are derived from shift patterns that don't exist
pre-2008 at tract level, and even state-level type-mixing is an approximation
pre-2000 since county boundaries and VRA geography shift.

### Option C — Senate across cycles (N≥4)

Adding 2014 Senate to the backtest harness would give N=4 midterms for
Senate. Senate seats are only ~33/cycle, so total state-cycle Senate obs ≈
132 — larger than governor but not dramatically so. Rank limit becomes
k·(cycles−1) = 15, still far below 500.

### Option D — Wait for 2026

The cleanest unblocker is the 2026 midterm. After 2026 the governor backtest
grows to N=4 and we can test interactions with an out-of-sample cycle. This
is the path that Hayden's priorities note implies ("needs more midterm data").

## Recommendation

Do not attempt to fit Type×fundamentals interactions with the current data —
the effective rank of the interaction signal is at most k·(cycles−1) ≤ 15 and
the pipeline will just zero the coefficients. Low-value experiment.

Priority-ordered ways to unblock, in decreasing ROI:

1. **Option A (QCEW backfill)** — gives true state-level fundamentals variation.
   Best ROI because it removes the rank cap immediately and reuses existing
   `src/prediction/state_economics.py`. Main risk is NAICS-version inconsistency.
2. **Option D (wait for 2026)** — zero work but zero progress until the next
   midterm.
3. **Option C (add 2014 Senate)** — modest data expansion, marginal improvement.
4. **Option B (pre-2010 governor via Algara)** — already discussed and
   largely exhausted by the pre-2000 data pull; would only help if combined
   with Option A.

**Next concrete step if this becomes priority 1:** write a `scripts/fetch_qcew_historical.py`
that pulls BLS QCEW 2008-2019 by year, normalizes NAICS codes, and appends to
the existing `qcew_county.parquet`. Then re-run `state_economics.build_state_econ_features`
for each midterm year and compare governor-backtest r at N=3 with and without
the type×state-econ interaction.

## Disposition

Leaving `#88 remaining` in priorities.md as-is (Hayden-hand-edited). This note
documents the blocker empirically so the next autonomous agent that picks up
this item has the feasibility analysis already done.
