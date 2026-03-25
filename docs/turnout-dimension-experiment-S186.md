# Turnout Dimension Experiment — P3.5 (S186)

**Date:** 2026-03-24
**Branch:** feat/turnout-dimension-experiment
**Script:** `experiments/turnout_dimension_experiment.py`

## Summary

Turnout shift columns already exist in the production shift matrix. This experiment
tests whether they carry independent signal, and whether changing their weight alters
type quality. Result: **keep turnout at current weight (x1.0)**. The production
baseline is already optimal.

## Hypothesis

Turnout shifts might encode structural mobilization patterns distinct from partisan
realignment. If so, treating them differently (up-weight, down-weight, or isolate)
could improve KMeans type coherence and holdout prediction.

## Setup

- **Model:** KMeans J=43, T=10 soft membership, presidential x2.5
- **Training:** 2008+ shift pairs (39 dims total: 26 D/R + 13 turnout)
- **Holdout:** `pres_d_shift_20_24`, `pres_r_shift_20_24`, `pres_turnout_shift_20_24`
- **Metric:** Mean Pearson r across 3 holdout dims (county-prior protocol)
- **Baseline reported:** r=0.8428 (from HDBSCAN experiment P3.4, same protocol)

## Variants

| Variant | Description | Dims |
|---------|-------------|------|
| A_BASELINE | Production: all shifts, pres D/R x2.5, turnout x1.0 | 39 |
| B_NO_TURNOUT | D/R shifts only — turnout cols dropped | 26 |
| C_TURNOUT_X2 | All shifts, turnout up-weighted x2.0 | 39 |
| D_TURNOUT_X0.5 | All shifts, turnout down-weighted x0.5 | 39 |
| E_TURNOUT_X0 | Equivalent to B but with same column set (turnout zeroed) | 26 |
| F_TURNOUT_ONLY | Turnout shifts only, no D/R; pres turnout x2.5 | 13 |

## Results

| Variant | Dims | Holdout r | Delta | Coherence | pres_d r | pres_r r | turn r |
|---------|------|-----------|-------|-----------|----------|----------|--------|
| A_BASELINE | 39 | **0.8491** | +0.0063 | 0.7498 | 0.8358 | 0.8358 | 0.8758 |
| B_NO_TURNOUT | 26 | 0.8284 | -0.0144 | 0.7348 | 0.8358 | 0.8358 | 0.8137 |
| C_TURNOUT_X2 | 39 | 0.8415 | -0.0013 | 0.7772 | 0.8211 | 0.8211 | 0.8823 |
| D_TURNOUT_X0.5 | 39 | 0.8338 | -0.0090 | 0.7417 | 0.8396 | 0.8396 | 0.8222 |
| E_TURNOUT_X0 | 26 | 0.8284 | -0.0144 | 0.7348 | 0.8358 | 0.8358 | 0.8137 |
| F_TURNOUT_ONLY | 13 | 0.7899 | -0.0529 | 0.7141 | 0.7474 | 0.7474 | 0.8749 |

Deltas are vs. the prior-experiment baseline of r=0.8428.

## Findings

**1. Current production setup is already optimal.**
The baseline (A) achieves the highest holdout r=0.8491 across all variants. No
re-weighting improves it.

**2. Removing turnout hurts.**
Dropping turnout cols (B/E) drops holdout r by -0.021, mostly concentrated in
the `pres_turnout_shift_20_24` holdout dim (0.8137 vs 0.8758 with turnout).
This is the clearest signal: turnout shifts predict turnout shifts.

**3. Up-weighting turnout hurts partisan prediction without helping turnout.**
Turnout x2.0 (C) gains slightly on pres_turnout holdout (0.8823) but loses
on the partisan dims (0.8211 vs 0.8358). Net effect is -0.0013 — near-zero
but in the wrong direction.

**4. Turnout alone predicts surprisingly well.**
F_TURNOUT_ONLY reaches r=0.7899 using only 13 turnout dims with no D/R signal.
The pres_turnout holdout dim hits r=0.8749 — nearly as good as the full model.
This confirms turnout shifts encode genuine electoral structure (mobilization
communities), not just noise.

**5. Pres d/r holdout r is identical for B, E (0.8358) — same as baseline.**
The partisan type structure is not damaged by dropping turnout; it's just the
turnout holdout dimension that suffers. This confirms turnout and partisan shifts
are partially orthogonal signals that each contribute independently.

## Recommendation

**No change to production weighting.** The current equal-weight treatment of
turnout shifts alongside partisan shifts is already optimal. Specifically:

- Do NOT drop turnout dims (costs -0.021 in holdout r)
- Do NOT up-weight turnout (costs -0.001 net, loses partisan resolution)
- Do NOT down-weight turnout (costs -0.009)

The x1.0 weight reflects a natural balance where turnout contributes its
mobilization signal without overwhelming the partisan signal.

## What This Closes

This is the final P3.x clustering experiment:
- P3.1: J selection sweep (J=43 optimal)
- P3.2: Presidential weight sweep (x2.5 optimal)
- P3.3: State centering for gov/Senate (confirmed necessary)
- P3.4: HDBSCAN (failed decisively vs KMeans, r=0.53)
- P3.5: Turnout as separate dimension (no improvement — current practice optimal)

All clustering experiment slots exhausted. KMeans J=43 with presidential x2.5
and equal-weight turnout is the confirmed production configuration.

## Next Steps

- National expansion (more states, more data)
- Poll bias correction (Silver Bulletin ratings integration)
- Senate and governor propagation (not just presidential)
