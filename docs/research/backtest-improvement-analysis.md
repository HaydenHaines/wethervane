# Backtest Improvement Analysis

**Date**: 2026-04-08
**Context**: Based on the 11-election backtest harness results (S530)

## What the Backtest Results Tell Us

### The Good
- **Direction accuracy is excellent** (88-100% across all elections)
- **Modern elections are well-modeled**: 2016+ presidential r > 0.89, bias < 2pp
- **Governor 2022 is near-perfect** (r=0.972) — validates the blended-prior approach
- **Senate consistently strong** (r=0.831-0.960) — the poll propagation through type structure works

### The Diagnostic Pattern
The temporal gradient (r improves monotonically toward 2024) reveals the model's main structural limitation: **the Ridge priors are anchored to the 2024 political landscape**. This means:

1. **2008/2012 D bias (-6 to -8pp)**: The 2024 priors "know" that rural counties are deep R and suburban counties shifted blue. In 2008, this wasn't yet true — Obama's coalition was structured differently. The model projects 2024-aligned priors backward, making every D prediction too Republican (hence negative bias = underpredicting Dems).

2. **2016 is the inflection point** (bias ≈ 0): Trump's realignment roughly matches the 2024 political map. The model's priors are "correct" for the 2016+ era.

3. **The gradient measures electoral realignment, not model error**: r=0.826 for 2008 isn't bad — it's the model honestly reflecting that the political geography was different 16 years ago.

## Improvement Strategies

### Strategy 1: Year-Adaptive Ridge Priors (HIGH IMPACT, MODERATE EFFORT)
**The idea**: Instead of training Ridge priors on 2024 data alone, train priors on data from the target election cycle's era. For backtesting 2008, use 2004-2008 data; for 2012, use 2008-2012; etc.

**Why it helps**: The -8pp bias in 2008 comes from the priors being structurally wrong about which counties are D vs R. Year-matched priors fix this directly.

**Implementation**:
- The `assemble_national_presidential_actuals.py` script (Hayden's commit) already generates per-year county actuals
- Train separate Ridge models on each cycle's demographic data → county outcomes
- Backtest harness selects the appropriate prior for each year
- **This is the single biggest lever** — it directly addresses the temporal gradient

**Caveat**: This changes the model for backtesting purposes only. For 2026 predictions, we'd still use 2024-trained priors (which is correct).

### Strategy 2: Fundamentals Slider Tuning (MODERATE IMPACT, LOW EFFORT)
**The idea**: Use the backtest as a calibration target for the `fundamentals_weight` parameter.

**What we have**: `prediction_params.json` has `fundamentals_weight: 0.3` (30% fundamentals, 70% generic ballot). Hayden added `snapshot_2022.json` and `snapshot_2024.json`.

**What to do**:
- Generate fundamentals snapshots for each backtest year (approval, GDP, unemployment, CPI)
- Sweep `fundamentals_weight` from 0.0 to 1.0 across all backtest years
- Find the weight that minimizes aggregate backtest RMSE
- This directly implements Hayden's vision of "optimal mix of polling/fundamentals/demo/candidate effect sliders"

**The sliders to tune**:
1. `fundamentals_weight` — how much to trust economic fundamentals vs generic ballot
2. `lam` (λ) — θ_national regularization (how much polls override priors)
3. `mu` (μ) — δ_race regularization (how much race-specific signal overrides national)
4. `half_life_days` — poll time decay
5. `_POLL_BLEND_SCALE` (k=5) — county prior vs type projection blend rate

### Strategy 3: Temporal Prior Blending (MODERATE IMPACT, MODERATE EFFORT)
**The idea**: Instead of Ridge priors from one election, blend priors from the N most recent elections, with exponential decay weighting.

**Why**: The 2024 priors capture the current alignment but miss historical patterns. A blended prior (e.g., 60% 2024, 25% 2020, 10% 2016, 5% 2012) would be more robust and partially address the temporal gradient.

**For backtesting**: Use all elections up to (but not including) the target year. For 2020 backtest, blend 2016+2012+2008 priors.

### Strategy 4: Type Definitions Are Already Good — Don't Change Them
**The temptation**: "Maybe types trained on 2008-2024 data would be different from types trained on just 2016-2024?"

**Why not**: KMeans on shift vectors already spans the full period. The types ARE the structural communities. The temporal gradient isn't from wrong types — it's from wrong priors about what those types do. The types capture the structure; the priors capture the current expression.

**Evidence**: Even in 2008 (r=0.826), the type structure provides enough information for the model to get direction right 90% of the time. The types work. The priors are what's temporally anchored.

### Strategy 5: Cross-Cycle Poll Calibration (LOW IMPACT, HIGH EFFORT)
**The idea**: Different eras had different polling biases. 2020 had a notorious D overcount. 2016 underestimated Trump. Calibrate poll weighting per era.

**Why lower priority**: Our backtest uses 538's final model averages, which already incorporate their bias corrections. Adding our own era-specific correction on top would be double-counting.

## Recommended Sequence

1. **Year-adaptive priors** (Strategy 1) — this is the 80/20 play. Fix the biggest source of temporal degradation.
2. **Slider sweep** (Strategy 2) — use the harness to optimize all 5 tunable parameters simultaneously. Grid search or Bayesian optimization.
3. **Temporal blending** (Strategy 3) — if Strategy 1 alone isn't sufficient, blend instead of switching.

## What NOT to Change
- **Type definitions** (J=100, KMeans, shift vectors) — these are structural and stable
- **Covariance matrix** — validated at r=0.915, working well
- **Poll propagation mechanics** — θ_national + δ_race decomposition is sound
- **County-residual blending** (k=5) — this fixed the type-compression bug and is working correctly

## Expected Outcome
With year-adaptive priors, backtest r should improve substantially for 2008/2012 (from 0.82→0.88+) and bias should drop below 3pp for all years. The temporal gradient should flatten but not disappear entirely — some gradient is real (earlier elections had genuinely different dynamics).

The slider sweep should find optimal parameters that minimize aggregate backtest RMSE. The current defaults (lam=1, mu=1, k=5, half_life=30) were calibrated on limited data — a full 11-election backtest is a much richer calibration target.
