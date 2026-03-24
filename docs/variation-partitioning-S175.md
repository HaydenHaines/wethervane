# Variation Partitioning: Types vs Demographics (S175)

## Question

How much do electoral types explain in 2020→2024 presidential shift variance,
compared to demographics alone — and is there unique information in the types
beyond what demographics capture?

## Setup

- **Counties**: 293 (FL + GA + AL)
- **Training dimensions**: 39 shift pairs (2008+, presidential×2.5 + state-centered gov/Senate)
- **Holdout**: 2020→2024 presidential shift (3 dimensions: D-share, R-share, turnout)
- **Type count (J)**: 43 (KMeans, production model)
- **Demographic features**: 40 (ACS, Census, RCMS, urbanicity, migration)

## Three Models

| Model | Description |
|-------|-------------|
| **Types only** | County baseline (mean training shift) + type covariance adjustment — exact production method |
| **Demographics only** | County baseline + RidgeCV on ACS/Census/RCMS/urbanicity/migration features |
| **Types + Demographics** | County baseline + RidgeCV on [type scores \| demographic features] combined |

All three models share the same county baseline (mean of training-column shifts).
The comparison isolates what each predictor adds beyond the baseline.

## Results

### Per-Dimension R²

| Dimension | Types | Demographics | Combined |
|-----------|-------|--------------|----------|
| `pres_d_shift_20_24` | 0.6721 | 0.6219 | 0.7316 |
| `pres_r_shift_20_24` | 0.6400 | 0.2531 | 0.7620 |
| `pres_turnout_shift_20_24` | 0.7441 | 0.7675 | 0.8369 |

### Aggregate (mean across holdout dimensions)

- R²(types only): **0.6854**
- R²(demographics only): **0.5475**
- R²(combined): **0.7768**

### Variance Partition

| Component | Fraction | Percentage |
|-----------|----------|------------|
| Unique to types | 0.2293 | 22.9% |
| Unique to demographics | 0.0914 | 9.1% |
| Shared (types + demo) | 0.4561 | 45.6% |
| Residual (unexplained) | 0.2232 | 22.3% |
| **Total** | **1.0000** | **100%** |

## Interpretation

Types explain **68.5%** of holdout variance on their own. Demographics explain **54.7%**. Together they explain **77.7%**.

Of the 77.7% explained by the combined model:
- **22.9%** is unique to types (types explain this, demographics cannot)
- **9.1%** is unique to demographics (demographics explain this, types cannot)
- **45.6%** is shared (both approaches capture it)

**Types add value beyond demographics.** The 22.9pp unique contribution means the KMeans type structure captures electoral behavior patterns that cannot be recovered from demographic proxies alone. This is consistent with the model's design: types are discovered from *how places shift*, not from who lives there — so they carry information about behavioral patterns that demographics approximate but do not fully explain.

## Methodology Notes

- **No data leakage**: demographic features are not derived from the holdout elections.
- **Ridge regression** with cross-validated alpha (RidgeCV) is used for demographics — a robust
  linear model appropriate for ~30 features on 293 counties.
- **Same county baseline** for all three models: mean of training-column shifts per county.
  This isolates the question to what each predictor adds to the baseline.
- **Shared variance** can be negative (suppressor effects) — this is mathematically valid.
- The types-only model uses the exact same method as `holdout_accuracy_county_prior` in
  `src/validation/validate_types.py`. The R² here should match the validation report.

## Files

- Script: `scripts/variation_partitioning.py`
- Tests: `tests/test_variation_partitioning.py`
- Generated: 2026-03-23 (Session 175)
