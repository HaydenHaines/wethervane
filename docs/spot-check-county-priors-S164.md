# County-Prior vs Type-Mean Prior Spot Check — S164 (2026-03-22)

## Setup

**N counties:** 293  
**J types:** 43  
**Prediction mode:** Prior-only (no poll update — isolates baseline quality)  
**County-prior:** each county's 2020 actual Dem share (most recent available before 2024)  
**Type-mean prior:** score-weighted average of population-weighted type means (computed from 2024 actuals)  
**Ground truth:** 2024 presidential Dem share

## Overall RMSE

| Approach | RMSE (pp) |
|----------|-----------|
| County-prior (2020 actual) | 2.67 |
| Type-mean prior | 9.00 |
| Improvement (type_mean - county_prior) | 6.33 |

County-prior better in **220/293 counties (75%)**.  
Type-mean better in 73/293 counties.

### Error Distribution (absolute error, pp)

| Approach | Median | p75 | p90 | Max |
|----------|--------|-----|-----|-----|
| County-prior | 2.08 | 3.16 | 3.78 | 9.51 |
| Type-mean | 4.58 | 8.88 | 15.44 | 35.11 |

## Top 10 Worst Errors — County-Prior Approach

| County | State | FIPS | Dom.Type | Actual 2024 | Prior 2020 | Pred | Error |
|--------|-------|------|----------|-------------|------------|------|-------|
| Miami-Dade County, FL | FL | 12086 | 40 | 43.9% | 53.4% | 53.4% | +9.5pp |
| Osceola County, FL | FL | 12097 | 33 | 48.7% | 56.4% | 56.4% | +7.7pp |
| Hendry County, FL | FL | 12051 | 1 | 30.4% | 38.1% | 38.1% | +7.7pp |
| Broward County, FL | FL | 12011 | 4 | 58.0% | 64.6% | 64.6% | +6.6pp |
| Palm Beach County, FL | FL | 12099 | 24 | 50.0% | 56.1% | 56.1% | +6.1pp |
| Hale County, AL | AL | 01065 | 36 | 53.0% | 59.0% | 59.0% | +6.1pp |
| Jefferson County, FL | FL | 12065 | 13 | 40.3% | 46.1% | 46.1% | +5.8pp |
| Hardee County, FL | FL | 12049 | 1 | 21.5% | 27.1% | 27.1% | +5.6pp |
| DeSoto County, FL | FL | 12027 | 1 | 28.2% | 33.6% | 33.6% | +5.4pp |
| Webster County, GA | GA | 13307 | 11 | 40.7% | 46.0% | 46.0% | +5.3pp |

## Top 10 Worst Errors — Type-Mean Approach

| County | State | FIPS | Dom.Type | Actual 2024 | Pred | Error |
|--------|-------|------|----------|-------------|------|-------|
| Hancock County, GA | GA | 13141 | 23 | 67.5% | 32.4% | -35.1pp |
| Wilcox County, AL | AL | 01131 | 5 | 65.5% | 30.6% | -34.9pp |
| Lowndes County, AL | AL | 01085 | 36 | 68.4% | 35.6% | -32.8pp |
| Dallas County, AL | AL | 01047 | 36 | 65.8% | 37.4% | -28.5pp |
| Gadsden County, FL | FL | 12039 | 13 | 64.9% | 36.6% | -28.4pp |
| Liberty County, GA | GA | 13179 | 10 | 58.4% | 31.5% | -26.9pp |
| Walton County, GA | GA | 13297 | 12 | 26.7% | 51.4% | +24.6pp |
| Clayton County, GA | GA | 13063 | 6 | 84.3% | 61.4% | -22.9pp |
| Chattahoochee County, GA | GA | 13053 | 7 | 41.5% | 19.0% | -22.5pp |
| Lee County, GA | GA | 13177 | 12 | 28.0% | 48.6% | +20.6pp |

## Black Belt County Errors

Black Belt RMSE — county_prior: **3.13 pp**, type_mean: **12.73 pp**

| County | State | FIPS | Dom.Type | Actual | Prior 2020 | County-Prior Pred | Type-Mean Pred | Err(County) | Err(Type) |
|--------|-------|------|----------|--------|------------|-------------------|----------------|-------------|-----------|
| Calhoun County, GA | GA | 13037 | 22 | 56.1% | 57.4% | 57.4% | 55.4% | +1.3pp | -0.7pp |
| Clay County, GA | GA | 13061 | 0 | 53.5% | 55.1% | 55.1% | 49.6% | +1.6pp | -3.9pp |
| Houston County, AL | AL | 01069 | 18 | 25.6% | 28.0% | 28.0% | 23.5% | +2.4pp | -2.1pp |
| Marengo County, AL | AL | 01091 | 36 | 47.8% | 50.3% | 50.3% | 35.1% | +2.5pp | -12.7pp |
| Appling County, GA | GA | 13001 | 38 | 18.7% | 21.3% | 21.3% | 27.7% | +2.6pp | +9.0pp |
| Barbour County, AL | AL | 01005 | 5 | 42.2% | 45.8% | 45.8% | 30.3% | +3.6pp | -11.9pp |
| Henry County, AL | AL | 01067 | 34 | 24.3% | 28.0% | 28.0% | 15.4% | +3.7pp | -8.9pp |
| Greene County, AL | AL | 01063 | 17 | 77.6% | 81.3% | 81.3% | 72.5% | +3.8pp | -5.1pp |
| Macon County, AL | AL | 01087 | 17 | 77.7% | 81.5% | 81.5% | 71.8% | +3.8pp | -5.8pp |
| Lowndes County, AL | AL | 01085 | 36 | 68.4% | 72.7% | 72.7% | 35.6% | +4.4pp | -32.8pp |

## Spotlight Counties: DeKalb, Cobb, Miami-Dade, Pinellas

| County | State | FIPS | Dom.Type | Actual | Prior 2020 | County-Prior Pred | Type-Mean Pred | Err(County) | Err(Type) |
|--------|-------|------|----------|--------|------------|-------------------|----------------|-------------|-----------|
| Miami-Dade County, FL | FL | 12086 | 40 | 43.9% | 53.4% | 53.4% | 44.1% | +9.5pp | +0.2pp |
| Pinellas County, FL | FL | 12103 | 24 | 46.9% | 49.6% | 49.6% | 43.5% | +2.7pp | -3.3pp |
| Cobb County, GA | GA | 13067 | 6 | 56.9% | 56.3% | 56.3% | 66.5% | -0.6pp | +9.6pp |
| DeKalb County, GA | GA | 13089 | 6 | 81.9% | 83.1% | 83.1% | 66.4% | +1.3pp | -15.5pp |

## Key Findings

_(Auto-generated section — see printed output for the full interpretation)_

1. **County-prior reduces RMSE by 6.33pp** (9.00pp → 2.67pp).
2. **Black Belt improvement: 9.59pp** (12.73pp → 3.13pp). County priors anchor Black Belt counties at their actual baseline, not the type mean.
3. **220/293 counties (75%) improved** with county-prior approach.

---
_Generated by scripts/spot_check_county_priors.py_