# Governor Forecast Audit — S491 (2026-04-07)

## Summary
- **9 REASONABLE**, **17 SUSPECT**, **10 WRONG** out of 36 predictions
- Model correlation vs 2022 actuals: r=0.714
- Raw 2024 presidential baseline is a better predictor (r=0.733) than the model
- Systematic D-incumbent underperformance: -2.9pp avg error
- Systematic R-incumbent D-favorable error: +2.5pp avg
- Range compression: model spread 40.6pp vs 49.4pp actual (governor) 

## Root Cause
"No cycle-type awareness. Ridge priors trained on 2024 presidential outcomes."
1. 2024 presidential was unusually R-favorable → R-tilted priors
2. Governor races systematically diverge from presidential (incumbency, split-ticket)
3. Open-seat vs incumbent-running dynamics absent
4. Predictions compressed toward 50%

## WRONG Predictions (10)
| State | Model | Should Be | Error |
|-------|-------|-----------|-------|
| CT | tossup -2.3% | lean_d/likely_d | D incumbent, D+7.4% pres |
| FL | tossup +0.9% | lean_r | R+6.6% pres, DeSantis won by 19.5pp |
| IA | tossup +0.7% | lean_r | R+6.7% pres, Reynolds won by 18.9pp |
| IL | lean_r -7.0% | likely_d/safe_d | Pritzker (D) won by 12.9pp, D+5.6% pres |
| MD | lean_d +3.5% | safe_d | Moore (D) won by 33.4pp, D+14.8% pres |
| OH | lean_d +4.7% | lean_r | DeWine won by 25pp, R+5.7% pres |
| OR | tossup +0.5% | lean_d | Kotek (D), D+7.4% pres |
| RI | tossup +0.2% | lean_d/likely_d | McKee won by 19.7pp, D+7.1% pres |
| TX | tossup +2.5% | lean_r | Abbott won by 10.6pp, R+6.9% pres |
| VT | lean_r -7.8% | lean_d | Scott (R) leaving, D+16.4% pres |

## _helpers.py Corrections Needed
1. MN (line 37): Walz did NOT vacate for VP. He lost the VP race, remains governor.
2. SD (line 49): Acting governor is Lt. Gov. Brock McEachin, NOT Dennis Daugaard (left office 2019).
3. NV (line 41+58): Clean up confusing D-then-override-to-R pattern.

## Recommendations
1. **Critical**: Implement voter behavior layer (τ/δ per cycle type)
2. **High**: Add incumbency prior (3-5pp advantage)
3. **High**: Separate open-seat vs incumbent-running signal
4. **Medium**: VT special case — Scott departure changes race fundamentally
5. **Medium**: Fix AK two-party calculation for RCV context
6. **Low**: Correct magnitude compression for safe states
