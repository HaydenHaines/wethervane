# Independent actual M02 result review

The completed output is internally coherent and source-bound, but the challenger does not deliver an A/B improvement. Retain FS5 as this experiment's comparator; no public/default or full-count promotion follows. This is a conditional historical aggregate D/R-share test, not a ranking of all stack layers.

## Exact output and admission verification

Root's original score action completed. I waited for complete_scores, verified terminal/receipt equality and all five advertised artifact hashes before decoding numeric results. Terminal SHA: `72f52aebb6806d61648c73d13dcc2a3212c2baf2eba31c36766e5e56fc2fe86d`; receipt SHA: `7e0b4ceae734aad265736bd2c7f13ffb0f83e52c565c9138567505c0cd89e076`; scores SHA: `fe52e19377e4ad1ce4086c420d8fcfa859e5071af8003505f55b050343c33238`. All source/runtime/admission/mask bindings match accepted scorer v2 and its review. Runtime hashes agree on 01ff4314…; authentic launch PID 1603537 and launch ≤ label-access ≤ terminal chronology match the supplied original action. I never controlled its handle or reran forecasts/scoring.

The separate review.py performs 10,879 bounded arithmetic checks with maximum absolute discrepancy 2.87e-15. It verifies exact rows/masks, per-race WIS/Brier/NLL points, per-race-to-cycle and equal-cycle tier means, saved-influence batch/chain MCSE and variance composition, nonlinear-chain residual/gate arithmetic, probability/PIT identities and descriptive calibration bins/coverage/width. CRPS points are checked through their saved per-race/cycle/tier aggregation; I reused the separately accepted CRPS source/oracles rather than repeating expensive full integrations. Raw forecast tensors and the outcome-source CSV were not opened; admitted truth values persisted in scores were used for arithmetic checks.

Common counts are 25/22/23/27; competitive 9/4/5/6; near-even 1/1/0/3 for 2018/2020/2022/2024. Every arm uses identical frozen masks. Empty 2022 near-even yields unavailable A-tier values and differences, never a zero or a 2018-only average. All original failed v1 scorer and numerical-refinement records remain preserved.

## Primary frozen common-mask comparison

Delta is FS6−FS5; positive means worse proper loss. Bands below are three times conditional Monte Carlo error, not electoral-generalization confidence intervals. All common polling-arm metric precision gates pass. All eight common A/B tier differences are resolved relative to their own 3-MCSE bands.

| Tier | Metric | Delta | 3 MCSE | Relative to FS5 |
|---|---|---:|---:|---:|
| A | brier | +0.000745662 | 0.000170845 | +1.2417% |
| A | crps | +0.000368890 | 0.000044399 | +1.9915% |
| A | negative_log | +0.001736463 | 0.000430860 | +0.9458% |
| A | wis | +0.000265360 | 0.000037349 | +1.8427% |
| B | brier | -0.001307445 | 0.000238015 | -2.9963% |
| B | crps | -0.000235062 | 0.000065281 | -1.2407% |
| B | negative_log | -0.004244263 | 0.000630718 | -2.8718% |
| B | wis | -0.000159597 | 0.000046089 | -1.1135% |

A is the designated 2018/2022 midterm tier and worsens on every metric. B 2020/2024 improves on every metric. A-common CRPS worsens 1.9915%; B improves 1.2407%. This is mixed evidence, not a robust gain across the project A/B comparison. The declared 0.0005/2% materiality numbers were review aids, not permission to hide individual regressions or substitute favorable masks.

| Cycle | CRPS delta | WIS delta | Brier delta | NLL delta |
|---|---:|---:|---:|---:|
| 2018 | +0.001029915 | +0.000866571 | -0.000985462 | -0.000806321 |
| 2020 | -0.000274580 | -0.000200795 | +0.000365129 | -0.000051559 |
| 2022 | -0.000292135 | -0.000335852 | +0.002476786 | +0.004279247 |
| 2024 | -0.000195543 | -0.000118398 | -0.002980018 | -0.008436967 |

2018 common CRPS worsens 0.001029915 (5.415%), with 3-MCSE0.000073733; WIS also worsens. Its binary scores improve. In 2022 CRPS/WIS improve, but Brier and NLL worsen by 0.002476786 and0.004279247, both resolved. In 2020 the common Brier point regression and tiny NLL point improvement are below their3-MCSE differences; the share-score gains are resolved.2024 improves all four common metrics.

All 19 positive per-cycle/mask/metric point regressions are retained in all-primary-cycle-regressions.csv; 18 exceed their conditional3-MCSE difference. This includes all four 2020 competitive/near-even regressions and 2022 competitive Brier/NLL/CRPS/WIS regressions. A-competitive's small CRPS/WIS gains do not erase its probability-score regressions or replace the frozen primary mask. B-competitive CRPS/WIS and B-near-even share-score differences are unresolved relative to numerical bands. The LP5 data-selection anchor is separately retained: full-series FS5 is not uniformly better than latest-pollster LP5; this does not change the predeclared FS6−FS5 target.

## Qualification and descriptives

Across the 240 arm/mask/cycle/metric cells, 132 polling-arm estimates qualify, 44 PO5 cells are analytic, 24 archived-F0 cells qualify, 20 empty-mask cells are unavailable, 11 F0 WIS cells remain dispersion-only, and nine F0 subset Brier/NLL precision cells remain unresolved. The nine failures are all reference-subset precision limits; they are retained, not silently dropped or repaired. F0 WIS has no certified nonlinear MCSE. No post-label sampling, probability floor or threshold relaxation was performed.

The reported bin counts and coverage/width recompute exactly. Illustrative 95% common-mask coverage and mean share-interval widths:

| Cycle | FS5 coverage | FS6 coverage | FS5 width | FS6 width |
|---|---:|---:|---:|---:|
| 2018 | 0.9200 | 0.9200 | 0.114759 | 0.116497 |
| 2020 | 0.8182 | 0.8636 | 0.122257 | 0.123955 |
| 2022 | 0.9130 | 0.9130 | 0.116377 | 0.117949 |
| 2024 | 1.0000 | 1.0000 | 0.115718 | 0.117473 |

FS6 intervals are slightly wider in all four cycles, with mixed coverage behavior at other levels. These are descriptive values across four national cycles; races are not independent elections. Continuous PIT values retain CDF/survival logs; archived empirical PIT uses a discrete bracket without randomization. The separate 2020 LP5 interval-display precision failure was preserved by Root and does not veto precise aggregate scores or trigger a retry.

## Limits and disposition

No computational inconsistency was found by this bounded review. The remaining limitations are substantive: finite inherited historical banks, unintegrated uncertainty in fitted MAP polling-error scales, shared earlier-data empirical-Bayes approximation, retrospective availability proxies, few revisited national cycles, and marginal rather than full-count scoring. Numerical resolution does not establish causal momentum, general population calibration or current 2026 performance. The earlier descriptive rising-candidate audit and this symmetric covariance challenger answer different questions.

The result does not support replacing FS5 with FS6 for this comparison. That is a bounded recommendation, not a public registry change or weakest-layer ranking. Root owns the actual scientific decision and journal. No label-informed model/mask/scale tuning or extra fit was attempted during review.


## Saved-row localization (no exclusion or tuning)

Independent arithmetic confirms 2018 common CRPS net +0.001029915076005821: NM contributes +0.0007571294733076023 and DE +0.00034702293436320334; the other 23 jointly contribute −0.00007423733166498455. Contributions use the original 1/25 race weights. In 2022 NV+PA contribute +0.004420109587548332 to Brier versus net +0.0024767855967449785, and +0.00913474564482121 to NLL versus net +0.004279246856082702; other 21 offset the excess. All 97 common-race deltas and original-weight contributions are retained in saved-row-contributions.json. This localizes the recorded losses without asserting a cause, selecting a different mask or changing the model after labels.
