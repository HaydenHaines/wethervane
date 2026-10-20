# House vote-share testing evidence, 2026-10-20

This frontend-owned adapter exposes selected-model/year summaries from completed, reviewed historical tests. It is separate from the scientific result/catalog schemas and does not modify their frozen parsers, importers, source files or statistical definitions. All sixteen published results remain experimental historical comparisons. No new model is fitted, no posterior array is read, and no current forecast, polling, map, chamber-control or tipping capability is admitted.

The source owner's final `HANDOFF.md` explicitly delivers derived results to the frontend publication workflow while retaining raw archives locally. It links the final scores, interpretation, report and precision evidence. No additional restriction on publishing these reviewed derived scores was found. The frontend has consumed that handoff and published its sixteen reviewed results. The House owner's 22:59:26 status marks its goal complete with no remaining work. This document records frontend publication acknowledgement without modifying the source checkout; a separate frontend outbound-message transport failure does not affect source completion or publication readiness.

## Source and identity admission

Read-only source checkout: `/home/hayden/projects/ballotmancy-house-benchmark`; paths below are relative to `research/house-benchmark-20261020/`.

| Artifact                                  | SHA-256                                                            |
| ----------------------------------------- | ------------------------------------------------------------------ |
| `HANDOFF.md`                              | `910692edc82c183704f37bb9737985ddc9b94cbd44b6f95a52a3754e237f9118` |
| `frontend-results-v2/catalog.json`        | `c189fe1100acec043f6ace325c81c1e2da8765c5f20a454b31e3825821d9cb7f` |
| `frontend-results-v2/review-receipt.json` | `70f371b56d63909c847d0aa539efb28a5c4a538c3ed2e33052ea4f0a28eea9ee` |
| `final-scores-v2/receipt.json`            | `c9e77fbf7d6d2263ecc1479ed324416e4284b995ea5ecca6509ba4271c0b7fbf` |
| `final-scores-v2/per-race-scores.csv`     | `be2fb76fdd18822a0d4dfedefb1e1f18509c5822b575e8e9a0b5f047145cfc81` |
| `final-scores-v2/cycle-scores.csv`        | `86b822702b74b82f9f0c1b04afefb11c5aca426f8969dfe1f9dcda431d94ab6c` |
| `panel-v6/receipt.json`                   | `b07c227432fa4768822e0db083107df6675d7d4168b7fc9a493180c16b1bfe80` |
| `panel-v6/outcomes-SEPARATE.csv`          | `45eb03bc69205763cfbef5014ce9e505dc0aa1cfe8a18ee8d7ca9b7a2d5f9c37` |
| `panel-v6/features.csv`                   | `3bbacb9487d45280fc131fa2f7754767b7939c32f5c282af4fc48af99d81a9c4` |
| `precision-v1/receipt.json`               | `0c9fb0eaf1d96ad9cefe8b3ed896765b63c3197f08f79590e6b283438fb31eca` |
| `report-v1/receipt.json`                  | `09e755eda4140f41d449b61f114f1d24364a3488f67a79de514f89b0dd9d9739` |
| `interpretation-v1/receipt.json`          | `fd73a4a2d36d775d7c3ff0d2e1c512a237b3b4e4ade38179af60405d841a0e00` |

The exporter reuses the reviewed comparison exporter to verify all sixteen source/public result bytes, target, nominal fourteen-day horizon, retrospective information mode, source policy, actual fit plans/receipts, corrected panel, target race identities, configuration hashes and joint-artifact references. It separately pins score/precision/report artifacts and verifies every score-receipt source/output/fit reference. The score receipt must contain each exact reviewed fit receipt. No NPZ/NetCDF contents are loaded by this exporter.

All 6,008 score rows join uniquely by configuration, cycle and race identity. Each model/year must have exactly the public race set: 371/383/373/375 races for 2018/2020/2022/2024. Candidate names and local IDs are independently checked using the reviewed exporter rule: official candidate name from the separate outcomes table when present, otherwise the feature roster name, with its exact party-scoped name hash. This matters in 2024, where official names differ in formatting from final-roster feature strings. These local IDs are not asserted to be FEC person identifiers.

Observed D share must agree with the separately pinned D and R vote totals. The observed winner, Democratic win probability, mean, and saved 5th/95th percentiles must agree with the exact public result. Absolute/squared error, coverage, interval width, interval score and both Brier scales are replayed algebraically. Saved CRPS is admitted from the pinned reviewed score file; its distributional computation is not reconstructed from quantiles. Source cycle-summary means/counts are checked against the replayed rows, and unexpected rows or joins reject the export.

## Units and meaning

All metrics concern conditional Democratic vote share `D/(D+R)`, not full-ballot shares. Third-party votes are outside this estimand. Excluded districts are retained by the existing coverage ledger but do not enter these averages.

| Frontend field      | Source identity       | Meaning                                                                                     |
| ------------------- | --------------------- | ------------------------------------------------------------------------------------------- |
| `meanAbsoluteError` | mean `absolute_error` | Average absolute difference between predicted mean D share and actual D share               |
| `covered` / `total` | `coverage`, `races`   | Count of actual shares inside the inclusive saved central 90% interval                      |
| `meanIntervalWidth` | mean `interval_width` | Average upper-minus-lower share interval width                                              |
| `crps`              | mean `crps`           | Reviewed continuous ranked probability score for the saved share distributions              |
| `intervalScore`     | mean `interval_score` | Proper central-90 interval score: width plus twenty times any distance outside the interval |

Metrics use `share_fraction`, with `intervalLevel: 0.9`. Multiply error, width, CRPS and interval score by 100 for share percentage-point display. For example, a predicted 55% share and actual 52% share give a three-share-point miss. Margin error is twice share error and must not replace it silently. Source binary win Brier is 0–1; the existing frontend categorical score is 0–2.

Coverage is `lower <= actual <= upper`. It must be displayed with width; high coverage alone does not demonstrate useful uncertainty. Source findings show wide intervals can achieve coverage while losing proper scores. CRPS and interval score are lower-is-better metrics, not overall grades or forecast certification.

Fixed-2024 source-replay checks: 375 races; 367 covered; mean absolute share error `0.021172829884947055`; mean 90% interval width `0.19147057264919518`; CRPS `0.01911193185910748`; interval score `0.19485721784279228`.

The read-only audit replayed all 32 source all/competitive summaries to a maximum difference of `2.8e-17`. All published share means/endpoints matched saved scores within approximately `1.3e-15`, and win probabilities matched exactly. The first export intentionally exposes only each selected year's all-included-races summary; it supplies no pooled cross-year ranking or competitive-subset UI.

## Runtime and qualifications

`data/house-score-evidence.json` contains summaries, source digests, model/result bindings, race IDs and plain qualifications. It includes no per-race observed vote records, raw source archives, local paths or predictive arrays. `lib/house-score-evidence.ts` checks its schema/source pins and an immutable canonical entries digest, then asynchronously checks the selected publication's exact source-file hash, configuration/year/map/scope/race support and canonical result contents. A changed score, prediction, observed result or stale publication makes the evidence unavailable while leaving ordinary result exploration usable. No fallback reconstructs a score from an unverified result.

The source precision receipt covers CRPS and binary Brier only. Its errors condition on finite fitted banks and preserve within-model dependent races; they do not measure source error, fitted-bank uncertainty, future accuracy or election-to-election uncertainty. This adapter exposes no error bars for MAE, coverage, width or any inferred model difference, and makes no significance/calibration claim.

Only four evaluation elections are available. Source editions, reconstructed presidential features and final candidate rosters are retrospective proxies rather than strict as-of replay. The 2018 Context model extrapolates beyond the presidential-party context in training. Numerical race gates pass, but eligible-seat subtotal precision fails in all sixteen cells. These qualifications remain in the manifest; no chamber qualification or current-model promotion follows.

Focused tests cover actual sixteen-result verification, exact source replay, immutable metrics, stale/tampered predictions and bindings, missing/duplicate scores, candidate/observed joins, units, Brier-scale distinction, inclusive interval boundaries, malformed CSV and changed source pins. Frozen House result, catalog, importer and scientific files remain unchanged.

## One-page presentation and validation

Following Luna's review, the existing three-card score summary stays in place. A collapsed “More about vote-share tests” disclosure adds selected-model/year results and the compared model's values when available. Mean error is explicitly Democratic vote-share error, not election-margin error. Coverage always appears beside mean range width. CRPS is expanded and explained in terms of the possible shares and probabilities assigned to them. Research stage remains separate from successful individual-race calculation checks.

The browser checks both the immutable summaries and the complete selected result before displaying evidence. Changed content cannot inherit evidence by claiming an old fingerprint. Primary and comparison evidence load independently; an unavailable reference retains the selected model's tests without stale reference values.

Local validation: all114 House assertions pass, including the nine new evidence checks and exact source replay. All40 affected desktop/mobile cases pass (ten new evidence cases and thirty existing comparison/release/layout cases). Production build and typecheck pass. Desktop and Pixel7 screenshots were inspected without JavaScript errors, page overflow or clipped metrics. The original source/public result bytes and five frozen parser/importer hashes remain unchanged.
