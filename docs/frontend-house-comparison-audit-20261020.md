# House same-year comparison audit — October 20, 2026

## Handoff acknowledgement

The frontend has consumed the House owner's final `research/house-benchmark-20261020/HANDOFF.md`. All sixteen reviewed historical results and the same-year comparison feature are **live at frontend commit `de6e80a`**; [CI run 39511922287 passed](https://github.com/HaydenHaines/wethervane/actions/runs/39511922287). Fixed 2024 remains the agreed reference, not a winner selected by historical scores. The source status updated `2026-10-20T22:59:26Z` reports the goal complete, sixteen qualified cells and no running tool session or remaining required work. The final handoff is not awaiting frontend consumption.

An inbound House READY notification was delivered. Frontend outbound native acknowledgements still fail with a transport error in this cycle; this document records acknowledgement without claiming outbound delivery or changing the scientific worktree.

The initial scientific review was read-only. The additive frontend implementation described below follows that review; it creates no new scientific results or qualification.

## Implemented comparison

The collapsed House builder now offers another reviewed model from the same election year. The selected model continues to drive the charts. A separate comparison note names both models; scorecards show both models' favorites-correct and probability-error scores on the identical tested districts. Candidate details show both winning chances and the selected-minus-compared change in percentage points. Very small changes are described without a misleading signed zero. Changing years clears the comparison rather than substituting a model from another election.

`scripts/build_house_comparisons.mjs` produces `data/house-comparisons.json` from the pinned source packet, checking its fit plans, review cells, policy and delivered files. `lib/house-comparison.ts` admits only the exact reviewed same-year groups. Browser verification checks canonical result contents as well as declared file hashes, candidate identities, observations, coverage and source bindings. All 48 ordered model pairs are supported. Missing, modified, cross-year or unsupported references retain the selected forecast and offer recovery without partial scores.

Optional comparison bindings extend the existing version1 local bookmark format without invalidating older views. Restoring a comparison verifies both exact results before changing the displayed model or URL. Stale and withdrawn entries remain visible and disabled. Shared links, district/view selections, and browser Back/Forward preserve the comparison. No new endpoint or scientific parser change is required.

This is a descriptive comparison of separately simulated models. It does not pair draws, quantify uncertainty in a difference, or activate chamber totals, control, tipping, polling, current2026 or district geometry.

Local validation: production build and TypeScript pass; all105 House assertions pass, including all48 ordered same-year pairs and regeneration from the pinned source packet. All16 new comparison browser cases and56 affected existing House/chamber/layout cases pass across desktop and Pixel7. Tests cover independently calculated scores, history/share links, same-primary saved restoration, atomic failed restores, mismatched/tampered results and stale/withdrawn secondary bindings. Final screenshots show readable comparison text, explicit model names, no signed-zero deltas, no page overflow and no JavaScript errors. All five frozen parser/importer hashes and all published scientific result bytes remain unchanged.

Public verification after deployment passed on desktop and Pixel7 at `2026-10-20T23:00:44–49Z`: the builder starts collapsed; Fixed 2024 shows 365/375 favorites correct and 0.045 categorical Brier, compared with Context's 366/375 and 0.035. Both exact result/configuration bindings survive local save, model switch, restore, share and reload. Comparison text is readable at 12px, tiny candidate deltas avoid signed zero, and no JavaScript errors, horizontal overflow or unsupported geometry/control appeared. Richer source testing evidence is being audited separately; this release does not claim that adapter or feature is complete.

## What is supported

Same-year descriptive comparisons among the four reviewed House laws are scientifically justified. The source handoff and `interpretation-v1/findings.md` explicitly compare identical corrected eligible races within each cycle. The four models differ in their intended equations, so their configuration/law identities should remain distinct rather than being required to match.

Independently inspected the raw published results and frozen `parseHouseResult` output. For each year, all four results have exactly identical coverage ledgers, race identities, election/map metadata, candidate/member names/IDs/parties/incumbency/source links, share denominator, observed winners, and race notes. Raw results have no extra top-level fields discarded by the parser. All sixteen still pass the frozen parser.

| Year | Common eligible races | Complete coverage ledger | Distinct joint IDs / joint source hashes |
| --- | ---: | ---: | ---: |
| 2018 | 371 | 435 | 4 / 4 |
| 2020 | 383 | 435 | 4 / 4 |
| 2022 | 373 | 435 | 4 / 4 |
| 2024 | 375 | 435 | 4 / 4 |

Useful quantities computable from the existing public fields, without reconstructing joint draws:

- For the exact same candidate and district: winning-chance change in percentage points; conditional D/R mean-share change; the displayed median and 90% endpoints/widths for each model. Expected D-minus-R margin is `100 × (2 × mean_D_share − 1)`; a change in mean margin is computable, but an interval for the difference between models is not.
- For the common observed races: categorical win Brier on the frontend's 0–2 scale, its signed change, and favorites correct/missed/tied using one shared tie policy. This Brier is twice the source binary Brier. It is not a vote-share score or percentage accuracy.
- Districts where the favored candidate changes, joined by identity rather than array position. Keep exact ties as ties, not first-listed winners.
- Mean displayed 90% interval width over the common races, explicitly a sharpness description, not evidence of better calibrated predictions. The source findings warn that narrower or wider intervals can have different proper-score costs.

Example checks from the actual 2024 public results (all 375 races):

| Model | Favorites correct | Categorical Brier | Mean 90% D/R share-range width, pp |
| --- | ---: | ---: | ---: |
| Fixed | 365 | 0.045203875 | 19.1471 |
| Slope | 365 | 0.041749296 | 18.6843 |
| Context | 366 | 0.035402029 | 15.3675 |
| Relative | 365 | 0.040303647 | 17.5430 |

Compared with Fixed, 2024 favored-candidate identities change in eight Slope races, five Context races, and eight Relative races. These descriptive counts are not independent hypothesis tests or proof of future superiority.

## Required admission and binding

Do not authorize a general comparison merely because two results parse, share a year, or contain D/R labels. The public frozen schema does not expose the scientific target, nominal horizon/cutoff, source panel identity, or complete source qualification as structured comparison keys. Its catalog is a selection catalog, not a general scientific compatibility certificate.

A minimal additive frontend adapter should use an explicit reviewed comparison-group receipt keyed by exact result bytes. It can leave the frozen result/parser/catalog unchanged. The receipt should bind:

1. This published catalog, the source catalog/review, and the source capability policy below; common target `house_aggregate_fundamentals_retrospective_v1`, nominal horizon 14 days, retrospective information mode, no polling, and conditional D/R estimand.
2. For each allowed run: exact `run.id`, `configurationId`, `configurationSha256`, `resultSha256`, year, and `jointDraws` identity/source hash. Different model configurations and joint identities are expected, not interchangeable.
3. Per comparison year: exact map edition, common eligible race-ID set, full 435-row coverage ledger including exclusions, candidate/member identity and party mapping, incumbency, denominator, election identity, and observed truth. Canonical fingerprints can sort these structures by identity; do not pair rows by display order.
4. Source numerical/review status and qualifications, including the failed eligible-seat-subtotal precision gate. The admitted operation is historical race comparison only.

At runtime, load and verify both immutable publications separately using the existing loader/catalog checks. Admit comparison only when both are in this group, have matching year/support/observed truth, and retain the recorded identities. A missing, changed, unavailable, incompatible, or partially loaded comparison must not relabel the displayed model or yield partial/intersection-only scores silently. If a supported subset is ever introduced, declare its identical membership and exclusion reasons explicitly.

Suggested small pure helper result: either `{status:'comparable', primaryIdentity, referenceIdentity, raceIds, candidateDeltas, winScoreSummary}` or `{status:'unavailable', reason}`. Include units, common race counts, tie exclusions, and signed difference direction. Keep fetching, UI state, and source admission outside the metric arithmetic. This is a recommendation, not an API declaration.

## What is not supported by these public summaries

- Pairing independently seeded model draws, claiming a same-draw counterfactual, or calculating a distribution/credible interval/probability for their difference. Draw identities are local to each model/year; matching row numbers or simulation counts do not establish coupling.
- Share CRPS, share absolute error, interval coverage or proper interval scores. The public observed object provides winner identity, not actual vote shares; three share quantiles do not determine a full predictive CDF or CRPS. The source's richer scores can be displayed only through a separate reviewed, hash-bound score adapter.
- The source's conditional score-difference MCSEs, calibration bins, competitive-subset findings, or near-even margin-event probabilities. These require the corresponding source definitions/receipts. Do not independently select different “competitive” races for each model and call the averages matched.
- Turning marginal race probabilities into joint seat/control simulations, multiplying race win probabilities, comparing chamber-control odds, reporting a majority tipping point, or treating the eligible subtotal as the full House. All sixteen seat-subtotal precision gates failed; full-chamber coverage and controls remain unavailable.
- Cross-year district/candidate deltas, carrying district numbers through map changes, pooling unequal year cohorts into an implied same-race result, or choosing a new default by target outcomes. The source's A/B equal-cycle historical summaries are separately defined evidence, not permission for arbitrary pooling or a current-2026 forecast claim.
- Giving numerical score differences statistical-significance or causal interpretations. Shared national shocks, finite fitted-bank uncertainty, retrospective sources and only four evaluated elections remain material limitations.

No verified district geometry is admitted. Same-year comparison can work in race lists/details without a map; a common map label alone does not authorize drawing district boundaries.

## Pinned evidence

Scientific root: `/home/hayden/projects/ballotmancy-house-benchmark/research/house-benchmark-20261020`.

| Artifact | SHA256 |
| --- | --- |
| Final `HANDOFF.md` inspected in this audit | `910692edc82c183704f37bb9737985ddc9b94cbd44b6f95a52a3754e237f9118` |
| `interpretation-v1/findings.md` | `f2883fe9688d5120a6d3e37c7c6ae380fa8e241c95738caf61e287654b355eba` |
| `interpretation-v1/receipt.json` | `fd73a4a2d36d775d7c3ff0d2e1c512a237b3b4e4ade38179af60405d841a0e00` |
| Source `frontend-results-v2/catalog.json` | `c189fe1100acec043f6ace325c81c1e2da8765c5f20a454b31e3825821d9cb7f` |
| Source `frontend-results-v2/review-receipt.json` | `70f371b56d63909c847d0aa539efb28a5c4a538c3ed2e33052ea4f0a28eea9ee` |
| Published frontend catalog | `82c63f9ba70bcd8a9fbbb2037d5bb48d3e9ee8190462f1583caa9a3c8ddb092b` |
| `source-capability-policy-v1.json` | `618e5870cc224d06d949e93bef4ba80e1d044445ff6a91877657837eb81f2897` |
| `panel-v6/receipt.json` | `b07c227432fa4768822e0db083107df6675d7d4168b7fc9a493180c16b1bfe80` |
| `final-scores-v2/receipt.json` | `c9e77fbf7d6d2263ecc1479ed324416e4284b995ea5ecca6509ba4271c0b7fbf` |

The earlier release audit records full-artifact/source score reconciliation. This comparison audit adds direct within-year identity/support equality and documents the limits of the currently published fields. No scientific worktree or frozen frontend files were changed.
