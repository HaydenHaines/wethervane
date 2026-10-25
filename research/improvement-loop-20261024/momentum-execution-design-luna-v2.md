# Matched M02 forecast comparison specification — v2 proposal

**Scope:** no-fit, no-score design for independent review. This corrects v1 while preserving that file and its review history. It does not authorize optimization, forecast generation, source admission, or promotion. Root selects calibration optimization and the target data operator separately. No target-score results were inspected in preparing this revision.

## Question

Test whether adding a persistent linear time covariance component changes retrospective D/R share and winner forecasts when training rows, target poll rows, F0 bank, target features, and outcomes are otherwise matched. For `x = days_before_election / 30`, the M02 covariance adds `tau_slope² x xᵀ`, anchored at Election Day. It is a covariance contribution, not a deterministic adjustment for a candidate who is rising.

The central matched contrast is F0 plus polls under the existing five-component Brownian error model (M01) versus the same F0 bank plus the same poll vector under the six-component slope model (M02). Each covariance fit re-estimates all its scales jointly using the same training rows and root-selected log-SD penalty. The contrast is conditional MAP calibration followed by the existing **marginal empirical-prior polling update**. It is not a calibration cut and does not integrate covariance-scale uncertainty.

## Historical folds and exact calibration

The sealed preparation plan binds three unique calibration datasets for four target folds. In particular, 2022 and 2024 share the same exact calibration input identity (`f5201d12ec69f378e03cf14af6aa0835ef84cfbbe213e6101dedc575297d16c9`); 2024 has no 2022 calibration source cycle.

| Target forecast | Strictly earlier source cycles | Selected rows | Target F0 races | Calibration fit identity |
|---|---|---:|---:|---|
| 2018 | 2012, 2016 | 930 | 30 | 2018 fold |
| 2020 | 2012, 2016, 2018 | 1,218 | 28 | 2020 fold |
| 2022 | 2012, 2016, 2018, 2020 | 1,514 | 30 | shared 2022/2024 fit |
| 2024 | 2012, 2016, 2018, 2020 | 1,514 | 30 | shared 2022/2024 fit |

Use the frozen calibration selector: all strictly earlier cycles, lead 0–100 days inclusive, latest observation by cycle/race/normalized pollster/fixed 14-day bin. Use exactly the sealed row IDs and D/R residuals in each M01/M02 pair. The six frozen scale medians (independent .10, cycle .06, race .06, house .03, Brownian movement .05, M02 slope .05) use the root-selected Normal penalty on log-SD with scale .7, same constant convention and no natural-scale Jacobian. This is the selected conditional-MAP objective, not marginal evidence. Root still owns the choice/timing of optimization.

Use the reviewed H14-bound F0 relative-context bank for the target fold, with all original 16,000 raw joint posterior rows, chain/draw identities and coefficient/scale dependence. Initial row weights are equal; polling changes their posterior weights. Do not factorize, shuffle, or independently mix columns; do not substitute pre-simulated F-only future replicates for joint parameter rows. The 2020 target keeps its 31 Texas calibration rows; the separate 1,483-row disjoint bank is not used. The 2022/2024 shared calibration identity allows one covariance calibration per arm to be reused for both target forecasts, subject to exact plan and source-hash validation.

## Correct target update law

For every F0-plus-poll arm, use the existing, qualified `polling_basis_v1.sample_poll_posterior` empirical-row Gibbs operator with `EmpiricalBasisFundamentals` and the declared Gaussian poll likelihood. The likelihood is not a stand-alone Gaussian of beta: it conditions on latent observed race states, the shared target-cycle effect, and Student-t race precision-mixture states. The Gibbs transition updates those states and the empirical bank-row index jointly. A transition may revisit/resample a row; each draw must retain the original source chain/draw identity, and all selected columns remain the same intact joint row. Posterior row weights are represented by sampled row-index frequencies, not required to remain equal.

Generate F0-only predictions under the same bank and output law with no target polls. For M01/M02 paired runs, use reproducible fold/operator/cutoff/arm-specific seeds and preserve bank-row provenance. Where common random numbers can be used validly (e.g., aligned posterior predictive draws conditional on paired states), use them and report how; do not claim Gibbs samples are aligned or independent just because seed integers match. Report chain diagnostics for row indices, latent states, shared cycle, precision-mixture states, coefficients and scales. If a numerical acceleration or alternative integration replaces the existing sampler, it is a new operator requiring independent equivalence validation before use.

## Comparators

Use these arms on identical race truth masks wherever each is defined:

| Arm | Target prior | Target polls | Error/update law | Role |
|---|---|---|---|---|
| F0-only | Reviewed fold-specific F0 joint empirical bank | None | Existing F0 predictive law | Fundamentals reference |
| Poll-only M01 | Fixed broad independent Gaussian prior on race outcome logits | Same exact poll vector | M01 covariance likelihood | Standalone poll reference |
| Poll-only M02 | Same fixed broad independent Gaussian prior | Same exact poll vector | M02 covariance likelihood | Standalone slope sensitivity |
| F0 + poll M01 | F0 joint bank | Same exact poll vector | Existing `polling_basis_v1` M01 Gibbs update | Combined incumbent |
| F0 + poll M02 | Same F0 joint bank | Same exact poll vector | Existing `polling_basis_v1` M02 Gibbs update | Combined challenger |

The **poll-only reference** deliberately does not learn a prior from historical election outcomes or call such a prior “no fundamentals.” Let each target race’s latent final D/R-share logit `z_r` have a fixed independent `Normal(0, 2.5²)` prior, then update using the same target poll vector and fold-specific five- or six-component covariance likelihood. The fixed symmetric prior uses no current covariates or training outcomes; `2.5` logit SD yields a central 95% prior share range of approximately 0.7%–99.3%, so it is a broad, proper reference rather than a learned forecast prior. Freeze it before scoring; do not tune it on these four cycles. Compute the Gaussian posterior for latent logits under the declared covariance and use common deterministic integration points for logistic-share scores if numerical integration is needed. This is a **reference sensitivity**, with a different prior/update operator from F0’s Student-t empirical-bank Gibbs law; do not treat poll-only versus combined score differences as a clean causal decomposition. The clean slope contrast remains F0+poll M01 versus F0+poll M02.

## Target operator and horizon choices remain open

The corrected coverage profile supports two distinct target poll operators; Root has not selected which is the forecast operator. Do not call one the primary execution condition until Root decides. Preserve them as separate, matched conditions:

- **Latest per race/pollster:** the source adapter’s current latest-observation convention, one exact event/pollster sample and one eligible question per sample. H14/native proxy profile: 170/134/171/226 rows over 25/22/23/27 covered races for 2018/2020/2022/2024.
- **Full eligible series:** every row admitted under the same cutoff, identity, population, question and deduplication rules. H14/native proxy profile: 202/190/220/351 rows over those same cycle-specific covered-race counts. It retains 25/32/38/74 repeated race-pollster groups. These rows can contribute time contrasts, but distinct poll/sample IDs do not establish respondent independence and the house covariance is not a proven overlap correction.

Within either target operator, M01 and M02 receive exactly the same target poll IDs. Do not pool data operators or call a full-series versus latest difference a kernel effect. The proposal evaluates H14 first and H7 as a separate poll-data cutoff condition. H7 reuses the same H14-bound F0 feature rows and bank; label it as a stale-feature sensitivity, not a complete H7 information set. Native availability is an archive field-end/creation-date proxy, not strict proof of publication before cutoff (`strict_asof_qualified:false`). Original publication/revision times and real respondent interview-date distributions are unavailable. Do not average over advertised intervals or expand rolling windows into daily observations. H30 is out until target features have dated H30 admission.

For every operator/horizon/target cycle, bind the exact cutoff timestamp, eligible-ID ledger, question rule, and extra-lag choice before forecast generation. Native lag 0 is the proposed profile only; any +1/+3/+7-day delay is a separately named sensitivity with exact date arithmetic. Do not assume numeric lags carry the same publication meaning across sources.

## Exact identity, target mask and outcomes

Before prediction, derive a complete mask by intersecting: (1) exact F0 target race ID, (2) source-proven regular-event poll identity, (3) target operator/cutoff eligibility, and (4) exact scored D/R outcome identity. Use the same race IDs in F0-only, poll-only, M01 and M02 for each paired comparison. Also report full-bank F0-only coverage separately; do not compare that score against a smaller poll-supported mask as if denominators matched.

The five roster gaps are event-identity gaps, not evidence of no polls. A separate intake proposes regular event matches for 2018 MN, 2018 MS, 2022 OK and 2024 CA, but proposals are not data admission. Require Root’s explicit admission and source receipt before using them. 2022 California remains unresolved between 538 races 8921/9480; exclude both from regular-event poll masks unless an authoritative question-ID/event link is added. Do not infer by row order or same-candidate name. Keep unresolved/unsupported races visible in the mask ledger with reasons and no zero-valued poll.

Keep the identity/mask decision independent of forecast outcomes and scores. Publish the masks, exclusions, race counts and source hashes before emitting score summaries. A failed identity gate invalidates that poll-updated comparison; no score file should conceal the missing or ambiguous event.

## Scoring

Every arm produces D/(D+R) share distributions and D/R winner probabilities on the same per-comparison truth mask. Report by race: predictive share mean/median, central 50/80/95% share intervals, D/R win probability, actual D/(D+R) share and official D/R winner. Keep D/R share separate from third-party share and margin; define tie handling once before scoring.

Predeclare these metrics: share CRPS; winner Brier and binary log score; share WIS from the 50/80/95% intervals; interval coverage and width. Winner accuracy may be shown as secondary descriptive information only. Declare the probability floor (if any) before scoring and also show raw prediction support. Average races equally within cycle; show all four cycle scores and paired per-race M02−M01 deltas, then equal-weight cycle macro-averages and A/B cycle summaries. Show each mask and denominator beside scores. Do not treat races within a cycle as independent election replications; four-cycle variation is descriptive, not a significance test.

## Diagnostic contract; numerical thresholds remain unselected

Before any new score is viewed, the execution plan must state numeric gates for all of the following. This proposal intentionally does not invent thresholds or select them from score outcomes:

- Covariance fit: exact training row IDs/residual hash; optimizer convergence; multiple-start spread; gradient norm; boundary hits; Hessian/profile conditioning and parameter uncertainty; prior-dominance/identifiability of Brownian versus slope scales.
- F0 Gibbs update: chain count/length, warmup, seed rules; R-hat, ESS, BFMI and convergence checks for all bank-row, state, cycle, scale and precision-mixture quantities; posterior weight concentration and per-chain row contribution; treatment on gate failure.
- Poll-only Gaussian integration: deterministic integration rule, convergence tolerance and error checks for share scores/interval endpoints.
- Predictive Monte Carlo: paired-run scheme, batch or chain MCSE for each score and interval endpoint, allowed MCSE relative to predeclared reporting precision, and reproducibility receipt.
- Invariants: covariance symmetry/positive definiteness; five-kernel equivalence at zero slope variance; independent objective/gradient oracle; exact mask consistency; normalized finite weights; all shares/probabilities in [0,1]; same D/R truth definitions and tie rule.

Do not substitute vague “small enough” gates during execution. Do not set them after seeing score signs. A numerical failure, prior-dominated fit, or poor effective row support is computational/inferential uncertainty, not evidence of a predictive regression or a win. Successful diagnostics do not establish improvement.

## Outcome interpretation and scope limits

This is retrospective development with approximate availability proxies, two A cycles and two B cycles. Report material gains and regressions by cycle, data operator, cutoff and mask; do not require every cycle to improve and do not promote on one noisy aggregate. Any change after observing outcomes or scores is exposed development and must preserve previous plans and results.

Excluded: H30; interval averaging; new poll acquisition; unresolved 2022 CA event mapping; unreviewed roster changes; reuse of the separate TX-disjoint training panel; a historical-outcome-learned “poll-only” prior; selected-vintage claims; calibration cut; integrated scale uncertainty; 2026 extrapolation or API activation.

**Open root selections before any run:** (1) latest-per-pollster versus full eligible series as the forecast operator (or whether to keep both as named conditions); (2) exact cutoff timestamps and native/extra lag; (3) source-proven event admissions and resulting score masks; (4) numeric optimizer, sampler, ESS and MCSE gates; (5) poll-only comparator inclusion versus treating it as separately reviewed sensitivity. This v2 is a proposal, not an execution authorization.

## Frozen references

- Root preflight review: `momentum-preflight-root-review-v1.md`, SHA-256 `22f25174f9412c3b876784753023bd7c229a91642b4a82b4aeaab41319d6e669`.
- Independent M02 design review: `momentum-challenger-review-sol-v1.md`, SHA-256 `0b6e855ce79551907938cbe2bb55009449dc5d7a5912bcc23f8497655d9be8a6`.
- Corrected coverage authority: `momentum-preflight-coverage-luna-v1/authoritative-index-v2.json`, SHA-256 `1fb3dd6905da7bfab2bd351dc486a3f525cc9f49c2da0bce9029d0f4d0c0c7da`.
- Completed bank compatibility: `momentum-preflight-bank-compatibility-sol-v1/compatibility.json`, SHA-256 `53337fcbd640c1ece9c31f4c2c44cf107bbdcd60bc5bee912716109a3806ba05`.
- Sealed calibration preparation plan: `momentum-calibration-preparation-sol-v1/prepared-v1/plan.json`, SHA-256 `f5201d12ec69f378e03cf14af6aa0835ef84cfbbe213e6101dedc575297d16c9`.
- Exact update implementation inspected: `src/forecasting/polling_basis_v1.py`; use the version/hash bound by the relevant frozen source receipt at execution time.
