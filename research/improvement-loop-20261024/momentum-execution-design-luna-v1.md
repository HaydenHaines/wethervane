# Matched M02 forecast comparison specification (proposal only)

**Status:** no-fit, no-score design proposal for independent review. It does not authorize or perform optimization, forecast generation, polling-source admission, or promotion. The calibration optimization decision belongs to Root. It makes no claim that polling or any other layer is the weakest layer.

## Question and estimand

Estimate whether adding one persistent linear time component to the existing poll-error covariance changes out-of-cycle D/R share and winner forecasts when all other inputs and source rows are matched. The time coefficient is anchored at Election Day: with `x = days_before_election / 30`, the slope covariance contribution is `tau_slope^2 * x xᵀ`; it is not a deterministic “rising candidate” adjustment. Forecasts use the target-cycle poll vector and F0's empirical fundamentals prior, with poll-likelihood weights updated across intact F0 posterior rows. The target is the **marginal empirical-prior polling update**, not a calibration cut and not an integrated covariance-scale posterior.

The minimal scientific contrast is the paired difference between the existing five-component Brownian total-error law (M01) and the six-component law adding the slope kernel (M02). Each arm re-estimates all its covariance scales jointly on exactly the same earlier-only calibration rows and regularizer. No scale is held fixed in M02. The zero-slope covariance limit is a mathematical identity test; a positive finite log-scale penalty means the optimizer may not attain zero exactly.

## Frozen fold, training and bank contract

Use the root-selected four expanding historical folds and exact F0 relative-context bank for each target cycle:

| Target | Strictly earlier calibration cycles | Selected calibration rows | F0 target races |
|---|---|---:|---:|
| 2018 | 2012, 2016 | 930 | 30 |
| 2020 | 2012, 2016, 2018 | 1,218 | 28 |
| 2022 | 2012, 2016, 2018, 2020 | 1,514 | 30 |
| 2024 | 2012, 2016, 2018, 2020, 2022 | 1,514 | 30 |

The historical poll calibration selector is the selected incumbent operator: all strictly earlier cycles, 0–100 days before the election, retaining the last observation by cycle/race/normalized pollster/fixed 14-day lead bin. Use the **same ordered source observation IDs**, D/R residuals, and race identities in M01 and M02. Fit five versus six log-SD parameters jointly, with the frozen Normal penalties on log SD (medians 0.10, 0.06, 0.06, 0.03, 0.05, and M02-only 0.05; log SD scale 0.7), no natural-SD Jacobian and no new sampling-error term. This is the incumbent conditional-MAP objective, not marginal evidence.

For each fold, use all 16,000 raw F0 joint posterior rows in original chain-major order, equal **initial** weights, preserving beta/tau dependence and draw identities. Never independently shuffle coefficient or scale columns, substitute the five already-simulated F-only future replicates for bank rows, or reuse a bank from a later target fold. Poll likelihoods update row weights; posterior row weights therefore need not remain equal. Use the existing six-column `EmpiricalBasisFundamentals` law and the exact F0 design order. Generate the target shared cycle effect and conditional race errors according to the existing updater law, shared across matched arms as appropriate for paired Monte Carlo. The same fold bank is shared by M01/M02; only the jointly fitted polling covariance differs.

These are the refreshed, source-pinned earlier-only F0 banks, not the older selected two-column fit. The 2020 Texas 31 rows stay in this selected MAP calibration target. Do not substitute the separate 1,483-row disjoint calibration dataset. The bank’s target feature receipts are H14-bound; retain the same H14 feature rows for H7 and label that explicitly as a horizon sensitivity. H30 is excluded until its own target-feature cutoff admission is established.

## Target poll operators and horizons

Keep the historical calibration selection above fixed. At the target election, compare two **separate** poll operators; never pool their IDs or attribute an operator change to the slope kernel:

1. **Primary: latest sample per race/pollster.** Apply the selected current-forecast rule after the H14 or H7 availability filter, using one latest observation per exact race/pollster and one question per sample as specified by the source adapter. This is the closest replay of the existing target operator. At H14 native proxy/no extra lag, coverage is 170/134/171/226 observations across 25/22/23/27 races in 2018/2020/2022/2024, respectively.
2. **Separate sensitivity: full eligible series.** Retain every source row that passes the same cutoff, identity, population, question and deduplication rules. Do not call its advantage “slope value” without the within-operator M01/M02 contrast. At H14 native proxy/no extra lag this is 202/190/220/351 rows across the same 25/22/23/27 races. In the full-series operator, 25/32/38/74 race-pollster groups repeat in 2018/2020/2022/2024. Distinct sample IDs do not establish independent respondents; the fitted shared house term is not a verified overlap correction.

H14 is the primary cutoff. H7 is a secondary poll-data cutoff sensitivity using the same H14 F0 target features, same target roster, same fold bank and the exact same two target operators. Do not interpret it as a fully refreshed H7 information set. For each cutoff, use field-end age as currently defined by the frozen adapter; apply only its native source-availability proxy with additional lag 0. Preserve separately labeled +1/+3/+7-day extra-lag sensitivities if later selected. Do not interpret numeric lags as equivalent publication-date information across sources. Original release/revision times are not verified (`strict_asof_qualified:false`).

Do not integrate advertised field intervals in this comparison. Interval averaging changes the observation operator and is not an established interview-date distribution; 90-day ActiVote metadata in particular is not 90 daily observations. Any interval operator must be a separately versioned follow-up, not a hidden preprocessing tweak.

### Exact race masks and identity gaps

For each target operator × cutoff, bind a complete selected-ID ledger and derive a single shared event-ID score mask before forecast generation. Intersect exact source-proven regular-event IDs, F0 bank target IDs, actual outcome IDs, and rows present under that operator/cutoff. Use the same mask for F0-only, polls-only, M01-combined and M02-combined scores. Also report F0-only results on the full bank target roster as a clearly separate coverage reference; do not compare a full-roster score directly with a smaller polled-race score.

The five identified event-identity gaps are not a zero-poll assertion. Use only source-proven event joins after explicit review/admission. 2018 MN, 2018 MS, 2022 OK and 2024 CA have proposed source-supported joins; the separate identity-intake packet is not itself admission. 2022 CA remains unresolved between 538 races 8921 and 9480; exclude both from poll-updated masks until a provider question-ID/event key proves the regular contest. Never merge the two events or infer one by row order/candidate identity. Record every excluded event and reason in the per-fold mask ledger.

## Comparator arms

Use the smallest arm set that separates the questions, without changing data between matched kernels:

| Arm | Fundamentals | Target poll data | Poll-error law | Purpose |
|---|---|---|---|---|
| F0-only | F0 relative-context empirical prior | None | None | Fundamentals reference; all bank races and, separately, common score mask |
| Poll-only-M01 | Pooled earlier-only outcome prior, no current covariates | Same exact target poll vector | Five-component | Standalone polling reference |
| Poll-only-M02 | Same pooled outcome prior | Same exact target poll vector | Six-component | Shows the slope law without F0 information |
| F0 + poll-M01 | F0 relative-context prior | Same exact target poll vector | Five-component | Existing combined baseline |
| F0 + poll-M02 | Same exact target poll vector and F0 prior | Six-component | Candidate combined model |

Define the **poll-only** prior explicitly rather than using an unspecified poll average: for each fold, fit a pooled Normal prior on the logit of observed actual D/(D+R) share among that fold’s exact strictly earlier F0 calibration races, using equal cycle weight and equal race weight within cycle. It has a single pooled mean and variance, no current-race covariates or incumbency inputs. For polled target races, update that prior with the same full-vector Gaussian likelihood, selected poll IDs, and fold-specific M01 or M02 covariance. For unpolled target races, retain the pooled prior prediction. Freeze this prior recipe and the handling of structural zero/one D/R denominators before any scores; if source-supported actual shares cannot be represented, keep them out of this poll-only prior and document the exact exclusion rather than silently clipping. The poll-only arms are descriptive standalone controls; the causal incremental contrast is not identified by them.

The key contrasts are (a) F0-only versus F0+poll-M01 for the incremental value of polls under the existing covariance law, and (b) F0+poll-M01 versus F0+poll-M02 for the incremental change from adding the slope component under matched inputs. Poll-only-M01/M02 and the kernel pair of standalone forecasts help diagnose whether a result depends on fundamentals, but do not multiply downstream variants or replace the paired combined contrast.

## Predictions, common errors and scores

Every arm predicts the D/(D+R) share distribution and D/R winner probability for the exact same target race IDs. For an updated F0 arm, condition on the complete multivariate poll vector: evaluate the likelihood under each intact bank row, normalize all empirical row weights jointly, and propagate the same draw's fundamentals and scales into each target race. Preserve dependence across races through the shared target-cycle effect and covariance terms. Do not independently mix marginal beta/tau distributions or convert missing polls into zero-valued observations. Use common random numbers/paired draw identities between M01 and M02 for variance reduction, without deleting or resampling bank rows; document any unavoidable RNG differences.

For each target race report: posterior predictive D/R share mean and median; central 50%, 80% and 95% share intervals; D/R win probability (with an explicitly consistent tie rule); and the archived observed D/R share and official winner used for scoring. Keep the two-party-share outcome separate from third-party vote and winner margins. No synthetic tipping-point or control interpretation.

Predeclare race-level proper scores:

- **Share CRPS** for the continuous D/(D+R) outcome.
- **Winner Brier score** and binary log score for the official D/R winner, with smaller values better. Log-score probabilities must be numerically bounded using a fixed, stated rule before scoring; report both raw predictive probability support and any numerical floor.
- **Share WIS** from the 50/80/95% central intervals, plus interval coverage and width. WIS and CRPS evaluate share forecasts; winner scores evaluate binary outcomes.
- Report simple winner accuracy only as a secondary descriptive statistic, not as a proper-score substitute.

Average races equally within each cycle. Report all four per-cycle values, paired per-race M02−M01 score differences, and a macro-average giving the four cycles equal weight; also separate A cycles (2018/2022) and B cycles (2020/2024). Never pool all races as if independent cycles or let poll counts weight a cycle. Report masks and race counts beside every score. Four cycles are a small retrospective development sample: cycle-to-cycle spread is descriptive, not a significance test or universal generalization.

## Numerical and diagnostic gates to freeze before execution

Root’s execution plan must set numeric thresholds before any new scores are inspected. At minimum:

1. **Calibration fit:** verify the exact source-ID ledger and fold-specific residual arrays before optimization. Use deterministic starts plus multiple dispersed starts for each arm; report optimizer success, objective and gradient norm, boundary hits, start-to-start parameter/objective spread, Hessian/profile conditioning and parameters on the natural scale. A six-scale solution that is nonidentified with Brownian movement or prior-dominated is “inconclusive,” not an automatic slope win. The fifth/sixth scale penalties must match the root-selected log-coordinate convention.
2. **Forecast integration:** report per-arm normalized empirical weights, effective sample size `1/sum(w²)`, maximum weight, and by-chain contribution. Set a minimum ESS and maximum-weight gate before scoring. If an arm collapses onto a few rows or chains, do not interpret its score; increase valid numerical integration or mark it inconclusive under a predeclared procedure. Preserve the original complete joint rows and provide weight-normalization and finite-value checks.
3. **Monte Carlo reliability:** batch/split the retained bank rows and any predictive simulation, compute MCSE for every reported score and central interval endpoint, and require MCSE to be small relative to the predeclared meaningful score resolution. Do not call between-cycle outcome variation MCSE. Verify predictive shares and win probabilities are finite and in [0,1], weights sum to one, observed truths are on the same D/R denominator, and paired M01/M02 draw alignment is reproducible.
4. **Law identity:** deterministic checks before any fit/score: five-term likelihood equals the six-term zero-slope-variance limit; covariance is symmetric/positive definite for supported inputs; the objective/gradient agree with an independent implementation at supplied points; and changing only M02’s slope coordinate leaves every other component unchanged. Check row normalization and F0-only law against the frozen bank implementation.
5. **Forecast masks:** one machine-readable mask file per fold/operator/cutoff binds exact IDs and truth hashes. Identical IDs are used in all arms within a paired comparison. Missing or ambiguous event joins fail closed; no score table is emitted for an invalid mask.

A solver, bank-weight or predictive-integration failure is a computational failure, separate from a predictive regression. Conversely, successful numerical diagnostics do not establish a forecast benefit.

## Interpretation, scope and permissible next step

The comparison is retrospective, based on approximate historical availability proxies, and has only four national cycles (two A and two B). A result can motivate another versioned challenger, but it cannot by itself establish causal mechanism, broad transfer, calibration qualification or a weakest-layer ranking. Inspect material regressions by cycle, operator, cutoff, race class and baseline; do not impose an all-cycles-win rule or promote based on a single noisy aggregate. Any post-score change is exposed development and must preserve the preceding plan/results unchanged.

Not included: H30 (feature cutoff not admitted), field-interval averaging, new historical polls, unresolved CA 2022 event joins, a different target race roster, extra Texas-disjoint training, alternate priors selected from outcomes, natural-scale Jacobian changes, a calibration cut, integrated covariance-scale uncertainty, API/UI activation, or 2026 extrapolation.

**Decision request for independent review:** approve or amend this frozen comparison contract before Root separately selects the permitted covariance-calibration optimization and any execution plan. No model fit or forecast comparison has been run under this proposal.

## Binding references

- Root preflight review: `momentum-preflight-root-review-v1.md`, SHA-256 `22f25174f9412c3b876784753023bd7c229a91642b4a82b4aeaab41319d6e669`.
- Independent M02 challenger review: `momentum-challenger-review-sol-v1.md`, SHA-256 `0b6e855ce79551907938cbe2bb55009449dc5d7a5912bcc23f8497655d9be8a6`.
- Corrected coverage authority: `momentum-preflight-coverage-luna-v1/authoritative-index-v2.json`, SHA-256 `1fb3dd6905da7bfab2bd351dc486a3f525cc9f49c2da0bce9029d0f4d0c0c7da`.
- Completed bank compatibility: `momentum-preflight-bank-compatibility-sol-v1/compatibility.json`, SHA-256 `53337fcbd640c1ece9c31f4c2c44cf107bbdcd60bc5bee912716109a3806ba05`.
- Root calibration-preparation selection: `checkpoints/momentum-calibration-preparation-root-selection-v1.json`, SHA-256 `f30f0a92920647276afa44f741515ff0cebfe5fe18417e273ae975d6133469e9`.
