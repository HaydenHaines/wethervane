# M02 persistent poll-trend challenger: no-fit experiment plan

**Date:** 2026-10-25  
**Status:** Specific experiment proposal only. No source, model, or output files were changed; no fit was run. This is a revisitable development experiment, not an outcome-blind confirmation.

## Purpose and decision

The v2 descriptive audit answers a narrow question about observed rising candidates. M02 asks a different forecast question: does a small, zero-centered persistent within-race trend component improve forecasts of the eventual conditional D/(D+R) share or winner probabilities over the selected polling model's existing date-dependent error law?

This is one replaceable component inside the polling measurement layer. It does not add a positive “momentum bonus,” treat a poll's age as a fixed lag, identify a causal change in opinion, or train only on the 12 threshold-qualified rising cases. The direct audit's 12 cases among 31 supported races remain evidence for the descriptive question, not enough independent information to set a forecast adjustment. Use the full eligible poll series under the frozen question/sample selection rules, and score every race supported by both arms.

Keep M01 as the exact incumbent and M02 as its sole challenger. No production change, 2026 adjustment, or claim that polling is the weakest layer follows from this proposal.

## Incumbent law actually selected

The selected standard poll likelihood is in [`poll_covariance_v1.py`](</home/hayden/projects/wethervane-redesign-20261006/integration/src/forecasting/poll_covariance_v1.py>) and [`polling_v1.py`](</home/hayden/projects/wethervane-redesign-20261006/integration/src/forecasting/polling_v1.py>). It compares conditional principal-contender log odds with a zero-mean Gaussian working-error covariance. Five owned components are added: independent poll error, cycle-wide error, race-persistent bias, same-pollster/same-cycle bias, and race Brownian movement. The time coordinate is days to election divided by 30. For observations `i,j` in the same race and cycle, the selected movement kernel is

\[
K^{BM}_{ij}=\min(d_i,d_j)/30,
\]

and zero across races. It is an independent-increment discrepancy anchored to zero on Election Day. Its fitted SD is `0.08071632721709084` conditional principal-contender logit units per square-root 30-day unit. The selected scale was a regularized MAP estimate using 1,514 historical observations across 2012, 2016, 2018, and 2020, 106 races, and 220 pollsters; its parameter uncertainty is not integrated. Earlier-only rolling folds must refit the covariance scales and must not reuse this all-four-cycle estimate for a target before 2020.

The current standard covariance training rules, selection, and MAP regularizer are documented in [`01-standard-plan-root-v1.json`](</home/hayden/projects/wethervane-redesign-20261006/integration/research/polling-layer-implementation-20261020/forecast-v2/source-snapshots/01-standard-plan-root-v1.json>) and [`07-calibration.json`](</home/hayden/projects/wethervane-redesign-20261006/integration/research/polling-layer-implementation-20261020/forecast-v2/source-snapshots/07-calibration.json>). The incumbent’s movement term describes timing uncertainty; it is not an estimated directional slope. Do not add a second nominal-N/MOE term to its historically fitted total error.

## Smallest persistent-trend alternative

Add one private, versioned random-slope covariance component, `persistent_linear`, alongside all five incumbent terms. It is the marginal covariance of one zero-centered race-specific slope shared by every poll in that race and election cycle:

\[
\beta_r \sim N(0,\tau_{slope}^2),\qquad x_i=d_i/30,\qquad
K^{slope}_{ij}=1(\text{same race, same cycle})x_i x_j.
\]

Equivalently, conditional poll means include `beta_r * x_i`, with the slope zero at Election Day. Units for `tau_slope` are principal-contender log odds per 30-day time unit. Because `d_i` counts days *before* Election Day, a positive realized slope means earlier polls for that oriented contender are higher and its support falls toward Election Day; a negative slope means it rises toward Election Day. The prior is symmetric, so neither direction receives an advance. This is a random-slope term, not a candidate win-probability bonus.

This rank-one within-race kernel models a drift that keeps its direction across the campaign. It is distinct from Brownian movement’s independent increments, but can compete strongly with it when a race has only a few polls or a narrow time span. The M02 forecast must report the fitted slope scale, the correlation/identifiability diagnostic against `movement`, and race-time coverage. A good aggregate score alone cannot identify the mechanism.

Retain the exact same source, poll/question selection, candidate orientation, Gaussian likelihood, fundamentals prior, race universe, measurement-error components, forecast cutoff, and score contract for M01 and M02. Use the [current private generic basis and likelihood contracts](</home/hayden/projects/wethervane-redesign-20261006/integration/src/forecasting/polling_basis_v1.py>) only if their existing support can express the augmented law without changing their estimand; otherwise add an experiment-only covariance implementation. Do not alter `poll_covariance_v1.py` in place.

### Shrinkage and estimation

Use a zero-centered, regularized slope distribution; no sign restriction or estimated nonzero population mean. For the first diagnostic version, use the same regularized log-scale MAP framework as the incumbent for all covariance scales, including `persistent_linear`, and disclose that scales remain conditional point estimates. Predeclare the slope log-SD regularizer median as 0.05 logit per 30 days, matching the incumbent movement component’s regularizer median as a transparent starting scale, with log-SD 0.7. This is a shrinkage convention to test, not a literature-derived truth. Run versioned 0.025 and 0.10 medians as sensitivity arms; do not choose a median by the held-out score.

Refit the complete vector of covariance scales jointly in every M01 and M02 training fold. M01 refits the five incumbent scales; M02 jointly fits those five and the new slope scale. Never add the slope to a frozen Brownian scale: that would confound the added mechanism with an unrefitted total-error decomposition. Keep covariance parameter-uncertainty treatment identical between arms. If a later version integrates scale uncertainty, create matched integrated-scale M01 and M02 arms; do not compare a Bayesian mixture in one arm with conditional MAP scales in the other.

The existing optimizer’s numerical box is not automatically scientific prior support. If the MAP solver needs finite bounds, state them in the new objective and make the assumed truncated log-scale prior explicit, including its normalization in any marginal-likelihood comparison. Prefer an unbounded optimizer where stable; in either case report estimates near any computational boundary and reject a boundary solution as inconclusive for this challenger.

## Poll timing and source availability

The current standard runner uses field-end age for `days_to_election`. Preserve that exact coordinate for the primary M01/M02 comparison so that adding a trend is the only change. The pinned 538 archive for 2018–24 has reported poll start/end dates, but these are poll-level field intervals, not interview-level response dates or response weights. There is no basis for inventing a respondent median date or saying all respondents were interviewed at the interval midpoint.

Predeclare one interval-aware **sensitivity**, separate from the primary comparison:

- Represent each reported interval by its inclusive day range `I_i`. Under an explicit uniform-within-reported-field-period working assumption, use `x_i = E[D_i]/30` for the slope kernel and replace Brownian `min(d_i,d_j)/30` with `E[min(D_i,D_j)]/30`, where `D_i,D_j` are independently drawn from their reported intervals. This integrates uncertainty over the interval without claiming those are actual respondent interview dates.
- Run that sensitivity only for polls whose exact source row supplies valid start and end dates; retain and report its smaller matched race mask. Do not substitute interval midpoints as actual dates, weights, or verified interview times.
- The older 2012–20 Tierney/Volfovsky author panel lacks field-start dates. Therefore a fully interval-aware four-fold comparison cannot be claimed on that training series; primary historical calibration stays on its documented field-end proxy. An interval-aware result for 538 target rows only is a labeled sensitivity, not the matched historical confirmation.

Both the historical panel’s `available_date` and the 538 archive’s `created_at + one day` are availability proxies. Historical original publication timestamps/revision histories are not verified. Require field end no later than the requested cutoff, apply the same created-plus-one-day proxy rule in both arms, and add predeclared release-lag sensitivities of 1, 3, and 7 days. Keep every result `strict_asof_qualified:false`; do not describe it as information verified public on that date. If source selection differs after the lag filter, refilter before deduplication identically for both arms.

## Evidence available and remaining limits

Two pinned inputs provide different, non-combinable evidence:

1. The admitted [historical intake](</home/hayden/projects/wethervane-redesign-20261006/integration/research/polling-layer-implementation-20261020/historical-intake-sol/README.md>) has 1,837 Tierney/Volfovsky poll records across 2012/2016/2018/2020. It is the existing development source for earlier-only covariance calibration. It lacks original publication times, field starts, candidate IDs/names, mode, nominal/effective sample size, sample IDs, and response overlap. Its candidate linkage is state/cycle roster based. Treat its availability as a field-end proxy.
2. The pinned 538 archive used in [momentum audit v2](</home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024/momentum-audit-v2/REPORT.md>) has 9,292 candidate-answer rows across 2018/2020/2022/2024 (2,016 / 2,559 / 2,089 / 2,628 raw rows; these are not independent polls). It preserves start/end, pollster, sample/question IDs, candidate identity, population, method, and `created_at`, but not respondent interview dates or a verified publication/version history. v2 has 113 supported ordinary D/R roster races overall, 31 races with adequate preceding/final windows and truth, and 12 meeting its 1-share-point rise threshold. Its three reviewed alias joins and NC FEC-ID trim must be retained; do not revert to v1’s lost 90 poll rows or loosen its other gates.

Do not concatenate these inputs as independent polls. Release-level overlap between sources is unknown. The primary proposal uses the Tierney panel only to fit earlier-only error scales, then the 538 catalog as the target-cycle poll source for 2018/2020/2022/2024. Source exchangeability and cross-source dependence remain limitations. A separate source-specific 538 calibration may be reported only when earlier-cycle support and a complete matching cutoff are available; it is not primary and must not double-count Tierney rows.

Before any fitting, run a **no-fit coverage audit** by target cycle × horizon × arm: count raw and deduplicated sample/questions, pollsters, races, start/end coverage, matched questions/candidate identities, same-pollster repeated time spans, and outcome-complete races after each availability lag. Include the retained D/R pair only; specials, RCV, runoffs, candidate changes, hypothetical questions, and different populations/candidate lineups follow the existing admissions rules. A cell with inadequate common support stays unavailable. More raw candidate rows do not establish enough independent election cycles or poll samples.

## Fair earlier-only replay

Use the same 2018/2020/2022/2024 Senate targets in both arms, with context tiers for a 2026 target: **A = 2018 and 2022 midterms; B = 2020 and 2024 presidential cycles**. These are revisitable development outcomes, not protected holdouts. Freeze each version’s poll rows, candidate identity, source pins, fundamentals bank, scale-training rows, horizon, seed policy, and score mask before scoring that version. Since these outcomes and earlier benchmark results have already been examined, label the result post-hoc development evidence; do not claim blind confirmation.

For each target cycle, train calibration and any learned nuisance parameters only on poll/error/outcome information from earlier cycles. In particular, the 2018 scale fit has only 2012/2016 training cycles and the 2020 fit only 2012/2016/2018. Treat those early scale fits as weakly supported if the components are prior-dominated or highly correlated. The candidate slope for a target race is inferred from its cutoff-eligible poll series under the zero-centered shrinkage law. Use the exact same dated poll archive/cutoff filters and all common races for M01/M02. Do not let target-cycle outcomes select poll questions, cutoff, prior scale, margin bands, or inclusion.

Recommended horizons: 14 days before Election Day as the primary common comparison, with 30-day and 7-day sensitivities. The older source has sparse 90-day cells (for example, 2012 has only 11 rows across 6 races and 2018 only 2 across 2 at 90 days), so 90 days is a coverage diagnostic unless a no-fit audit shows adequate matched support for all four target cycles. Preserve the benchmark convention of applying availability first, then selecting one latest sample per race/pollster or one question per sample under the incumbent rules. M01 and M02 may not use different numbers of observations to improve scores.

Historical fundamentals banks must be reconstructed within each earlier-cycle cutoff under the same baseline method. The currently selected 2026 prior trained through 2020 cannot serve as the 2018 or 2020 bank. The existing historical poll intake explicitly says these earlier-fold fundamentals draws still need regeneration. Keep poll-likelihood and fundamentals uncertainty aligned by common draw identity; do not independently shuffle coefficients/scales or add upstream fundamental variance into total polling error.

## Scores, diagnostics and interpretation

Primary score: empirical conditional D/(D+R) share CRPS on the common race mask, equal-weight races within cycle. Report per-cycle losses and equal-cycle A and B summaries. Also report winner Brier and log score, share WIS/80% and 95% coverage and widths, baseline/always-half/frequency references, poll coverage, and score MCSE. Keep polling-share errors in share percentage points separate from D-minus-R margin errors (a one-point conditional-share error is two margin points). Show all four cycles; do not pool thousands of poll rows as independent elections.

Use the same saved fundamentals/posterior bank, poll question/sample selection, random-number budget, target race IDs, and draw representation for each matched pair. Verify covariance SPD/Cholesky, rank/condition of the augmented race-time covariance, finite positive scales, optimizer convergence from multiple starts, and slope/Brownian parameter correlation. Require no silent fallback or changed input when a gate fails. For the sampler, use at least four chains, at least 2,000 warmup and 4,000 retained draws (or stricter current production settings), rank R-hat ≤1.01, bulk/tail ESS ≥1,000 for nonconstant parameters, and report win/score MCSE. A failed gate preserves that attempt and bars substantive score interpretation until repaired under a new version.

Use the existing practical CRPS review aid (0.0005 absolute or 2% relative) and the current “flag losses beyond two numerical MCSE” rule as prompts to inspect effects, not automatic adoption/rejection thresholds. Investigate material regressions in either A cycle, B transfer, horizon, source-lag, and interval sensitivity. No requirement that M02 win every cycle; no one-cycle gain is enough to claim generalized improvement. Retain M01 if the score advantage is within MC error, highly cycle/source dependent, driven by methodological changes, or unsupported by adequate time-spanning same-pollster evidence. A good score does not identify the historical mechanism; a fitting trend does not establish predictive value.

The v2 descriptive 12-case result and earlier model scores are already visible. They must not be treated as untouched validation. Preserve those prior values, show each new version against its own matched M01 baseline, and record any version changes made after seeing prior scores. No A/B/C tier is an automatic veto and no sealed holdout is imposed; the requested iterated development loop may revisit these cycles while stating that this is retrospective model development.

## Execution dependencies and stop conditions

This plan is **not executable as a forecast-quality comparison yet**. The movement and poll likelihood code exist, and the four-cycle target archive plus 2012–20 historical panel are present. Remaining prerequisites are:

1. A source-pinned no-fit coverage report for each cycle/horizon/lag and common M01/M02 race/sample masks; no original publication vintages become verified by the current proxies.
2. Earlier-only historical fundamentals posterior banks for 2018, 2020, 2022, and 2024 under the exact same baseline law and source/output identities.
3. A separate M02 covariance contract/implementation with a documented `persistent_linear` owner, explicit scale prior, complete six-scale refit, and tests against analytic small-matrix random-slope covariance/conditional forecasts; M01 stays byte-pinned.
4. A frozen scoring/evaluation version binding both A and B cycles, equal-cycle summaries, horizons, lag/interval sensitivities, numerical gates, and source receipts before its first score read.

If the slope scale runs to a solver boundary, posterior/brownian scales are not separable, time-spanning source support collapses under matching, or apparent gains vanish under the same-pollster/source/lag checks, report that and retain the incumbent. The plan does not claim any result in advance.
