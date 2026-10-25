# Polling momentum: what is known and what a fair test needs

**Date:** 2026-10-24. **Status:** literature pass and preregistration sketch only. No poll archive audit or fit was run here. The separate archive-frequency audit is owned by root.

## Question and distinctions

The proposed claim is conditional: among candidates whose support rose late, is the eventual result more often above than below the final polls’ estimate for that same candidate? That is different from the established average pattern that polls tend to become more accurate nearer Election Day. It is also different from whether a particular pollster’s estimate is stale when published.

Keep three dates separate:

- **Fieldwork dates** say when respondents supplied answers. A poll estimates opinion during that interval, not on its publication day.
- **Release/publication date** says when the public could first use the result. Release lag varies; exclude any poll that was released after a forecast cutoff even if its fieldwork had finished before it.
- **Election day** is the outcome date. Any poll-result difference combines survey error, preference changes after fieldwork, turnout/participation changes, and possible vote-count differences. It does not identify a public-opinion “lag” by itself.

A fixed 1–2 week lag claim is not supported by the sources reviewed. AAPOR’s disclosure standards call for collection dates to be reported, which allows the field period to be distinguished from release date; the field period can itself span several days ([AAPOR Transparency Initiative](https://aapor.org/standards-and-ethics/transparency-initiative/)).

## Primary-literature assessment

1. **Campaign movement is real on average, but noisy poll-to-poll changes are not all movement.** Wlezien & Erikson analyze aggregate polls in 15 U.S. presidential campaigns (1944–2000), explicitly separating preference movement from survey error and estimating persistence of campaign shocks. This supports time-aware modeling of opinions; the population, decades, and presidential target do not establish a late-trend bonus for House/Senate candidates. [Journal of Politics, “The Timeline of Presidential Election Campaigns”](https://doi.org/10.1111/1468-2508.00159).
2. **Average poll error tends to fall toward the vote, but this is not directional evidence for the candidate trending up.** Jennings & Wlezien analyze 30,000+ national polls from 351 elections in 45 countries (1942–2017); errors evolve systematically over the campaign, with average polls becoming more informative as Election Day approaches, and no general increase in final-week error over time. The study focuses on election-level poll accuracy, not the conditional frequency `P(actual > final poll | recent within-pollster rise)`. Cross-national national-election results are not U.S. district-level effect sizes. [Nature Human Behaviour, “Election polling errors across time and space”](https://doi.org/10.1038/s41562-018-0315-6).
3. **Poll error is substantially wider than ordinary sampling margins suggest.** Shirani-Mehr et al. examine 4,221 polls in 608 U.S. state-level presidential, Senate, and gubernatorial elections (1998–2014), decomposing errors into election-level bias and variance; the reported average RMSE is about 3.5 percentage points. A small apparent weekly rise can therefore be noise or a change in poll composition. Their average-error result does not establish which side of the error a candidate with a rising trend will land on. [JASA, “Disentangling Bias and Variance in Election Polls”](https://doi.org/10.1080/01621459.2018.1448823).
4. **Naively subtracting polls from results confounds shifting preferences with polling error and is sensitive to the time window.** Tierney & Volfovsky’s 2025 hidden-Markov analysis of U.S. presidential polls (2004–2020) and Senate polls (1992–2020) models preferences as evolving over time; it reports substantially lower bias estimates than direct poll-result comparisons and argues that time-window choice matters. This is a direct warning against treating election-day poll residuals as pure pollster bias or a fixed lag. Citation verified against the journal’s issue record: [JRSS A, “Bias and excess variance in election polling: a not-so-hidden Markov model,” 188(2), 566–582 (2025), DOI 10.1093/jrsssa/qnae066](https://academic.oup.com/jrsssa/article/188/2/566/7720791).

**Evidence conclusion:** Literature supports that opinion can move, survey noise is material, and average forecasts improve closer to Election Day. It does **not** establish that late-rising candidates beat final polls more often than they miss low, nor a universal 7–14-day delay. The conditional frequency is an empirical question for the bounded archive.

## Proposed frequency audit (root to bind to the admitted archive)

### Estimand

For each race, identify the candidate direction from a **same-pollster** change between (a) polls whose fieldwork ends 7–13 days before Election Day and (b) polls whose fieldwork ends 0–6 days before Election Day. Use a fixed seven-day maximum fieldwork length, matching election, candidate list/question, population (registered-voter vs likely-voter), mode/method where recorded, and pollster organization. Do not pair one firm’s earlier result with another firm’s later result to define a trend.

Operationalize “rising” as an increase in a candidate-oriented two-major-candidate margin, not a rise in raw topline vote share; this avoids claiming to measure undecided allocation or candidate preference separately. Orient the signed poll lead to the candidate who rose between the windows. The primary outcome is whether that candidate’s actual two-candidate margin exceeds an **independent** final-week poll estimate. Build the estimate from the final-week polls (fieldwork end 0–6 days before Election Day) excluding every polling organization used to define that race’s trend. Keep one final estimate per race (predeclared median of pollster-level estimates) so races with many polls do not count as more independent evidence. Positive residual means the rising candidate outperformed the independent final-week poll estimate; negative means underperformed; exact equality is a tie, retained as a third outcome rather than assigned to either side.

This cross-pollster outcome avoids the direct mathematical coupling from using the same late poll both to declare a candidate “rising” and to calculate that candidate’s poll error. Also report a secondary, clearly labeled **same-pollster final-poll residual** against that firm’s last poll; because the same final observation contributes to both the rise and residual, regression-to-the-mean coupling is a known limitation and that result is not the primary test.

### Thresholds and robustness

- Primary “rising” threshold: a within-pollster increase of at least **1.0 percentage point** in the candidate-oriented margin between the two fieldwork windows; require at least one eligible poll in each window and at least one independent final-week pollster.
- Report every eligible pair’s signed change, not just the thresholded cases. Estimate the continuous relation between recent change and out-of-pollster final-week residual, with race and election-cycle structure; don’t select a best-looking threshold after seeing outcomes.
- Sensitivities fixed in advance: 0.5pp and 2.0pp thresholds; alternative bins 14–8 vs 7–0 and 21–8 vs 7–0; separate two-party vote-share and margin targets if both are available. Keep candidate support and margin measures consistent between polls and final returns; don’t reallocate undecided voters post hoc.
- Match on candidate identity and ballot/question wording. Treat substitutions, withdrawals, new candidates, runoff/RCV contests, and polls with materially different candidate lists as ineligible for a simple same-pair comparison or place them in a separately specified analysis.

### Test and dependence

The direct descriptive question is: among races meeting the threshold, what share have positive versus negative out-of-pollster residual, with ties shown separately? Compare that frequency to 50% **only descriptively**; add cycle-stratified uncertainty and the rising/falling comparison. Use one equal-weight observation per race, and cluster uncertainty at the election-cycle level (with race nested in cycle); never treat poll records as independent election replications. Report the number of independent cycles and races. With only a few cycles, present per-cycle frequencies and leave-one-cycle-out results; do not make a high-confidence generalization from hundreds of polls in a handful of elections.

Use the exact release timestamp to impose each forecast cutoff, but use fieldwork end date to assign timing windows. Record fieldwork start/end, release date/time, pollster/fieldwork vendor, population, mode, sample size, candidate/question set, topline definitions, sample type, and source URL. Keep release-publication delay as its own descriptive variable; test a field-end-to-release interval only if coverage is complete enough and the source timestamp is credible. It must not be interpreted as polls “lagging the public.”

### Bias and selection checks

Pollsters that publish multiple waves may differ systematically from those that do not; same-pollster pairing reduces composition changes but limits the sample to repeated firms. Trend-source pollsters are removed from the final benchmark, which reduces coupling but may leave fewer independent final polls. Report eligible/excluded races and missing-fieldwork metadata. Include a pre-specified check for pollster/mode/population changes; do not attribute a change caused by likely-voter modeling or survey mode to actual voter movement. Preserve partisan-sponsored and candidate-sponsored polls as a separate stratum or exclude them by a preregistered rule; selective publication can distort observed trends. No post-cutoff poll or outcome-aware inclusion rule is allowed.

## What would count against the momentum claim

Do not retain a “late rise predicts overperformance” mechanism if the direction is near 50/50, varies sign across cycles, disappears after same-pollster matching or independent-pollster outcome construction, is driven by a single high-profile cycle, tracks methodology/source-composition changes, or loses to the no-momentum baseline on cycle-held-out final-margin predictive scores. If mean absolute poll error declines closer to Election Day but rising candidates do not have positive conditional residuals, report the established average finding and reject the directional momentum claim.

No result in this literature review establishes a Ballotmancy-specific momentum effect. Use the root-owned bounded Senate archive audit first, and only propose a replaceable model layer after its coverage, independent-cycle count, and leakage checks are known.
