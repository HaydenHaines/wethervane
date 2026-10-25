# Sol: inter-layer complementarity research and round-two critique

Date: 2026-10-24. Scope: source inspection, primary-source research and bounded functional verification. No scientific model, fit, live process, source admission or public default changed. This report does not rank an empirically weakest layer.

Scientific code anchors below refer to `/home/hayden/projects/wethervane-redesign-20261006/integration/`. The improvement lab is a separate workspace. The House owner owns the four-fundamentals-arm F/FP historical 14-day polling benchmark; this report proposes no duplicate implementation or fit.

## Functional baseline and what the checks establish

All eight canonical roles already have substantive research implementations. The initial pass should skip implemented roles rather than interpret an unpublished control as an absent model. `research/senate-stack-integration-20261020/modular-count-graph-astra-v1/components.py` contains Community at line 59, Environment at 98, Fundamentals at 122, Participation at 150, Choice at 277, RoleChoice at 384, CountsRules at 389 and Measurement at 450. `standard_registry()` and `baseline_composition()` at lines 552–590 connect them. ProtectedPopulation at line 51 preserves an upstream parent; the stochastic population implementations live upstream, not in that passthrough.

The fresh functional check total is **141 passing checks: 110 population-provider checks plus 31 downstream/compiler checks**. These are not 141 predictive backtests or independent election shocks.

- Root's population log: `/tmp/ballotmancy-population-functional-20261024.log`, ending `110 passed in 2.03s`. Root ran source-v2 `test_source.py`, the source independent-v2 review, path-v1 `test_provider.py` and its independent review with importlib import mode. I read the completed log; I did not personally run those 110 checks. Root owns their detailed invocation and snapshot hashes.
- My independent downstream run: `research/senate-stack-integration-20261020/modular-count-graph-review-panel-sol-v1/test_independent_v1.py`, ending `31 passed, 1 warning in 202.17s`.
- Downstream receipt: `/tmp/ballotmancy-downstream-20261024-PczADt/verification-summary.json`; log: `independent-tests.log`; before/after source hashes: `tested-source-hashes.json` in the same directory. Log SHA256: `a5385b612ca1db14d43459031aebc4aad45b3556d90bad7bd2fdd5b051d5b907`.
- Frozen actual-graph witness: `modular-count-graph-review-panel-sol-v1/actual-graph-witness-v2.json`, SHA256 `662a1a06d46cdf2abb922c17cc1f695d59fcbd39699bdca8913b0043639f2789`. Its five compiler source hashes still match the current files. It records the first actual 64-variable-group/662-coordinate graph and all eight original population masses of 0.125. It explicitly compiled no actual likelihood and called no sampler. I did not rerun that witness.

Exact downstream command, from the scientific repository root:

```bash
env PYTHONDONTWRITEBYTECODE=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONPATH=.redesign-runtime/response-20261007/reference-dependencies:. \
PYTENSOR_FLAGS=base_compiledir=/tmp/ballotmancy-downstream-20261024-PczADt/pytensor \
.venv/bin/python -m pytest -p no:cacheprovider \
--basetemp /tmp/ballotmancy-downstream-20261024-PczADt/pytest -q \
research/senate-stack-integration-20261020/modular-count-graph-review-panel-sol-v1/test_independent_v1.py
```

The downstream suite exercises full nonzero supplied-state log density, every prior/likelihood/Jacobian factor, named gradients and physical outputs against the predecessor; unique source ownership; duplicate-evidence rejection; typed ports; original parent-mass invariance; and external replacement/configuration/dependency/target boundaries. Its inherited replacement classes establish interoperability, not the predictive value of alternative laws. PyTensor warned that it could not link BLAS; the numerical assertions passed, but this run qualifies neither fit performance nor forecasting skill. All inspected scientific source hashes remained unchanged; outputs and compiler caches were isolated in `/tmp`.

Computational coherence is therefore supported at the tested boundaries. Empirical complementarity requires jointly refitted comparisons. A component can be mathematically coherent yet add no predictive information, or compete with another component over the same unidentified residual.

## Round one: four bounded candidates

### 1. Integrate uncertainty in polling-error covariance before adding new shocks

**Classification:** uncertainty-propagation gap in an existing measurement/environment boundary, not a missing correlation layer. This is the strongest first research candidate here, not a claim that measurement is empirically the weakest layer.

**Existing coverage:** `src/forecasting/poll_covariance_v1.py:1–16` defines a total-error covariance with independent, cycle, race, house and movement components. `poll_persistence_v1.py` implements recurring-effect alternatives. `src/forecasting/dynamic.py:1–8` already contains coefficient evolution and regional/shared measurement errors. `research/polling-layer-implementation-20261020/final-decision-root-v2.md:34–42` preserves real losses and explicitly states that error scales are conditioned on rather than integrated. It also reports that the recurring exact-pollster/state challenger did not demonstrate practical value.

**Mechanism/target:** uncertainty in estimated covariance parameters can affect the tails of the combined forecast and its joint seat distribution. A presidential/midterm scale difference is a second, separable hypothesis; it must not be asserted as a permanent fact from two dramatic presidential misses. The 2024 Economist authors identify covariance and nonsampling-error estimation as consequential choices, not a universally successful algorithm: [Gelman, Goodrich and Han, 2024](https://doi.org/10.1162/99608f92.a919e3fa).

**Low-risk alternative:** first integrate uncertainty over the existing total-error parameters. Only subsequently test one strongly pooled context-dependent scale contrast, with the equal-scale model as its nested baseline. Do not add another Brownian movement, house-bias or binomial/MOE term atop an already fitted total-error component. The compact ownership revision already prohibits that duplication.

**Data and replay:** existing historical poll/error panels support the declared A2018/A2022 and B2020/B2024 development comparisons, with their recorded retrospective availability proxies. Each target-cycle scale fit must use earlier cycles; a covariance calibrated through 2024 cannot silently be a 2018 predictor. No additional live poll collection is needed for the first matched algorithm comparison. Exact historical release-version evidence remains a separate issue from a matched proxy replay.

**Matched test and rejection:** retain the same source rows, race support, cutoffs, fundamentals and common shock ownership; refit the affected joint downstream block. Compare share/count CRPS and interval scores, competitive-race Brier/log scores and supported joint seat scores. Reject extra complexity when useful concentration does not improve at the declared practical threshold, gains are only wider intervals, or material-regression guards are breached. Report each A and B cycle; adoption need not win every metric or cycle.

### 2. Couple reported participation to the realized event state, rather than adding generic survey weights

**Classification:** a specific bridge challenger already proposed in the design but absent from the selected implementation; not a newly discovered selection layer.

**Existing coverage:** `modular-count-graph-astra-v1/components.py:479–495` has source community, source-by-state community, turnout, choice, response, uncertainty and false-reporting parameters. Its reported-participation probability is `sigmoid(turnout_logit + source_turnout)` at line 495. `design-astra-v1/design.md:79` explicitly suggests comparing a bridge adding `log G_e`; that dependence is not present in the inspected expression. `src/forecasting/selection.py:1–6,29–48` already separates expected electorate preference from a common poll-sample composition tilt and disclaims identified response probabilities.

**Mechanism/target:** survey participation answers and realized election participation may share event-level variation beyond the population propensity. This is a test of the measurement/participation relationship, not a causal assertion that an engagement variable caused a polling miss. AAPOR's investigation documents substantial 2020 error without identifying a single sufficient explanation: [AAPOR 2020 evaluation](https://aapor.org/wp-content/uploads/2022/11/AAPOR-Task-Force-on-2020-Pre-Election-Polling_Report-FNL.pdf).

**Low-risk alternative:** one pooled bridge coefficient on the existing event state, keeping source-specific reporting and false-reporting terms uncertain. Use the same person/source evidence once. Do not introduce a second participation likelihood from a validation outcome already included in the survey table.

**Data and replay:** earlier-wave CES/ANES self-reports and separately admitted voter-validation linkages are candidates; their exact field definitions, record universes, match uncertainty and version dates must be checked. A2018 earlier-wave version authentication is a documented gap. For B2020, A2022 and B2024, use only respectively earlier surveys/releases, not a later cumulative archive merely containing older event years. [Official CES source index](https://tischcollege.tufts.edu/research-faculty/research-centers/cooperative-election-study/data-downloads) identifies annual datasets and questionnaires but does not authenticate old downloaded bytes.

**Confounding/test/rejection:** event frailty, source turnout shifts, source composition and report bias can explain overlapping discrepancies. Refit them jointly. Evaluate participation/report predictive scores plus candidate counts and source-stratified residuals; include a deletion test for the added bridge. Reject a prior-determined coefficient or a change that only moves discrepancy between channels without useful predictive gains. The compact report/count approximation at `design.md:93–99` must retain its own approximation checks.

### 3. Residual within-person dependence across simultaneous offices

**Classification:** a genuine omitted mechanism in the compact model after existing shared-state dependence is conditioned on. It is a scope-dependent challenger, not a reason to add another national election shock.

**Existing coverage:** `design-astra-v1/design.md:87` explicitly chooses conditional independence across office responses given demographic/community cell and common state. The existing participant-pattern table, shared event count and latent utilities already induce dependence. `src/model_stack_event_v1/README.md` and `mixed-count-graph-astra-v1/README.md` preserve supplied joint rows rather than reconstructing independence. Legacy tract split-ticket features in `src/tracts/build_tract_features.py` are aggregate share differences, not identified individual response dependence.

**Mechanism/alternative:** one shrinkage-controlled association for a declared simultaneous contest pair, preserving its supplied marginal probabilities. A finite joint-pattern or copula construction could isolate the remaining dependence without changing the marginal winner model. It must own one joint observation factor, not independent copies of the same margins.

**Evidence:** actual CVRs identify same-ballot combinations in covered jurisdictions. They do not prove national representativeness or improved forecast skill. The published 2020 database covers 42.7 million ballots in 20 states: [Kuriwaki and colleagues, Scientific Data](https://www.nature.com/articles/s41597-024-04017-1). It contains no individual demographic identities, so it cannot by itself identify demographic-specific effects.

**Data and replay:** the cited database article was published **November 28, 2024**, after the B2024 election and 14-day forecast cutoff. Earlier dataset availability requires its own release proof. Earlier CES waves can provide same-person survey responses, subject to source/report-error and original-version checks. A2018/A2022 and B2020/B2024 must not borrow target-cycle ballots as training predictors. Current CVRs can be retrospective evaluation data without being forecast-time inputs.

**Test/rejection:** use pair-pattern log scores and count covariance, then any supported joint electoral outcomes; retain marginal score guards. Reject if apparent benefit vanishes after jurisdiction composition is controlled or the coefficient is only ecologically identified from marginal returns. Defer this for the single-contest Texas pilot: it adds little to that immediate target and needs informative multi-contest evidence. Simultaneous regular/special Senate contests or cross-office forecasts are the more relevant scope.

### 4. Propagate uncertainty in source-vintage growth revisions

**Classification:** an omitted compact bridge or source/adaptation sensitivity, not an absent population-uncertainty layer.

**Existing coverage:** `national-population-joint-astra-v2/README.md` supplies a source-period stochastic law; `national-population-path-astra-v1/README.md` declares its correlated temporal paths and fixed PEP growth covariates. `src/population/` already has source-error/dynamic implementations. `src/population_vintage_benchmark_v1/models.py:1–7` explicitly labels its resident-path replay as a report-proxy diagnostic, not the full population target. Source-period sampling uncertainty, temporal process noise and revisions are different mechanisms.

**Mechanism/evidence:** changes in the growth covariate across vintages may alter population projections and induced turnout predictions. Census documents source and methodology changes between vintages: [Census annual methodology updates](https://www.census.gov/newsroom/blogs/random-samplings/2026/01/updates-data-methodology-population-estimates.html). That is evidence that revisions occur, not proof of a calibrated revision-error distribution or election-score improvement.

**Low-risk alternative:** first compare matched historical-vintage inputs and report sensitivity. Only then consider one pooled revision-error component learned from earlier revisions. It belongs upstream and must not add an independent second observation of the same ACS/PEP controls or downstream feedback into protected population masses.

**Data and replay:** A2018 has prepared 2012–2016 source statistics; annual-path and release-version admission still require checking. For B2020/A2022/B2024 select the latest genuinely available prior-period PUMS/PEP release at each horizon and retain its frozen bytes. Current 2020–2024 PUMS is not an A2018 input. Existing source archives and vintage benchmark code are reusable; this review did not certify complete four-cycle paired revision panels.

**Test/rejection:** compare source/report predictive scores, population-path sensitivity and resulting counts with identical behavioral evidence. A current-revised-versus-old-vintage exercise diagnoses information sensitivity; it is not an information-matched algorithm gain. Reject an added term that merely absorbs genuine growth, duplicates existing temporal/source error or remains unidentified.

## Round two: cross-critique and boundaries

### Weighted survey alternatives are not inherently invalid

`src/survey_measurement/weighted_pattern_dirichlet_v1.py:1–9` already implements a normalized approximate same-sample weighted-release law, with a fixed common weighting operator applied once. `src/survey_measurement/ces_conditional_profile_dm_v2.py:1–7` conditions raw response counts on source/wave/state/stratum totals; it is a supplied-theta factor, not an empirical transport fitter. `src/survey_transport_v2/README.md` explicitly distinguishes respondent reconstruction from population transport and preserves negative empirical interval results.

The strongest duplication risk is adding survey-weight exponents to an existing selection-adjusted, already weighted observation law. That can correct the same selection mechanism twice, reinterpret weighted totals as raw people, or invent independent evidence from overlapping tables.

A weighted pseudo-posterior can nevertheless be a legitimate **separately versioned alternative target**. [Savitsky and Toth](https://arxiv.org/abs/1507.07050) provide conditions for such inference, involving inclusion-probability assumptions; the existence of released weights does not establish those conditions for an opt-in election survey. Normalize the resulting conditional law separately for every protected parent and preserve original parent masses. Explicitly declare the changed likelihood/target, weight scale, design/dependence assumptions and uncertainty treatment. Normalizing weights to sum to sample size alone proves none of that. Compare one defensible replacement against the current generative source law; do not silently stack corrections.

### Critique of the companion proposals

- **Economy and approval:** already present in legacy `src/prediction/fundamentals.py`, not the selected compact fundamentals. Porting one variable is an adaptation challenger, not filling an unknown layer. The legacy table is House-share-change, midterm-only evidence, not automatically a Senate A/B model. National approval and economic change may compete with shared cycle and turnout/choice effects; jointly refit and perform a deletion comparison rather than combine posterior estimates.
- **Availability correction for the 14-day target:** the companion report gives national October employment publication dates in early November. All four are later than the corresponding October 14-day forecast cutoffs. Thus the October observation is unavailable at h14 in **all four cycles**, even where it appears before Election Day. Use the latest release actually available at each horizon and its then-current vintage. Do not infer availability from observation month or state-series release dates.
- **Registration-to-vote transition:** not the same as adding a convenient covariate. FL registration features already exist in `src/assembly/build_voter_registration_features.py`; they do not establish a national transition law. Registration universes differ, inactive/stale records can matter, and same-day registration breaks an assumed strictly earlier gate. The selected CVAP-exposure route does not assert a known finite eligible denominator. A proposed product of registration and voting probabilities needs its own support/measurement model; do not relabel standardized exposure propensity or administrative registrations as legal eligibility. Test a calibrated measurement bridge or dated covariate before adding an unidentified pair of transitions.
- **Validated turnout:** useful possible evidence, not a new model by itself. A 2024-source voter-history file mentioning 2018/2020/2022 does not establish that those rows or matched universes were available for earlier forecasts. Match failures are not nonvoters; current respondents' historical records do not automatically represent each earlier survey cohort. Evaluate match/coverage error and avoid double-use with existing self-report factors or validation-adjusted weights.
- **Mobility/substate turnout:** `src/assembly/fetch_irs_migration.py` and `build_irs_migration_features.py` already implement migration intake/features. IRS tax-return movers are not all residents or voters. Introducing county paths into a statewide compact model requires dated boundaries, exposure allocation, household/nonfiler treatment and shared-event conservation; that is a geographic adaptation, not a new generic turnout layer.
- **Candidate experience/resources:** the compact model already has incumbency, person residuals and a ballot-role hierarchy. Legacy `src/prediction/candidate_effects.py` estimates poll-conditioned race deviations, and `candidate_ctov.py` carries past candidate overperformance; neither is a verified ex ante experience/resources model. Dated prior elected office may be a defensible narrow candidate feature. Money is heavily endogenous to viability and polling; filing-date joins and a causal interpretation must not be confused. Defer it until comprehensive candidate identities and historical reporting cutoffs are available.

## Iterative verification and decision rule

Follow `.agents/skills/verify-election-model/SKILL.md`: for a 2026 midterm target, A is 2018/2022 and B is 2020/2024, with older cycles as C stress evidence. Fit coefficients, representations, covariance and source bridges from earlier information within each replay. Target outcomes are scoring labels, not coefficient inputs. The historical cycles are reusable development evidence; no sealed-holdout requirement is introduced here.

Change one mechanism at a time. Keep the incumbent full configuration, protected-parent law/masses, observed-source identities, target, horizons, outcome support and score definitions bound. Refit affected downstream parameters jointly rather than splice independently fitted columns. Preserve common prediction rows and their dependence when comparing uncertainty. A small supplied-state oracle or compiler pass precedes but does not replace empirical evaluation.

Declare practical improvements and material-regression guards before each consequential run, reporting each A and B cycle and equal-cycle summaries. Investigate losses and evaluate useful concentration as well as coverage. Adoption does not require every metric or cycle to improve. Retaining the incumbent is a valid outcome. No empirical weakest-layer ordering follows from these research notes, the 141 functional checks, or standalone fits.

The immediate recommendation is to inventory the existing covariance-scale uncertainty gap and design one matched challenger. This is a research priority based on a documented assumption, not permission to start a fit, replace the incumbent or activate a public model. The House owner's separate four-arm F/FP historical 14-day benchmark remains outside this assignment.
