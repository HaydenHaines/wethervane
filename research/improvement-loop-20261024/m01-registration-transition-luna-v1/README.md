# Registration-to-participation transition: source and identification review

Date: 2026-10-26

## Finding

A registration hurdle is a plausible model hypothesis, but the currently admitted evidence does not support a directly observed population law of the form `P(registered) × P(voted | registered)`. No joint refit or predictive comparison was run. Existing coverage already includes demographic/community/state turnout, survey turnout masks, and state event totals; “no separate registration stage” is a structural gap, not proof the model is weak.

CPS Voting Supplement microdata are available for election years including 2018–2024, with state identifiers and person replicate weights. The 2024 technical documentation establishes the key instrument limitation: PES1 is in the edited survey universe of adults age 18+ with qualifying citizenship codes, subject to CPS interview/nonresponse and editing; it is not a census of every legally eligible voter. A “yes” response skips the registration question, while PES2 is asked for PES1=No, don’t know, refused, or no response (-9). Consequently, the data do not directly observe registration status among reported voters. A joint table can be constructed only by adding an assumption or outside source about voter registration status. That is consequential in jurisdictions allowing same-day registration, and in North Dakota where voter registration does not exist. CPS answers are self/proxy reports, and the voting question is about the election generally, not a specific federal contest.

EAVS supplies administrative registration and counted-ballot totals, but not a person-linked R×V table. Its registration counts (including active/inactive categories) and participation totals have different definitions and denominators. EAC documentation notes state reporting variation, North Dakota's no-registration regime, same-day registration, and changes to question definitions. CVAP is a useful scale denominator, not an exact count of legally eligible voters. EAVS cannot by itself identify which registrants voted.

## Existing graph and admitted data

The selected `Participation` provider maps a turnout logit to `q = sigmoid(turnout_logit)` and multiplies exposure, community profile, and q to form event intensity. `Measurement` computes survey turnout as `h = sigmoid(turnout_logit + source_turnout[source])`; it separately attaches one Negative Binomial factor to each participation report. A new `q = r × v` implementation cannot leave the existing turnout-logit interface ambiguously representing either stage: it must expose a compatible marginal logit for `q`, or version/update the measurement provider and all affected contracts.

The strict current survey row schema is `record_id, cell_id, community_mask, turnout_mask, answer_mask`; it has no registration outcome, design weight, or replicate-weight covariance. Thus CPS is not drop-in data. The current prepared TX problem (`prepared-v4/problem.json`, purpose `actual_current`, cutoff 2026-10-20) has one state (TX), events 2020/2024/2026, two EAVS reports (2020 and 2024), 33 CES survey blocks (historical 2020), and no CPS registration records. Both existing EAVS report factors already inform event participation. A new likelihood using the same EAVS totals must replace/deduplicate the old factor; do not count one report twice.

## What can be tested, after the gate

A defensible next experiment is a feasibility/identification gate first, then a matched full-stack replay only if it passes:

1. Pin a dated CPS microdata and person replicate-weight vintage for each historical election; specify the citizen/age universe, state and demographic crosswalk, treatment of proxy/nonresponse/not-in-universe answers, and the survey-weighted likelihood or summary-statistic covariance. Preserve CPS as reported behavior, not administrative ground truth.
2. Do not silently impute `registered=1` for all voters. Either model the skip mechanism and a regime-specific voter-registration-status rule (including same-day registration), or mark the joint stage unidentified in unsupported jurisdictions. Use EAVS jurisdiction-level registration/SDR items only as separate aggregate observations, with definition/version and release date pinned. Explicitly handle ND.
3. Reconcile target concepts: CPS reports whether a respondent voted in the election; EAVS F1a is counted ballots, and A1a/A1b represent administrative registration categories. State the population/exposure and errors for every factor. CVAP is not an exact eligibility cap.
4. On a frozen baseline roster, compare the incumbent participation/report measurement against a registration-stage replacement with the necessary compatible Measurement update. Keep all other contracts, cutoffs, and observations matched, and jointly refit every parameter in the full graph. A same-source report is owned once: replace the existing EAVS report likelihood if an alternative EAVS likelihood uses that same evidence.
5. Evaluate per-cycle joint candidate-count/event-total proper scores and calibration. Separate 2018/2022 midterms from 2020/2024 presidential elections. Target-cycle post-election CPS/EAVS may score a forecast or train later cycles only after the actual publication date. The current 2020→2024 EAVS carry-forward comparison is exposed development evidence, not independent validation.

This gate could establish whether new registration information identifies anything beyond total turnout. It would not by itself establish forecast improvement. Current evidence supports a barrier report and an adapter proposal, not a completed scientific experiment.

## Primary sources and local evidence

- Census CPS Voting Supplement 2024 technical documentation, especially Overview and record-layout/questionnaire pages: https://www2.census.gov/programs-surveys/cps/techdocs/cpsnov24.pdf
- Census Voting and Registration datasets (2024, 2022, 2020, 2018 files and replicate weights): https://www.census.gov/data/datasets/time-series/demo/cps/cps-supp_cps-repwgt/cps-voting.html
- Census November CPS voting API: https://api.census.gov/data/2024/cps/voting/nov.html
- EAC Studies and Reports; 2024 EAVS v1 release June 30, 2025 and v2 release Feb 12, 2026; 2022 release Jun 29, 2023: https://www.eac.gov/research-and-data/studies-and-reports
- EAC 2024 EAVS Comprehensive Report (registration and turnout definitions, jurisdiction variation, North Dakota, denominator discussion, revisions): https://www.eac.gov/sites/default/files/2025-06/2024_EAVS_Report_508c.pdf
- EAC 2024 EAVS release (June 30, 2025): https://www.eac.gov/news/2025/06/30/us-election-assistance-commission-releases-2024-election-administration-and-voting
- Sol's independent interface/identification challenge, with exact-rational synthetic counterexample: `../participation-challenge-sol-v1/README.md` (hash in evidence.json). This is an algebraic illustration, not empirical evidence.

## Scope and limits

This review read source/code/contracts and public official documentation; it did not download CPS records, build a new adapter, fit a model, run an owner job, or alter model/data inputs. The 2024 documentation is evidence about 2024 instrument design; older questionnaire versions must be checked before assuming identical paths across every historical cycle. The interface counterexample shows that marginal turnout alone does not identify a two-stage factorization; it is not proof that a richer CPS+EAVS likelihood could never identify one.
