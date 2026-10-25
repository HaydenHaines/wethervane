# Luna research notes: mechanism review and round-two cross-critique

Date: 2026-10-24
Scope: read-only review of the Senate modular model and candidate missing mechanisms; no model or fit changes.

## Round 1: model coverage and potential challengers

The selected modular graph in `research/senate-stack-integration-20261020/modular-count-graph-astra-v1/components.py` already has state lean, prior national presidential results, president/midterm context, and incumbency in `Fundamentals`; `Environment` owns shared cycle and race residuals. There is no selected economic or presidential-approval predictor. `src/prediction/fundamentals.py` is a separate legacy midterm pipeline with approval, Q2 GDP growth, unemployment and CPI; `data/raw/fundamentals/midterm_history.csv` contains midterm rows through 2018 and 2022 only, and its target is House share change. It has no 2020 or 2024 rows, so it is not A/B evidence for those cycles.

`unemployment_oct` is the **national** seasonally adjusted UNRATE series: `scripts/fetch_fred_fundamentals.py` maps FRED `UNRATE` to that key. Do not infer its availability from state unemployment releases. National October Employment Situation releases occurred Nov. 2, 2018; Nov. 6, 2020; Nov. 4, 2022; and Nov. 1, 2024. Thus a pre-Election-Day forecast cutoff could use the October national observation for 2018/2022/2024, but not 2020. However, the internal historical table has no 2020/2024 rows, so the point is only relevant if those features are later ported to B-cycle inputs. Source: BLS archives, https://www.bls.gov/news.release/archives/empsit_11022018.htm, https://www.bls.gov/news.release/archives/empsit_11062020.htm, https://www.bls.gov/news.release/archives/empsit_11042022.htm, https://www.bls.gov/news.release/archives/empsit_11012024.htm.

Candidate-adjacent coverage: `src/forecasting/senate_data.py` joins official incumbent markers and strictly prior presidential results. `src/fundamentals_source_v1/README.md` describes prior-nomination/incumbency traits for Texas 2020 and its 2014 predecessor only. Candidate experience, retirement status and fundraising are not selected features; full A/B candidate-history support is not demonstrated. FEC cycle records exist but receipts are endogenous to viability and require reporting-date cutoffs.

Candidate research assessment:

- State economy: test one time-safe state measure under `Fundamentals`, e.g. real personal-income growth, rather than porting a bundle. Hummel & Rothschild (2014) report state personal-income growth as informative for Senate forecasts; Atkeson & Partin (1995) find no state-economy accountability effect for Senate elections in their 1986/1990 study. Mixed literature; not evidence of Ballotmancy improvement. Sources: https://doi.org/10.1016/j.electstud.2014.05.002 and https://www.cambridge.org/core/product/identifier/S0003055400095423/type/journal_article. Compare paired A/B scores and a deletion ablation, with release-vintage joins.
- Presidential approval: legacy midterm pipeline/data already include approval on A cycles, but selected law has prior presidential vote and party/midterm context. Consider only a separate, strongly shrunk coefficient if pre-election values are admitted for all cycles; national common shocks over four test cycles and high overlap make identification weak. Highton (2008) studies **senator** home-state approval, not presidential approval, and finds effects on retirement, candidate emergence, spending and outcomes: https://doi.org/10.3162/036298008784311019.
- Candidate experience/open-seat: test only verified prior elected office under candidate `Choice`, separate from incumbency; broad history/cutoff support is missing. Risks include overlap with prior nomination/incumbency and strategic entrant selection. Do not call the variable generic “quality.”
- Fundraising/resources: FEC archived data are available for all four cycles, https://www.fec.gov/data/browse-data/?cf=hs_elec&tab=candidates, but only cumulative amounts filed by a fixed pre-election cutoff should be tested. Endogeneity with candidate strength/viability is severe; defer until candidate ID and filing-vintage joins are complete.

No additional election-context term recommended: prior presidential share, party and midterm are already represented in selected `Fundamentals`, and shared cycle residuals are owned by `Environment`.

## Round 2: cross-critique

The selected compact graph is not entirely missing turnout or survey-source structure. `ProtectedPopulation` passes fixed exposure; population support includes 24 citizen cells and a noncitizen cell. Community composition is state-by-demographic with 12 profiles. Participation models demographic/education/community/state/cycle propensity, then event Gamma variation. Measurement includes source-specific turnout, source-by-state community discrepancy, choice and response/answer terms, plus separate official F1a report error. Precisely: selected `Measurement` has `source_community_state[qi, si]`, but `source_turnout[qi]` is source-level, not a source-by-state turnout interaction. F1a report error is not voter validation or registration.

Broader research exists but is limited/unselected:

- `src/forecasting/selection.py` is a conditional D/R poll-sample composition tilt, explicitly not identified response probabilities; it is not the selected measurement law.
- `src/forecasting/selection_pilot.py` is a Florida masked-origin experiment with 2016/2020 survey records filtered on self-reported turnout, not voter-file validated turnout.
- `src/senate_count_stack_v1/population.py` and the national-population research use statewide citizen demographic population; they do not integrate substate mobility/population paths in the compact model.

Recommended max-three test shortlist (none is a claim of improvement):

1. **Registration → voting transition** under `Participation`: compare `P(register | citizen exposure) × P(vote | registered)` against the current direct turnout propensity. Official EAC EAVS supports registration and participation at state/local scale across 2018/2020/2022/2024; Census CPS Voting Supplement provides demographic registration/vote estimates for all four cycles. EAC: https://www.eac.gov/research-and-data/eavs-retrospective. Census: https://www.census.gov/topics/public-sector/voting/technical-documentation/questionnaires.html. Complementarity test: keep the current source/community/state features fixed, then compare held-out EAVS counted-participant totals and CPS cell rates under direct vs decomposed turnout. Main risks: state registration definitions, stale records, shared EAVS reporting error, and state-vs-cell resolution mismatch.
2. **Survey inclusion and validated-turnout calibration** under `Measurement`, rather than adding another participant likelihood: compare existing source shifts to a model calibrated on earlier validated records. CCES 2024 validated-vote data include 2018, 2020, 2022 and 2024 historical vote-validation material: https://doi.org/10.7910/DVN/X11EP6. Complementarity test: use held-out survey/voter-file matches and EAVS, with record/source identity kept distinct; never train and score on the same validated turnout outcome. Unmatched/private records stay unresolved, not nonvoters. Main risk: inclusion bias, match error and current source shifts can explain the same residual.
3. **One state economic change** under `Fundamentals`, as described above: same-cohort paired refit and deletion test against prior presidential share plus party/midterm, using archived state releases for each target cycle. Main risk: state economic covariates overlap with state lean, shared cycle effects and turnout context; use one prespecified feature with shrinkage.

Defer **substate turnout/mobility** for this statewide Senate target: it would expand geography, population projection and boundary-crosswalk requirements, while existing components already provide demographic/community/state turnout heterogeneity. Defer presidential approval as a first new layer because it is already in a legacy A-only midterm model and substantially overlaps national election context. Defer candidate experience/resources pending comprehensive dated A/B history and fixed reporting cutoffs; incumbency is already present and fundraising is endogenous.

## Source/data references

- EAC EAVS time-series datasets and local/county views (2004–2024): https://www.eac.gov/research-and-data/eavs-retrospective
- Census CPS voting/registration supplements: https://www.census.gov/topics/public-sector/voting.html
- CCES validated-turnout dataset/guide: https://doi.org/10.7910/DVN/X11EP6
- BLS national October Employment Situation archives listed above.
- FEC cycle candidate data: https://www.fec.gov/data/browse-data/?cf=hs_elec&tab=candidates

Research does not establish value added by the selected model. New mechanisms should be challenged one at a time under the existing role, with affected parameters jointly refit, time-bounded source data, matched A/B targets and explicit retention of failed challengers.
