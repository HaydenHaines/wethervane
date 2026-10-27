# House probability wording in the Election Night Guide and Evaluation

Read-only source audit of SiteGround commit `317ff150f012871aec6c71ae680d7e76eec5b4db`. Source hashes are in `source-pins.json`. No UI, forecast, or publication files were changed and no model was run.

## Findings

### Guide: per-race outcomes

`ElectionNightForecast` preserves a race's full listed `outcomes` array. Each outcome has a `winProbability`, `kind` (`candidate` or `group`), and member roster (`lib/election-night-forecast.ts:22–29`; source `ExplorerOutcome` at `lib/explorer-types.ts:62–75`). The forecast projection requires a nonempty outcome roster and probabilities summing to one, but does not carry a separate structural flag (`lib/election-night-forecast.ts:72–103`). A one-outcome race therefore exposes enough information to infer that only one outcome is listed, but no UI copy states that fact. A multi-member group is correctly described as a group chance that is not allocated among members in the race details (`components/ElectionNightGuide.tsx:91–93`) and in the main section (`:251–262`). Call rows likewise identify group chance and distinguish multi-member winners (`:50–61`).

All the race probability surfaces use `probability()`: modal pick in featured facts, modal call rows/table, and each listed outcome in details (`components/ElectionNightGuide.tsx:55–61,77–80,90–93,252–254`). Exact `0` and `1` format as `0.0%` and `100.0%`; near endpoints `<0.1%` and `>99.9%` are distinct (`lib/presentation.ts:100,109–114`). None of these values gets an endpoint qualification. The only existing nearby zero caveat is in the optional pair-event explanation, and only when all tail-event metrics are present (`ElectionNightGuide.tsx:97–104`); it does not qualify winner/group endpoints or 100%, and it disappears in the missing-tail branch.

### Guide: chamber

The underlying scenario does carry chamber control probabilities: `ControlScenario` has Democratic, Republican, and unresolved probabilities (`lib/site-types.ts:47–52`), and `ExplorerScenario` contains that control object plus seat PMFs, means, and quantiles (`lib/explorer-types.ts:112–120`). The Guide does **not** render those control probability fields. It displays a deterministic point call assembled from each race's modal outcome and retained seats, with an explicit non-guarantee/non-most-common explanation (`components/ElectionNightGuide.tsx:263–270`; `lib/election-night-calls.ts:3–19`). Its chamber uncertainty card shows expected seat totals and 5th–95th seat quantiles (`:359–366`), not control probabilities. The printout says “The model’s control chances and shared uncertainty are shown below” (`:269`), but only the shared seat summaries appear below; this is inaccurate copy. Avoid adding 0/1 control-probability wording until those probabilities are actually rendered.

### Evaluation table

The locked result table shows only each race's modal pick name, `kind`, and probability (`components/ElectionNightEvaluation.tsx:75–78`). Its projection schema has no outcome roster, support count, or structural flag (`lib/election-night-evaluation-view.ts:27–37`). Thus the component cannot tell a one-listed-group `100%` from a multi-outcome model endpoint using its current view. Group picks are labeled “(group),” but the chance is not distinguished from a member chance beyond that label. Exact and near endpoints use the same formatter above; no boundary caveat appears.

## Recommended changes and tests

- In the Guide, distinguish single-listed-outcome support from empirical endpoint estimates. For multiple listed outcomes, exact 0/100 wording should say those are forecast endpoints and do not make an outcome impossible or guaranteed. Do not infer event counts, independent draws, or an effective sample size from displayed probabilities.
- For the locked Evaluation view, either carry authenticated support count/structural semantics in its projection, or use generic endpoint wording that does not falsely call a structural assignment a simulation result. Keep the group-versus-person wording intact.
- Correct the Guide sentence at `ElectionNightGuide.tsx:269` or render the control probabilities it promises. If rendered, test D, R, and unresolved endpoints separately from deterministic point-call seats.
- Add component-render tests with invented House fixtures: one structural single-group race; multiple groups with exact 0 and 1; near-boundary values (for example `.0009` and `.9991`); candidate and group outcomes including a multi-member Other group; and evaluation rows for structural and modeled support. Assert both visible text and accessible labels. Add a chamber scenario with each control outcome at 0/1 and confirm actual probabilities, if exposed, remain distinct from seat quantiles and point call.

## Existing test coverage

Forecast tests validate shape, probability mass, ties, and preserve House group probabilities; evaluation tests exercise scoring with endpoints and structural one-outcome data. These are data/algorithm tests, not React rendering tests for `ElectionNightGuide` or `ElectionNightEvaluation`. Search found no test asserting endpoint or support wording in either component. The test suite does include unrelated accessible/guide browser coverage for navigation, but no exact endpoint fixture for these displays.
