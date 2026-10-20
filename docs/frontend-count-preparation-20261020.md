# Preparing count and participation views

This preparation follows the standing frontend integration task and the Senate owner's `docs/development/frontend-count-stack-preview-20261020.md`, SHA256 `dbf74bbe8bdc1b789ad7025d8f8483f1c6e92d11c5c91ff56529f8f981ea7fe7`. That handoff is a proposal, not an available forecast or public API. The scientific implementation `senate_compact_count_v1` and target `population_cut_v1` are not options in the current reviewed frontend catalog.

The existing 48 aggregate forecasts, two scoped 2018 studies, best-guess default and basic electoral-history preset remain separate. Their conditional shares cannot supply population, participation, all-candidate votes or missing legal outcomes for a count forecast.

## Presentation model

The prototype is a frontend view model, with explicitly invented fixtures. It is not a proposed backend request schema. A future source-bound adapter must translate the final reviewed public packet into presentation data and verify its identities before any public activation.

The view follows three questions: how many people are represented; how many participate in one physical election event; how those participants' ballots are represented in each applicable contest. Each quantity carries its own units, denominator, uncertainty intervals and missing-data reason.

| Quantity | Display and boundary |
| --- | --- |
| Adult residents | Includes noncitizen adults; distinct from citizen exposure and voters |
| Citizen-adult exposure | The uncertain population used by the working model, not a verified count of legal eligibility or a hard participation cap |
| Predicted event participants | People participating in one event, from the same joint count result; not an administrative observation |
| Administrative participation report | A separately dated reported count, potentially noisy; never relabeled as the model's participant forecast |
| Named candidate votes | Every admitted named candidate remains distinct, including minor-party and declared write-in candidates |
| Candidate shares | The denominator is explicit. All-candidate votes and Democratic–Republican votes are different quantities |
| Blank, invalid, nonresponse and lawful nonperson outcomes | Separate admitted categories; pooled survey answers or write-in totals never become an invented candidate named Other |
| No applicable contest | Distinct from a contest with an undervote or no valid candidate votes; participation can exist where a Senate contest does not |
| Unresolved race outcome | Tied maximum and no valid candidate votes remain visible. The frontend never breaks ties or renormalizes the remaining winner chances |
| Chamber organizing scenario | A 50–50 Senate and the vice president's role are distinct from tied candidate counts |

Count means and interval endpoints may be fractional even when underlying simulated counts are integers. Turnout divided by citizen-adult exposure may exceed one in the proposed incidence model; the UI must not clamp it to 100%. The mean of a ratio is not generally the ratio of means. Marginal interval endpoints cannot be added to prove same-event accounting.

## Scope and evidence

Always identify the fixture and its partial scope before displaying numbers. A Texas pilot does not establish national support. Actual coverage must separately count population/participation targets, physical events, contest settings and retained seats. The planned national goals of 51 state/DC targets, 35 current contests and 100 seats including 65 retained seats are distinct checks, not assumptions inferred from a roster.

Numerical checks, source admission, rule admission, empirical evidence, publication availability and execution lifecycle remain separate. A pending calculation has no queued/running claim without an actual scheduler receipt. A failed or missing population parent suppresses incomplete predictive summaries; its probability mass must not silently disappear. An administrative report can remain visible with its original observed label only when its own dated source is independently admitted and qualified. Holding that source hides the report without hiding otherwise qualified predictions.

Coverage counts use the declared expected scope. Unexpected event or contest identities invalidate predictive coverage instead of inflating a displayed count. Blank/invalid ballot counts, response counts and people keep their distinct units.

The prototype consumes explicitly weighted summaries tied to an immutable joint-artifact reference. It does not read research NPZ files, average draw rows or infer chamber outcomes from candidate marginal probabilities. Parent-mass and same-event reconciliation declarations are preparation for backend evidence, not proofs reconstructed from the summaries.

## Planned contribution explanations

Demographics, education, education interactions and community are descriptive rows only. They are not invented catalog keys or executable switches. The first proposed population support has 24 age × racial/ethnic × education cells and no sex/gender dimension; the 12 proposed community profiles use religion × attendance and are not an exhaustive definition of community. Effects can contribute to both turnout and candidate choice. Turning a contribution off changes the jointly fitted law while retaining the population cells and source universe.

## Admission requirements for a real integration

Before a public count view or request is enabled, obtain the final exported catalog and option schemas; source-bound request/resolution/result contracts; immutable manifests and weighted summaries; a qualified real request/result example; actual event, contest, candidate and rule axes; source and legal qualification; complete parent-mass and same-event accounting evidence; actual execution/job behavior; and replay checks. Record the packet version and hashes. Unsupported election rules, missing parents and incomplete chamber accounting remain explicit limitations.

Private preparation fixtures must cover partial Texas support, missing national support, failed parent mass, unsupported election rules, a tied candidate maximum, no valid votes, absent contests, a separate chamber organizing tie, unknown quantities, and a participation/exposure ratio above one. No fixture is imported by the production page or exposed through an API.
