# Frontend readiness for a future national joint count output

**Read-only audit, 2026-10-25.** No frontend edits, API activation, output import, or scientific process interaction. The v23 checkpoint describes active research and constructor progress; it does not provide an actual posterior or qualified public result.

## What is now concretely supported in research artifacts

- Checkpoint v23 records an actual, source-bound constructor for 20 of 35 Senate settings, attached to eight population parents. It is explicitly incomplete: `full_national_assembly_complete:false`, `posterior_issued:false`, and the full run was still in warmup at checkpoint issuance. No actual posterior or forecast was complete then. The checkpoint says to preserve run session 72164; this audit did not inspect or control it.
- The mixed-rule artifact work can serialize and reduce **supplied-state** rows containing a shared physical event, finite candidate counts, ranked raw-mark signatures, runoff state and distinct open-person allocations. Its current artifact scope is `supplied_states_not_posterior`; the README says the example is invented and there is no production registration. It demonstrates row structure and invariants, not forecast values.
- The target-aware count binding has a typed prediction protocol and verifier. It requires actual initialized model shapes, same-row prediction deterministics, exact parent masses, axes, target and source identities, plus registered execution evidence. It explicitly rejects chamber reduction for the current 20-setting scope (and an incomplete 23-setting successor). The registry has no production registration. Thus this is an important future adapter boundary, not a qualified public payload.
- The owner’s prototypes preserve multiway named support, person-level open outcomes, unresolved identity/eligibility, conditional runoff event semantics and round-level raw ranked counts. The producer README explicitly says pooled write-in totals are not a person, no rule tabulation is performed by the frontend, and supplied rows are not posterior draws.

## What the site already handles safely

`docs/frontend-count-preparation-20261020.md`, `lib/count-view.ts` and the private `CountForecastPreview.tsx` already establish useful invariants: independent population/event/contest quantities; explicit denominators; named candidates distinct from noncandidate categories; unavailable distinct from zero; missing parent mass suppresses predictive summaries; immutable joint-summary binding; explicit required event/contest axes; shared-event participant reconciliation; unresolved ties/no-valid-vote probabilities; and chamber control gated on a complete fixed-seat ledger and named organizing scenario.

The private `ElectionOutcomePreview` also separates final winner, runoff advancement and conditional later-event results. It requires one supplied ranked example row with all rounds and inactive ballots conserved. Both previews remain presentation fixtures: they are not imported by production pages and are not raw research artifact parsers.

## Specific contract gaps for multiway shares, margins and ranked rounds

The current `CountView` cannot yet represent the new output without lossy reinterpretation:

1. `CountContestView.rule.family` supports only `single_mark_plurality | unsupported`. It has no admitted multiway/ranked family, ranked-round axis, continuing-ballot denominator, round-specific inactive/exhausted category, or rule-supplied margin. Do not widen this union to “ranked” until the producer’s final summary contract names exactly what each round statistic means.
2. Candidate share currently means **all-candidate count divided by a complete `countable_candidate_total`**. That is not automatically a D/R share, first-choice share, share of active ballots after an elimination, or final-round share. A ranked presentation needs a supplied denominator identity and interpretation for every displayed row/round (for example, all countable first choices versus active continuing ballots in round N). The frontend must not derive a margin by subtracting marginal means or shares from separate summaries.
3. Contest outcome probabilities currently support candidate wins, tied maximum and no valid candidate votes. The future count output also needs person-scoped outcomes, unresolved identity/eligibility/procedure cases, and rule-specific round/final outcomes. Unknown or pooled write-in mass must stay a distinct nonperson category; the open-person model’s residual tail must not be rendered as a named “Other” candidate.
4. `rankedExample` is one same-row example sequence, not a weighted round-by-round forecast distribution. The new contract will need producer-supplied round summaries bound to the same joint artifact and rule scenario, with per-round candidate counts/shares, inactive/exhausted ballots and denominator. Keep the example-row disclosure separate from any mean/interval progression.
5. Partial coverage must remain scoped to exact settings, contests and events. The existing model already blocks full chamber control absent a complete seat ledger; preserve that boundary. A 20/35 conditional constructor or supplied-state artifact cannot be labeled a national Senate result. Missing/failed population-parent mass must remain missing and cannot be renormalized away. If a future reviewed output supports qualified local rows while national coverage is incomplete, display only those rows with explicit coverage and no chamber-level aggregate.

## Safe preparation, once the public projection is specified

Keep backend bytes and scientific artifacts outside the component layer. Build one versioned adapter from a reviewed exported projection into a view model that:

- binds target, cutoff, source/rule admission, joint artifact, weighted summary, exact event/contest/person axes and parent masses;
- carries all-candidate, D/R and per-round statistics as distinct quantities, each with a supplied denominator key, units, intervals and unavailable reason;
- carries winner/final outcomes, advancement, unresolved reasons and conditional-event participation as separate summaries from the same joint result;
- includes complete expected/supplied coverage ledgers and a clear local-versus-national scope declaration;
- passes exact same-row accounting and parent-mass/coverage checks before exposing any number.

Until that output projection is frozen and qualified, safe local work is limited to invented-fixture tests for arbitrary candidate counts, D/R versus all-candidate denominators, per-round active/inactive counts, no-valid-vote/tie/unidentified outcomes, missing settings and parent mass. Do not build live controls or adapt the packed research artifact directly. Reuse the existing fixture surface rather than duplicating the House catalog or adding another result page.

## Binding source files and hashes

Senate checkout `/home/hayden/projects/wethervane-redesign-20261006/integration`:

| File | SHA-256 |
|---|---|
| `research/senate-stack-integration-20261020/checkpoint-root-v23.json` | `3a8679aabf020af81115f0b037b391bcc76ba1274e9e8d285a274c28272ca2fe` |
| `research/senate-stack-integration-20261020/national-problem-assembly-astra-v1/README.md` | `168aed7d740dfb428ab24ee61f8101ffa63a3dca80c3dcc3bfe751a8248508dd` |
| `research/senate-stack-integration-20261020/national-problem-assembly-astra-v1/receipt-v1.json` | `5cc92ecb54328a28eb40d51837279660050743b2f3e76452d6fe0593177c2d39` |
| `research/senate-stack-integration-20261020/national-mixed-rule-integration-astra-v1/README.md` | `c85c1fcb851478cf6a9dacad002b3a5d7904baea177ef771d93cb62e154917c5` |
| `research/senate-stack-integration-20261020/national-mixed-rule-integration-astra-v1/receipt-v1.json` | `a476a4a77dc1abb568b9e5924395fa8e84217592f3e74c266c38843b1606cbd5` |
| `research/senate-stack-integration-20261020/national-count-prediction-binding-astra-v1/README.md` | `c5998218e5171c099bd739394f1cd2ded72a11b9f095bf18b3a8d6367d1fccc0` |
| `research/senate-stack-integration-20261020/national-count-prediction-binding-astra-v1/producer-protocol-v2.json` | `3cf20cdfc824ed981ece779d19d561187e9151866fa230090aeb6b974799e377` |
| `research/senate-stack-integration-20261020/national-count-prediction-binding-astra-v1/receipt-v1.json` | `cbe1db7468de36c464498d20fcef0c440a5fbdecef4a5bea820407ed61c9024d` |

Frontend checkout `/home/hayden/projects/ballotmancy-siteground`:

| File | SHA-256 |
|---|---|
| `docs/frontend-count-preparation-20261020.md` | `44fb5713f9fb32e30d107274ed7fe8f51fdc4404760a5b7ef5f94fc1dfa1212a` |
| `lib/count-view.ts` | `24944e496eff888929d014f2b9aac0a0edecce15f5e0df3d7eee425c03960d04` |
| `lib/election-outcome-view.ts` | `1b4926920f4b4697b75274819555129edd52eb35c2d6a384e599de2868fe3304` |
| `components/CountForecastPreview.tsx` | `b06c66ae0766498c38c0aa349ef393b7411161c32f6037b114896915b9b6e941` |
| `components/ElectionOutcomePreview.tsx` | `ad847f3d9de5b4870c600fdaf3e6ded16fd0b383034d499cb0b5e8f92031a813` |

No files were changed in either checkout.
