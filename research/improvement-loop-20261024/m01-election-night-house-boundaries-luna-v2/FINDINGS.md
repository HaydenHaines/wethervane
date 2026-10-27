# Independent review: Election Night endpoint wording follow-up

Reviewed the isolated worktree `/home/hayden/projects/ballotmancy-house-boundary-display-20261026`, base commit `27da59b5e66d774831f4024cdfd2050019af191b`. The tracked diff SHA256 is `335fbcd31537c7add36fa4b96fd388936be2f070211c58e57e7be4a5cdb46f14`; the new helper and SSR test are separately pinned in `source-pins.json`.

## Assessment

The endpoint presentation follow-up addresses the previous Guide/Evaluation gaps within its stated scope.

- `electionNightProbability(value, chamber, listedOutcomes?)` preserves the original `probability()` output and appends wording only for House endpoints. If exactly one outcome is listed and its probability is 1, it says the value is assigned to the only listed outcome and does not guarantee the actual result. With multiple outcomes, 0 says “model estimate; not impossible” and 1 says “model estimate; not certain.” It does not infer counts, effective sample size, or a sampling interval.
- The helper is applied to Guide call rows, printable call table, modal-pick facts, outcome details, conditional pair-event values, and both the headline and all-race views (`components/ElectionNightGuide.tsx`). Candidate/group labels and the multi-member group note remain intact.
- Pair-event probabilities intentionally omit winner-roster count and receive the generic model-endpoint wording, so a one-listed winner outcome does not incorrectly determine the support semantics of a separate pair event.
- The locked evaluation projection has only modal-pick name/kind/probability, not the full roster or structural flag. Its table calls the helper without a listed-outcome count, so it uses the generic endpoint wording rather than inventing single-group support. This is an appropriate conservative response to unknown support under the current schema.
- The inaccurate statement that chamber control chances appear below is corrected. The Guide now says shared seat ranges are shown. The actual call remains a deterministic construction from modal race outcomes and retained seats; no control probabilities are newly displayed.

No material defect found in the changed scope. One possible future refinement: a locked evaluation row with unknown roster at 100% receives “model estimate; not certain,” which is deliberately generic. If the evaluation contract later preserves structural/support metadata, that surface could distinguish the single-listed case too.

## Reproduced checks

Ran:

```sh
node --experimental-strip-types --test tests/election-night-probability-display.test.mjs
```

All four invented SSR tests passed. They exercise: structural one-group support and non-allocation for a multi-member group in both call row and printable row; multiple-outcome exact 0/1; `.9999`/`.0001` near endpoints; no House qualifier for Senate; independent conditional-pair endpoints; and locked House evaluation with unknown support (generic wording, no structural assertion). Test module compiles local TS components into a temporary directory and removes it in an `after` hook.

## Limits

This is a source and invented-fixture review. It is not a publication/deployment authorization, does not validate forecast accuracy, and does not add control probability display. The locked evaluation schema still cannot distinguish one listed outcome from multi-outcome support, so its endpoint language remains generic by design.
