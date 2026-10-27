# Independent review of House boundary-display preparation

## Scope

Reviewed isolated worktree `/home/hayden/projects/ballotmancy-house-boundary-display-20261026`, based at `317ff150f012871aec6c71ae680d7e76eec5b4db`, with six modified source/test files. The worktree also has an untracked `node_modules/`; I did not inspect or modify it. I did not change the site, run forecasts, or open prediction/draw arrays. Exact base/diff/source hashes are in `source-pins.json`.

## Verdict

The patch materially improves the current House detail, map accessibility label, chamber summary and live serpentine ribbon: modeled supported endpoint probabilities receive a non-certainty statement, and an omitted group can be labelled “not in listed outcomes” rather than being shown as a modelled zero. No false claims about weighted draws or effective sample size were added; the patch does not invent numerator/denominator counts.

Two review gaps remain:

1. The shared `HouseGroupRibbon` supplies the new endpoint/support wording only when `supportedBlocs` is passed. The live adapter now passes it, but historical `HouseGroupResults` passes `result.races` directly (`components/HouseGroupResults.tsx:100`); the `HouseGroupRace` type contains `groups`, `probabilities`, and `structural` but no `supportedBlocs` (`lib/house-group-result.ts:17–33`). So for historical ribbon rows `chanceText` omits both absent-group status and endpoint caveat even though the detail pane does have them. Its tile `title`/ARIA description is built from `chanceText` (`components/HouseGroupRibbon.tsx:66–70`), so this is also an accessibility gap. The adapter can derive support from `race.groups.map(g => g.id)`.
2. The new ribbon helper accepts contradictory structural claims. `layoutGroupRibbon()` validates support/probability coherence, then independently sets `probabilitySource` from `race.structural` (`lib/group-ribbon-layout.ts:81–92`). A reproducible fixture with two supported groups, `D=.5,R=.5,O=0`, and `structural:true` is accepted as `probabilitySource:"structural"`. Existing live/historical producers appear to set the flag consistently (live derives it from one outcome; historical parser enforces structural iff one group), so this is not evidence of a current rendered misclassification from those trusted paths. It is a gap in the helper’s fail-closed contract and negative tests; when support is present, require structural iff support length is one (or derive structural there).

There is also wording refinement: a structurally single-group live row is currently given the same `chanceText` suffix as a finite simulated endpoint (`"100.0% model estimate; not certain"`) and then a separate `"fixed under the listed group support"` statement. The appended support statement helps, but the first suffix calls a structural assignment a “model estimate.” Prefer distinct text such as “all modeled mass is assigned to the only listed group; this does not guarantee the actual winner.”

## Source assessment

- The live race details correctly separate three cases. If there is one outcome group, the text says it is the only group included and is not a guarantee (`HouseLiveResults.tsx:90`). With multiple groups and any exact 0/1 value, the text says a finite simulation endpoint does not imply impossibility/guarantee (`:91`). A group outcome is not allocated among its members (`:101`).
- The current live ribbon passes group support from `race.outcomes` and marks one-outcome races structural (`HouseLiveResults.tsx:157–166`). Its synthetic D/R/O zero fill is now paired with support metadata; `group-ribbon-layout.ts:82–86` rejects nonzero probability for an absent supported bloc. `HouseGroupRibbon.tsx:10–16,66–70` puts support/endpoint wording in the focused text, `title`, and `aria-label`.
- The live map option adds boundary wording to the map label/title for a leading probability of exactly 0/1 (`DistrictMap.tsx:258–260`). For a structural one-outcome probability of 1, however, it says “model estimate; not certain” rather than explaining single listed support. The detail panel has the more precise wording when the user selects that district.
- Chamber D/R/unresolved probabilities get an endpoint disclaimer if any is 0/1 (`HouseLiveResults.tsx:139–140`). It does not supply draw counts. This avoids unsupported count precision claims; existing numerical-review text gives the global joint-draw count and aggregate chamber MCSE elsewhere.
- No new statement assumes unweighted independent trials or uses N/effective N to form an interval. “Finite simulations” is conservative and remains valid when output probabilities may be weighted. Do not infer 0/N by multiplying serialized probabilities by the global draw count; the code currently does not do that.
- Existing multi-member group wording continues to state that a group’s chance is not split among its members (`HouseLiveResults.tsx:101`) and member identities/affiliations are listed separately (`:102–108`). Existing rendering tests cover mixed Other membership, but the new endpoint tests do not combine multi-member group semantics with a structural/0/1 fixture.

## Remaining endpoint surfaces not covered

The preparation changes `HouseLiveResults`, `DistrictMap`, and the shared group ribbon, but not every House-facing probability surface:

- `components/ElectionNightGuide.tsx:55–63,74–102,253` displays the modal House call, probability details and call table with `probability()` but has no 0/1 boundary or structural-support wording. `ChamberExplorer.tsx` opens this guide for either chamber (`:6–7,113`).
- `components/ElectionNightEvaluation.tsx:77` displays modal-pick probabilities in the House evaluation table without the new boundary wording.
- Historical group detail (`HouseGroupRaceDetails.tsx:54–58,78–82`) already distinguishes structural one-group support from modeled joint-draw 0/1 endpoints and avoids certainty language. Its ribbon accessibility descriptions still need the `supportedBlocs` propagation described above.
- Other historical/legacy House probability views such as `HouseExplorer` have their own endpoint prose; this patch does not extend the shared policy across all probability displays.

## Reproduction

Ran the focused suite in the isolated worktree:

```sh
node --experimental-strip-types --test tests/group-ribbon-layout.test.mjs tests/house-live-results.test.mjs tests/house-group-ribbon-preview.test.mjs
```

Result: 22 passed, 0 failed, 5 skipped (browser/optional preview tests). Node emitted the existing typeless-module warning and React emitted its `<title>` children warning. The live boundary tests also check rounded near-boundary estimates (`.9999`/`.0001`) separately from exact endpoints. A separate invented-object probe is preserved as `structural-flag-probe.mjs`; it confirms a two-supported-group row marked structural is currently accepted. Its result has no site/district data.

## Recommended follow-up tests

- Add malformed fixture cases: multi-support + `structural:true`; one support + `structural:false`; duplicate/unknown support; ensure invalid declarations fail closed or are derived from authenticated source support.
- Server-render historical group ribbon fixtures for a structural one-group case and a modeled 0/1 case; assert visible focused text, tile `title`, and `aria-label` distinguish unavailable group, finite simulation endpoint, and structural single-group assignment.
- Add current live tests for structural O-only and D/R multi-member groups, ensuring the endpoint message never allocates group chance to members and map/ribbon language matches detail.
- Exercise House election-night call, details, and evaluation table at 0/1 and one-group support; verify the disclosed wording is adjacent and accessible there too.
- Keep probability values as the existing estimates; no per-draw counts or Monte Carlo intervals should be added unless the authenticated draw decoder supplies exact counts and a suitable dependence/weight interpretation.
