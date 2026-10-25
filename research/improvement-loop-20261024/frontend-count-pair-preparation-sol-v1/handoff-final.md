# Final private candidate-pair comparison handoff

This final handoff supersedes the earlier handoff/manifest/patch in this folder. Earlier versions are preserved as development history. The only changes since that preliminary handoff are the requested tiny-positive/near-one share-mass display, nonwrapping loss units, focused rendered tests and corresponding documentation. No validator/data/scientific semantics changed.

Worktree: `/home/hayden/projects/ballotmancy-count-comparison-prep-20261025`. Branch: `research/count-comparison-prep-20261025`. Exact base: `801dda8912f575b01af034702455b055f956874e`. Six new files only; no commit, push or deployment. No applicable AGENTS.md exists in this checkout or its ancestors. The local node_modules symlink is a runtime convenience, excluded from the patch.

The separate invented-fixture validator, component and fixtures preserve exact declared source/context/endpoint/date/pair-denominator binding, explicit undefined/unavailable/defined states and scored/expected coverage. The view cannot turn a pair lead into a legal winner or chamber result. It distinguishes prediction ranges from errors, calculation precision and permission for public use. The old one-page app, CSS, CountView/ElectionOutcomePreview, APIs and controls are byte-identical to the deployed base. No production module imports the new preview.

The copy explains CRPS, weighted interval score, percentage-point units, lower-is-better/zero-is-perfect and mean/median. It explains why a zero pair total prevents calculating that share and why remaining outcomes do not receive extra weight. Technical identity records are collapsed, model headings use h4 below pair h3, and scores use h5. Tiny positive undefined probabilities display “less than 0.1%”; values just below one display “more than 99.9%”; exact zero/one remain distinct. Every numeric error stays beside its points unit on one line.

## Final validation

- 62 passed, 0 failed, 1 skipped across the new and existing private preview suites; all 21 new tests ran. The skip is the optional legacy ElectionOutcome HTML export.
- TypeScript and the production build pass. No new page or API appears in the route list.
- Chromium desktop 1440×1000 and mobile 390×844: no horizontal overflow or page errors. Text-fragment checks confirm every score value/unit occupies one line. Final HTML/screenshots/metrics use the v3 suffix and were visually inspected.
- The dedicated rendered regression tests tiny positive mass (1e-100 and .0001), a value just below one and exact one.
- Source patch reverse-application check passes, and existing app/native-preview bytes match the exact base.
- Earlier test/build/render evidence remains preserved. The first value-wrap measurement incorrectly used stretched flex-item height; its corrected text-fragment check is documented in visual-checker-correction.json. No UI code changed to accommodate that checker.

## Integration limits

This is a private presentation model, not an owner-approved browser contract or real artifact parser. Every number/person/source-hash declaration is invented; matching declarations do not authenticate actual evidence. There is no public route, live-data loader, owner change, fit, real scoring, scientific process control, commit, push or deployment. The prototype intentionally supports one ordinary candidate-mark pair per race/contest, one cycle and a declared H14 cutoff.

Questions for owner/frontend review remain in docs/frontend-candidate-pair-preparation-20261025.md: the authenticated summary/label/source/estimand contract, canonical identity and alias joins, parent/row weights, actual numerical/forecast evidence, truth-independent F/FP summaries, per-functional precision and any ranked/winner/national output. A real adapter must be reviewed separately; no existing native validator should be widened to admit this fixture.

## Final source hashes

| File | SHA-256 |
|---|---|
| `components/CandidatePairComparisonPreview.tsx` | `8eb432b2c7ff32f7eaaddbba10d4e34258f28d52cde589019b4fcd100fb98ba8` |
| `lib/candidate-pair-comparison-view.ts` | `4df0217d985637ce4dbfe546bb01835ebc6f647aa4ee6db54a83bd9bb03929fe` |
| `tests/fixtures/candidate-pair-comparison.mjs` | `ba5fd6b1a1cb3bf05ae2293022d9f8234646fde07c2dfdd01ff36d3127847eae` |
| `tests/candidate-pair-comparison-view.test.mjs` | `2587ea89eb7489f23894a10774147d7598e2199d2cfd8ba35b0dc62d5ab28f39` |
| `tests/candidate-pair-comparison-preview.test.mjs` | `c898805e5023a5e0c92221b51f647682442e4f897a3dc921d7aa31187d673e59` |
| `docs/frontend-candidate-pair-preparation-20261025.md` | `2b0bf04f0e08d6183aa9fb2837b849129457315f8105c7f2d278fa9f57a5e3b2` |

Final patch SHA-256: `e7b049ee610a6a720dba75c7148b989d8a1d89ff28778d2e52804ce77f5783c2`.
