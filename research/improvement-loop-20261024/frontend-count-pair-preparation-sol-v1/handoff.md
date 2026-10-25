# Private candidate-pair comparison handoff

Isolated worktree: `/home/hayden/projects/ballotmancy-count-comparison-prep-20261025`. Branch: `research/count-comparison-prep-20261025`. Base: `801dda8912f575b01af034702455b055f956874e`. Six new source/doc files only; no commits, pushes or deployments. No applicable AGENTS.md was present in this checkout or its ancestors.

The separate private validator/component and invented fixtures are ready for review. The one-page app, styles, existing CountView/ElectionOutcomePreview, public APIs and selectors are byte-identical to the base. The component reuses the established private count-preview design; no public page or route imports it. `change.patch` contains all six proposed additions. The local node_modules symlink is a dependency convenience and is excluded from that patch.

The view keeps an explicit first/(first+second) candidate denominator, exact declared model/source/context/date/endpoint matching, missing/undefined/defined states, per-model and matched scored/expected coverage, unavailable calculation precision and separate experimental/unapproved status. It shows no overall mean from a successful subset and never turns a pair lead into a legal-winner or chamber claim. All people, values and hash declarations are invented. The checks reconcile those declarations; they are not source authentication or an owner-approved export schema.

Copy review is incorporated: CRPS and weighted interval score are explained, scores use percentage-point units, zero-is-perfect/lower-is-better and mean/median definitions are visible, and undefined share probability/calculation error use plain language. Technical IDs/hashes are collapsed. Heading order is h3 pair → h4 model → h5 errors.

## Validation

- Combined new + existing private-preview suite: 61 passed, 0 failed, 1 skipped (the optional legacy ElectionOutcome HTML export). All 20 new tests ran, including the new private HTML export.
- `npm run typecheck`: passed.
- `npm run build`: passed with the existing route list; no added page/API.
- Real Chromium standalone renders: desktop 1440×1000 and mobile 390×844, no horizontal overflow or page errors. Final evidence is preview-v2.html, desktop-v2-*.png, mobile-v2-*.png and visual-checks-v2.json. Both sizes were visually inspected.
- Patch reverse-application check passed. Existing app/native preview bytes were checked against the exact base.
- Initial and copy-review iteration logs/HTML/screenshots remain preserved; v2 is final.

## Integration boundaries and owner questions

This work does not parse owner research artifacts or load real data. No forecast, fitting, scoring of real predictions, deployment or scientific process control occurred. `docs/frontend-candidate-pair-preparation-20261025.md` records the unresolved owner projection questions: authenticated label/source/estimand bindings, canonical identities/aliases and exact context/cutoff, truth-independent F/FP summaries, parent/row weighting and qualified numerical/forecast evidence, per-functional score/precision statuses, and any wider ranked/winner/national contract. The draft remains limited to one ordinary candidate-mark pair per race/contest, one cycle and a declared H14 cutoff. It must not be treated as a frozen scientific/browser protocol. A future real adapter requires separate owner and frontend review; do not relax existing native validators to admit this fixture.

## Proposed source hashes

| File | SHA-256 |
|---|---|
| `components/CandidatePairComparisonPreview.tsx` | `a6d59696e7967c9a1cd32f90caefa474bb5528019be4959675a9c4621fd8e848` |
| `lib/candidate-pair-comparison-view.ts` | `4df0217d985637ce4dbfe546bb01835ebc6f647aa4ee6db54a83bd9bb03929fe` |
| `tests/fixtures/candidate-pair-comparison.mjs` | `ba5fd6b1a1cb3bf05ae2293022d9f8234646fde07c2dfdd01ff36d3127847eae` |
| `tests/candidate-pair-comparison-view.test.mjs` | `2587ea89eb7489f23894a10774147d7598e2199d2cfd8ba35b0dc62d5ab28f39` |
| `tests/candidate-pair-comparison-preview.test.mjs` | `be49f508313f4747d82a48c66038858dd0a3a2ac91409d2f1a4f518c2532aa7d` |
| `docs/frontend-candidate-pair-preparation-20261025.md` | `a597bae54785927b233801a0a630c52f58e4627142b79405399c9bfa8591d387` |

Patch SHA-256: `de184063279f075ee3af6b44a7ee72156d24f60643352786bd305e109e40038e`.
