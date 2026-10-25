# Private forecast-quality scoreboard preparation

`components/EvaluationScoreboardPreview.tsx` is a private React renderer for the invented `PrivateEvaluationScoreboardDraftV1` in `lib/evaluation-scoreboard-view.ts`. No page, API, model choice, or live result loader imports it. It is a one-page composition study using the existing count-preview styles. The existing candidate-pair preview and public validators are separate.

The parent supplies `view`, `selectedCycleId`, `selectedModelId`, and optionally `compareModelId`. There is no selector or session state inside the component. Model ingredients, cycles, ordered named-pair identities, and administrative-report rosters come from the supplied draft. Every expected pair and report remains in coverage even when a forecast or observation is absent. The static invented example and direct server-render test are in `tests/fixtures/evaluation-scoreboard.mjs` and `tests/evaluation-scoreboard-preview.test.mjs`.

The view model accepts **supplied** CRPS, weighted interval score, Brier, natural-log loss, and whole-vector energy values. It does not calculate scoring rules, an aggregate from successful rows, an equal-cycle mean, or a forecast. A supplied partial value is labeled partial and lists its scored IDs; a complete value requires the entire expected roster. Pair-share native fractions become percentage points on display; pair-margin values are already in points. Report errors stay in counts, while Brier and log losses are dimensionless. Range coverage and range width appear together. The strictly-leading named-pair event excludes ties and is not legal-winner or chamber probability. Energy belongs to the entire ordered share vector, with its square-root dimension scaling; it is not decomposed into pair effects.

Run, research evidence, adoption, scope match, input-information match, metric availability, and metric precision remain separate. Different model inputs permit a **descriptive** score difference only when the scored target, ordered roster, cutoff, observed source/view and actual observed values, metric definition, weighting, units, report profile, and scored IDs match. The display does not attribute such a difference to polling or another component, or predict future accuracy. Positive undefined mass, absent observations, inactive events, active reports of zero, unavailable/not-requested outputs, exact zero loss, and infinite log loss have distinct states. A positive-weight missing vector coordinate withholds the whole energy score.

Administrative totals are named reports under their exact supplied release/profile and definition. They are not independently established exact physical participation. The source-quality flags, unknown source variance, and release/state-effect provenance stay in report details. In particular, a 2018 participating-voter profile must remain distinct from later counted-ballot profiles; this prototype contains no actual historical labels or scores. Report and cycle totals are always derived from the supplied fixture, with no fixed national or year counts.

## Local verification

```sh
npm run test:count-preview
npm run typecheck
npm run build
```

Run typecheck and build sequentially because Next regenerates `.next/types` during a build. For standalone invented HTML, set `BALLOTMANCY_EVALUATION_PREVIEW_OUTPUT` to a local output path when running `node --experimental-strip-types --test tests/evaluation-scoreboard-preview.test.mjs`. The lab handoff at `frontend-evaluation-scoreboard-preparation-sol-v1/author-handoff.md` records source hashes, visual checks, and remaining integration boundaries.

This V1 draft is **not** an owner-approved browser contract. Its declared hashes do not authenticate source files. The accepted evaluation interfaces have synthetic/source-reader evidence, not an actual qualified full-count forecast score or public adoption. Future real integration requires an authenticated owner projection, source and identity binding, accepted precision metadata, and explicit publication approval.
