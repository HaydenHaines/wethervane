# Private named candidate-pair comparison preparation

This work is an invented presentation study. It is not an owner-approved browser contract, raw artifact parser, public page or live-data adapter. No production module imports the new preview. The existing one-page site, CountView, ElectionOutcomePreview, APIs, controls and styles remain unchanged.

`components/CandidatePairComparisonPreview.tsx` uses the existing private count-preview styles. `lib/candidate-pair-comparison-view.ts` checks a separate invented draft with three model slots: an experimental vote-count model and the preserved F/FP comparisons, described in the display as earlier models without/with race polls.

The fixed expected pair roster binds two named people in an explicit order, their declared source/model IDs and identity hashes, a race/contest/physical event and election date. Each model slot must match that roster, comparison context, pair specification and its declared source artifact. The draft is limited to ordinary candidate marks and a consistent H14 cutoff in one cycle. These are choices for the invented study, not a promise about the owner's eventual export.

Every displayed share divides the first person's votes by the two named people's combined votes. It cannot substitute all-candidate share, ranked-round share, a winner probability or chamber control. Positive probability of a zero pair total leaves that share unavailable; remaining outcomes are not given extra weight. Tiny positive chances display as “less than 0.1%,” and values just below one as “more than 99.9%”; exact zero/one remain distinct. Missing support stays unknown rather than becoming zero. Missing pairs remain in the coverage denominator, and no overall mean error is formed from a successful subset.

Prediction availability, score availability, calculation precision and qualification are separate. A missing observed share can leave a predictive summary visible while withholding scores. All supplied losses must be finite and nonnegative; the absolute error of the mean must agree with the displayed mean and observation. CRPS and weighted interval score remain supplied illustrative values: the validator does not recreate distributions or certify those calculations. It also does not authenticate the declared hashes or real observations.

The display explains means/medians, predictive ranges, CRPS, weighted interval score and average-share miss. Scores are converted from share fractions to percentage-point units, with lower-is-better and zero-is-perfect labels. It explains the difference between uncertainty about election outcomes and variation from running a simulation again. Model checks and permission for public use are explicitly unestablished; technical IDs/hashes stay in collapsed details. Model headings are h4 below each pair's h3. Each numeric error stays on one line with its points unit.

## Verification

From this isolated worktree:

```sh
node --experimental-strip-types --test \
  tests/candidate-pair-comparison-view.test.mjs \
  tests/candidate-pair-comparison-preview.test.mjs \
  tests/count-view.test.mjs tests/count-preview.test.mjs \
  tests/election-outcome-view.test.mjs tests/election-outcome-preview.test.mjs
npm run typecheck
npm run build
```

The new tests cover changed/reversed endpoint and source identities, event/date/context mismatch, pair-versus-all-candidate denominators, unavailable/undefined support, explicit legitimate zeroes, missing observations, coverage, inconsistent scalar summaries/scores, invalid qualification claims, rendering, readable labels and production-import isolation. The optional `BALLOTMANCY_PAIR_PREVIEW_OUTPUT` test exports a standalone invented HTML page for visual review without adding a route. Existing tests and validators remain unchanged.

## Questions for a future owner export

The source audit is recorded in the lab's `frontend-national-output-readiness-sol-v2.json` (SHA `9101b44e613d7061ce206a3fc97e65d39a8d58714f57f4de88704b75f7f70fbb`). The new owner interface binds 113 historical principal pairs, but that implementation receipt is not a completed count posterior or forecast qualification.

Before real integration, owner and frontend review must settle a separately versioned exported summary with authenticated source/rule/target/cutoff, exact person and alias crosswalks, event/contest axes, parent weighting and numerical/forecast evidence. Observed values need their own authenticated label/source/estimand binding; the observation in this draft is only invented. The comparison row currently keeps display names/event/date/source IDs in a separate pinned pair specification, and baseline summaries are only available inside scores when truth exists. A real projection needs an explicit choice for truth-independent summaries and missing values.

The meanings and availability of per-functional calculation precision, score-difference precision, incomplete support, competitive masks, same-row winner summaries and any national coverage claims also remain owner questions. Different training/input information does not isolate a model contribution. No fixture field or validator result resolves these questions, admits a real source, qualifies a forecast or permits a public-default change. Do not widen the existing count/outcome validators or connect a public API based on this study.
