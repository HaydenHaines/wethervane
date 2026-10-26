# Election-night refresh and evaluation runtime

This note describes the current forecast and the separate, pinned evaluation path. The runtime can display a locked forecast beside the automatically updated current forecast. A new current forecast does not rewrite an old evaluation lock.

## Current forecast and historical evaluation

The public forecast continues to follow the currently published edition selected by the normal publication workflow. The evaluation endpoint loads a specific model forecast and baseline from an immutable evaluation lock. It uses that lock's race roster, candidate identities, dates, protocols, comparison contract, and grading cohorts even after the current forecast changes.

The browser should show the two editions as separate records. A changed current forecast SHA means the live forecast has moved to a new edition; it does not change the frozen forecast used in the scorecard. The lock's `frozenAt` value records manifest metadata. By itself, it does not prove that the lock was made before outcomes were known. That timing claim needs a separate, reviewed admission record.

The earlier [live-mode design](../election-night-live-mode-design.md) describes county vote ingestion and live race calls. Those features are not implemented by this evaluation runtime. This runtime consumes reviewed outcome records; it does not collect precinct or county returns, make race calls, or alter a forecast based on reported votes.

## Runtime publication config

The Node route is `GET /api/election-night/evaluation?chamber=senate` or `...?chamber=house`. It accepts only the chamber selector and reads this fixed server-side file:

```text
data/election-night/evaluation-publication.json
```

Its schema is `ElectionNightEvaluationPublicationConfigV1`. The top-level `chambers` object may contain `senate`, `house`, or both. Each configured chamber has a `lock` and a `ledger`; each contains `subject`, `publication`, and `admission` artifact pins. A pin consists of a SHA-256 digest, byte length, and the matching content-addressed path `blobs/sha256/<digest>`. The store root is fixed at:

```text
data/election-night/evaluation-artifacts/
```

Do not add user-controlled paths, revision-specific environment variables, or query-string hashes. The route derives the artifact paths from the fixed store and authenticated pins. With no config or no chamber entry, the API returns `lock_not_configured`. Invalid or unverifiable configured artifacts return a generic `unavailable` response with HTTP 503; filesystem paths and artifact bytes are not returned to the browser.

Each lock pins both original forecast sources and their projections, both evaluator protocols, the comparison contract and cohort, the rubric decision, and the result policy. Its review pins and the trusted publication and admission manifests must bind the exact artifact graph. The loader verifies bytes and relationships; that verification is not a substitute for scientific, operator, or publication review.

The result ledger pin likewise requires a trusted publication and admission manifest bound to that exact ledger and evaluation lock. The ledger manifest selects an explicit `chosenHeadSha256` and contains the full ordered revision chain. It is not resolved by looking for the newest timestamp.

## Preparing and publishing an evaluation lock

Use the normal forecast workflow to publish and review the current forecast edition first. Then prepare a separate evaluation lock that names the exact model and baseline edition SHAs, fixed display and scoring cohorts, result protocols, comparison method, and rubric decision. Preserve the original forecast bytes and normalized projections. Do not point a historical lock at a mutable current-edition alias.

The Senate preparation script, `scripts/prepare_election_night_senate_lock.mjs`, creates a **non-admitted draft** for private review. It does not issue a trusted publication manifest, a scientific-admission manifest, or a completed lock. It must not be treated as a production publisher. A valid runtime entry requires an independently reviewed lock and both required trust manifests before its pins are placed in the config file.

Before publication, reviewers must verify the exact source-to-projection identities, candidate and event mapping, locked cohort and exclusions, grading policy, and the timing/evidence basis for any pre-outcome claim. House locks may carry the native null target/law identifiers only as allowed by the House binding validator; do not fill these fields with invented IDs. Senate bindings require their reviewed target/law identifiers. The lock loader enforces these edition-specific contracts.

## Recording outcomes and advancing the ledger

There is no production result-ingestion or ledger-publication command in this repository yet. **No real election outcome records or evaluation publication config are installed in this checkout.** Do not copy synthetic test records into production data or hand-author an admission decision to make the endpoint show a grade.

When a reviewed ingestion/publisher is available, it must:

1. Bind each observation to the exact locked race, event, round, rule, candidate or party-group outcome, and result policy.
2. Preserve the source document bytes and supporting evidence as immutable content-addressed artifacts; keep provenance and public URLs with the record.
3. Represent a new result correction as a new immutable revision. Keep the old record and revision bytes, name the superseded record digest, and link the new revision to the previous ledger head.
4. Create a new ledger manifest whose `revisions` list is the complete chain through its explicitly selected `chosenHeadSha256` (or null for an empty ledger). Create fresh trusted publication/admission manifests for the new ledger and replace only that chamber's ledger pins in the runtime config.
5. Publish the config update atomically after the artifacts and manifests are present and verified. Never overwrite or edit a content-addressed blob.

The evaluation loader and scorer then compare the selected ledger head with the fixed lock. Called or provisional observations remain distinct from final/certified outcomes. Partial results can expose the evaluator's raw matched-row summaries, but they do not receive a letter grade. A rubric grade is exposed only when the grader reports a complete, gradeable fixed cohort. Senate display-only races remain visible and can show separately labeled evaluator diagnostics, but do not enter the predeclared grade denominators. House winner evaluation is a D/R/O party-group event and remains distinct from any named-candidate-pair margin.

## Operational release checks

Before enabling either chamber in the config, confirm all six chamber pins refer to the intended lock and ledger subjects, their trusted publication/admission manifests are byte-authenticated and independently approved, and every artifact exists under the fixed store. Check that the lock references the intended historical forecast editions rather than the current pointer, that the selected ledger head is the reviewed head, and that the source/candidate/finality mappings are complete.

The endpoint is `no-store` and chamber-only. Its status, fixed cohort counts, per-race finality, locked forecast projections, admitted source references, and grade status are public-facing data. Never include raw vote feeds, private source captures, local file paths, reviewer credentials, or unreviewed results in the runtime response or publication config.
