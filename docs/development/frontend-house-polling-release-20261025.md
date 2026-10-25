# Historical House polling integration

Historical House users can enable polling within the eight-component builder, preserve their model and election year, and compare the result with the same fundamentals model without polls. The page keeps all 435 districts in the map and ribbon, shows local polling coverage separately, and loads the polling model's exact two-party vote shares inside the district detail. The original default, original result bytes and saved configurations remain available.

## Published scope and source identity

The collection contains the original sixteen full-roster fundamentals results plus sixteen polling results: four fundamentals models across 2018, 2020, 2022 and 2024. Polling is an optional checkbox. The builder starts collapsed; the original House default exposes a button to reach the all-district polling options without changing the default automatically.

The scientific owner's final `research/house-polling-benchmark-20261024/frontend-results-v1` packet is pinned to:

- Catalog: `9b35f2a63c895cf6b01de9ac3542b2f7abcb68a555aabdc73fedcb5d55683b0c`.
- Source policy: `0fe3dca9f2a2eb1bb2098de4bfc73e0d4f634553805c15c4eae874c5fa25cdac`.
- Output audit: `86e537ae80784dd937e4c90dc6655afe3f0de1b6cb737ce3c5bd89011282b1e9`.
- Frontend collection catalog: `d016ef7dff403467bd8079803f24bbc4920af01a7588b608f0e7d6dd68611036`.

The importer verifies 437 declared source pins, matches all sixteen F/FP pairs, preserves the original F artifacts and checks its output destination before staging. A separate Luna review checked 327 source/evidence pins and independently recomputed the sixteen categorical Brier scores. The counts reflect different traversals, not different scientific packets.

These are experimental historical reconstructions. Poll availability uses interview dates and the day after archive ingestion as a proxy; original publication timestamps and final-roster vintages do not establish a strict historical replay. Adding polls improved some comparisons and hurt others. No general best-model promotion follows from this release. There is no 2026 House forecast, qualified chamber control, tipping district or count-stack activation.

## Integration boundaries

`house-polling-result.ts` and `house-polling-shares.ts` separately validate the new native targets. The frozen F validators remain unchanged. `house-reviewed-*` modules combine the selectable collection, verify each result and map, retain exact saved bindings, and allow the explicitly reviewed F/FP pairs. Existing F/F comparison checks remain in use. Display aliases are pinned to reviewed race/party identities, not accepted from similar names alone.

`/api/chambers/house/groups` serves the reviewed collection. The separate `/api/chambers/house/groups/shares` endpoint serves exact polling conditional-share results. The browser verifies source and canonical-content fingerprints before displaying shares. Shares exclude Other from their denominator; their conditional pair-winning probabilities are not used as actual group-winning probabilities. The original F share link remains explicitly labeled as the original forecast.

The scoreboard scores D/R/O group winners. Mixed F/FP comparisons use matching all-scoreable, original-share-subset and newly included district groups. Different competitive definitions are not equated. Local-polled/unpolled scores describe subsets within the polling result; the UI does not present their difference as polling's causal effect. Maps and the ribbon consume result arrays, without manually generated chart variants.

## Review and validation

Independent implementation review found a log-score-floor mismatch between normalized and saved views and a staging-parent symlink gap. The parser now uses the owner's `1e-6` floor, derived views/bindings were regenerated, and the importer checks the resolved parent before staging. Contradictory duplicate DTO summaries fail closed. A second review verified the repaired F/FP roundtrips, both comparison directions and rejection of altered scores/diagnostics. The first browser run crossed a data regeneration and is not release evidence; the rebuilt frozen collection is the test target.

Validation recorded before publication:

- TypeScript and optimized Next.js build pass.
- Original House suite: 199 pass, one pre-existing optional check skipped.
- Eleven new integration tests pass, including all 32 exact result/map roundtrips, sixteen F/FP pairs, old/new saved views, API/browser tamper rejection, reproduction from the actual scientific checkout and protected-source handling.
- Eighteen desktop/mobile browser cases pass for existing group views and the new polling flow. After the final default-to-builder entry point was added, all eight polling browser cases passed on a fresh build (twenty distinct browser cases in total).
- Desktop scoreboard inspected visually; no horizontal overflow. Browser cases also check mobile width, maps/ribbon, local-poll explanations, missing share coverage, saved pairs and failed-load recovery.

The default and original scientific result files are unchanged. The final browser check passed; this release is ready for publication within the standing frontend authorization. The calibration/economic research and the Senate owner's active full count fit remain separate work.
