# House historical release audit — October 20, 2026

The initial read-only audit below reviewed the actual sixteen-result packet against the frozen frontend. The owner’s final `research/house-benchmark-20261020/HANDOFF.md` subsequently declared publication readiness and confirmed the Fixed-law 2024 reference. The frontend owner then ran the frozen importer into `data/house`. All eighteen collection files (manifest, catalog and sixteen results) are byte-identical to the audited staging directory. Source files remain unchanged.

## Identities and admission

Source packet: `/home/hayden/projects/ballotmancy-house-benchmark/research/house-benchmark-20261020/frontend-results-v2`.

Private staging: `/tmp/ballotmancy-house-release-20261020-v2`.

Verified SHA256 identities:

| Artifact | SHA256 |
| --- | --- |
| Source `catalog.json` | `c189fe1100acec043f6ace325c81c1e2da8765c5f20a454b31e3825821d9cb7f` |
| Source `review-receipt.json` | `70f371b56d63909c847d0aa539efb28a5c4a538c3ed2e33052ea4f0a28eea9ee` |
| Staged catalog | `82c63f9ba70bcd8a9fbbb2037d5bb48d3e9ee8190462f1583caa9a3c8ddb092b` |

All sixteen catalog result fingerprints and review output fingerprints match existing bytes. Staged results are byte-identical to source results. The review's source-score receipt fingerprint also matches. All sixteen entries are publication-eligible, reviewed, and have passed race numerical/structural-score checks.

The source catalog has no scientifically selected default. The owner’s final handoff explicitly chooses the 2024 fixed-carryover result as the initial Fixed-law reference; the staged importer follows that choice. This does not establish that it outperforms the other models.

## Frozen frontend compatibility

All sixteen actual artifacts pass `parseHouseResult`, catalog/publication parsing, and chamber-readiness checks. The actual staged loader returns `available`, sixteen catalog entries, initial run `house_history_refreshed_v1-2024-h14`, and `geography:null`.

| Election | Included races per model | Excluded districts | Coverage ledger |
| --- | ---: | ---: | ---: |
| 2018 | 371 | 64 | 435 |
| 2020 | 383 | 52 | 435 |
| 2022 | 373 | 62 | 435 |
| 2024 | 375 | 60 | 435 |

Every result has `eligible_race_subset` scope and `full_chamber:false`. All sixteen report race display available and map/control/tipping unavailable. No geometry, chamber proof, or tipping proof is supplied. The joint-draw references do not establish full-chamber coverage. No current-election or polling forecast is supplied.

Candidate/outcome IDs are unique and remain local artifact identities. Each race has one named D and one named R candidate, each with one member. All 6,008 race/model records have matching observed outcome ID, winner name, member name, and party. Source observed D/R outcomes agree with the source actual two-party shares. There are no observed vote-share ties or exact favorite-probability ties in this packet.

## Independent score reconciliation

Recomputed each race's frontend categorical Brier as `sum((p - observed_indicator)^2)` across its two outcomes, then averaged races. Across all sixteen artifacts this agrees with source `multiclass_brier` and **twice** source `binary_brier`; the 0–2 frontend scale must not be compared directly with the source's 0–1 binary scale.

Every race's D win probability, mean conditional share, and 5th/95th percentiles reconcile to the source per-race scores. Maximum absolute discrepancies: win probability `0`; mean share `1.23e-15`; interval endpoint `1.12e-16`; categorical Brier `5.56e-17`. Source all-race cycle-score means also agree within `1e-12`.

Fixed browser assertions for the staged initial 2024 view:

- Run: `house_history_refreshed_v1-2024-h14`.
- Result SHA256: `0548d05a65b7575780da7e1e0a6d4c3cdb4d356a300fd877e8dd019c4d302a0c`.
- Configuration SHA256: `6fb45986f5acdeefe90bad6aaa594ccb0a872f196a4f275467d76d8809b72cd0`.
- Favorites won: **365 / 375**, with **0** tied favorites excluded.
- Frontend probability error: **0.045**; unrounded categorical Brier `0.04520387457833331`.
- Source binary Brier: `0.022601937289166662`.
- Tested coverage: **375 / 435**; excluded: **60**.
- Observed party winners among included races: D **188**, R **187**. These are not full-House seat totals.
- Favorite misses: CA-22, CA-45, CO-08, MI-08, NY-17, NY-19, OH-09, OR-05, PA-07, WA-03.

## Scope, copy, and disclosure findings

The result limitations preserve retrospective source/final-roster qualifications, strictly earlier fitted labels, revisited development comparisons, conditional D/R shares, omitted candidates/counts/turnout, unavailable geometry/control/current forecast, fitted-bank uncertainty, local candidate IDs, and the failed eligible-seat-subtotal precision gate. **All sixteen subtotal precision checks failed**; race numerical success must not be relabeled as qualified seat/control output.

No local filesystem paths, research paths, runtime snapshot paths, or `file://` links were found in staged result payloads. Candidate and observed-result source links point to the House Clerk's election-statistics PDFs. The public catalog contains sanitized file references rather than source checkout locations.

Two presentation issues deserve a local copy layer before public use; neither requires changing frozen artifacts:

1. The excluded-district list currently displays raw codes such as `D_nominee_count_0`, `majority_runoff_or_jungle_rule_outside_pair_scope`, and `two_incumbents_outside_engine_support`. Explain these in plain language while retaining the exact recorded reasons in advanced detail.
2. Some nondefault results contain simulated probabilities exactly 0%/100% (30 outcome entries across all sixteen results). A short explanation should state that finite saved simulations are not certainty. The initial 2024 fixed-carryover result itself has no exact-boundary probabilities.

Each result reports 80,000 saved predictive simulations; this does not mean 80,000 independent fitted-parameter estimates. Retrospective qualifications and failed seat-subtotal precision must remain visible; a reviewed packet is not prospective validation or a national House-control forecast.

## Presentation and integration

The public interface translates every unique note and all 24 exclusion-code combinations without altering source result bytes. The advanced fingerprint section links to the exact selected result, including original codes and source notes. Finite simulation probabilities of zero or one have adjacent qualifications and a general explanation that they are not certainty. Scorecards state that these are historical reconstructions using later-assembled inputs and revisited development elections; they are not pre-election published forecasts or proof of future accuracy. The 2018 Context model explains its presidential-party extrapolation in its selected-model description.

All eight model responsibilities are visible when the collapsed builder is opened. House has one selectable saved-model responsibility, three fixed built-in responsibilities, and four unavailable components. Candidate context, uncertainty and race rules are not misrepresented as independently switchable. The current Senate default and all 48 Senate result files remain unchanged.

The five frozen importer/parser fingerprints remain identical to the House owner's preflight copies. Native cross-thread delivery still fails; the completed source handoff and this shared document are the coordination record.

Final local validation: production build and typecheck pass, along with 92 House assertions. The full 116-case desktop/mobile run passed 114 cases and found two obsolete expectations that House remained preparing. Those assertions now verify the real historical subset stays neutral; all six layout cases pass on rerun. Independent actual-data desktop/mobile review passed all sixteen selections, coverage/score checks, exact saved-view restoration, finite-simulation endpoints and 2018 Context qualifications, with no JavaScript errors or horizontal overflow.
