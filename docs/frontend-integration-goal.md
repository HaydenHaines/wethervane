# Standing frontend integration goal

Authorized by Hayden on October 20, 2026. This frontend session anticipates the work of the Senate methodology and House modeling sessions, builds reusable support ahead of their releases, then integrates, tests and publishes completed features on SiteGround.

## Current integration status

The House owner has delivered `research/house-benchmark-20261020/HANDOFF.md`, explicitly authorizing frontend staging and publication of all sixteen reviewed historical results. The confirmed initial reference is `house_history_refreshed_v1-2024-h14`. The frozen importer loaded the final packet into `data/house`; source and catalog fingerprints match the independently audited staging packet. Historical models remain experimental and use reconstructed inputs; the test elections were revisited during development. Maps, polling, current2026, chamber control, tipping and seat subtotals remain unavailable. See [release evidence](frontend-house-release-audit-20261020.md).

The full eight-role builder now distinguishes built-in fixed responsibilities from components that are not connected. Optional polling remains a checkbox; alternatives use dropdowns; required models cannot be switched off. The builder starts collapsed. House source notes, exclusion reasons and model descriptions are translated in a presentation-only layer; exact source bytes and frozen parsers are unchanged. Finite simulation endpoints are explicitly qualified.

The sixteen-result release is live at frontend commit `65c8574`; CI and public desktop/mobile selection checks passed. The next additive update implements same-year House score/probability comparisons and exact saved comparison bookmarks using source-bound review groups. See [comparison evidence and limits](frontend-house-comparison-audit-20261020.md).

A direct House READY message has now arrived as well. Outbound native cross-thread acknowledgements still fail with a transport error; no delivery of an outbound message is claimed. The final handoff has been consumed and the results are published. Earlier progress entries below are retained as history and do not override this current release state.

## Ownership and coordination

- Frontend: one-page experience, component editors, saved configurations/comparisons, presentation adapters, readiness checks, accessibility, responsive behavior and deployment. Thread `01a15a54-c818-75b0-86f3-8797c3d1cfa2`; checkout `/home/hayden/projects/ballotmancy-siteground`.
- Senate: scientific components, standard methodology, execution and Senate source/results qualification. Thread `01a11695-7d58-7eb2-b2ce-e9c6c14fc007`; checkout `/home/hayden/projects/wethervane-redesign-20261006/integration`.
- House: district data, historical benchmarks, House configurations and scientific publication qualification. Thread `01a15ae1-4e5d-7711-bec6-113673b68a1d`; checkout `/home/hayden/projects/ballotmancy-house-benchmark`.

Read the latest handoffs and shared status at the start of each integration cycle. Ask owners for upcoming schemas, component IDs, representative artifacts, compatibility rules and acceptance criteria before features complete. Share frontend blockers and adapter needs early. Treat local drafts as preparation; reconcile them with the reviewed release packet before publication. If direct messaging is unavailable, use the owners' shared handoff documents without changing their files or scientific work.

## Release workflow

1. Inventory upcoming capabilities and distinguish implemented, proposed, reviewed and publishable states.
2. Prepare reusable selectors, adapters and visualizations against declared contracts and representative fixtures. Keep unreleased capabilities out of public results.
3. Integrate exact reviewed result/configuration identities, source dates, coverage, candidate membership, evidence labels and limitations. An available result does not establish predictive accuracy.
4. Verify contract compatibility, preserved configurations/comparisons, numerical presentation, map/ribbon behavior and desktop/mobile accessibility. Use actual result artifacts when available.
5. Deploy completed features to `deploy/siteground`, then verify the public site. Inform modeling owners which exact results are live.

Preserve the original best-guess polling and basic electoral-history presets unless an explicit scientific adoption decision changes the default. Pending/failed selections must not relabel the displayed forecast. Do not describe required computation as queued or running without a real execution service. Graphs consume result arrays and chamber metadata; do not create one-off graphs for individual configurations.

Whole-chamber control and tipping require the corresponding verified joint outputs and accounting. Race subsets are not full-chamber forecasts. District maps require matching dated geography. Missing results remain unavailable. New terminology needs a plain explanation where readers encounter it.

## Initial readiness backlog

| Workstream | Current evidence | Frontend next step |
| --- | --- | --- |
| Senate composition | Reviewed v2 handoff; eight roles, seven concurrent providers and real 2018 F/FP results | Scoped study adapter and controls implemented. Next, accept additional reviewed contexts/resolutions without extrapolating enum compatibility or changing full-chamber defaults |
| Senate saved comparisons | Ten completed current comparisons in `research/fundamentals-iteration-20261020/frontend-contract-root-v2/interface-v1/current-comparisons.json` | Integrated as ten selectable F/FP alternatives with joint-derived tipping, exact input cutoffs and separate evidence; preserve all original 38 run artifacts and both defaults |
| Execution lifecycle | Reviewed Python driver exists; no HTTP executor, durable scheduler or process recovery service supplied | Support ready/requires-computation/incompatible/failed states and separate requested/displayed identities; do not invent a live job service |
| House historical configurations | Final reviewed packet: four models across 2018/2020/2022/2024, coverage 371/383/373/375 of 435; source corrections and numerical refinements reviewed | Exact sixteen-result publication imported after explicit HANDOFF.md; Fixed2024 is the confirmed reference, not a best-score selection. Validate and deploy historical-only capabilities |
| House maps/control | Historical race-subset preflight passes; verified geography, full-chamber accounting and current forecast remain unavailable | Retain reusable map/ribbon components and explicit capability gates; activate only the capabilities justified by completed artifacts |
| House saved views and comparisons | Exact local bookmarks plus a reviewed source-bound same-year comparison manifest for the sixteen published results | Compare historical scores and candidate probabilities on identical districts; save and restore both exact identities together. Keep stale entries visible, preserve the main result on failure, and admit future comparison groups only after source review |
| Source transparency | Current Senate source receipt is live; historical per-poll list remains unpublished; initial House release uses no polling | Accept future source-selection receipts independently of forecasts; preserve the distinction between no polling in a model, source-intake failures and excluded observations |
| Future population/count stack | New `frontend-count-stack-preview-20261020.md` is preparation only; Texas pilot lacks the final qualified public packet | Private typed view model and invented fixtures prepare weighted summaries, participation, candidate counts, missing values and unresolved outcomes. No request/provider activation or public fixture route |
| Terminology correction from new handoff | `frontend-poll-admission-handoff-20261020.md` distinguishes the historical recurring pollster/state challenger from race-persistent error already in standard polling | Corrected choice explanation and presentation label; same selected polls, different learned recurring-error adjustment |

## Initial source review

The newer Senate handoff supersedes the implementation-status sections of the original composition memo. Its custom executable examples cover30 ordinary D/R contests, not arbitrary complete Senate configurations. They have numerical evidence but empirical status `unassessed`; they must not populate a full-chamber forecast without the separate scientific accounting.

The current House contract is `research/house-benchmark-20261020/frontend-contract-draft.json`, explicitly a draft. The final16-cell catalog and corrected results are still being qualified. Initial House capability is fundamentals-only, `polling:false`, and no current/full-chamber forecast. Source-intake failures are recorded in `research/house-benchmark-20261020/poll-source-inventory-v2/admission-decision.json`; these must not be described as polls silently discarded after model admission.

Previous public frontend: `4aee86a`, with 64 desktop/mobile checks passing and live verification. This integration preserves those original scientific result files.

## First integration batch

- `scripts/export_current_comparisons.py` pins the reviewed Senate packet, all execution inputs and saved joint outputs. It derives tipping from those actual draws and checks race shares, chamber totals, control and distributions before emitting any bundles. The registry contains 48 results. New results remain experimental; passing calculation checks is separate from predictive evidence.
- Five new polling alternatives share the exact admitted observations, likelihood, run specification and receipt of the existing polling collection. Their hashes are now included in the source-selection receipt. Included/excluded records and all original receipt bindings stay unchanged.
- Numeric comparisons require matching year, date, target and forecast law. The new alternatives can be compared with each other; the older default has a different forecast definition. The UI explains that boundary.
- House `/api/chambers/house?run=ID` accepts an exact published selection. Catalog entries carry result hashes, availability, year, model identity, race coverage and evidence. The UI preserves the previous result on failed loads, maintains model family across years, resets district identity across map vintages, and supports URL sharing and browser history. The builder starts collapsed.
- House publication remains `{schemaVersion:1,status:"preparing"}`. Do not import provisional outputs. The owner requested an explicit ready handoff after auditing historical incumbency; the proposed presentation default is `house_history_refreshed_v1-2024-h14` (Fixed), subject to confirmation, and is not selected by historical score.

On the completed House handoff, run `node --experimental-strip-types scripts/import_house_publication.mjs --source-root <house-worktree> --catalog <relative-catalog-path> --review-receipt <relative-review-path> --default-run <confirmed-reference-id> --output-dir <frontend-staging-directory>`. The importer requires all five arguments, validates the review receipt, fit plans, identities, checksums and publication eligibility, and switches the manifest only after the full collection passes. Review the staged results in the one-page interface before copying the publication into `data/house` for deployment. No fit or source file is modified by the importer.

The existing three-slot catalog has saved results for all its supported current combinations; its pending lifecycle tests use an explicit synthetic fixture. The finer v2 study now connects its own source-backed pending examples. Do not misrepresent a completed current result as uncomputed, or expose the 30-race study as a complete chamber forecast.

The next UI batch follows Hayden’s full-model direction and two Luna reviews: all eight roles stay visible, unavailable slices are disabled with reasons, required parts are labeled, and optional polling uses an include checkbox with supported alternatives. See [the slice editor decisions and v2 study contract](frontend-model-slices-20261020.md). The scoped study now connects the two 30-race examples through an independent server adapter and the same one-page interface. Current full-chamber forecast support stays unchanged. House’s source correction has progressed to `panel-v6` and a fresh 16-cell batch; await the explicit `frontend-results-v2` ready packet.

## Scoped component-study integration

The 2018 study provides two exact saved setups with distinct scientific-plan bindings, source-pinned candidate/observed joins, and honest 80%/95% intervals. It exposes seven concurrent controls, retains the previous result when a requested calculation is unavailable, and distinguishes numerical checks from unassessed predictive evidence. A race-only scorecard, probability ribbon, map, saved views and comparisons work without invented chamber outputs. All original 48 forecast artifacts and defaults are preserved.

Next integration targets remain owner-qualified House publication, additional reviewed component contexts/resolutions, and the Senate population pilot. The pilot and all future roles remain disabled until their completed contracts justify activation. House has frozen a read-only copy of the importer and parser for its own publication review; those files and hashes remain unchanged in this batch.

## House slices and future count preparation

The House builder uses the same eight-role renderer with its own adapter. Only the saved district-level fundamentals models are selectable; shared national uncertainty remains bundled, and other roles explain their actual House availability. Capability flags alone cannot create executable component choices. Full-chamber result fixtures correctly describe existing fixed chamber calculations without inventing a separate count/rule selector. The public House endpoint remains preparing.

The Senate owner supplied a new count-stack preview. [Preparation notes and admission requirements](frontend-count-preparation-20261020.md) record its hash and semantic boundaries. `CountForecastPreview` is an internal fixture component, absent from production pages and APIs. It separates population, event participants, administrative reports, candidate votes, noncandidate outcomes and chamber organization. It preserves declared interval levels and denominators, withholds incomplete joint summaries, retains unresolved probability, and never creates count forecasts from the existing share results. Real activation awaits a final reviewed public contract and source/rule qualification.

Earlier comparison/House preparation validation: 74 desktop/mobile checks and 78 House parser/importer/readiness assertions passed, along with the existing data, composition, score, state, ribbon and artifact checks.

Scoped-study validation: production build and typecheck pass. All 92 desktop/mobile cases pass after narrowing an ambiguous test alert locator; the final 10 study checks were rerun after copy edits. Seven export/source-replay checks and 12 API/presentation tests pass. The original 23 scoreboard/presentation checks, explorer state and artifact checks pass. Desktop/mobile screenshots show no page overflow or JavaScript errors. All 48 prior forecast files and both defaults remain unchanged; frozen House parser hashes still match.

House-slice/count-preparation validation: all 98 desktop/mobile cases, 83 House assertions and 24 private count-view/render checks pass. Production build and typecheck pass. The private prototype was inspected on desktop and mobile without page overflow. Independent administrative source admission and declared event/contest scope are enforced. No production entry imports the prototype, and no public count endpoint exists. All result data and the five frozen House importer/parser hashes remain unchanged.

Latest coordination evidence: House status at `2026-10-20T22:06:52Z` reports all 16 original corrected fits finished, 14 passing, with the two predeclared numerical refinements and final review outstanding. No `frontend-results-v2` ready packet is present. Outbound native thread updates failed with a transport error; these notes record frontend readiness without claiming delivery to either owner. Continue from the shared status/handoffs until messaging recovers.

## Exact House saved views

The collapsed House builder now includes a collapsed “Saved House views” section. Views are local to the browser/device, with no account or sync claim. Each stores a bounded name, exact run/result/configuration identity, election year, district map edition, selected race and forecast/actual mode. It stores no forecast arrays. Catalog preflight is separate from final result verification. Missing or withdrawn entries remain visible and deletable; an updated result never silently replaces the saved one.

Saving does not navigate. Restoring loads and verifies the original result before changing the URL or displayed data. Failed restores preserve the old view; retry retains the original immutable binding. Back/Forward and reload retain the selected district and map mode. Browser-storage write failures retain the old saved list and report the failure. A historical actual-winner view can preserve an unknown district result without replacing it with a forecast.

These are view bookmarks, not evidence that two models have comparable dates, targets or testing support. Numeric House comparison remains dependent on the reviewed scientific comparison metadata. The same five frozen importer/parser files remain unchanged.

Validation: 92 House assertions and all 42 affected desktop/mobile House selection and chamber-navigation cases pass. Production build/typecheck pass. Desktop/mobile saved-list screenshots were inspected, including stale entries and map-mode labels; no page overflow or JavaScript errors. All Senate result data, defaults and House publication state are preserved. The live 2022 Context refinement process was revalidated while this work proceeded; final House publication still awaits the owner-qualified packet.

Final local validation: production build and typecheck pass, along with 92 House assertions. The full 116-case desktop/mobile run passed 114 cases and found two obsolete expectations that House remained preparing. Those assertions now verify the real historical subset stays neutral; all six layout cases pass on rerun. Independent actual-data desktop/mobile review passed all sixteen selections, coverage/score checks, exact saved-view restoration, finite-simulation endpoints and 2018 Context qualifications, with no JavaScript errors or horizontal overflow.
