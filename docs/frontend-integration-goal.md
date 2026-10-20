# Standing frontend integration goal

Authorized by Hayden on October 20, 2026. This frontend session anticipates the work of the Senate methodology and House modeling sessions, builds reusable support ahead of their releases, then integrates, tests and publishes completed features on SiteGround.

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
| Senate composition | Reviewed `docs/development/frontend-model-composition-handoff-20261020-v2.md`; eight roles, declarative option schemas, four lifecycle states and real 2018 F/FP results | Inventory and adapt the new catalog/request/result contracts; support provider/version/options without hardcoding every complete forecast |
| Senate saved comparisons | Ten completed current comparisons in `research/fundamentals-iteration-20261020/frontend-contract-root-v2/interface-v1/current-comparisons.json` | Reconcile with existing public runs; prepare imports/comparisons for additional reviewed results with exact cutoffs and scope |
| Execution lifecycle | Reviewed Python driver exists; no HTTP executor, durable scheduler or process recovery service supplied | Support ready/requires-computation/incompatible/failed states and separate requested/displayed identities; do not invent a live job service |
| House historical configurations | Four model families across 2018/2020/2022/2024; corrected panel coverage 371/383/373/375 of435; final batch/review pending | Prepare configuration/year selection, scoped scorecards and coverage/source explanations against the reviewed publication schema |
| House maps/control | Historical race-subset preflight passes; verified geography, full-chamber accounting and current forecast remain unavailable | Retain reusable map/ribbon components and explicit capability gates; activate only the capabilities justified by completed artifacts |
| Source transparency | Current Senate source receipt is live; historical per-poll list remains unpublished; initial House release uses no polling | Accept future source-selection receipts independently of forecasts; preserve the distinction between no polling in a model, source-intake failures and excluded observations |
| Terminology correction from new handoff | `frontend-poll-admission-handoff-20261020.md` distinguishes the historical recurring pollster/state challenger from race-persistent error already in standard polling | Refine the existing persistent-polling choice explanation and label before extending the catalog; same selected polls, different learned recurring-error adjustment |

## Initial source review

The newer Senate handoff supersedes the implementation-status sections of the original composition memo. Its custom executable examples cover30 ordinary D/R contests, not arbitrary complete Senate configurations. They have numerical evidence but empirical status `unassessed`; they must not populate a full-chamber forecast without the separate scientific accounting.

The current House contract is `research/house-benchmark-20261020/frontend-contract-draft.json`, explicitly a draft. The final16-cell catalog and corrected results are still being qualified. Initial House capability is fundamentals-only, `polling:false`, and no current/full-chamber forecast. Source-intake failures are recorded in `research/house-benchmark-20261020/poll-source-inventory-v2/admission-decision.json`; these must not be described as polls silently discarded after model admission.

Current public frontend: `4aee86a`, with64 desktop/mobile checks passing and live verification. This standing goal does not itself change the deployed scientific results.
