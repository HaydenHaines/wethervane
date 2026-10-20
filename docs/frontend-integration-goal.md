# Standing frontend integration goal

Authorized by Hayden on October 20, 2026. This frontend session prepares reusable support for the Senate and House owners, then integrates, tests and publishes their completed, reviewed capabilities on SiteGround. This is the current status and backlog; earlier integration history is preserved in git.

## Current release and coordination

The **last verified public release is `de6e80a` at [ballotmancy.com](https://ballotmancy.com)**. [CI run 39511922287 passed](https://github.com/HaydenHaines/wethervane/actions/runs/39511922287). Public desktop and Pixel7 checks passed for House selection, same-year comparison, exact saved-view restoration, sharing and reload, with no JavaScript errors or horizontal overflow. See [House release evidence](frontend-house-release-audit-20261020.md) and [comparison evidence and limits](frontend-house-comparison-audit-20261020.md).

The House final `research/house-benchmark-20261020/HANDOFF.md` has been consumed. All sixteen reviewed results are published: four models across 2018/2020/2022/2024, covering 371/383/373/375 of 435 districts respectively. `house_history_refreshed_v1-2024-h14` is the confirmed initial reference, not a model selected for the best historical score. House status updated `2026-10-20T22:59:26Z` reports the goal complete, sixteen qualified cells and no running tool session or remaining required work. Older pending/refinement fields describe preserved history, not a publication blocker.

House models remain experimental historical reconstructions with some later-assembled inputs and test elections revisited during development. Only historical race capabilities are available. No district geometry, polling, current-2026 forecast, seat subtotals, chamber control or tipping is activated; all sixteen eligible-seat-subtotal precision gates failed. Same-year comparisons describe separately simulated models, not paired-draw uncertainty or prospective validation.

The eight-role builders start collapsed and distinguish required controls, fixed built-in responsibilities and components that are not connected. House has one selectable saved-model responsibility, three fixed responsibilities and four unavailable components. Optional Senate polling uses a positive checkbox with supported alternatives. Pending/failed requests retain the exact displayed result and its label. Source explanations and finite-simulation endpoint qualifications are presentation-only; scientific artifacts and frozen parsers are unchanged.

An inbound House READY message was delivered. Frontend outbound native acknowledgements still fail with a transport error in this cycle; this document acknowledges consumption and publication without claiming outbound delivery. Continue through shared handoff/status documents while transport is unavailable. Source-side notification completion and frontend outbound delivery are separate facts.

## Ownership

- **Frontend:** one-page experience, editors, exact saved views/comparisons, adapters, readiness checks, accessibility, responsive behavior and deployment. Thread `01a15a54-c818-75b0-86f3-8797c3d1cfa2`; checkout `/home/hayden/projects/ballotmancy-siteground`.
- **Senate:** scientific components, methodology, execution, source/results qualification and population/count contracts. Thread `01a11695-7d58-7eb2-b2ce-e9c6c14fc007`; checkout `/home/hayden/projects/wethervane-redesign-20261006/integration`.
- **House:** district data, historical benchmarks, configurations and scientific publication qualification. Thread `01a15ae1-4e5d-7711-bec6-113673b68a1d`; checkout `/home/hayden/projects/ballotmancy-house-benchmark`.

Read owners' latest handoffs/status each cycle. Request upcoming schemas, provider/instance IDs, representative artifacts, compatibility rules and acceptance criteria early. Frontend preparation does not authorize changing scientific worktrees or publishing drafts. Report adapter blockers through the available coordination channel.

## Current backlog

| Workstream | Current evidence and next requirement |
| --- | --- |
| Senate saved forecasts | Preserve all 48 aggregate results: the original 38 plus ten reviewed current F/FP alternatives, with joint-derived tipping and exact cutoffs. Preserve the best-guess polling default and basic electoral-history preset unless the owner explicitly adopts a replacement. The older default and newer alternatives have different forecast definitions; comparisons must respect that boundary. |
| Senate component study | Two ready 2018 F/FP examples cover 30 ordinary D/R races, with seven concurrent controls, source-bound candidate/observed joins and supplied 80%/95% intervals. Numerical checks pass; empirical evidence remains `unassessed`. Exact candidate-off examples require computation. Additional contexts/options need reviewed server bindings; enum membership is insufficient. No full-chamber output follows from this subset. |
| Execution lifecycle | No HTTP executor, durable scheduler or process-recovery service supplied. Preserve ready/requires-computation/incompatible/failed states and requested/displayed identities. Never claim queued/running computation without a real service receipt. |
| House publication/comparison | Sixteen historical results, all 48 ordered same-year pairs and exact two-result bookmarks are live. Preserve immutable bindings, common district/candidate/observed support, stale-entry visibility and atomic restore. New groups require reviewed source-bound evidence. |
| Richer House testing evidence | This update adds a separate source-bound adapter and collapsed disclosure for all sixteen models: mean absolute Democratic vote-share miss, central-90 coverage paired with width, and saved CRPS. It verifies the exact reviewed result and score identities; see [source audit](frontend-house-score-evidence-audit-20261020.md). Calibration, competitive-subset displays and score-difference precision remain future adapters with their own definitions; public winner summaries cannot supply them. |
| House future capabilities | No admitted geometry, current forecast, independent polling/turnout/count components or qualified full-chamber accounting. Keep reusable district/ribbon components and readiness gates. Publisher map labels and common draw-cycle IDs do not establish geometry or chamber readiness. |
| Source transparency | Current Senate question-selection receipt is live and hash-bound to applicable exact polling results; historical per-poll lists remain unpublished. House uses no polling; source-intake failures differ from observations excluded after model admission. Preserve recurring pollster/state error explanations separately from standard race-persistent error. |
| Population/count stack | The latest Senate preview remains an unqualified proposal. A private typed prototype and invented fixtures exist, without a production import, public fixture route or activated provider/request. A Texas pilot is not national support. Follow [count admission requirements](frontend-count-preparation-20261020.md) before activation. |

## Handoff requirements and release gates

1. Obtain explicit owner publication approval, exact catalog/result/review hashes, evidence states, source dates, coverage, limitations and confirmed presentation default. Calculation completion, source admission, numerical qualification and predictive evidence remain distinct.
2. Require request/resolution/result contracts, provider/version/instance IDs, option defaults, exact supported bindings and representative ready/unavailable/failure examples. A capability flag alone cannot create a selectable or executable component.
3. Verify immutable result/configuration identity, race/outcome/member support, observed truth, target, information date/mode, horizon, denominator and matched comparison cohort. Group probabilities are not individual-candidate probabilities. Preserve same-state specials and reset districts across map editions.
4. Require independently admitted matching dated geography before coloring districts. Control/tipping requires complete chamber accounting and qualified joint outputs. Never convert marginals, partial coverage or failed seat-subtotal summaries into chamber forecasts, pair independent simulation rows or silently score unmatched intersections.
5. Count activation requires a final qualified public catalog/options and request/result packet, immutable weighted summaries/manifests, a real reviewed example, event/contest/candidate/rule axes, source/legal qualification, parent-mass and same-event reconciliation, declared intervals/denominators, execution behavior and replay checks. Preserve missing parents, unsupported rules and unresolved outcomes. Existing conditional shares cannot supply missing population, participation or candidate counts; retain the detailed [preparation boundaries](frontend-count-preparation-20261020.md).
6. Integrate through adapters without changing owner artifacts. The frozen House importer requires explicit source root, catalog, review receipt, confirmed default and staging output; validate the whole collection before switching its manifest. Keep local paths and raw source contents out of browser payloads.
7. Test actual records, numeric presentation, failure/retry, saved bindings, URL/history, accessibility and desktop/mobile layout. Deploy to `deploy/siteground`, verify the public site, record exact live identities and notify owners when transport permits.

Graphs consume result arrays and chamber metadata rather than one-off configuration logic. Defaults and scientific bytes stay unchanged without an explicit reviewed handoff. Explain unfamiliar terminology where readers encounter it.

## Unchanged Senate handoff fingerprints

These four documents under the Senate checkout's `docs/development/` are unchanged in this cycle:

| Document | SHA256 |
| --- | --- |
| `frontend-model-composition-handoff-20261020.md` | `51bd24a28f6e19666dff97aa3f490fcba2695f7db4b5580831ebe2fb1558b73b` |
| `frontend-model-composition-handoff-20261020-v2.md` | `32d2b419bcdea4dc3beb26c906e6c1a06cf37bdd1507ebd119589eeb45998951` |
| `frontend-poll-admission-handoff-20261020.md` | `d955798f52d04d879886cec204b35b110fe667a5b73df5f6076790b187f063fa` |
| `frontend-count-stack-preview-20261020.md` | `dbf74bbe8bdc1b789ad7025d8f8483f1c6e92d11c5c91ff56529f8f981ea7fe7` |
