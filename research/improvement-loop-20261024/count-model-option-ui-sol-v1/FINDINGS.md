# Count-model optional-choice UI preparation

Prepared in isolated worktree `/home/hayden/projects/count-option-ui-sol-v1`, branch `research/count-option-ui-sol-v1`, based on freshly fetched `origin/deploy/siteground` commit `2c8b04bd88ee70ac08b44d0de0651964a7830582`. No deployment, push, shared-worktree content edit or owner-model change occurred. Changes are uncommitted for review.

## Implemented preparation

The existing collapsed **Build your perspective** panel contains one additional count-model availability section. It has a disabled “Count model · Not available” action, identifies the forecast actually displayed, and separates missing connected result, diagnostics and race/chamber coverage. It explains that participation/candidate counts and election rules form a different target from the saved aggregate component choices, and that an optional future result need not become the default. There is no competing page, top-level toggle, enabled count selector, request handler, fabricated forecast, queued job or new production schema.

Existing component/preset controls, requested-versus-displayed status, historical results, default reset and all public data remain unchanged. Unsupported/pending aggregate requests cannot make this count section look selected or ready. Its availability is deliberately not derived from an aggregate result's numerical or predictive status. No global claim is made that count computations do not exist: the text says no count result/diagnostics/coverage is **connected to this view**.

Changed source files are `components/StackControls.tsx`, new `components/CountModelAvailability.tsx`, narrowly scoped styles in `app/globals.css`, and new `tests/count-model-availability.test.mjs`. This is preparation, not a connected count-result adapter.

## Why the existing contracts are insufficient

The public composition catalog and Node resolver bind exact saved configurations to three aggregate slots (`base`, `polling`, `uncertainty`) and saved ExplorerBundle identities. Their current targets describe aggregate principal-pair or historical categorical winner forecasts. They do not describe population parent masses, all-candidate counts, participant/exposure quantities, joint integer rows, full roster/rule coverage or count-result diagnostic qualifications. Adding a count ID to the existing `base` choices would silently change the target and make other controls imply unsupported compatibility.

The reviewed generic composition handoff supplies useful identities, separate lifecycle/evidence status, source-bound configurations and selected-versus-displayed rules. Its real custom execution target is ordinary D/R shares; saved full-chamber aggregate comparisons have additional explicit adapters. The same handoff says the complete count target needs its own versioned driver and qualification. Generic lifecycle conventions can be reused, but do not authorize an invented count-result parser.

The repository's `CountViewDraft` and `CountForecastPreview` are explicitly private **invented-fixture presentation preparation**. The type requires `provenance.kind = invented_fixture`; it is not an owner result schema. Replacing that literal, importing its synthetic numbers into public data, or treating fixture readiness checks as a certified owner adapter would be misleading. Neither a production count result contract nor a source-bound count-model public exporter was supplied in the inspected frontend handoffs. Thus no ready-state count integration is claimed here.

## Exact next owner handoff

To connect an optional result, the frontend needs a separately versioned, source-bound owner packet with:

1. Exact result/configuration/request/law identities; selected provider versions and scientific target; artifact paths/hashes and public export permission; a real serialized schema/example plus parser/adapter validation. A model nickname or an active sampler PID is insufficient.
2. Explicit lifecycle, actual calculation/diagnostic status and limitations, with separately labeled predictive evidence. Optional listing is not conditional on beating the aggregate model. Diagnostic limits and any withheld quantities must be preserved; comparative performance controls default promotion separately.
3. Dated geography, event/contest/candidate identities and complete/partial coverage, ballot/rule support and accounting categories, and separate observed-report versus predictive quantities with units/denominators/reference and publication dates. A TX-only result may support a clearly scoped TX option; it cannot populate a full-Senate map/control panel as if all seats were modeled.
4. Authenticated original population-parent weights and conditional weighting/normalization diagnostics; exact joint artifact and owner-generated weighted summaries. Do not reconstruct counts or joint chamber uncertainty from marginal summaries or independent race coin flips.
5. For chamber outputs, full incoming-seat plus retained-seat ledger, scenario/caucus assumptions, vice-presidential tie rule and owner confirmation that winners/control derive from the same qualified joint integer rows. Unsupported chamber or margin/tipping outputs remain absent, with explicit reasons.
6. Actual structural/polling/source cutoffs and vintage qualifications, compatible output capabilities, diagnostic/failure receipts and stable versioned result identity. No fabricated timestamps or best-performance claim.

Once supplied, introduce a versioned target/family adapter and exact saved-result entry within the existing panel. Choose the **whole identified count configuration** before exposing only its advertised compatible component controls. Keep the older displayed forecast explicitly named while loading/failing; accept only a matching requested identity. A partial count view should display only supported races/quantities and suppress unrelated aggregate chamber/map/tipping panels rather than reuse their numbers under a count label. Preserve the default registry pointer until separate evidence supports a most-performant default decision. No production contract or threshold is invented in this preparation.

## Checks and limits

Three new tests render the actual React StackControls using real checked-in 2026/2018/2022 bundles. They verify the collapsed placement, disabled option, separate missing statuses, unsupported/pending selection behavior, existing displayed identity and default preservation. The alternate request-state objects are test-only metadata and contain no count forecast values.

`npm run typecheck` passed. Forty-seven existing explorer identity/state, composition resolver and count view/preview tests passed. All public data files were compared byte-for-byte with the base commit; none changed. Tests used a temporary node_modules symlink to the existing installed dependencies read-only; the symlink was removed afterward. No dependency install, full Next build, browser visual run, scientific job or service restart occurred. The new HTML was exercised through real React server rendering, not a browser screenshot. Initial new-test fixture metadata omitted required empty arrays and failed; supplying the complete existing resolution fields corrected the test without changing production contracts.

See logs and `evidence.json` for exact source pins and validation. This work is ready for independent code/UX review, not release or result publication.
