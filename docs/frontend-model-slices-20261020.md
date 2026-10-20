# Full model-slice editor

Hayden requested that the configuration UI show the eventual model structure now, with unavailable slices visible and disabled. Two Luna reviews covered the scientific taxonomy and interaction design. A follow-up review addressed mobile status visibility.

## Interaction decision

Keep the existing collapsed “Build your perspective” entry point, years, exact saved presets, comparisons and saved views. Inside, show all eight responsibilities in one visible list. Each row has a purpose, availability and its actual supported controls. Multiple controls can belong to one row; these are not mutually exclusive model layers or percentages to blend.

- Use a positive native checkbox only for a real off/identity choice. Polling is the currently supported optional slice; switching it off removes polling evidence. Historical alternatives appear in a dropdown when there is more than one supported method. The last valid method is restored when polling is re-enabled; year changes retain the existing supported-selection rules.
- Required parts use “Always included” instead of a misleading toggle. Their available settings still have selectors.
- Future slices use unchecked disabled controls, muted styling and a visible reason connected by `aria-describedby`. Their text remains readable in the document even though disabled inputs leave the keyboard tab order. No tooltip is needed to discover a limitation.
- Keep requested and displayed configurations separate. The status appears before the role list, and an unresolved selection remains visible while scrolling. A disclosure explains the precise compatibility/computation reason. A pending calculation is never shown as a scheduled job.
- Use one column on mobile. Keep every role visible; no nested role accordions or horizontal table scrolling.

This follows [USWDS checkbox guidance](https://designsystem.digital.gov/components/checkbox/) on positive toggle labels and accessible hit areas, [GOV.UK select guidance](https://design-system.service.gov.uk/components/select/) on keeping selection tasks simple, and [WAI-ARIA keyboard guidance](https://www.w3.org/WAI/ARIA/apg/practices/keyboard-interface/) on disabled-control discoverability. Detailed methodological evidence stays in the existing disclosure rather than crowding each row.

## Scientific roles and current adaptation

| Canonical role | Display label | Current saved-forecast controls |
| --- | --- | --- |
| population | Population & geography | Disabled: fixed contest support is not an active population model |
| community | Community composition | Disabled: group composition is not estimated |
| fundamentals | Electoral history | Required starting-model selector; current alternatives still bundle several contributions |
| environment | National environment | Required uncertainty selector, including settings for how races can move together |
| measurement | Polls & surveys | Optional polling checkbox; historical polling-method dropdown |
| participation | Turnout & participation | Disabled: no voter-count/turnout model connected |
| choice | Candidate choice | Built in: saved results define their possible winners; no independent full candidate-choice control |
| counts_rules | Vote counts & election rules | Built in: saved Senate results retain their fixed chamber, independent-senator and tie rules; no independent vote-count control |

`lib/model-slices.ts` adapts the existing saved-forecast request to row/control descriptors. `ModelSliceControls` renders those descriptors without knowing complete forecast names. This separates the full role structure from each backend contract's actual choices. It accepts multiple controls per role and only offers an inclusion checkbox when its descriptor declares an off value. Preset defaults and all result bytes are unchanged.

## Finer contract and scoped study

`scripts/export_component_catalog.py` produces `data/model-components-v2.json` from the pinned reviewed Senate interface and ready-result receipts. It preserves scientific role/provider/version/instance IDs and declarative option schemas. The exported metadata contains eight roles, seven providers and two ordinary D/R historical examples, with separate numerical and evidence states. Source paths and probability arrays are excluded; no execution endpoint is implied.

These finer providers apply to a 30-race 2018 study. They cannot be substituted into the 35-race current Senate forecast through UI labels alone. The main editor continues using exact saved combinations. Its “Explore the 2018 component study” action opens the finer contract in the same page, with an explicit scope and a return to the prior full forecast. Both builders start collapsed.

`scripts/export_component_results.py` now joins candidates from the pinned target panel and observed D/R votes from separate labels. The exporter verifies all receipts, input identities, exact 30-race joins, probability sums and share quantiles. The UI displays the supplied 80% and 95% intervals; it never relabels them as the original forecasts' 90% interval. Candidates and polls retain their retrospective qualifications. D/R identifiers describe outcomes within each race, not persistent person IDs.

The study shows all seven provider instances concurrently: structure/candidate/race under fundamentals; carryover/context/cycle under environment; polling under measurement. Five unconnected roles stay disabled. Metadata enum membership alone does not prove compatibility. Alternative values without reviewed bindings are disabled. Both exact F/FP examples are ready; their exact candidate-off counterparts require computation. F and FP bind different scientific plans on the server. Switching polling selects the entire corresponding context.

`/api/component-studies` validates the pinned metadata and result index before advertising results. POST accepts only a known study ID and its seven provider-instance selections; source bindings cannot be supplied by the browser. It fills declared provider defaults for comparison, preserving explicit overrides such as presidential context `off`. Ready responses identify the original saved artifact. Exact pending selections have no result and no scheduled job. Other drafts explain that this deployment lacks a reviewed binding, without claiming scientific impossibility. GET result responses preserve the pinned JSON bytes so clients can verify the advertised fingerprint.

The study has a race-only scorecard, dynamic probability ribbon, map, 10-race expandable list, and detail sidebar. `deriveRaceScoreboard` reuses the original categorical scoring rules independently of chamber estimates. The probability error is the 0–2 categorical Brier score; vote-share error is the mean absolute D/R share error. Comparison requires matching races, outcome identities, observed results, target, date and denominator. Saved study views bind the exact result fingerprint; unresolved choices cannot be saved as completed results. Share links describe the displayed result, never an uncomputed draft.

No retained seats, chamber-control probabilities, tipping marker or count forecasts are synthesized. Map states outside the study are labeled “not included,” rather than “no election.” The original 48 full forecasts and preserved defaults are unchanged.

The earlier House source-pending status has been superseded: the owner delivered the final `frontend-results-v2` packet and explicit `HANDOFF.md`; all sixteen reviewed historical results and their same-year comparisons are live at frontend commit `de6e80a`. House status at `2026-10-20T22:59:26Z` reports the goal complete with no running tool session. This activates only the reviewed historical race capabilities, not geometry, polling, current forecasts or chamber totals/control/tipping. See [current integration status](frontend-integration-goal.md) and [comparison evidence](frontend-house-comparison-audit-20261020.md).

## Verification

Production build and typecheck pass. The full desktop/mobile suite passes all 82 checks; after final readability changes, all 28 affected configuration/layout checks pass again. Seven provider-metadata checks pass, including replay from the pinned source and rejection of changed identities/evidence. Desktop/mobile screenshots were inspected, including the sticky unresolved-selection message. All 48 existing forecast hashes, the registry and defaults remain unchanged.

The subsequent scoped-study batch adds 10 desktop/mobile flows, seven export checks, and 12 API/presentation tests. All 92 browser cases pass, and the final 10 study flows pass again after the source-copy review. These cover actual source-bound results, unavailable computation, failure recovery, saved views, browser history, exact interval levels and absent-map coverage. The build, typecheck, original scoring tests and artifact checks pass.

## Built-in responsibilities

A follow-up Luna review found that fixed responsibilities should not look switched off. Rows with no independent control now distinguish missing components (disabled unchecked box) from responsibilities already built into a saved model (checkmark and “Built into this model”). House uncertainty, candidate context and supported race rules are fixed within each saved result. Senate candidate options, chamber accounting and tie rules are likewise fixed. This does not claim that either collection predicts vote counts or that the partial House results establish chamber control.

Two further Luna reviews confirmed the inventory against the Senate and House handoffs and the interaction design against primary accessibility guidance. Live desktop and mobile inspection confirmed all eight Senate roles, the collapsed initial builder, readable unavailable explanations and no horizontal overflow. House population copy now explicitly says that district maps are unavailable; dated district rosters alone do not supply a displayable map. The proposed count-stack packet does not yet justify adding any further public controls.
