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
| choice | Candidate choice | Disabled: no independent full candidate-choice layer; saved results define their possible winners |
| counts_rules | Vote counts & election rules | Disabled as an interchangeable layer; saved Senate results retain their fixed chamber, independent-senator and tie rules |

`lib/model-slices.ts` adapts the existing saved-forecast request to row/control descriptors. `ModelSliceControls` renders those descriptors without knowing complete forecast names. This separates the full role structure from each backend contract's actual choices. It accepts multiple controls per role and only offers an inclusion checkbox when its descriptor declares an off value. Preset defaults and all result bytes are unchanged.

## Next contract, prepared separately

`scripts/export_component_catalog.py` produces `data/model-components-v2.json` from the pinned reviewed Senate interface and ready-result receipts. It preserves scientific role/provider/version/instance IDs and declarative option schemas. The exported metadata contains eight roles, seven providers and two ordinary D/R historical examples, with separate numerical and evidence states. Source paths and probability arrays are excluded; no execution endpoint is implied.

These finer providers currently apply to a 30-race 2018 study. They cannot be substituted into the 35-race current Senate forecast through UI labels alone. The current editor therefore continues using exact saved combinations until the scoped adapter connects the finer contract.

The subsequent adapter needs candidate/observed joins from the pinned target panel and separate labels, explicit retrospective roster/poll qualifications, and interval levels `[0.025, 0.1, 0.5, 0.9, 0.975]`. Do not silently label them as the existing 90% interval. Preserve provider-instance identity, source cutoff and each result's exact support. No chamber control or tipping can be derived from these race-only examples.

The source owner’s current House successor is `panel-v6`, with a new 16-cell fit batch and an expected `frontend-results-v2` packet. Source qualification remains pending; keep House results unpublished until the explicit ready handoff.

## Verification

Production build and typecheck pass. The full desktop/mobile suite passes all 82 checks; after final readability changes, all 28 affected configuration/layout checks pass again. Seven provider-metadata checks pass, including replay from the pinned source and rejection of changed identities/evidence. Desktop/mobile screenshots were inspected, including the sticky unresolved-selection message. All 48 existing forecast hashes, the registry and defaults remain unchanged.
