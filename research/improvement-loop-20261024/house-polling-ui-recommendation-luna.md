# House historical polling option: UI recommendation

## Keep the choice inside the existing builder

Retain the collapsed-by-default one-page builder and original default/saved views. Add one optional checkbox under the existing fundamentals/model selection:

- **Checkbox:** “Include historical polls”
- **Dropdown when checked:** “Polling adjustment” with the saved FP method variants for the selected election year and fundamentals.
- **Year/fundamentals behavior:** Keep the selected year and fundamentals fixed when toggling. Resolve only the matching FP result. If no matching FP run exists, leave the checkbox disabled with a short reason; do not silently fall back to a different fundamentals model.
- **Summary:** “{year} · {fundamentals} · polls included” or “{year} · {fundamentals} · polls off”. Keep “Full 435 D/R/O group forecasts” visible so users know the winner target did not change.

Suggested helper copy: “Polls adjust a shared national factor. Districts without district-level polls still receive that adjustment. Polling coverage varies by year; it does not mean every district has a local poll.” The all-district group forecast count and local poll-coverage count must be shown separately.

## Coverage copy

Show two distinct quantities near the checkbox or in its helper:

- “Group-win forecasts: 435 districts.”
- “Districts with local polls in this historical set: 115 (2018), 82 (2020), 65 (2022), 62 (2024). The national polling adjustment also applies to districts without local polls.”

For the original two-party share result, use its own availability counts: “Earlier D/R share forecast available for 371 (2018), 383 (2020), 373 (2022), and 375 (2024) districts.” Keep this as a separate target and coverage statement; do not imply those counts describe the new D/R/O group forecasts or local polling coverage.

Where source dates are not strict, call these **historical polling reconstructions** and disclose that publication-date support is partly based on proxies. Avoid saying polls were known by the nominal forecast date.

## Comparison and score masks

Offer comparisons only between results with the same year, fundamentals, D/R/O group-win target, and scoring definition. Name the mask in every score label, for example:

- “All scoreable districts” (state the numerator and denominator; 2018 excludes the unscored annulled NC-09 result).
- “Districts with local polls” (count for that year).
- “Districts without local polls.”
- “Earlier D/R-share-forecast subset” and “newly covered districts” if showing that split; clarify both rows score the **group-win** target.

For poll-on versus poll-off, provide the same-mask side-by-side score, not a single aggregate that compares different sets. In particular, do not present 2020 as “polling made the model worse” based on a broad metric summary. State the precise metric and mask where results differ; describe mixed metrics as mixed, and say scores are retrospective comparisons rather than independent validation. Do not use score changes to imply polls improved or harmed accuracy without a supported comparison design.

## Evidence from current UI

The existing builder is already collapsed by default and uses the shared optional-slice checkbox/dropdown pattern (`components/ModelSliceControls.tsx`). The group controls keep year/fundamentals selection and winner target distinct, and the current method notes say polling is not connected. This recommendation is a small extension of that structure: one optional polling slice whose choices are matching saved FP runs, with explicit poll coverage and matched score masks. No new page, global blend, control projection, or tipping-point control is needed.
