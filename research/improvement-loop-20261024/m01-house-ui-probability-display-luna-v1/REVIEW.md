# House probability display audit (canonical Siteground checkout)

## Scope and verdict

Read-only audit of `/home/hayden/projects/ballotmancy-siteground` at commit `317ff150f012871aec6c71ae680d7e76eec5b4db`, branch `deploy/siteground`. No files were changed and no deployment or forecast job was run.

There is no House-specific live presentation path in this frontend checkout. The House artifacts under `data/house-live` contain a catalog, race-level probability summaries, a joint-draw manifest and a publication manifest, but no `web` component, route or API client references them. The existing probability headline is explicitly for the Senate. The current map shows county/type partisan share, not House race win probabilities. Consequently I cannot identify a current House UI location that rounds a House result to 0% or 100%; the relevant risk is in any future integration and in analogous existing presentation helpers.

## What the code currently displays

- `web/app/forecast/page.tsx:4–10` redirects to `/forecast/senate` and describes Senate/Governor forecasts. `web/app/forecast/senate/page.tsx:42–50` includes `ChamberProbabilityBanner`; `web/app/forecast/governor/page.tsx:14–21` says governors have no chamber-control view. The route tree contains no House route.
- `web/lib/api.ts:227–263` defines/fetches Senate overview fields (rating, margin, poll count and projected seats), and `web/lib/api.ts:403–421` defines/fetches Senate chamber probabilities (control/majority percentages, seat distribution and `n_simulations`). There is no House API type or fetcher. A repository search of `web` for `house-live`, `/house`, or `House` returned no references outside unrelated household text.
- `web/components/forecast/ChamberProbabilityBanner.tsx:29–70` formats Senate simulation percentages with `.toFixed(0)`, which can render a finite simulation result as `0%`/`100%`. It includes “based on N simulations” only in the clear-leader branch (line 69), not the near-tossup branch. The component has no structural-support distinction.
- `web/components/forecast/QuantileDotplot.tsx:56–97,142–172` is not a Monte Carlo draw view: it synthesizes a fixed grid of normal quantiles from `predDemShare` and `predStd` (default `.08`), counts dots above `.5`, and shows 100 desktop / 50 mobile scenarios. A 0/100 dot count is a deterministic threshold result on that constructed grid, not House posterior support or a draw count. Do not reuse it as a display for actual House joint-draw probabilities.
- `web/components/forecast/TippingPointBar.tsx:32–41,49–71` is the Senate seat “snake”: it sorts races by margin and highlights the 18th seat, describing 18 of 33 seats for a majority. It displays ratings/margins, not probabilities.
- `web/components/map/MapShell.tsx:450–473,517–535` colors tracts from type-level `mean_pred_dem_share`; its tooltip says `Predicted: D+/R+ (xx.x% Dem)` and demographic values. `MapTooltip.tsx` only renders those strings. This is vote-share/lean information, not House race win probability. In the exact 50% case, the tooltip branch at lines 520–523 labels it `D+0.0`.

The internal House artifact schema has richer information than the frontend contract: the pinned result has 435 race records; outcome groups carry `p_win`; conditional two-candidate share summaries are separate; and the joint-draw manifest declares two million coherent rows across eight chains with D/R/O winner encoding. The artifact’s precision object has aggregate chamber MCSE fields, but the race outcome schema has no explicit outcome draw count/Monte Carlo error field. The front end currently receives none of this House information. The joint-draw bytes were not opened or inspected.

## Support and certainty distinctions needed for any future UI

Do not render “0%” or “100%” alone as certainty. Preserve the supplied estimate, while adjacent text identifies it as a simulation estimate and exposes the relevant actual denominator/count when it is an unweighted finite draw. Prefer wording such as “0 of N simulated elections” / “N of N simulated elections”; if the numeric estimate is rounded, retain enough precision or show the count so small positive values do not silently round to 0% and values below 100% do not silently round to 100%.

The data/API contract must distinguish at least:

1. **Supported outcome, zero observed wins:** outcome exists in the modeled outcome support but its finite draw count is zero. Show the observed count and N, and make clear this is a boundary in the simulation sample, not a structural impossibility.
2. **Structurally unavailable or unmodeled outcome:** do not manufacture a numeric 0% from absence. State “not modeled / unavailable under this scenario” and explain the model’s outcome grouping if needed.
3. **Single supported group:** do not label a sole group’s 100% as empirical certainty. State that it is the only represented/supported outcome under the declared model/aggregation; this is different from a finite sample with competing outcomes supported but unobserved.

The current artifact’s presence of outcome-group records and `p_win` values is not a frontend-ready structural-support indicator, nor does the existing app expose per-outcome raw counts. Add explicit contract fields for support status, observed outcome count, denominator and sampling design. If draws are weighted, do not infer a binomial confidence bound or denominator from weighted effective sample size; use the declared estimator/uncertainty method, or display only the estimate and an appropriately qualified precision measure. Do not treat all candidate/group members as independent draws: the House artifact describes joint election rows shared across districts.

## Tests required before a House UI release

- Formatter/component tests for a supported group at `0/N`, `N/N`, `1/N`, `N-1/N`, and tiny nonzero probabilities that would round to `0%`; verify no bare “certain”, “impossible”, “will win”, or “will lose” language.
- Separate fixtures for structural absence, a single supported group, and competing supported groups with a zero observed count; assert each produces distinct text/accessibility labels and never coerces missing support to numeric zero.
- Verify numerator/denominator and point estimate agree for the declared unweighted draw contract. For weighted output, assert no binomial interval is derived from effective sample size and the estimator’s documented precision field is the only uncertainty shown.
- Check every view (district card, district map tooltip/legend, chamber/control headline, snake/tipping display and accessible text) uses the same support and simulation qualifier. A margin/rating/tipping rank must never be described as win probability.
- Test server/API schema validation so absent or stale support/draw metadata fails closed instead of rendering a confident percentage.

## Frozen evidence and limits

Exact reviewed frontend and House artifact hashes are in `source-pins.json`. The catalog/publication manifest labels House artifacts available internally, and `qualification.json` says publication status is reviewed, but its limitations also state the 2026 forecast has no observed outcomes or demonstrated accuracy and that a House-specific tipping statistic is not qualified. This audit makes no release/qualification recommendation beyond the display safeguards above. It does not inspect raw House draws or district probability values.
