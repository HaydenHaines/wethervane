# WetherVane Frontend v2 Audit Report

**Date:** 2026-03-29
**Auditor:** Claude Opus 4.6 (Playwright + visual inspection)
**Target:** https://wethervane.hhaines.duckdns.org (production)
**Branch:** fix/frontend-remaining-polish (deployed)
**Rubric:** docs/frontend-v2-evaluation-rubric.md

---

## Issue Catalog

Every issue is written as a user story with severity, rubric category, and a description of expected behavior.

---

### CRITICAL ISSUES (ship-blocking)

---

#### C1. Governor races show no model predictions — all 36 display "EVEN"

**Rubric:** 2.1 (Margins display correctly), 2.5 (Cross-page data agreement)
**Severity:** Critical

> As a visitor exploring the governor forecast, I expect to see model-derived margins for each race, so I can evaluate which governor races are competitive.

**Observed:** All 36 governor races at `/forecast/governor` show "EVEN" with "Tossup" rating and 0 polls. The API endpoint `/api/v1/governor/overview` returns `{"races": []}` — zero races. The frontend generates placeholder cards with no data.

**Expected:** Either (a) governor races should have model priors from the type structure (same as Senate), or (b) if governor predictions aren't ready, the page should say so explicitly — "Governor race predictions are coming as polling data arrives" — instead of presenting 36 fake "EVEN" cards that look like real predictions.

**Why this matters:** Showing 36 identical "EVEN" cards with "Tossup" badges trains users to distrust the site's data. A visitor who sees California and Wyoming both rated "Tossup EVEN" immediately knows something is wrong.

---

#### C2. Type names are misleading — "Majority-Black Urban" is R+10.6

**Rubric:** 2.4 (Type names non-misleading), 7.7 (Uncertainty communication)
**Severity:** Critical

> As a visitor reading about Type 34 "Majority-Black Urban," I expect the name to describe what I'm looking at, so I can build a mental model of America's electoral communities.

**Observed:** Type 34 "Majority-Black Urban" has `mean_pred_dem_share=0.394` — that's R+10.6. The name implies a Democratic-leaning urban Black community, but the prediction is solidly Republican. Similarly, the type page header shows both "Likely R" badge AND the super-type "Black-Belt Evangelical" — the juxtaposition of "Urban" in the name with "Evangelical" in the super-type is confusing.

Looking at the demographics: 52.6% White, 36.5% Black, median income $40,837, pop density 27.1. This is a rural low-density Black Belt community, not an "urban" community. The name appears to be auto-generated and wrong.

**Expected:** Type names should be descriptive of the actual demographics and geography. "Rural Black Belt Mixed" or "Black Belt Working-Class" would be more accurate for Type 34. Names should never contradict the observable data on the same page.

---

#### C3. Type detail "Mean Dem share" shows "EVEN" for all types

**Rubric:** 2.1 (Margins display correctly), 1.5 (Configuration over hardcoding)
**Severity:** Critical

> As a visitor on a type detail page, I expect to see the type's predicted lean, so I can understand its political character.

**Observed:** On `/type/34`, the Demographics panel shows "Mean Dem share: EVEN" — but the page header shows "R+10.6" and the API returns `mean_pred_dem_share=0.394`. The demographic panel's Political section is displaying a different (or broken) value than the header.

**Expected:** The "Mean Dem share" field in the demographics panel should show the actual value (39.4% or R+10.6), matching the header badge. This appears to be a formatting/threshold bug similar to the old EVEN bug on the forecast page.

---

#### C4. Explore/Map "Types" toggle shows forecast legend, not type colors

**Rubric:** 1.5 (Configuration over hardcoding), 7.2 (Party color discipline)
**Severity:** Critical

> As a visitor exploring the stained-glass map in "Types" mode, I expect to see the actual super-type colors with named labels, so I can understand the community structure.

**Observed:** The explore/map page at `/explore/map` has a "Types / Forecast" toggle. When "Types" is selected, the map still shows the forecast legend (Safe D / Likely D / Lean D / Tossup / Lean R / Likely R / Safe R / No race). The map itself appears to show state-level forecast colors, not the sub-county type stained-glass pattern.

**Expected:** In "Types" mode, the map should show the stained-glass choropleth colored by super-type (Rural Young, Black-Belt Evangelical, Hispanic Exurban, Rural Evangelical, Asian-Pacific Urban), with the super-type legend using the Dusty Ink palette. This is the signature visualization of the entire project.

---

#### C5. Data consistency: Types page says 100 types / 5 super-types, but memory says model is J=100 with 8 super-types

**Rubric:** 2.3 (Model parameter consistency)
**Severity:** Critical

> As a visitor reading about the methodology, I expect the numbers to be consistent everywhere, so I can trust the rigor of the analysis.

**Observed:**
| Location | Fine types | Super-types |
|----------|-----------|-------------|
| Types index page | 100 | 5 |
| Methodology page | 100 | 5 |
| Methodology "Current Status" | 100 | 5 |
| API `/types` | 100 | (5 unique super_type_ids: 0-4) |
| API `/super-types` | — | 5 |
| MEMORY.md | 100 (J=100) | 8 |
| Explore/Shifts | — | 5 panels |

The API and frontend are consistent at 100/5 now — but MEMORY.md says "8 tract super-types (S238: relabeled from 6)." This suggests either the memory is describing the in-progress tract model (not yet deployed) or the production model has fallen behind. The numbers on the site are internally consistent, which is good, but if the tract model with 8 super-types is supposed to be deployed, it isn't.

**Expected:** All references to model parameters should match the currently deployed model. MEMORY.md should be updated to clarify which numbers describe the production vs. in-progress model.

---

### MAJOR ISSUES (fix before public launch)

---

#### M1. Embed widget has 6 React hydration errors

**Rubric:** 4.9 (No console errors), 8.8 (No console errors in production)
**Severity:** Major

> As a journalist embedding a WetherVane widget on my article, I expect it to render without JavaScript errors, so it doesn't break my page.

**Observed:** `/embed/2026-ga-senate` produces 5x React error #418 and 1x error #423 in the console. These are server/client hydration mismatches — the HTML rendered on the server doesn't match what React expects on the client.

**Expected:** Zero console errors. The embed should be a self-contained, lightweight component with no hydration issues.

---

#### M2. Embed widget renders full site footer and nav shell

**Rubric:** 8.5 (Error boundaries), general UX
**Severity:** Major

> As a journalist embedding a WetherVane widget, I expect only the forecast card to render — not the full site footer.

**Observed:** The embed page at `/embed/2026-ga-senate` renders the "Methodology | About | GitHub" footer and "WetherVane — Community-based electoral forecasting" tagline below the widget. In an iframe, this footer is confusing and wastes space.

**Expected:** The embed layout should suppress all global navigation, footer, and site chrome. Only the widget card + "Powered by WetherVane" attribution link should render.

---

#### M3. No Senate/Governor tab or toggle on forecast page

**Rubric:** 3.2 (Governor races accessible)
**Severity:** Major

> As a visitor on the forecast page, I expect to easily switch between Senate and Governor forecasts, so I can explore all 2026 races.

**Observed:** The forecast page at `/forecast` redirects to `/forecast/senate`. There is no visible tab, toggle, or link to `/forecast/governor` anywhere on the page. The only way to reach governor races is by manually typing the URL or navigating from the landing page (which also doesn't link to governors).

**Expected:** A Senate/Governor toggle or tab bar at the top of the forecast page, similar to the Types/Map/Shifts tabs on the explore page.

---

#### M4. "View on Map" links go to `/forecast` instead of focused view

**Rubric:** 3.3 (Cross-linking completeness)
**Severity:** Major

> As a visitor on a county or type detail page, I expect "View on Map" to show me that specific county/type highlighted on the map, so I can see its geographic context.

**Observed:** On `/county/13121` (Fulton County) and `/type/34`, the "View on Map" link goes to `/forecast` — the national forecast page with no focus on the relevant entity. The user lands on a full national map with no indication of where their county/type is.

**Expected:** "View on Map" should navigate to `/explore/map?focus=13121` or similar, with the map zoomed to and highlighting the relevant county/type.

---

#### M5. County detail pages have no election history visualization

**Rubric:** 1.4 (Context beats raw numbers), 7.3 (Historical context)
**Severity:** Major

> As a visitor on Fulton County's page, I expect to see how it voted in past elections, so I can understand its political trajectory.

**Observed:** `/county/13121` shows "County-level shift history coming soon" with a link to the type's shift chart as a placeholder. There is no county-level election history data displayed.

**Expected:** A simple bar or line chart showing Dem margin by election year (2008-2024), even if it's just the raw numbers in a table. This is the most natural question a visitor has about a county: "How has it voted?"

---

#### M6. Landing page balance bar segments are not interactive

**Rubric:** 3.1 (Forecast to race detail), 7.4 (Interactive exploration depth)
**Severity:** Major

> As a visitor looking at the 47D-53R balance bar, I expect to click on contested segments to see which races they represent, so I can explore close races.

**Observed:** The balance bar on the landing page shows segment buttons (GA: D+10.4, ME: D+0.7, etc.) but they are `<button>` elements that don't navigate anywhere. On the forecast page, the same segments exist but also don't link to race detail pages — they only interact with the map.

**Expected:** Balance bar segments should link to the corresponding race detail page (`/forecast/2026-ga-senate`), or at minimum show a tooltip with the race details and a "View details" link.

---

#### M7. favicon.ico returns 404 on every page load

**Rubric:** 4.9 (No console errors)
**Severity:** Major (minor technical, but every page load generates an error)

> As a visitor, I expect the site to have a favicon, so the browser tab has an identifiable icon and no console errors are generated.

**Observed:** Every page load triggers `Failed to load resource: 404` for `/favicon.ico`.

**Expected:** A favicon.ico file (even a simple "W" or weather vane icon) served from the public directory.

---

#### M8. Methodology accuracy page has contradictory sentence

**Rubric:** 1.8 (Show your work), 2.5 (Cross-page data agreement)
**Severity:** Major

> As a reader evaluating the model's credibility, I expect the accuracy analysis to be internally consistent, so I can trust the methodology.

**Observed:** On `/methodology/accuracy`, the "What This Means" section contains: "The standard holdout r (0.698) is slightly higher than LOO (0.711) is lower because the standard metric allows each county to predict its own type mean."

This sentence contradicts itself — it says holdout (0.698) is "slightly higher" than LOO (0.711), but 0.698 < 0.711. The sentence also has garbled grammar ("is slightly higher than LOO (0.711) is lower because").

**Expected:** "The standard holdout r (0.698) is slightly lower than LOO ensemble r (0.711). The LOO ensemble benefits from 160 additional features beyond type scores. The basic LOO type-mean baseline (0.448) is the honest structural-model-only metric."

---

#### M9. Methodology sections all expanded by default

**Rubric:** 1.2 (Progressive disclosure)
**Severity:** Major

> As a visitor landing on the methodology page, I expect a scannable overview that lets me drill into sections of interest, so I'm not overwhelmed by a wall of text.

**Observed:** All 8 methodology sections are expanded on page load. The page is very long and requires extensive scrolling to reach later sections. The TOC on the left is helpful but the open-by-default pattern defeats the purpose of collapsible sections.

**Expected:** Only the first section ("The Key Insight") should be expanded by default. Remaining sections collapsed, with the TOC providing jump links. Users who want to read everything can expand as they go.

---

#### M10. Explore/Types scatter plot dots have no hover tooltips

**Rubric:** 7.4 (Interactive exploration depth), 5.4 (Keyboard navigation)
**Severity:** Major

> As a visitor exploring the demographic scatter plot, I expect to hover over a dot to see which type it represents, so I can connect visual patterns to specific communities.

**Observed:** On `/types`, the scatter plot shows colored dots but hovering over them shows no tooltip. You must click a dot to get details — but the click behavior isn't visually indicated (no cursor change, no hover highlight).

**Expected:** Hover shows a tooltip with type name, super-type, and key metrics. Click navigates to the type detail page. Cursor changes to pointer on hover. On mobile, tap shows the tooltip.

---

### MINOR ISSUES (polish before or after launch)

---

#### m1. Landing page "Explore electoral types" says "100 community types" — should be dynamic

**Rubric:** 1.5 (Configuration over hardcoding)
**Severity:** Minor

> As a developer retraining the model, I expect the landing page to reflect the current model without code changes.

**Observed:** The entry point card on the landing page says "Discover the 100 community types that drive American elections" — hardcoded text.

**Expected:** Pull the count from the API dynamically, or use vague language like "Discover the community types that drive American elections."

---

#### m2. Breadcrumb "Map" link on types/explore pages goes to `/forecast`

**Rubric:** 3.4 (Breadcrumb correctness)
**Severity:** Minor

> As a visitor navigating via breadcrumbs, I expect "Map" to take me to the map, and "Home" to take me home — not the same place.

**Observed:** On `/types`, the breadcrumb shows "Map / Types" where "Map" links to `/forecast`. On county pages, "Home" also links to `/`. The conceptual hierarchy is unclear — is "Map" the parent of "Types"?

**Expected:** Breadcrumbs should follow the information architecture: Home → Explore → Types, or Home → Forecast → [Race]. "Map" should link to `/explore/map` if used.

---

#### m3. "Races tracked: 18" on methodology page — actual count is 33 Senate + 36 Governor

**Rubric:** 2.3 (Model parameter consistency)
**Severity:** Minor

> As a visitor reading the methodology, I expect the "races tracked" number to match reality.

**Observed:** Methodology "Current Status" section says "18 Races tracked." The Senate overview API returns 33 races. The governor page shows 36 races. Neither matches 18.

**Expected:** Dynamic count from the API, or accurate static count. "18 competitive races" might refer to races within a certain margin — if so, specify the threshold.

---

#### m4. Type detail "Similar Types" cards have truncated names

**Rubric:** 6.5 (Readability)
**Severity:** Minor

> As a visitor browsing similar types, I expect to read their full names.

**Observed:** On `/type/34`, the Similar Types cards show names like "Majority-Black Evangelic..." with ellipsis truncation. The cards are wide enough to fit more text.

**Expected:** Either wrap the text to a second line or use a smaller font size to fit the full name.

---

#### m5. County detail breadcrumb missing state level

**Rubric:** 3.4 (Breadcrumb correctness)
**Severity:** Minor

> As a visitor on a county page, I expect breadcrumbs to show the navigation hierarchy including state.

**Observed:** `/county/13121` shows "Home / Fulton County" — no state in the hierarchy.

**Expected:** "Home / Georgia / Fulton County" with Georgia linking to a state-level view or the forecast map focused on Georgia.

---

#### m6. "Updated 6d ago" on landing page — staleness concern

**Rubric:** 1.9 (Data provenance)
**Severity:** Minor

> As a visitor evaluating the site's credibility, I expect the data to feel current or to see a clear explanation of the update schedule.

**Observed:** Landing page shows "Updated 6d ago - 58 polls." Six days is borderline stale for a forecast site. No indication of when the next update will occur.

**Expected:** Show an absolute date ("Updated March 23, 2026") plus the update cadence ("Polls updated weekly"). If the scraper runs on a schedule, show when the next run is expected.

---

#### m7. Forecast page has no "About this race" context for races with 0 polls

**Rubric:** 1.4 (Context beats raw numbers), 1.9 (Data provenance)
**Severity:** Minor

> As a visitor looking at a race with 0 polls, I expect the site to explain what the prediction is based on.

**Observed:** Many race cards show "0 polls" (e.g., MN, NM, VA). On the forecast overview, there's no indication of what drives the prediction when there are no polls. Users may assume it's a guess.

**Expected:** On race detail pages with 0 polls (like `/forecast/2026-mn-senate`), there should be a brief note: "This forecast is based solely on the structural model prior — no state-specific polls have been incorporated yet."

---

#### m8. Type detail "Member Geography" map says "Loading map..." but loads

**Rubric:** 4.7 (Skeleton screens)
**Severity:** Minor

> As a visitor scrolling to the Member Geography section, I expect a smooth loading experience.

**Observed:** The "Member Geography" section briefly shows "Loading map..." text before the map appears. It's functional but bare.

**Expected:** A skeleton placeholder shaped like a map outline, matching the section's dimensions, instead of plain text.

---

#### m9. Mobile forecast page truncates rating badges

**Rubric:** 6.5 (Readability at 375px), 6.1 (Layout transformation)
**Severity:** Minor

> As a mobile visitor, I expect to see the full rating badge for each race.

**Observed:** At 375px, the race cards in the forecast show "Tossup" badges that are slightly cut off on the right edge for the third card in each row (e.g., ME in the Key Races row).

**Expected:** Either reduce to 2 cards per row on mobile or ensure badges are fully visible.

---

#### m10. No search/filter on types listing page (100 items)

**Rubric:** 7.4 (Interactive exploration depth)
**Severity:** Minor

> As a visitor looking for a specific type, I expect to search or filter rather than scroll through 100 cards.

**Observed:** `/types` lists all 100 types in a long scrollable page. No search box, no filter by super-type, no jump links to super-type sections.

**Expected:** At minimum, jump links to each super-type group at the top. Ideally, a search-as-you-type filter.

---

#### m11. Anchor link "#" text visible in methodology section headings

**Rubric:** General polish
**Severity:** Minor

> As a reader of the methodology, I expect clean headings without distracting link symbols.

**Observed:** Each methodology section heading shows a "#" symbol as part of the anchor link text (e.g., "The Key Insight #"). The "#" is part of the button text and always visible.

**Expected:** The "#" anchor link should be hidden by default and shown only on hover (`:hover` or `:focus-visible`), following standard documentation site patterns.

---

#### m12. No `prefers-reduced-motion` handling observed

**Rubric:** 5.7 (Reduced motion)
**Severity:** Minor

> As a visitor with motion sensitivity, I expect animations to be suppressed when my OS preferences say so.

**Observed:** Could not confirm whether animations respect `prefers-reduced-motion`. The methodology scrollytelling sections have expand/collapse animations. No explicit reduced-motion CSS was observed.

**Expected:** All animations should check `prefers-reduced-motion: reduce` and fall back to instant transitions.

---

#### m13. Explore map has no visible legend in "Types" mode

**Rubric:** 5.3 (No color-only encoding), 7.1 (Population-normalized geography)
**Severity:** Minor

> As a visitor exploring the map in Types mode, I expect a legend that tells me what the colors mean.

**Observed:** When "Types" is selected on `/explore/map`, the legend still shows forecast categories (Safe D, Lean D, etc.) instead of super-type names. Since the map is showing type colors (which don't correspond to D/R lean), the legend is actively misleading.

**Expected:** In Types mode, show a legend with the 5 super-type names and their assigned colors.

---

### TASTE / DESIGN SUGGESTIONS (non-blocking, opinionated)

---

#### T1. Landing page hero could be more impactful

> As a first-time visitor, I want to immediately understand what makes WetherVane different from FiveThirtyEight, so I can decide whether to engage.

**Observed:** The landing page hero says "Republicans Strongly Favored to Hold the Senate" with the 47-53 split. This is good but generic — any forecast site has this.

**Suggestion:** Add a one-line differentiator below the subtitle: "Based on 100 electoral communities discovered from how places move together — not polls alone." This immediately communicates the unique methodology.

---

#### T2. Balance bar on landing could use state labels on hover

> As a visitor studying the balance bar, I want to know which state each segment represents without having to decode the colors.

**Observed:** The balance bar shows colored segments with state abbreviations on hover (in the accessibility tree) but no visual tooltip.

**Suggestion:** Show state abbreviation + margin on hover/tap for contested segments. The segments already have this data (e.g., "GA: D+10.4").

---

#### T3. Race detail page could show 90% CI explicitly

> As a quantitatively-minded visitor, I want to see the confidence interval around the margin, so I know how uncertain the prediction is.

**Observed:** The race detail page shows the margin (D+10.4), the dotplot (95/5 scenarios), and the "±7pp std" in the dotplot description. But the 90% CI isn't shown as a range next to the margin.

**Suggestion:** Add "90% CI: D+3.4 to D+17.4" below the margin, using the ±7pp std to compute it. This directly serves the "uncertainty is the story" principle.

---

#### T4. Shift history charts could annotate election events

> As a visitor studying the shift patterns, I want to see which elections caused the shifts, so I can connect patterns to real-world events.

**Observed:** The shift history charts on type detail and explore/shifts pages show cycle labels ('04, '08, etc.) but no annotation of major events (Obama, Tea Party, Trump, etc.).

**Suggestion:** Add subtle event labels above notable inflection points. The methodology accuracy page already does this for cross-election validation — extend the pattern.

---

#### T5. Footer is minimal — could include more navigation

> As a visitor at the bottom of a long page, I want quick links to all major sections.

**Observed:** Footer shows only "Methodology | About | GitHub" on the left and tagline on the right.

**Suggestion:** Add links to Forecast, Types, Shifts, and About in the footer. Also add "Built by Hayden Haines" credit.

---

## Rubric Scoring

| Category | Score | Pass? | Notes |
|----------|-------|-------|-------|
| 1. Core Design Principles | **2.0** | Borderline | P1 (uncertainty) good on race detail via dotplot. P2 (progressive disclosure) works landing→forecast→detail. P3 (big number) strong. P5 (config) mostly good but hardcoded "100" in landing text. P6 (mobile) good transformations. P7 (speed) skeleton screens missing in spots. |
| 2. Data Accuracy | **1.3** | **FAIL** | Governor races all EVEN (C1). Type names misleading (C2). Type detail "Mean Dem share" shows EVEN (C3). Methodology accuracy contradicts itself (M8). |
| 3. Navigation | **1.9** | Borderline | Race cards now link to detail (fixed from prior audit). But no Senate/Governor toggle (M3), "View on Map" goes nowhere useful (M4), breadcrumbs inconsistent (m2, m5). |
| 4. Performance | **2.3** | Pass | Pages load fast. Map progressive loading works. Skeleton screens inconsistent (m8). Embed hydration errors (M1). favicon 404 (M7). |
| 5. Accessibility | **2.1** | Pass | Skip links present. ARIA regions good. Dark mode works well. No color-only issues detected on forecast. Keyboard nav untested in depth. Anchor "#" text visible (m11). |
| 6. Mobile | **2.3** | Pass | Good layout transformation at 375px. Balance bar becomes text list. Race cards stack. Hamburger menu works. Minor badge truncation (m9). |
| 7. Election Viz Quality | **1.8** | **FAIL** | Map doesn't show stained-glass types (C4). No population normalization — state-level geographic map. Shift history exists (good). Uncertainty via dotplot (good). But "Types" mode on map is broken. |
| 8. Code Quality | **2.0** | Borderline | Build apparently clean. Race cards are now links (good). Embed hydration errors (M1). Config-driven rendering mostly works but "Mean Dem share: EVEN" bug (C3) suggests formatter issue. |
| **Overall** | **1.96** | **FAIL** | Two categories fail. Critical data accuracy issues must be resolved. |

---

## Top 10 Priorities (ordered)

1. **Fix governor forecasts** — either ship model priors or remove the page until ready (C1)
2. **Fix type "Mean Dem share: EVEN" bug** — formatter/threshold issue in demographics panel (C3)
3. **Fix explore/map Types mode** — show actual super-type colors and legend, not forecast colors (C4)
4. **Review and fix misleading type names** — audit all 100 type names against their demographics (C2)
5. **Add Senate/Governor toggle** on the forecast page (M3)
6. **Fix embed hydration errors** and strip site chrome from embed layout (M1, M2)
7. **Fix methodology accuracy contradictory sentence** (M8)
8. **Fix "View on Map" links** to focus on the relevant entity (M4)
9. **Add favicon** (M7)
10. **Update "Races tracked" count** on methodology page (m3)

---

## What's Working Well

1. **Senate forecast page is solid.** Margins display correctly, race cards link to detail pages, balance bar works, map+panel split is effective.
2. **Race detail pages are excellent.** Quantile dotplots are the signature viz. Poll tables are clean. Electoral types breakdown provides real insight. Forecast blend sliders are a power-user feature done right.
3. **Methodology page is publication-quality.** Clear writing, good structure, honest about limitations. The 8-step narrative is compelling.
4. **Dark mode works well across the site.** No broken cards, good contrast, map colors preserved.
5. **Mobile layout transformations are genuinely different.** Balance bar becomes text summary. Cards stack. Hamburger menu. This isn't just responsive CSS.
6. **Historical shifts page is beautiful.** Small multiples with shared Y-axis, confidence bands, clear super-type labels. Best data viz on the site.
7. **Landing page hero nails the "one big number" principle.** 47D — 53R is immediately comprehensible.
8. **404 page has proper branding and navigation.** Major improvement from default Next.js 404.
9. **Breadcrumbs and cross-linking are mostly complete.** County → type, type → member counties, race → types in state. The architecture is there.
10. **Config-driven rendering works for demographics.** Type and county detail pages auto-display new fields from the API without code changes.

---

## Screenshots

All screenshots saved to `/home/hayden/workspace/audit/`:
- `01-landing-desktop.png` — Landing page (light)
- `02-forecast-senate-desktop.png` — Forecast page (light)
- `03-race-detail-ga-senate.png` — GA Senate race detail
- `04-types-index.png` — Types listing
- `05-type-detail-34.png` — Type 34 detail (full page)
- `06-landing-dark-mode.png` — Landing page (dark)
- `07-forecast-dark-mode.png` — Forecast page (dark)
- `08-landing-mobile-375.png` — Landing page (mobile 375px)
- `09-forecast-mobile-375.png` — Forecast page (mobile 375px)
- `10-explore-shifts.png` — Historical shifts page
- `11-embed-widget.png` — Embed widget
- `12-explore-map.png` — Explore map (Types mode)
