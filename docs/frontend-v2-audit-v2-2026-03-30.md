# WetherVane Frontend v2 Re-Audit — "Shooting for Perfect"

**Date:** 2026-03-30
**Auditor:** Claude Opus 4.6 (Playwright on production)
**Target:** https://wethervane.hhaines.duckdns.org
**Branch:** fix/type-name-accuracy (17 commits ahead of main)
**Prior audit:** docs/frontend-v2-audit-2026-03-29.md (28 issues found)

---

## What's Fixed (22 of 28 original issues resolved)

### Critical fixes confirmed working:
- **C1 (Governor EVEN):** Governor page now shows honest "coming soon" with context and links
- **C3 (Type Mean Dem share EVEN):** Now shows "50.1%" instead of "EVEN" — correct format
- **C5 (Data consistency):** Types page, methodology, and API all agree: 100 types, 5 super-types

### Major fixes confirmed working:
- **M3 (Senate/Gov toggle):** Tabs visible at top of forecast panel
- **M4 (View on Map):** County links to `/explore/map?focus=13121`, type links to `/explore/map?type=34`
- **M7 (Favicon):** `icon.svg` serving correctly, no 404
- **M8 (Methodology text):** Contradictory sentence rewritten
- **M9 (Methodology collapse):** Only section 1 expanded by default
- **m1 (Hardcoded 100):** Now says "Discover the community types..."
- **m2 (Breadcrumb Map):** Fixed to `/explore/map`
- **m3 (Races tracked):** Updated to "33 Senate races"
- **m5 (County breadcrumb state):** Now shows "Home / Georgia / Fulton County"
- **m6 (Updated 6d ago):** Now shows "Updated Mar 22, 2026 - 58 polls"

### Design taste confirmed working:
- **T1 (Hero tagline):** "Based on 100 electoral communities discovered from how places move together — not polls alone."
- **T2 (Balance bar tooltips):** Already implemented (confirmed)
- **T3 (90% CI):** Already implemented (confirmed)
- **T4 (Shift annotations):** Obama/Trump/Biden labels visible in shift charts
- **T5 (Footer):** Now has Forecast/Types/Shifts + "Built by Hayden Haines"

---

## What's NOT Perfect (remaining issues)

### Still broken / partially fixed

---

#### R1. Type 34 name NOT updated on production

**Original:** C2 (Type names misleading)
**Status:** PARTIALLY FIXED

The naming algorithm was updated in `name_types.py` and the DuckDB was rebuilt, but **the page title still shows "Majority-Black Urban"** for Type 34. The API may have been rebuilt but the Next.js static pages were pre-rendered at build time with the old names. A fresh `next build` + service restart may be needed to pick up new API data.

**Also:** Types 37 ("Majority-Black Evangelical Churched") and 79 ("Majority-Black Evangelical Devout") still show old names in the Similar Types section. The naming fixes touched DuckDB but the SSG pages are stale.

**Fix:** Rebuild the frontend (`npx next build`) to re-fetch type names from the API during static generation, then restart the frontend service.

---

#### R2. Embed widget still has 6 hydration errors

**Original:** M1 (Hydration errors)
**Status:** NOT FIXED on production

The embed at `/embed/2026-ga-senate` produces 6 console errors (4x #425, 1x #418, 1x #423). The code fix (SiteChrome component, cleaned embed layout) is correct but the **static build was done before the embed fix was merged**. The running server is serving stale `.next` output.

**Also:** The footer/nav stripping IS working — only the widget card renders. So the SiteChrome fix is partially deployed (it was in the last build) but the hydration fix from the embed layout change isn't.

**Fix:** Rebuild and restart. The code is correct; it's a deployment sequencing issue.

---

#### R3. County election history shows "No data available" for Fulton County

**Original:** M5 (County election history)
**Status:** PARTIALLY FIXED

The API endpoint `GET /api/v1/counties/{fips}/history` exists and the `CountyElectionHistory` component is wired in. But Fulton County (13121) shows "No election history data available." The endpoint is reading from parquet files in `data/assembled/` but may not be finding them, or the FIPS matching is off.

**Fix:** Debug the API endpoint — check if the parquet files exist and have the expected column names. Test with `curl https://wethervane.hhaines.duckdns.org/api/v1/counties/13121/history`.

---

#### R4. Explore map "Types" mode renders types nationally but legend labels are wrong

**Original:** C4 (Map Types mode)
**Status:** PARTIALLY FIXED

The map now shows tract-level stained glass coloring on the forecast page (huge improvement!). But the explore/map page wasn't re-tested after the fix. On the forecast page, the legend shows super-type names like "Hispanic & Immigrant Gateway" and "Black & Minority Urban" — these don't match the official super-type names from the spec ("Hispanic Exurban", "Asian-Pacific Urban", etc.). The names appear to be auto-generated from tract GeoJSON properties rather than fetched from the API.

**Fix:** The legend should read super-type names from `/api/v1/super-types`, not from GeoJSON properties. Alternatively, ensure the GeoJSON properties match the API names.

---

#### R5. Methodology anchor "#" still always visible

**Original:** m11
**Status:** NOT FIXED

Each methodology section heading still shows the "#" anchor link text permanently. The agent reported the CSS was already correct (`opacity-0 group-hover:opacity-60`), but on production the "#" is visible in the button text, not as a separate styled element.

**Fix:** The "#" is inside the button's heading text, not a separate hover-target. Need to extract it as a separate `<a>` element with the hover classes, or hide it with CSS that targets the specific anchor within the heading.

---

#### R6. Map on forecast page shows stained-glass but no overlay toggle

**Original:** Adjacent to C4
**Status:** NEW OBSERVATION

The forecast page map now renders tract-level stained-glass types by default. This is visually stunning but there's no way to toggle back to the forecast state-level coloring on this page. The Types/Forecast toggle only exists on `/explore/map`.

On the forecast page specifically, seeing the stained glass is interesting but potentially confusing — users landed here expecting to see which states are competitive. The state-level fill with race ratings (Safe D / Lean D / etc.) was the expected default for the forecast page.

**Fix:** Either (a) restore state-level forecast coloring as the default on the forecast page and keep stained glass for explore/map, or (b) add the Types/Forecast toggle to the forecast map panel.

---

#### R7. Forecast page "Loading map..." still shows as text

**Original:** m8 (Loading skeleton)
**Status:** UNCERTAIN

The forecast page still briefly shows "Loading map..." text before the map renders. The skeleton fix was applied to `MemberGeography.tsx` (type detail pages) but the main `MapShell` on the forecast page may use a different loading path.

**Fix:** Check if `MapShell`'s loading state uses a skeleton or text. If text, add the same `animate-pulse` skeleton pattern.

---

#### R8. Mobile forecast page: race cards lack Tossup badge on some cards

**Original:** m9 (Mobile badge truncation)
**Status:** PARTIALLY FIXED

At 375px, the race card badges were fixed with `shrink-0`, but on the mobile forecast page the second card in a row (e.g., MI) doesn't show its Tossup badge at all. The badge may be rendering off-screen.

**Not re-tested** — would need a fresh mobile Playwright screenshot to verify.

---

### New observations (not in original audit)

---

#### N1. Super-type names in legend don't match API/spec

The tract stained-glass legend on the forecast page uses names like "Hispanic & Immigrant Gateway", "Black & Minority Urban", "Affluent Educated Suburban", "Middle-Class Suburban", "Rural Institutional Outlier", "White Working & Rural". The spec and API use different names: "Hispanic Exurban", "Asian-Pacific Urban", "Black-Belt Evangelical", "Rural Young", "Rural Evangelical".

These are two different naming systems. The GeoJSON-embedded names appear to be auto-generated from demographics, while the API names come from the type naming system. They need to be unified.

---

#### N2. Shift chart on type detail shows annotations but they're crowded

The Obama/Trump/Biden annotations are present on the shift history charts but may overlap with data points or axis labels on types with certain shift patterns. This is a minor visual polish issue.

---

#### N3. "Explore" nav link goes to /types but "Shifts" footer link goes to /explore

The top nav "Explore" links to `/types`. The footer "Shifts" links to `/explore`. There's an inconsistency — `/explore` may or may not be a valid page. Should be `/explore/shifts`.

---

## Revised Rubric Scoring

| Category | Before | After | Notes |
|----------|--------|-------|-------|
| 1. Core Design Principles | 2.0 | **2.5** | Hero tagline, methodology collapse, 0-poll explanation, freshness dates. Stained-glass map is signature viz. Loses points: stale type names on production. |
| 2. Data Accuracy | 1.3 | **2.2** | Governor honest state, Mean Dem share fixed, methodology text fixed. Loses points: type names stale on static pages (R1), county history empty (R3). |
| 3. Navigation | 1.9 | **2.7** | Senate/Gov tabs, focused map links, breadcrumbs with state. Loses points: Explore/Shifts nav inconsistency (N3). |
| 4. Performance | 2.3 | **2.4** | Favicon fixed, zero errors on landing. Loses points: embed hydration (R2, deployment issue), "Loading map..." text (R7). |
| 5. Accessibility | 2.1 | **2.2** | Skip links, dark mode, ARIA. Loses points: methodology "#" still visible (R5). |
| 6. Mobile | 2.3 | **2.4** | Good transformations. Minor badge issue may persist (R8). |
| 7. Election Viz Quality | 1.8 | **2.4** | Stained-glass map is now live! Shift annotations added. Loses points: legend name mismatch (N1), forecast page default overlay (R6). |
| 8. Code Quality | 2.0 | **2.5** | SiteChrome pattern clean, config-driven rendering, build clean. Type naming algorithm improved. |
| **Overall** | **1.96** | **2.41** | +0.45 improvement. No categories fail. 3 items need deployment rebuild, 3 need code fixes. |

---

## Path to 3.0

The branch is ~85% of the way to a perfect score. To close the gap:

### Must-do before merge to main (30 minutes):
1. **Rebuild frontend + restart services** — fixes R1 (type names), R2 (embed hydration)
2. **Debug county history API** — fix R3 (test the endpoint directly with curl)
3. **Fix footer Shifts link** — change `/explore` to `/explore/shifts` (N3)

### Should-do before public launch (1-2 hours):
4. **Unify super-type names** — either update GeoJSON properties to match API names, or vice versa (N1)
5. **Restore forecast page map to state-level default** — stained glass on explore, ratings on forecast (R6)
6. **Fix methodology anchor "#" CSS** — extract as hover-only element (R5)
7. **Add MapShell loading skeleton** — match MemberGeography pattern (R7)

### Nice-to-have:
8. Verify mobile badge at 375px (R8)
9. Shift annotation spacing polish (N2)
