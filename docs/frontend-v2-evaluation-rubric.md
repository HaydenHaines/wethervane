# WetherVane Frontend v2 Evaluation Rubric

**Purpose:** Structured criteria for evaluating whether the frontend redesign achieved its goals. Derived from the v2 design spec (10 governing principles), the March 28 Playwright audit (29 issues found), and current data visualization / election viz / web performance best practices.

**Scoring:** Each criterion is rated 0-3:
- **0** = Not met / broken
- **1** = Partially met, significant gaps
- **2** = Met with minor issues
- **3** = Fully met or exceeds expectations

**Passing threshold:** Category average >= 2.0 across all categories. No category below 1.5. No individual critical criterion scored 0.

---

## Category 1: Core Design Principles (from spec)

These map directly to the 10 governing principles in the design spec. They are the project's own success criteria.

| # | Criterion | What to check | Weight |
|---|-----------|---------------|--------|
| 1.1 | **Uncertainty is the story (P1)** | Every forecast shows a distribution or CI, not just a point estimate. Quantile dotplots on race detail pages. Confidence bands on poll trackers. | Critical |
| 1.2 | **Progressive disclosure (P2)** | Landing page answers "who's ahead?" Race detail answers "how confident?" Type detail answers "why?" Each layer is reachable without dead ends. | Critical |
| 1.3 | **One Big Number anchor (P3)** | Every forecast page leads with a dominant metric in large type. Race detail has the margin front and center. Senate overview leads with seat count. | High |
| 1.4 | **Context beats raw numbers (P4)** | Historical comparisons available (shift history, election history). Every margin has a rating badge. Relative positioning shown. | High |
| 1.5 | **Configuration over hardcoding (P5)** | Zero hardcoded model parameters in JSX. New API fields auto-display via config system. Model retrain requires zero frontend code changes. Verify: add a fake field to API response and confirm it renders. | Critical |
| 1.6 | **Mobile is a different product (P6)** | Maps become cards, tables become lists, charts simplify at <768px. Not just CSS media queries. Balance bar has text fallback. Dotplot reduces to 50 dots. Touch targets >= 44px. | Critical |
| 1.7 | **Speed creates trust (P7)** | First meaningful paint < 1s. Skeleton screens during all data fetches. No bare "Loading..." text. Progressive loading of geographic detail (state → tract). | High |
| 1.8 | **Show your work (P8)** | Methodology is a first-class section accessible from global nav. Scrollytelling narrative covers 8 steps. Accuracy page with metrics. | High |
| 1.9 | **Data provenance on every display (P9)** | Every number answers: when was this updated? How many sources? How confident? Freshness timestamp on landing page. Poll counts on race cards. | Medium |
| 1.10 | **No tech debt from day one (P10)** | Every component has loading, error, and populated states. All colors tokenized. All values config-driven. TypeScript strict mode with no `any`. | High |

**Category pass:** Average >= 2.0, no critical criterion at 0.

---

## Category 2: Data Accuracy & Consistency

The audit found data display bugs were the most trust-destroying issues. A political forecast site with wrong numbers is dead on arrival.

| # | Criterion | What to check | Weight |
|---|-----------|---------------|--------|
| 2.1 | **Margins display correctly** | All race margins match API values. No "EVEN" bug. Verify 5+ races against raw API response. | Critical |
| 2.2 | **Rating badges match margins** | Rating colors (Safe/Likely/Lean/Tossup) are directional and consistent between map, cards, and detail pages. | Critical |
| 2.3 | **Model parameter consistency** | Type count, super-type count, county/tract count are consistent across About, Methodology, Types listing, and API `/health`. No stale numbers. | Critical |
| 2.4 | **Type names are non-misleading** | No partisan suffixes (Blue/Red) that contradict individual county/tract data. Descriptive names only. | Critical |
| 2.5 | **Cross-page data agreement** | Same race shows same margin on forecast overview, race detail, and embed widget. Same type shows same demographics on type detail and county detail. | High |
| 2.6 | **Map colors match data** | Map choropleth coloring corresponds to actual margins/ratings. State fill matches the race card rating. | High |

**Category pass:** Average >= 2.5. All critical criteria >= 2.

---

## Category 3: Navigation & Information Architecture

Users must be able to find what they're looking for without dead ends.

| # | Criterion | What to check | Weight |
|---|-----------|---------------|--------|
| 3.1 | **Forecast to race detail** | Clicking a race card on the forecast page navigates to `/forecast/[slug]`. No dead-end buttons. | Critical |
| 3.2 | **Governor races accessible** | Governor races reachable from the forecast page (tab, toggle, or separate section). | High |
| 3.3 | **Cross-linking completeness** | County → type detail. Type → member counties on map. Race → types in state. Methodology → accuracy. Every entity has outbound links. | High |
| 3.4 | **Breadcrumb correctness** | Breadcrumbs reflect actual hierarchy. No synonymous links (Home vs Map). Back navigation works. | Medium |
| 3.5 | **Global navigation** | All major sections (Forecast, Explore, Methodology, About) reachable from every page. Logo/brand links home. Footer with secondary links. | High |
| 3.6 | **404 experience** | Custom 404 page with branding, navigation links, and helpful suggestions. Not default Next.js. | Medium |
| 3.7 | **Deep link stability** | Direct URL access to `/forecast/2026-ga-senate`, `/type/42`, `/county/13121` works without navigation. Shareable URLs. | High |

**Category pass:** Average >= 2.0. Critical criterion >= 2.

---

## Category 4: Performance

Data visualization sites live or die by load time. Heavy maps and chart libraries are the primary risk.

| # | Criterion | What to check | Tool |Weight |
|---|-----------|---------------|------|-------|
| 4.1 | **LCP < 2.5s** | Largest Contentful Paint on landing, forecast, race detail (desktop) | Lighthouse | Critical |
| 4.2 | **INP < 200ms** | Interaction to Next Paint when clicking race cards, toggling tabs, filtering | Lighthouse / manual | High |
| 4.3 | **CLS < 0.1** | No layout shifts from late-loading charts, legends, or dynamic filters | Lighthouse | High |
| 4.4 | **Mobile TTI < 3s** | Time to interactive on mid-range device (throttled 4G) for landing page | Lighthouse mobile | High |
| 4.5 | **Lighthouse Performance >= 90** | Overall Lighthouse performance score for landing page | Lighthouse | Medium |
| 4.6 | **Progressive map loading** | National view loads states instantly; tract polygons load on state click, not upfront | Manual observation | High |
| 4.7 | **Skeleton screens** | Every async component shows a skeleton (not "Loading..." text) during data fetch | Manual / Playwright slow network | Medium |
| 4.8 | **JS bundle size** | Total compressed JS < 300KB for initial page load (stretch: < 200KB) | Build output / Webpack analyzer | Medium |
| 4.9 | **No console errors** | Zero JS errors in console on any page (no hydration mismatches, no 404s for assets) | Browser DevTools | Medium |

**Category pass:** Average >= 2.0. LCP criterion >= 2.

---

## Category 5: Accessibility (WCAG 2.1 AA)

A public-facing political information site has a responsibility to be accessible.

| # | Criterion | What to check | Tool | Weight |
|---|-----------|---------------|------|--------|
| 5.1 | **Text contrast >= 4.5:1** | All text passes AA contrast ratio in both light and dark themes | axe-core / Lighthouse | Critical |
| 5.2 | **Chart element contrast >= 3:1** | Non-text elements (bars, dots, map fills) have sufficient contrast against background | Manual | High |
| 5.3 | **No color-only encoding** | No chart uses color as the sole differentiator. Patterns, labels, or shapes supplement. Map legend has text labels. | Manual | High |
| 5.4 | **Keyboard navigation** | All interactive elements (map states, race cards, tabs, sliders) reachable and operable via keyboard | Manual | High |
| 5.5 | **Alt text on visualizations** | Charts have descriptive alt text or hidden data tables. SVGs have `<title>` and `<desc>`. | Automated scan | Medium |
| 5.6 | **Dark mode cards/surfaces** | No light-background cards in dark mode. All surfaces respect theme. | Manual | Medium |
| 5.7 | **Reduced motion** | `prefers-reduced-motion` disables animations. Scrollytelling becomes static. | Manual | Medium |
| 5.8 | **Lighthouse Accessibility >= 90** | Overall Lighthouse accessibility score | Lighthouse | High |
| 5.9 | **Skip-to-content link** | Present and functional | Manual | Low |

**Category pass:** Average >= 2.0. Critical criterion >= 2.

---

## Category 6: Mobile Experience

Per spec P6, mobile is a different product, not a squeezed desktop.

| # | Criterion | What to check | Weight |
|---|-----------|---------------|--------|
| 6.1 | **Layout transformation** | At 375px: maps stack above content, tables become lists, grids become carousels or stacked cards. Not just scaled-down desktop. | Critical |
| 6.2 | **Touch targets >= 48px** | All tappable elements meet minimum size. No tiny close buttons or cramped links. | High |
| 6.3 | **No gesture conflicts** | Chart pinch-zoom doesn't hijack page scroll. No scroll hijacking anywhere. | High |
| 6.4 | **Chart simplification** | Dotplot reduces dots, line charts reduce series, tables reduce columns. Data density appropriate for small screens. | High |
| 6.5 | **Readability at 375px** | All text >= 12px. No truncated labels on critical information. No horizontal scroll on content. | High |
| 6.6 | **Thumb zone controls** | Primary interactive controls (filters, toggles) in bottom 40% of viewport. | Medium |
| 6.7 | **Balance bar mobile fallback** | Text summary ("47D - 53R") replaces interactive segments on mobile. | Medium |

**Category pass:** Average >= 2.0. Critical criterion >= 2.

---

## Category 7: Election Data Visualization Quality

Criteria specific to political/election data presentation, drawn from Senaratna's 10 principles and peer site analysis (FiveThirtyEight, The Economist, NYT).

| # | Criterion | What to check | Weight |
|---|-----------|---------------|--------|
| 7.1 | **Population-normalized geography** | Maps weight by votes/population, not land area. Small rural counties don't dominate visual. Either cartogram, hex map, or balanced visual treatment. | High |
| 7.2 | **Party color discipline** | Partisan colors (blue/red) used exclusively for party-affiliated data. Non-party elements (demographics, methodology) use neutral palette. | High |
| 7.3 | **Historical context available** | At least one prior election comparison accessible per race and per type. Shift history or trend charts. | High |
| 7.4 | **Interactive exploration depth** | >= 2 filter/drill-down dimensions available (geography, time, type, demographic). Users can explore on their terms. | High |
| 7.5 | **Meaningful aggregation** | Data presented at the level elections are decided (state for Senate/Governor). Raw tract data accessible but not the default view. | Medium |
| 7.6 | **Minimal clutter** | <= 7 distinct visual elements per view. No unnecessary widgets. White space is used intentionally. | Medium |
| 7.7 | **Uncertainty communication** | Forecasts always show range, not just point estimate. Language reflects uncertainty ("favored" not "will win"). Quantile dotplots are the signature visualization. | Critical |
| 7.8 | **Methodology transparency** | Methodology is detailed, accessible, written for educated non-experts. Model assumptions stated. Accuracy metrics published. | High |

**Category pass:** Average >= 2.0. Critical criterion >= 2.

---

## Category 8: Code & Architecture Quality

Technical health of the codebase — maintainability, testability, correctness.

| # | Criterion | What to check | Weight |
|---|-----------|---------------|--------|
| 8.1 | **TypeScript strict, no `any`** | `strict: true` in tsconfig. `grep -r ": any"` returns zero hits in app code (libraries excluded). | High |
| 8.2 | **Component size <= 200 lines** | No component file exceeds 200 lines (per spec). Check with `wc -l`. | Medium |
| 8.3 | **SWR for all data fetching** | No raw `fetch()` or `useEffect` for data loading. All API calls go through SWR hooks. | High |
| 8.4 | **Config-driven rendering** | `display.ts`, `palette.ts`, `methodology.ts` drive all display logic. No inline color values, no hardcoded field labels. | Critical |
| 8.5 | **Error boundaries everywhere** | Every data-consuming section wrapped in error boundary with fallback UI. Verify by killing API and checking graceful degradation. | High |
| 8.6 | **Build clean** | `next build` produces zero TypeScript errors and zero warnings. | High |
| 8.7 | **Test coverage** | Playwright e2e tests cover critical user flows (landing → forecast → race detail → back). >= 30 e2e tests. | High |
| 8.8 | **No console errors in production build** | Zero hydration errors, no 404s for assets, no React warnings in `next start`. | Medium |

**Category pass:** Average >= 2.0. Critical criterion >= 2.

---

## Category 9: Post-Launch Engagement (measure after 2 weeks live)

These can't be evaluated pre-launch but should be tracked to determine long-term success.

| # | Criterion | What to check | Tool | Target |
|---|-----------|---------------|------|--------|
| 9.1 | **Engagement rate** | % of sessions with meaningful interaction | GA4 | >= 60% |
| 9.2 | **Bounce rate** | % of single-page sessions | GA4 | < 48% |
| 9.3 | **Avg session duration** | Time spent per visit | GA4 | >= 2 min |
| 9.4 | **Chart interaction rate** | % of sessions where user interacts with a viz (click, hover, filter) | Custom events | >= 30% |
| 9.5 | **Pages per session** | Depth of exploration | GA4 | >= 2 |
| 9.6 | **Return visitor rate** | % of visitors who come back within 14 days | GA4 | >= 25% |
| 9.7 | **Race detail page visits** | % of sessions that reach a race detail page (proxy for progressive disclosure working) | GA4 | >= 40% |
| 9.8 | **Methodology page visits** | % of sessions that visit methodology (proxy for trust/transparency) | GA4 | >= 10% |

**Category pass:** Meet >= 6 of 8 targets after 2 weeks of traffic.

---

## Evaluation Procedure

### Pre-launch (categories 1-8)

1. **Automated sweep**: Run Lighthouse on 5 representative pages (landing, forecast overview, race detail, type detail, methodology) in both desktop and mobile modes. Record Performance, Accessibility, and Best Practices scores.

2. **Data accuracy spot check**: Pick 10 races. For each, compare the margin shown on the forecast page, the race detail page, and the raw API response. All three must match.

3. **Config-driven test**: Temporarily add a fake demographic field to one API endpoint. Confirm it auto-renders on the frontend without code changes. Remove it.

4. **Mobile walkthrough**: On a 375px viewport (or real phone), complete the flow: landing → forecast → race detail → type detail → methodology. Note any dead ends, truncated text, or touch target failures.

5. **Keyboard-only navigation**: Complete the same flow using only Tab, Enter, and arrow keys. Every interactive element must be reachable.

6. **Dark mode audit**: Switch to dark theme. Check every page for light-background cards, low-contrast text, or missing theme tokens.

7. **Cross-page consistency**: Verify type count, super-type count, and model metrics match across About, Methodology, Types listing, and API `/health`.

8. **Build & test verification**: `next build` clean. All Playwright tests pass. All Python tests pass.

### Post-launch (category 9)

9. Set up GA4 (or Plausible/Umami for privacy) with custom events for chart interactions, filter usage, and page transitions.

10. Review engagement metrics at 2 weeks and 4 weeks. Compare against targets.

---

## Scorecard Template

| Category | Score (0-3) | Pass? |
|----------|-------------|-------|
| 1. Core Design Principles | ___ / 3.0 | >= 2.0, no critical at 0 |
| 2. Data Accuracy | ___ / 3.0 | >= 2.5, all critical >= 2 |
| 3. Navigation | ___ / 3.0 | >= 2.0, critical >= 2 |
| 4. Performance | ___ / 3.0 | >= 2.0, LCP >= 2 |
| 5. Accessibility | ___ / 3.0 | >= 2.0, critical >= 2 |
| 6. Mobile | ___ / 3.0 | >= 2.0, critical >= 2 |
| 7. Election Viz Quality | ___ / 3.0 | >= 2.0, critical >= 2 |
| 8. Code Quality | ___ / 3.0 | >= 2.0, critical >= 2 |
| 9. Engagement (post-launch) | ___ / 8 targets | >= 6 of 8 |
| **Overall** | **___ / 3.0** | **All categories pass** |

**Ship decision:** All categories 1-8 pass → ship. Any category fails → fix critical gaps first.
