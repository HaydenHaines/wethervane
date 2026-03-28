# Design Spec: Frontend Redesign — Senate Forecast Landing + Progressive Loading + Explore

**Date**: 2026-03-28
**Status**: APPROVED design decisions, awaiting final user review
**Scope**: Transform WetherVane from a research visualization tool into a 538-style public election forecast site.

## Motivation

The current frontend is a modeling experiment UI — blank grey map on load, state/year/election dropdowns, "Recalculate" button required to see anything, defaults to AK Governor. Users see nothing compelling until they manually configure the view. The site needs to answer the public question immediately: "Who's going to win the Senate?"

## Design Decisions (Confirmed)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Default landing view | 2026 Senate national overview | The marquee midterm question |
| Color palette | Dusty Ink (C) — muted, aged atlas aesthetic | Academic authority without partisan garishness |
| Desktop layout | Dual-mode: Content (A) default + Dashboard (B) toggle | A is "correct" for storytelling; B for power users |
| Mobile layout | Content (A) only | Overlay panel doesn't work at small viewports |
| Race scope | Senate on landing; Governor via toggle | Senate control is the collective stakes question |
| State drill-down | Hybrid: dashboard zooms in-page, content expands inline | Respects each layout's interaction model |
| State zoom visual | Desaturate surrounding states on drill-down | Communicates prediction scope is state-specific |
| Map loading | Progressive: states → tracts on state click | Solves 81K tract performance problem |
| Explore + Compare | Merge into single "Explore" tab | Scatter plot selects types, comparison populates below |

## Color Palette: Dusty Ink

```
Safe D:     #2d4a6f
Likely D:   #4b6d90
Lean D:     #7e9ab5
Tossup:     #b5a995
Lean R:     #c4907a
Likely R:   #9e5e4e
Safe R:     #6e3535

Background: #fafaf8
Text:       #3a3632
Muted text: #6e6860
Subtle text:#8a8478
Card bg:    #f5f3ef
Border:     #e0ddd8
Map empty:  #eae7e2
```

Typography: Georgia/serif for headings, system-ui for UI elements and labels.

## Component Architecture

### 1. Senate Forecast Landing (`/forecast` — default route)

**Above the fold (no scroll needed):**
- **Headline**: "Republicans Favored to retain control of the Senate" — dynamically generated from model output. Large serif heading.
- **Senate Control Bar**: Horizontal spectrum, Safe D → Tossup → Safe R. Each segment = one contested race, labeled with state abbreviation. Width proportional to competitiveness (tossup seats wider). Hover shows race detail tooltip. Click navigates to race detail (content mode) or zooms map (dashboard mode).
- **Seat counts**: "Dem seats: 47 + races | 50 for majority | GOP seats: 53 + races" below the bar.

**Below the fold:**
- **State map** (content mode): SVG or lightweight canvas map of US with 51 state polygons. States with 2026 Senate races colored by race rating (Dusty Ink palette). States without Senate races in `#eae7e2` (map empty color). Click a state → expands that race's detail card inline.
- **Race cards**: Grid of cards for each contested Senate race, sorted by competitiveness (tossups first). Each card shows: state name, rating badge (Tossup/Lean/Likely/Safe), margin, poll count, trend indicator. Click → race detail page.

**Dashboard mode (toggle):**
- Senate bar stays fixed at top.
- Map fills the viewport below (deck.gl, state polygons initially).
- Right overlay panel (35% width) shows race detail when a state is clicked.
- Overlay includes: race name, rating, margin, recent polls, weight sliders, Recalculate button.
- Clicking a state: map zooms to state, loads tract polygons for that state only, surrounding states desaturate to ~30% opacity. Back button returns to national view.

**Senate/Governor toggle:**
- Small segmented control above the control bar: "Senate | Governor"
- Switching to Governor replaces the control bar with governor races and recolors the map.
- Governor view has no "control" narrative (no collective stakes), just race-by-race ratings.

### 2. Progressive Map Loading

**Problem**: Loading 81K tract polygons on page load causes multi-second delays and high memory usage.

**Solution**: Three-level progressive loading:

**Level 1 — National (instant):**
- 51 state polygons from a lightweight state GeoJSON (~200KB).
- Colored by race rating using Dusty Ink palette.
- States without a 2026 Senate/Governor race are greyed out (#eae7e2).
- This is what the user sees on page load. No spinner, no delay.

**Level 2 — State zoom (on click):**
- When user clicks a state, fetch that state's tract community polygons from the API or a per-state GeoJSON split.
- Typical state: ~500-3,000 community polygons. Loads in <1s.
- Surrounding states desaturate (opacity 0.3, color shifted toward grey).
- The clicked state renders with full stained glass tract-level detail.

**Level 3 — Tract detail (on tract click):**
- Clicking a community polygon shows a popup with: type name, super-type, demographics, prediction, margin.
- No additional data load — information comes from the already-loaded state GeoJSON properties.

**Data preparation needed:**
- Generate `web/public/states-us.geojson` — 51 state polygons with properties: `state_abbr`, `state_fips`, `has_senate_2026`, `has_governor_2026`, `senate_rating`, `governor_rating`.
- Split the existing `tracts-us.geojson` into per-state files: `web/public/tracts/GA.geojson`, `web/public/tracts/FL.geojson`, etc. Each ~500KB-2MB.
- API endpoint or static JSON for race ratings data that the control bar and map coloring consume.

### 3. State Drill-Down Behavior

**Dashboard mode (Layout B):**
1. User clicks Georgia on national map.
2. Map smoothly animates zoom to Georgia's bounding box.
3. State polygons outside Georgia desaturate (opacity 0.3, color blended toward #eae7e2).
4. Georgia's tract community polygons load and render (stained glass pattern).
5. Right overlay panel slides in showing Georgia Senate race detail.
6. Recalculate button in panel updates tract-level predictions with poll data + weight sliders.
7. "← Back to national" button in panel returns to national view (reverse animation).

**Content mode (Layout A):**
1. User clicks Georgia on the map or the Georgia race card.
2. Map zooms to Georgia, surrounding states desaturate (same visual treatment).
3. Page scrolls down to an expanded Georgia section showing: polls, type breakdown, prediction detail.
4. "Back to all races" link scrolls back up and resets map to national.

**Mobile (always Content mode):**
1. Map is small at top, shows state-level only (no tract loading).
2. Tapping a state scrolls to that race's card below.
3. Race cards are the primary interaction surface on mobile.

### 4. Merged Explore Tab

**Concept**: Scatter plot + type comparison in one integrated interface.

**Layout:**
- **Top section**: Full-width scatter plot of all types. X and Y axis selectable from dropdowns (demographics, electoral shifts, predicted margins). Each dot = one type, sized by tract count, colored by super-type. Larger than current (~400px tall minimum).
- **Bottom section**: Comparison table that populates when user selects types. Click a dot on the scatter plot → that type appears in the comparison. Click a second dot → side-by-side comparison. "Clear selection" resets.
- **Map**: Updates to highlight selected types on the stained glass map (selected types at full opacity, others dimmed).

**Interaction flow:**
1. User opens Explore tab, sees scatter plot of all 130 types.
2. Notices an outlier — clicks it. Comparison panel below shows that type's full profile.
3. Clicks another dot to compare. Side-by-side table appears.
4. Map on the left highlights where those two types exist geographically.

**Axis options (from feature registry):**
- Demographics: pct_white_nh, pct_black, pct_hispanic, pct_asian, pct_ba_plus, median_hh_income, median_age
- Electoral: pres_shift by cycle, turnout shifts, predicted 2026 margin
- Behavior: τ (turnout ratio), δ (choice shift) — unique to WetherVane

### 5. Visual Polish Checklist

**Must-fix for launch:**
- Map loads colored on first render (state-level, no grey blank).
- Favicon (use a weathervane icon).
- Persistent popups cleared on tab navigation.
- Consistent type naming (descriptive names everywhere, not "Super-type 1").
- 90% CI column: either populate or remove from county/tract tables.
- FIPS codes replaced with tract/county names in all tables.

**Typography:**
- Headings: Georgia, serif (already in use).
- Body/UI: system-ui stack.
- Numbers/data: monospace for alignment in tables.

**Spacing:**
- Increase padding on control elements (current feels cramped).
- Race cards need breathing room between them.
- Scatter plot needs larger minimum size.

## API Changes Needed

### New endpoint: `GET /api/v1/senate/overview`

Returns the national Senate forecast summary:

```json
{
  "headline": "Republicans Favored",
  "subtitle": "to retain control of the Senate",
  "dem_seats_safe": 47,
  "gop_seats_safe": 53,
  "races": [
    {
      "state": "GA",
      "race": "2026 GA Senate",
      "slug": "2026-ga-senate",
      "rating": "tossup",
      "margin": -0.6,
      "n_polls": 15,
      "dem_candidate": null,
      "gop_candidate": null
    }
  ],
  "dem_win_probability": 0.35,
  "most_likely_split": "48D-52R"
}
```

Race `rating` is derived from margin: Safe (>15), Likely (8-15), Lean (3-8), Tossup (<3).

### New endpoint: `GET /api/v1/tracts/{state_abbr}`

Returns tract community polygons for a single state (for progressive loading). Could also be served as static GeoJSON files.

### Existing endpoints

- `GET /api/v1/forecast/races` — already works
- `POST /api/v1/forecast/polls` — already works with tract data
- `GET /api/v1/forecast/race/{race}` — needs update to serve tract-level predictions

## Non-Goals

- Election night live results (future)
- User accounts or saved forecasts
- Custom scenario builder ("what if Dems win GA but lose NC")
- House forecasts (depends on tract→district crosswalk)
- Candidate-specific information (photos, bios)
- Social media sharing cards (already have og:image, sufficient for now)

## Risks

- **Performance**: Even with progressive loading, per-state GeoJSON could be 2MB+ for large states (TX, CA). May need further simplification or vector tiles eventually.
- **Race rating thresholds**: The Safe/Likely/Lean/Tossup boundaries are somewhat arbitrary. Need to document and be consistent.
- **Scope**: This is a large frontend redesign. May need to ship in phases: Phase 1 (Senate landing + progressive loading), Phase 2 (dashboard mode + Explore merge).
