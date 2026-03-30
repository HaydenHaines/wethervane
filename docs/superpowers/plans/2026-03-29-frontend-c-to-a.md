# Frontend C+ to A: Color, Layout, and Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the systemic color/rating bug, add missing landing page components (map + balance bar), center race ticker, and fix explore page — turning a C+ execution into an A.

**Architecture:** The root cause of most visual bugs is that the API returns bare ratings (`"lean"`, `"likely"`, `"safe"`) without directional suffixes (`_d`/`_r`). The frontend palette maps `lean_d`→`#7e9ab5` and `lean_r`→`#c4907a`, but `"lean"` doesn't match any key, so everything falls back to tossup purple. Fix the API, then fix every downstream consumer.

**Tech Stack:** Python (FastAPI API), TypeScript/React (Next.js frontend), Tailwind CSS, deck.gl, visx

---

## Root Cause: Rating Direction Bug

The API function `_margin_to_rating()` in `api/routers/senate.py` returns `"lean"`, `"likely"`, `"safe"` — but the frontend's `RATING_COLORS` expects `"lean_d"`, `"lean_r"`, `"likely_d"`, `"likely_r"`, `"safe_d"`, `"safe_r"`. This one bug cascades to:

- All "lean" badges render as purple (tossup)
- The balance bar is a solid purple bar
- The map colors fall through to beige for lean/likely/safe states
- Race cards show wrong colors everywhere

---

### Task 1: Fix API rating direction (root cause)

**Files:**
- Modify: `api/routers/senate.py:48-60` — `_margin_to_rating()`
- Modify: `api/routers/senate.py:63-65` — `_rating_sort_key()`
- Modify: `api/routers/senate.py:68-100` — `_build_headline()` (uses ratings internally)
- Modify: `api/routers/senate.py:255-258` — `_RATING_COLORS` dict
- Modify: `api/tests/test_senate.py` — update test expectations
- Reference: `web/lib/config/palette.ts` — the canonical rating keys

- [ ] **Step 1: Fix `_margin_to_rating` to return directional ratings**

```python
def _margin_to_rating(margin: float) -> str:
    """Convert signed Dem margin to a directional rating label.

    margin = state_pred - 0.5 (positive = Dem-favored, negative = GOP-favored)
    Returns: tossup | lean_d | lean_r | likely_d | likely_r | safe_d | safe_r
    """
    abs_m = abs(margin)
    if abs_m < _TOSSUP_MAX:
        return "tossup"
    direction = "_d" if margin > 0 else "_r"
    if abs_m < _LEAN_MAX:
        return f"lean{direction}"
    if abs_m < _LIKELY_MAX:
        return f"likely{direction}"
    return f"safe{direction}"
```

- [ ] **Step 2: Fix `_rating_sort_key` for directional ratings**

```python
def _rating_sort_key(rating: str) -> int:
    """Sort races with tossups first, safe last."""
    return {
        "safe_d": 0, "likely_d": 1, "lean_d": 2,
        "tossup": 3,
        "lean_r": 4, "likely_r": 5, "safe_r": 6,
    }.get(rating, 3)
```

- [ ] **Step 3: Fix `_build_headline` to use directional ratings**

In `_build_headline`, the counting logic groups by margin sign, not rating string. Check that the `dem_favored`/`gop_favored` counting still works with directional ratings. The function counts `margin > 0` for dem-favored, so it should be unaffected — but verify.

- [ ] **Step 4: Fix the `_RATING_COLORS` dict in the overview endpoint**

Replace the map colors dict at line ~255:

```python
_RATING_COLORS = {
    "safe_d": "#2d4a6f", "likely_d": "#4b6d90", "lean_d": "#7e9ab5",
    "tossup": "#8a6b8a",   # Purple — NOT gray/beige
    "lean_r": "#c4907a", "likely_r": "#9e5e4e", "safe_r": "#6e3535",
}
```

Also fix `_PARTY_COLORS` for non-contested states. Currently uses `"#b5a995"` (beige) for delegation colors, which looks like "no data". Change to:

```python
_PARTY_COLORS = {
    "D": "#3a5f8a",  # Muted dark blue — clearly "Dem-held, no race"
    "R": "#7a4a4a",  # Muted dark red — clearly "GOP-held, no race"
    "split": "#5a5a5a",  # Dark gray — ambiguous delegation
}
```

- [ ] **Step 5: Update tests**

Update `test_margin_to_rating` to expect directional strings:
```python
def test_tossup():
    assert _margin_to_rating(0.01) == "tossup"
    assert _margin_to_rating(-0.01) == "tossup"

def test_lean():
    assert _margin_to_rating(0.05) == "lean_d"
    assert _margin_to_rating(-0.05) == "lean_r"

def test_likely():
    assert _margin_to_rating(0.10) == "likely_d"
    assert _margin_to_rating(-0.10) == "likely_r"

def test_safe():
    assert _margin_to_rating(0.20) == "safe_d"
    assert _margin_to_rating(-0.20) == "safe_r"
```

Update `TestBuildHeadline` test race fixtures — ratings in test data must now use `"lean_d"`, `"lean_r"`, etc.

- [ ] **Step 6: Run tests**

Run: `uv run pytest api/tests/test_senate.py -v`
Expected: All pass

- [ ] **Step 7: Commit**

```bash
git add api/routers/senate.py api/tests/test_senate.py
git commit -m "fix: return directional ratings (lean_d/lean_r) from API

The API was returning bare ratings (lean, likely, safe) without
directional suffixes. The frontend palette expects lean_d/lean_r etc.
This mismatch caused everything to fall back to tossup purple.

Also fixes map state colors: tossup is now purple (#8a6b8a) not beige,
and non-contested states use muted party colors instead of beige."
```

---

### Task 2: Fix landing page — center ticker, add balance bar, add mini map

**Files:**
- Modify: `web/app/page.tsx` — landing page
- Modify: `web/components/landing/RaceTicker.tsx` — center the flex container
- Modify: `web/components/landing/HeroHeadline.tsx` — color-code the 47D/53R numbers
- Create: `web/components/landing/MiniBalanceBar.tsx` — simplified balance bar for landing
- Create: `web/components/landing/MiniMap.tsx` — non-interactive state map

- [ ] **Step 1: Center the race ticker**

In `web/components/landing/RaceTicker.tsx`, change the ticker container from left-aligned overflow scroll to centered with wrap:

```tsx
// Change this:
<div className="flex gap-3 overflow-x-auto px-4 pb-2">

// To this:
<div className="flex flex-wrap justify-center gap-3 px-4 pb-2">
```

- [ ] **Step 2: Color-code the 47D / 53R hero numbers**

In `web/components/landing/HeroHeadline.tsx` (or wherever the hero is), the `47D` should be in `--forecast-safe-d` blue and `53R` in `--forecast-safe-r` red. Find the current rendering and add `style={{ color: DUSTY_INK.safeD }}` and `style={{ color: DUSTY_INK.safeR }}`.

- [ ] **Step 3: Add a mini balance bar to the landing page**

Create `web/components/landing/MiniBalanceBar.tsx` — a simplified version of `BalanceBar.tsx` that shows all 100 senators as thin segments. Non-contested seats use muted party colors. Contested seats use the directional rating color and are slightly taller (highlighted). The bar should clearly communicate "there are 100 seats, here's the split, and here are the ones in play."

Props: `{ races, demSafeSeats, gopSafeSeats }`. Non-contested D seats = `demSafeSeats` segments in muted blue. Non-contested R seats = `gopSafeSeats` segments in muted red. Contested seats colored by directional rating, placed in the center between safe D and safe R.

Import and render on the landing page between the hero and the race ticker.

- [ ] **Step 4: Add a mini map to the landing page**

Create `web/components/landing/MiniMap.tsx` — a non-interactive simplified US map using the state GeoJSON already at `web/public/states-us.geojson`. Use a basic SVG render (NOT deck.gl — too heavy for a static landing preview). States colored by their forecast rating (from `state_colors` in the senate overview data). Click navigates to `/forecast`.

If SVG rendering of all 51 states is complex, use a simpler approach: a static image generated at build time, or a deck.gl `StaticMap` component with `interactive={false}`.

Import via `next/dynamic` with `ssr: false`. Place between hero and race ticker per spec.

- [ ] **Step 5: Rebuild, deploy, verify**

```bash
cd web && rm -rf .next && npx next build
cp -r public/ .next/standalone/public/ && cp -r .next/static/ .next/standalone/.next/static/
systemctl --user restart wethervane-frontend
```

- [ ] **Step 6: Commit**

```bash
git add web/
git commit -m "feat: landing page polish — centered ticker, balance bar, mini map, color-coded hero"
```

---

### Task 3: Fix explore page scatter plot rendering

**Files:**
- Modify: `web/components/explore/ScatterPlot.tsx` — debug why dots aren't rendering
- Reference: `web/app/explore/types/page.tsx` — known working version

- [ ] **Step 1: Diagnose scatter plot**

The scatter plot shows axis selectors but no dots. Check:
1. Is the SWR hook fetching data? Check browser console for API calls to `/types/scatter-data` or `/types`
2. Is the visx SVG rendering but with wrong dimensions (dots outside viewport)?
3. Is the data shape different from what the component expects?

Compare the working `/explore/types` page version with the `/types` page version.

- [ ] **Step 2: Fix the issue**

Most likely cause: the ScatterPlot component fetches its own data via SWR but the API endpoint it targets doesn't exist or returns a different shape. Fix the data fetching or pass data as props from the parent.

- [ ] **Step 3: Test visually**

Navigate to `/types` in Playwright, wait for SWR fetch, screenshot. Dots should appear colored by super-type.

- [ ] **Step 4: Commit**

---

### Task 4: Fix forecast balance bar to show all 100 seats

**Files:**
- Modify: `web/components/forecast/BalanceBar.tsx` — redesign to show all 100 seats

- [ ] **Step 1: Redesign the balance bar**

Current bar shows only competitive races as equal-width segments. Spec says: "Each segment = one race" with ALL senators represented.

Redesign: 100 thin segments total.
- `demSafeSeats` segments in muted blue (left side)
- Contested D-leaning races in gradient blue shades (lean_d, likely_d)
- Tossup races in purple
- Contested R-leaning races in gradient red shades (lean_r, likely_r)
- `gopSafeSeats` segments in muted red (right side)

Contested seats should be slightly taller (e.g., 36px vs 28px) to highlight them.

Each contested segment is still a clickable button with a tooltip and link.

- [ ] **Step 2: Test visually — the bar should show a clear D-blue|contested|R-red split**

- [ ] **Step 3: Commit**

---

### Task 5: Fix map state colors for no-race states

**Files:**
- Modify: `web/components/map/MapLegend.tsx` — update "No race" legend color

- [ ] **Step 1: After Task 1 API fix, verify map colors**

With the API returning proper directional ratings and non-beige colors for non-contested states, the map should already look better. If states without races still look too light, adjust the `_PARTY_COLORS` values in the API to be darker/more distinct from tossup.

- [ ] **Step 2: Update legend**

The map legend shows "No race" as a color swatch. Update it to show as the muted party delegation color instead of beige, or remove the "No race" entry and instead show "Dem held (no race)" and "GOP held (no race)" as separate entries.

- [ ] **Step 3: Commit**

---

### Task 6: Full suite test + deploy

- [ ] **Step 1: Run all tests**
```bash
uv run pytest --tb=short -q
```
Expected: 2,497+ pass

- [ ] **Step 2: Clean build**
```bash
cd web && rm -rf .next && npx next build
```

- [ ] **Step 3: Deploy**
```bash
cp -r public/ .next/standalone/public/ && cp -r .next/static/ .next/standalone/.next/static/
systemctl --user restart wethervane-frontend && systemctl --user restart wethervane-api
```

- [ ] **Step 4: Full Playwright visual verification**

Screenshot every page: landing, forecast/senate, forecast/governor, race detail, types, type detail, county detail, methodology, methodology/accuracy. Verify:
- Landing: centered ticker, mini balance bar, mini map, color-coded hero
- Forecast: map with proper state colors (dark blue/red for non-contested, gradient for contested, purple for tossup)
- Balance bar: 100 segments with gradient colors, contested highlighted
- Race cards: lean badges show blue/red, not purple
- All other pages: no regressions

- [ ] **Step 5: Commit final state**
