# Dashboard Mode + Merged Explore Tab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dashboard mode toggle (Layout B: map-dominant with overlay panel) alongside the existing content mode (Layout A), and merge the Compare and Explore tabs into a single integrated Explore experience.

**Architecture:** A layout mode toggle switches between Content (A, scrollable page) and Dashboard (B, full-viewport map with overlay panel). Dashboard mode loads ForecastView in a sliding right panel when a state is clicked. The Explore tab merges the scatter plot with type comparison — clicking dots on the scatter populates a comparison table below.

**Tech Stack:** Next.js 14, React 18, Deck.gl 9, Observable Plot, TypeScript

---

## File Structure

### New Files
- `web/components/DashboardOverlay.tsx` — right-side sliding panel for dashboard mode (race detail + controls)
- `web/components/LayoutToggle.tsx` — toggle button between Content and Dashboard modes

### Modified Files
- `web/components/MapContext.tsx` — add layoutMode state
- `web/app/(map)/layout.tsx` — conditionally render split-pane (dashboard) vs stacked (content)
- `web/components/MapShell.tsx` — full-viewport in dashboard mode
- `web/app/(map)/explore/page.tsx` — render merged ExploreView
- `web/components/ShiftExplorer.tsx` — enlarge scatter plot, add click-to-compare
- `web/components/TypeCompareTable.tsx` — integrate as sub-component of Explore
- `web/components/TabBar.tsx` — remove Compare tab, rename Explore

---

## Task 1: Layout Mode Toggle + MapContext

**Files:**
- Create: `web/components/LayoutToggle.tsx`
- Modify: `web/components/MapContext.tsx`

- [ ] **Step 1: Add layoutMode to MapContext**

In `web/components/MapContext.tsx`, add:
```typescript
layoutMode: "content" | "dashboard";
setLayoutMode: (mode: "content" | "dashboard") => void;
```

Default to `"content"`. Persist to localStorage so it survives page reloads.

- [ ] **Step 2: Create LayoutToggle component**

```typescript
// web/components/LayoutToggle.tsx
"use client";
import { useMapContext } from "./MapContext";
import { DUSTY_INK } from "@/lib/colors";

export function LayoutToggle() {
  const { layoutMode, setLayoutMode } = useMapContext();
  return (
    <div style={{
      display: "flex", gap: 0, borderRadius: 4, overflow: "hidden",
      border: `1px solid ${DUSTY_INK.border}`, fontSize: 11,
      fontFamily: "var(--font-sans)",
    }}>
      {(["content", "dashboard"] as const).map((mode) => (
        <button
          key={mode}
          onClick={() => setLayoutMode(mode)}
          style={{
            padding: "4px 10px", border: "none", cursor: "pointer",
            background: layoutMode === mode ? DUSTY_INK.text : DUSTY_INK.cardBg,
            color: layoutMode === mode ? "#fff" : DUSTY_INK.textMuted,
            fontFamily: "inherit", fontSize: "inherit",
          }}
        >
          {mode === "content" ? "Article" : "Dashboard"}
        </button>
      ))}
    </div>
  );
}
```

- [ ] **Step 3: Commit**

```bash
git add web/components/LayoutToggle.tsx web/components/MapContext.tsx
git commit -m "feat: layout mode toggle — content vs dashboard mode"
```

---

## Task 2: Dashboard Layout with Overlay Panel

**Files:**
- Create: `web/components/DashboardOverlay.tsx`
- Modify: `web/app/(map)/layout.tsx`
- Modify: `web/components/MapShell.tsx`

- [ ] **Step 1: Create DashboardOverlay component**

A sliding right panel (35% width) that appears when a state is zoomed in dashboard mode. Contains: race name, rating, margin, poll list, weight sliders, Recalculate button. Essentially a compact version of ForecastView.

```typescript
// web/components/DashboardOverlay.tsx
"use client";
import { useState, useEffect, useCallback } from "react";
import { DUSTY_INK } from "@/lib/colors";
import { useMapContext } from "./MapContext";
import { fetchPolls, feedMultiplePolls, type PollRow, type ForecastRow } from "@/lib/api";

export function DashboardOverlay() {
  const { zoomedState, forecastChoropleth, setForecastChoropleth } = useMapContext();
  const [polls, setPolls] = useState<PollRow[]>([]);
  const [statePred, setStatePred] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const YEAR = "2026";

  // Load polls when state changes
  useEffect(() => {
    if (!zoomedState) { setPolls([]); return; }
    const race = `${YEAR} ${zoomedState} Senate`;
    fetchPolls({ race }).then(setPolls).catch(() => setPolls([]));
  }, [zoomedState]);

  const recalculate = useCallback(async () => {
    if (!zoomedState || polls.length === 0) return;
    setLoading(true);
    try {
      const race = `${YEAR} ${zoomedState} Senate`;
      const result = await feedMultiplePolls({
        cycle: YEAR, state: zoomedState, race,
        section_weights: { model_prior: 1.0, state_polls: 1.0, national_polls: 1.0 },
      });
      // Build choropleth from results
      const typeSum = new Map<number, number>();
      const typeCount = new Map<number, number>();
      const stateRows = result.counties.filter(r => r.state_abbr === zoomedState);
      stateRows.forEach(r => {
        const tid = (r as any).dominant_type;
        if (tid != null && r.pred_dem_share != null) {
          typeSum.set(tid, (typeSum.get(tid) ?? 0) + r.pred_dem_share);
          typeCount.set(tid, (typeCount.get(tid) ?? 0) + 1);
        }
      });
      const choro = new Map<string, number>();
      typeSum.forEach((sum, tid) => choro.set(String(tid), sum / (typeCount.get(tid) ?? 1)));
      setForecastChoropleth(choro);
      setStatePred(stateRows[0]?.state_pred ?? null);
    } finally { setLoading(false); }
  }, [zoomedState, polls, setForecastChoropleth]);

  if (!zoomedState) return null;

  const race = `${YEAR} ${zoomedState} Senate`;
  const marginText = statePred != null
    ? (statePred > 0.5 ? `D+${((statePred - 0.5) * 100).toFixed(1)}` : `R+${((0.5 - statePred) * 100).toFixed(1)}`)
    : "—";

  return (
    <div style={{
      position: "absolute", top: 0, right: 0, bottom: 0, width: "340px",
      background: DUSTY_INK.background, borderLeft: `1px solid ${DUSTY_INK.border}`,
      padding: "16px", overflowY: "auto", zIndex: 5,
      fontFamily: "var(--font-sans)", fontSize: 13,
    }}>
      <h3 style={{ fontFamily: "var(--font-serif)", fontSize: 18, margin: "0 0 4px", color: DUSTY_INK.text }}>
        {zoomedState} Senate
      </h3>
      {statePred != null && (
        <div style={{ fontSize: 14, fontWeight: 600, color: DUSTY_INK.text, marginBottom: 12 }}>
          {marginText}
        </div>
      )}

      {polls.length > 0 && (
        <div style={{ marginBottom: 12 }}>
          <div style={{ fontSize: 11, color: DUSTY_INK.textSubtle, marginBottom: 4 }}>
            Polls ({polls.length})
          </div>
          {polls.slice(0, 5).map((p, i) => (
            <div key={i} style={{ display: "flex", justifyContent: "space-between", fontSize: 11, padding: "2px 0", color: DUSTY_INK.textMuted }}>
              <span>{p.pollster?.slice(0, 20)}</span>
              <span style={{ color: p.dem_share > 0.5 ? DUSTY_INK.safeD : DUSTY_INK.safeR }}>
                {p.dem_share > 0.5 ? `D+${((p.dem_share - 0.5) * 100).toFixed(0)}` : `R+${((0.5 - p.dem_share) * 100).toFixed(0)}`}
              </span>
            </div>
          ))}
        </div>
      )}

      <button
        onClick={recalculate}
        disabled={loading || polls.length === 0}
        style={{
          width: "100%", padding: "8px", borderRadius: 4, border: "none",
          background: polls.length > 0 ? DUSTY_INK.safeD : DUSTY_INK.border,
          color: polls.length > 0 ? "#fff" : DUSTY_INK.textMuted,
          cursor: polls.length > 0 ? "pointer" : "default",
          fontFamily: "var(--font-sans)", fontSize: 12,
        }}
      >
        {loading ? "Calculating..." : `Recalculate${polls.length > 0 ? ` (${polls.length} polls)` : ""}`}
      </button>
    </div>
  );
}
```

- [ ] **Step 2: Update layout.tsx for dashboard mode**

In `web/app/(map)/layout.tsx`, conditionally render:
- **Content mode**: Map (left) + scrollable panel (right) side by side (current layout)
- **Dashboard mode**: Map fills viewport, no right panel (DashboardOverlay renders absolutely positioned inside MapShell's container)

Add the LayoutToggle in the header/nav area.

- [ ] **Step 3: Update MapShell for dashboard mode**

When `layoutMode === "dashboard"`, the map container should fill the viewport. The DashboardOverlay renders inside the map container (position: absolute).

- [ ] **Step 4: Build and verify**

```bash
cd web && npm run build && cp -r public/ .next/standalone/public/ && cp -r .next/static/ .next/standalone/.next/static/
systemctl --user restart wethervane-frontend.service
```

- [ ] **Step 5: Commit**

```bash
git add web/components/DashboardOverlay.tsx web/app/\(map\)/layout.tsx web/components/MapShell.tsx
git commit -m "feat: dashboard mode — full-viewport map with overlay panel"
```

---

## Task 3: Merge Explore + Compare into Single Tab

**Files:**
- Modify: `web/components/ShiftExplorer.tsx` — enlarge plot, add click-to-select
- Modify: `web/app/(map)/explore/page.tsx` — render integrated view
- Modify: `web/components/TabBar.tsx` — remove Compare tab
- Remove: `web/app/(map)/compare/page.tsx` (or redirect to explore)

- [ ] **Step 1: Enlarge scatter plot in ShiftExplorer**

The current plot is ~250x200px. Set minimum height to 400px. Make it responsive (fill available width).

- [ ] **Step 2: Add click-to-select on scatter dots**

When a dot (type) is clicked in the scatter plot, add its type_id to the comparison selection. Use `addToComparison` from MapContext. Highlight selected dots with a ring.

- [ ] **Step 3: Integrate TypeCompareTable below the scatter plot**

In the Explore page, render ShiftExplorer on top and TypeCompareTable below. The compare table auto-populates when types are selected from the scatter plot.

```typescript
// web/app/(map)/explore/page.tsx
"use client";
import { ShiftExplorer } from "@/components/ShiftExplorer";
import { TypeCompareTable } from "@/components/TypeCompareTable";
import { useMapContext } from "@/components/MapContext";

export default function ExplorePage() {
  const { compareTypeIds } = useMapContext();
  return (
    <div>
      <ShiftExplorer />
      {compareTypeIds.length > 0 && (
        <div style={{ borderTop: "1px solid var(--color-border)", marginTop: 16, paddingTop: 16 }}>
          <TypeCompareTable />
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 4: Update TabBar — remove Compare, rename**

In `web/components/TabBar.tsx`, remove the Compare tab entry. The tabs become: Forecast | Explore | About.

If someone navigates to `/compare`, redirect to `/explore`.

- [ ] **Step 5: Build and verify**

```bash
cd web && npm run build && cp -r public/ .next/standalone/public/ && cp -r .next/static/ .next/standalone/.next/static/
systemctl --user restart wethervane-frontend.service
```

- [ ] **Step 6: Commit**

```bash
git add web/components/ShiftExplorer.tsx web/components/TypeCompareTable.tsx web/app/\(map\)/explore/page.tsx web/components/TabBar.tsx web/app/\(map\)/compare/page.tsx
git commit -m "feat: merge Explore + Compare — scatter plot with integrated type comparison"
```

---

## Task 4: Final Polish + Deploy

- [ ] **Step 1: Mobile check — dashboard mode hidden on mobile**

In LayoutToggle, hide the toggle on screens < 768px (dashboard doesn't work on mobile).

- [ ] **Step 2: Full test suite**

```bash
cd /home/hayden/projects/wethervane && uv run pytest tests/ -q --tb=short 2>&1 | tail -5
```

- [ ] **Step 3: Build, deploy, push**

```bash
cd web && npm run build && cp -r public/ .next/standalone/public/ && cp -r .next/static/ .next/standalone/.next/static/
systemctl --user restart wethervane-api.service wethervane-frontend.service
TOKEN=$(gh auth token) && git push "https://$TOKEN@github.com/HaydenHaines/wethervane.git" main
```
