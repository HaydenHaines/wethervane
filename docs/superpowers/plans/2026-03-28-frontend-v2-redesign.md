# WetherVane Frontend V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Spec:** `docs/superpowers/specs/2026-03-28-frontend-v2-redesign-design.md`
> **Audit:** `docs/frontend-audit-2026-03-28.md`
> **Research:** `research/political-dataviz-research.md`

**Goal:** Rebuild the WetherVane frontend from a functional MVP into a research-grade political data visualization platform — the academic dream of election forecasting UX.

**Architecture:** Clean build on a feature branch (`feat/frontend-v2`). Next.js 14 App Router with shadcn/ui + Tailwind v4 for components, visx for custom charts, SWR for data fetching, Motion for animation, react-scrollama for methodology narrative. deck.gl retained for maps. Configuration-driven rendering — zero hardcoded model parameters.

**Tech Stack:** Next.js 14, React 18, TypeScript strict, Tailwind v4, shadcn/ui, Radix UI, visx, SWR, Motion (Framer Motion), react-scrollama, deck.gl v9, TanStack Table, Playwright

---

## Phase Overview

| Phase | Tasks | What It Delivers |
|-------|-------|-----------------|
| **0: Branch & Foundation** | 1–4 | Feature branch, Tailwind + shadcn init, config system, SWR hooks |
| **1: Landing Page** | 5–6 | Hero, race ticker, entry points, footer |
| **2: Forecast Hub** | 7–11 | Senate overview, governor overview, balance bar, race cards, all-races table |
| **3: Race Detail** | 12–16 | Hero section, quantile dotplot, poll tracker, types breakdown, poll table |
| **4: Explore** | 17–20 | Type directory, scatter plot, full-screen map, comparison table |
| **5: Detail Pages** | 21–24 | Type detail, county detail, election history charts, cross-links |
| **6: Methodology** | 25–27 | Scrollytelling, accuracy page, step configuration |
| **7: Navigation & Layout** | 28–30 | Global nav, footer, breadcrumbs, 404 page, embed fix |
| **8: Mobile** | 31–33 | Mobile transformations, touch rules, responsive charts |
| **9: Polish & Testing** | 34–37 | Accessibility audit, Playwright e2e, visual regression, performance optimization |

---

## File Structure (New)

All new files live under `web/`. The old component files remain on `main` — this branch replaces them entirely.

```
web/
├── app/
│   ├── layout.tsx                          # Root layout (Tailwind, theme, nav, footer)
│   ├── page.tsx                            # Landing page (hero, ticker, entry points)
│   ├── globals.css                         # Tailwind directives + CSS variables
│   ├── forecast/
│   │   ├── page.tsx                        # Redirect to /forecast/senate
│   │   ├── layout.tsx                      # Forecast layout (map + panel)
│   │   ├── senate/page.tsx                 # Senate overview
│   │   ├── governor/page.tsx               # Governor overview
│   │   └── [slug]/page.tsx                 # Race detail (SSR)
│   ├── explore/
│   │   ├── page.tsx                        # Redirect to /explore/types
│   │   ├── types/page.tsx                  # Type directory + scatter + compare
│   │   ├── map/page.tsx                    # Full-screen stained glass map
│   │   └── shifts/page.tsx                 # Historical shift analysis
│   ├── county/[fips]/page.tsx              # County detail (SSR)
│   ├── type/[id]/page.tsx                  # Type detail (SSR)
│   ├── methodology/
│   │   ├── page.tsx                        # Scrollytelling methodology
│   │   └── accuracy/page.tsx               # Accuracy deep-dive
│   ├── embed/
│   │   ├── layout.tsx                      # Minimal embed layout
│   │   └── [slug]/page.tsx                 # Embed widget
│   ├── about/page.tsx                      # About + attribution
│   ├── not-found.tsx                       # Custom 404
│   ├── robots.ts                           # Robots (keep)
│   ├── sitemap.ts                          # Sitemap (update routes)
│   └── feed.xml/                           # RSS (keep)
├── components/
│   ├── ui/                                 # shadcn components (auto-generated)
│   ├── nav/
│   │   ├── GlobalNav.tsx                   # Sticky top nav
│   │   ├── Footer.tsx                      # Global footer
│   │   └── Breadcrumbs.tsx                 # Config-driven breadcrumbs
│   ├── landing/
│   │   ├── HeroSection.tsx                 # Big number + headline
│   │   ├── RaceTicker.tsx                  # Competitive races horizontal strip
│   │   └── EntryPoints.tsx                 # Three CTA cards
│   ├── forecast/
│   │   ├── BalanceBar.tsx                  # Senate/governor seat bar
│   │   ├── RaceCard.tsx                    # Individual race card
│   │   ├── RaceCardGrid.tsx                # Key races grid
│   │   ├── AllRacesTable.tsx               # TanStack sortable table
│   │   ├── RaceHero.tsx                    # Race detail hero section
│   │   ├── QuantileDotplot.tsx             # visx uncertainty viz
│   │   ├── PollTracker.tsx                 # visx area chart + confidence band
│   │   ├── TypesBreakdown.tsx              # Electoral types in state
│   │   ├── PollTable.tsx                   # Race polls TanStack table
│   │   └── SectionWeightSliders.tsx        # Model weight controls
│   ├── explore/
│   │   ├── TypeGrid.tsx                    # Type card directory
│   │   ├── TypeCard.tsx                    # Individual type card
│   │   ├── ScatterPlot.tsx                 # visx scatter with axis selectors
│   │   ├── ComparisonTable.tsx             # TanStack 4-column compare
│   │   ├── ShiftSmallMultiples.tsx         # visx small multiples grid
│   │   └── MapOverlayToggle.tsx            # Forecast/Types/Shifts toggle
│   ├── detail/
│   │   ├── DemographicsPanel.tsx           # Config-driven demographics grid
│   │   ├── ShiftHistoryChart.tsx           # visx line chart (type or county)
│   │   ├── ElectionHistoryChart.tsx         # visx bar chart (county)
│   │   ├── MemberGeography.tsx             # Mini filtered map
│   │   ├── CorrelatedTypes.tsx             # Related types from covariance
│   │   └── SimilarCounties.tsx             # Same-type counties
│   ├── methodology/
│   │   ├── ScrollySection.tsx              # react-scrollama wrapper
│   │   ├── StepViz.tsx                     # Visualization renderer per step
│   │   └── MetricsCard.tsx                 # Model metric display card
│   ├── map/
│   │   ├── MapShell.tsx                    # deck.gl container (refactored)
│   │   ├── MapLegend.tsx                   # Floating legend
│   │   ├── MapTooltip.tsx                  # Hover tooltip
│   │   └── MapControls.tsx                 # Zoom, reset, overlay controls
│   └── shared/
│       ├── RatingBadge.tsx                 # Lean D / Tossup / etc badge
│       ├── MarginDisplay.tsx               # Formatted margin (D+3.2)
│       ├── FreshnessStamp.tsx              # "Updated 2 hours ago"
│       ├── ProvenanceLabel.tsx             # "Based on 7 polls"
│       ├── Sparkline.tsx                   # Tiny visx line for tables
│       └── ErrorAlert.tsx                  # SWR error fallback
├── lib/
│   ├── config/
│   │   ├── display.ts                      # Field → label/format/section mapping
│   │   ├── palette.ts                      # Colors: ratings, super-types, choropleth
│   │   ├── methodology.ts                  # Scrollytelling steps + model metrics
│   │   └── navigation.ts                   # Route definitions for breadcrumbs
│   ├── hooks/
│   │   ├── use-senate-overview.ts          # SWR: GET /senate/overview
│   │   ├── use-race-detail.ts              # SWR: GET /races/{slug}
│   │   ├── use-type-detail.ts              # SWR: GET /types/{id}
│   │   ├── use-county-detail.ts            # SWR: GET /counties/{fips}
│   │   ├── use-polls.ts                    # SWR: GET /polls
│   │   ├── use-types.ts                    # SWR: GET /types
│   │   ├── use-super-types.ts              # SWR: GET /super-types
│   │   ├── use-type-scatter.ts             # SWR: GET /types/scatter-data
│   │   └── use-forecast.ts                 # SWR: GET /forecast
│   ├── api.ts                              # Raw fetch functions (consumed by SWR hooks)
│   ├── format.ts                           # Format functions: currency, percent, per1000, margin
│   └── types.ts                            # Shared TypeScript interfaces
├── public/
│   ├── states-us.geojson                   # State boundaries (keep)
│   └── favicon.ico                         # Fix the 404
└── e2e/
    ├── landing.spec.ts                     # Landing page flows
    ├── forecast.spec.ts                    # Forecast hub flows
    └── navigation.spec.ts                  # Cross-page navigation
```

---

## Phase 0: Branch & Foundation

### Task 1: Create Feature Branch and Install Dependencies

**Files:**
- Modify: `web/package.json`
- Modify: `web/tsconfig.json`

**What we're building:** The foundation that every other task depends on. Tailwind v4, shadcn/ui, and all new dependencies.

- [ ] **Step 1: Create feature branch**

```bash
cd /home/hayden/projects/wethervane
git checkout -b feat/frontend-v2
```

- [ ] **Step 2: Install Tailwind v4 and PostCSS**

```bash
cd web
npm install tailwindcss @tailwindcss/postcss postcss
```

- [ ] **Step 3: Install shadcn/ui dependencies**

```bash
npx shadcn@latest init
```

When prompted:
- Style: Default
- Base color: Neutral
- CSS variables: Yes
- Tailwind config location: (accept default)
- Component alias: `@/components/ui`
- Utility alias: `@/lib/utils`

- [ ] **Step 4: Install visx modules**

```bash
npm install @visx/shape @visx/scale @visx/axis @visx/tooltip @visx/group @visx/responsive @visx/text @visx/curve @visx/geo @visx/gradient
```

- [ ] **Step 5: Install SWR, Motion, TanStack Table, react-scrollama**

```bash
npm install swr motion @tanstack/react-table react-scrollama react-intersection-observer
```

- [ ] **Step 6: Verify build succeeds**

```bash
npm run build
```

Expected: Build succeeds (existing pages still work, new deps are unused).

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "chore: install frontend v2 dependencies (tailwind, shadcn, visx, swr, motion)"
```

---

### Task 2: Tailwind + Theme Configuration

**Files:**
- Rewrite: `web/app/globals.css`
- Create: `web/tailwind.config.ts` (if shadcn init didn't create it)
- Create: `web/postcss.config.mjs`

**What we're building:** The Dusty Ink v2 design system as Tailwind CSS variables. Every color, spacing, and typography decision from the spec lives here.

> **Research basis:** Spec §Design System. Purple for tossup (not gray). Muted academic tones. Both light and dark themes.

- [ ] **Step 1: Write globals.css with Tailwind directives and CSS variables**

```css
@import "tailwindcss";

@layer base {
  :root {
    /* Surface */
    --color-bg: 247 248 250;          /* #f7f8fa */
    --color-surface: 255 255 255;      /* #ffffff */
    --color-surface-elevated: 255 255 255;
    --color-border: 224 224 224;       /* #e0e0e0 */
    --color-border-subtle: 240 238 235;

    /* Text */
    --color-text: 34 34 34;           /* #222222 */
    --color-text-muted: 102 102 102;  /* #666666 */
    --color-text-subtle: 138 132 120;

    /* Partisan — Spec §Design System */
    --forecast-safe-d: 45 74 111;     /* #2d4a6f */
    --forecast-likely-d: 75 109 144;  /* #4b6d90 */
    --forecast-lean-d: 126 154 181;   /* #7e9ab5 */
    --forecast-tossup: 138 107 138;   /* #8a6b8a — PURPLE, not gray */
    --forecast-lean-r: 196 144 122;   /* #c4907a */
    --forecast-likely-r: 158 94 78;   /* #9e5e4e */
    --forecast-safe-r: 110 53 53;     /* #6e3535 */

    /* Focus */
    --color-focus: 33 102 172;        /* #2166ac */

    --font-serif: Georgia, "Times New Roman", serif;
    --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }

  [data-theme="dark"] {
    --color-bg: 26 27 30;
    --color-surface: 37 38 43;
    --color-surface-elevated: 44 45 50;
    --color-border: 61 63 68;
    --color-border-subtle: 42 43 47;

    --color-text: 224 224 224;
    --color-text-muted: 160 160 160;
    --color-text-subtle: 120 118 112;

    --forecast-safe-d: 91 155 213;
    --forecast-likely-d: 75 130 185;
    --forecast-lean-d: 110 155 195;
    --forecast-tossup: 160 130 160;
    --forecast-lean-r: 210 160 140;
    --forecast-likely-r: 200 130 110;
    --forecast-safe-r: 232 116 106;

    --color-focus: 91 155 213;
  }

  @media (prefers-color-scheme: dark) {
    [data-theme="system"] {
      --color-bg: 26 27 30;
      --color-surface: 37 38 43;
      --color-surface-elevated: 44 45 50;
      --color-border: 61 63 68;
      --color-border-subtle: 42 43 47;
      --color-text: 224 224 224;
      --color-text-muted: 160 160 160;
      --color-text-subtle: 120 118 112;
      --forecast-safe-d: 91 155 213;
      --forecast-likely-d: 75 130 185;
      --forecast-lean-d: 110 155 195;
      --forecast-tossup: 160 130 160;
      --forecast-lean-r: 210 160 140;
      --forecast-likely-r: 200 130 110;
      --forecast-safe-r: 232 116 106;
      --color-focus: 91 155 213;
    }
  }
}

@layer base {
  body {
    font-family: var(--font-sans);
    background-color: rgb(var(--color-bg));
    color: rgb(var(--color-text));
    transition: background-color 0.2s ease, color 0.2s ease;
  }

  h1, h2, h3 {
    font-family: var(--font-serif);
  }

  @media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
      animation-duration: 0.01ms !important;
      animation-iteration-count: 1 !important;
      transition-duration: 0.01ms !important;
    }
  }
}
```

- [ ] **Step 2: Verify Tailwind compiles**

```bash
npm run dev
```

Visit http://localhost:3001 — page should load with new base styles applied. Colors may look different from the old globals.css — that's expected.

- [ ] **Step 3: Create ThemeToggle component**

Create `web/components/shared/ThemeToggle.tsx`:

```typescript
"use client";

import { useEffect, useState } from "react";

type Theme = "system" | "light" | "dark";

const CYCLE: Theme[] = ["system", "light", "dark"];
const ICONS: Record<Theme, string> = { system: "◑", light: "☀", dark: "☾" };

/**
 * Theme toggle button: system → light → dark → system.
 * Stores preference in localStorage under 'wethervane-theme'.
 * Applies via data-theme attribute on <html> (read by CSS variables).
 */
export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>("system");

  useEffect(() => {
    const stored = localStorage.getItem("wethervane-theme") as Theme | null;
    if (stored && CYCLE.includes(stored)) setTheme(stored);
  }, []);

  function cycle() {
    const next = CYCLE[(CYCLE.indexOf(theme) + 1) % CYCLE.length];
    setTheme(next);
    localStorage.setItem("wethervane-theme", next);
    document.documentElement.setAttribute("data-theme", next);
  }

  return (
    <button
      onClick={cycle}
      className="text-lg w-8 h-8 flex items-center justify-center rounded hover:bg-[rgb(var(--color-border))] transition-colors"
      aria-label={`Theme: ${theme}. Click to cycle.`}
      title={`Theme: ${theme}`}
    >
      {ICONS[theme]}
    </button>
  );
}
```

- [ ] **Step 4: Commit**

```bash
git add web/app/globals.css web/tailwind.config.ts web/postcss.config.mjs web/components/shared/ThemeToggle.tsx
git commit -m "feat: tailwind v4 + dusty ink v2 CSS variables + theme toggle (light/dark/system)"
```

---

### Task 3: Configuration System

**Files:**
- Create: `web/lib/config/display.ts`
- Create: `web/lib/config/palette.ts`
- Create: `web/lib/config/navigation.ts`
- Create: `web/lib/format.ts`
- Create: `web/lib/types.ts`

**What we're building:** The config-driven rendering backbone. When the model changes from J=130 to J=150 or adds a new demographic field, no component code changes — only these config files.

> **Research basis:** Spec §Configuration Architecture. Spec Principle P5: "Configuration Over Hardcoding." Audit finding: stale "100 types / 5 super-types" on about/methodology pages.

- [ ] **Step 1: Create display config**

Create `web/lib/config/display.ts`:

```typescript
export interface FieldConfig {
  label: string;
  format: "percent" | "currency" | "number" | "per1000_to_pct" | "margin" | "raw";
  section: "economics" | "education" | "demographics" | "culture" | "housing" | "shift" | "other";
  sortOrder: number;
  /** If true, show a zero-line rule on charts */
  hasZeroLine?: boolean;
}

/**
 * Maps API field names to display metadata.
 * Consumed by: DemographicsPanel, ComparisonTable, ScatterPlot axis labels, tooltips.
 *
 * MAINTENANCE: When the model pipeline adds a new feature, add one entry here.
 * When a feature is removed, delete its entry. No component changes needed.
 * Unknown API fields render with raw key name as label (see format.ts fallback).
 */
export const FIELD_DISPLAY: Record<string, FieldConfig> = {
  // Demographics
  pct_white_nh: { label: "White (Non-Hispanic)", format: "percent", section: "demographics", sortOrder: 1 },
  pct_black: { label: "Black", format: "percent", section: "demographics", sortOrder: 2 },
  pct_hispanic: { label: "Hispanic", format: "percent", section: "demographics", sortOrder: 3 },
  pct_asian: { label: "Asian", format: "percent", section: "demographics", sortOrder: 4 },
  median_age: { label: "Median Age", format: "number", section: "demographics", sortOrder: 5 },
  pop_total: { label: "Population", format: "number", section: "demographics", sortOrder: 6 },

  // Economics
  median_hh_income: { label: "Median Household Income", format: "currency", section: "economics", sortOrder: 1 },
  pct_owner_occupied: { label: "Homeownership Rate", format: "percent", section: "housing", sortOrder: 1 },
  pct_wfh: { label: "Work From Home", format: "percent", section: "economics", sortOrder: 3 },
  pct_management: { label: "Management Occupations", format: "percent", section: "economics", sortOrder: 4 },
  log_pop_density: { label: "Log Population Density", format: "number", section: "demographics", sortOrder: 7 },

  // Education
  pct_bachelors_plus: { label: "Bachelor's Degree+", format: "percent", section: "education", sortOrder: 1 },

  // Culture / Religion — NOTE: adherence_rate is per-1,000, NOT a fraction
  // See CLAUDE.md Gotchas: "Religious adherence rate is per-1,000"
  evangelical_share: { label: "Evangelical Protestant", format: "percent", section: "culture", sortOrder: 1 },
  mainline_share: { label: "Mainline Protestant", format: "percent", section: "culture", sortOrder: 2 },
  catholic_share: { label: "Catholic", format: "percent", section: "culture", sortOrder: 3 },
  black_protestant_share: { label: "Black Protestant", format: "percent", section: "culture", sortOrder: 4 },
  congregations_per_1000: { label: "Congregations per 1,000", format: "number", section: "culture", sortOrder: 5 },
  religious_adherence_rate: { label: "Religious Adherence", format: "per1000_to_pct", section: "culture", sortOrder: 6 },

  // Partisan
  mean_dem_share: { label: "Mean Democratic Share", format: "percent", section: "other", sortOrder: 1 },
  pred_dem_share: { label: "Predicted Dem Share", format: "percent", section: "other", sortOrder: 2 },

  // Shifts — these have zero-lines on charts
  pres_d_shift_00_04: { label: "Pres Dem Shift 2000→2004", format: "margin", section: "shift", sortOrder: 1, hasZeroLine: true },
  pres_d_shift_04_08: { label: "Pres Dem Shift 2004→2008", format: "margin", section: "shift", sortOrder: 2, hasZeroLine: true },
  pres_d_shift_08_12: { label: "Pres Dem Shift 2008→2012", format: "margin", section: "shift", sortOrder: 3, hasZeroLine: true },
  pres_d_shift_12_16: { label: "Pres Dem Shift 2012→2016", format: "margin", section: "shift", sortOrder: 4, hasZeroLine: true },
  pres_d_shift_16_20: { label: "Pres Dem Shift 2016→2020", format: "margin", section: "shift", sortOrder: 5, hasZeroLine: true },
  pres_d_shift_20_24: { label: "Pres Dem Shift 2020→2024", format: "margin", section: "shift", sortOrder: 6, hasZeroLine: true },
};

/**
 * Get display config for a field, with fallback for unknown fields.
 * Unknown fields render with the raw API key as label.
 */
export function getFieldConfig(key: string): FieldConfig {
  return FIELD_DISPLAY[key] ?? {
    label: key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()),
    format: "raw",
    section: "other",
    sortOrder: 999,
  };
}
```

- [ ] **Step 2: Create palette config**

Create `web/lib/config/palette.ts`:

```typescript
/**
 * Rating colors — the partisan lean scale.
 * Research basis: Purple for tossup, not gray (gray reads as "no data").
 * See spec §Design System, research §5 Anti-pattern 3.
 */
export const RATING_COLORS: Record<string, string> = {
  safe_d: "#2d4a6f",
  likely_d: "#4b6d90",
  lean_d: "#7e9ab5",
  tossup: "#8a6b8a",
  lean_r: "#c4907a",
  likely_r: "#9e5e4e",
  safe_r: "#6e3535",
};

/** Human-readable rating labels */
export const RATING_LABELS: Record<string, string> = {
  safe_d: "Safe D",
  likely_d: "Likely D",
  lean_d: "Lean D",
  tossup: "Tossup",
  lean_r: "Lean R",
  likely_r: "Likely R",
  safe_r: "Safe R",
};

/**
 * Super-type colors — indexed by super_type_id from API.
 * Super-type NAMES come from the /super-types API endpoint, never hardcoded.
 * Only colors live here. When super-types grow from 6→8→10, add entries.
 *
 * MAINTENANCE: These are RGB tuples for deck.gl (which expects [r,g,b]).
 * The hex equivalents are in comments for CSS usage.
 */
export const SUPER_TYPE_COLORS: [number, number, number][] = [
  [220, 120, 55],   // #dc7837 — amber-orange
  [115, 45, 140],   // #732d8c — deep violet
  [220, 110, 110],  // #dc6e6e — rose-salmon
  [170, 35, 50],    // #aa2332 — deep crimson
  [38, 145, 145],   // #269191 — teal-cyan
  [195, 155, 25],   // #c39b19 — deep gold
  [65, 140, 210],   // #418cd2 — sky blue
  [40, 140, 85],    // #288c55 — emerald green
  // Overflow slots for future super-types
  [180, 100, 180],  // #b464b4 — lavender
  [140, 160, 60],   // #8ca03c — olive
];

/** Convert RGB tuple to CSS hex string */
export function rgbToHex([r, g, b]: [number, number, number]): string {
  return `#${[r, g, b].map((c) => c.toString(16).padStart(2, "0")).join("")}`;
}

/** Get super-type color, with fallback for out-of-range IDs */
export function getSuperTypeColor(id: number): [number, number, number] {
  return SUPER_TYPE_COLORS[id] ?? [150, 150, 150];
}

/**
 * Choropleth color interpolation for dem share.
 * Maps 0.0–1.0 dem_share to the partisan color scale.
 */
export const CHOROPLETH_CONFIG = {
  demMin: 0.3,
  demMax: 0.7,
  steps: 9,
} as const;

export function choroplethColor(demShare: number): [number, number, number, number] {
  const clamped = Math.max(CHOROPLETH_CONFIG.demMin, Math.min(CHOROPLETH_CONFIG.demMax, demShare));
  const t = (clamped - CHOROPLETH_CONFIG.demMin) / (CHOROPLETH_CONFIG.demMax - CHOROPLETH_CONFIG.demMin);

  // Interpolate: safe_r (t=0) → tossup (t=0.5) → safe_d (t=1)
  const safeR: [number, number, number] = [110, 53, 53];
  const tossup: [number, number, number] = [181, 169, 149];
  const safeD: [number, number, number] = [45, 74, 111];

  let rgb: [number, number, number];
  if (t < 0.5) {
    const s = t / 0.5;
    rgb = safeR.map((c, i) => Math.round(c + (tossup[i] - c) * s)) as [number, number, number];
  } else {
    const s = (t - 0.5) / 0.5;
    rgb = tossup.map((c, i) => Math.round(c + (safeD[i] - c) * s)) as [number, number, number];
  }

  return [rgb[0], rgb[1], rgb[2], 200];
}
```

- [ ] **Step 3: Create navigation config**

Create `web/lib/config/navigation.ts`:

```typescript
export interface NavItem {
  label: string;
  href: string;
}

export const MAIN_NAV: NavItem[] = [
  { label: "Forecast", href: "/forecast" },
  { label: "Explore", href: "/explore" },
  { label: "Methodology", href: "/methodology" },
];

export const FOOTER_NAV: NavItem[] = [
  { label: "About", href: "/about" },
  { label: "Methodology", href: "/methodology" },
  { label: "GitHub", href: "https://github.com/HaydenHaines/wethervane" },
];

/**
 * Breadcrumb route mapping.
 * Dynamic segments ([fips], [id], [slug]) are resolved at render time
 * using data from the page's API response.
 */
export const BREADCRUMB_ROUTES: Record<string, string> = {
  "/": "Home",
  "/forecast": "Forecast",
  "/forecast/senate": "Senate",
  "/forecast/governor": "Governor",
  "/explore": "Explore",
  "/explore/types": "Types",
  "/explore/map": "Map",
  "/explore/shifts": "Shifts",
  "/methodology": "Methodology",
  "/methodology/accuracy": "Accuracy",
  "/about": "About",
};
```

- [ ] **Step 4: Create format utilities**

Create `web/lib/format.ts`:

```typescript
import { getFieldConfig, type FieldConfig } from "@/lib/config/display";

/** Format a value based on its field config format type */
export function formatValue(value: number | null | undefined, format: FieldConfig["format"]): string {
  if (value == null) return "—";

  switch (format) {
    case "percent":
      return `${(value * 100).toFixed(1)}%`;
    case "currency":
      return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 }).format(value);
    case "number":
      return new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 }).format(value);
    case "per1000_to_pct":
      // RCMS adherence rate is per-1,000. Display as percentage.
      // See CLAUDE.md Gotchas: "Religious adherence rate is per-1,000, not a fraction."
      return `${(value / 10).toFixed(1)}%`;
    case "margin":
      return formatMargin(value);
    case "raw":
      return String(value);
  }
}

/** Format a value using the display config for a given field key */
export function formatField(key: string, value: number | null | undefined): string {
  const config = getFieldConfig(key);
  return formatValue(value, config.format);
}

/**
 * Format a margin as "D+3.2" or "R+1.8" or "EVEN".
 * Input: decimal where positive = Dem advantage (0.032 = D+3.2pp).
 * Threshold: 0.005 (0.5pp) for EVEN.
 *
 * NOTE: The old SenateControlBar used 0.5 instead of 0.005, causing
 * every race to show "EVEN". See audit §Critical Bug 1. This function
 * uses the correct threshold.
 */
export function formatMargin(margin: number): string {
  const abs = Math.abs(margin);
  if (abs < 0.005) return "EVEN";
  const pct = (abs * 100).toFixed(1);
  return margin > 0 ? `D+${pct}` : `R+${pct}`;
}

/**
 * Format a margin for display with the party-colored prefix.
 * Returns { text: "D+3.2", party: "d" | "r" | "even" }
 */
export function parseMargin(margin: number): { text: string; party: "d" | "r" | "even" } {
  const abs = Math.abs(margin);
  if (abs < 0.005) return { text: "EVEN", party: "even" };
  const pct = (abs * 100).toFixed(1);
  return margin > 0 ? { text: `D+${pct}`, party: "d" } : { text: `R+${pct}`, party: "r" };
}

/** Format relative time: "2 hours ago", "3 days ago", etc. */
export function timeAgo(date: Date | string): string {
  const now = new Date();
  const then = new Date(date);
  const diffMs = now.getTime() - then.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  if (diffMins < 1) return "just now";
  if (diffMins < 60) return `${diffMins} min ago`;
  const diffHours = Math.floor(diffMins / 60);
  if (diffHours < 24) return `${diffHours} hour${diffHours === 1 ? "" : "s"} ago`;
  const diffDays = Math.floor(diffHours / 24);
  return `${diffDays} day${diffDays === 1 ? "" : "s"} ago`;
}
```

- [ ] **Step 5: Create shared types**

Create `web/lib/types.ts`:

```typescript
/**
 * Shared TypeScript interfaces for API responses.
 * These match the FastAPI response models in api/models.py.
 *
 * MAINTENANCE: When the API contract changes, update these types
 * and run `npm run build` to catch any downstream breakage.
 */

export interface SenateRaceData {
  state: string;
  race: string;
  slug: string;
  rating: string;
  margin: number;
  n_polls: number;
}

export interface SenateOverviewData {
  headline: string;
  subtitle: string;
  dem_seats_safe: number;
  gop_seats_safe: number;
  races: SenateRaceData[];
  state_colors: Record<string, string>;
}

export interface TypeSummary {
  type_id: number;
  super_type_id: number;
  display_name: string;
  n_counties: number;
  mean_pred_dem_share: number | null;
  median_hh_income: number | null;
  pct_bachelors_plus: number | null;
  pct_white_nh: number | null;
  log_pop_density: number | null;
}

export interface TypeCounty {
  county_fips: string;
  county_name: string | null;
  state_abbr: string;
}

export interface TypeDetail extends TypeSummary {
  counties: TypeCounty[];
  demographics: Record<string, number>;
  shift_profile: Record<string, number> | null;
  narrative: string | null;
}

export interface SuperTypeSummary {
  super_type_id: number;
  display_name: string;
  member_type_ids: number[];
  n_counties: number;
}

export interface CountyDetail {
  county_fips: string;
  county_name: string;
  state_abbr: string;
  dominant_type: number;
  super_type: number;
  type_display_name: string;
  super_type_display_name: string;
  narrative: string | null;
  pred_dem_share: number;
  demographics: Record<string, number>;
  sibling_counties: Array<{
    county_fips: string;
    county_name: string;
    state_abbr: string;
    pred_dem_share: number | null;
  }>;
}

export interface ForecastRow {
  county_fips: string;
  county_name: string | null;
  state_abbr: string;
  race: string;
  pred_dem_share: number | null;
  pred_std: number | null;
  pred_lo90: number | null;
  pred_hi90: number | null;
  state_pred: number | null;
  poll_avg: number | null;
}

export interface PollRow {
  race: string;
  geography: string;
  geo_level: string;
  dem_share: number;
  n_sample: number;
  date: string | null;
  pollster: string | null;
}

export interface TypeScatterPoint {
  type_id: number;
  super_type_id: number;
  display_name: string;
  n_counties: number;
  demographics: Record<string, number>;
  shift_profile: Record<string, number>;
}
```

- [ ] **Step 6: Commit**

```bash
git add web/lib/config/ web/lib/format.ts web/lib/types.ts
git commit -m "feat: config system (display, palette, navigation) + format utils + shared types"
```

---

### Task 4: SWR Data Hooks

**Files:**
- Create: `web/lib/hooks/use-senate-overview.ts`
- Create: `web/lib/hooks/use-types.ts`
- Create: `web/lib/hooks/use-super-types.ts`
- Create: `web/lib/hooks/use-type-detail.ts`
- Create: `web/lib/hooks/use-county-detail.ts`
- Create: `web/lib/hooks/use-polls.ts`
- Create: `web/lib/hooks/use-type-scatter.ts`
- Create: `web/lib/hooks/use-forecast.ts`
- Modify: `web/lib/api.ts` (keep raw fetch functions, add new ones)

**What we're building:** SWR hooks wrapping every API call. Components never call fetch directly. Each hook handles caching, revalidation, error state, and loading state.

> **Research basis:** Spec §Data Layer. Principle P7: "Speed Creates Trust." SWR stale-while-revalidate gives instant cache hits with background refresh.

- [ ] **Step 1: Update api.ts with complete fetch functions**

Keep the existing functions in `web/lib/api.ts` and add any missing ones. The existing file already has all the fetch functions we need. Verify it exports:
- `fetchSenateOverview()`
- `fetchTypes()`
- `fetchSuperTypes()`
- `fetchTypeDetail(id)`
- `fetchCounties()`
- `fetchForecast(race?, state?)`
- `fetchPolls(params)`
- `fetchTypeScatterData()`

Add one missing function to `web/lib/api.ts`:

```typescript
export async function fetchCountyDetail(fips: string): Promise<CountyDetail> {
  const res = await fetch(`${API_BASE}/counties/${fips}`);
  if (!res.ok) throw new Error(`/counties/${fips} failed: ${res.status}`);
  return res.json();
}
```

Import the `CountyDetail` type from `@/lib/types` at the top.

- [ ] **Step 2: Create SWR hooks**

Create `web/lib/hooks/use-senate-overview.ts`:

```typescript
import useSWR from "swr";
import { fetchSenateOverview } from "@/lib/api";
import type { SenateOverviewData } from "@/lib/types";

export function useSenateOverview() {
  return useSWR<SenateOverviewData>("senate-overview", fetchSenateOverview, {
    revalidateOnFocus: false,
    dedupingInterval: 300_000, // 5 min
  });
}
```

Create `web/lib/hooks/use-types.ts`:

```typescript
import useSWR from "swr";
import { fetchTypes } from "@/lib/api";
import type { TypeSummary } from "@/lib/types";

export function useTypes() {
  return useSWR<TypeSummary[]>("types", fetchTypes, {
    revalidateOnFocus: false,
    dedupingInterval: 1_800_000, // 30 min
  });
}
```

Create `web/lib/hooks/use-super-types.ts`:

```typescript
import useSWR from "swr";
import { fetchSuperTypes } from "@/lib/api";
import type { SuperTypeSummary } from "@/lib/types";

export function useSuperTypes() {
  return useSWR<SuperTypeSummary[]>("super-types", fetchSuperTypes, {
    revalidateOnFocus: false,
    dedupingInterval: 3_600_000, // 60 min
  });
}
```

Create `web/lib/hooks/use-type-detail.ts`:

```typescript
import useSWR from "swr";
import { fetchTypeDetail } from "@/lib/api";
import type { TypeDetail } from "@/lib/types";

export function useTypeDetail(id: number | null) {
  return useSWR<TypeDetail>(
    id != null ? `type-${id}` : null,
    () => fetchTypeDetail(id!),
    { revalidateOnFocus: false, dedupingInterval: 1_800_000 },
  );
}
```

Create `web/lib/hooks/use-county-detail.ts`:

```typescript
import useSWR from "swr";
import { fetchCountyDetail } from "@/lib/api";
import type { CountyDetail } from "@/lib/types";

export function useCountyDetail(fips: string | null) {
  return useSWR<CountyDetail>(
    fips ? `county-${fips}` : null,
    () => fetchCountyDetail(fips!),
    { revalidateOnFocus: false, dedupingInterval: 1_800_000 },
  );
}
```

Create `web/lib/hooks/use-polls.ts`:

```typescript
import useSWR from "swr";
import { fetchPolls } from "@/lib/api";
import type { PollRow } from "@/lib/types";

export function usePolls(params: { race?: string; state?: string; cycle?: string }) {
  const key = `polls-${params.race ?? ""}-${params.state ?? ""}-${params.cycle ?? ""}`;
  return useSWR<PollRow[]>(
    key,
    () => fetchPolls(params),
    { revalidateOnFocus: false, dedupingInterval: 900_000 }, // 15 min
  );
}
```

Create `web/lib/hooks/use-type-scatter.ts`:

```typescript
import useSWR from "swr";
import { fetchTypeScatterData } from "@/lib/api";
import type { TypeScatterPoint } from "@/lib/types";

export function useTypeScatter() {
  return useSWR<TypeScatterPoint[]>("type-scatter", fetchTypeScatterData, {
    revalidateOnFocus: false,
    dedupingInterval: 1_800_000,
  });
}
```

Create `web/lib/hooks/use-forecast.ts`:

```typescript
import useSWR from "swr";
import { fetchForecast } from "@/lib/api";
import type { ForecastRow } from "@/lib/types";

export function useForecast(race?: string, state?: string) {
  const key = `forecast-${race ?? "all"}-${state ?? "all"}`;
  return useSWR<ForecastRow[]>(
    key,
    () => fetchForecast(race, state),
    { revalidateOnFocus: false, dedupingInterval: 300_000 },
  );
}
```

- [ ] **Step 3: Verify TypeScript compiles**

```bash
cd web && npx tsc --noEmit
```

Expected: No type errors.

- [ ] **Step 4: Commit**

```bash
git add web/lib/hooks/ web/lib/api.ts
git commit -m "feat: SWR data hooks for all API endpoints with caching config"
```

---

## Phase 1: Landing Page

### Task 5: Shared Components (RatingBadge, MarginDisplay, FreshnessStamp, ErrorAlert)

**Files:**
- Create: `web/components/shared/RatingBadge.tsx`
- Create: `web/components/shared/MarginDisplay.tsx`
- Create: `web/components/shared/FreshnessStamp.tsx`
- Create: `web/components/shared/ErrorAlert.tsx`

**What we're building:** Reusable atoms used across every page. Build these first so pages can compose them.

> **Research basis:** Spec Principle P9: "Data Provenance on Every Display." Every data display must answer: when, how many, how confident.

- [ ] **Step 1: Install shadcn Badge and Alert components**

```bash
cd web
npx shadcn@latest add badge alert
```

- [ ] **Step 2: Create RatingBadge**

Create `web/components/shared/RatingBadge.tsx`:

```typescript
"use client";

import { Badge } from "@/components/ui/badge";
import { RATING_COLORS, RATING_LABELS } from "@/lib/config/palette";

interface RatingBadgeProps {
  rating: string;
  className?: string;
}

/**
 * Displays a partisan rating as a colored badge.
 * Rating values come from the API (e.g., "safe_d", "tossup", "lean_r").
 * Colors and labels come from palette config — not hardcoded.
 */
export function RatingBadge({ rating, className }: RatingBadgeProps) {
  const color = RATING_COLORS[rating] ?? RATING_COLORS.tossup;
  const label = RATING_LABELS[rating] ?? rating;

  return (
    <Badge
      className={className}
      style={{ backgroundColor: color, color: "#fff", border: "none" }}
    >
      {label}
    </Badge>
  );
}
```

- [ ] **Step 3: Create MarginDisplay**

Create `web/components/shared/MarginDisplay.tsx`:

```typescript
import { parseMargin } from "@/lib/format";

interface MarginDisplayProps {
  margin: number;
  size?: "sm" | "md" | "lg" | "xl";
  className?: string;
}

const SIZE_CLASSES = {
  sm: "text-sm font-medium",
  md: "text-lg font-semibold",
  lg: "text-2xl font-bold",
  xl: "text-5xl font-bold tracking-tight",
};

/**
 * Formatted margin display: "D+3.2" in blue, "R+1.8" in red, "EVEN" in purple.
 * Uses the corrected 0.005 threshold (not 0.5 — see audit §Critical Bug 1).
 */
export function MarginDisplay({ margin, size = "md", className }: MarginDisplayProps) {
  const { text, party } = parseMargin(margin);

  const colorClass =
    party === "d"
      ? "text-[rgb(var(--forecast-safe-d))]"
      : party === "r"
        ? "text-[rgb(var(--forecast-safe-r))]"
        : "text-[rgb(var(--forecast-tossup))]";

  return (
    <span className={`${SIZE_CLASSES[size]} ${colorClass} ${className ?? ""}`}>
      {text}
    </span>
  );
}
```

- [ ] **Step 4: Create FreshnessStamp**

Create `web/components/shared/FreshnessStamp.tsx`:

```typescript
import { timeAgo } from "@/lib/format";

interface FreshnessStampProps {
  updatedAt?: string | Date;
  pollCount?: number;
  className?: string;
}

/**
 * Shows data provenance: when data was updated and how many sources.
 * Research basis: Spec Principle P9 — every display answers "when?" and "how many?"
 */
export function FreshnessStamp({ updatedAt, pollCount, className }: FreshnessStampProps) {
  const parts: string[] = [];
  if (updatedAt) parts.push(`Updated ${timeAgo(updatedAt)}`);
  if (pollCount != null) parts.push(`${pollCount} poll${pollCount === 1 ? "" : "s"}`);

  if (parts.length === 0) return null;

  return (
    <span className={`text-sm text-[rgb(var(--color-text-muted))] ${className ?? ""}`}>
      {parts.join(" · ")}
    </span>
  );
}
```

- [ ] **Step 5: Create ErrorAlert**

Create `web/components/shared/ErrorAlert.tsx`:

```typescript
"use client";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

interface ErrorAlertProps {
  title?: string;
  message?: string;
  retry?: () => void;
}

/**
 * Error fallback for SWR data loading failures.
 * Shows a non-intrusive alert with optional retry button.
 */
export function ErrorAlert({ title = "Failed to load data", message, retry }: ErrorAlertProps) {
  return (
    <Alert variant="destructive" className="my-4">
      <AlertTitle>{title}</AlertTitle>
      <AlertDescription className="flex items-center gap-2">
        {message ?? "Something went wrong. Please try again."}
        {retry && (
          <button
            onClick={retry}
            className="underline text-sm font-medium hover:no-underline"
          >
            Retry
          </button>
        )}
      </AlertDescription>
    </Alert>
  );
}
```

- [ ] **Step 6: Commit**

```bash
git add web/components/shared/ web/components/ui/
git commit -m "feat: shared components (RatingBadge, MarginDisplay, FreshnessStamp, ErrorAlert)"
```

---

### Task 6: Landing Page

**Files:**
- Rewrite: `web/app/page.tsx`
- Rewrite: `web/app/layout.tsx`
- Create: `web/components/landing/HeroSection.tsx`
- Create: `web/components/landing/RaceTicker.tsx`
- Create: `web/components/landing/EntryPoints.tsx`
- Create: `web/components/nav/GlobalNav.tsx`
- Create: `web/components/nav/Footer.tsx`

**What we're building:** The front door. Hero headline with one big number, competitive race ticker, three entry points, global nav, and footer. SSR the hero at build time.

> **Research basis:** Spec §Landing Page. Principle P3: "One Big Number." "5 seconds to orient, then navigate deeper."

- [ ] **Step 1: Install shadcn Card and Skeleton components**

```bash
npx shadcn@latest add card skeleton separator
```

- [ ] **Step 2: Create GlobalNav**

Create `web/components/nav/GlobalNav.tsx`:

```typescript
"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { MAIN_NAV } from "@/lib/config/navigation";

/**
 * Sticky top navigation bar.
 * Active tab highlighted based on current pathname.
 */
export function GlobalNav() {
  const pathname = usePathname();

  return (
    <nav className="sticky top-0 z-50 border-b border-[rgb(var(--color-border))] bg-[rgb(var(--color-bg))]/95 backdrop-blur-sm">
      <div className="mx-auto flex h-12 max-w-6xl items-center justify-between px-4">
        <Link
          href="/"
          className="font-serif text-lg font-bold text-[rgb(var(--color-text))] hover:opacity-80 transition-opacity"
        >
          WetherVane
        </Link>
        <div className="flex gap-6">
          {MAIN_NAV.map((item) => {
            const isActive = pathname.startsWith(item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`text-sm font-medium transition-colors ${
                  isActive
                    ? "text-[rgb(var(--color-text))] border-b-2 border-[rgb(var(--color-focus))]"
                    : "text-[rgb(var(--color-text-muted))] hover:text-[rgb(var(--color-text))]"
                }`}
              >
                {item.label}
              </Link>
            );
          })}
        </div>
      </div>
    </nav>
  );
}
```

- [ ] **Step 3: Create Footer**

Create `web/components/nav/Footer.tsx`:

```typescript
import Link from "next/link";
import { FOOTER_NAV } from "@/lib/config/navigation";

export function Footer() {
  return (
    <footer className="border-t border-[rgb(var(--color-border))] mt-16 py-8 px-4">
      <div className="mx-auto max-w-6xl flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="flex gap-6">
          {FOOTER_NAV.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className="text-sm text-[rgb(var(--color-text-muted))] hover:text-[rgb(var(--color-text))] transition-colors"
              {...(item.href.startsWith("http") ? { target: "_blank", rel: "noopener" } : {})}
            >
              {item.label}
            </Link>
          ))}
        </div>
        <p className="text-sm text-[rgb(var(--color-text-subtle))]">
          Built by Hayden Haines · WetherVane Electoral Model
        </p>
      </div>
    </footer>
  );
}
```

- [ ] **Step 4: Create HeroSection**

Create `web/components/landing/HeroSection.tsx`:

```typescript
import { MarginDisplay } from "@/components/shared/MarginDisplay";
import { Skeleton } from "@/components/ui/skeleton";
import type { SenateOverviewData } from "@/lib/types";

interface HeroSectionProps {
  data: SenateOverviewData | undefined;
  isLoading: boolean;
}

/**
 * Landing page hero: one big number + headline.
 * Research basis: Spec Principle P3 — "One Big Number" anchor.
 */
export function HeroSection({ data, isLoading }: HeroSectionProps) {
  if (isLoading) {
    return (
      <section className="py-16 text-center">
        <Skeleton className="mx-auto h-8 w-96 mb-4" />
        <Skeleton className="mx-auto h-16 w-48 mb-4" />
        <Skeleton className="mx-auto h-4 w-72" />
      </section>
    );
  }

  if (!data) return null;

  const { headline, subtitle, dem_seats_safe, gop_seats_safe, races } = data;
  const totalDem = dem_seats_safe;
  const totalGop = gop_seats_safe;

  return (
    <section className="py-16 text-center">
      <h1 className="font-serif text-3xl sm:text-4xl font-bold mb-4 text-[rgb(var(--color-text))]">
        {headline}
      </h1>
      <div className="flex items-center justify-center gap-4 mb-4">
        <span className="text-5xl font-bold text-[rgb(var(--forecast-safe-d))]">{totalDem}</span>
        <span className="text-2xl text-[rgb(var(--color-text-muted))]">–</span>
        <span className="text-5xl font-bold text-[rgb(var(--forecast-safe-r))]">{totalGop}</span>
      </div>
      <p className="text-[rgb(var(--color-text-muted))] max-w-lg mx-auto">
        {subtitle} · Based on {races.length} races and the WetherVane covariance model.
      </p>
    </section>
  );
}
```

- [ ] **Step 5: Create RaceTicker**

Create `web/components/landing/RaceTicker.tsx`:

```typescript
import Link from "next/link";
import { RatingBadge } from "@/components/shared/RatingBadge";
import { MarginDisplay } from "@/components/shared/MarginDisplay";
import { Skeleton } from "@/components/ui/skeleton";
import type { SenateRaceData } from "@/lib/types";

interface RaceTickerProps {
  races: SenateRaceData[] | undefined;
  isLoading: boolean;
}

/**
 * Horizontal strip of the most competitive races.
 * Research basis: Spec §Landing Page — "Race ticker" with 6-8 most competitive.
 */
export function RaceTicker({ races, isLoading }: RaceTickerProps) {
  if (isLoading) {
    return (
      <div className="flex gap-3 overflow-x-auto px-4 py-2">
        {Array.from({ length: 6 }).map((_, i) => (
          <Skeleton key={i} className="h-20 w-36 shrink-0 rounded-lg" />
        ))}
      </div>
    );
  }

  if (!races?.length) return null;

  // Sort by absolute margin (most competitive first), take top 8
  const competitive = [...races]
    .sort((a, b) => Math.abs(a.margin) - Math.abs(b.margin))
    .slice(0, 8);

  return (
    <div className="flex gap-3 overflow-x-auto px-4 py-2 scrollbar-thin">
      {competitive.map((race) => (
        <Link
          key={race.slug}
          href={`/forecast/${race.slug}`}
          className="shrink-0 rounded-lg border border-[rgb(var(--color-border))] bg-[rgb(var(--color-surface))] p-3 w-36 hover:border-[rgb(var(--color-focus))] transition-colors"
        >
          <div className="text-xs text-[rgb(var(--color-text-muted))] mb-1">
            {race.state} Senate
          </div>
          <MarginDisplay margin={race.margin} size="md" />
          <div className="mt-1">
            <RatingBadge rating={race.rating} />
          </div>
        </Link>
      ))}
    </div>
  );
}
```

- [ ] **Step 6: Create EntryPoints**

Create `web/components/landing/EntryPoints.tsx`:

```typescript
import Link from "next/link";
import { Card, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";

const ENTRIES = [
  {
    title: "See the full forecast",
    description: "Senate and governor race predictions with poll tracking",
    href: "/forecast",
  },
  {
    title: "Explore electoral types",
    description: "130 community types that shape American elections",
    href: "/explore/types",
  },
  {
    title: "How the model works",
    description: "The methodology behind covariance-based forecasting",
    href: "/methodology",
  },
];

export function EntryPoints() {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 px-4 max-w-4xl mx-auto">
      {ENTRIES.map((entry) => (
        <Link key={entry.href} href={entry.href}>
          <Card className="h-full hover:border-[rgb(var(--color-focus))] transition-colors cursor-pointer">
            <CardHeader>
              <CardTitle className="text-base font-serif">{entry.title} →</CardTitle>
              <CardDescription>{entry.description}</CardDescription>
            </CardHeader>
          </Card>
        </Link>
      ))}
    </div>
  );
}
```

- [ ] **Step 7: Rewrite root layout**

Rewrite `web/app/layout.tsx`:

```typescript
import type { Metadata } from "next";
import { GlobalNav } from "@/components/nav/GlobalNav";
import { Footer } from "@/components/nav/Footer";
import "./globals.css";

export const metadata: Metadata = {
  title: "WetherVane — Electoral Forecast Model",
  description: "Covariance-based election forecasting using 130 electoral community types.",
};

const THEME_INIT_SCRIPT = `
(function() {
  var stored = localStorage.getItem('wethervane-theme');
  if (stored === 'light' || stored === 'dark') {
    document.documentElement.setAttribute('data-theme', stored);
  } else {
    document.documentElement.setAttribute('data-theme', 'system');
  }
})();
`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_INIT_SCRIPT }} />
      </head>
      <body className="min-h-screen flex flex-col">
        <a
          href="#main-content"
          className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-[100] focus:bg-[rgb(var(--color-surface))] focus:p-2 focus:rounded"
        >
          Skip to main content
        </a>
        <GlobalNav />
        <main id="main-content" className="flex-1">
          {children}
        </main>
        <Footer />
      </body>
    </html>
  );
}
```

- [ ] **Step 8: Rewrite landing page**

Rewrite `web/app/page.tsx`:

```typescript
"use client";

import { HeroSection } from "@/components/landing/HeroSection";
import { RaceTicker } from "@/components/landing/RaceTicker";
import { EntryPoints } from "@/components/landing/EntryPoints";
import { FreshnessStamp } from "@/components/shared/FreshnessStamp";
import { useSenateOverview } from "@/lib/hooks/use-senate-overview";

export default function LandingPage() {
  const { data, error, isLoading } = useSenateOverview();

  return (
    <div className="max-w-6xl mx-auto">
      <HeroSection data={data} isLoading={isLoading} />

      <section className="mb-12">
        <h2 className="font-serif text-lg font-semibold px-4 mb-3">Key Races</h2>
        <RaceTicker races={data?.races} isLoading={isLoading} />
      </section>

      <EntryPoints />

      <div className="text-center mt-12 mb-8">
        <FreshnessStamp
          pollCount={data?.races.reduce((sum, r) => sum + r.n_polls, 0)}
        />
      </div>
    </div>
  );
}
```

- [ ] **Step 9: Verify the landing page renders**

```bash
npm run dev
```

Visit http://localhost:3001. Should see: hero headline, seat count, race ticker, three entry-point cards, footer. No "EVEN" bug — margins display correctly via `formatMargin` with the `0.005` threshold.

- [ ] **Step 10: Commit**

```bash
git add web/app/layout.tsx web/app/page.tsx web/components/landing/ web/components/nav/ web/components/ui/
git commit -m "feat: landing page with hero, race ticker, entry points, global nav, footer"
```

---

## Phase 2: Forecast Hub

### Task 7: Forecast Layout with Map

**Files:**
- Create: `web/app/forecast/layout.tsx`
- Create: `web/app/forecast/page.tsx` (redirect to /forecast/senate)

**What we're building:** The forecast section layout. Map on the left (desktop) or top (mobile), panel content on the right/bottom. Redirect `/forecast` to `/forecast/senate`.

- [ ] **Step 1: Create forecast layout**

Create `web/app/forecast/layout.tsx`:

```typescript
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Forecast — WetherVane",
  description: "2026 Senate and Governor race forecasts",
};

export default function ForecastLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex flex-col lg:flex-row h-[calc(100vh-3rem)]">
      {/* Map pane — will be added in Task 28 (map refactor). Placeholder for now. */}
      <div className="hidden lg:block lg:w-1/2 bg-[rgb(var(--color-bg))] border-r border-[rgb(var(--color-border))]">
        <div className="flex items-center justify-center h-full text-[rgb(var(--color-text-muted))]">
          Map loads here
        </div>
      </div>
      {/* Panel pane */}
      <div className="flex-1 overflow-y-auto p-4 lg:p-6">
        {children}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Create forecast redirect page**

Create `web/app/forecast/page.tsx`:

```typescript
import { redirect } from "next/navigation";

export default function ForecastPage() {
  redirect("/forecast/senate");
}
```

- [ ] **Step 3: Commit**

```bash
git add web/app/forecast/
git commit -m "feat: forecast layout with map placeholder + redirect to /forecast/senate"
```

---

### Task 8: Balance Bar Component

**Files:**
- Create: `web/components/forecast/BalanceBar.tsx`

**What we're building:** The Senate seat balance visualization — a horizontal stacked bar where each segment is a race colored by rating. Hover shows tooltip. Click navigates to race detail.

> **Research basis:** Spec §Forecast Hub. The existing SenateControlBar concept, but functional. Real margins, real colors, real links.

- [ ] **Step 1: Install shadcn Tooltip**

```bash
npx shadcn@latest add tooltip
```

- [ ] **Step 2: Create BalanceBar**

Create `web/components/forecast/BalanceBar.tsx`:

```typescript
"use client";

import { useRouter } from "next/navigation";
import { RATING_COLORS, RATING_LABELS } from "@/lib/config/palette";
import { formatMargin } from "@/lib/format";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import type { SenateRaceData } from "@/lib/types";

interface BalanceBarProps {
  races: SenateRaceData[];
  demSeats: number;
  gopSeats: number;
}

/** Rating sort order: most D → tossup → most R */
const RATING_ORDER: Record<string, number> = {
  safe_d: 0, likely_d: 1, lean_d: 2, tossup: 3, lean_r: 4, likely_r: 5, safe_r: 6,
};

/**
 * Senate balance bar: horizontal stacked segments, one per race.
 * Research basis: Spec §Forecast Hub — "real margins, real colors, real links."
 */
export function BalanceBar({ races, demSeats, gopSeats }: BalanceBarProps) {
  const router = useRouter();

  const sorted = [...races].sort(
    (a, b) => (RATING_ORDER[a.rating] ?? 3) - (RATING_ORDER[b.rating] ?? 3),
  );

  const totalSegments = sorted.length;
  if (totalSegments === 0) return null;

  return (
    <TooltipProvider delayDuration={100}>
      <div className="mb-6">
        {/* Seat counts */}
        <div className="flex justify-between mb-2 text-sm font-semibold">
          <span className="text-[rgb(var(--forecast-safe-d))]">{demSeats}D</span>
          <span className="text-[rgb(var(--color-text-muted))]">
            51 needed for control
          </span>
          <span className="text-[rgb(var(--forecast-safe-r))]">{gopSeats}R</span>
        </div>

        {/* Bar */}
        <div className="flex h-8 rounded-md overflow-hidden border border-[rgb(var(--color-border))]">
          {sorted.map((race) => (
            <Tooltip key={race.slug}>
              <TooltipTrigger asChild>
                <button
                  className="h-full transition-opacity hover:opacity-80 focus:outline-none focus:ring-2 focus:ring-[rgb(var(--color-focus))] min-w-[8px]"
                  style={{
                    flex: 1,
                    backgroundColor: RATING_COLORS[race.rating] ?? RATING_COLORS.tossup,
                  }}
                  onClick={() => router.push(`/forecast/${race.slug}`)}
                  aria-label={`${race.state}: ${formatMargin(race.margin)}`}
                />
              </TooltipTrigger>
              <TooltipContent>
                <p className="font-semibold">{race.state}</p>
                <p>{formatMargin(race.margin)} · {RATING_LABELS[race.rating] ?? race.rating}</p>
                <p className="text-xs text-muted-foreground">
                  {race.n_polls} poll{race.n_polls === 1 ? "" : "s"}
                </p>
              </TooltipContent>
            </Tooltip>
          ))}
        </div>

        {/* 50-seat midline indicator */}
        <div className="relative h-0">
          <div
            className="absolute top-[-32px] h-8 w-px bg-[rgb(var(--color-text))] opacity-30"
            style={{ left: "50%" }}
          />
        </div>
      </div>
    </TooltipProvider>
  );
}
```

- [ ] **Step 3: Commit**

```bash
git add web/components/forecast/BalanceBar.tsx web/components/ui/
git commit -m "feat: BalanceBar component with tooltips and race navigation"
```

---

### Task 9: Race Card + Race Card Grid

**Files:**
- Create: `web/components/forecast/RaceCard.tsx`
- Create: `web/components/forecast/RaceCardGrid.tsx`

**What we're building:** Individual race cards showing margin, rating, poll count — and a grid that sorts them by competitiveness. Cards are links to race detail pages (fixing the critical audit finding where cards didn't navigate).

> **Research basis:** Spec §Forecast Hub. Audit §Critical Bug: "Clicking a race card only changes the map view but does not navigate to the race detail page."

- [ ] **Step 1: Create RaceCard**

Create `web/components/forecast/RaceCard.tsx`:

```typescript
import Link from "next/link";
import { Card, CardContent } from "@/components/ui/card";
import { RatingBadge } from "@/components/shared/RatingBadge";
import { MarginDisplay } from "@/components/shared/MarginDisplay";
import { FreshnessStamp } from "@/components/shared/FreshnessStamp";
import type { SenateRaceData } from "@/lib/types";

interface RaceCardProps {
  race: SenateRaceData;
}

/**
 * Individual race card — always a link to the race detail page.
 * Audit fix: old RaceCard was a button that only changed the map.
 * This version is a <Link> wrapping a <Card>.
 */
export function RaceCard({ race }: RaceCardProps) {
  return (
    <Link href={`/forecast/${race.slug}`}>
      <Card className="h-full hover:border-[rgb(var(--color-focus))] transition-colors cursor-pointer">
        <CardContent className="p-4">
          <div className="flex items-start justify-between mb-2">
            <div>
              <div className="text-xs text-[rgb(var(--color-text-muted))] mb-1">
                {race.state}
              </div>
              <MarginDisplay margin={race.margin} size="lg" />
            </div>
            <RatingBadge rating={race.rating} />
          </div>
          <FreshnessStamp pollCount={race.n_polls} />
        </CardContent>
      </Card>
    </Link>
  );
}
```

- [ ] **Step 2: Create RaceCardGrid**

Create `web/components/forecast/RaceCardGrid.tsx`:

```typescript
import { RaceCard } from "./RaceCard";
import type { SenateRaceData } from "@/lib/types";

interface RaceCardGridProps {
  races: SenateRaceData[];
  title: string;
}

/**
 * Grid of race cards, sorted by competitiveness (smallest margin first).
 */
export function RaceCardGrid({ races, title }: RaceCardGridProps) {
  const sorted = [...races].sort(
    (a, b) => Math.abs(a.margin) - Math.abs(b.margin),
  );

  return (
    <section className="mb-8">
      <h2 className="font-serif text-lg font-semibold mb-3">{title}</h2>
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-2 xl:grid-cols-3 gap-3">
        {sorted.map((race) => (
          <RaceCard key={race.slug} race={race} />
        ))}
      </div>
    </section>
  );
}
```

- [ ] **Step 3: Commit**

```bash
git add web/components/forecast/RaceCard.tsx web/components/forecast/RaceCardGrid.tsx
git commit -m "feat: RaceCard and RaceCardGrid — cards link to race detail pages"
```

---

### Task 10: Senate Overview Page

**Files:**
- Create: `web/app/forecast/senate/page.tsx`

**What we're building:** The Senate forecast overview — balance bar + key races grid + all-races table. The main forecast page that users land on.

- [ ] **Step 1: Create Senate page**

Create `web/app/forecast/senate/page.tsx`:

```typescript
"use client";

import { useSenateOverview } from "@/lib/hooks/use-senate-overview";
import { BalanceBar } from "@/components/forecast/BalanceBar";
import { RaceCardGrid } from "@/components/forecast/RaceCardGrid";
import { ErrorAlert } from "@/components/shared/ErrorAlert";
import { Skeleton } from "@/components/ui/skeleton";

/** Tossup + lean races are "key races" */
const KEY_RATINGS = new Set(["tossup", "lean_d", "lean_r"]);

export default function SenatePage() {
  const { data, error, isLoading, mutate } = useSenateOverview();

  if (error) {
    return <ErrorAlert title="Failed to load Senate forecast" retry={() => mutate()} />;
  }

  if (isLoading || !data) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-8 w-full" />
        <div className="grid grid-cols-3 gap-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-28 rounded-lg" />
          ))}
        </div>
      </div>
    );
  }

  const keyRaces = data.races.filter((r) => KEY_RATINGS.has(r.rating));
  const otherRaces = data.races.filter((r) => !KEY_RATINGS.has(r.rating));

  return (
    <div>
      <h1 className="font-serif text-2xl font-bold mb-4">2026 United States Senate</h1>

      <BalanceBar
        races={data.races}
        demSeats={data.dem_seats_safe}
        gopSeats={data.gop_seats_safe}
      />

      {keyRaces.length > 0 && (
        <RaceCardGrid races={keyRaces} title="Key Races" />
      )}

      {otherRaces.length > 0 && (
        <RaceCardGrid races={otherRaces} title="Other Races" />
      )}
    </div>
  );
}
```

- [ ] **Step 2: Verify the Senate page renders**

```bash
npm run dev
```

Visit http://localhost:3001/forecast/senate. Should see balance bar with real margins and colors, key races grid, other races grid. Every card links to `/forecast/[slug]`.

- [ ] **Step 3: Commit**

```bash
git add web/app/forecast/senate/
git commit -m "feat: Senate overview page with balance bar + race card grids"
```

---

### Task 11: Governor Overview Page

**Files:**
- Create: `web/app/forecast/governor/page.tsx`

**What we're building:** Governor forecast overview. Same structure as Senate but with governor races. This fixes the audit finding: "No Governor races visible on the forecast page."

> **Research basis:** Audit §Major Issue 5: "No governor races on the forecast page. Only Senate races shown despite 36 governor slugs."

NOTE: The API currently does not have a `/governor/overview` endpoint. This page will use the existing `/forecast/races` endpoint to list governor races and display them with available data. If a dedicated governor overview API is needed, that should be added to the backend as a separate task.

- [ ] **Step 1: Create Governor page**

Create `web/app/forecast/governor/page.tsx`:

```typescript
"use client";

import { useState, useEffect } from "react";
import { RaceCardGrid } from "@/components/forecast/RaceCardGrid";
import { ErrorAlert } from "@/components/shared/ErrorAlert";
import { Skeleton } from "@/components/ui/skeleton";
import type { SenateRaceData } from "@/lib/types";

/**
 * Governor overview page.
 * Uses /forecast/races to get governor race slugs, then constructs race data.
 * TODO: When a /governor/overview API endpoint exists, switch to a SWR hook.
 */
export default function GovernorPage() {
  const [races, setRaces] = useState<SenateRaceData[] | null>(null);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const res = await fetch("/api/v1/forecast/races");
        if (!res.ok) throw new Error("Failed to load races");
        const allRaces: string[] = await res.json();
        const govRaces = allRaces
          .filter((r) => r.includes("Governor"))
          .map((race) => {
            const state = race.split(" ")[1]; // "2026 GA Governor" → "GA"
            return {
              state,
              race,
              slug: race.toLowerCase().replace(/\s+/g, "-"),
              rating: "tossup" as const, // Placeholder until API provides ratings
              margin: 0,
              n_polls: 0,
            };
          });
        setRaces(govRaces);
      } catch (e) {
        setError(e as Error);
      }
    }
    load();
  }, []);

  if (error) return <ErrorAlert title="Failed to load Governor races" />;

  if (!races) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-8 w-64" />
        <div className="grid grid-cols-3 gap-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-28 rounded-lg" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div>
      <h1 className="font-serif text-2xl font-bold mb-4">2026 Governor Races</h1>
      <p className="text-[rgb(var(--color-text-muted))] mb-6">
        {races.length} governor races tracked. Detailed predictions available as polling data arrives.
      </p>
      <RaceCardGrid races={races} title="All Governor Races" />
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add web/app/forecast/governor/
git commit -m "feat: Governor overview page (fixes audit: governor races invisible)"
```

---

## Phases 3–9: Remaining Tasks (Summary)

The plan continues with the same granularity for all remaining phases. Each task below follows the identical format: files, research citation, step-by-step with code, verify command, commit.

---

### Task 12: Race Detail — Hero + Confidence Interval

**Files:**
- Create: `web/app/forecast/[slug]/page.tsx`
- Create: `web/components/forecast/RaceHero.tsx`

Build the race detail page hero section: candidate labels, large margin number, rating badge, and 90% confidence interval text. SSR via `generateStaticParams` using `/forecast/races`. The hero section embodies Principle P3 (One Big Number) with uncertainty context (Principle P1).

- [ ] Steps: Create page with SSR data fetch, create RaceHero component with MarginDisplay + RatingBadge + CI text, verify with `npm run dev` and visit `/forecast/2026-ga-senate`, commit.

---

### Task 13: Quantile Dotplot

**Files:**
- Create: `web/components/forecast/QuantileDotplot.tsx`

> **Research basis:** The Economist's quantile dotplots. "In 73 of 100 scenarios, the Democrat wins." See spec §Race Detail, research §4 Inspiration Gallery.

Build a visx-based 100-dot distribution visualization. Input: `pred_dem_share` and `pred_std`. Generate 100 quantile values from a normal distribution. Arrange as a beeswarm/strip. Color dots by party outcome (blue if >0.5, red if <0.5). Show frequency text below.

- [ ] Steps: Create component with visx `Circle` marks + `scaleLinear`, add frequency label, add `prefers-reduced-motion` static fallback, integrate into race detail page, verify, commit.

---

### Task 14: Poll Tracker Chart

**Files:**
- Create: `web/components/forecast/PollTracker.tsx`

> **Research basis:** visx area chart with stepped gradient fan chart. See spec §Race Detail, research §2 Pattern: Fan Chart.

Build a visx area chart showing polling average over time. Confidence band rendered as stepped gradient (3 opacity levels for 60%/80%/95%). Uses `usePolls` hook. Falls back to "No polls yet" message when empty.

- [ ] Steps: Create component with visx `AreaClosed` + `LinePath` + `scaleTime` + `scaleLinear`, add stepped confidence bands, add event dot annotations, integrate into race detail page, verify, commit.

---

### Task 15: Electoral Types Breakdown

**Files:**
- Create: `web/components/forecast/TypesBreakdown.tsx`

Show the top 5-8 types in a state with super-type color dots, descriptive names (no partisan suffix), predicted margins, and population weight. Each type name links to `/type/[id]`. This is WetherVane's unique analytical lens.

- [ ] Steps: Create component using data from race detail API, render as a compact list with super-type colors from palette config, add links, integrate into race detail page, verify, commit.

---

### Task 16: Poll Table + Section Weight Sliders

**Files:**
- Create: `web/components/forecast/PollTable.tsx`
- Create: `web/components/forecast/SectionWeightSliders.tsx`

> **Research basis:** Spec §Race Detail. Data provenance — show pollster, date, sample size, result.

Build a TanStack Table for polls (sortable by date, pollster, result). Build shadcn Slider components for model prior / state polls / national polls weights.

- [ ] Steps: Install shadcn `slider` + `table`, create PollTable with TanStack columns, create SectionWeightSliders with labeled shadcn Sliders, integrate both into race detail page, verify, commit.

---

### Task 17: Type Directory (`/explore/types`)

**Files:**
- Create: `web/app/explore/page.tsx` (redirect)
- Create: `web/app/explore/types/page.tsx`
- Create: `web/components/explore/TypeGrid.tsx`
- Create: `web/components/explore/TypeCard.tsx`

> **Research basis:** Spec §Explore. Audit §Major: "Super-type labels generic ('Super-type 1')." Audit §Major: "Type names with Blue/Red suffixes."

Type directory with cards grouped by super-type. Search bar filters. Super-type names come from API `/super-types`, not hardcoded. Type cards show descriptive names only — strip any "Red"/"Blue" suffix in the display config if present.

- [ ] Steps: Create TypeCard, create TypeGrid with grouping by super_type_id + search filter, create page, verify super-type names display from API, commit.

---

### Task 18: Scatter Plot (visx)

**Files:**
- Create: `web/components/explore/ScatterPlot.tsx`

> **Research basis:** Spec §Explore. Audit fixes: human-readable axis labels (via display config), hover tooltips, super-type named legend.

Rebuild the scatter plot with visx. X/Y axis dropdowns use `FIELD_DISPLAY` labels. Dots colored by super-type from palette config. Hover tooltip via `@visx/tooltip`. Click opens type detail Sheet.

- [ ] Steps: Create with visx `Circle` + `scaleLinear` + `AxisBottom` + `AxisLeft`, add axis selector dropdowns with config-driven labels, add tooltip, add legend from `useSuperTypes()` hook, integrate into explore/types page, verify, commit.

---

### Task 19: Comparison Table

**Files:**
- Create: `web/components/explore/ComparisonTable.tsx`

> **Research basis:** Spec §Explore. 4-column comparison with config-driven rendering.

Rebuild with TanStack Table + shadcn. Combobox type selector (searchable over 130 types). Rows driven by `FIELD_DISPLAY` config. Color-coded cells for relative values. Shareable URL via query params.

- [ ] Steps: Install shadcn `combobox` (or build with `command` + `popover`), create ComparisonTable, wire up URL query params for selected type IDs, verify, commit.

---

### Task 20: Full-Screen Map Page (`/explore/map`)

**Files:**
- Create: `web/app/explore/map/page.tsx`
- Create: `web/components/map/MapShell.tsx` (refactored from old)
- Create: `web/components/map/MapLegend.tsx`
- Create: `web/components/map/MapTooltip.tsx`
- Create: `web/components/explore/MapOverlayToggle.tsx`

> **Research basis:** Spec §Explore/Map. Full-viewport deck.gl. Floating legend with named super-types. Overlay toggle: Forecast | Types | Shifts.

Refactor the existing MapShell.tsx to the new component structure. Keep deck.gl GeoJsonLayer logic. Add floating legend reading super-type names from `useSuperTypes()`. Add overlay toggle. State click → fly-to + tract load. Tract click → shadcn Sheet with details.

- [ ] Steps: Refactor MapShell (extract tooltip and legend into separate components), create MapLegend with API-driven names, create overlay toggle, create page, verify, commit.

---

### Task 21: Type Detail Page (`/type/[id]`)

**Files:**
- Rewrite: `web/app/type/[id]/page.tsx`
- Create: `web/components/detail/DemographicsPanel.tsx`
- Create: `web/components/detail/ShiftHistoryChart.tsx`

> **Research basis:** Spec §Detail Pages. Config-driven demographics rendering. Type names without partisan suffix.

SSR type detail page. Hero with descriptive name (no Red/Blue suffix), super-type badge, lean. Demographics panel reads from display config — new model features auto-display. Shift history chart built with visx line chart.

- [ ] Steps: Create DemographicsPanel (iterates `Object.entries(demographics)`, looks up each in `getFieldConfig`, groups by section, sorts by sortOrder), create ShiftHistoryChart (visx LinePath + AreaClosed around zero), rewrite page, verify with `/type/1`, commit.

---

### Task 22: County Detail Page (`/county/[fips]`)

**Files:**
- Rewrite: `web/app/county/[fips]/page.tsx`
- Create: `web/components/detail/ElectionHistoryChart.tsx`
- Create: `web/components/detail/SimilarCounties.tsx`

> **Research basis:** Spec §Detail Pages. Audit fix: "No election history shown." Audit fix: "View on Map goes to national."

SSR county detail. Hero with county name, type link, big margin number. Election history bar chart (visx). Demographics via DemographicsPanel (same component as type pages). "View on Map" links to `/explore/map?focus=[fips]`. Similar counties from API `sibling_counties`.

- [ ] Steps: Create ElectionHistoryChart (visx Bar), create SimilarCounties, rewrite page with cross-links to type detail and map, verify with `/county/13121`, commit.

---

### Task 23: Correlated Types + Member Geography

**Files:**
- Create: `web/components/detail/CorrelatedTypes.tsx`
- Create: `web/components/detail/MemberGeography.tsx`

Correlated types: show 3-4 most similar types from the covariance matrix (requires API support — if not available, show types from the same super-type as a fallback). Member geography: mini filtered deck.gl map showing only this type's counties.

- [ ] Steps: Create both components, integrate into type detail page, verify, commit.

---

### Task 24: Breadcrumbs Component

**Files:**
- Create: `web/components/nav/Breadcrumbs.tsx`

> **Research basis:** Spec §Navigation. Audit: "Breadcrumbs: Home and Map are synonymous." Fix: distinct labels, truncation on mobile.

Config-driven breadcrumbs reading from `BREADCRUMB_ROUTES`. Dynamic segments resolved from page data. Mobile: truncate middle segments.

- [ ] Steps: Create Breadcrumbs component, integrate into all detail pages (type, county, race, methodology), verify, commit.

---

### Task 25: Methodology Scrollytelling

**Files:**
- Create: `web/app/methodology/page.tsx`
- Create: `web/lib/config/methodology.ts`
- Create: `web/components/methodology/ScrollySection.tsx`
- Create: `web/components/methodology/StepViz.tsx`
- Create: `web/components/methodology/MetricsCard.tsx`

> **Research basis:** Spec §Methodology. Principle P8: "Show Your Work." Research §2 Pattern: Scrollytelling (WaPo, Reuters, Pudding). react-scrollama with IntersectionObserver.

8-step scroll-driven narrative. Text on left, sticky visualization on right. Steps configured in `methodology.ts`. Each step specifies vizType + vizConfig. `prefers-reduced-motion` renders static sections.

- [ ] Steps: Create methodology config with 8 steps, create ScrollySection wrapper, create StepViz renderer (dispatches to map/scatter/heatmap/diagram based on vizType), create MetricsCard, create page, verify scroll behavior, verify reduced-motion fallback, commit.

---

### Task 26: Methodology Accuracy Page

**Files:**
- Rewrite: `web/app/methodology/accuracy/page.tsx`

> **Research basis:** Spec §Methodology. Audit: "Contradictory sentence about holdout r vs LOO r."

Rebuild with shadcn components. Add a visx predicted-vs-actual scatter plot. Fix the contradictory text. Pull metrics from `MODEL_METRICS` config (not hardcoded).

- [ ] Steps: Create page with MetricsCards reading from config, create scatter plot, fix contradictory text, verify, commit.

---

### Task 27: Historical Shifts Page (`/explore/shifts`)

**Files:**
- Create: `web/app/explore/shifts/page.tsx`
- Create: `web/components/explore/ShiftSmallMultiples.tsx`

> **Research basis:** Spec §Explore/Shifts. Research §2 Pattern: Small multiples. "Realignment at a glance."

Small multiples grid: one mini visx chart per super-type showing Dem margin shift across 2008→2024. Consistent scales. Lower priority — can ship after core pages.

- [ ] Steps: Create ShiftSmallMultiples with visx LinePath in a CSS grid, create page, verify, commit.

---

### Task 28: Map Refactor and Integration

**Files:**
- Refactor: `web/components/map/MapShell.tsx`
- Create: `web/components/map/MapControls.tsx`

> **Research basis:** Spec §Performance. deck.gl retained. Tract GeoJSON on demand.

Refactor the existing MapShell into the new component structure. Extract tooltip, legend, and controls into separate focused components (<200 lines each). Wire map into the forecast layout (replacing the placeholder from Task 7). Add overlay toggle from Task 20.

- [ ] Steps: Refactor MapShell (move palette to config, move tooltip to MapTooltip, move legend to MapLegend), create MapControls (zoom, reset), wire into forecast layout, verify map loads with forecast coloring, commit.

---

### Task 29: Custom 404 + About Page + Embed Fix

**Files:**
- Create: `web/app/not-found.tsx`
- Create: `web/app/about/page.tsx`
- Rewrite: `web/app/embed/[slug]/page.tsx`
- Create: `web/public/favicon.ico`

> **Research basis:** Audit: "404 page has no branding." Audit: "embed has 6 hydration errors." Audit: "favicon.ico 404."

Custom 404 with WetherVane branding + nav links. About page pulling type/super-type counts from API (not hardcoded). Fix embed hydration errors. Add favicon.

- [ ] Steps: Create not-found.tsx with nav links, create about page with dynamic counts from `useSuperTypes` + `useTypes`, fix embed page SSR/client mismatch, add favicon, verify, commit.

---

### Task 30: Sitemap Update

**Files:**
- Rewrite: `web/app/sitemap.ts`

Update sitemap to include all new routes: `/`, `/forecast/senate`, `/forecast/governor`, all `/forecast/[slug]` races, `/explore/types`, `/explore/map`, `/explore/shifts`, `/methodology`, `/methodology/accuracy`, `/about`.

- [ ] Steps: Rewrite sitemap.ts to fetch race slugs from API, generate URLs for all routes, verify with `curl localhost:3001/sitemap.xml`, commit.

---

### Task 31: Mobile Transformations — Forecast

**Files:**
- Modify: `web/components/forecast/BalanceBar.tsx`
- Modify: `web/components/forecast/RaceCardGrid.tsx`
- Modify: `web/components/forecast/QuantileDotplot.tsx`
- Modify: `web/components/forecast/PollTracker.tsx`

> **Research basis:** Spec §Mobile Strategy. Mobile rules codified in spec. Balance bar → text summary. Race cards → carousel. Dotplot → 50 dots. Poll tracker → trend only.

Apply mobile breakpoint transformations per the spec's mobile transformation table.

- [ ] Steps: Add responsive classes to BalanceBar (text summary at <768px), convert RaceCardGrid to horizontal scroll at <768px, reduce QuantileDotplot to 50 dots, simplify PollTracker, verify at 375px viewport, commit.

---

### Task 32: Mobile Transformations — Explore + Detail

**Files:**
- Modify: `web/components/explore/ScatterPlot.tsx`
- Modify: `web/components/explore/ComparisonTable.tsx`
- Modify: `web/components/detail/DemographicsPanel.tsx`
- Modify: `web/components/nav/GlobalNav.tsx`

> **Research basis:** Spec §Mobile Rules. "No horizontal scroll tables — use lists. No multi-column comparison >2."

Scatter: bottom sheet axis selectors. Comparison: max 2 columns on mobile. Demographics: single column. Nav: hamburger menu.

- [ ] Steps: Add responsive variants to each component, install shadcn `sheet` for mobile nav, verify at 375px, commit.

---

### Task 33: Touch Interaction Rules Codification

**Files:**
- Create: `web/lib/config/touch-rules.md` (documentation, not code)
- Modify: Various components to add long-press for tooltip

> **Research basis:** Spec §Mobile — 10 codified touch interaction rules.

Document the 10 touch rules in a reference file. Add `onTouchStart`/`onTouchEnd` long-press handlers to map tooltips and chart tooltips. Verify all interactive elements ≥44px.

- [ ] Steps: Write touch-rules.md, add long-press to MapTooltip, audit touch target sizes, fix any <44px targets, commit.

---

### Task 34: Accessibility Audit

**Files:**
- Modify: Various components for WCAG fixes

> **Research basis:** Spec §Accessibility. WCAG 2.1 AA.

Run axe-core audit. Fix any contrast failures. Add `aria-label` to map regions. Add visually-hidden data tables as chart alternatives. Verify skip-link works. Verify keyboard navigation through nav and race cards.

- [ ] Steps: Install `@axe-core/playwright`, run audit script, fix findings, re-run until clean, commit.

---

### Task 35: Playwright E2E Tests

**Files:**
- Create: `web/e2e/landing.spec.ts`
- Create: `web/e2e/forecast.spec.ts`
- Create: `web/e2e/navigation.spec.ts`

> **Research basis:** Spec §Testing Strategy. Critical flows: landing → forecast → race detail → back.

Write Playwright tests for the three critical user flows. Include visual regression screenshot for the stained glass map. Verify no console errors on any page.

- [ ] Steps: Write landing tests (hero renders, ticker links work), write forecast tests (balance bar renders, race cards navigate to detail), write navigation tests (breadcrumbs, back links, 404), run `npx playwright test`, commit.

---

### Task 36: Final Build Verification + Cleanup

**Files:**
- Modify: `web/package.json` (update scripts)
- Remove: Old component files no longer imported

Final verification: `npm run build` succeeds, `npm run start` serves the site, all Playwright tests pass. Remove any old components from the v1 codebase that are no longer imported. Update the `wethervane-frontend.service` if the build output path changed.

- [ ] Steps: Run `npm run build`, run `npm start`, visit all pages manually, run Playwright suite, remove dead code, update systemd service if needed, commit, push branch.

---

### Task 37: Performance Optimization — ISR, Dynamic Imports, Lazy Loading

**Files:**
- Modify: `web/app/forecast/[slug]/page.tsx` (add ISR revalidate)
- Modify: `web/app/forecast/senate/page.tsx` (add ISR)
- Modify: `web/app/explore/map/page.tsx` (dynamic import MapShell)
- Modify: Various chart components (dynamic import visx)

**What we're building:** The performance layer from the spec. SSR pages get ISR revalidation. Heavy client components (deck.gl, visx) load via `next/dynamic` with `ssr: false`. Below-fold sections lazy-load via IntersectionObserver.

> **Research basis:** Spec §Performance Architecture. Principle P7: "Speed Creates Trust." First meaningful paint under 1 second.

- [ ] **Step 1: Add ISR to forecast pages**

In `web/app/forecast/[slug]/page.tsx`, add:

```typescript
export const revalidate = 300; // 5 minutes
```

In `web/app/forecast/senate/page.tsx` — this is client-side (SWR handles caching). No ISR needed.

- [ ] **Step 2: Dynamic import deck.gl MapShell**

In any page that uses the map, import via:

```typescript
import dynamic from "next/dynamic";

const MapShell = dynamic(() => import("@/components/map/MapShell"), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full bg-[rgb(var(--color-bg))] animate-pulse" />
  ),
});
```

This prevents deck.gl's WebGL dependencies from breaking SSR.

- [ ] **Step 3: Dynamic import visx chart components**

For each visx-heavy component (QuantileDotplot, PollTracker, ScatterPlot, ShiftSmallMultiples, ShiftHistoryChart, ElectionHistoryChart), use dynamic imports in their parent pages:

```typescript
const QuantileDotplot = dynamic(
  () => import("@/components/forecast/QuantileDotplot").then(m => ({ default: m.QuantileDotplot })),
  { ssr: false, loading: () => <Skeleton className="h-48 w-full" /> },
);
```

- [ ] **Step 4: Lazy load below-fold sections**

Wrap below-fold sections (comparison table, methodology viz steps, county tables) with an IntersectionObserver trigger:

```typescript
"use client";
import { useInView } from "react-intersection-observer";

function LazySection({ children }: { children: React.ReactNode }) {
  const { ref, inView } = useInView({ triggerOnce: true, rootMargin: "200px" });
  return <div ref={ref}>{inView ? children : <Skeleton className="h-64 w-full" />}</div>;
}
```

- [ ] **Step 5: Verify performance**

```bash
npm run build && npm start
```

Check with browser DevTools Lighthouse: First Contentful Paint should be under 1.5s on landing page. Race detail pages should render hero instantly (SSR) with charts loading client-side.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "perf: ISR, dynamic imports for deck.gl/visx, lazy loading below-fold sections"
```

---

## Execution Notes

**Branch:** All work happens on `feat/frontend-v2`. The old frontend remains on `main` and serves production until the branch is merged.

**Commit frequency:** Every task ends with a commit. This gives 36 clean, bisectable commits.

**Testing checkpoints:** After Tasks 6 (landing), 10 (senate), 12 (race detail), 20 (map), 25 (methodology) — pause and verify the full user flow works before continuing.

**API dependencies:** Tasks 11 (governor) and 23 (correlated types) may need new API endpoints. If the endpoint doesn't exist, build the page with available data and mark the API gap as a follow-up task.

**Documentation to consult:**
- shadcn/ui: https://ui.shadcn.com/docs
- visx: https://airbnb.io/visx/docs
- SWR: https://swr.vercel.app/docs
- Motion: https://motion.dev/docs
- react-scrollama: https://github.com/jsonkao/react-scrollama
- deck.gl GeoJsonLayer: https://deck.gl/docs/api-reference/layers/geojson-layer
- TanStack Table: https://tanstack.com/table/latest
