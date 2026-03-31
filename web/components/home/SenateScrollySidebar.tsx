"use client";

import { cn } from "@/lib/utils";
import { DUSTY_INK } from "@/lib/config/palette";

/**
 * The ordered sequence of narrative zones, left-to-right (D side to R side).
 * This matches the seat ordering in the balance bar and the narrative arc.
 */
const ZONE_ORDER = [
  "not_up_d",
  "safe_up_d",
  "contested_d",
  "tossup",
  "contested_r",
  "safe_up_r",
  "not_up_r",
] as const;

type Zone = (typeof ZONE_ORDER)[number];

/** Dusty Ink hex colors per zone — maps narrative zones to the partisan palette. */
const ZONE_COLORS: Record<Zone, string> = {
  not_up_d:    DUSTY_INK.safeD,
  safe_up_d:   DUSTY_INK.likelyD,
  contested_d: DUSTY_INK.leanD,
  tossup:      DUSTY_INK.tossup,
  contested_r: DUSTY_INK.leanR,
  safe_up_r:   DUSTY_INK.likelyR,
  not_up_r:    DUSTY_INK.safeR,
};

interface SenateScrollySidebarProps {
  /** Seat counts per zone as returned by the API (zone_counts field). */
  zoneCounts: Record<string, number>;
  /** The zone currently in view — segments in this zone are fully opaque. */
  activeZone: string | null;
  className?: string;
}

/**
 * Senate seat sidebar: 100 seat segments laid out in zone order, colored by
 * partisan rating. The active zone is fully opaque; inactive zones are dimmed.
 *
 * Desktop (lg+): vertical strip on the left, sticky below the nav.
 * Mobile: horizontal strip pinned below the nav (rendered separately in page.tsx).
 */
export function SenateScrollySidebar({
  zoneCounts,
  activeZone,
  className,
}: SenateScrollySidebarProps) {
  // Build the array of seat segments in zone order.
  // Each segment is { zone, color }.
  const segments: { zone: Zone; color: string }[] = [];
  for (const zone of ZONE_ORDER) {
    const count = zoneCounts[zone] ?? 0;
    for (let i = 0; i < count; i++) {
      segments.push({ zone, color: ZONE_COLORS[zone] });
    }
  }

  return (
    <>
      {/* Desktop: vertical sidebar */}
      <aside
        className={cn(
          "sticky top-16 self-start h-[calc(100vh-4rem)] w-5 flex-col gap-px overflow-hidden rounded",
          className,
        )}
        aria-label="Senate seat map sidebar"
        aria-hidden="true"
      >
        <div className="flex h-full flex-col gap-px">
          {segments.map((seg, i) => (
            <div
              key={i}
              className="flex-1 min-h-[4px] rounded-sm transition-opacity duration-300"
              style={{
                backgroundColor: seg.color,
                opacity: isSegmentActive(seg.zone, activeZone) ? 1 : 0.2,
              }}
            />
          ))}
        </div>
      </aside>

      {/* Mobile: horizontal strip (always rendered alongside desktop, visibility
          controlled by className from parent) */}
    </>
  );
}

/**
 * Determines if a sidebar segment should be highlighted for the active zone.
 * When the battleground zone (tossup) is active, all competitive segments
 * (contested_d, tossup, contested_r) highlight together.
 */
function isSegmentActive(segmentZone: string, activeZone: string | null): boolean {
  if (activeZone === null) return true; // nothing active = all visible
  if (segmentZone === activeZone) return true;
  // Battleground zone highlights all competitive seats
  if (activeZone === "tossup") {
    return segmentZone === "contested_d" || segmentZone === "contested_r";
  }
  return false;
}

/**
 * Mobile-only version of the sidebar — a thin horizontal strip.
 * Rendered separately in the page layout so it can be pinned below the nav
 * with a different flex direction.
 */
export function SenateScrollySidebarMobile({
  zoneCounts,
  activeZone,
}: Omit<SenateScrollySidebarProps, "className">) {
  const segments: { zone: Zone; color: string }[] = [];
  for (const zone of ZONE_ORDER) {
    const count = zoneCounts[zone] ?? 0;
    for (let i = 0; i < count; i++) {
      segments.push({ zone, color: ZONE_COLORS[zone] });
    }
  }

  return (
    <div
      className="sticky top-12 z-10 flex h-2 w-full gap-px overflow-hidden lg:hidden"
      aria-label="Senate seat map strip"
      aria-hidden="true"
    >
      {segments.map((seg, i) => (
        <div
          key={i}
          className="flex-1 transition-opacity duration-300"
          style={{
            backgroundColor: seg.color,
            opacity: isSegmentActive(seg.zone, activeZone) ? 1 : 0.2,
          }}
        />
      ))}
    </div>
  );
}
