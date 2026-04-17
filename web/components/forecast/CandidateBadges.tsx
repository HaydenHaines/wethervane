"use client";

/**
 * CandidateBadges — displays sabermetric badge pills for candidates in a race.
 *
 * Fetches candidate badge data via SWR and renders compact badge pills
 * for each candidate, colored by badge dimension.  Also shows the CEC
 * (Candidate Effect Consistency) as a small indicator.
 *
 * This is a client component because it uses SWR for data fetching and
 * tooltip interactivity.
 */

import Link from "next/link";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useRaceCandidates } from "@/lib/hooks/use-race-candidates";
import type { CandidateBadge, RaceCandidateSummary } from "@/lib/api";
import { CTOVRadarChart } from "./CTOVRadarChart";

interface CandidateBadgesProps {
  /** Race key in the format used by candidates_2026.json, e.g. "2026 GA Senate". */
  raceKey: string;
}

/**
 * Consistent color map for badge dimensions.
 *
 * "Low" prefix badges use the same hue but at reduced opacity to indicate
 * the inverse (weakness in that dimension).  Colors are chosen to be
 * distinguishable on the Dusty Ink muted background.
 */
const BADGE_COLORS: Record<string, { bg: string; text: string; border: string }> = {
  "Hispanic Appeal": {
    bg: "rgba(234, 88, 12, 0.10)",
    text: "rgb(180, 70, 10)",
    border: "rgba(234, 88, 12, 0.30)",
  },
  "Black Community Strength": {
    bg: "rgba(124, 58, 237, 0.10)",
    text: "rgb(100, 45, 190)",
    border: "rgba(124, 58, 237, 0.30)",
  },
  "Senior Whisperer": {
    bg: "rgba(20, 120, 140, 0.10)",
    text: "rgb(15, 95, 115)",
    border: "rgba(20, 120, 140, 0.30)",
  },
  "Suburban Professional": {
    bg: "rgba(59, 130, 246, 0.10)",
    text: "rgb(40, 95, 190)",
    border: "rgba(59, 130, 246, 0.30)",
  },
  "Rural Populist": {
    bg: "rgba(161, 98, 7, 0.10)",
    text: "rgb(130, 80, 5)",
    border: "rgba(161, 98, 7, 0.30)",
  },
  "Faith Coalition": {
    bg: "rgba(168, 85, 120, 0.10)",
    text: "rgb(140, 65, 100)",
    border: "rgba(168, 85, 120, 0.30)",
  },
  "Turnout Monster": {
    bg: "rgba(34, 197, 94, 0.10)",
    text: "rgb(20, 130, 65)",
    border: "rgba(34, 197, 94, 0.30)",
  },
};

/** Default color for unrecognized badge names. */
const DEFAULT_BADGE_COLOR = {
  bg: "rgba(148, 163, 184, 0.10)",
  text: "rgb(85, 95, 110)",
  border: "rgba(148, 163, 184, 0.30)",
};

/**
 * Color for PCA-discovered badges.
 *
 * Distinct teal/cyan hue to visually separate data-driven discoveries from
 * curated catalog badges.  Uses dashed border to indicate these are
 * algorithmically derived rather than hand-curated.
 */
const DISCOVERED_BADGE_COLOR = {
  bg: "rgba(6, 182, 212, 0.08)",
  text: "rgb(8, 145, 178)",
  border: "rgba(6, 182, 212, 0.30)",
};

/**
 * Resolve the color for a badge name, handling "Low" prefix variants.
 *
 * "Low" badges (e.g. "Low Rural Populist") use the same hue as their
 * positive counterpart but with reduced opacity to visually signal a
 * weakness rather than a strength.
 */
function getBadgeColor(badgeName: string): { bg: string; text: string; border: string } {
  // Direct match
  if (badgeName in BADGE_COLORS) return BADGE_COLORS[badgeName];

  // "Low X" -> look up "X" and dim it
  if (badgeName.startsWith("Low ")) {
    const base = badgeName.slice(4);
    const baseColor = BADGE_COLORS[base];
    if (baseColor) {
      return {
        bg: baseColor.bg.replace("0.10", "0.06"),
        text: baseColor.text,
        border: baseColor.border.replace("0.30", "0.18"),
      };
    }
    // "Low Turnout Monster" -> "Turnout Monster" not found, try partial
    for (const [key, color] of Object.entries(BADGE_COLORS)) {
      if (base.includes(key) || key.includes(base)) {
        return {
          bg: color.bg.replace("0.10", "0.06"),
          text: color.text,
          border: color.border.replace("0.30", "0.18"),
        };
      }
    }
  }

  return DEFAULT_BADGE_COLOR;
}

/** Format CEC as a percentage label. */
function formatCEC(cec: number): string {
  return `${Math.round(cec * 100)}%`;
}

/** Party color accessor. */
const PARTY_LABEL_COLORS: Record<string, string> = {
  D: "var(--color-dem)",
  R: "var(--color-rep)",
  I: "var(--color-text-muted)",
};

function BadgePill({ badge }: { badge: CandidateBadge }) {
  const isSignature = badge.kind === "signature";
  const isDiscovered = badge.kind === "discovered";
  const isLow = badge.name.startsWith("Low ");

  // Color selection: discovered badges use teal, signature uses muted gray,
  // catalog badges use the curated color map.
  const color = isDiscovered
    ? DISCOVERED_BADGE_COLOR
    : isSignature
    ? DEFAULT_BADGE_COLOR
    : getBadgeColor(badge.name);

  // Strip the "PCA: " prefix for display to keep pills compact.
  const displayName = isDiscovered
    ? badge.name.replace(/^PCA:\s+/, "")
    : badge.name;

  // Score formatting: discovered badges show PCA score as σ units;
  // signature badges show z-scores; catalog shows percentage points.
  const scoreFormatted = isDiscovered || isSignature
    ? `${badge.score >= 0 ? "+" : ""}${badge.score.toFixed(3)} vs party peers`
    : badge.score > 0
    ? `+${(badge.score * 100).toFixed(1)}pp vs expected`
    : `${(badge.score * 100).toFixed(1)}pp vs expected`;

  const tooltipSuffix = badge.provisional ? " (1 race — provisional)" : "";

  // Build a richer tooltip for discovered badges that shows their demographic basis.
  const discoveredDemoSuffix =
    isDiscovered && badge.top_demographics && badge.top_demographics.length > 0
      ? ` | Key dims: ${badge.top_demographics
          .slice(0, 2)
          .map(([col, corr]) => `${col} (r=${corr >= 0 ? "+" : ""}${corr.toFixed(2)})`)
          .join(", ")}`
      : "";

  // Discovered badges always get a dashed border to signal algorithmic origin.
  const borderStyle = isDiscovered || badge.provisional ? "1px dashed" : "1px solid";

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger
          render={
            <span
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "3px",
                padding: "1px 8px",
                borderRadius: "9999px",
                fontSize: "0.65rem",
                fontWeight: 500,
                letterSpacing: "0.02em",
                cursor: "default",
                userSelect: "none",
                background: color.bg,
                color: color.text,
                border: `${borderStyle} ${color.border}`,
                opacity: isLow ? 0.8 : 1,
              }}
            >
              {isLow && !isSignature && !isDiscovered && (
                <span aria-hidden="true" style={{ fontSize: "0.55rem" }}>&#9661;</span>
              )}
              {isDiscovered && (
                <span aria-hidden="true" style={{ fontSize: "0.55rem", marginRight: "2px" }}>&#9670;</span>
              )}
              {displayName}
            </span>
          }
        />
        <TooltipContent side="top">
          <span style={{ fontSize: "0.75rem" }}>
            {badge.name}: {scoreFormatted}{tooltipSuffix}{discoveredDemoSuffix}
            {isDiscovered && " [data-driven]"}
          </span>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

function CandidateRow({ candidate }: { candidate: RaceCandidateSummary }) {
  const partyColor = PARTY_LABEL_COLORS[candidate.party] ?? "var(--color-text-muted)";

  return (
    <div style={{ display: "flex", gap: "12px", alignItems: "flex-start" }}>
      {/* Radar chart — only renders when there are ≥3 meaningful badge dimensions */}
      {candidate.badge_scores && Object.keys(candidate.badge_scores).length > 0 && (
        <div style={{ flexShrink: 0, paddingTop: "2px" }}>
          <CTOVRadarChart
            badgeScores={candidate.badge_scores}
            party={candidate.party}
            size={120}
          />
        </div>
      )}

      {/* Name, CEC indicator, and badge pills */}
      <div className="flex flex-col gap-1.5" style={{ flex: 1, minWidth: 0 }}>
        <div className="flex items-center gap-2">
          <span
            className="text-sm font-semibold"
            style={{ color: partyColor }}
          >
            {candidate.bioguide_id ? (
              <Link
                href={`/candidates/${candidate.bioguide_id}`}
                style={{
                  color: "inherit",
                  textDecoration: "none",
                }}
                className="hover:underline"
              >
                {candidate.name}
              </Link>
            ) : (
              candidate.name
            )}
            <span
              className="ml-1 text-xs font-normal"
              style={{ color: "var(--color-text-muted)" }}
            >
              ({candidate.party})
            </span>
          </span>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger
                render={
                  <span
                    style={{
                      display: "inline-flex",
                      alignItems: "center",
                      padding: "0px 5px",
                      borderRadius: "4px",
                      fontSize: "0.6rem",
                      fontWeight: 600,
                      letterSpacing: "0.04em",
                      cursor: "default",
                      userSelect: "none",
                      background: "rgba(148, 163, 184, 0.08)",
                      color: "var(--color-text-muted)",
                      border: "1px solid rgba(148, 163, 184, 0.20)",
                    }}
                  >
                    CEC {formatCEC(candidate.cec)}
                  </span>
                }
              />
              <TooltipContent side="top">
                <span style={{ fontSize: "0.75rem" }}>
                  Candidate Effect Consistency: {formatCEC(candidate.cec)} --
                  how stable this candidate&apos;s performance pattern is across races.
                  Higher = more predictable.
                </span>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
        {candidate.badges.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {candidate.badges.map((badge) => (
              <BadgePill key={badge.name} badge={badge} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export function CandidateBadges({ raceKey }: CandidateBadgesProps) {
  const { data, isLoading, error } = useRaceCandidates(raceKey);

  // Don't render anything if loading, error, or no candidates have badges
  if (isLoading || error || !data) return null;
  if (data.candidates.length === 0) return null;

  return (
    <section
      className="mb-6 rounded-md px-4 py-3"
      style={{
        background: "var(--color-surface)",
        border: "1px solid var(--color-border)",
      }}
    >
      <h3
        className="text-xs font-semibold mb-2.5 tracking-wide uppercase"
        style={{ color: "var(--color-text-muted)" }}
      >
        Candidate Performance Profile
      </h3>
      <div className="space-y-3">
        {data.candidates.map((candidate) => (
          <CandidateRow key={candidate.bioguide_id} candidate={candidate} />
        ))}
      </div>
    </section>
  );
}
