"use client";

import { useState, useMemo } from "react";
import Link from "next/link";
import { DUSTY_INK } from "@/lib/config/palette";

// ── Types ──────────────────────────────────────────────────────────────────

interface WetherVaneData {
  pred_dem_share: number | null;
  pred_std: number | null;
  rating: string | null;
  n_counties: number | null;
}

interface RaceRow {
  race_id: string;
  slug: string;
  year: number;
  state_abbr: string;
  race_type: string;
  wethervane: WetherVaneData;
  cook: string | null;
  sabato: string | null;
  inside: string | null;
}

interface ComparisonTableProps {
  races: RaceRow[];
}

// ── Rating helpers ─────────────────────────────────────────────────────────

/**
 * Normalize any external rating string to one of our 7 canonical keys.
 * External forecasters use slightly different labels (Solid vs Safe, Toss-up vs Tossup).
 */
function normalizeRating(raw: string | null): string | null {
  if (!raw) return null;
  const lower = raw.toLowerCase().replace(/[-\s]+/g, "_");
  const MAP: Record<string, string> = {
    solid_d:  "safe_d",
    safe_d:   "safe_d",
    likely_d: "likely_d",
    lean_d:   "lean_d",
    tilt_d:   "lean_d",   // Inside Elections "Tilt D" — treat as lean
    toss_up:  "tossup",
    tossup:   "tossup",
    tilt_r:   "lean_r",   // Inside Elections "Tilt R" — treat as lean
    lean_r:   "lean_r",
    likely_r: "likely_r",
    solid_r:  "safe_r",
    safe_r:   "safe_r",
  };
  return MAP[lower] ?? null;
}

const RATING_COLORS: Record<string, string> = {
  safe_d:   DUSTY_INK.safeD,
  likely_d: DUSTY_INK.likelyD,
  lean_d:   DUSTY_INK.leanD,
  tossup:   DUSTY_INK.tossup,
  lean_r:   DUSTY_INK.leanR,
  likely_r: DUSTY_INK.likelyR,
  safe_r:   DUSTY_INK.safeR,
};

const RATING_DISPLAY: Record<string, string> = {
  safe_d:   "Safe D",
  likely_d: "Likely D",
  lean_d:   "Lean D",
  tossup:   "Toss-up",
  lean_r:   "Lean R",
  likely_r: "Likely R",
  safe_r:   "Safe R",
};

/** Absolute distance from 50% — smaller = more competitive. */
function competitivenessScore(row: RaceRow): number {
  if (row.wethervane.pred_dem_share !== null) {
    return Math.abs(row.wethervane.pred_dem_share - 0.5);
  }
  // Fallback to Cook rating competitiveness order
  const RATING_DIST: Record<string, number> = {
    tossup: 0.01, lean_d: 0.05, lean_r: 0.05,
    likely_d: 0.10, likely_r: 0.10, safe_d: 0.25, safe_r: 0.25,
  };
  const cook = normalizeRating(row.cook);
  return cook ? (RATING_DIST[cook] ?? 0.30) : 0.35;
}

// ── Rating cell ────────────────────────────────────────────────────────────

function RatingCell({ raw }: { raw: string | null }) {
  if (!raw) {
    return (
      <span style={{ color: "var(--color-text-subtle)", fontSize: 12 }}>—</span>
    );
  }
  const key = normalizeRating(raw);
  const color = key ? (RATING_COLORS[key] ?? "var(--color-text-muted)") : "var(--color-text-muted)";
  return (
    <span
      style={{
        display: "inline-block",
        padding: "2px 8px",
        borderRadius: 4,
        fontSize: 12,
        fontWeight: 600,
        background: color,
        color: "#fff",
        whiteSpace: "nowrap",
      }}
    >
      {raw}
    </span>
  );
}

function WetherVaneCell({ data }: { data: WetherVaneData }) {
  if (data.pred_dem_share === null || data.rating === null) {
    return (
      <span style={{ color: "var(--color-text-subtle)", fontSize: 12 }}>—</span>
    );
  }
  const pct = (data.pred_dem_share * 100).toFixed(1);
  const margin = data.pred_dem_share - 0.5;
  const marginPp = (Math.abs(margin) * 100).toFixed(1);
  const party = margin >= 0 ? "D" : "R";
  const color = RATING_COLORS[data.rating] ?? DUSTY_INK.tossup;
  const displayLabel = RATING_DISPLAY[data.rating] ?? data.rating;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <span
        style={{
          display: "inline-block",
          padding: "2px 8px",
          borderRadius: 4,
          fontSize: 12,
          fontWeight: 600,
          background: color,
          color: "#fff",
          whiteSpace: "nowrap",
        }}
      >
        {displayLabel}
      </span>
      <span style={{ fontSize: 11, color: "var(--color-text-muted)" }}>
        {pct}% D &middot; {party}+{marginPp}
      </span>
    </div>
  );
}

// ── Filter types ───────────────────────────────────────────────────────────

type RaceTypeFilter = "all" | "senate" | "governor";

// ── Main component ─────────────────────────────────────────────────────────

export function ComparisonTable({ races }: ComparisonTableProps) {
  const [raceTypeFilter, setRaceTypeFilter] = useState<RaceTypeFilter>("all");
  const [showOnlyCompetitive, setShowOnlyCompetitive] = useState(false);

  const filtered = useMemo(() => {
    let result = [...races];

    if (raceTypeFilter === "senate") {
      result = result.filter((r) => r.race_type.toLowerCase() === "senate");
    } else if (raceTypeFilter === "governor") {
      result = result.filter((r) => r.race_type.toLowerCase() === "governor");
    }

    if (showOnlyCompetitive) {
      const competitive = new Set(["tossup", "lean_d", "lean_r", "likely_d", "likely_r"]);
      result = result.filter((r) => {
        const wvRating = r.wethervane.rating;
        const cookNorm = normalizeRating(r.cook);
        return (
          (wvRating && competitive.has(wvRating)) ||
          (cookNorm && competitive.has(cookNorm))
        );
      });
    }

    result.sort((a, b) => competitivenessScore(a) - competitivenessScore(b));
    return result;
  }, [races, raceTypeFilter, showOnlyCompetitive]);

  const cellStyle: React.CSSProperties = {
    padding: "10px 14px",
    textAlign: "center" as const,
  };

  const headerCellStyle: React.CSSProperties = {
    padding: "10px 14px",
    textAlign: "center" as const,
    fontWeight: 600,
    color: "var(--color-text-muted)",
    whiteSpace: "nowrap" as const,
    minWidth: 110,
  };

  return (
    <div>
      {/* Filter controls */}
      <div
        className="mb-4 flex flex-wrap gap-3 items-center"
        style={{ fontSize: 13 }}
      >
        <div className="flex gap-1">
          {(["all", "senate", "governor"] as RaceTypeFilter[]).map((f) => (
            <button
              key={f}
              onClick={() => setRaceTypeFilter(f)}
              style={{
                padding: "4px 12px",
                borderRadius: 4,
                border: "1px solid var(--color-border)",
                background:
                  raceTypeFilter === f ? "var(--color-text)" : "transparent",
                color:
                  raceTypeFilter === f
                    ? "var(--color-bg)"
                    : "var(--color-text-muted)",
                cursor: "pointer",
                fontSize: 12,
                fontWeight: 500,
              }}
            >
              {f === "all" ? "All Races" : f === "senate" ? "Senate" : "Governor"}
            </button>
          ))}
        </div>

        <label
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
            cursor: "pointer",
            color: "var(--color-text-muted)",
            userSelect: "none",
          }}
        >
          <input
            type="checkbox"
            checked={showOnlyCompetitive}
            onChange={(e) => setShowOnlyCompetitive(e.target.checked)}
            style={{ cursor: "pointer" }}
          />
          Competitive only (Toss-up / Lean / Likely)
        </label>

        <span
          style={{
            marginLeft: "auto",
            color: "var(--color-text-subtle)",
            fontSize: 12,
          }}
        >
          {filtered.length} race{filtered.length !== 1 ? "s" : ""}
        </span>
      </div>

      {/* Responsive table */}
      <div
        style={{
          overflowX: "auto",
          borderRadius: 8,
          border: "1px solid var(--color-border)",
        }}
      >
        <table
          style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}
        >
          <thead>
            <tr
              style={{
                borderBottom: "2px solid var(--color-border)",
                background: "var(--color-surface)",
              }}
            >
              {/* Race column — sticky so state name stays visible on mobile scroll */}
              <th
                style={{
                  padding: "10px 14px",
                  textAlign: "left",
                  fontWeight: 600,
                  color: "var(--color-text)",
                  whiteSpace: "nowrap",
                  position: "sticky",
                  left: 0,
                  background: "var(--color-surface)",
                  zIndex: 2,
                  minWidth: 120,
                }}
              >
                Race
              </th>
              <th
                style={{
                  ...headerCellStyle,
                  color: "var(--color-text)",
                  minWidth: 140,
                }}
              >
                WetherVane
              </th>
              <th style={headerCellStyle}>Cook</th>
              <th style={headerCellStyle}>Sabato</th>
              <th style={headerCellStyle}>Inside Elections</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((row, i) => {
              const rowBg =
                i % 2 === 0 ? "var(--color-bg)" : "var(--color-surface)";
              return (
                <tr
                  key={row.race_id}
                  style={{
                    borderBottom:
                      i < filtered.length - 1
                        ? "1px solid var(--color-border)"
                        : "none",
                    background: rowBg,
                  }}
                >
                  {/* Race name — sticky */}
                  <td
                    style={{
                      padding: "10px 14px",
                      position: "sticky",
                      left: 0,
                      background: rowBg,
                      zIndex: 1,
                    }}
                  >
                    <Link
                      href={`/forecast/${row.slug}`}
                      style={{
                        textDecoration: "none",
                        color: "var(--color-text)",
                        fontWeight: 500,
                      }}
                    >
                      <span
                        style={{
                          color: "var(--color-text-muted)",
                          marginRight: 4,
                          fontSize: 11,
                        }}
                      >
                        {row.state_abbr}
                      </span>
                      {row.race_type}
                    </Link>
                  </td>

                  <td style={cellStyle}>
                    <WetherVaneCell data={row.wethervane} />
                  </td>
                  <td style={cellStyle}>
                    <RatingCell raw={row.cook} />
                  </td>
                  <td style={cellStyle}>
                    <RatingCell raw={row.sabato} />
                  </td>
                  <td style={cellStyle}>
                    <RatingCell raw={row.inside} />
                  </td>
                </tr>
              );
            })}

            {filtered.length === 0 && (
              <tr>
                <td
                  colSpan={5}
                  style={{
                    padding: "32px",
                    textAlign: "center",
                    color: "var(--color-text-muted)",
                  }}
                >
                  No races match the current filters.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
