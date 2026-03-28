"use client";
import { useState } from "react";
import { DUSTY_INK, ratingColor, ratingLabel, type Rating } from "@/lib/colors";

export interface ControlBarRace {
  state: string;
  rating: Rating;
  margin: number;
  slug: string;
  race: string;
}

export interface SenateControlBarProps {
  races: ControlBarRace[];
  demSeats: number;
  gopSeats: number;
  onRaceClick: (slug: string, state?: string) => void;
}

const RATING_ORDER: Rating[] = [
  "safe_d", "likely_d", "lean_d", "tossup", "lean_r", "likely_r", "safe_r",
];

function formatMargin(margin: number): string {
  const abs = Math.abs(margin);
  if (abs < 0.5) return "EVEN";
  const party = margin > 0 ? "D" : "R";
  return `${party}+${abs.toFixed(1)}`;
}

export function SenateControlBar({ races, demSeats, gopSeats, onRaceClick }: SenateControlBarProps) {
  const [hoveredSlug, setHoveredSlug] = useState<string | null>(null);

  // Group races by rating, preserving order
  const grouped = new Map<Rating, ControlBarRace[]>();
  for (const r of RATING_ORDER) {
    grouped.set(r, []);
  }
  for (const race of races) {
    const bucket = grouped.get(race.rating as Rating);
    if (bucket) bucket.push(race);
  }

  // Count races per side
  const demRaces = races.filter(r =>
    r.rating === "safe_d" || r.rating === "likely_d" || r.rating === "lean_d"
  ).length;
  const gopRaces = races.filter(r =>
    r.rating === "safe_r" || r.rating === "likely_r" || r.rating === "lean_r"
  ).length;
  const tossupRaces = races.filter(r => r.rating === "tossup").length;

  return (
    <div style={{ marginBottom: "24px" }}>
      {/* Bar */}
      <div style={{
        display: "flex",
        height: "36px",
        borderRadius: "4px",
        overflow: "hidden",
        border: "1px solid " + DUSTY_INK.border,
      }}>
        {RATING_ORDER.map((rating) => {
          const bucket = grouped.get(rating) ?? [];
          if (bucket.length === 0) return null;
          const color = ratingColor(rating);
          const isTossup = rating === "tossup";

          return bucket.map((race) => (
            <div
              key={race.slug}
              onClick={() => onRaceClick(race.slug, race.state)}
              onMouseEnter={() => setHoveredSlug(race.slug)}
              onMouseLeave={() => setHoveredSlug(null)}
              style={{
                flex: isTossup ? 2 : 1,
                background: color,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                cursor: "pointer",
                position: "relative",
                opacity: hoveredSlug === race.slug ? 0.85 : 1,
                transition: "opacity 0.12s ease",
                borderRight: "1px solid rgba(255,255,255,0.25)",
              }}
            >
              <span style={{
                fontFamily: "var(--font-sans)",
                fontSize: "11px",
                fontWeight: 700,
                color: "#fff",
                letterSpacing: "0.5px",
                textShadow: "0 1px 2px rgba(0,0,0,0.3)",
              }}>
                {race.state}
              </span>

              {/* Tooltip */}
              {hoveredSlug === race.slug && (
                <div style={{
                  position: "absolute",
                  top: "calc(100% + 6px)",
                  left: "50%",
                  transform: "translateX(-50%)",
                  background: DUSTY_INK.text,
                  color: "#fff",
                  padding: "6px 10px",
                  borderRadius: "4px",
                  fontSize: "11px",
                  fontFamily: "var(--font-sans)",
                  whiteSpace: "nowrap",
                  zIndex: 100,
                  pointerEvents: "none",
                  lineHeight: 1.5,
                }}>
                  <div style={{ fontWeight: 700 }}>{race.race}</div>
                  <div>{ratingLabel(rating)} · {formatMargin(race.margin)}</div>
                </div>
              )}
            </div>
          ));
        })}
      </div>

      {/* Seat counts below bar */}
      <div style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        marginTop: "8px",
        fontSize: "12px",
        fontFamily: "var(--font-sans)",
        color: DUSTY_INK.textMuted,
      }}>
        <span>
          <strong style={{ color: DUSTY_INK.safeD }}>{demSeats}</strong>
          {demRaces > 0 && <> + {demRaces} race{demRaces !== 1 ? "s" : ""}</>}
        </span>
        <span style={{ fontSize: "11px", color: DUSTY_INK.textSubtle }}>
          50 for majority
        </span>
        <span>
          {gopRaces > 0 && <>{gopRaces} race{gopRaces !== 1 ? "s" : ""} + </>}
          <strong style={{ color: DUSTY_INK.safeR }}>{gopSeats}</strong>
        </span>
      </div>
    </div>
  );
}
