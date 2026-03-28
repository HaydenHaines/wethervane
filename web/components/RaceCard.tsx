"use client";
import { DUSTY_INK, ratingColor, ratingLabel, type Rating } from "@/lib/colors";

export interface RaceCardProps {
  state: string;
  race: string;
  slug: string;
  rating: Rating;
  margin: number;
  nPolls: number;
  onClick: () => void;
}

function formatMargin(margin: number): string {
  const abs = Math.abs(margin);
  if (abs < 0.5) return "EVEN";
  const party = margin > 0 ? "D" : "R";
  return `${party}+${abs.toFixed(1)}`;
}

export function RaceCard({ state, race, slug, rating, margin, nPolls, onClick }: RaceCardProps) {
  const color = ratingColor(rating);
  const label = ratingLabel(rating);

  // Extract office from race string: "2026 FL Senate" → "Senate"
  const parts = race.split(/\s+/);
  const office = parts.length >= 3 ? parts.slice(2).join(" ") : race;

  return (
    <button
      onClick={onClick}
      style={{
        display: "block",
        width: "100%",
        textAlign: "left",
        padding: "14px 16px",
        background: DUSTY_INK.cardBg,
        border: "1px solid " + DUSTY_INK.border,
        borderLeft: `4px solid ${color}`,
        borderRadius: "4px",
        cursor: "pointer",
        transition: "background 0.15s ease",
      }}
      onMouseEnter={(e) => { e.currentTarget.style.background = DUSTY_INK.border; }}
      onMouseLeave={(e) => { e.currentTarget.style.background = DUSTY_INK.cardBg; }}
    >
      {/* State name */}
      <div style={{
        fontFamily: "var(--font-serif)",
        fontSize: "16px",
        fontWeight: 700,
        color: DUSTY_INK.text,
        marginBottom: "4px",
      }}>
        {state} {office}
      </div>

      {/* Rating badge + margin + poll count */}
      <div style={{
        display: "flex",
        alignItems: "center",
        gap: "10px",
        flexWrap: "wrap",
      }}>
        <span style={{
          display: "inline-block",
          padding: "2px 8px",
          borderRadius: "10px",
          fontSize: "11px",
          fontFamily: "var(--font-sans)",
          fontWeight: 600,
          color: "#fff",
          background: color,
          letterSpacing: "0.3px",
        }}>
          {label}
        </span>

        <span style={{
          fontFamily: "var(--font-sans)",
          fontSize: "13px",
          fontWeight: 600,
          color: DUSTY_INK.text,
        }}>
          {formatMargin(margin)}
        </span>

        {nPolls > 0 && (
          <span style={{
            fontFamily: "var(--font-sans)",
            fontSize: "11px",
            color: DUSTY_INK.textMuted,
          }}>
            {nPolls} poll{nPolls !== 1 ? "s" : ""}
          </span>
        )}
      </div>
    </button>
  );
}
