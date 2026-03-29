"use client";

/**
 * MapTooltip — floating tooltip that appears on hover over map features.
 *
 * Supports two display modes:
 *  - Single-line (state name): renders as a compact label.
 *  - Multi-line (tract details): renders a headline + secondary rows.
 *    Lean lines ("Lean: D+5") receive partisan color treatment.
 */

interface MapTooltipProps {
  x: number;
  y: number;
  /** Newline-delimited string. First line is the headline. */
  text: string;
}

export function MapTooltip({ x, y, text }: MapTooltipProps) {
  const lines = text.split("\n");
  const [headline, ...rest] = lines;

  return (
    <div
      style={{
        position: "absolute",
        left: x + 14,
        top: y + 14,
        background: "rgba(20, 24, 32, 0.93)",
        borderRadius: "6px",
        padding: "8px 12px",
        fontSize: "12px",
        fontFamily: "var(--font-sans)",
        pointerEvents: "none",
        boxShadow: "0 4px 12px rgba(0,0,0,0.35)",
        minWidth: "160px",
        maxWidth: "280px",
        zIndex: 20,
      }}
    >
      <div
        style={{
          color: "#f0f4f8",
          fontWeight: 600,
          fontSize: "13px",
          marginBottom: rest.length > 0 ? "4px" : 0,
        }}
      >
        {headline}
      </div>

      {rest.map((line, i) => {
        if (line.startsWith("Lean:")) {
          const lean = line.replace("Lean: ", "");
          const isDem = lean.startsWith("D");
          return (
            <div
              key={i}
              style={{
                color: isDem ? "#6baed6" : "#fc8d59",
                fontWeight: 700,
                fontSize: "13px",
                marginBottom: "4px",
              }}
            >
              {lean}
            </div>
          );
        }

        return (
          <div
            key={i}
            style={{
              color: "#b0bec5",
              fontSize: "11px",
              marginTop: i === 0 ? 0 : "2px",
            }}
          >
            {line}
          </div>
        );
      })}
    </div>
  );
}
