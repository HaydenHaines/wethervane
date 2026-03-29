"use client";

/**
 * MapOverlayToggle — segmented control for switching between map overlay modes.
 *
 * Two modes:
 *  - "types"    : Stained glass super-type coloring (default)
 *  - "forecast" : Partisan-lean choropleth
 *
 * The active mode is communicated upward via `onChange`.
 * The parent (MapPage) is responsible for pushing the selection into MapContext.
 */

export type MapOverlay = "types" | "forecast";

interface MapOverlayToggleProps {
  value: MapOverlay;
  onChange: (overlay: MapOverlay) => void;
}

const OPTIONS: Array<{ value: MapOverlay; label: string }> = [
  { value: "types",    label: "Types" },
  { value: "forecast", label: "Forecast" },
];

export function MapOverlayToggle({ value, onChange }: MapOverlayToggleProps) {
  return (
    <div
      role="group"
      aria-label="Map overlay"
      style={{
        display: "inline-flex",
        borderRadius: 6,
        border: "1px solid var(--color-border, #e0ddd8)",
        overflow: "hidden",
        background: "var(--color-surface, #fafaf8)",
        fontSize: 12,
        fontFamily: "var(--font-sans)",
      }}
    >
      {OPTIONS.map((opt) => {
        const isActive = opt.value === value;
        return (
          <button
            key={opt.value}
            role="radio"
            aria-checked={isActive}
            onClick={() => onChange(opt.value)}
            style={{
              padding: "5px 14px",
              border: "none",
              borderRight: opt.value === "types" ? "1px solid var(--color-border, #e0ddd8)" : "none",
              background: isActive ? "var(--color-text, #3a3632)" : "transparent",
              color: isActive ? "var(--color-surface, #fafaf8)" : "var(--color-text-muted, #6e6860)",
              cursor: "pointer",
              fontFamily: "inherit",
              fontSize: "inherit",
              fontWeight: isActive ? 600 : 400,
              transition: "background 0.15s, color 0.15s",
            }}
          >
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}
