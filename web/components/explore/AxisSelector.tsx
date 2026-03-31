"use client";
import { prettifyKey } from "./ShiftScatterPlot";

// ── Axis option configuration ─────────────────────────────────────────────────

export const DEMO_KEYS: Array<{ key: string; label: string }> = [
  { key: "median_hh_income", label: "Median income" },
  { key: "median_age", label: "Median age" },
  { key: "pct_bachelors_plus", label: "Bachelor's+" },
  { key: "pct_white_nh", label: "White (non-Hispanic)" },
  { key: "pct_black", label: "Black" },
  { key: "pct_hispanic", label: "Hispanic" },
  { key: "pct_asian", label: "Asian" },
  { key: "pct_owner_occupied", label: "Owner-occupied" },
  { key: "pct_wfh", label: "Work from home" },
  { key: "pct_transit", label: "Transit commuters" },
  { key: "evangelical_share", label: "Evangelical" },
  { key: "mainline_share", label: "Mainline Protestant" },
  { key: "catholic_share", label: "Catholic" },
  { key: "black_protestant_share", label: "Black Protestant" },
  { key: "congregations_per_1000", label: "Congregations/1K" },
  { key: "log_pop_density", label: "Pop density (log)" },
  { key: "mean_pred_dem_share", label: "Predicted margin" },
];

// ── Props ─────────────────────────────────────────────────────────────────────

export interface AxisSelectorProps {
  xKey: string;
  yKey: string;
  shiftKeys: string[];
  onXChange: (key: string) => void;
  onYChange: (key: string) => void;
}

// ── Component ─────────────────────────────────────────────────────────────────

export function AxisSelector({
  xKey,
  yKey,
  shiftKeys,
  onXChange,
  onYChange,
}: AxisSelectorProps) {
  const selectStyle: React.CSSProperties = {
    fontSize: "12px",
    padding: "4px 8px",
    border: "1px solid var(--color-border)",
    borderRadius: 3,
    background: "var(--color-surface)",
    color: "var(--color-text)",
    flex: 1,
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "6px", marginBottom: "10px" }}>
      <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
        <span style={{ fontSize: "12px", color: "var(--color-text-muted)", width: 20 }}>X</span>
        <select
          value={xKey}
          onChange={(e) => onXChange(e.target.value)}
          style={selectStyle}
          aria-label="X-axis variable"
        >
          <optgroup label="Demographics">
            {DEMO_KEYS.map(({ key, label }) => (
              <option key={key} value={key}>{label}</option>
            ))}
          </optgroup>
          {shiftKeys.length > 0 && (
            <optgroup label="Shifts">
              {shiftKeys.map((key) => (
                <option key={key} value={key}>{prettifyKey(key)}</option>
              ))}
            </optgroup>
          )}
        </select>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
        <span style={{ fontSize: "12px", color: "var(--color-text-muted)", width: 20 }}>Y</span>
        <select
          value={yKey}
          onChange={(e) => onYChange(e.target.value)}
          style={selectStyle}
          aria-label="Y-axis variable"
        >
          {shiftKeys.length > 0 && (
            <optgroup label="Shifts">
              {shiftKeys.map((key) => (
                <option key={key} value={key}>{prettifyKey(key)}</option>
              ))}
            </optgroup>
          )}
          <optgroup label="Demographics">
            {DEMO_KEYS.map(({ key, label }) => (
              <option key={key} value={key}>{label}</option>
            ))}
          </optgroup>
        </select>
      </div>
    </div>
  );
}
