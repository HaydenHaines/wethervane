"use client";

import Link from "next/link";
import { getSuperTypeColor, rgbToHex } from "@/lib/config/palette";
import { formatMargin } from "@/lib/format";

export interface StateTypeItem {
  type_id: number;
  super_type_id: number;
  display_name: string;
  n_counties: number;
  mean_pred_dem_share: number | null;
}

interface StateTypeDistributionProps {
  types: StateTypeItem[];
  totalCounties: number;
}

/**
 * Horizontal stacked bar showing each type's share of the state's counties,
 * plus a table of types sorted by county count.
 *
 * The stacked bar gives a quick visual overview of which electoral communities
 * dominate the state; the table provides the detail.
 */
export function StateTypeDistribution({ types, totalCounties }: StateTypeDistributionProps) {
  if (types.length === 0) {
    return (
      <p className="text-sm italic" style={{ color: "var(--color-text-muted)" }}>
        No type data available.
      </p>
    );
  }

  // Sort by county count descending
  const sorted = [...types].sort((a, b) => b.n_counties - a.n_counties);

  return (
    <div>
      {/* Stacked bar */}
      <div
        style={{
          display: "flex",
          height: 20,
          borderRadius: 6,
          overflow: "hidden",
          marginBottom: 16,
          gap: 1,
        }}
        role="img"
        aria-label="Type distribution stacked bar"
      >
        {sorted.map((t) => {
          const pct = totalCounties > 0 ? (t.n_counties / totalCounties) * 100 : 0;
          const color = rgbToHex(getSuperTypeColor(t.super_type_id));
          return (
            <div
              key={t.type_id}
              title={`${t.display_name}: ${t.n_counties} counties (${pct.toFixed(1)}%)`}
              style={{
                width: `${pct}%`,
                background: color,
                flexShrink: 0,
              }}
            />
          );
        })}
      </div>

      {/* Type rows */}
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {sorted.map((t) => {
          const pct = totalCounties > 0 ? (t.n_counties / totalCounties) * 100 : 0;
          const color = rgbToHex(getSuperTypeColor(t.super_type_id));
          const isDem = t.mean_pred_dem_share !== null && t.mean_pred_dem_share > 0.505;
          const isGop = t.mean_pred_dem_share !== null && t.mean_pred_dem_share < 0.495;
          const marginColor = isDem
            ? "var(--forecast-safe-d)"
            : isGop
            ? "var(--forecast-safe-r)"
            : "var(--forecast-tossup)";

          return (
            <div
              key={t.type_id}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 12,
                padding: "6px 12px",
                borderRadius: 6,
                background: "var(--color-surface)",
                border: "1px solid var(--color-border)",
                borderLeft: `3px solid ${color}`,
              }}
            >
              {/* Color swatch */}
              <div
                style={{
                  width: 10,
                  height: 10,
                  borderRadius: "50%",
                  background: color,
                  flexShrink: 0,
                }}
              />

              {/* Name + count */}
              <div style={{ flex: 1, minWidth: 0 }}>
                <Link
                  href={`/type/${t.type_id}`}
                  style={{
                    color: "var(--color-dem)",
                    textDecoration: "none",
                    fontSize: 13,
                    fontWeight: 600,
                  }}
                >
                  {t.display_name}
                </Link>
              </div>

              {/* County count + pct */}
              <span style={{ fontSize: 12, color: "var(--color-text-muted)", whiteSpace: "nowrap" }}>
                {t.n_counties} {t.n_counties === 1 ? "county" : "counties"} ({pct.toFixed(0)}%)
              </span>

              {/* Lean */}
              <span
                style={{
                  fontFamily: "var(--font-mono, monospace)",
                  fontSize: 13,
                  fontWeight: 700,
                  color: marginColor,
                  whiteSpace: "nowrap",
                  minWidth: 52,
                  textAlign: "right",
                }}
              >
                {formatMargin(t.mean_pred_dem_share)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
