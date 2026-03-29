import Link from "next/link";
import { formatMargin } from "@/lib/format";
import { getSuperTypeColor, rgbToHex } from "@/lib/config/palette";

export interface TypeBreakdownItem {
  type_id: number;
  display_name: string;
  n_counties: number;
  mean_pred_dem_share: number | null;
}

interface TypesBreakdownProps {
  types: TypeBreakdownItem[];
  stateName: string;
}

/**
 * Types breakdown — shows the top electoral types in the state by county count.
 *
 * Each row links to /type/[id] and uses the super-type color from the palette.
 * Since the race detail API returns type_id but not super_type_id, we derive
 * a color from the type_id itself as a fallback (consistent but not semantically
 * meaningful). The color is primarily decorative — the link text is the primary signal.
 */
export function TypesBreakdown({ types, stateName }: TypesBreakdownProps) {
  if (types.length === 0) {
    return (
      <p className="text-sm italic" style={{ color: "var(--color-text-muted)" }}>
        No electoral type data available for this state.
      </p>
    );
  }

  return (
    <div>
      <p className="text-sm mb-3" style={{ color: "var(--color-text-muted)" }}>
        The most common electoral types in {stateName}, by county count.
        Each type&apos;s average model prediction is shown.
      </p>
      <div className="flex flex-col gap-2">
        {types.map((t) => {
          // Derive a color from type_id for consistent visual identity.
          // Ideally we'd use super_type_id; use type_id % 10 as fallback.
          const color = rgbToHex(getSuperTypeColor(t.type_id % 10));
          const isDem =
            t.mean_pred_dem_share !== null && t.mean_pred_dem_share > 0.505;
          const isGop =
            t.mean_pred_dem_share !== null && t.mean_pred_dem_share < 0.495;
          const marginColor = isDem
            ? "var(--forecast-safe-d)"
            : isGop
            ? "var(--forecast-safe-r)"
            : "var(--forecast-tossup)";

          return (
            <div
              key={t.type_id}
              className="flex items-center justify-between gap-4 rounded px-4 py-2"
              style={{
                background: "var(--color-surface)",
                border: "1px solid var(--color-border)",
                borderLeft: `3px solid ${color}`,
              }}
            >
              <div className="flex items-center gap-3 min-w-0">
                <Link
                  href={`/type/${t.type_id}`}
                  className="text-sm font-semibold truncate"
                  style={{ color: "var(--forecast-safe-d)", textDecoration: "none" }}
                >
                  {t.display_name}
                </Link>
                <span
                  className="text-xs flex-shrink-0"
                  style={{ color: "var(--color-text-muted)" }}
                >
                  {t.n_counties} {t.n_counties === 1 ? "county" : "counties"}
                </span>
              </div>
              <span
                className="font-mono text-sm font-bold flex-shrink-0"
                style={{ color: marginColor }}
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
