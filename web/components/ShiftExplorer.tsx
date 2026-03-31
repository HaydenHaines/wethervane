"use client";
import { useEffect, useState, useCallback } from "react";
import { fetchTypeScatterData, type TypeScatterPoint } from "@/lib/api";
import { useMapContext } from "@/components/MapContext";
import { ShiftScatterPlot, SuperTypeLegend, Tooltip } from "@/components/explore/ShiftScatterPlot";
import { AxisSelector } from "@/components/explore/AxisSelector";
import type { TooltipState, SuperTypeLegendEntry } from "@/components/explore/ShiftScatterPlot";

// Default axes: education (pct_bachelors_plus) vs. 2020→2024 presidential shift.
// These axes best illustrate the two primary dimensions of electoral type variance.
const DEFAULT_X_AXIS = "pct_bachelors_plus";
const DEFAULT_Y_AXIS = "pres_d_shift_20_24";

// ── Main component ────────────────────────────────────────────────────────────

export function ShiftExplorer() {
  const { compareTypeIds, addToComparison } = useMapContext();
  const [data, setData] = useState<TypeScatterPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [xKey, setXKey] = useState(DEFAULT_X_AXIS);
  const [yKey, setYKey] = useState(DEFAULT_Y_AXIS);
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);

  const selectedIds = new Set(compareTypeIds);

  const handleClick = useCallback((typeId: number) => {
    addToComparison(typeId);
  }, [addToComparison]);

  useEffect(() => {
    fetchTypeScatterData()
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  // Derive available shift keys from the first data point once loaded
  const shiftKeys = data.length > 0
    ? Object.keys(data[0].shift_profile).sort()
    : [];

  // Fall back to the first available shift key if the default yKey is not in the data
  const resolvedYKey = shiftKeys.includes(yKey) || yKey in (data[0]?.demographics ?? {})
    ? yKey
    : shiftKeys[0] ?? yKey;

  // Build unique, sorted super-type legend entries from the loaded data
  const superTypeEntries: SuperTypeLegendEntry[] = [];
  const seenSuperTypes = new Set<number>();
  for (const d of data) {
    if (!seenSuperTypes.has(d.super_type_id)) {
      seenSuperTypes.add(d.super_type_id);
      superTypeEntries.push({ super_type_id: d.super_type_id, name: `Super-type ${d.super_type_id + 1}` });
    }
  }
  superTypeEntries.sort((a, b) => a.super_type_id - b.super_type_id);

  return (
    <div style={{ padding: "12px 16px", minWidth: 0, overflow: "hidden" }}>
      <h3 style={{
        margin: "0 0 4px",
        fontFamily: "var(--font-serif)",
        fontSize: "16px",
      }}>
        Shift Explorer
      </h3>
      <p style={{ margin: "0 0 12px", fontSize: "12px", color: "var(--color-text-muted)" }}>
        Each dot is one of the 100 electoral types. Size = county count. Click a dot to compare.
      </p>

      {loading && (
        <div style={{ color: "var(--color-text-muted)", fontSize: "13px", padding: "20px 0" }}>
          Loading...
        </div>
      )}

      {error && (
        <div style={{ color: "var(--color-rep)", fontSize: "13px", padding: "8px 0" }}>
          Error: {error}
        </div>
      )}

      {!loading && !error && (
        <>
          <AxisSelector
            xKey={xKey}
            yKey={resolvedYKey}
            shiftKeys={shiftKeys}
            onXChange={setXKey}
            onYChange={setYKey}
          />

          {/* Scatter plot — click a dot to add it to comparison */}
          <ShiftScatterPlot
            data={data}
            xKey={xKey}
            yKey={resolvedYKey}
            onHover={setTooltip}
            onClick={handleClick}
            selectedIds={selectedIds}
          />

          <SuperTypeLegend entries={superTypeEntries} />

          <p style={{ margin: "4px 0 0", fontSize: "11px", color: "var(--color-text-muted)" }}>
            {data.length} types shown
          </p>
        </>
      )}

      {tooltip && <Tooltip state={tooltip} />}
    </div>
  );
}
