"use client";

import { useMemo } from "react";
import { scaleLinear, scalePoint } from "@visx/scale";
import { LinePath } from "@visx/shape";
import { Group } from "@visx/group";
import { AxisLeft, AxisBottom } from "@visx/axis";
import { useTypeScatter } from "@/lib/hooks/use-type-scatter";
import { useSuperTypes } from "@/lib/hooks/use-super-types";
import { getSuperTypeColor, rgbToHex } from "@/lib/config/palette";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/**
 * Presidential Dem shift cycles in chronological order.
 * Each entry is the suffix used in the API's shift_profile keys.
 */
const PRES_CYCLES: { suffix: string; label: string }[] = [
  { suffix: "00_04", label: "'04" },
  { suffix: "04_08", label: "'08" },
  { suffix: "08_12", label: "'12" },
  { suffix: "12_16", label: "'16" },
  { suffix: "16_20", label: "'20" },
  { suffix: "20_24", label: "'24" },
];

const CHART_WIDTH = 240;
const CHART_HEIGHT = 140;
const MARGIN = { top: 16, right: 12, bottom: 32, left: 40 };

const INNER_WIDTH = CHART_WIDTH - MARGIN.left - MARGIN.right;
const INNER_HEIGHT = CHART_HEIGHT - MARGIN.top - MARGIN.bottom;

/** Y-axis extent cap: clamp at ±25pp to prevent outlier distortion. */
const Y_CAP = 0.25;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ShiftPoint {
  label: string;
  value: number;
}

interface SuperTypeShifts {
  superTypeId: number;
  displayName: string;
  color: string;
  points: ShiftPoint[];
}

// ---------------------------------------------------------------------------
// Mini chart
// ---------------------------------------------------------------------------

interface MiniChartProps {
  entry: SuperTypeShifts;
  yScale: ReturnType<typeof scaleLinear<number>>;
  xScale: ReturnType<typeof scalePoint<string>>;
  yZero: number;
}

function MiniChart({ entry, yScale, xScale, yZero }: MiniChartProps) {
  const { displayName, color, points } = entry;

  return (
    <div
      style={{
        border: "1px solid var(--color-border)",
        borderRadius: 8,
        padding: "12px 8px 4px",
        background: "var(--color-surface)",
      }}
    >
      {/* Super-type label */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 6,
          marginBottom: 4,
          paddingLeft: MARGIN.left,
        }}
      >
        <span
          style={{
            display: "inline-block",
            width: 10,
            height: 10,
            borderRadius: "50%",
            background: color,
            flexShrink: 0,
          }}
        />
        <span
          style={{
            fontSize: 11,
            fontWeight: 700,
            color: "var(--color-text)",
            lineHeight: 1.3,
          }}
        >
          {displayName}
        </span>
      </div>

      {/* visx chart */}
      <svg width={CHART_WIDTH} height={CHART_HEIGHT} style={{ display: "block" }}>
        <Group left={MARGIN.left} top={MARGIN.top}>
          {/* Zero reference line */}
          <line
            x1={0}
            x2={INNER_WIDTH}
            y1={yZero}
            y2={yZero}
            stroke="var(--color-border)"
            strokeWidth={1}
            strokeDasharray="4 3"
          />

          {/* Positive fill (Dem) */}
          <path
            d={buildAreaPath(points, xScale, yScale, yZero, true)}
            fill={color}
            fillOpacity={0.15}
          />

          {/* Negative fill (Rep) */}
          <path
            d={buildAreaPath(points, xScale, yScale, yZero, false)}
            fill="#c4707a"
            fillOpacity={0.12}
          />

          {/* Line */}
          <LinePath
            data={points}
            x={(d) => xScale(d.label) ?? 0}
            y={(d) => yScale(Math.max(-Y_CAP, Math.min(Y_CAP, d.value)))}
            stroke={color}
            strokeWidth={2}
            strokeLinejoin="round"
            strokeLinecap="round"
          />

          {/* Data points */}
          {points.map((d) => (
            <circle
              key={d.label}
              cx={xScale(d.label) ?? 0}
              cy={yScale(Math.max(-Y_CAP, Math.min(Y_CAP, d.value)))}
              r={3}
              fill={color}
              stroke="var(--color-surface)"
              strokeWidth={1.5}
            />
          ))}

          {/* Left axis */}
          <AxisLeft
            scale={yScale}
            numTicks={3}
            tickFormat={(v) => `${(Number(v) * 100).toFixed(0)}pp`}
            stroke="var(--color-border)"
            tickStroke="var(--color-border)"
            tickLabelProps={{ fontSize: 9, fill: "var(--color-text-muted)", dx: -2 }}
            tickLength={3}
          />

          {/* Bottom axis */}
          <AxisBottom
            top={INNER_HEIGHT}
            scale={xScale}
            stroke="var(--color-border)"
            tickStroke="var(--color-border)"
            tickLabelProps={{ fontSize: 9, fill: "var(--color-text-muted)", dy: 4 }}
            tickLength={3}
          />
        </Group>
      </svg>
    </div>
  );
}

/**
 * Build a filled area path between the line and the zero baseline.
 * `positive` = true renders above zero (Dem territory), false = below (Rep).
 */
function buildAreaPath(
  points: ShiftPoint[],
  xScale: ReturnType<typeof scalePoint<string>>,
  yScale: ReturnType<typeof scaleLinear<number>>,
  yZero: number,
  positive: boolean,
): string {
  if (points.length === 0) return "";

  const coords = points.map((d) => ({
    x: xScale(d.label) ?? 0,
    y: yScale(Math.max(-Y_CAP, Math.min(Y_CAP, d.value))),
    raw: d.value,
  }));

  let pathStr = "";

  for (let i = 0; i < coords.length; i++) {
    const { x, y, raw } = coords[i];
    const include = positive ? raw >= 0 : raw < 0;

    if (include) {
      if (pathStr === "") {
        // Start from baseline
        pathStr += `M ${x} ${yZero} L ${x} ${y}`;
      } else {
        pathStr += ` L ${x} ${y}`;
      }

      // If this is the last point in the segment, close back to baseline
      const next = coords[i + 1];
      if (!next || (positive ? next.raw < 0 : next.raw >= 0)) {
        pathStr += ` L ${x} ${yZero} Z`;
      }
    }
  }

  return pathStr;
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

/**
 * ShiftSmallMultiples — renders one mini visx LinePath chart per super-type.
 *
 * Shows presidential Dem shift across 2000→2024 election cycles.
 * All charts share the same Y scale to enable cross-super-type comparison.
 */
export function ShiftSmallMultiples() {
  const { data: scatter, isLoading: scatterLoading } = useTypeScatter();
  const { data: superTypes, isLoading: superTypesLoading } = useSuperTypes();

  const isLoading = scatterLoading || superTypesLoading;

  // Aggregate type-level shift data to super-type means
  const superTypeShifts = useMemo<SuperTypeShifts[]>(() => {
    if (!scatter || !superTypes) return [];

    // Build super-type name/color lookup
    const stMeta = new Map<number, { displayName: string; color: string }>(
      superTypes.map((st) => [
        st.super_type_id,
        {
          displayName: st.display_name,
          color: rgbToHex(getSuperTypeColor(st.super_type_id)),
        },
      ]),
    );

    // Accumulate per-cycle shift values per super-type
    const accumulator = new Map<number, Map<string, number[]>>();
    for (const pt of scatter) {
      const { super_type_id, shift_profile } = pt;
      if (!accumulator.has(super_type_id)) {
        accumulator.set(super_type_id, new Map());
      }
      const cycleMap = accumulator.get(super_type_id)!;

      for (const { suffix } of PRES_CYCLES) {
        const key = `pres_d_shift_${suffix}`;
        if (shift_profile[key] !== undefined) {
          if (!cycleMap.has(suffix)) cycleMap.set(suffix, []);
          cycleMap.get(suffix)!.push(shift_profile[key]);
        }
      }
    }

    // Convert accumulator to mean points, sorted by super_type_id
    return Array.from(accumulator.entries())
      .sort(([a], [b]) => a - b)
      .map(([superTypeId, cycleMap]) => {
        const meta = stMeta.get(superTypeId);
        const points: ShiftPoint[] = PRES_CYCLES.map(({ suffix, label }) => {
          const vals = cycleMap.get(suffix) ?? [];
          const mean = vals.length > 0 ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
          return { label, value: mean };
        });
        return {
          superTypeId,
          displayName: meta?.displayName ?? `Super-type ${superTypeId}`,
          color: meta?.color ?? "#888888",
          points,
        };
      });
  }, [scatter, superTypes]);

  // Shared Y scale — computed across all super-types for visual comparability
  const { yScale, yZero, xScale } = useMemo(() => {
    if (superTypeShifts.length === 0) {
      const yFallback = scaleLinear({ domain: [-0.2, 0.2], range: [INNER_HEIGHT, 0] });
      return {
        yScale: yFallback,
        yZero: yFallback(0),
        xScale: scalePoint({
          domain: PRES_CYCLES.map((c) => c.label),
          range: [0, INNER_WIDTH],
          padding: 0.2,
        }),
      };
    }

    const allValues = superTypeShifts.flatMap((st) => st.points.map((p) => p.value));
    const rawMin = Math.min(...allValues);
    const rawMax = Math.max(...allValues);

    // Clamp at cap, then add 15% padding
    const yMin = Math.max(-Y_CAP, rawMin);
    const yMax = Math.min(Y_CAP, rawMax);
    const pad = (yMax - yMin) * 0.15 || 0.02;

    const ys = scaleLinear({
      domain: [yMin - pad, yMax + pad],
      range: [INNER_HEIGHT, 0],
      nice: true,
    });

    const xs = scalePoint({
      domain: PRES_CYCLES.map((c) => c.label),
      range: [0, INNER_WIDTH],
      padding: 0.2,
    });

    return { yScale: ys, yZero: ys(0), xScale: xs };
  }, [superTypeShifts]);

  if (isLoading) {
    return (
      <div style={{ color: "var(--color-text-muted)", fontSize: 14, padding: "24px 0" }}>
        Loading shift data…
      </div>
    );
  }

  if (superTypeShifts.length === 0) {
    return (
      <div style={{ color: "var(--color-text-muted)", fontSize: 14, padding: "24px 0" }}>
        Shift data unavailable.
      </div>
    );
  }

  return (
    <div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))",
          gap: 16,
        }}
      >
        {superTypeShifts.map((entry) => (
          <MiniChart
            key={entry.superTypeId}
            entry={entry}
            yScale={yScale}
            xScale={xScale}
            yZero={yZero}
          />
        ))}
      </div>

      <p style={{ fontSize: 12, color: "var(--color-text-muted)", marginTop: 16 }}>
        Each panel shows the mean presidential Dem shift across cycles within that super-type family.
        Positive = Dem gain relative to prior cycle. Negative = Rep gain.
        Y-axis is shared across all panels — magnitudes are directly comparable.
      </p>
    </div>
  );
}
