/**
 * ShiftHistoryChart — visx line chart showing presidential Dem shift over cycles.
 *
 * Displays how the type's mean Democratic presidential vote share has shifted
 * across election cycles (e.g., '08→'12, '12→'16, etc.).
 *
 * A midpoint line at 0 separates rightward from leftward cycles. Positive values
 * indicate a Democratic gain; negative values indicate a Republican gain.
 */

"use client";

import { useMemo } from "react";
import { scaleLinear, scalePoint } from "@visx/scale";
import { LinePath } from "@visx/shape";
import { AxisBottom, AxisLeft } from "@visx/axis";
import { Group } from "@visx/group";
import { curveMonotoneX } from "@visx/curve";
import { DUSTY_INK } from "@/lib/config/palette";
import { formatMargin } from "@/lib/format";

interface ShiftHistoryChartProps {
  shiftProfile: Record<string, number>;
}

interface ShiftPoint {
  label: string;
  sortKey: string;
  value: number;
}

const MARGIN = { top: 36, right: 24, bottom: 40, left: 56 };

/**
 * Notable cycle annotations shown above inflection points.
 * Key matches the axis label (e.g., "'04→'08") from parseShiftKey.
 */
const CYCLE_ANNOTATIONS: Record<string, string> = {
  "'04→'08": "Obama",
  "'12→'16": "Trump",
  "'16→'20": "Biden",
};
const CHART_HEIGHT = 200;

/** Parse a shift field key into a display label and sort key. */
function parseShiftKey(key: string): { label: string; sortKey: string } | null {
  const match = /^pres_d_shift_(\d{2})_(\d{2})$/.exec(key);
  if (!match) return null;
  const [, from, to] = match;
  return {
    label: `'${from}→'${to}`,
    sortKey: `${from}_${to}`,
  };
}

export function ShiftHistoryChart({ shiftProfile }: ShiftHistoryChartProps) {
  const points: ShiftPoint[] = useMemo(() => {
    const raw: ShiftPoint[] = [];
    for (const [key, value] of Object.entries(shiftProfile)) {
      const parsed = parseShiftKey(key);
      if (!parsed) continue;
      raw.push({ label: parsed.label, sortKey: parsed.sortKey, value });
    }
    return raw.sort((a, b) => a.sortKey.localeCompare(b.sortKey));
  }, [shiftProfile]);

  if (points.length === 0) {
    return (
      <p style={{ color: "var(--color-text-muted)", fontSize: 14 }}>
        No shift data available.
      </p>
    );
  }

  const width = 680; // intrinsic width; ResponsiveContainer would be ideal but adds complexity
  const innerWidth = width - MARGIN.left - MARGIN.right;
  const innerHeight = CHART_HEIGHT - MARGIN.top - MARGIN.bottom;

  const labels = points.map((p) => p.label);
  const values = points.map((p) => p.value);
  const minVal = Math.min(...values, -0.05);
  const maxVal = Math.max(...values, 0.05);
  // Symmetric around zero with some padding
  const extent = Math.max(Math.abs(minVal), Math.abs(maxVal)) * 1.15;

  const xScale = scalePoint({ domain: labels, range: [0, innerWidth], padding: 0.2 });
  const yScale = scaleLinear({ domain: [-extent, extent], range: [innerHeight, 0], nice: true });

  const zeroY = yScale(0);

  return (
    <div>
      <svg
        viewBox={`0 0 ${width} ${CHART_HEIGHT}`}
        style={{ width: "100%", height: "auto", overflow: "visible" }}
        aria-label="Electoral shift line chart"
      >
        <Group left={MARGIN.left} top={MARGIN.top}>
          {/* Zero line */}
          <line
            x1={0}
            x2={innerWidth}
            y1={zeroY}
            y2={zeroY}
            stroke="var(--color-border)"
            strokeWidth={1}
            strokeDasharray="4 3"
          />

          {/* Partisan fill under the curve */}
          {points.map((pt, i) => {
            if (i === 0) return null;
            const prev = points[i - 1];
            const x0 = xScale(prev.label) ?? 0;
            const x1 = xScale(pt.label) ?? 0;
            const y0 = yScale(prev.value);
            const y1 = yScale(pt.value);
            const isDem = (pt.value + prev.value) / 2 >= 0;
            return (
              <polygon
                key={pt.sortKey}
                points={`${x0},${zeroY} ${x0},${y0} ${x1},${y1} ${x1},${zeroY}`}
                fill={isDem ? DUSTY_INK.leanD : DUSTY_INK.leanR}
                opacity={0.12}
              />
            );
          })}

          {/* Main line */}
          <LinePath
            data={points}
            x={(p) => xScale(p.label) ?? 0}
            y={(p) => yScale(p.value)}
            stroke={points[points.length - 1]?.value >= 0 ? DUSTY_INK.leanD : DUSTY_INK.leanR}
            strokeWidth={2}
            curve={curveMonotoneX}
          />

          {/* Data points */}
          {points.map((pt) => {
            const cx = xScale(pt.label) ?? 0;
            const cy = yScale(pt.value);
            const isDem = pt.value >= 0;
            return (
              <g key={pt.sortKey}>
                <circle
                  cx={cx}
                  cy={cy}
                  r={4}
                  fill={isDem ? DUSTY_INK.leanD : DUSTY_INK.leanR}
                  stroke="var(--color-bg)"
                  strokeWidth={1.5}
                />
              </g>
            );
          })}

          {/* Event annotations above notable inflection points */}
          {points.map((pt) => {
            const annotation = CYCLE_ANNOTATIONS[pt.label];
            if (!annotation) return null;
            const cx = xScale(pt.label) ?? 0;
            return (
              <text
                key={`anno-${pt.sortKey}`}
                x={cx}
                y={-8}
                textAnchor="middle"
                fontSize={9}
                fontFamily="var(--font-sans)"
                fill="var(--color-text-subtle, var(--color-text-muted))"
                opacity={0.65}
              >
                {annotation}
              </text>
            );
          })}

          {/* Axes */}
          <AxisBottom
            scale={xScale}
            top={innerHeight}
            tickLabelProps={{
              fontSize: 11,
              fill: "var(--color-text-muted)",
              textAnchor: "middle",
              fontFamily: "var(--font-sans)",
            }}
            stroke="var(--color-border)"
            tickStroke="var(--color-border)"
            tickLength={4}
          />
          <AxisLeft
            scale={yScale}
            tickFormat={(v) => formatMargin(0.5 + Number(v))}
            tickLabelProps={{
              fontSize: 10,
              fill: "var(--color-text-muted)",
              textAnchor: "end",
              dx: -4,
              fontFamily: "var(--font-sans)",
            }}
            numTicks={5}
            stroke="var(--color-border)"
            tickStroke="var(--color-border)"
            tickLength={4}
          />
        </Group>
      </svg>
      <p
        style={{
          fontSize: 12,
          color: "var(--color-text-subtle, var(--color-text-muted))",
          marginTop: 4,
        }}
      >
        Presidential Dem shift by cycle — mean across member counties. Positive = Dem gain.
      </p>
    </div>
  );
}
