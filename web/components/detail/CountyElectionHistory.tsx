/**
 * CountyElectionHistory — visx line chart showing county Dem vote share by election cycle.
 *
 * Displays actual presidential, Senate, and gubernatorial results as separate
 * series so visitors can see the county's electoral trajectory over time.
 *
 * A midpoint line at 0.5 separates Democratic (>50%) from Republican (<50%) territory.
 */

"use client";

import { useMemo } from "react";
import { scaleLinear, scalePoint } from "@visx/scale";
import { LinePath } from "@visx/shape";
import { AxisBottom, AxisLeft } from "@visx/axis";
import { Group } from "@visx/group";
import { curveMonotoneX } from "@visx/curve";
import { DUSTY_INK } from "@/lib/config/palette";

export interface ElectionHistoryPoint {
  year: number;
  election_type: "president" | "senate" | "governor";
  dem_share: number;
  total_votes: number | null;
}

interface CountyElectionHistoryProps {
  history: ElectionHistoryPoint[];
}

const MARGIN = { top: 20, right: 24, bottom: 44, left: 60 };
const CHART_HEIGHT = 220;

const SERIES_COLORS: Record<string, string> = {
  president: DUSTY_INK.leanD,
  senate:    "#8B7BAF",   // muted purple
  governor:  "#7BAF8B",   // muted green
};

const SERIES_LABELS: Record<string, string> = {
  president: "President",
  senate:    "Senate",
  governor:  "Governor",
};

function formatShare(v: number): string {
  const margin = v - 0.5;
  const pp = Math.abs(margin * 100).toFixed(1);
  return margin >= 0 ? `D+${pp}` : `R+${pp}`;
}

function yTickFormat(v: { valueOf(): number }): string {
  // Show axis ticks as D+X / R+X around 0.5 midpoint
  const margin = v.valueOf() - 0.5;
  const pp = Math.abs(margin * 100).toFixed(0);
  if (Math.abs(margin) < 0.001) return "Even";
  return margin > 0 ? `D+${pp}` : `R+${pp}`;
}

export function CountyElectionHistory({ history }: CountyElectionHistoryProps) {
  const { series, allYears } = useMemo(() => {
    const byType: Record<string, ElectionHistoryPoint[]> = {};
    const yearsSet = new Set<number>();

    for (const pt of history) {
      if (!byType[pt.election_type]) byType[pt.election_type] = [];
      byType[pt.election_type].push(pt);
      yearsSet.add(pt.year);
    }

    // Sort each series by year
    for (const key of Object.keys(byType)) {
      byType[key].sort((a, b) => a.year - b.year);
    }

    const allYears = Array.from(yearsSet).sort((a, b) => a - b);
    return { series: byType, allYears };
  }, [history]);

  if (history.length === 0) {
    return (
      <p style={{ color: "var(--color-text-muted)", fontSize: 14 }}>
        No election history data available for this county.
      </p>
    );
  }

  const width = 680;
  const innerWidth = width - MARGIN.left - MARGIN.right;
  const innerHeight = CHART_HEIGHT - MARGIN.top - MARGIN.bottom;

  const allShares = history.map((p) => p.dem_share);
  const minShare = Math.min(...allShares);
  const maxShare = Math.max(...allShares);
  // Pad around the observed range, always including 0.5 midpoint
  const pad = 0.04;
  const yMin = Math.min(minShare - pad, 0.38);
  const yMax = Math.max(maxShare + pad, 0.62);

  // Use string year labels so scalePoint works cleanly
  const yearLabels = allYears.map(String);

  const xScale = scalePoint({
    domain: yearLabels,
    range: [0, innerWidth],
    padding: 0.2,
  });

  const yScale = scaleLinear({
    domain: [yMin, yMax],
    range: [innerHeight, 0],
    nice: true,
  });

  const midY = yScale(0.5);

  const seriesKeys = Object.keys(series).sort(); // stable order

  return (
    <div>
      <svg
        viewBox={`0 0 ${width} ${CHART_HEIGHT}`}
        style={{ width: "100%", height: "auto", overflow: "visible" }}
        aria-label="County election history chart"
      >
        <Group left={MARGIN.left} top={MARGIN.top}>
          {/* Midpoint (50%) line */}
          <line
            x1={0}
            x2={innerWidth}
            y1={midY}
            y2={midY}
            stroke="var(--color-border)"
            strokeWidth={1}
            strokeDasharray="4 3"
          />

          {/* Party fill above/below midpoint for presidential series */}
          {series.president &&
            series.president.map((pt, i) => {
              if (i === 0) return null;
              const prev = series.president[i - 1];
              const x0 = xScale(String(prev.year)) ?? 0;
              const x1 = xScale(String(pt.year)) ?? 0;
              const y0 = yScale(prev.dem_share);
              const y1 = yScale(pt.dem_share);
              const isDem = (pt.dem_share + prev.dem_share) / 2 >= 0.5;
              return (
                <polygon
                  key={`fill-${prev.year}-${pt.year}`}
                  points={`${x0},${midY} ${x0},${y0} ${x1},${y1} ${x1},${midY}`}
                  fill={isDem ? DUSTY_INK.leanD : DUSTY_INK.leanR}
                  opacity={0.10}
                />
              );
            })}

          {/* Lines per series */}
          {seriesKeys.map((key) => {
            const pts = series[key];
            const color = SERIES_COLORS[key] || "#999";
            return (
              <LinePath
                key={key}
                data={pts}
                x={(p) => xScale(String(p.year)) ?? 0}
                y={(p) => yScale(p.dem_share)}
                stroke={color}
                strokeWidth={key === "president" ? 2.5 : 1.75}
                strokeDasharray={key === "senate" ? "5 3" : key === "governor" ? "3 3" : undefined}
                curve={curveMonotoneX}
              />
            );
          })}

          {/* Data points per series */}
          {seriesKeys.map((key) =>
            series[key].map((pt) => {
              const cx = xScale(String(pt.year)) ?? 0;
              const cy = yScale(pt.dem_share);
              const color = SERIES_COLORS[key] || "#999";
              return (
                <g key={`${key}-${pt.year}`}>
                  <circle
                    cx={cx}
                    cy={cy}
                    r={key === "president" ? 4 : 3}
                    fill={color}
                    stroke="var(--color-bg)"
                    strokeWidth={1.5}
                  />
                  <title>{`${SERIES_LABELS[key] ?? key} ${pt.year}: ${formatShare(pt.dem_share)}`}</title>
                </g>
              );
            })
          )}

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
            tickFormat={yTickFormat}
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

      {/* Legend */}
      <div
        style={{
          display: "flex",
          gap: 16,
          flexWrap: "wrap",
          marginTop: 8,
          fontSize: 12,
          color: "var(--color-text-muted)",
        }}
      >
        {seriesKeys.map((key) => (
          <span key={key} style={{ display: "flex", alignItems: "center", gap: 5 }}>
            <svg width={24} height={10} style={{ flexShrink: 0 }}>
              <line
                x1={0}
                y1={5}
                x2={24}
                y2={5}
                stroke={SERIES_COLORS[key] || "#999"}
                strokeWidth={key === "president" ? 2.5 : 1.75}
                strokeDasharray={key === "senate" ? "5 3" : key === "governor" ? "3 3" : undefined}
              />
            </svg>
            {SERIES_LABELS[key] ?? key}
          </span>
        ))}
        <span style={{ marginLeft: "auto", fontStyle: "italic" }}>
          Dem two-party vote share
        </span>
      </div>
    </div>
  );
}
