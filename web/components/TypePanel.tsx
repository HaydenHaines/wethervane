"use client";
import { useEffect, useRef, useState } from "react";
import * as Plot from "@observablehq/plot";
import { fetchTypeDetail, type TypeDetail } from "@/lib/api";
import { SUPER_TYPE_COLORS, SUPER_TYPE_NAMES } from "@/components/MapShell";

interface Props {
  typeId: number;
  onClose: () => void;
}

function ShiftSparkline({ shiftProfile }: { shiftProfile: Record<string, number> }) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!ref.current) return;

    const entries = Object.entries(shiftProfile)
      .filter(([k]) => k.startsWith("pres_d_shift_"))
      .map(([key, value]) => {
        const parts = key.split("_");
        const year = parseInt("20" + parts[parts.length - 1], 10);
        return { year, shift: value, label: key.replace("pres_d_shift_", "") };
      })
      .sort((a, b) => a.year - b.year);

    if (entries.length === 0) return;

    const plot = Plot.plot({
      width: 260,
      height: 80,
      marginLeft: 28,
      marginRight: 4,
      marginTop: 8,
      marginBottom: 20,
      x: { label: null, tickFormat: (d: number) => `'${String(d).slice(2)}` },
      y: { label: "shift", grid: true, zero: true },
      marks: [
        Plot.ruleY([0], { stroke: "#ccc" }),
        Plot.line(entries, {
          x: "year",
          y: "shift",
          stroke: entries.some((e) => e.shift > 0) ? "var(--color-dem)" : "var(--color-rep)",
          strokeWidth: 1.5,
        }),
        Plot.dot(entries, { x: "year", y: "shift", r: 2.5, fill: "var(--color-text)" }),
      ],
    });

    ref.current.innerHTML = "";
    ref.current.appendChild(plot);

    return () => {
      ref.current?.removeChild(plot);
    };
  }, [shiftProfile]);

  return <div ref={ref} />;
}

function DemographicRow({ label, value, fmt }: { label: string; value: number | undefined; fmt?: "pct" | "dollar" | "num" }) {
  if (value == null) return null;
  let display: string;
  if (fmt === "dollar") display = `$${Math.round(value).toLocaleString()}`;
  else if (fmt === "pct") display = `${(value * 100).toFixed(1)}%`;
  else display = value.toFixed(1);
  return (
    <div style={{ display: "flex", justifyContent: "space-between", padding: "3px 0", fontSize: "12px" }}>
      <span style={{ color: "var(--color-text-muted)" }}>{label}</span>
      <span style={{ fontWeight: 500 }}>{display}</span>
    </div>
  );
}

export function TypePanel({ typeId, onClose }: Props) {
  const [detail, setDetail] = useState<TypeDetail | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    fetchTypeDetail(typeId)
      .then(setDetail)
      .catch(() => setDetail(null))
      .finally(() => setLoading(false));
  }, [typeId]);

  const superColor = detail
    ? SUPER_TYPE_COLORS[detail.super_type_id % SUPER_TYPE_COLORS.length]
    : [127, 127, 127];

  return (
    <div style={{
      position: "absolute",
      top: 0,
      right: 0,
      width: "320px",
      height: "100%",
      background: "white",
      borderLeft: "1px solid var(--color-border)",
      display: "flex",
      flexDirection: "column",
      boxShadow: "-2px 0 8px rgba(0,0,0,0.08)",
      zIndex: 10,
    }}>
      {/* Header */}
      <div style={{
        padding: "16px 16px 12px",
        borderBottom: "1px solid var(--color-border)",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "flex-start",
      }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <div style={{
              width: 14, height: 14, borderRadius: 3,
              background: `rgb(${superColor.join(",")})`,
            }} />
            <h3 style={{ margin: 0, fontFamily: "var(--font-serif)", fontSize: "16px" }}>
              {detail?.display_name ?? `Type ${typeId}`}
            </h3>
          </div>
          {detail && (
            <p style={{ margin: "4px 0 0", fontSize: "12px", color: "var(--color-text-muted)" }}>
              {detail.n_counties} counties
              {SUPER_TYPE_NAMES[detail.super_type_id] && (
                <> &middot; {SUPER_TYPE_NAMES[detail.super_type_id]}</>
              )}
            </p>
          )}
        </div>
        <button
          onClick={onClose}
          style={{ border: "none", background: "none", cursor: "pointer", fontSize: "20px", color: "var(--color-text-muted)", lineHeight: 1 }}
        >&times;</button>
      </div>

      {loading && (
        <div style={{ padding: "20px", color: "var(--color-text-muted)", fontSize: "13px" }}>
          Loading...
        </div>
      )}

      {detail && !loading && (
        <div style={{ flex: 1, overflow: "auto", padding: "12px 16px" }}>
          {/* Shift sparkline */}
          {detail.shift_profile && Object.keys(detail.shift_profile).length > 0 && (
            <div style={{ marginBottom: "16px" }}>
              <p style={{ margin: "0 0 6px", fontSize: "11px", textTransform: "uppercase", letterSpacing: "0.5px", color: "var(--color-text-muted)" }}>
                Shift pattern
              </p>
              <ShiftSparkline shiftProfile={detail.shift_profile} />
            </div>
          )}

          {/* Demographics */}
          {Object.keys(detail.demographics).length > 0 && (
            <div style={{ marginBottom: "16px" }}>
              <p style={{ margin: "0 0 6px", fontSize: "11px", textTransform: "uppercase", letterSpacing: "0.5px", color: "var(--color-text-muted)" }}>
                Demographics
              </p>
              <DemographicRow label="Median income" value={detail.demographics.median_hh_income} fmt="dollar" />
              <DemographicRow label="Median age" value={detail.demographics.median_age} fmt="num" />
              <DemographicRow label="White (non-Hispanic)" value={detail.demographics.pct_white_nh} fmt="pct" />
              <DemographicRow label="Black" value={detail.demographics.pct_black} fmt="pct" />
              <DemographicRow label="Hispanic" value={detail.demographics.pct_hispanic} fmt="pct" />
              <DemographicRow label="Asian" value={detail.demographics.pct_asian} fmt="pct" />
              <DemographicRow label="Bachelor's+" value={detail.demographics.pct_bachelors_plus} fmt="pct" />
            </div>
          )}

          {/* County list */}
          <div>
            <p style={{ margin: "0 0 8px", fontSize: "11px", textTransform: "uppercase", letterSpacing: "0.5px", color: "var(--color-text-muted)" }}>
              Counties ({detail.n_counties})
            </p>
            {detail.counties.map((fips) => (
              <div key={fips} style={{
                padding: "5px 0",
                borderBottom: "1px solid var(--color-border)",
                fontSize: "13px",
              }}>
                {fips}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
