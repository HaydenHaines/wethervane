"use client";
import { useEffect, useRef, useState } from "react";
import * as Plot from "@observablehq/plot";
import { fetchCommunityDetail, type CommunityDetail } from "@/lib/api";

interface Props {
  communityId: number;
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
      y: { label: "Δ D%", grid: true, zero: true },
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

export function CommunityPanel({ communityId, onClose }: Props) {
  const [detail, setDetail] = useState<CommunityDetail | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    fetchCommunityDetail(communityId)
      .then(setDetail)
      .finally(() => setLoading(false));
  }, [communityId]);

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
          <h3 style={{ margin: 0, fontFamily: "var(--font-serif)", fontSize: "16px" }}>
            {detail?.display_name ?? `Community ${communityId}`}
          </h3>
          {detail && (
            <p style={{ margin: "4px 0 0", fontSize: "12px", color: "var(--color-text-muted)" }}>
              {detail.n_counties} counties · {detail.states.join(", ")}
            </p>
          )}
        </div>
        <button
          onClick={onClose}
          style={{ border: "none", background: "none", cursor: "pointer", fontSize: "20px", color: "var(--color-text-muted)", lineHeight: 1 }}
        >×</button>
      </div>

      {loading && (
        <div style={{ padding: "20px", color: "var(--color-text-muted)", fontSize: "13px" }}>
          Loading…
        </div>
      )}

      {detail && !loading && (
        <div style={{ flex: 1, overflow: "auto", padding: "12px 16px" }}>
          {/* Shift sparkline */}
          <div style={{ marginBottom: "16px" }}>
            <p style={{ margin: "0 0 6px", fontSize: "11px", textTransform: "uppercase", letterSpacing: "0.5px", color: "var(--color-text-muted)" }}>
              Presidential swing pattern
            </p>
            <ShiftSparkline shiftProfile={detail.shift_profile} />
          </div>

          {/* County list */}
          <div>
            <p style={{ margin: "0 0 8px", fontSize: "11px", textTransform: "uppercase", letterSpacing: "0.5px", color: "var(--color-text-muted)" }}>
              Counties ({detail.n_counties})
            </p>
            {detail.counties.map((county) => {
              const share = county.pred_dem_share;
              const isD = share !== null && share > 0.5;
              return (
                <div key={county.county_fips} style={{
                  display: "flex",
                  justifyContent: "space-between",
                  padding: "5px 0",
                  borderBottom: "1px solid var(--color-border)",
                  fontSize: "13px",
                }}>
                  <span>{county.county_name ?? county.county_fips}</span>
                  {share !== null && (
                    <span style={{ color: isD ? "var(--color-dem)" : "var(--color-rep)", fontWeight: "600" }}>
                      {isD ? "D" : "R"} {Math.abs(share * 100 - 50).toFixed(1)}
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
