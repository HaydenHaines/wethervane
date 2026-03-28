"use client";

import { useState } from "react";
import { formatMargin } from "@/lib/typeDisplay";

interface ForecastToggleProps {
  slug: string;
  nPolls: number;
  statePredNational: number | null;
  statePredLocal: number | null;
  candidateEffectMargin: number | null;
}

export default function ForecastToggle({
  slug,
  nPolls,
  statePredNational,
  statePredLocal,
  candidateEffectMargin,
}: ForecastToggleProps) {
  const hasPolls = nPolls > 0;
  const [mode, setMode] = useState<"national" | "local">(
    hasPolls ? "local" : "national"
  );

  const currentPred =
    mode === "national" ? statePredNational : statePredLocal;

  return (
    <div
      style={{
        margin: "16px 0",
        padding: "12px 16px",
        background: "var(--color-surface, #f8f9fa)",
        border: "1px solid var(--color-border, #e0e0e0)",
        borderRadius: "8px",
      }}
    >
      <div
        style={{
          display: "flex",
          gap: "4px",
          marginBottom: "8px",
        }}
        role="radiogroup"
        aria-label="Forecast mode"
      >
        <button
          role="radio"
          aria-checked={mode === "national"}
          onClick={() => setMode("national")}
          style={{
            padding: "6px 14px",
            border: "1px solid var(--color-border, #ccc)",
            borderRadius: "4px 0 0 4px",
            background:
              mode === "national"
                ? "var(--color-primary, #2563eb)"
                : "var(--color-bg, white)",
            color: mode === "national" ? "white" : "var(--color-text, #333)",
            cursor: "pointer",
            fontSize: "13px",
            fontWeight: 600,
          }}
        >
          National Environment
        </button>
        <button
          role="radio"
          aria-checked={mode === "local"}
          onClick={() => hasPolls && setMode("local")}
          disabled={!hasPolls}
          title={
            hasPolls ? undefined : "No polls available for this race"
          }
          style={{
            padding: "6px 14px",
            border: "1px solid var(--color-border, #ccc)",
            borderRadius: "0 4px 4px 0",
            background:
              mode === "local"
                ? "var(--color-primary, #2563eb)"
                : "var(--color-bg, white)",
            color: mode === "local" ? "white" : "var(--color-text, #333)",
            cursor: hasPolls ? "pointer" : "not-allowed",
            fontSize: "13px",
            fontWeight: 600,
            opacity: hasPolls ? 1 : 0.4,
          }}
        >
          Local Polling
        </button>
      </div>

      <p
        style={{
          fontSize: "12px",
          color: "var(--color-text-muted, #888)",
          margin: "0 0 4px",
        }}
      >
        {mode === "national"
          ? "Based on national political environment — no race-specific polling applied."
          : `Based on national environment + ${nPolls} poll${nPolls !== 1 ? "s" : ""} for this race.`}
      </p>

      {currentPred !== null && currentPred !== undefined && (
        <p
          style={{
            fontSize: "15px",
            fontWeight: 700,
            margin: "4px 0 0",
            color:
              currentPred > 0.5
                ? "var(--color-dem, #2563eb)"
                : "var(--color-rep, #dc2626)",
          }}
        >
          {formatMargin(currentPred)}
        </p>
      )}

      {candidateEffectMargin !== null &&
        candidateEffectMargin !== undefined &&
        hasPolls && (
          <p
            style={{
              fontSize: "11px",
              color: "var(--color-text-muted, #888)",
              margin: "4px 0 0",
            }}
          >
            Local polling shifts this race{" "}
            {candidateEffectMargin > 0 ? "D" : "R"}+
            {Math.abs(candidateEffectMargin * 100).toFixed(1)}pp from
            national environment.
          </p>
        )}
    </div>
  );
}
