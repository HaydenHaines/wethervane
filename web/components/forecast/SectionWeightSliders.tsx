"use client";

import { useState, useCallback } from "react";

export interface SectionWeights {
  model_prior: number;
  state_polls: number;
  national_polls: number;
}

interface SectionWeightSlidersProps {
  /** Initial weights (0-100 scale, must sum to 100). */
  initial?: SectionWeights;
  /** Called when weights change. Debounced: fires after slider release. */
  onChange?: (weights: SectionWeights) => void;
}

const DEFAULT_WEIGHTS: SectionWeights = {
  model_prior: 60,
  state_polls: 30,
  national_polls: 10,
};

const LABEL: Record<keyof SectionWeights, string> = {
  model_prior: "Model prior",
  state_polls: "State polls",
  national_polls: "National polls",
};

/**
 * Three-section weight sliders for the forecast blend.
 *
 * Weights are constrained to sum to 100: when one slider moves, the remaining
 * weight is distributed proportionally among the other two.
 */
export function SectionWeightSliders({
  initial = DEFAULT_WEIGHTS,
  onChange,
}: SectionWeightSlidersProps) {
  const [weights, setWeights] = useState<SectionWeights>(initial);

  const handleChange = useCallback(
    (key: keyof SectionWeights, rawValue: number) => {
      const clampedValue = Math.max(0, Math.min(100, rawValue));
      const remaining = 100 - clampedValue;
      const otherKeys = (Object.keys(weights) as (keyof SectionWeights)[]).filter(
        (k) => k !== key,
      );
      const otherTotal = otherKeys.reduce((s, k) => s + weights[k], 0);

      const mutable: Record<keyof SectionWeights, number> = { ...weights };
      if (otherTotal === 0) {
        // Distribute evenly
        const share = remaining / otherKeys.length;
        mutable[key] = clampedValue;
        otherKeys.forEach((k) => { mutable[k] = share; });
      } else {
        mutable[key] = clampedValue;
        otherKeys.forEach((k) => {
          mutable[k] = Math.round((weights[k] / otherTotal) * remaining);
        });
        // Fix any rounding drift so total = 100
        const total = Object.values(mutable).reduce((s, v) => s + v, 0);
        if (total !== 100) {
          mutable[otherKeys[otherKeys.length - 1]] += 100 - total;
        }
      }
      const updated: SectionWeights = mutable;

      setWeights(updated);
      onChange?.(updated);
    },
    [weights, onChange],
  );

  return (
    <div
      className="rounded-md p-4 space-y-4"
      style={{
        background: "var(--color-surface)",
        border: "1px solid var(--color-border)",
      }}
    >
      <p className="text-xs font-semibold uppercase tracking-wide" style={{ color: "var(--color-text-muted)" }}>
        Forecast blend
      </p>
      {(Object.keys(weights) as (keyof SectionWeights)[]).map((key) => (
        <div key={key} className="space-y-1">
          <div className="flex justify-between text-sm">
            <label htmlFor={`slider-${key}`} className="font-medium">
              {LABEL[key]}
            </label>
            <span className="font-mono text-muted-foreground">
              {weights[key]}%
            </span>
          </div>
          <input
            id={`slider-${key}`}
            type="range"
            min={0}
            max={100}
            step={5}
            value={weights[key]}
            onChange={(e) => handleChange(key, parseInt(e.target.value, 10))}
            className="w-full accent-[var(--forecast-safe-d)]"
            aria-label={`${LABEL[key]} weight: ${weights[key]}%`}
          />
        </div>
      ))}
      <p className="text-xs" style={{ color: "var(--color-text-muted)" }}>
        Weights must sum to 100. Adjusting one redistributes the others proportionally.
      </p>
    </div>
  );
}
