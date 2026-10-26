"use client";

import { createContext, useContext } from "react";

export const HouseEditionContext = createContext<{
  currentAvailable: boolean;
  availabilityPending: boolean;
  onCurrent: () => void;
  onHistorical: (year: number) => void;
} | null>(null);

/** Match the Senate builder's year selector; supported historical choices stay catalog-owned. */
export default function HouseElectionYear({ year, years, busy, onYear, isYearAvailable = () => true }: {
  year: number;
  years: number[];
  busy: boolean;
  onYear: (year: number) => void;
  isYearAvailable?: (year: number) => boolean;
}) {
  const edition = useContext(HouseEditionContext);
  const choices = [...new Set([...(edition ? [2026] : []), ...years])].sort((a, b) => b - a);
  return <div className="stack-period"><div className="stack-field">
    <label>Election year</label>
    <div className="segmented year-control" role="group" aria-label="Election year">
      {choices.map(value => <button key={value} type="button" aria-pressed={year === value}
        disabled={busy || (value !== 2026 && !isYearAvailable(value)) || (value === 2026 && year !== 2026 && !!edition && !edition.currentAvailable)}
        onClick={() => value === 2026 && edition ? edition.onCurrent() : year === 2026 && edition ? edition.onHistorical(value) : onYear(value)}>{value}</button>)}
    </div>
    {edition && !edition.currentAvailable && year !== 2026 && <p className="small">{edition.availabilityPending
      ? "Checking whether a current House forecast is available."
      : "The current House forecast is unavailable. Published historical results remain separate."}</p>}
  </div></div>;
}
