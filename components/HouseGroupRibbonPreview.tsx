"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { layoutHouseGroupRibbon, type HouseGroupRibbonEntry } from "../lib/house-group-ribbon.ts";
import type { HouseWinnerFixture } from "../lib/house-winner-view.ts";
import { COLORS, probability } from "../lib/presentation.ts";

const labels = { D: "Democratic group", R: "Republican group", O: "Other group" };
const chanceText = (row: HouseGroupRibbonEntry) => `Democratic ${probability(row.probabilities.D)}, Republican ${probability(row.probabilities.R)}, Other ${probability(row.probabilities.O)}`;
function color(row: HouseGroupRibbonEntry) {
  const top = row.leaders[0], chance = row.probabilities[top];
  if (row.leaders.length > 1 || chance <= .6) return COLORS.competitive;
  return chance > .9 ? COLORS[top] : { D: COLORS.likelyD, R: COLORS.likelyR, O: COLORS.likelyO }[top];
}

/** Private fixture-only preparation. This component does not accept a public result. */
export default function HouseGroupRibbonPreview({ fixtures }: { fixtures: HouseWinnerFixture[] }) {
  const container = useRef<HTMLDivElement>(null);
  const buttons = useRef(new Map<string, HTMLButtonElement>());
  const [columns, setColumns] = useState(12);
  const [showLabels, setShowLabels] = useState(fixtures.length <= 60);
  const [selectedId, setSelectedId] = useState(fixtures[0]?.district || "");
  const [activeId, setActiveId] = useState(fixtures[0]?.district || "");
  useEffect(() => {
    if (!container.current) return;
    const observer = new ResizeObserver(([entry]) => setColumns(Math.max(1, Math.min(showLabels ? 20 : 60, Math.floor(entry.contentRect.width / (showLabels ? 60 : 20))))));
    observer.observe(container.current);
    return () => observer.disconnect();
  }, [showLabels]);
  const layout = useMemo(() => layoutHouseGroupRibbon(fixtures, columns), [fixtures, columns]);
  const ready = [...layout.main.rows, ...layout.other.rows, ...layout.mixedTies];
  const entries = [...ready, ...layout.unavailable];
  useEffect(() => {
    if (!entries.some(row => row.id === selectedId)) setSelectedId(entries[0]?.id || "");
    if (!entries.some(row => row.id === activeId)) setActiveId(entries[0]?.id || "");
  }, [fixtures, selectedId, activeId]); // Selection survives a different ordering or column count.
  const selected = ready.find(row => row.id === selectedId);
  const missing = layout.unavailable.find(row => row.id === selectedId);

  function navigation(id: string) {
    return {
      ref: (node: HTMLButtonElement | null) => { if (node) buttons.current.set(id, node); else buttons.current.delete(id); },
      tabIndex: id === activeId ? 0 : -1,
      "aria-pressed": id === selectedId,
      onFocus: () => setActiveId(id),
      onClick: () => setSelectedId(id),
      onKeyDown: (event: React.KeyboardEvent<HTMLButtonElement>) => {
        const index = entries.findIndex(row => row.id === id);
        const next = event.key === "Home" ? 0 : event.key === "End" ? entries.length - 1
          : ["ArrowRight", "ArrowDown"].includes(event.key) ? Math.min(entries.length - 1, index + 1)
            : ["ArrowLeft", "ArrowUp"].includes(event.key) ? Math.max(0, index - 1) : null;
        if (next === null) return;
        event.preventDefault(); buttons.current.get(entries[next].id)?.focus();
      },
    };
  }
  function tile(row: HouseGroupRibbonEntry, position?: { row: number; column: number }) {
    const top = Math.max(...Object.values(row.probabilities));
    const description = `${row.districtKey}; ${chanceText(row)}; ${row.leaders.length > 1 ? "top chances tied: " : "highest chance: "}${row.leaders.map(key => labels[key]).join(" and ")}; ${row.probabilitySource === "structural" ? "fixed under the listed candidates" : "modeled probabilities"}`;
    return <button key={row.id} {...navigation(row.id)} className="house-ribbon-cell" data-district={row.districtKey}
      aria-label={description} title={description}
      style={{ ...(position ? { gridRow: position.row + 1, gridColumn: position.column + 1 } : {}), background: color(row), color: top > .6 && top <= .9 && row.leaders.length === 1 ? "var(--ink)" : "#fff" }}>
      <span>{row.districtKey}</span>
    </button>;
  }
  return <section className={`count-preview tipping-card house-ribbon${showLabels ? "" : " is-compact"}`} aria-label="Invented House group ribbon preview">
    <div className="count-fixture-banner" role="note"><strong>Illustrative data only · Private preview</strong><p>These districts and probabilities are invented to test the interface.</p></div>
    <h2>House group forecast ribbon</h2>
    <p>{fixtures.length} districts represented · {ready.length} with predictions · {layout.unavailable.length} unavailable.</p>
    <label className="house-ribbon-density"><input type="checkbox" checked={showLabels} onChange={event => setShowLabels(event.target.checked)} /> Show district labels and larger tiles</label>
    <p className="house-ribbon-focus" aria-live="polite" data-selection="true">
      {selected ? <><strong>{selected.districtKey}</strong> · {chanceText(selected)}. {selected.leaders.length > 1 ? "Top chances tied: " : "Highest chance: "}{selected.leaders.map(key => labels[key]).join(" and ")}. {selected.probabilitySource === "structural" && "Fixed under the listed candidates."}</>
        : missing ? <><strong>{missing.districtKey}</strong> · Forecast unavailable. {missing.reason}</> : "No districts supplied."}
    </p>
    <div ref={container}>
      {layout.main.rows.length > 0 && <section aria-label="Democratic and Republican paths">
        <h3>Democratic and Republican paths</h3>
        <p>Ordered by the Democratic chance minus the Republican chance. Other-group chances remain part of each district’s forecast.</p>
        <div className="house-ribbon-ends"><span>Higher Democratic chance</span><span>Higher Republican chance</span></div>
        <div className="house-ribbon-grid" data-segment="main" style={{ gridTemplateColumns: `repeat(${layout.main.columnCount}, minmax(0, 1fr))` }}>{layout.main.rows.map(row => tile(row, row))}</div>
      </section>}
      {layout.other.rows.length > 0 && <section aria-label="Other group leads">
        <h3>Other group leads</h3><p>These districts have a higher Other-group chance than either major party. Ordered by that chance, highest first.</p>
        <div className="house-ribbon-grid" data-segment="other" style={{ gridTemplateColumns: `repeat(${layout.other.columnCount}, minmax(0, 1fr))` }}>{layout.other.rows.map(row => tile(row, row))}</div>
      </section>}
      {layout.mixedTies.length > 0 && <section aria-label="Top ties involving Other">
        <h3>Top ties involving Other</h3><p>These districts have equal highest chances for Other and at least one major party. Listed by district, without a partisan rank.</p>
        <div className="house-ribbon-grid" data-segment="ties" style={{ gridTemplateColumns: `repeat(${Math.min(columns, layout.mixedTies.length)}, minmax(0, 1fr))` }}>{layout.mixedTies.map(row => tile(row))}</div>
      </section>}
      {layout.unavailable.length > 0 && <section aria-label="No group prediction available">
        <h3>No group prediction available</h3><p>These districts stay outside the probability rankings. Select one to see why its forecast is unavailable.</p>
        <div className="house-ribbon-grid" data-segment="unavailable" style={{ gridTemplateColumns: `repeat(${Math.min(columns, layout.unavailable.length)}, minmax(0, 1fr))` }}>{layout.unavailable.map(row => {
          const description = `${row.districtKey}, forecast unavailable: ${row.reason}`;
          return <button key={row.id} {...navigation(row.id)} className="house-ribbon-cell" data-district={row.districtKey}
            aria-label={description} title={description} style={{ background: COLORS.inactive, color: "var(--ink)" }}><span>{row.districtKey}</span></button>;
        })}</div>
      </section>}
    </div>
    <p className="small">Arrow keys follow the displayed sequence; Enter selects a district. Blue, red and neutral tiles identify the leading group; pale shades mean a chance above 60% through 90%, dark shades above 90%. Purple means no group is above 60%.</p>
    <p className="house-coverage-note">This ordering compares chances of winning. It is not a vote margin and does not estimate House control or a tipping point.</p>
  </section>;
}
