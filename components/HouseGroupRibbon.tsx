"use client";

import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";
import type { HouseGroupBloc } from "../lib/house-group-result.ts";
import { groupRibbonMajorityPositions, layoutGroupRibbon, type GroupRibbonEntry, type GroupRibbonRace, type RibbonChamber } from "../lib/group-ribbon-layout.ts";
import { COLORS, probability } from "../lib/presentation.ts";
import HouseGroupLegend from "./HouseGroupLegend";

const groupName: Record<HouseGroupBloc, string> = { D: "Democratic group", R: "Republican group", O: "Other group" };
const chanceText = (row: GroupRibbonEntry) => `Democratic ${probability(row.probabilities.D)}, Republican ${probability(row.probabilities.R)}, Other ${probability(row.probabilities.O)}`;
function tileColor(row: GroupRibbonEntry) {
  const top = row.leaders[0], chance = row.probabilities[top];
  if (row.leaders.length > 1 || chance <= .6) return COLORS.competitive;
  if (chance > .9) return COLORS[top];
  return { D: COLORS.likelyD, R: COLORS.likelyR, O: COLORS.likelyO }[top];
}

/** Group-win chances, with optional seat-count reference markers for a complete chamber. */
export default function HouseGroupRibbon({ races, selectedRaceId, onSelect, chamber }: { races: readonly GroupRibbonRace[]; selectedRaceId: string; onSelect: (id: string) => void; chamber?: RibbonChamber }) {
  const container = useRef<HTMLDivElement>(null);
  const buttons = useRef(new Map<string, HTMLButtonElement>());
  const [columns, setColumns] = useState(12);
  const [showLabels, setShowLabels] = useState(races.length <= 60);
  const [activeId, setActiveId] = useState(selectedRaceId || races[0]?.id || "");
  useEffect(() => {
    if (!container.current || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(([entry]) => setColumns(Math.max(1, Math.min(showLabels ? 20 : 60, Math.floor(entry.contentRect.width / (showLabels ? 60 : 20))))));
    observer.observe(container.current);
    return () => observer.disconnect();
  }, [showLabels]);
  const layout = useMemo(() => layoutGroupRibbon(races, columns), [races, columns]);
  const majorityPositions = groupRibbonMajorityPositions(layout, chamber);
  const majorityLabel = (from: ("D" | "R")[]) => from.length === 2 ? "from either end"
    : from[0] === "D" ? "from the Democratic end" : "from the Republican end";
  const ready = [...layout.main.rows, ...layout.other.rows, ...layout.mixedTies];
  const entries = [...ready, ...layout.unavailable];
  useEffect(() => {
    if (!entries.some(row => row.id === activeId)) setActiveId(selectedRaceId || entries[0]?.id || "");
  }, [entries, activeId, selectedRaceId]);
  const selected = ready.find(row => row.id === selectedRaceId);
  const missing = layout.unavailable.find(row => row.id === selectedRaceId);

  function navigation(id: string) {
    return {
      ref: (node: HTMLButtonElement | null) => { if (node) buttons.current.set(id, node); else buttons.current.delete(id); },
      tabIndex: id === activeId ? 0 : -1,
      "aria-pressed": id === selectedRaceId,
      onFocus: () => setActiveId(id),
      onClick: () => onSelect(id),
      onKeyDown: (event: KeyboardEvent<HTMLButtonElement>) => {
        const index = entries.findIndex(row => row.id === id);
        const next = event.key === "Home" ? 0 : event.key === "End" ? entries.length - 1
          : ["ArrowRight", "ArrowDown"].includes(event.key) ? Math.min(entries.length - 1, index + 1)
            : ["ArrowLeft", "ArrowUp"].includes(event.key) ? Math.max(0, index - 1) : null;
        if (next === null || next < 0) return;
        event.preventDefault(); buttons.current.get(entries[next].id)?.focus();
      },
    };
  }
  function tile(row: GroupRibbonEntry, position?: { row: number; column: number; direction: string }) {
    const majority = majorityPositions.find(marker => marker.id === row.id);
    const top = Math.max(row.probabilities.D, row.probabilities.R, row.probabilities.O);
    const description = `${row.districtKey}; ${chanceText(row)}; ${row.leaders.length > 1 ? "Top chances tied: " : "Highest chance: "}${row.leaders.map(key => groupName[key]).join(" and ")}; ${row.probabilitySource === "structural" ? "fixed under the listed group support" : "modeled group probabilities"}${majority ? `; majority marker, ${chamber!.majorityThreshold} seats ${majorityLabel(majority.from)}` : ""}`;
    const color = tileColor(row);
    return <button key={row.id} {...navigation(row.id)} className={`house-ribbon-cell${majority ? " majority-position" : ""}`} data-district={row.districtKey}
      data-direction={position?.direction} data-majority-from={majority?.from.join(" ")}
      aria-label={description} title={description}
      style={{ ...(position ? { gridRow: position.row + 1, gridColumn: position.column + 1 } : {}), background: color, color: top > .6 && top <= .9 && row.leaders.length === 1 ? "var(--ink)" : "#fff" }}>
      <span>{row.districtKey}</span><small aria-hidden="true">{position?.direction === "right-to-left" ? "←" : "→"}</small>
    </button>;
  }
  // Coordinates from the shared serpentine helper are expressed against this configured width.
  const gridStyle = () => ({ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` });
  return <section className={`tipping-card house-ribbon${showLabels ? "" : " is-compact"}`} aria-label="House group-win probability ribbon">
    <h2>House group-win chances by district</h2>
    <p>{races.length} districts · {ready.length} with a usable group prediction · {layout.unavailable.length} unavailable.</p>
    <p>Each probability is the chance that a Democratic, Republican, or Other candidate group wins the district.</p>
    {majorityPositions.length > 0 ? <div className="house-majority-summary">
      <strong><span className="house-majority-key" aria-hidden="true" />Majority marker · {chamber!.majorityThreshold} seats</strong>
      <div>{majorityPositions.map(marker => <span key={marker.id}>
        <button type="button" onClick={() => onSelect(marker.id)}>{marker.districtKey}</button>{" "}{majorityLabel(marker.from)}
      </span>)}</div>
      <p>The bar marks the district reached by counting {chamber!.majorityThreshold} seats inward from each end of the ranked path.
        {layout.other.rows.length + layout.mixedTies.length > 0 && " Districts where Other leads or ties stay separate, so the two markers differ."}
        {" "}This is a seat-count guide, not the chance that a district decides control. See the chamber forecast below the map.</p>
    </div> : <p>This ribbon does not show House control or a tipping point.</p>}
    <label className="house-ribbon-density"><input type="checkbox" checked={showLabels} onChange={event => setShowLabels(event.target.checked)} /> Show district labels and larger tiles</label>
    <p className="house-ribbon-focus" aria-live="polite">
      {selected ? <><strong>{selected.districtKey}</strong> · {chanceText(selected)}. {selected.leaders.length > 1 ? "Top chances tied: " : "Highest chance: "}{selected.leaders.map(key => groupName[key]).join(" and ")}. {selected.probabilitySource === "structural" && "Fixed under the listed group support."}</>
        : missing ? <><strong>{missing.districtKey}</strong> · Group prediction unavailable. {missing.reason}</> : "No districts supplied."}
    </p>
    <div ref={container} className="house-ribbon-container">
      {layout.main.rows.length > 0 && <section aria-label="Democratic and Republican path" style={{ marginTop: 24 }}>
        <h3>Democratic and Republican path</h3>
        <p>Ordered from higher Democratic chance to higher Republican chance by the difference between those two chances. Other-group probability remains shown in each district’s detail.</p>
        <div className="house-ribbon-ends"><span>Higher Democratic chance</span><span>Higher Republican chance</span></div>
        <div className="house-ribbon-grid" data-segment="main" style={gridStyle()}>{layout.main.rows.map(row => tile(row, row))}</div>
      </section>}
      {layout.other.rows.length > 0 && <section aria-label="Other group leads" style={{ marginTop: 32 }}>
        <h3>Other group leads</h3><p>Other has the highest chance in these districts. Ordered by Other-group chance, highest first.</p>
        <div className="house-ribbon-grid" data-segment="other" style={gridStyle()}>{layout.other.rows.map(row => tile(row, row))}</div>
      </section>}
      {layout.mixedTies.length > 0 && <section aria-label="Top ties involving Other" style={{ marginTop: 32 }}>
        <h3>Top ties involving Other</h3><p>Other ties with at least one other group for the highest chance. These districts have no Democratic-versus-Republican rank.</p>
        <div className="house-ribbon-grid" data-segment="ties" style={gridStyle()}>{layout.mixedTies.map(row => tile(row))}</div>
      </section>}
      {layout.unavailable.length > 0 && <section aria-label="Group prediction unavailable" style={{ marginTop: 32 }}>
        <h3>Group prediction unavailable</h3><p>These districts stay outside the probability rankings; select one to see why.</p>
        <div className="house-ribbon-grid" data-segment="unavailable" style={gridStyle()}>{layout.unavailable.map(row => {
          const description = `${row.districtKey}, group prediction unavailable: ${row.reason}`;
          return <button key={row.id} {...navigation(row.id)} className="house-ribbon-cell" data-district={row.districtKey} aria-label={description} title={description}
            style={{ background: COLORS.inactive, color: "var(--ink)" }}><span>{row.districtKey}</span></button>;
        })}</div>
      </section>}
    </div>
    <p className="small">Arrow keys follow the displayed sequence; Enter or Space selects a district. Colors indicate the leading group; purple indicates no group is above 60%. Direction arrows show each row’s reading direction.</p>
    <HouseGroupLegend unavailableLabel={layout.unavailable.length ? "Prediction unavailable" : undefined} />
  </section>;
}
