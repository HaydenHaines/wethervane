"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { HouseResult } from "../lib/house-result";
import { layoutHouseRibbon } from "../lib/house-ribbon";
import { probability, raceColor, raceRating } from "../lib/presentation";

export default function HouseRibbon({
  result,
  selectedId,
  onSelect,
  showMajorityPosition,
}: {
  result: HouseResult;
  selectedId: string;
  onSelect: (id: string) => void;
  showMajorityPosition: boolean;
}) {
  const container = useRef<HTMLDivElement>(null);
  const buttons = useRef(new Map<string, HTMLButtonElement>());
  const [columns, setColumns] = useState(12);
  const [showLabels, setShowLabels] = useState(result.races.length <= 60);
  const [activeId, setActiveId] = useState(selectedId);
  useEffect(() => {
    setActiveId(selectedId);
  }, [selectedId]);
  useEffect(() => {
    const node = container.current;
    if (!node) return;
    const observer = new ResizeObserver(([entry]) =>
      setColumns(
        Math.max(
          1,
          Math.min(
            showLabels ? 20 : 60,
            Math.floor(entry.contentRect.width / (showLabels ? 56 : 18)),
          ),
        ),
      ),
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, [showLabels]);
  const layout = useMemo(
    () =>
      layoutHouseRibbon(
        result.races.map((race) => ({
          id: race.id,
          districtKey: race.district.districtKey,
          democraticWinProbability: race.outcomes.find(
            (outcome) => outcome.party === "D",
          )!.winProbability,
        })),
        columns,
      ),
    [result, columns],
  );
  const byId = new Map(result.races.map((race) => [race.id, race]));
  const active = layout.rows.find((entry) => entry.id === activeId);
  return (
    <section
      className={`tipping-card house-ribbon${showLabels ? "" : " is-compact"}`}
      aria-label="House race ribbon"
    >
      <div className="tipping-heading">
        <h2 className="section-title">The path through the House</h2>
        <span className="small">
          {result.races.length} districts ·{" "}
          {result.metadata.full_chamber
            ? "Full chamber"
            : "Eligible races only"}
        </span>
      </div>
      <p className="small">
        Follow the ribbon from the strongest Democratic chances, through the
        closest races, to the strongest Republican chances. Each tile is one
        district.
      </p>
      <label className="house-ribbon-density">
        <input
          type="checkbox"
          checked={showLabels}
          onChange={(event) => setShowLabels(event.target.checked)}
        />{" "}
        Show district labels and larger tiles
      </label>
      <div className="house-ribbon-ends">
        <span>Start: more Democratic</span>
        <span>End: more Republican</span>
      </div>
      {active && (
        <p className="house-ribbon-focus" aria-live="polite">
          <strong>{active.districtKey}</strong> · Democratic win chance{" "}
          {probability(active.democraticWinProbability)} · Rank{" "}
          {active.index + 1} of {layout.rows.length}
        </p>
      )}
      <div ref={container} className="house-ribbon-container">
        <div
          className="house-ribbon-grid"
          style={{
            gridTemplateColumns: `repeat(${layout.columnCount}, minmax(0, 1fr))`,
          }}
        >
          {layout.rows.map((entry) => {
            const race = byId.get(entry.id)!;
            const majority =
              showMajorityPosition &&
              entry.index + 1 === result.metadata.majority_threshold;
            const label = `${entry.districtKey}, ${raceRating(race)}, Democratic win chance ${probability(entry.democraticWinProbability)}, rank ${entry.index + 1} of ${layout.rows.length}${majority ? ", majority position" : ""}`;
            return (
              <button
                key={entry.id}
                ref={(node) => {
                  if (node) buttons.current.set(entry.id, node);
                  else buttons.current.delete(entry.id);
                }}
                className={`house-ribbon-cell${majority ? " majority-position" : ""}`}
                data-race-id={entry.id}
                data-rank={entry.index + 1}
                data-direction={entry.direction}
                aria-label={label}
                title={label}
                aria-pressed={entry.id === selectedId}
                tabIndex={entry.id === activeId ? 0 : -1}
                style={{
                  gridRow: entry.row + 1,
                  gridColumn: entry.column + 1,
                  background: raceColor(race),
                  color: raceRating(race) === "Likely" ? "var(--ink)" : "#fff",
                }}
                onFocus={() => setActiveId(entry.id)}
                onClick={() => onSelect(entry.id)}
                onKeyDown={(event) => {
                  const nextIndex =
                    event.key === "Home"
                      ? 0
                      : event.key === "End"
                        ? layout.rows.length - 1
                        : ["ArrowRight", "ArrowDown"].includes(event.key)
                          ? Math.min(layout.rows.length - 1, entry.index + 1)
                          : ["ArrowLeft", "ArrowUp"].includes(event.key)
                            ? Math.max(0, entry.index - 1)
                            : null;
                  if (nextIndex === null) return;
                  event.preventDefault();
                  buttons.current.get(layout.rows[nextIndex].id)?.focus();
                }}
              >
                <span>{entry.districtKey}</span>
                {entry.index % columns === 0 && (
                  <small aria-hidden="true">
                    {entry.direction === "left-to-right" ? "→" : "←"}
                  </small>
                )}
              </button>
            );
          })}
        </div>
      </div>
      <p className="small">
        Arrow keys follow the ranked sequence; Enter selects a district. Safe
        &gt;90% · Likely &gt;60–90% · Tossup 40–60%.
      </p>
      {showMajorityPosition ? (
        <p className="house-coverage-note">
          The outlined 218th district marks the majority position in this
          probability ranking. A district’s frequency at the tipping point
          requires joint simulations.
        </p>
      ) : (
        <p className="house-coverage-note">
          Chamber tipping point unavailable. This set of races does not
          establish a complete House forecast.
        </p>
      )}
    </section>
  );
}
