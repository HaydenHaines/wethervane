import { useEffect, useRef } from "react";
import type { ExplorerRace } from "../lib/explorer-types";
import { STATE_NAMES, probability, raceColor, raceRating } from "../lib/presentation";
import { ribbonMinimumWidth } from "../lib/ribbon-layout";

/** A race-only ribbon. Ordering and width come from the result, with no seat threshold. */
export default function RaceProbabilityRibbon({ races, selectedId, onSelect }: {
  races: ExplorerRace[]; selectedId: string; onSelect: (id: string) => void;
}) {
  const scroller = useRef<HTMLDivElement>(null);
  const democraticChance = (race: ExplorerRace) => race.outcomes.find((o) => o.party === "D")!.winProbability;
  const ordered = [...races].sort((a, b) => democraticChance(b) - democraticChance(a) || a.id.localeCompare(b.id));
  useEffect(() => {
    const node = scroller.current;
    const tile = node?.querySelector<HTMLElement>("[aria-pressed=true]");
    if (node && tile && node.scrollWidth > node.clientWidth)
      node.scrollLeft = tile.offsetLeft - node.offsetLeft - (node.clientWidth - tile.offsetWidth) / 2;
  }, [selectedId, races]);
  return <section className="tipping-card" aria-labelledby="study-ribbon-title">
    <div className="tipping-heading"><div>
      <h2 id="study-ribbon-title" className="section-title">The races, in order</h2>
      <p className="small">All {races.length} study races, from strongest Democratic chance through tossups to strongest Republican chance.</p>
    </div></div>
    <div ref={scroller} className="tipping-scroll" role="region" tabIndex={0} aria-label="Study races ordered by Democratic win probability">
      <div className="tipping-track study-race-track" style={{ minWidth: ribbonMinimumWidth(races.length) }}>
        {ordered.map((race) => <button key={race.id} className={`race-tile${selectedId === race.id ? " selected" : ""}${raceRating(race) === "Likely" ? " likely" : ""}`}
          style={{ background: raceColor(race) }} aria-pressed={selectedId === race.id} onClick={() => onSelect(race.id)}
          aria-label={`${STATE_NAMES[race.state]}, ${raceRating(race)}, Democratic chance ${probability(democraticChance(race))}`}>
          <span className="tile-state">{race.state}</span><span className="tile-seat">{probability(democraticChance(race))} D</span>
        </button>)}
      </div>
    </div>
    <p className="small">There is no majority marker: this study covers a subset of races and supplies no chamber-control or tipping-point result.</p>
  </section>;
}
