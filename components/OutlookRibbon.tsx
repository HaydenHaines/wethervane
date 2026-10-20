import { useEffect, useRef } from "react";
import { ribbonMinimumWidth, ribbonSeatLabel } from "../lib/ribbon-layout";
import type { ExplorerRace, ExplorerScenario } from "../lib/explorer-types";
import {
  STATE_NAMES,
  competitiveOrder,
  leadingOutcome,
  probability,
  raceColor,
  raceRating,
} from "../lib/presentation";

export default function OutlookRibbon({
  races,
  scenario,
  selectedId,
  onSelect,
}: {
  races: ExplorerRace[];
  scenario: ExplorerScenario;
  selectedId: string;
  onSelect: (id: string) => void;
}) {
  const scroller = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const node = scroller.current;
    if (!node || node.scrollWidth <= node.clientWidth) return;
    const button = Array.from(
      node.querySelectorAll<HTMLButtonElement>("button"),
    ).find((b) => b.dataset.raceId === selectedId);
    if (button)
      node.scrollLeft =
        button.offsetLeft -
        node.offsetLeft -
        (node.clientWidth - button.offsetWidth) / 2;
  }, [selectedId, races]);
  return (
    <section className="tipping-card" aria-labelledby="outlook-title">
      <div className="tipping-heading">
        <div>
          <h2 className="section-title" id="outlook-title">
            The election outlook
          </h2>
          <p className="small">
            All {races.length} contests, ordered from least to most confident
            leading outcome.
          </p>
        </div>
      </div>
      <p className="ribbon-fallback">
        Historical vote margins and tipping-point summaries are unavailable.
        Tiles show outcome win probabilities, without inferred control-threshold
        bars.
      </p>
      <div
        className="tipping-scroll"
        ref={scroller}
        role="region"
        tabIndex={0}
        aria-label="Historical race ribbon, ordered by leading outcome win probability"
      >
        <div
          className="tipping-track"
          style={{ minWidth: ribbonMinimumWidth(races.length) }}
        >
          <div className="retained-endcap">
            <strong>{scenario.retainedSeats.D}</strong>
            <span>
              D caucus
              <br />
              retained
            </span>
          </div>
          {[...races].sort(competitiveOrder).map((race) => {
            const lead = leadingOutcome(race);
            return (
              <button
                key={race.id}
                data-race-id={race.id}
                className={`race-tile${selectedId === race.id ? " selected" : ""}${raceRating(race) === "Likely" ? " likely" : ""}`}
                style={{ background: raceColor(race) }}
                aria-label={`${STATE_NAMES[race.state]}, ${race.electionLabel}, ${raceRating(race)}, ${lead.name}${lead.kind === "group" ? " group" : ""} ${probability(lead.winProbability)} chance of winning`}
                aria-pressed={selectedId === race.id}
                onClick={() => onSelect(race.id)}
              >
                <span className="tile-state">{race.state}</span>
                <span className="tile-seat">{ribbonSeatLabel(race)}</span>
              </button>
            );
          })}
          <div className="retained-endcap">
            <strong>{scenario.retainedSeats.R}</strong>
            <span>
              R caucus
              <br />
              retained
            </span>
          </div>
        </div>
      </div>
      <p className="small">
        {scenario.label}.{" "}
        {scenario.retainedSeats.U > 0
          ? `${scenario.retainedSeats.U} retained seats remain unaligned.`
          : ""}{" "}
        Party colors preserve the identity of the leading candidate or group.
      </p>
    </section>
  );
}
