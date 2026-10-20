"use client";

import { useEffect, useRef } from "react";
import type { ModelTippingSummary } from "../lib/tipping-types";
import type { RaceForecast } from "../lib/site-types";
import {
  COLORS,
  STATE_NAMES,
  percent,
  probability,
  raceColor,
  raceRating,
} from "../lib/presentation";

type Props = {
  summary: ModelTippingSummary;
  scenario: string;
  races: RaceForecast[];
  selectedId: string;
  onSelect: (raceId: string) => void;
};

function projectedMargin(race: RaceForecast): string {
  const margin = (2 * race.firstShareMean - 1) * 100;
  const leader = race.candidates[margin >= 0 ? 0 : 1];
  return `${leader.party} +${Math.abs(margin).toFixed(1)} pp`;
}

export default function TippingRibbon({
  summary,
  scenario,
  races,
  selectedId,
  onSelect,
}: Props) {
  const scroller = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const container = scroller.current;
    if (!container || container.scrollWidth <= container.clientWidth) return;
    const tile = Array.from(
      container.querySelectorAll<HTMLButtonElement>("button"),
    ).find((button) => button.dataset.raceId === selectedId);
    if (tile)
      container.scrollLeft =
        tile.offsetLeft -
        container.offsetLeft -
        (container.clientWidth - tile.offsetWidth) / 2;
  }, [summary, scenario, selectedId]);
  const caucuses =
    summary.scenarios[scenario as keyof typeof summary.scenarios];
  if (!caucuses) return null;
  const byId = new Map(races.map((race) => [race.id, race]));
  const selected = byId.get(selectedId);
  const guaranteedR = new Set(caucuses.R.guaranteedRaceIds);
  const guaranteedD = new Set(caucuses.D.guaranteedRaceIds);
  return (
    <section className="tipping-card" aria-labelledby="tipping-title">
      <div className="tipping-heading">
        <div>
          <h2 id="tipping-title" className="section-title">
            The path to a majority
          </h2>
          <p className="small">
            {races.length} elections, ordered by projected principal-contender
            margin.
          </p>
        </div>
        <span className="ribbon-key">
          <span className="threshold-key">D</span>
          <span className="threshold-key republican">R</span>Projected control
          thresholds
        </span>
      </div>
      <div className="ribbon-axis" aria-hidden="true">
        <span>Democratic / independent advantage</span>
        <span>Republican advantage</span>
      </div>
      <div
        ref={scroller}
        className="tipping-scroll"
        role="region"
        aria-label="Senate race ribbon, ordered by projected margin. Scroll horizontally for all 35 elections."
        tabIndex={0}
      >
        <div className="tipping-track">
          <div className="retained-endcap" style={{ color: COLORS.D }}>
            <strong>{caucuses.D.retainedSeats}</strong>
            <span>
              D caucus
              <br />
              retained
            </span>
          </div>
          {summary.raceOrder.map((id) => {
            const race = byId.get(id);
            if (!race) return null;
            const isD = caucuses.D.projectedTippingRaceId === id;
            const isR = caucuses.R.projectedTippingRaceId === id;
            const guaranteed = guaranteedR.has(id)
              ? "R"
              : guaranteedD.has(id)
                ? "D"
                : null;
            const label = `${STATE_NAMES[race.state]}, ${/special/.test(id) ? "special election" : "regular election"}, ${raceRating(race)}, projected margin ${projectedMargin(race)}${isD ? ", projected D control threshold" : ""}${isR ? ", projected R control threshold" : ""}${guaranteed ? `, ${guaranteed} caucus regardless of winner` : ""}. At the D control threshold in ${percent(caucuses.D.frequencyByRace[id] || 0)} of simulations. At the R control threshold in ${percent(caucuses.R.frequencyByRace[id] || 0)} of simulations.`;
            return (
              <button
                key={id}
                data-race-id={id}
                className={`race-tile${selectedId === id ? " selected" : ""}${raceRating(race) === "Likely" ? " likely" : ""}${guaranteed ? " guaranteed" : ""}`}
                style={{ background: raceColor(race) }}
                aria-label={label}
                aria-pressed={selectedId === id}
                onClick={() => onSelect(id)}
                title={label}
              >
                {(isD || isR) && (
                  <span className="tile-markers" aria-hidden="true">
                    {isD && isR ? (
                      <span className="threshold-combined">
                        D{caucuses.D.thresholdSeats} / R
                        {caucuses.R.thresholdSeats} · {race.state}
                      </span>
                    ) : (
                      <>
                        {isD && (
                          <span className="threshold-key">
                            D{caucuses.D.thresholdSeats}
                          </span>
                        )}
                        {isR && (
                          <span className="threshold-key republican">
                            R{caucuses.R.thresholdSeats}
                          </span>
                        )}
                      </>
                    )}
                  </span>
                )}
                <span className="tile-state">{race.state}</span>
                <span className="tile-seat">
                  {/special/.test(id) ? "SP" : "’26"}
                </span>
                {guaranteed && (
                  <span className="tile-guaranteed" aria-hidden="true">
                    {guaranteed}*
                  </span>
                )}
              </button>
            );
          })}
          <div className="retained-endcap" style={{ color: COLORS.R }}>
            <strong>{caucuses.R.retainedSeats}</strong>
            <span>
              R caucus
              <br />
              retained
            </span>
          </div>
        </div>
      </div>
      <div className="tipping-summaries">
        {(["D", "R"] as const).map((party) => {
          const caucus = caucuses[party];
          const race = caucus.projectedTippingRaceId
            ? byId.get(caucus.projectedTippingRaceId)
            : undefined;
          const mostFrequent = [...races].sort(
            (a, b) =>
              caucus.frequencyByRace[b.id] - caucus.frequencyByRace[a.id] ||
              a.id.localeCompare(b.id),
          )[0];
          return (
            <div className="tipping-summary" key={party}>
              <span
                className={`threshold-key${party === "R" ? " republican" : ""}`}
              >
                {party}
              </span>
              <div>
                <strong>
                  {race ? (
                    <button onClick={() => onSelect(race.id)}>
                      {STATE_NAMES[race.state]}
                      {/special/.test(race.id) ? " special" : ""}
                    </button>
                  ) : (
                    "Threshold unreachable"
                  )}
                </strong>
                <span>
                  {caucus.thresholdSeats} caucus seats for control
                  {race && caucus.projectedMarginPp !== null
                    ? ` · projected margin ${caucus.projectedMarginPp >= 0 ? "+" : "−"}${Math.abs(caucus.projectedMarginPp).toFixed(1)} pp`
                    : ""}
                </span>
                {mostFrequent && (
                  <span>
                    Most frequent in simulations:{" "}
                    <button onClick={() => onSelect(mostFrequent.id)}>
                      {STATE_NAMES[mostFrequent.state]}
                    </button>{" "}
                    · {probability(caucus.frequencyByRace[mostFrequent.id])}
                  </span>
                )}
              </div>
            </div>
          );
        })}
      </div>
      {selected && (
        <div className="selected-threshold" aria-live="polite">
          <strong>
            {STATE_NAMES[selected.state]} · {projectedMargin(selected)}
          </strong>
          <span>
            At the D control threshold in{" "}
            <b>{probability(caucuses.D.frequencyByRace[selectedId] || 0)}</b> of
            simulations.
          </span>
          <span>
            At the R control threshold in{" "}
            <b>{probability(caucuses.R.frequencyByRace[selectedId] || 0)}</b> of
            simulations.
          </span>
        </div>
      )}
      {guaranteedR.size > 0 && (
        <p className="ribbon-guarantee">
          <b>R*</b> {guaranteedR.size} independent–Republican contests count
          toward the R caucus whichever contender wins in this scenario. These
          remain separate from the {caucuses.R.retainedSeats} retained R caucus
          seats.
        </p>
      )}
      <details className="ribbon-method">
        <summary>How to read the thresholds</summary>
        <p>
          The ribbon follows mean projected margins, from the Democratic or
          independent contender’s advantage to the Republican contender’s
          advantage. Tile colors show the contender’s genuine party and Safe,
          Likely, or Tossup category. The alignment selector above governs
          caucus membership, not party identity.
        </p>
        <p>
          The D and R markers identify each caucus’s last needed eligible race
          in projected order: {caucuses.D.requiredContestedWins} contested wins
          for D and {caucuses.R.requiredContestedWins} for R, after retained and
          guaranteed seats. D needs {caucuses.D.thresholdSeats} seats and R
          needs {caucuses.R.thresholdSeats}, reflecting the modeled tie rule. A
          marker’s margin is measured from that caucus contender’s perspective.
        </p>
        <p>
          Simulation frequencies come from{" "}
          {summary.simulationCount.toLocaleString()} saved joint draws. In each
          draw, eligible races are ranked by their caucus contender’s share, and
          the race at the control threshold is recorded. Draws where the caucus
          falls short still count. D and R frequencies are separate
          distributions; they do not add to one total. A frequency is not the
          probability that changing one race alone would change chamber control.
        </p>
      </details>
    </section>
  );
}
