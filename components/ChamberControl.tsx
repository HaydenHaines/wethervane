import { forecastLabel } from "../lib/forecast-copy";
import { explainResearchNote } from "../lib/forecast-copy";
import type { ExplorerScenario } from "../lib/explorer-types";
import { COLORS, ppChange, probability } from "../lib/presentation";

export default function ChamberControl({
  scenario,
  baseline,
  chamber = "senate",
}: {
  scenario: ExplorerScenario;
  baseline?: ExplorerScenario;
  chamber?: "senate" | "house";
}) {
  const stats = [
    {
      label: "Democratic control",
      p: scenario.control.democraticMajorityProbability,
      base: baseline?.control.democraticMajorityProbability,
      color: COLORS.D,
    },
    {
      label: "Republican control",
      p: scenario.control.republicanMajorityProbability,
      base: baseline?.control.republicanMajorityProbability,
      color: COLORS.R,
    },
    {
      label: "Control depends on alliances",
      p: scenario.control.unresolvedOrganizingProbability,
      base: baseline?.control.unresolvedOrganizingProbability,
      color: COLORS.unresolved,
    },
  ];
  return (
    <section className="control-card" aria-labelledby="control-title">
      <div className="section-top">
        <div>
          <h2 id="control-title" className="section-title">
            Who could control the chamber?
          </h2>
          <div className="small">
            {forecastLabel(scenario.label)} · D needs {scenario.thresholds.D}; R needs{" "}
            {scenario.thresholds.R}
          </div>
        </div>
      </div>
      <div className="control-probs" aria-live="polite">
        {stats.map((x) => (
          <div key={x.label} style={{ color: x.color }}>
            <span className="prob-label">
              <span className="dot" />
              {x.label}
            </span>
            <span className="prob-number">{probability(x.p)}</span>
            {x.base !== undefined && (
              <span className="comparison-delta">
                {ppChange(x.p - x.base)} vs comparison
              </span>
            )}
          </div>
        ))}
      </div>
      <div className="prob-track" aria-hidden="true">
        {stats.map((x) => (
          <span
            key={x.label}
            style={{ width: `${x.p * 100}%`, background: x.color }}
          />
        ))}
      </div>
      <p className="control-note">
        {explainResearchNote(scenario.description)} Unresolved control means the winners are known in a simulation, but the winning members’ party alliances have not determined who organizes the chamber.
      </p>
      <p className="control-note">A caucus is the group working with a party to organize the chamber, including any allied independents. {chamber === "senate" ? "The party holding the vice presidency can control a 50–50 Senate through its tie-breaking vote." : "Control requires a majority of House seats; there is no vice-presidential tie-breaking vote in the House."}</p>
      <div className="chamber-seat-summary">
        {(["D", "R", "U"] as const).map((party) => (
          <span key={party}>
            {party === "U" ? "Unaligned" : `${party} caucus`}:{" "}
            <b>{scenario.seats[party].expectedSeats.toFixed(1)}</b> expected
            seats{" "}
            <span className="small">
              ({scenario.seats[party].seatQuantiles.p05}–
              {scenario.seats[party].seatQuantiles.p95}, modeled 90% range)
            </span>
          </span>
        ))}
      </div>
      <p className="detail-note">Expected seats are the average across simulated elections, so they can include fractions. The 90% range spans the 5th to 95th percentiles; outcomes outside that range remain possible.</p>
    </section>
  );
}
