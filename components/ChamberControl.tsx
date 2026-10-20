import type { ExplorerScenario } from "../lib/explorer-types";
import { COLORS, ppChange, probability } from "../lib/presentation";

export default function ChamberControl({
  scenario,
  baseline,
}: {
  scenario: ExplorerScenario;
  baseline?: ExplorerScenario;
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
      label: "Unresolved alignment",
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
            {scenario.label} · D needs {scenario.thresholds.D}; R needs{" "}
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
                {ppChange(x.p - x.base)} vs default
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
        {scenario.description} Unresolved refers to organizing arrangements, not
        an undecided election.
      </p>
      <div className="chamber-seat-summary">
        {(["D", "R", "U"] as const).map((party) => (
          <span key={party}>
            {party === "U" ? "Unaligned" : `${party} caucus`}:{" "}
            <b>{scenario.seats[party].expectedSeats.toFixed(1)}</b> expected
            seats{" "}
            <span className="small">
              ({scenario.seats[party].seatQuantiles.p05}–
              {scenario.seats[party].seatQuantiles.p95}, 90% interval)
            </span>
          </span>
        ))}
      </div>
    </section>
  );
}
