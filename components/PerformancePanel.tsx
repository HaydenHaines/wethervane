import type { ExplorerBundle, ExplorerScenario } from "../lib/explorer-types";
import SeatDistribution from "./SeatDistribution";

export default function PerformancePanel({
  bundle,
  scenario,
  baseline,
}: {
  bundle: ExplorerBundle;
  scenario: ExplorerScenario;
  baseline?: ExplorerScenario;
}) {
  const p = scenario.performance;
  const reference = baseline?.performance;
  return (
    <section className="performance-card" aria-labelledby="performance-title">
      <h2 className="section-title" id="performance-title">
        {p ? "Against the observed result" : "What has been tested?"}
      </h2>
      {p ? (
        <>
          <p className="small">
            Historical outcomes · {scenario.label} · Lower scores indicate
            better predictive performance.
          </p>
          <div className="performance-grid">
            {[
              {
                label: "D seat CRPS",
                value: p.dSeatCrps,
                base: reference?.dSeatCrps,
              },
              {
                label: "R seat CRPS",
                value: p.rSeatCrps,
                base: reference?.rSeatCrps,
              },
              {
                label: "Control Brier score",
                value: p.controlBrier,
                base: reference?.controlBrier,
              },
            ].map((x) => (
              <div className="performance-metric" key={x.label}>
                <span>{x.label}</span>
                <strong>{x.value.toFixed(3)}</strong>
                {x.base !== undefined && (
                  <span>
                    {x.value - x.base >= 0 ? "+" : ""}
                    {(x.value - x.base).toFixed(3)} vs default
                  </span>
                )}
              </div>
            ))}
          </div>
          <div className="seat-distributions">
            {(["D", "R"] as const).map((party) => (
              <SeatDistribution
                key={party}
                party={party}
                distribution={scenario.seats[party]}
                observed={p.observedSeats[party]}
                baseline={baseline?.seats[party]}
              />
            ))}
          </div>
          <p className="performance-note">
            Bars show the probability of each seat total. The vertical line
            marks the observed result
            {baseline ? "; the dashed outline shows the default stack" : ""}.
          </p>
          <p className="performance-note">
            Observed control:{" "}
            {p.observedControl === "U"
              ? "unresolved"
              : `${p.observedControl} caucus`}
            . Observed seats: D {p.observedSeats.D}, R {p.observedSeats.R},
            unaligned {p.observedSeats.U}.{" "}
            {p.intervals
              .map(
                (i) =>
                  `${i.party} ${(i.level * 100).toFixed(0)}% interval ${i.lower}–${i.upper}: ${i.covered ? "covered" : "missed"}`,
              )
              .join(". ")}
            .
          </p>
          <details className="assumptions">
            <summary>How these scores work</summary>
            <p className="detail-note">
              CRPS measures the full predicted seat-count distribution against
              the observed count. Control Brier compares probabilities for D, R,
              and unresolved control against the observed category. This is a
              descriptive comparison of saved runs, not evidence that a selected
              stack is calibrated or superior in future elections.
            </p>
          </details>
        </>
      ) : (
        <p className="performance-note">
          The 2026 election has not been observed. Historical performance
          belongs to the 2018 and 2022 winner laws; it is not a score for this
          current principal-contender model.
        </p>
      )}
      {bundle.historicalNote && (
        <p className="performance-note">{bundle.historicalNote}</p>
      )}
      <details className="assumptions">
        <summary>Scope and limitations</summary>
        {bundle.limitations.map((text, i) => (
          <p className="detail-note" key={i}>
            {text}
          </p>
        ))}
      </details>
    </section>
  );
}
