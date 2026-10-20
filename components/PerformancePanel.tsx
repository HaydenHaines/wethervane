import { explainResearchNote } from "../lib/forecast-copy";
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
    <section
      id="historical-performance"
      className="performance-card"
      aria-labelledby="performance-title"
    >
      <h2 className="section-title" id="performance-title">
        {p ? "Against the actual result" : "What has been tested?"}
      </h2>
      {p ? (
        <>
          <p className="small">
            Historical results · {scenario.label} · Lower scores indicate
            better predictive performance.
          </p>
          <p className="performance-note">Seat-count error (CRPS) scores all the possible seat totals and their probabilities against the actual total, in seat units. Chamber-control error (Brier) scores the probabilities of Democratic, Republican and unresolved control on a 0–2 scale. For both, 0 is perfect and lower is better.</p>
          <div className="performance-grid">
            {[
              {
                label: "Democratic seat-count error (CRPS)",
                value: p.dSeatCrps,
                base: reference?.dSeatCrps,
              },
              {
                label: "Republican seat-count error (CRPS)",
                value: p.rSeatCrps,
                base: reference?.rSeatCrps,
              },
              {
                label: "Chamber-control error (Brier)",
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
                    {(x.value - x.base).toFixed(3)} vs comparison
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
            marks the actual result
            {baseline ? "; the dashed outline shows the comparison setup" : ""}.
          </p>
          <p className="performance-note">
            Actual control:{" "}
            {p.observedControl === "U"
              ? "unresolved"
              : `${p.observedControl} caucus`}
            . Actual seats: D {p.observedSeats.D}, R {p.observedSeats.R},
            unaligned {p.observedSeats.U}.{" "}
            {p.intervals
              .map(
                (i) =>
                  `${i.party} ${(i.level * 100).toFixed(0)}% modeled range ${i.lower}–${i.upper}: ${i.covered ? "included the actual result" : "missed the actual result"}`,
              )
              .join(". ")}
            .
          </p>
          <details className="assumptions">
            <summary>How these scores work</summary>
            <p className="detail-note">
              CRPS stands for continuous ranked probability score. For seat totals, it sums the squared differences between the forecast’s chance of being at or below each total and whether the actual result was at or below it. Brier adds squared errors across the three possible control outcomes. Neither score by itself establishes that this setup will predict future elections well.
            </p>
          </details>
        </>
      ) : (
        <p className="performance-note">
          The 2026 election has not happened yet. Select 2018 or 2022 to see how historical reconstructions compare with actual results. Those views use a different model of possible winners, so their scores do not measure the accuracy of this 2026 two-contender forecast.
        </p>
      )}

      <details className="assumptions">
        <summary>Scope and limitations</summary>
        {bundle.limitations.map((text, i) => (
          <p className="detail-note" key={i}>
            {explainResearchNote(text)}
          </p>
        ))}
      </details>
    </section>
  );
}
