import { housePartyShareView, type HousePartyShareFixture } from "../lib/house-party-share-view.ts";

const percent = (value: number) => `${(value * 100).toFixed(1)}%`;
const level = (value: number) => `${new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 }).format(value * 100)}%`;
const votes = (value: number) => value.toLocaleString("en-US");

/** Private fixture preview. No public route or publication adapter imports this component. */
export default function HousePartySharePreview({ fixture }: { fixture: HousePartyShareFixture }) {
  const view = housePartyShareView(fixture);
  return <section className="count-preview" aria-label="Invented House party share preview">
    <div className="count-fixture-banner" role="note">
      <strong>Illustrative data only · Private preview</strong>
      <p>These examples are invented to test the interface. They are not election forecasts.</p>
    </div>
    <header>
      <h2>{view.district} · First ballot</h2>
      <p><strong>Democratic share of first-ballot D+R votes.</strong> Adds votes for all Democratic nominees and divides by votes for all Democratic and Republican nominees. Third-party votes are excluded from this denominator.</p>
    </header>
    <div className="count-grid">
      <section className="count-quantity" data-quantity="estimate">
        <h3>Party vote share</h3>
        {view.estimate.status === "modeled" ? <>
          <span className="count-kind">Model estimate</span>
          <p className="count-value">{percent(view.estimate.mean)}</p>
          <p>Average across modeled elections.</p>
          <ul className="count-intervals">{view.estimate.intervals.map(interval => <li key={interval.level}>
            Middle {level(interval.level)} of modeled shares: {percent(interval.lower)}–{percent(interval.upper)}
          </li>)}</ul>
        </> : view.estimate.status === "structural" ? <>
          <span className="count-kind">Fixed by party absence</span>
          <p className="count-value">{percent(view.estimate.value)}</p>
          <p>No {view.estimate.absentParty === "D" ? "Democratic" : "Republican"} nominee is present. The party on the ballot is assumed to receive a positive vote total.</p>
          <p>{view.estimate.reason}</p>
          <p>This endpoint follows party absence; it is not a simulation result.</p>
        </> : <p className="count-unavailable"><strong>Not available.</strong> {view.estimate.reason}</p>}
      </section>
      <section className="count-quantity" data-quantity="observed">
        <h3>Recorded party vote share</h3>
        {view.observed.status === "reported" ? <>
          <span className="count-kind">Recorded result · Invented example</span>
          <p className="count-value">{percent(view.observed.share)}</p>
          <p>{votes(view.observed.numerator)} Democratic votes out of {votes(view.observed.denominator)} Democratic and Republican votes.</p>
        </> : <p className="count-unavailable"><strong>Not available.</strong> {view.observed.reason}</p>}
      </section>
    </div>
    <section className="count-scope" data-quantity="winner">
      <h3>Individual candidate win probabilities</h3>
      <p><strong>Not available.</strong> {view.candidateWinner.reason}</p>
      <p>The party-share measure does not determine a winner. Ranked-choice and runoff final winners are separate outputs. No winner score or chamber-control estimate is supplied by this preview.</p>
    </section>
    <details className="count-checks">
      <summary>About this example</summary>
      <p>{view.context}</p>
      <p>Stage: First ballot. Calculation: D / (D + R). Units: share fraction, displayed as percentages. Source status: invented fixture.</p>
    </details>
  </section>;
}
