import { houseWinnerView, type HouseWinnerFixture } from "../lib/house-winner-view.ts";

const groupLabel = (bloc: "D" | "R" | "O") => ({
  D: "Democratic group",
  R: "Republican group",
  O: "Other candidates & write-ins",
})[bloc];
const percent = (value: number) => value > 0 && value < .001 ? "<0.1%"
  : value < 1 && value > .999 ? ">99.9%" : `${(value * 100).toFixed(1)}%`;
const ruleLabel = { plurality: "Plurality", ranked_choice: "Ranked choice", majority_runoff: "Majority with a possible runoff" };

/** Private presentation preparation; this consumes invented fixtures, not a public result. */
export default function HouseWinnerPreview({ fixture }: { fixture: HouseWinnerFixture }) {
  const view = houseWinnerView(fixture);
  const prediction = view.prediction;
  const observed = view.truth.status === "observed" ? view.truth : null;
  const observedGroup = observed ? view.groups.find(group => group.id === observed.groupId) : undefined;
  return <section className="count-preview" aria-label="Invented House group winner preview">
    <div className="count-fixture-banner" role="note">
      <strong>Illustrative data only · Private preview</strong>
      <p>These examples are invented to test the interface. They are not election forecasts.</p>
    </div>
    <header>
      <h2>{view.district} · Chance each party or group wins</h2>
      <p>Each percentage applies to a whole group. It does not give the chance for each person in that group.</p>
      <p>Election rule: <strong>{ruleLabel[view.ruleFamily]}</strong>. This model approximates the final group winner. It does not simulate ranked-choice transfers or a later runoff.</p>
    </header>
    {prediction.status === "unavailable" && <p className="count-warning"><strong>Winner forecast unavailable.</strong> {prediction.reason}</p>}
    <div className="count-grid">
      {view.groups.map(group => {
        const chance = prediction.status === "modeled"
          ? prediction.probabilities.find(row => row.groupId === group.id)!.probability
          : prediction.status === "structural" ? 1 : null;
        return <section className="count-quantity" data-group={group.id} key={group.id} aria-label={groupLabel(group.bloc)}>
          <h3>{groupLabel(group.bloc)}</h3>
          {prediction.status === "structural" ? <>
            <span className="count-kind">All listed candidates are in this group</span>
            <p className="count-value">100% <span>among the listed candidates</span></p>
            <p>Every possible winner represented by this setup belongs to this group. This depends on the listed candidates; it is not a guarantee about the election.</p>
            <p>{prediction.reason}</p>
          </> : chance !== null ? <>
            <span className="count-kind">Modeled chance to win</span>
            <p className="count-value">{percent(chance)}</p>
            {(chance === 0 || chance === 1) && <p>The model assigns {percent(chance)} in this example. This is not a guarantee about the election.</p>}
          </> : <p className="count-unavailable">Chance unavailable</p>}
          <h4>Candidates and categories represented</h4>
          <ul>{group.members.map(member => <li key={member.id}>
            {member.kind === "named_person" ? member.label : <><strong>Pooled report:</strong> {member.label}</>}
          </li>)}</ul>
          {group.members.some(member => member.kind === "pooled_or_residual") && <p>A pooled report combines entries without identifying each person. It is not one candidate.</p>}
          <p>Individual candidate chances are not supplied by this group forecast.</p>
        </section>;
      })}
    </div>
    <section className="count-scope" data-quantity="observed">
      <h3>Recorded result</h3>
      {observed && observedGroup ? <>
        <span className="count-kind">Recorded outcome · Invented example</span>
        <p>Recorded winner’s group: <strong>{groupLabel(observedGroup.bloc)}</strong>.</p>
        {!!observed.winnerNames?.length && <p>Reported winner: {observed.winnerNames.join("; ")}.</p>}
      </> : <p><strong>Recorded winner unavailable. Not scored.</strong> {view.truth.status === "unavailable" && view.truth.reason}</p>}
    </section>
    <p>First-ballot party vote share is a separate measure. This preview supplies no vote-share estimate, historical score, chamber-control estimate or tipping point.</p>
    <details className="count-checks">
      <summary>About this example</summary>
      <p>{view.approximationNote}</p>
      {observed && <p>Recorded outcome basis: {observed.sourceBasis}.</p>}
      <p>The fixture describes the final party/group outcome. Group membership and recorded outcomes are supplied separately; a name match or a one-entry source row does not establish an individual candidate forecast.</p>
    </details>
  </section>;
}
