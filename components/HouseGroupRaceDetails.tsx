import type { HouseGroupBloc, HouseGroupMember, HouseGroupRace, HouseResultGroup } from "../lib/house-group-result.ts";
import { COLORS, probability } from "../lib/presentation.ts";

const blocName: Record<HouseGroupBloc, string> = { D: "Democratic group", R: "Republican group", O: "Other group" };
const ruleName = { plurality: "Plurality election", ranked_choice: "Ranked-choice election", majority_runoff: "Majority-runoff election" };
function ruleExplanation(rule: HouseGroupRace["ruleFamily"]) {
  if (rule === "plurality") return "In this format, the candidate with the most votes wins.";
  if (rule === "ranked_choice") return "Voters rank candidates, and those rankings are counted through rounds.";
  return "In this format, candidates may advance to another election if no one reaches the required majority.";
}
function sourceBasisLabel(sourceBasis: string) {
  return ({
    FEC_plurality_maximum_or_sole_unopposed: "Federal Election Commission winner record",
    FEC_GE_winner_indicator: "Federal Election Commission winner record",
    Alaska_2024_official_RCV_final_round: "Alaska official ranked-choice final-round result",
    Clerk_2024_candidate_totals: "Clerk candidate totals",
    Clerk_2024_final_RCV_round: "Clerk final ranked-choice round result",
    Clerk_unopposed_no_vote_count: "Clerk record for an unopposed candidate; no vote count",
  } as Record<string, string>)[sourceBasis] || "Recorded election result";
}
function memberKind(member: HouseGroupMember) {
  if (member.kind === "source_reported_name") return "Name listed in a source · not confirmed here as a candidate in this race";
  if (member.kind === "pooled_unidentified") return "Combined or unidentified entry";
  return "Named candidate";
}
function resolvesIndividual(group: HouseResultGroup) {
  return group.id !== "O" && group.namedSingletonResolutionEligible && group.members.length === 1
    && group.members[0].kind === "named_candidate" && group.individualCandidateProbability !== null;
}
function diff(current: number, compared: number) {
  const value = (current - compared) * 100;
  return `${value > 0 ? "+" : ""}${value.toFixed(1)} percentage points (selected minus compared)`;
}

export default function HouseGroupRaceDetails({ race, comparisonRace }: { race: HouseGroupRace; comparisonRace?: HouseGroupRace | null }) {
  const compared = comparisonRace && comparisonRace.districtKey === race.districtKey && comparisonRace.cycle === race.cycle ? comparisonRace : null;
  const groups = (["D", "R", "O"] as const).map(id => race.groups.find(group => group.id === id)).filter((group): group is HouseResultGroup => !!group);
  return <section aria-label={`${race.districtKey} group-win details`}>
    <h3 className="detail-label">Chance of group winning</h3>
    {compared && <p className="house-candidate-comparison">Selected and compared forecasts use the same district and group-win target. Differences are descriptive; no precision claim is implied.</p>}
    {groups.map(group => {
      const chance = race.probabilities[group.id];
      const individual = resolvesIndividual(group);
      return <article className="candidate" key={group.id}>
        <div className="candidate-top">
          <div><div className="candidate-name">{group.label || blocName[group.id]}</div><span className="party-label">{blocName[group.id]}</span></div>
          <strong className="candidate-prob" style={{ color: COLORS[group.id] }}>{probability(chance)}</strong>
        </div>
        {compared && <p className="house-candidate-comparison">Compared forecast: <strong>{probability(compared.probabilities[group.id])}</strong> · {diff(chance, compared.probabilities[group.id])}.</p>}
        <div className="candidate-track"><span style={{ width: `${chance * 100}%`, background: COLORS[group.id] }} /></div>
        {race.structural && chance === 1
          ? <p className="detail-note">Only this group appears in the source candidate list, so the model assigns it 100% of the group probability. This is a model rule, not a guarantee of the election result.</p>
          : !race.structural && (chance === 0 || chance === 1)
            ? <p className="detail-note">{chance === 1 ? "Every saved simulation draw shown here yielded this group as the winner." : "No saved simulation draw shown here yielded this group as the winner."} That does not make the outcome guaranteed or impossible.</p>
            : null}
        <p className="detail-note">{individual
          ? <>The source lists one named candidate for this D/R group. Candidate chance in this result: <strong>{probability(group.individualCandidateProbability!)}</strong>.</>
          : <>This is a group chance. The result does not assign individual chances to members of this group.</>}</p>
        <ul className="detail-note">
          {group.members.map(member => <li key={member.id}><strong>{member.name}</strong> · {memberKind(member)}</li>)}
        </ul>
      </article>;
    })}
    <p className="detail-note">These probabilities estimate which party/candidate group ultimately wins; they are not vote shares. Other-group probability stays separate and is not distributed among named people or write-ins.</p>
    <p className="detail-note"><strong>Election format:</strong> {ruleName[race.ruleFamily]}. {ruleExplanation(race.ruleFamily)} This result uses an aggregate approximation: it does not simulate vote counts, ranked-choice counting rounds, or runoff rounds.</p>
    <p className="detail-note"><strong>How to read the probability:</strong> {race.structural
      ? "Only one group appears in the source candidate list, so the model assigns it all of the group probability. This depends on the source list and does not guarantee the actual result."
      : race.probabilityMethod === "modeled_joint_draw_frequency"
        ? "The probability is the share of model simulation runs in which the group won. It depends on the saved model estimates."
        : "The probability comes from the saved model's group-winner calculation."}</p>
    {race.observed.status === "observed" ? <p className="observed-result winner">
      Recorded result: <strong>{blocName[race.observed.groupId]} won</strong>{race.observed.winnerNames.length ? <> · Name in record: <strong>{race.observed.winnerNames.join(", ")}</strong></> : null}
      <span> · Result source: {sourceBasisLabel(race.observed.sourceBasis)}</span>
    </p> : <p className="observed-result">Recorded result unavailable: this election was annulled, so no winner is available to score here.</p>}
    {!race.groupScoreEligible && <p className="detail-note">This district is not included in the scored coverage.</p>}
  </section>;
}
