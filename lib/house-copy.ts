/** Presentation-only explanations of the reviewed House benchmark; source bytes stay intact. */
export const HOUSE_HISTORICAL_CONTEXT = "These are historical reconstructions using some inputs assembled after the election. Each model was fitted on earlier election results, but these test elections were also examined during development. The scores describe this study, not forecasts published before election day or proof of future accuracy.";

const notes: Record<string, string> = {
  "Current editions of historical inputs and final candidate rosters are retrospective proxies; this is not a strict as-of replay.": "Some past inputs and candidate lists use versions assembled after the election. This is a historical reconstruction, not a replay of exactly what forecasters knew at the time.",
  "All fitted labels are from strictly earlier cycles. These are revisited historical development comparisons, not prospective validation.": "The model was fitted using earlier election results, then compared with later past elections. Those test elections were also examined during development; this is not an independent test of forecasts made before those elections.",
  "Shares condition on Democratic and Republican candidate votes; other candidates, turnout and vote counts are not predicted.": "Vote share counts Democratic and Republican candidate votes only. Votes for other candidates, turnout and total vote counts are not modeled.",
  "No full-chamber, chamber-control, chamber-tipping, or current-election forecast is supplied.": "These results do not cover the whole House or predict its current election. They cannot show who controls the chamber or which district tips the majority.",
  "District geometry is not admitted by this result. Map identities must be verified separately.": "District boundaries have not yet been verified for these elections, so these results do not include a district map.",
  "Simulation diagnostics are conditional on the saved historical bank. Predictive resampling does not remove fitted-bank uncertainty.": "Simulation checks depend on the saved model estimates. Simulating more elections from those same estimates can reduce simulation noise, but does not eliminate uncertainty or numerical error in estimating the model itself.",
  "Candidate member IDs are local artifact identities, not official FEC identifiers.": "Candidate IDs match records within this saved result. They are not official Federal Election Commission identifiers.",
  "The eligible-seat subtotal fails the unchanged Senate subtotal-mean Monte Carlo precision gate; race diagnostics are reported separately.": "The numerical precision check for adding expected seats across the included districts did not pass, so no seat subtotal is reported. Individual race calculations passed their separate checks. Neither result establishes who controls the whole House.",
  "Historical fundamentals benchmark; source vintages are retrospective proxies.": "This historical model uses electoral history. Some source information was assembled after the election.",
};

export function houseResearchNote(note: string, hasVerifiedGeography = false): string {
  if (hasVerifiedGeography && note === "District geometry is not admitted by this result. Map identities must be verified separately.")
    return "District boundaries were checked separately against this election’s map. They locate the published races without changing the model’s predictions or adding forecasts for excluded districts.";
  const coverage = /^Eligible D\/R benchmark for (\d+) of 435 districts; excluded districts remain in the coverage ledger\.$/.exec(note);
  return coverage
    ? `This comparison includes ${coverage[1]} of 435 districts that met the study's Democratic–Republican race rules. Excluded districts and their reasons are listed below.`
    : notes[note] ?? note;
}

export function houseExclusionReason(reason: string | null): string {
  if (!reason) return "No exclusion reason was supplied.";
  return reason.split(";").map(code => {
    const nominees = /^([DR])_nominee_count_(\d+)$/.exec(code);
    if (nominees) return `${nominees[2]} ${nominees[1] === "D" ? "Democratic" : "Republican"} nominees in the source roster; this model requires exactly one nominee from each party.`;
    return ({
      ranked_choice_outside_pair_scope: "Ranked-choice election; this model does not simulate its counting rounds.",
      majority_runoff_or_jungle_rule_outside_pair_scope: "A runoff or all-candidate election format falls outside this model's supported two-candidate rules.",
      two_incumbents_outside_engine_support: "Two sitting House members competed; this model does not support that matchup.",
      annulled_election_no_certified_target: "The election was annulled and has no certified winner for this study.",
    } as Record<string, string>)[code] ?? code;
  }).join(" ");
}

export function houseModelDescription(configurationId: string, year: number): string {
  const description = ({
    house_history_refreshed_v1: "Uses each district's earlier presidential result as a starting point, with a fixed relationship between presidential and House voting.",
    house_presidential_slope_v1: "Learns from earlier elections how strongly a district's presidential result predicts its House vote, instead of fixing that relationship.",
    house_relative_lean_v1: "Separates a district's earlier presidential lean from the national presidential result, then learns how both relate to House voting.",
    house_relative_context_v1: "Adds the party holding the White House and whether it is a midterm election to the district and national voting patterns.",
  } as Record<string, string>)[configurationId];
  const qualification = configurationId === "house_relative_context_v1" && year === 2018
    ? " For this 2018 reconstruction, the training elections covered only Democratic presidents, so applying the White House effect under a Republican president extends beyond the situations observed in training."
    : "";
  return (description || "A saved House electoral-history model fitted on earlier elections.") + qualification + " Its components were fitted together and cannot be independently removed from this saved result.";
}

export const houseEvidenceLabel = (label: string): string => label === "Reviewed historical race benchmark; retrospective source proxies"
  ? "Reviewed calculations on past elections; some inputs were reconstructed later"
  : label;

export const houseSelectionNote = (note: string): string => note === "All prespecified laws retained; no winner selected from the target outcomes."
  ? "All four models planned for this study are available. None was selected as the winner based on these historical results."
  : note;
