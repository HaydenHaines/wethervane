/** Plain-language presentation of saved research notes; never changes source artifacts. */
export const HISTORICAL_CONTEXT = "Recreated in 2026 using some inputs assembled after the election. These elections were also examined during model development, so this is not a test made with only information available at the time, or proof of how the 2026 model will perform.";

const pair = "This forecast considers only the two named contenders. Other candidates, write-ins and undecided responses are outside this model; the shares do not describe the whole ballot.";
const plurality = "The model compares only the two named contenders and gives the win to the one with more modeled votes. Other eligible candidates are outside this forecast.";
const notes: Record<string, string> = {
  "Conditional principal-contender support; listed omitted candidates and valid write-ins/nonresponse are not modeled. No full-ballot vote share or candidate-count claim.": pair,
  "Conditional principal-contender support; listed omitted candidates and valid write-ins/nonresponse are not modeled. No full-ballot vote share or candidate-count claim. Independent opposition remains I. Prior Democratic presidential offset and historical D/R coefficients are borrowed as an unvalidated opposition proxy; no independent-specific advantage is added.": pair + " The independent keeps their party label, but the model borrows Democratic presidential voting patterns and historical Democratic–Republican relationships to estimate their support. That substitution has not been validated for independents, and there is no extra independent-candidate adjustment.",
  "Conditional two-principal plurality proxy; omitted valid options are excluded from modeled winner support.": plurality,
  "Conditional two-principal plurality proxy; omitted valid options are excluded from modeled winner support. Class III unexpired-term special is treated with regular-race discrepancy; this is an unvalidated transfer.": plurality + " This special election fills an unfinished Senate term. The model uses the same race-level uncertainty assumptions as a regular election; that transfer has not been validated.",
  "Conditional two-principal plurality proxy; omitted valid options are excluded from modeled winner support. For RCV, transfer/exhaustion rounds are approximated by head-to-head final-preference share; not a first-round share forecast.": pair + " Ranked-choice voting is approximated as a final head-to-head contest. The model does not separately simulate transferred votes or ballots that stop counting when no ranked candidate remains. These are not first-round shares.",
  "Conditional two-principal plurality proxy; omitted valid options are excluded from modeled winner support. General/runoff sequence is collapsed to an eventual two-principal majority proxy; runoff turnout and calendar are unmodeled.": pair + " The general election and any required runoff are approximated as one final head-to-head contest. Changes in turnout or timing between rounds are not modeled.",
  "Conditional two-principal plurality proxy; omitted valid options are excluded from modeled winner support. Louisiana 2026 U.S.Senate general is a plurality contest under current official election guidance; no inherited jungle-primary/runoff rule is applied.": plurality + " For Louisiana's 2026 Senate general election, the winner is the candidate with the most votes. The model does not apply the state's former all-candidate primary and runoff rules to this contest.",
  "Probabilities describe modeled candidate groups. Multiple candidates in the same party group have no separate winner probabilities.": "Some candidates from the same party are combined into one possible outcome. The chance of that group winning does not tell us which member would win.",
  "The incoming seat is defined after the final decisive voting event, including a required runoff; this is not an election-night forecast.": "The forecast concerns who ultimately takes the seat, including any required runoff election. It does not predict when the result will be known.",
  "Binary matchup/effective-share proxies; no full election-rule or all-candidate simulation.": pair,
  "Community identification: Not estimated in this baseline; no causal subgroup or composition attribution from toplines": "Overall poll percentages do not tell this model why particular communities vote as they do, or whether a change comes from different people responding.",
  "Current scope: Audited source subset with visible uncovered races, not an exhaustive polling census": "Only a reviewed set of poll sources is included. Some races have no included polls, and this collection does not contain every available poll.",
  "Fundamentals-only forecast; no current polling or candidate-quality term.": "This setup uses electoral history without current polls or a separate adjustment for candidate strength.",
  "Historical poll publication and candidate-support vintages use qualified retrospective proxies; this is not a strict real-time historical replay.": "Some historical poll availability dates and candidate information were reconstructed afterward. We cannot claim that every input was available in exactly this form before the election.",
  "Initial rerun: A computational forecast comparison; no demonstrated forecast improvement": "Completing this calculation does not demonstrate that it predicts elections better.",
  "No participation or vote-count forecast.": "The model does not predict how many people will vote or the total number of votes cast.",
  "Predictive Monte Carlo error is conditional on saved posterior draws; resampling does not increase posterior ESS.": "Simulation percentages have numerical noise. Reusing the same saved model estimates to simulate more elections does not add independent information about those estimates.",
  "Public comparators: Descriptive only; no tuning to public win probabilities": "Other public forecasts are used for comparison; this model was not adjusted to match their win chances.",
  "Retrospective development benchmark; 2018 and 2022 were revisited during model development and are not blind holdouts.": "The 2018 and 2022 elections were examined during model development. Their scores are not an independent test on elections the developers had never examined.",
  "Saved posterior diagnostics passed; protected 2022 predictions remain unscored.": "Checks on the saved model estimation passed. This current model's separate, reserved 2022 predictions have not been scored; the historical explorer uses a different model.",
  "Scattered/other unnamed write-in residuals are explicit source support hypotheses, not identified individual candidates or proof of exhaustive ballot eligibility.": "Some source results combine unnamed write-in or other candidates. These categories do not identify every candidate or establish that every eligible ballot option is represented.",
  "Scores use fixed observed organizing alignment and the actual vice-presidential tie-break; changing a forecast caucus scenario does not change observed truth.": "Historical scores use the party alliances and vice-presidential tie-breaking vote that actually occurred. Changing forecast assumptions does not change those actual results.",
  "The historical categorical winner-gate law differs from the saved 2026 principal-pair forecast; these results do not directly calibrate that artifact.": "Historical views estimate which candidate group wins. The 2026 view estimates a two-contender contest with vote shares. Success in one model does not establish the accuracy of the other.",
  "Winner draws support group probabilities and chamber outcomes, not full-field vote-share margins or margin-ranked tipping frequencies.": "Historical simulations estimate winning groups and chamber control. They do not provide vote-share leads or the tipping-point rankings that require those shares.",
  "Counterfactual: all four modeled independent opposition contenders join Democratic caucus; King and Sanders remain there.": "What if all four modeled independent challengers joined the Democrats if elected? King and Sanders remain with the Democrats. This is an assumption, not a prediction of those decisions.",
  "Counterfactual: all four modeled independent opposition contenders join Republican caucus; King and Sanders remain Democratic.": "What if all four modeled independent challengers joined the Republicans if elected? King and Sanders remain with the Democrats. This is an assumption, not a prediction of those decisions.",
  "Counterfactual: groups otherwise unaligned join the Democratic caucus; existing alignments are preserved.": "Assume otherwise unaligned winning groups work with the Democrats to organize the chamber. Existing party alliances stay the same.",
  "Counterfactual: groups otherwise unaligned join the Republican caucus; existing alignments are preserved.": "Assume otherwise unaligned winning groups work with the Republicans to organize the chamber. Existing party alliances stay the same.",
  "Existing King and Sanders remain in Democratic caucus; all four modeled independent opposition contenders unassigned, matching stated campaign pledges without treating pledges as certain future conduct.": "King and Sanders continue working with the Democrats. The four modeled independent challengers are assigned to neither party if elected, consistent with their stated pledges. Their eventual decisions could differ.",
  "Preserve the recorded pre-election organizing alignments; other groups remain unaligned.": "Use the party alliances recorded before the election. Other winning groups are assigned to neither party for chamber control.",
};

export const explainResearchNote = (text: string): string => notes[text] ?? text;

/** Translate only known labels; scientific identifiers and saved values stay intact. */
export function forecastLabel(label: string): string {
  return label.replace("common-cycle SD", "shared election-wide error size")
    .replace("Common-cycle error SD", "Shared election-wide error size")
    .replace("All polling-error SDs", "All polling-error sizes")
    .replace("all scales", "all error sizes")
    .replace("Polling error scales", "Polling-error sizes")
    .replace("Electoral-history aggregate", "Electoral-history starting estimate")
    .replace("Winner-only", "Winner reference model")
    .replace("Fundamentals", "Electoral history")
    .replace("persistent polling errors", "polling errors that persist")
    .replace(/ · loading ([\d.]+)/, " · shared national shift ×$1")
    .replace("National loading", "Shared national shift")
    .replace("As-of alignment; others unaligned", "Pre-election alliances; others unaligned")
    .replace("caucus with", "work with");
}
