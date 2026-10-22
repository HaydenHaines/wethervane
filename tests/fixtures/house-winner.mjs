/** Entirely invented scenarios; no probabilities or names are scientific outputs. */
const member = (id, label, kind = "named_person") => ({ id, label, kind });
export const modeledWinnerFixture = () => ({ fixtureOnly: true, sourceStatus: "invented_fixture", district: "AA-01", stage: "ultimate_outcome", resolution: "party_group", ruleFamily: "plurality", approximationNote: "Invented retrospective aggregate ultimate-party/group approximation; no ballot-transfer or vote-count law.", groups: [
  { id: "D", bloc: "D", members: [member("example-d", "Dana Example")] },
  { id: "R", bloc: "R", members: [member("example-r", "Robin Example")] },
  { id: "O", bloc: "O", members: [member("example-o", "Other reported write-ins", "pooled_or_residual")] },
], prediction: { status: "modeled", probabilities: [{ groupId: "D", probability: .6 }, { groupId: "R", probability: .35 }, { groupId: "O", probability: .05 }] }, truth: { status: "observed", groupId: "D", sourceBasis: "Invented official ultimate-outcome report", winnerNames: ["Example, Dana"] } });
export const samePartyWinnerFixture = () => {
  const fixture = modeledWinnerFixture();
  fixture.groups = [{ id: "D", bloc: "D", members: [member("example-d1", "Dana Example"), member("example-d2", "Devon Example")] }];
  fixture.prediction = { status: "structural", groupId: "D", reason: "This invented roster contains only Democratic nominees. It does not identify which person wins." };
  return fixture;
};
export const residualWinnerFixture = () => {
  const fixture = modeledWinnerFixture(); fixture.groups[2].members = [member("residual", "Scattered", "pooled_or_residual")]; return fixture;
};
export const unadmittedWinnerFixture = () => {
  const fixture = modeledWinnerFixture(); fixture.groups = [fixture.groups[2]];
  fixture.prediction = { status: "unavailable", reason: "The reported candidate list has not been checked for this example. Listing one group does not establish its chance of winning." };
  fixture.truth = { status: "observed", groupId: "O", sourceBasis: "Invented source outcome" }; return fixture;
};
export const rankedChoiceWinnerFixture = () => {
  const fixture = modeledWinnerFixture(); fixture.ruleFamily = "ranked_choice";
  fixture.groups[0].members.push(member("example-d2", "Devon Example"));
  fixture.truth = { status: "observed", groupId: "R", sourceBasis: "Invented official final ranked-choice round", winnerNames: ["Example, Robin"] }; return fixture;
};
export const uncertifiedWinnerFixture = () => ({ ...modeledWinnerFixture(), truth: { status: "unavailable", reason: "Invented annulled election: no certified ultimate outcome." } });
