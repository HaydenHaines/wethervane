/** Invented presentation fixtures only; no real people, sources or predictions. */
const hashes = { request: "a".repeat(64), context: "b".repeat(64), pair: "c".repeat(64), count: "d".repeat(64), F: "e".repeat(64), FP: "f".repeat(64) };
const endpoint = (place, side) => ({ personId: `${place}-${side}`, name: `${side === "first" ? "Alex" : "Blair"} ${place} Example`, sourceCandidateId: `source-${place}-${side}`, modelCandidateId: `model-${place}-${side}`, identitySha256: side === "first" ? "1".repeat(64) : "2".repeat(64) });
const pair = place => ({ id: `pair-${place}`, raceId: `invented-${place}-race`, contestId: `invented-${place}-contest`, eventId: `invented-${place}-event`, electionDate: "2024-11-05", first: endpoint(place, "first"), second: endpoint(place, "second") });
const binding = (p, arm) => {
  const omitName = endpoint => Object.fromEntries(Object.entries(endpoint).filter(([key]) => key !== "name"));
  return { comparisonContextSha256: hashes.context, pairSpecificationSha256: hashes.pair, sourceId: `invented-${arm}-source`, artifactSha256: hashes[arm], pairId: p.id, raceId: p.raceId, contestId: p.contestId, eventId: p.eventId, electionDate: p.electionDate, first: omitName(p.first), second: omitName(p.second), estimand: "candidate_marks", denominator: { kind: "named_pair_counts", firstPersonId: p.first.personId, secondPersonId: p.second.personId } };
};
const available = (p, arm, mean, observed) => ({ binding: binding(p, arm), distribution: { status: "available", undefinedMass: 0, summary: { mean, median: mean-.01, intervals: [{ level: .8, lower: .3, upper: .75 }] } }, evaluation: { status: "scored", score: { crps: .08, weightedIntervalScore: .12, absoluteMeanError: Math.abs(mean-observed) } }, precision: { status: "unavailable", reason: "No estimate of variation from running the calculation again is supplied." } });
const undefinedShare = (p, arm) => ({ binding: binding(p,arm), distribution: { status: "undefined_share", undefinedMass: .2, summary: null, reason: "Both named candidates receive zero votes in some invented outcomes." }, evaluation: { status: "unavailable", score: null, reason: "Some possible outcomes have no combined votes for this pair, so a share cannot be calculated." }, precision: { status: "unavailable", reason: "No estimate of calculation variation is supplied." } });
const unavailable = (p, arm) => ({ binding: binding(p,arm), distribution: { status: "unavailable", undefinedMass: null, summary: null, reason: "The invented result was not supplied." }, evaluation: { status: "unavailable", score: null, reason: "No score was supplied." }, precision: { status: "unavailable", reason: "No estimate of calculation variation was supplied." } });

export function partialPairComparisonFixture() {
  const harbor=pair("Harbor"), valley=pair("Valley"), ridge=pair("Ridge");
  return { kind: "candidate_pair_comparison_preparation", provenance: { kind: "invented_fixture", label: "Invented ordinary candidate-mark pair examples" }, label: "Named candidate-pair comparison",
    binding: { requestSha256: hashes.request, comparisonContextSha256: hashes.context, pairSpecificationSha256: hashes.pair, cycle: 2024, informationCutoff: "2024-10-22T00:00:00Z", horizonDays: 14, sources: Object.fromEntries(["count","F","FP"].map(arm=>[arm,{id:`invented-${arm}-source`,artifactSha256:hashes[arm]}])) },
    qualification: { experimental: true, posterior: "not_established", adoption: "not_qualified", informationSets: "different", explanation: "These invented models use different historical data and assumptions. Their errors cannot show that polling alone caused a change or approve any real forecast." },
    requiredPairs: [harbor,valley,ridge], rows: [
      { pairId: harbor.id, observedFirstPairShare: .52, arms: { count: available(harbor,"count",.6,.52), F: available(harbor,"F",.55,.52), FP: available(harbor,"FP",.58,.52) } },
      { pairId: valley.id, observedFirstPairShare: null, arms: { count: undefinedShare(valley,"count"), F: unavailable(valley,"F"), FP: unavailable(valley,"FP") } },
    ],
  };
}

export function missingObservationFixture() {
  const view=partialPairComparisonFixture();const row=view.rows[0];row.observedFirstPairShare=null;
  for(const arm of Object.values(row.arms)) arm.evaluation={status:"unavailable",score:null,reason:"Observed pair share is not supplied in this example."};
  return view;
}

export function incompatiblePairFixture() {
  const view=partialPairComparisonFixture();
  view.rows[0].arms.FP.binding.first.modelCandidateId="different-person";
  return view;
}
