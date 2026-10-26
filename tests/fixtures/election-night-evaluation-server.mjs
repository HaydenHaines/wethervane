import { ElectionNightArtifactStore, loadElectionNightEvaluationLock } from "../../lib/election-night-artifacts.ts";
import { HOUSE_PARTY_WINNER_INTERPRETATION_SHA256 } from "../../lib/election-night-evaluation.ts";
import { ELECTION_NIGHT_RUBRIC } from "../../lib/election-night-rubric.ts";
import { mkdirSync, readFileSync } from "node:fs";

// Synthetic software fixture only. Names, forecasts, review decisions, result
// records and source bytes are invented; this fixture admits no real result.
const H = char => char.repeat(64);
const unique = rows => [...new Map(rows.map(row => [row.sha256, row])).values()];

export function makeElectionNightEvaluationFixture(root, { chamber = "senate", count = 2, displayOnlyCount = 0 } = {}) {
  mkdirSync(root, { recursive: true });
  const store = new ElectionNightArtifactStore(root);
  const reviews = Object.fromEntries(["comparison", "cohort", "rubric", "resultPolicy", "projection"].map(role =>
    [role, store.putJson({ schema: "INVENTED-review", role })]));
  const resultPolicy = store.putJson({ schema: "INVENTED-result-policy" });
  const rubric = store.put(readFileSync(new URL("../../data/election-night-rubric-decision.json", import.meta.url)));
  if (rubric.sha256 !== ELECTION_NIGHT_RUBRIC.decisionSha256) throw Error("Synthetic rubric fixture does not match source policy.");
  const raceIds = Array.from({ length: count }, (_, i) => `INVENTED-${chamber.toUpperCase()}-race-${i + 1}`);
  const cohort = store.putJson({ schema: "INVENTED-fixed-cohort", raceIds });
  const modelSource = store.put(Buffer.from("INVENTED model original forecast bytes"));
  const baselineSource = store.put(Buffer.from("INVENTED baseline original forecast bytes"));
  const denominator = "INVENTED named-pair votes";
  if (!Number.isSafeInteger(displayOnlyCount) || displayOnlyCount < 0 || displayOnlyCount >= count)
    throw Error("Synthetic display-only count must leave a nonempty scoring cohort.");
  const scoredRaceIds = raceIds.slice(0, count - displayOnlyCount);
  const displayOnlyRaceIds = raceIds.slice(count - displayOnlyCount);
  const winnerDenominator = chamber === "house" ? "INVENTED House winner event" : denominator;
  const bind = (side, source) => ({ chamber, runId: `INVENTED-${side}-${chamber}`, resultSha256: source.sha256,
    nativeForecastSha256: chamber === "senate" ? H(side === "model" ? "a" : "b") : source.sha256,
    targetId: chamber === "house" ? null : "INVENTED-target", lawId: chamber === "house" ? null : "INVENTED-law",
    issuedAt: "2026-11-01T13:00:00Z", cutoffAt: "2026-11-01T12:00:00Z", horizonCalendarDays: 1 });
  const bindings = { model: bind("model", modelSource), baseline: bind("baseline", baselineSource) };
  const makeRace = (id, side) => {
    const firstId = `${id}:first`, secondId = `${id}:second`, otherId = `${id}:other`;
    const firstP = side === "model" ? .7 : .6;
    const outcomes = [{ id: firstId, name: "INVENTED First Candidate", party: "D", kind: "candidate", winProbability: firstP,
      members: [{ id: firstId, name: "INVENTED First Candidate", party: "D", sourceUrls: [] }] },
    { id: secondId, name: "INVENTED Second Candidate", party: "R", kind: "candidate", winProbability: 1 - firstP,
      members: [{ id: secondId, name: "INVENTED Second Candidate", party: "R", sourceUrls: [] }] }];
    const winnerOutcomes = outcomes;
    if (chamber === "house") {
      winnerOutcomes[0] = { id: `${id}:D`, name: "INVENTED Democratic party group", party: "D", kind: "group",
        winProbability: firstP, members: outcomes[0].members };
      winnerOutcomes[1] = { id: `${id}:R`, name: "INVENTED Republican party group", party: "R", kind: "group",
        winProbability: 1 - firstP, members: outcomes[1].members };
    }
    return { id, state: "XX", seatId: id, electionLabel: "INVENTED contest", ruleFamily: "plurality", method: "INVENTED method",
      notes: [], sourceUrls: [], outcomes: winnerOutcomes,
      pair: { firstCandidateId: firstId, secondCandidateId: secondId, firstCandidateName: "INVENTED First Candidate",
        secondCandidateName: "INVENTED Second Candidate", denominator, units: "percentage_points", orientation: "first_minus_second",
        meanMarginPp: side === "model" ? 2 : 0, medianMarginPp: 0,
        range90: side === "model" ? { lower: -5, upper: 9 } : { lower: -10, upper: 10 }, firstExceedsHalf: firstP,
        winnerRelationship: "same_event", closeWithin2Pp: null, firstAtLeast10Pp: null, secondAtLeast10Pp: null, qualifications: [] },
      modalPick: chamber === "house" ? { outcomeId: `${id}:D`, name: "INVENTED Democratic party group", kind: "group",
        probability: firstP, tiedOutcomeIds: [`${id}:D`] } :
        { outcomeId: firstId, name: "INVENTED First Candidate", kind: "candidate", probability: firstP, tiedOutcomeIds: [firstId] } };
  };
  const forecasts = Object.fromEntries(["model", "baseline"].map(side => [side, {
    schema: "ElectionNightForecastV1", chamber, editionStatus: "dated_snapshot",
    run: { id: bindings[side].runId, label: `INVENTED ${side}`, year: 2026, configuration: {}, artifactSha256: bindings[side].resultSha256,
      issuedAt: bindings[side].issuedAt, cutoffAt: bindings[side].cutoffAt, strictAsOf: false, sourceCutoffs: [],
      targetId: bindings[side].targetId, lawId: bindings[side].lawId },
    qualification: [], sourceUrls: [], sourceHashes: chamber === "senate" ? { forecastSha256: bindings[side].nativeForecastSha256 } : {},
    sourceNotes: [], inputNotices: [], races: raceIds.map(id => makeRace(id, side)), chamberScenario: null,
    summary: { raceCount: count, pairRaceCount: count, expectedModalMisses: count * (side === "model" ? .3 : .4),
      topTieRaceCount: 0, tiePolicy: "lexicographic_outcome_id", interpretation: "INVENTED synthetic forecast" } }]));
  const winnerTargets = scoredRaceIds.map(id => ({ raceId: id, eventId: "INVENTED-election", roundId: "INVENTED-final-round",
    resultRuleId: "INVENTED-final-rule", denominator: winnerDenominator, estimand: chamber === "house" ? "house_official_party_winner" : "senate_principal_pair_lead",
    outcomeIds: chamber === "house" ? ["D", "R", "O"] : [`${id}:first`, `${id}:second`],
    firstCandidateId: chamber === "house" ? null : `${id}:first`, secondCandidateId: chamber === "house" ? null : `${id}:second` }));
  const intervalTargets = scoredRaceIds.map(id => ({ raceId: id, eventId: "INVENTED-election", roundId: "INVENTED-final-round",
    resultRuleId: "INVENTED-final-rule", denominator, estimand: "named_pair_margin", firstCandidateId: `${id}:first`,
    secondCandidateId: `${id}:second`, units: "percentage_points", orientation: "first_minus_second", intervalLevel: .9 }));
  const contract = { schema: "ElectionNightRubricComparisonV1", id: "INVENTED-comparison", sha256: H("c"), rubricDecisionSha256: rubric.sha256,
    comparisonMode: "matched_horizon", label: "INVENTED matched forecast editions", model: bindings.model, baseline: bindings.baseline,
    comparisonReview: { id: "INVENTED-comparison-review", sha256: reviews.comparison.sha256 },
    cohortDefinition: { id: "INVENTED-cohort", sha256: cohort.sha256 }, resultPolicy: { id: "INVENTED-result-policy", sha256: resultPolicy.sha256 },
    earlierBenchmarkPins: null, houseMappingReviewSha256: chamber === "house" ? HOUSE_PARTY_WINNER_INTERPRETATION_SHA256 : null,
    winnerCohort: winnerTargets, intervalCohort: intervalTargets, intervalExclusions: [],
    ...(displayOnlyCount ? { displayRaceIds: raceIds, ungradedDisplayRaceIds: displayOnlyRaceIds } : {}) };
  const protocol = side => ({ schema: "ElectionNightEvaluationProtocolV1", id: `INVENTED-${side}-protocol`,
    forecast: { chamber, runId: bindings[side].runId, artifactSha256: bindings[side].resultSha256,
      issuedAt: bindings[side].issuedAt, cutoffAt: bindings[side].cutoffAt },
    reviewedDefinition: { sha256: reviews.resultPolicy.sha256, url: "https://example.invalid/INVENTED-definition" },
    races: raceIds.map(id => ({ raceId: id,
      winner: { estimand: chamber === "house" ? "house_official_party_winner" : "conditional_principal_pair_winner",
        eventId: "INVENTED-election", roundId: "INVENTED-final-round", ruleFamily: "plurality",
        resultRuleId: "INVENTED-final-rule", outcomeIds: chamber === "house" ? ["D", "R", "O"] : [`${id}:first`, `${id}:second`],
        minimumStatus: "certified", ...(chamber === "house" ? { denominator: "INVENTED House winner event",
          interpretationSha256: HOUSE_PARTY_WINNER_INTERPRETATION_SHA256 } : {}) },
      margin: { estimand: "named_pair_margin", eventId: "INVENTED-election", roundId: "INVENTED-final-round",
        ruleFamily: "plurality", resultRuleId: "INVENTED-final-rule", firstCandidateId: `${id}:first`, secondCandidateId: `${id}:second`,
        denominator, units: "percentage_points", orientation: "first_minus_second", pointRule: "mean", minimumStatus: "certified" } })) });
  const artifacts = { modelSource, baselineSource, modelForecast: store.putJson(forecasts.model), baselineForecast: store.putJson(forecasts.baseline),
    modelProtocol: store.putJson(protocol("model")), baselineProtocol: store.putJson(protocol("baseline")),
    comparisonContract: store.putJson(contract), cohort, rubric, resultPolicy };
  const manifest = { schema: "ElectionNightEvaluationLockV1", id: "INVENTED-lock", frozenAt: "2026-11-01T14:00:00Z",
    forecastBindings: bindings, artifacts, reviews,
    projections: { model: { sourceSha256: modelSource.sha256, projectionSha256: artifacts.modelForecast.sha256, reviewSha256: reviews.projection.sha256 },
      baseline: { sourceSha256: baselineSource.sha256, projectionSha256: artifacts.baselineForecast.sha256, reviewSha256: reviews.projection.sha256 } } };
  const lockPin = store.putJson(manifest);
  function authority(subject, scope, lockSha256, reviewPins, extraPins) {
    const admission = store.putJson({ schema: "ElectionNightArtifactAdmissionV1", id: `INVENTED-${scope}-admission`, scope,
      subject, evaluationLockSha256: lockSha256, reviews: unique(reviewPins) });
    const publication = store.putJson({ schema: "ElectionNightArtifactPublicationV1", id: `INVENTED-${scope}-publication`,
      artifacts: unique([subject, admission, ...reviewPins, ...extraPins]) });
    return { publication, admission };
  }
  const lockAuth = authority(lockPin, "evaluation_lock", lockPin.sha256, Object.values(reviews), [...Object.values(artifacts), ...Object.values(reviews)]);
  const lockRequest = { lock: lockPin, trustedPublication: lockAuth.publication, trustedAdmission: lockAuth.admission };
  const records = [];
  function addRecord({ index = 0, revision = "1", winnerStatus = "certified", marginStatus = "certified", marginPp = -7 } = {}) {
    const raceId = raceIds[index], source = store.put(Buffer.from(`INVENTED result source bytes ${index} revision ${revision}`));
    const evidence = store.putJson({ schema: "INVENTED-evidence", raceId, revision });
    const outcomeId = `${raceId}:first`;
    const record = { schema: "ElectionNightOutcomeRecordV1", id: `INVENTED-result-${raceId}`, revision, raceId,
      source: { sha256: source.sha256, url: "https://example.invalid/INVENTED-official-result",
        publishedAt: "2026-11-04T12:00:00Z", retrievedAt: "2026-11-05T12:00:00Z" },
      winner: { status: winnerStatus, estimand: chamber === "house" ? "house_official_party_winner" : "conditional_principal_pair_winner",
        eventId: "INVENTED-election", roundId: "INVENTED-final-round", ruleFamily: "plurality", resultRuleId: "INVENTED-final-rule",
        outcomeId: chamber === "house" ? "D" : outcomeId, kind: chamber === "house" ? "group" : "candidate", candidateId: outcomeId,
        reason: null, ...(chamber === "house" ? { denominator: "INVENTED House winner event" } : {}) },
      margin: { status: marginStatus, estimand: "named_pair_margin", eventId: "INVENTED-election", roundId: "INVENTED-final-round",
        ruleFamily: "plurality", resultRuleId: "INVENTED-final-rule", firstCandidateId: `${raceId}:first`, secondCandidateId: `${raceId}:second`,
        denominator, units: "percentage_points", orientation: "first_minus_second", marginPp, reason: null } };
    const recordPin = store.putJson(record), revisionManifest = { schema: "ElectionNightResultRevisionV1", id: `INVENTED-revision-${raceId}-${revision}`,
      ledgerId: "INVENTED-ledger", evaluationLockSha256: lockPin.sha256,
      predecessorSha256: records.at(-1)?.revisionPin.sha256 ?? null,
      supersedesRecordSha256: records.find(row => row.raceId === raceId)?.recordPin.sha256 ?? null,
      record: recordPin, source, evidence: [evidence] };
    const revisionPin = store.putJson(revisionManifest);
    records.push({ raceId, recordPin, revisionPin, source, evidence, record });
    return records.at(-1);
  }
  function makeLedger() {
    const ledgerPin = store.putJson({ schema: "ElectionNightResultLedgerV1", id: "INVENTED-ledger",
      evaluationLockSha256: lockPin.sha256, chosenHeadSha256: records.at(-1)?.revisionPin.sha256 ?? null,
      revisions: records.map(row => row.revisionPin) });
    const ledgerAuth = authority(ledgerPin, "result_ledger", lockPin.sha256, [reviews.resultPolicy],
      records.flatMap(row => [row.revisionPin, row.recordPin, row.source, row.evidence]));
    return { ledger: ledgerPin, trustedPublication: ledgerAuth.publication, trustedAdmission: ledgerAuth.admission };
  }
  return { root, store, chamber, count, lockPin, lockRequest, records, addRecord, makeLedger };
}
