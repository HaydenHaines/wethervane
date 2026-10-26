import assert from "node:assert/strict";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createHash } from "node:crypto";
import test from "node:test";
import { ElectionNightArtifactStore, loadElectionNightEvaluationLock, loadElectionNightResultLedger } from "../lib/election-night-artifacts.ts";
import { assertEarlierSenateCertifiedMinimums, buildElectionNightScoring } from "../lib/election-night-scoring.ts";
import { HOUSE_PARTY_WINNER_INTERPRETATION_SHA256 } from "../lib/election-night-evaluation.ts";
import { toHouseElectionNightForecast, toSenateElectionNightForecast } from "../lib/election-night-forecast.ts";
import { houseLiveView } from "../lib/house-live-view.ts";
import { parseHouseLiveResult } from "../lib/house-live-result.ts";
import { ELECTION_NIGHT_RUBRIC, gradeElectionNightRubric, validElectionNightRubricBinding } from "../lib/election-night-rubric.ts";

// Apart from the explicitly named public-forecast metadata check, fixtures use
// invented names, observations, reviews and source bytes. No real 2026 result
// is admitted or graded.
const H = char => char.repeat(64);
const unique = pins => [...new Map(pins.map(pin => [pin.sha256, pin])).values()];
const near = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-12, `${actual} vs ${expected}`);
const ownerMappingPath = "/home/hayden/projects/wethervane-redesign-20261006/integration/research/senate-polling-refresh-20261026/scorecard-baseline-compatibility-sol/candidate-pair-event-mapping-v1.json";
function setup(t, chamber = "senate", count = 2, { currentHouse = false, gradedCount = count,
  excludeLastInterval = false, winnerMinimumStatus = "final", marginMinimumStatus = "final", mutateModelProtocol = null,
  houseNullIdentities = false } = {}) {
  const root = mkdtempSync(join(tmpdir(), "election-night-scoring-"));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const store = new ElectionNightArtifactStore(root);
  const reviews = Object.fromEntries(["comparison", "cohort", "rubric", "resultPolicy", "projection"].map(role =>
    [role, store.putJson({ schema: "INVENTED-review", role })]));
  const resultPolicy = store.putJson({ schema: "INVENTED-result-policy" });
  const rubric = store.put(readFileSync(new URL("../data/election-night-rubric-decision.json", import.meta.url)));
  assert.equal(rubric.sha256, ELECTION_NIGHT_RUBRIC.decisionSha256);
  const raceIds = Array.from({ length: count }, (_, index) => `INVENTED-race-${index + 1}`);
  const cohort = store.putJson({ schema: "INVENTED-fixed-cohort", raceIds });
  const modelSource = store.put(Buffer.from("INVENTED model original bytes"));
  const baselineSource = store.put(Buffer.from("INVENTED baseline original bytes"));
  const binding = (side, pin) => ({ chamber, runId: currentHouse ? `house-current-${side === "model" ? "polling" : "fundamentals"}` : `INVENTED-${side}`, resultSha256: pin.sha256,
    nativeForecastSha256: chamber === "senate" ? H(side === "model" ? "a" : "b") : pin.sha256,
    targetId: houseNullIdentities ? null : "INVENTED-target", lawId: houseNullIdentities ? null : "INVENTED-law", issuedAt: "2026-11-01T13:00:00Z",
    cutoffAt: "2026-11-01T12:00:00Z", horizonCalendarDays: 2 });
  const bindings = { model: binding("model", modelSource), baseline: binding("baseline", baselineSource) };
  const denominator = "INVENTED named-pair votes";
  const winnerDenominator = chamber === "house" ? "INVENTED official party winner" : denominator;
  const makeRace = (id, side) => {
    const firstP = side === "model" ? 0.7 : 0.6;
    const outcomes = ["D", "R"].map((party, index) => ({ id: chamber === "house" ? `${id}:${party}` : `${id}:${index ? "second" : "first"}`,
      name: `INVENTED ${party}`, party, kind: "candidate", winProbability: index ? 1 - firstP : firstP,
      members: [{ id: `${id}:${index ? "second" : "first"}`, name: `INVENTED ${party}`, party, sourceUrls: [] }] }));
    const firstId = outcomes[0].members[0].id, secondId = outcomes[1].members[0].id;
    return { id, state: "XX", seatId: id, electionLabel: "INVENTED contest", ruleFamily: "plurality", method: "INVENTED",
      notes: [], sourceUrls: [], outcomes,
      pair: { firstCandidateId: firstId, secondCandidateId: secondId, firstCandidateName: "INVENTED D", secondCandidateName: "INVENTED R",
        denominator, units: "percentage_points", orientation: "first_minus_second", meanMarginPp: side === "model" ? 2 : 0,
        medianMarginPp: 0, range90: side === "model" ? { lower: -5, upper: 5 } : { lower: -10, upper: 10 },
        firstExceedsHalf: firstP, winnerRelationship: chamber === "house" ? "separate_conditional_event" : "same_event",
        closeWithin2Pp: null, firstAtLeast10Pp: null, secondAtLeast10Pp: null, qualifications: [] },
      modalPick: { outcomeId: outcomes[0].id, name: outcomes[0].name, kind: "candidate", probability: firstP, tiedOutcomeIds: [outcomes[0].id] } };
  };
  const makeForecast = side => ({ schema: "ElectionNightForecastV1", chamber, editionStatus: "dated_snapshot",
    run: { id: bindings[side].runId, label: `INVENTED ${side}`, year: 2026, configuration: {},
      artifactSha256: bindings[side].resultSha256, issuedAt: bindings[side].issuedAt, cutoffAt: bindings[side].cutoffAt,
      strictAsOf: false, sourceCutoffs: [], targetId: bindings[side].targetId, lawId: bindings[side].lawId },
    qualification: [], sourceUrls: [], sourceHashes: chamber === "senate" ? { forecastSha256: bindings[side].nativeForecastSha256 } : {},
    sourceNotes: [], inputNotices: [], races: raceIds.map(id => makeRace(id, side)), chamberScenario: null,
    summary: { raceCount: count, pairRaceCount: count, expectedModalMisses: count * (side === "model" ? 0.3 : 0.4),
      topTieRaceCount: 0, tiePolicy: "lexicographic_outcome_id", interpretation: "INVENTED" } });
  const forecasts = { model: makeForecast("model"), baseline: makeForecast("baseline") };
  const allWinnerTargets = raceIds.map(id => ({ raceId: id, eventId: "INVENTED-event", roundId: "INVENTED-final-round",
    resultRuleId: "INVENTED-final-rule", denominator: winnerDenominator,
    estimand: chamber === "house" ? "house_official_party_winner" : "senate_principal_pair_lead",
    outcomeIds: chamber === "house" ? ["D", "R", "O"] : [`${id}:first`, `${id}:second`],
    firstCandidateId: chamber === "house" ? null : `${id}:first`, secondCandidateId: chamber === "house" ? null : `${id}:second` }));
  const allIntervalTargets = raceIds.map(id => ({ raceId: id, eventId: "INVENTED-event", roundId: "INVENTED-final-round",
    resultRuleId: "INVENTED-final-rule", denominator, estimand: "named_pair_margin",
    firstCandidateId: `${id}:first`, secondCandidateId: `${id}:second`, units: "percentage_points",
    orientation: "first_minus_second", intervalLevel: 0.9 }));
  const contract = { schema: "ElectionNightRubricComparisonV1", id: "INVENTED-comparison", sha256: H("c"),
    rubricDecisionSha256: rubric.sha256, comparisonMode: "matched_horizon", label: "INVENTED matched edition",
    model: bindings.model, baseline: bindings.baseline,
    comparisonReview: { id: "INVENTED-comparison-review", sha256: reviews.comparison.sha256 },
    cohortDefinition: { id: "INVENTED-cohort", sha256: cohort.sha256 },
    resultPolicy: { id: "INVENTED-result-policy", sha256: resultPolicy.sha256 },
    earlierBenchmarkPins: null, houseMappingReviewSha256: chamber === "house" ? HOUSE_PARTY_WINNER_INTERPRETATION_SHA256 : null,
    winnerCohort: allWinnerTargets.slice(0, gradedCount),
    intervalCohort: allIntervalTargets.slice(0, gradedCount - Number(excludeLastInterval)),
    intervalExclusions: excludeLastInterval ? [{ raceId: raceIds[gradedCount - 1], reason: "INVENTED predeclared pair unavailability" }] : [],
    ...(gradedCount < count ? { displayRaceIds: raceIds, ungradedDisplayRaceIds: raceIds.slice(gradedCount) } : {}) };
  const makeProtocol = side => ({ schema: "ElectionNightEvaluationProtocolV1", id: `INVENTED-${side}-protocol`,
    forecast: { chamber, runId: bindings[side].runId, artifactSha256: bindings[side].resultSha256,
      issuedAt: bindings[side].issuedAt, cutoffAt: bindings[side].cutoffAt },
    reviewedDefinition: { sha256: reviews.resultPolicy.sha256, url: "https://example.invalid/INVENTED-review" },
    races: raceIds.map(id => ({ raceId: id,
      winner: { estimand: chamber === "house" ? "house_official_party_winner" : "conditional_principal_pair_winner",
        eventId: "INVENTED-event", roundId: "INVENTED-final-round", ruleFamily: "plurality", resultRuleId: "INVENTED-final-rule",
        outcomeIds: chamber === "house" ? ["D", "R", "O"] : [`${id}:first`, `${id}:second`], minimumStatus: winnerMinimumStatus,
        ...(chamber === "house" ? { denominator: winnerDenominator,
          interpretationSha256: HOUSE_PARTY_WINNER_INTERPRETATION_SHA256 } : {}) },
      margin: { estimand: "named_pair_margin", eventId: "INVENTED-event", roundId: "INVENTED-final-round",
        ruleFamily: "plurality", resultRuleId: "INVENTED-final-rule", firstCandidateId: `${id}:first`, secondCandidateId: `${id}:second`, denominator,
        units: "percentage_points", orientation: "first_minus_second", pointRule: "mean", minimumStatus: marginMinimumStatus } })) });
  const modelProtocol = makeProtocol("model"); mutateModelProtocol?.(modelProtocol);
  const artifacts = { modelSource, baselineSource, modelForecast: store.putJson(forecasts.model),
    baselineForecast: store.putJson(forecasts.baseline), modelProtocol: store.putJson(modelProtocol),
    baselineProtocol: store.putJson(makeProtocol("baseline")), comparisonContract: store.putJson(contract),
    cohort, rubric, resultPolicy };
  const lockManifest = { schema: "ElectionNightEvaluationLockV1", id: "INVENTED-lock", frozenAt: "2026-11-01T14:00:00Z",
    forecastBindings: bindings, artifacts, reviews,
    projections: { model: { sourceSha256: modelSource.sha256, projectionSha256: artifacts.modelForecast.sha256,
      reviewSha256: reviews.projection.sha256 }, baseline: { sourceSha256: baselineSource.sha256,
      projectionSha256: artifacts.baselineForecast.sha256, reviewSha256: reviews.projection.sha256 } } };
  const lockPin = store.putJson(lockManifest);
  const authority = (subject, scope, lockSha256, reviewPins, extraPins) => {
    const admission = store.putJson({ schema: "ElectionNightArtifactAdmissionV1", id: `INVENTED-${scope}-admission`,
      scope, subject, evaluationLockSha256: lockSha256, reviews: unique(reviewPins) });
    const publication = store.putJson({ schema: "ElectionNightArtifactPublicationV1", id: `INVENTED-${scope}-publication`,
      artifacts: unique([subject, admission, ...reviewPins, ...extraPins]) });
    return { trustedPublication: publication, trustedAdmission: admission };
  };
  const lockAuthority = authority(lockPin, "evaluation_lock", lockPin.sha256, Object.values(reviews),
    [...Object.values(artifacts), ...Object.values(reviews)]);
  const loadedLock = loadElectionNightEvaluationLock(store, { lock: lockPin, ...lockAuthority });
  function revision({ raceId = raceIds[0], id = "INVENTED-revision-1", recordRevision = "1", predecessor = null,
    supersedes = null, marginPp = -7, winnerGroup = "D", winningCandidateId = `${raceId}:first`,
    winnerStatus = "final", marginStatus = "final", sourceText = "INVENTED official source bytes" } = {}) {
    const source = store.put(Buffer.from(sourceText));
    const evidence = store.putJson({ schema: "INVENTED-result-evidence", id });
    const record = { schema: "ElectionNightOutcomeRecordV1", id: `INVENTED-result-${raceId}`, revision: recordRevision, raceId,
      source: { sha256: source.sha256, url: "https://example.invalid/INVENTED-result",
        publishedAt: "2026-11-04T12:00:00Z", retrievedAt: "2026-11-05T12:00:00Z" },
      winner: { status: winnerStatus, estimand: chamber === "house" ? "house_official_party_winner" : "conditional_principal_pair_winner",
        eventId: "INVENTED-event", roundId: "INVENTED-final-round", ruleFamily: "plurality", resultRuleId: "INVENTED-final-rule",
        outcomeId: chamber === "house" ? winnerGroup : winningCandidateId,
        kind: chamber === "house" ? "group" : "candidate", candidateId: winningCandidateId, reason: null,
        ...(chamber === "house" ? { denominator: winnerDenominator } : {}) },
      margin: { status: marginStatus, estimand: "named_pair_margin", eventId: "INVENTED-event",
        roundId: "INVENTED-final-round", ruleFamily: "plurality", resultRuleId: "INVENTED-final-rule", firstCandidateId: `${raceId}:first`,
        secondCandidateId: `${raceId}:second`, denominator, units: "percentage_points",
        orientation: "first_minus_second", marginPp, reason: null } };
    const recordPin = store.putJson(record);
    const manifest = { schema: "ElectionNightResultRevisionV1", id, ledgerId: "INVENTED-ledger",
      evaluationLockSha256: loadedLock.lockSha256, predecessorSha256: predecessor,
      supersedesRecordSha256: supersedes, record: recordPin, source, evidence: [evidence] };
    return { pin: store.putJson(manifest), recordPin, source, evidence, record };
  }
  function ledger(revisions) {
    const ledgerPin = store.putJson({ schema: "ElectionNightResultLedgerV1", id: "INVENTED-ledger",
      evaluationLockSha256: loadedLock.lockSha256, chosenHeadSha256: revisions.at(-1)?.pin.sha256 ?? null,
      revisions: revisions.map(row => row.pin) });
    const auth = authority(ledgerPin, "result_ledger", loadedLock.lockSha256, [reviews.resultPolicy],
      revisions.flatMap(row => [row.pin, row.recordPin, row.source, row.evidence]));
    return loadElectionNightResultLedger(store, { evaluationLock: loadedLock, ledger: ledgerPin, ...auth });
  }
  return { store, loadedLock, contract, forecasts, raceIds, revision, ledger, lockPin };
}

test("paired Senate losses derive from both issued projections and the exact same admitted record bytes", t => {
  const f = setup(t), first = f.revision(), ledger = f.ledger([first]);
  const scored = buildElectionNightScoring(f.loadedLock, ledger), input = scored.rubricInput;
  assert.equal(scored.lockSha256, f.loadedLock.lockSha256);
  assert.equal(scored.chosenHeadSha256, ledger.chosenHeadSha256);
  assert.equal(scored.modelEvaluation.rows.length, 2); assert.equal(scored.baselineEvaluation.rows.length, 2);
  assert.deepEqual(scored.displayOnlyRaceIds, []);
  assert.equal(input.winnerRows.length, 2); assert.equal(input.intervalRows.length, 2);
  const winner = input.winnerRows[0], interval = input.intervalRows[0];
  near(winner.model.loss, 0.18); near(winner.baseline.loss, 0.32);
  near(interval.model.loss, 50); near(interval.baseline.loss, 20);
  assert.equal(winner.model.observedValue, `${f.raceIds[0]}:first`);
  assert.equal(interval.model.observedValue, -7);
  for (const pair of [winner, interval]) {
    assert.deepEqual(pair.model.result, pair.baseline.result);
    assert.equal(pair.model.result.recordSha256, first.recordPin.sha256);
    assert.equal(pair.model.result.sourceSha256, first.source.sha256);
    assert.equal(pair.model.result.resultPolicySha256, f.contract.resultPolicy.sha256);
  }
  assert.equal(input.winnerRows[1].model.status, "unresolved");
  assert.equal(input.winnerRows[1].model.loss, null);
  assert.equal(input.intervalRows[1].model.status, "unresolved");
});

test("authenticated lock through ledger and evaluator earns only an invented complete-cohort analytic grade", t => {
  const f = setup(t, "senate", 1), head = f.revision({ marginPp: 0 });
  const bridge = buildElectionNightScoring(f.loadedLock, f.ledger([head]));
  const grade = gradeElectionNightRubric(bridge.rubricInput);
  near(grade.winner.raw.modelMeanLoss, 0.09);
  near(grade.winner.raw.baselineMeanLoss, 0.16);
  near(grade.interval.raw.modelMeanLoss, 10);
  near(grade.interval.raw.baselineMeanLoss, 20);
  assert.equal(grade.winner.coverage.final, 1);
  assert.equal(grade.interval.coverage.final, 1);
  assert.equal(grade.winner.grade, "A");
  assert.equal(grade.interval.grade, "A");
  assert.equal(grade.overall.grade, "A");
  assert.equal(grade.status, "graded");
  assert.deepEqual(grade.issues, []);
  assert.equal(bridge.rubricInput.winnerRows[0].model.result.recordSha256, head.recordPin.sha256);
});

test("owner-bound earlier Senate protocols require certified targets across all 35 displayed rows", t => {
  const f = setup(t, "senate", 35, { gradedCount: 32, winnerMinimumStatus: "certified", marginMinimumStatus: "certified" });
  const contract = structuredClone(f.contract);
  contract.comparisonMode = "explicit_earlier_benchmark";
  contract.resultPolicy.sha256 = ELECTION_NIGHT_RUBRIC.earlierBenchmark.resultPolicySha256;
  const model = structuredClone(f.loadedLock.modelProtocol), baseline = structuredClone(f.loadedLock.baselineProtocol);
  const check = () => assertEarlierSenateCertifiedMinimums(contract, f.forecasts.model, f.forecasts.baseline, model, baseline);
  assert.doesNotThrow(check);
  model.races[34].winner.minimumStatus = "final"; // Displayed conditional race outside the 32-row primary grade.
  assert.throws(check, /all 35 displayed races/);
  model.races[34].winner.minimumStatus = "certified";
  baseline.races[0].margin.minimumStatus = "final";
  assert.throws(check, /all 35 displayed races/);
  baseline.races[0].margin.minimumStatus = "certified";
  baseline.races[34].margin = null;
  assert.throws(check, /all 35 displayed races/);
  baseline.races[34].margin = structuredClone(f.loadedLock.baselineProtocol.races[34].margin);
  assert.doesNotThrow(check);
  const revisions = [];
  for (const [index, raceId] of f.raceIds.entries()) revisions.push(f.revision({ raceId,
    id: `INVENTED-final-revision-${index}`, predecessor: revisions.at(-1)?.pin.sha256 ?? null,
    winnerStatus: "final", marginStatus: "final" }));
  const bridge = buildElectionNightScoring(f.loadedLock, f.ledger(revisions));
  assert.deepEqual(bridge.displayOnlyRaceIds, f.raceIds.slice(32));
  assert.equal(bridge.modelEvaluation.rows.length, 35);
  assert.equal(bridge.rubricInput.winnerRows.length, 32);
  assert.ok(bridge.rubricInput.winnerRows.every(row => row.model.status === "unresolved" && row.baseline.status === "unresolved"));
  assert.ok(bridge.rubricInput.intervalRows.every(row => row.model.loss === null && row.baseline.loss === null));
  assert.equal(gradeElectionNightRubric(bridge.rubricInput).status, "not_graded");
});

test("final-only invented records cannot grade when both frozen target minimums require certification", t => {
  const f = setup(t, "senate", 1, { winnerMinimumStatus: "certified", marginMinimumStatus: "certified" });
  const head = f.revision({ winnerStatus: "final", marginStatus: "final" });
  const bridge = buildElectionNightScoring(f.loadedLock, f.ledger([head]));
  for (const row of [bridge.rubricInput.winnerRows[0], bridge.rubricInput.intervalRows[0]]) {
    assert.equal(row.model.status, "unresolved"); assert.equal(row.baseline.status, "unresolved");
    assert.equal(row.model.loss, null); assert.equal(row.baseline.loss, null);
    assert.equal(row.model.result.finality, "final");
  }
  const grade = gradeElectionNightRubric(bridge.rubricInput);
  assert.equal(grade.status, "not_graded");
  assert.equal(grade.overall.grade, null);
});

test("an authenticated empty ledger keeps every locked row pending with no score or grade", t => {
  const f = setup(t, "senate", 2), bridge = buildElectionNightScoring(f.loadedLock, f.ledger([]));
  assert.equal(bridge.chosenHeadSha256, null);
  assert.equal(bridge.modelEvaluation.rows.length, 2);
  assert.equal(bridge.baselineEvaluation.rows.length, 2);
  for (const row of [...bridge.rubricInput.winnerRows, ...bridge.rubricInput.intervalRows]) {
    assert.equal(row.model.status, "unresolved"); assert.equal(row.baseline.status, "unresolved");
    assert.equal(row.model.result, null); assert.equal(row.baseline.result, null);
    assert.equal(row.model.loss, null); assert.equal(row.baseline.loss, null);
  }
  const grade = gradeElectionNightRubric(bridge.rubricInput);
  assert.equal(grade.status, "not_graded");
  assert.equal(grade.overall.grade, null);
});

test("new immutable result head changes both arms together and leaves the prior score detached", t => {
  const f = setup(t), first = f.revision(), original = buildElectionNightScoring(f.loadedLock, f.ledger([first]));
  const corrected = f.revision({ id: "INVENTED-correction-2", recordRevision: "2", predecessor: first.pin.sha256,
    supersedes: first.recordPin.sha256, marginPp: 9, sourceText: "INVENTED corrected source bytes" });
  const latest = buildElectionNightScoring(f.loadedLock, f.ledger([first, corrected]));
  assert.equal(original.rubricInput.intervalRows[0].model.observedValue, -7);
  assert.equal(latest.rubricInput.intervalRows[0].model.observedValue, 9);
  assert.equal(latest.rubricInput.intervalRows[0].model.result.revision, "2");
  assert.equal(latest.rubricInput.winnerRows[0].baseline.result.recordSha256, corrected.recordPin.sha256);
  assert.notEqual(original.rubricInput.intervalRows[0].model.result.recordSha256, corrected.recordPin.sha256);
  assert.notEqual(original.chosenHeadSha256, latest.chosenHeadSha256);
});

test("called or unavailable results remain visible with null losses until their declared target is scoreable", t => {
  const f = setup(t), called = f.revision({ winnerStatus: "called", marginStatus: "called" });
  const partial = buildElectionNightScoring(f.loadedLock, f.ledger([called]));
  assert.equal(partial.rubricInput.winnerRows[0].model.status, "unresolved");
  assert.equal(partial.rubricInput.intervalRows[0].model.status, "unresolved");
  assert.equal(partial.rubricInput.intervalRows[0].model.loss, null);
  assert.equal(partial.rubricInput.winnerRows[0].model.result.finality, "called");
  assert.equal(partial.rubricInput.intervalRows[0].model.result.finality, "called");
  assert.equal(partial.modelEvaluation.rows[1].winner.status, "unresolved");
  const provisionalFixture = setup(t, "senate", 1, { winnerMinimumStatus: "called" });
  const provisional = buildElectionNightScoring(provisionalFixture.loadedLock,
    provisionalFixture.ledger([provisionalFixture.revision({ winnerStatus: "called", marginStatus: "called" })]));
  assert.equal(provisional.rubricInput.winnerRows[0].model.status, "scored");
  assert.equal(provisional.rubricInput.winnerRows[0].model.result.finality, "called");
  assert.equal(provisional.rubricInput.intervalRows[0].model.status, "unresolved");
  const tied = setup(t, "senate", 1);
  const tie = buildElectionNightScoring(tied.loadedLock,
    tied.ledger([tied.revision({ winnerStatus: "unsupported", marginStatus: "final", marginPp: 0 })]));
  assert.equal(tie.rubricInput.winnerRows[0].model.status, "unsupported");
  assert.equal(tie.rubricInput.winnerRows[0].model.loss, null);
  assert.equal(tie.rubricInput.intervalRows[0].model.status, "scored");
  assert.equal(tie.rubricInput.intervalRows[0].model.observedValue, 0);
});

test("explicit graded subset retains every forecast and protocol row for display", t => {
  const f = setup(t, "senate", 2, { gradedCount: 1 });
  const first = f.revision(), second = f.revision({ raceId: f.raceIds[1], id: "INVENTED-revision-2",
    predecessor: first.pin.sha256, sourceText: "INVENTED second race result source" });
  const scored = buildElectionNightScoring(f.loadedLock, f.ledger([first, second]));
  assert.deepEqual(scored.displayOnlyRaceIds, [f.raceIds[1]]);
  assert.equal(scored.modelEvaluation.rows.length, 2);
  assert.equal(scored.modelEvaluation.rows[1].winner.status, "scored");
  assert.equal(scored.rubricInput.winnerRows.length, 1);
  assert.equal(scored.rubricInput.intervalRows.length, 1);
  assert.deepEqual(scored.rubricInput.contract.ungradedDisplayRaceIds, [f.raceIds[1]]);
});

test("a predeclared pair exclusion removes only its grade row, while all evaluator margins remain visible", t => {
  const f = setup(t, "senate", 2, { excludeLastInterval: true });
  const scored = buildElectionNightScoring(f.loadedLock, f.ledger([f.revision()]));
  assert.equal(scored.rubricInput.winnerRows.length, 2);
  assert.equal(scored.rubricInput.intervalRows.length, 1);
  assert.deepEqual(scored.rubricInput.contract.intervalExclusions,
    [{ raceId: f.raceIds[1], reason: "INVENTED predeclared pair unavailability" }]);
  assert.equal(scored.modelEvaluation.rows.length, 2);
  assert.equal(scored.modelEvaluation.rows[1].margin.status, "unresolved");
});

test("an admitted but scientifically mismapped protocol is rejected before any paired loss is built", t => {
  const f = setup(t, "senate", 1, { mutateModelProtocol: protocol => {
    protocol.races[0].winner.eventId = "INVENTED-different-event";
  } });
  assert.throws(() => buildElectionNightScoring(f.loadedLock, f.ledger([f.revision()])), /Winner target mapping/);
  const changedRule = setup(t, "senate", 1, { mutateModelProtocol: protocol => {
    protocol.races[0].winner.resultRuleId = "INVENTED-different-counting-rule";
  } });
  assert.throws(() => buildElectionNightScoring(changedRule.loadedLock,
    changedRule.ledger([changedRule.revision()])), /Senate principal-pair lead target/);
});

test("House Other winner absent from native support remains a scored three-class party event", t => {
  const f = setup(t, "house", 1), result = f.revision({ winnerGroup: "O", winningCandidateId: "INVENTED-write-in" });
  const scored = buildElectionNightScoring(f.loadedLock, f.ledger([result]));
  const winner = scored.rubricInput.winnerRows[0];
  assert.equal(winner.model.status, "scored"); assert.equal(winner.model.observedValue, "O");
  near(winner.model.loss, 1.58); near(winner.baseline.loss, 1.52);
  assert.deepEqual(scored.modelEvaluation.rows[0].winner.score.logLoss, { status: "infinite", value: null });
  assert.equal(scored.modelEvaluation.rows[0].winner.score.housePartyWinner.zeroOrigin.O, "structural_absent");
  assert.equal(scored.modelEvaluation.rows[0].winner.score.housePartyWinner.candidateRoster.listedInFrozenRoster, false);
});

test("an authenticated invented House lock can score explicit absent native IDs without inventing target or law", t => {
  const f = setup(t, "house", 1, { houseNullIdentities: true });
  assert.equal(f.loadedLock.modelForecast.run.targetId, null);
  assert.equal(f.loadedLock.modelForecast.run.lawId, null);
  const bridge = buildElectionNightScoring(f.loadedLock, f.ledger([f.revision({ marginPp: 0 })]));
  const result = gradeElectionNightRubric(bridge.rubricInput);
  assert.equal(result.status, "graded");
  assert.equal(result.overall.grade, "A"); // Entirely invented one-race fixture.
  const currentLike = setup(t, "house", 1, { currentHouse: true, houseNullIdentities: true });
  assert.throws(() => buildElectionNightScoring(currentLike.loadedLock,
    currentLike.ledger([currentLike.revision()])), /435 party winners/);
  assert.throws(() => setup(t, "senate", 1, { houseNullIdentities: true }), /bindings differ/);
});

test("the actual published House V1 projection has the explicit absent native IDs accepted by the lock contract", () => {
  const root = new URL("../", import.meta.url);
  const catalog = JSON.parse(readFileSync(new URL("data/house-live/catalog.json", root)));
  const entry = catalog.entries.find(row => row.id === catalog.default_id);
  const bytes = readFileSync(new URL(`data/house-live/${entry.result.path}`, root));
  assert.equal(createHash("sha256").update(bytes).digest("hex"), entry.result.sha256);
  const native = parseHouseLiveResult(JSON.parse(bytes));
  assert.ok(native);
  const forecast = toHouseElectionNightForecast(houseLiveView(native, entry));
  assert.equal(forecast.run.targetId, null); assert.equal(forecast.run.lawId, null);
  assert.equal(forecast.races.length, 435); assert.equal(forecast.summary.pairRaceCount, 412);
  assert.equal(validElectionNightRubricBinding({ chamber: "house", runId: forecast.run.id,
    resultSha256: forecast.run.artifactSha256, nativeForecastSha256: entry.result.sha256,
    targetId: forecast.run.targetId, lawId: forecast.run.lawId, issuedAt: forecast.run.issuedAt,
    cutoffAt: forecast.run.cutoffAt, horizonCalendarDays: 0 }), true);
  // This checks metadata compatibility only; it does not lock or score an election.
});

test("only authenticated store products enter scoring and current House editions require 435/412", t => {
  const f = setup(t), result = f.revision(), ledger = f.ledger([result]);
  assert.throws(() => buildElectionNightScoring({ ...f.loadedLock }, ledger), /authenticated loader/);
  assert.throws(() => buildElectionNightScoring(f.loadedLock, { ...ledger }), /authenticated loader/);
  const house = setup(t, "house", 1), houseLedger = house.ledger([house.revision()]);
  assert.throws(() => buildElectionNightScoring(f.loadedLock, houseLedger), /admitted evaluation lock/);
  // A fixture claiming the actual current House run name cannot grade a tiny cohort.
  // Its locked bytes remain invented; this tests the additional current-edition guard.
  const currentLike = setup(t, "house", 1, { currentHouse: true });
  assert.equal(currentLike.loadedLock.modelForecast.races.length, 1);
  assert.throws(() => buildElectionNightScoring(currentLike.loadedLock, currentLike.ledger([currentLike.revision()])), /435 party winners/);
});

test("actual October 20/26 Senate projections retain the reviewed ordered 35 named pairs without result scoring", { skip: !existsSync(ownerMappingPath) }, () => {
  const read = relative => JSON.parse(readFileSync(new URL(`../${relative}`, import.meta.url), "utf8"));
  const mappingBytes = readFileSync(ownerMappingPath);
  assert.equal(createHash("sha256").update(mappingBytes).digest("hex"),
    "9024069ac0f397c2d379663d653d384e32532548ef26772035eda61ce55265e0");
  const mapping = JSON.parse(mappingBytes), registry = read("public/data/explorer/registry.json");
  const forecasts = ["current-fundamentals", "current-polling"].map(id => {
    const entry = registry.runs.find(row => row.id === id), bytes = readFileSync(new URL(`../public${entry.href}`, import.meta.url));
    assert.equal(createHash("sha256").update(bytes).digest("hex"), entry.sha256);
    return toSenateElectionNightForecast(JSON.parse(bytes), entry);
  });
  assert.equal(mapping.races.length, 35);
  for (const forecast of forecasts) {
    assert.equal(forecast.races.length, 35);
    forecast.races.forEach((race, index) => {
      const locked = mapping.races[index];
      assert.equal(race.id, locked.race_id);
      assert.equal(race.pair.firstCandidateId, locked.first_candidate.id);
      assert.equal(race.pair.secondCandidateId, locked.second_candidate.id);
      assert.equal(locked.denominator, "sum_of_named_principal_candidate_votes_under_declared_rule_proxy");
      assert.equal(race.pair.denominator, "Share of votes for the two named principal contenders");
    });
  }
});
