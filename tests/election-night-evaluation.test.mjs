import assert from "node:assert/strict";
import test from "node:test";
import { multicategoryBrier, categoricalLogLoss, central90IntervalScore, evaluateElectionNightForecast } from "../lib/election-night-evaluation.ts";

// All forecast numbers, people, results, source pins and protocol reviews below are
// INVENTED software fixtures. They are not real returns or scientific admission.
const H = "a".repeat(64), review = { sha256: "b".repeat(64), url: "https://example.test/INVENTED-review" };
const source = { sha256: "c".repeat(64), url: "https://example.test/INVENTED-result", publishedAt: "2026-11-04T12:00:00Z", retrievedAt: "2026-11-04T13:00:00Z" };
const denominator = "Share of votes for the two named principal contenders";
const clone = structuredClone;
const near = (a, b) => assert.ok(Math.abs(a - b) < 1e-12, `${a} vs ${b}`);
function frozen(v) { if (v && typeof v === "object") { Object.freeze(v); Object.values(v).forEach(frozen); } return v; }
function fixture(count = 1) {
  const races = Array.from({ length: count }, (_, i) => {
    const id = `INVENTED-race-${i + 1}`;
    const member = (suffix, name) => ({ id: `${id}:${suffix}`, name, party: suffix[0], sourceUrls: [] });
    const outcomes = [
      { id: `${id}:A`, name: "Invented A", party: "D", kind: "candidate", members: [member("a", "Invented A")], winProbability: 0.6 },
      { id: `${id}:B`, name: "Invented pooled B", party: "O", kind: "group", members: [member("b1", "Invented B1"), member("b2", "Invented B2")], winProbability: 0.3 },
      { id: `${id}:C`, name: "Invented C", party: "R", kind: "candidate", members: [member("c", "Invented C")], winProbability: 0.1 },
    ];
    return { id, state: "XX", seatId: `XX-${i}`, electionLabel: "INVENTED contest", ruleFamily: "plurality", method: "INVENTED proxy",
      notes: ["INVENTED listed support"], sourceUrls: [], outcomes,
      pair: { firstCandidateId: outcomes[0].members[0].id, secondCandidateId: outcomes[1].members[0].id,
        firstCandidateName: "Invented A", secondCandidateName: "Invented B1", denominator, units: "percentage_points", orientation: "first_minus_second",
        meanMarginPp: 2, medianMarginPp: 1, range90: { lower: -5, upper: 5 }, firstExceedsHalf: 0.55,
        winnerRelationship: "separate_conditional_event", closeWithin2Pp: 0.2, firstAtLeast10Pp: 0.1, secondAtLeast10Pp: 0.1, qualifications: ["INVENTED pair"] },
      modalPick: { outcomeId: outcomes[0].id, name: outcomes[0].name, kind: "candidate", probability: 0.6, tiedOutcomeIds: [outcomes[0].id] } };
  });
  const forecast = { schema: "ElectionNightForecastV1", chamber: "house", editionStatus: "dated_snapshot",
    run: { id: "INVENTED-saved-run", label: "INVENTED forecast", year: 2026, configuration: {}, artifactSha256: H,
      issuedAt: "2026-11-01T13:00:00Z", cutoffAt: "2026-11-01T12:00:00Z", strictAsOf: false, sourceCutoffs: [], targetId: null, lawId: null },
    qualification: ["INVENTED software fixture"], inputNotices: [], sourceUrls: [], sourceHashes: {}, sourceNotes: [], races, chamberScenario: null,
    summary: { raceCount: count, pairRaceCount: count, expectedModalMisses: 0.4 * count, topTieRaceCount: 0, tiePolicy: "lexicographic_outcome_id", interpretation: "INVENTED" } };
  const protocol = { schema: "ElectionNightEvaluationProtocolV1", id: "INVENTED-protocol", reviewedDefinition: clone(review),
    forecast: { chamber: forecast.chamber, runId: forecast.run.id, artifactSha256: H, issuedAt: forecast.run.issuedAt, cutoffAt: forecast.run.cutoffAt },
    races: races.map(r => ({ raceId: r.id, winner: { estimand: "listed_modeled_winner", eventId: "INVENTED-event", roundId: "INVENTED-ultimate-round",
      ruleFamily: r.ruleFamily, outcomeIds: r.outcomes.map(o => o.id), minimumStatus: "called" },
      margin: { estimand: "named_pair_margin", eventId: "INVENTED-event", roundId: "INVENTED-pair-round", ruleFamily: r.ruleFamily,
        firstCandidateId: r.pair.firstCandidateId, secondCandidateId: r.pair.secondCandidateId, denominator, units: "percentage_points",
        orientation: "first_minus_second", pointRule: "mean", minimumStatus: "final" } })) };
  const records = races.map((r, i) => ({ schema: "ElectionNightOutcomeRecordV1", id: `INVENTED-record-${i + 1}`, revision: "INVENTED-revision-1", raceId: r.id,
    source: clone(source), winner: { status: "certified", estimand: "listed_modeled_winner", eventId: "INVENTED-event", roundId: "INVENTED-ultimate-round",
      ruleFamily: r.ruleFamily, outcomeId: r.outcomes[0].id, kind: "candidate", candidateId: r.outcomes[0].members[0].id, reason: null },
    margin: { status: "certified", estimand: "named_pair_margin", eventId: "INVENTED-event", roundId: "INVENTED-pair-round", ruleFamily: r.ruleFamily,
      firstCandidateId: r.pair.firstCandidateId, secondCandidateId: r.pair.secondCandidateId, denominator, units: "percentage_points",
      orientation: "first_minus_second", marginPp: -7, reason: null } }));
  return { forecast, protocol, records };
}
function evaluate(f) { return evaluateElectionNightForecast(f.forecast, f.protocol, f.records); }
function coverageTotal(c) { assert.equal(c.total, c.scored + c.unresolved + c.unsupported + c.mismatched); assert.equal(c.scored, c.scoredCalled + c.scoredFinal + c.scoredCertified); }

test("proper categorical scores use full unordered support, natural logs and explicit binary normalization", () => {
  const p = [{ id: "A", winProbability: 0.5 }, { id: "B", winProbability: 0.3 }, { id: "C", winProbability: 0.2 }];
  near(multicategoryBrier(p, "C"), 0.98); near(categoricalLogLoss(p, "C").value, Math.log(5));
  const binary = [{ id: "A", winProbability: 0.7 }, { id: "B", winProbability: 0.3 }];
  near(multicategoryBrier(binary, "A"), 0.18); near(multicategoryBrier(binary, "A"), 2 * (0.7 - 1) ** 2);
  near(categoricalLogLoss(binary, "A").value, -Math.log(0.7));
  assert.deepEqual(categoricalLogLoss([{ id: "A", winProbability: 1 }], "A"), { status: "finite", value: 0 });
  assert.throws(() => multicategoryBrier(p, "outside"), /outside declared support/);
  assert.throws(() => categoricalLogLoss([{ id: "A", winProbability: 0.8 }], "A"), /invalid categorical/);
  assert.throws(() => multicategoryBrier([{ id: "A", winProbability: 0.5 }, { id: "A", winProbability: 0.5 }], "A"), /invalid categorical/);
});

test("zero-probability observed outcomes retain infinite log loss through JSON without clipping", () => {
  const p = [{ id: "A", winProbability: 0.5 }, { id: "B", winProbability: 0.5 }, { id: "C", winProbability: 0 }];
  near(multicategoryBrier(p, "C"), 1.5);
  assert.deepEqual(JSON.parse(JSON.stringify(categoricalLogLoss(p, "C"))), { status: "infinite", value: null });
  const tiny = [{ id: "A", winProbability: 1 }, { id: "B", winProbability: Number.MIN_VALUE }];
  assert.equal(categoricalLogLoss(tiny, "B").status, "finite");
  near(categoricalLogLoss(tiny, "B").value, -Math.log(Number.MIN_VALUE));
  const f = fixture(); f.forecast.races[0].outcomes.forEach((o, i) => o.winProbability = i === 0 ? 0 : 0.5);
  const result = JSON.parse(JSON.stringify(evaluate(f)));
  assert.deepEqual(result.rows[0].winner.score.logLoss, { status: "infinite", value: null });
  assert.equal(result.rows[0].winner.status, "scored");
});

test("a tied modal call grades the fixed outcome-ID choice, not any equally probable winner", () => {
  const f = fixture(), race = f.forecast.races[0];
  race.outcomes.forEach((outcome, index) => { outcome.winProbability = index < 2 ? 0.5 : 0; });
  race.modalPick.tiedOutcomeIds = race.outcomes.slice(0, 2).map(outcome => outcome.id);
  const tied = race.outcomes[1];
  f.records[0].winner = { ...f.records[0].winner, outcomeId: tied.id, kind: "group", candidateId: tied.members[0].id };
  let score = evaluate(f).rows[0].winner.score;
  assert.equal(score.modalPickCorrect, false);
  near(score.brierSum, 0.5);
  near(score.logLoss.value, Math.log(2));
  f.records[0].winner = { ...f.records[0].winner, outcomeId: race.outcomes[0].id, kind: "candidate", candidateId: race.outcomes[0].members[0].id };
  score = evaluate(f).rows[0].winner.score;
  assert.equal(score.modalPickCorrect, true);
  near(score.brierSum, 0.5);
});

test("central 90% interval scoring includes width, both tail penalties and inclusive endpoint coverage", () => {
  for (const observed of [-5, 0, 5]) {
    const s = central90IntervalScore(-5, 5, observed); assert.equal(s.covered, true); assert.equal(s.widthPp, 10); assert.equal(s.scorePp, 10);
  }
  assert.equal(central90IntervalScore(-5, 5, -10).scorePp, 110);
  assert.equal(central90IntervalScore(-5, 5, 10).scorePp, 110);
  assert.equal(central90IntervalScore(-5, 5, 10).covered, false);
  assert.equal(central90IntervalScore(-20, 20, 0).scorePp, 40); // Covering everything with a wider range is not free.
  assert.throws(() => central90IntervalScore(1, -1, 0), /invalid central/);
  assert.throws(() => central90IntervalScore(-1, 1, Infinity), /invalid central/);
});

test("certified named-pair margins use observed-minus-predicted orientation and an explicit mean or median rule", () => {
  const f = fixture(); let r = evaluate(f); const s = r.rows[0].margin.score;
  assert.equal(s.predictedMarginPp, 2); assert.equal(s.observedMarginPp, -7);
  assert.equal(s.signedErrorPp, -9); assert.equal(s.absoluteErrorPp, 9); assert.equal(s.interval90.scorePp, 50);
  assert.equal(s.orientation, "first_minus_second"); assert.equal(s.denominator, denominator);
  assert.equal(s.estimand, "named_pair_margin"); assert.equal(s.roundId, "INVENTED-pair-round");
  assert.equal(r.rows[0].winner.score.estimand, "listed_modeled_winner");
  assert.equal(r.rows[0].winner.score.roundId, "INVENTED-ultimate-round");
  assert.equal(r.rows[0].margin.provisional, false); assert.equal(r.coverage.margin.scoredCertified, 1);
  f.protocol.races[0].margin.pointRule = "median"; r = evaluate(f);
  assert.equal(r.rows[0].margin.score.predictedMarginPp, 1); assert.equal(r.rows[0].margin.score.signedErrorPp, -8);
  assert.equal(r.rows[0].margin.score.interval90.scorePp, 50); assert.equal(r.grade, "not_graded");
});

test("called winners and final/certified margins have separate statuses and coverage counts", () => {
  const f = fixture(3);
  f.records[0].winner.status = "called"; f.records[0].margin.status = "provisional";
  f.records[1].winner.status = "final"; f.records[1].margin.status = "final";
  const r = evaluate(f);
  assert.equal(r.rows[0].winner.status, "scored"); assert.equal(r.rows[0].winner.provisional, true);
  assert.equal(r.rows[0].margin.status, "unresolved"); assert.equal(r.rows[0].margin.score, null);
  assert.equal(r.coverage.winner.scoredCalled, 1); assert.equal(r.coverage.winner.scoredFinal, 1); assert.equal(r.coverage.winner.scoredCertified, 1);
  assert.equal(r.coverage.margin.scoredCalled, 0); assert.equal(r.coverage.margin.scoredFinal, 1); assert.equal(r.coverage.margin.scoredCertified, 1);
  coverageTotal(r.coverage.winner); coverageTotal(r.coverage.margin);
  f.protocol.races[0].winner.minimumStatus = "final"; let limited = evaluate(f);
  assert.equal(limited.rows[0].winner.status, "unresolved"); assert.equal(limited.rows[0].winner.provisional, true);
  assert.equal(limited.coverage.winner.scoredCalled, 0);
  f.protocol.races[1].margin.minimumStatus = "certified"; limited = evaluate(f);
  assert.equal(limited.rows[1].margin.status, "unresolved");
  f.records[0].margin.status = "called"; assert.equal(evaluate(f).rows[0].margin.status, "unresolved");
});

test("group outcome scoring preserves pooled membership without creating individual probabilities", () => {
  const f = fixture(), group = f.forecast.races[0].outcomes[1];
  f.records[0].winner.outcomeId = group.id; f.records[0].winner.kind = "group"; f.records[0].winner.candidateId = group.members[1].id;
  let r = evaluate(f); assert.equal(r.rows[0].winner.status, "scored");
  near(r.rows[0].winner.score.brierSum, 0.86); near(r.rows[0].winner.score.logLoss.value, -Math.log(0.3));
  assert.equal(r.rows[0].winner.score.modalPickCorrect, false); assert.equal(r.rows[0].winner.score.kind, "group");
  assert.ok(!("candidateProbability" in r.rows[0].winner.score));
  f.records[0].winner.candidateId = null; assert.equal(evaluate(f).rows[0].winner.status, "scored");
  f.records[0].winner.kind = "candidate"; assert.equal(evaluate(f).rows[0].winner.status, "mismatched");
  f.records[0].winner.kind = "group"; f.records[0].winner.candidateId = f.forecast.races[0].outcomes[0].members[0].id;
  assert.equal(evaluate(f).rows[0].winner.status, "mismatched");
});

test("a Senate proxy is never silently scored as full-ballot truth", () => {
  const f = fixture(); f.forecast.chamber = f.protocol.forecast.chamber = "senate";
  const race = f.forecast.races[0];
  race.outcomes = [race.outcomes[0], race.outcomes[2]]; race.outcomes[1].winProbability = 0.4;
  race.pair.secondCandidateId = race.outcomes[1].members[0].id; race.pair.secondCandidateName = race.outcomes[1].name;
  race.pair.firstExceedsHalf = 0.6; race.pair.winnerRelationship = "same_event";
  f.protocol.races[0].winner.outcomeIds = race.outcomes.map(o => o.id);
  f.protocol.races[0].margin.secondCandidateId = f.records[0].margin.secondCandidateId = race.pair.secondCandidateId;
  f.protocol.races[0].winner.estimand = "conditional_principal_pair_winner";
  f.records[0].winner.estimand = "full_ballot_winner";
  assert.equal(evaluate(f).rows[0].winner.status, "mismatched");
  f.records[0].winner.estimand = "conditional_principal_pair_winner";
  assert.equal(evaluate(f).rows[0].winner.status, "scored");
  f.records[0].winner.outcomeId = "INVENTED-unlisted-actual-winner";
  assert.equal(evaluate(f).rows[0].winner.status, "unsupported");
  race.outcomes[0].kind = "group";
  assert.equal(evaluate(f).rows[0].winner.status, "mismatched");
});

test("event, round, target, named candidates, denominator, scale and finality require explicit matches", () => {
  const changes = [
    ["eventId", "INVENTED-other-event"], ["roundId", "INVENTED-first-round"], ["estimand", "full_ballot_margin"],
    ["firstCandidateId", "INVENTED-other-person"], ["secondCandidateId", "INVENTED-other-person"],
    ["denominator", "all valid ballot votes"], ["orientation", "second_minus_first"], ["units", "share_fraction"], ["ruleFamily", "ranked_choice"],
  ];
  for (const [field, value] of changes) {
    const f = fixture(); f.records[0].margin[field] = value;
    assert.equal(evaluate(f).rows[0].margin.status, "mismatched", field);
  }
  for (const field of ["eventId", "roundId", "estimand", "ruleFamily"]) {
    const f = fixture(); f.records[0].winner[field] = "INVENTED-different";
    assert.equal(evaluate(f).rows[0].winner.status, "mismatched", field);
  }
  for (const value of [null, NaN, Infinity, 101, -101]) {
    const f = fixture(); f.records[0].margin.marginPp = value;
    assert.equal(evaluate(f).rows[0].margin.status, "mismatched");
  }
  const f = fixture(); f.records[0].margin.firstCandidateId = f.forecast.races[0].pair.secondCandidateId;
  f.records[0].margin.secondCandidateId = f.forecast.races[0].pair.firstCandidateId;
  f.records[0].margin.marginPp *= -1; assert.equal(evaluate(f).rows[0].margin.status, "mismatched");
});

test("every frozen race remains in unresolved, unsupported, mismatched or scored accounting", () => {
  const f = fixture(4); f.records.pop();
  f.forecast.races[0].pair = null; f.protocol.races[0].margin = null;
  f.records[1].winner.outcomeId = "INVENTED-outside-support";
  f.records[2].winner.roundId = "INVENTED-wrong-round";
  const r = evaluate(f); assert.equal(r.rows.length, 4);
  assert.deepEqual(r.coverage.winner, { total: 4, scored: 1, unresolved: 1, unsupported: 1, mismatched: 1, scoredCalled: 0, scoredFinal: 0, scoredCertified: 1 });
  assert.equal(r.rows[0].margin.status, "unsupported"); assert.equal(r.rows[3].margin.status, "unresolved");
  coverageTotal(r.coverage.margin);
  f.records[0].winner.status = "unsupported"; f.records[0].winner.reason = "INVENTED source does not resolve this target";
  assert.equal(evaluate(f).rows[0].winner.reason, f.records[0].winner.reason);
  const empty = fixture(3); empty.records = [];
  const pending = evaluate(empty);
  assert.equal(pending.coverage.winner.unresolved, 3); assert.equal(pending.coverage.margin.unresolved, 3);
  assert.equal(pending.coverage.winner.scored, 0); assert.equal(pending.rows.length, 3);
});

test("duplicate revisions and extra/missing target records are explicit and never silently selected", () => {
  const f = fixture(2); const duplicate = clone(f.records[0]); duplicate.id = "INVENTED-new-revision"; duplicate.revision = "INVENTED-2";
  f.records.push(duplicate, { ...clone(f.records[0]), id: "INVENTED-extra", raceId: "INVENTED-outside-cohort" });
  let r = evaluate(f); assert.equal(r.rows[0].winner.status, "mismatched"); assert.equal(r.rows[0].records.length, 2);
  assert.equal(r.coverage.unmatchedRecords, 1); assert.equal(r.unmatchedRecords[0].recordId, "INVENTED-extra");
  assert.equal(r.coverage.resultRecords, 4); assert.equal(r.rows[1].winner.status, "scored");
  f.protocol.races.pop(); r = evaluate(f); assert.equal(r.rows[1].winner.status, "mismatched"); assert.equal(r.protocolIssues.length, 1);
  f.protocol.races.push(clone(f.protocol.races[0])); r = evaluate(f); assert.ok(r.protocolIssues.some(i => i.reason.includes("Duplicate")));
  f.protocol.races.push({ ...clone(f.protocol.races[0]), raceId: "INVENTED-outside-protocol" });
  assert.ok(evaluate(f).protocolIssues.some(i => i.raceId === "INVENTED-outside-protocol"));
  const sameId = fixture(2); sameId.records[1].id = sameId.records[0].id;
  assert.ok(evaluate(sameId).rows.every(row => row.winner.status === "mismatched"));
});

test("protocol binding and source metadata fail closed; pins alone do not authorize mismatched target mappings", () => {
  for (const field of ["chamber", "runId", "artifactSha256", "issuedAt", "cutoffAt"]) {
    const f = fixture(); f.protocol.forecast[field] = "INVENTED-other";
    assert.throws(() => evaluate(f), /exact forecast edition/, field);
  }
  const badReview = fixture(); badReview.protocol.reviewedDefinition.sha256 = "invalid";
  assert.throws(() => evaluate(badReview), /exact forecast edition/);
  for (const change of [r => r.source.sha256 = "invalid", r => r.source.url = "file:///invented", r => r.source.publishedAt = "2027-01-01T00:00:00Z", r => r.revision = ""]) {
    const f = fixture(); change(f.records[0]); const row = evaluate(f).rows[0];
    assert.equal(row.winner.status, "mismatched"); assert.equal(row.margin.status, "mismatched");
  }
  const f = fixture(); f.protocol.races[0].winner.outcomeIds.pop();
  assert.equal(evaluate(f).rows[0].winner.status, "mismatched"); assert.equal(evaluate(f).rows[0].margin.status, "scored");
  f.protocol.races[0].winner.minimumStatus = "toString";
  assert.equal(evaluate(f).rows[0].winner.status, "mismatched");
});

test("evaluation never mutates the forecast, protocol or result records and returns detached source references", () => {
  const f = fixture(2); const before = JSON.stringify(f); frozen(f);
  const r = evaluate(f); assert.equal(JSON.stringify(f), before);
  r.rows[0].records[0].source.url = "https://example.test/changed-output-only";
  r.reviewedDefinition.url = "https://example.test/changed-review-output-only";
  r.rows[0].margin.score.interval90.lower = -99;
  assert.equal(JSON.stringify(f), before); assert.equal(JSON.parse(JSON.stringify(r)).grade, "not_graded");
  assert.ok(!("evaluationLocked" in r)); assert.ok(!("accuracy" in r));
});
