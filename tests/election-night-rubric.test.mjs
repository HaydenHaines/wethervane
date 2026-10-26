import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { ELECTION_NIGHT_RUBRIC as policy, gradeElectionNightRubric } from "../lib/election-night-rubric.ts";

// All results, losses and contracts below are INVENTED analytic fixtures.
// No real 2026 observation or grade is loaded or assigned by these tests.
const hash = char => char.repeat(64);
function fixture({ chamber = "senate", count = 1, winnerModel = 0.19, winnerBaseline = 0.2, intervalModel = 19, intervalBaseline = 20 } = {}) {
  const forecast = (runId, sha) => ({ chamber, runId, resultSha256: hash(sha), nativeForecastSha256: hash(sha),
    targetId: "INVENTED-native-target", lawId: "INVENTED-law", issuedAt: "2026-11-01T13:00:00Z", cutoffAt: "2026-11-01T12:00:00Z", horizonCalendarDays: 2 });
  const model = forecast("INVENTED-model", "a"), baseline = forecast("INVENTED-history", "b");
  const winnerCohort = Array.from({ length: count }, (_, i) => ({ raceId: `INVENTED-${i}`, eventId: "INVENTED-event", roundId: "INVENTED-ultimate-round",
    resultRuleId: "INVENTED-final-rule", denominator: chamber === "house" ? "One official D/R/O winner" : "Votes for exactly the two named principals",
    estimand: chamber === "house" ? "house_official_party_winner" : "senate_principal_pair_lead",
    outcomeIds: chamber === "house" ? ["D", "R", "O"] : [`INVENTED-${i}:first`, `INVENTED-${i}:second`],
    firstCandidateId: chamber === "house" ? null : `INVENTED-${i}:first`, secondCandidateId: chamber === "house" ? null : `INVENTED-${i}:second` }));
  const intervalCohort = winnerCohort.map(t => ({ raceId: t.raceId, eventId: t.eventId, roundId: chamber === "senate" ? t.roundId : "INVENTED-pair-round", resultRuleId: t.resultRuleId,
    estimand: "named_pair_margin", firstCandidateId: `${t.raceId}:first`, secondCandidateId: `${t.raceId}:second`, denominator: "Votes for exactly the two named principals",
    units: "percentage_points", orientation: "first_minus_second", intervalLevel: 0.9 }));
  const contract = { schema: "ElectionNightRubricComparisonV1", id: "INVENTED-reviewed-comparison", sha256: hash("c"),
    rubricDecisionSha256: policy.decisionSha256, comparisonMode: "matched_horizon", label: "INVENTED same-horizon benchmark",
    model, baseline, comparisonReview: { id: "INVENTED-comparison-review", sha256: hash("d") },
    cohortDefinition: { id: "INVENTED-pre-outcome-cohort", sha256: hash("e") }, resultPolicy: { id: "INVENTED-finality-revisions", sha256: hash("f") },
    earlierBenchmarkPins: null, houseMappingReviewSha256: chamber === "house" ? policy.houseMappingReviewSha256 : null,
    winnerCohort, intervalCohort, intervalExclusions: [] };
  function rows(targets, winner) {
    return targets.map(target => {
      const result = { recordId: `INVENTED-result-${target.raceId}`, revision: "INVENTED-final-revision-1", recordSha256: hash("1"), sourceSha256: hash("2"),
        resultPolicySha256: contract.resultPolicy.sha256, finality: "certified" };
      const loss = (f, value) => ({ forecastSha256: f.resultSha256, target: structuredClone(target), status: "scored", metric: winner ? "multicategory_brier_sum" : "central90_interval_score_pp",
        result: structuredClone(result), loss: winner ? value * 2 : value, observedValue: winner ? target.outcomeIds[0] : 1, reason: null });
      return { raceId: target.raceId, comparisonContractSha256: contract.sha256, cohortDefinitionSha256: contract.cohortDefinition.sha256,
        model: loss(model, winner ? winnerModel : intervalModel), baseline: loss(baseline, winner ? winnerBaseline : intervalBaseline) };
    });
  }
  return { contract, modelForecast: structuredClone(model), baselineForecast: structuredClone(baseline), winnerRows: rows(winnerCohort, true), intervalRows: rows(intervalCohort, false) };
}
const near = (a, b) => assert.ok(Math.abs(a - b) < 1e-10, `${a} vs ${b}`);
function freeze(v) { if (v && typeof v === "object") { Object.freeze(v); Object.values(v).forEach(freeze); } return v; }
function assertCoverage(c) { assert.equal(c.total, c.final + c.provisional + c.pending + c.mismatched); }

test("exported editorial table, tolerance and V2 earlier-benchmark exception are immutable", () => {
  assert.deepEqual(policy.bands.map(b => [b.grade, b.minimumSkill]), [["A", 0.2], ["B", 0.1], ["C", 0], ["D", -0.1], ["F", null]]);
  assert.equal(policy.equalityTolerance, 1e-12);
  assert.equal(policy.decisionSha256, "edb82165b1e6fc76335912108b887b40ded71f0c0de76f3262dcdf39f3e0b79e");
  assert.ok(Object.isFrozen(policy)); assert.ok(Object.isFrozen(policy.bands)); assert.ok(Object.isFrozen(policy.bands[0]));
  assert.ok(Object.isFrozen(policy.earlierBenchmark.model));
  assert.throws(() => policy.bands[0].minimumSkill = 0);
});

test("all four reviewed analytic counterexamples produce the specified component floor and coin-flip gate", () => {
  const coin = gradeElectionNightRubric(fixture({ winnerModel: 0.25, winnerBaseline: 0.3, intervalModel: 10, intervalBaseline: 20 }));
  near(coin.winner.skill, 1 / 6); near(coin.winner.coinFlipSkill, 0);
  assert.equal(coin.winner.historyGrade, "B"); assert.equal(coin.winner.grade, "B"); assert.equal(coin.interval.grade, "A"); assert.equal(coin.overall.grade, "D");
  assert.equal(coin.winner.coinFlipCapApplied, false); assert.equal(coin.overall.coinFlipCapApplied, true);
  const wrong = gradeElectionNightRubric(fixture({ winnerModel: 0.9025, winnerBaseline: 0.4, intervalModel: 10, intervalBaseline: 20 }));
  near(wrong.winner.skill, -1.25625); assert.equal(wrong.winner.grade, "F"); assert.equal(wrong.overall.grade, "F");
  const vague = gradeElectionNightRubric(fixture({ winnerModel: 0.1875, winnerBaseline: 0.25, intervalModel: 200, intervalBaseline: 20 }));
  assert.equal(vague.winner.grade, "A"); near(vague.interval.skill, -9); assert.equal(vague.interval.grade, "F"); assert.equal(vague.overall.grade, "F");
  const mild = gradeElectionNightRubric(fixture());
  near(mild.winner.skill, 0.05); near(mild.interval.skill, 0.05); near(mild.winner.coinFlipSkill, 0.24);
  assert.equal(mild.winner.grade, "C"); assert.equal(mild.interval.grade, "C"); assert.equal(mild.overall.grade, "C");
  near(mild.winner.raw.modelMeanLoss, 0.19); near(mild.winner.raw.meanPairedLossDifference, -0.01);
});

test("unrounded skills meet inclusive boundaries with the declared absolute equality tolerance", () => {
  for (const [skill, expected] of [[0.2, "A"], [0.1, "B"], [0, "C"], [-0.1, "D"], [-0.10000001, "F"], [0.19999999, "B"]]) {
    const result = gradeElectionNightRubric(fixture({ winnerModel: 0.2 * (1 - skill), winnerBaseline: 0.2, intervalModel: 20 * (1 - skill), intervalBaseline: 20 }));
    assert.equal(result.interval.grade, expected, String(skill)); assert.equal(result.winner.grade, expected, String(skill));
  }
  const roundoff = gradeElectionNightRubric(fixture({ winnerModel: 0.16, winnerBaseline: 0.2, intervalModel: 16, intervalBaseline: 20 }));
  assert.equal(roundoff.winner.grade, "A"); // Binary floating-point 1-.16/.2 lies fractionally below .2.
});

test("House fixed D/R/O coin flip is distinct from uniform thirds and retains actual Other winners", () => {
  const f = fixture({ chamber: "house", winnerModel: 0.25, winnerBaseline: 0.3, intervalModel: 10 });
  let result = gradeElectionNightRubric(f);
  near(result.winner.raw.coinFlipMeanLoss, 0.25); near(result.winner.raw.uniformThreeCategoryMeanLoss, 1 / 3);
  near(result.winner.coinFlipSkill, 0); near(result.winner.uniformThreeCategorySkill, 0.25);
  assert.equal(result.winner.grade, "B"); assert.equal(result.overall.grade, "D");
  f.winnerRows[0].model.observedValue = f.winnerRows[0].baseline.observedValue = "O";
  f.winnerRows[0].model.loss = 1.5; f.winnerRows[0].baseline.loss = 1.7;
  result = gradeElectionNightRubric(f);
  near(result.winner.raw.coinFlipMeanLoss, 0.75); near(result.winner.raw.uniformThreeCategoryMeanLoss, 1 / 3);
  assert.equal(result.winner.coverage.final, 1); assert.equal(result.winner.raw.count, 1); assert.equal(result.overall.grade, "D");
  const full = fixture({ chamber: "house", count: 3 });
  for (let i = 0; i < 3; i++) full.winnerRows[i].model.observedValue = full.winnerRows[i].baseline.observedValue = ["D", "R", "O"][i];
  result = gradeElectionNightRubric(full); near(result.winner.raw.coinFlipMeanLoss, 5 / 12);
  assert.equal(result.winner.coverage.final, 3);
  const droppedO = structuredClone(full); droppedO.contract.winnerCohort[0].outcomeIds = ["D", "R"];
  assert.equal(gradeElectionNightRubric(droppedO).overall.grade, null);
});

test("zero baseline withholds component and overall letters while preserving raw paired losses and differences", () => {
  const result = gradeElectionNightRubric(fixture({ winnerModel: 0.2, winnerBaseline: 0 }));
  assert.equal(result.winner.complete, true); assert.equal(result.winner.skill, null); assert.equal(result.winner.grade, null); assert.equal(result.overall.grade, null);
  near(result.winner.raw.modelMeanLoss, 0.2); near(result.winner.raw.baselineMeanLoss, 0); near(result.winner.raw.meanPairedLossDifference, 0.2);
  assert.match(result.winner.reason, /zero/);
  const interval = gradeElectionNightRubric(fixture({ intervalModel: 1, intervalBaseline: 0 }));
  assert.equal(interval.interval.grade, null); assert.equal(interval.overall.grade, null); near(interval.interval.raw.meanPairedLossDifference, 1);
});

test("called/provisional, missing, unsupported, duplicate and unexpected rows cannot earn complete-cohort letters", () => {
  for (const finality of ["called", "provisional", "unresolved", "unsupported"]) {
    const f = fixture(); for (const row of [...f.winnerRows, ...f.intervalRows]) row.model.result.finality = row.baseline.result.finality = finality;
    const result = gradeElectionNightRubric(f);
    assert.equal(result.overall.grade, null); assert.equal(result.winner.grade, null); assert.equal(result.interval.grade, null);
    const unavailable = finality === "unresolved" || finality === "unsupported";
    assert.equal(result.winner.coverage.provisional, unavailable ? 0 : 1);
    assert.equal(result.winner.coverage.pending, unavailable ? 1 : 0);
    if (unavailable) { assert.equal(result.winner.raw.count, 0); assert.equal(result.winner.raw.modelMeanLoss, null); }
    else near(result.winner.raw.modelMeanLoss, 0.19);
    assert.equal(result.winner.raw.scope, "available_matched_rows");
  }
  const missing = fixture({ count: 2 }); missing.winnerRows.pop();
  let result = gradeElectionNightRubric(missing); assert.equal(result.winner.coverage.pending, 1); assert.equal(result.winner.grade, null);
  assert.equal(result.overall.grade, null); assert.equal(result.winner.raw.count, 1); assertCoverage(result.winner.coverage);
  const unsupported = fixture(); unsupported.winnerRows[0].model.status = "unsupported";
  result = gradeElectionNightRubric(unsupported); assert.equal(result.winner.grade, null); assert.match(result.winner.rows[0].reason, /unsupported/);
  const duplicate = fixture(); duplicate.winnerRows.push(structuredClone(duplicate.winnerRows[0]));
  result = gradeElectionNightRubric(duplicate); assert.equal(result.winner.coverage.mismatched, 1); assert.equal(result.winner.grade, null);
  const extra = fixture(); extra.intervalRows.push({ ...structuredClone(extra.intervalRows[0]), raceId: "INVENTED-outside-cohort" });
  result = gradeElectionNightRubric(extra); assert.equal(result.interval.coverage.unexpected, 1); assert.equal(result.unexpectedRows.length, 1); assert.equal(result.overall.grade, null);
});

test("winner-only and structurally excluded pair cohorts are explicit; no margins are invented", () => {
  const f = fixture({ chamber: "house", count: 2 });
  const removed = f.contract.intervalCohort.pop(); f.intervalRows.pop(); f.contract.intervalExclusions = [{ raceId: removed.raceId, reason: "INVENTED pre-outcome unsupported pair" }];
  let result = gradeElectionNightRubric(f); assert.equal(result.interval.coverage.total, 1); assert.equal(result.overall.grade, "C");
  assert.deepEqual(result.intervalExcluded, f.contract.intervalExclusions);
  f.contract.intervalExclusions = []; result = gradeElectionNightRubric(f); assert.equal(result.overall.grade, null); assert.ok(result.issues.length > 0);
  const winnerOnly = fixture(); winnerOnly.contract.intervalExclusions = [{ raceId: winnerOnly.contract.winnerCohort[0].raceId, reason: "INVENTED no admitted pair" }];
  winnerOnly.contract.intervalCohort = []; winnerOnly.intervalRows = [];
  result = gradeElectionNightRubric(winnerOnly); assert.equal(result.winnerOnlyGrade, "C"); assert.equal(result.overall.grade, null);
  winnerOnly.winnerRows[0].model.loss = 0.5; winnerOnly.winnerRows[0].baseline.loss = 0.6;
  result = gradeElectionNightRubric(winnerOnly); assert.equal(result.winner.grade, "B"); assert.equal(result.winnerOnlyGrade, "D");
});

test("contract, forecast, metric, event, cohort and result revisions must bind exactly", () => {
  const changes = [
    f => f.modelForecast.resultSha256 = hash("9"), f => f.contract.rubricDecisionSha256 = hash("9"),
    f => f.contract.comparisonReview.sha256 = "invalid", f => f.contract.cohortDefinition.id = "",
    f => f.winnerRows[0].comparisonContractSha256 = hash("9"), f => f.winnerRows[0].cohortDefinitionSha256 = hash("9"),
    f => f.winnerRows[0].model.metric = "binary_brier_already_divided", f => f.winnerRows[0].baseline.target.eventId = "INVENTED-other-event",
    f => f.intervalRows[0].baseline.target.roundId = "INVENTED-different-round", f => f.intervalRows[0].baseline.target.denominator = "all votes",
    f => f.intervalRows[0].baseline.target.firstCandidateId = "INVENTED-other-person", f => f.winnerRows[0].baseline.result.revision = "INVENTED-newer-revision",
    f => f.winnerRows[0].baseline.result.recordSha256 = hash("9"), f => f.winnerRows[0].model.result.resultPolicySha256 = hash("9"),
    f => f.winnerRows[0].baseline.observedValue = "INVENTED-0:second", f => f.intervalRows[0].baseline.observedValue = 2,
    f => { f.contract.intervalCohort[0].eventId = "INVENTED-other-event"; for (const r of [f.intervalRows[0].model,f.intervalRows[0].baseline]) r.target.eventId = "INVENTED-other-event"; },
    f => f.winnerRows[0].model.loss = 2.1, f => f.intervalRows[0].model.loss = -1,
  ];
  for (const change of changes) {
    const f = fixture(); change(f); const result = gradeElectionNightRubric(f);
    assert.equal(result.overall.grade, null, String(change));
    assert.ok(result.issues.length || result.winner.coverage.mismatched || result.interval.coverage.mismatched, String(change));
  }
  const changed = fixture(); changed.winnerRows[0].baseline.result.revision = "INVENTED-changed";
  const row = gradeElectionNightRubric(changed).winner.rows[0];
  near(row.modelLoss, 0.19); near(row.baselineLoss, 0.2); assert.equal(row.pairedLossDifference, null);
  for (const value of [null, {}, { verified: true, approved: true, locked: true }]) assert.equal(gradeElectionNightRubric(value).overall.grade, null);
});

function earlierFixture() {
  const f = fixture({ count: 35 }), e = policy.earlierBenchmark;
  f.contract.id = e.comparisonContractId; f.contract.comparisonMode = "explicit_earlier_benchmark";
  f.contract.label = "October 20 basic history (earlier issued)"; f.contract.comparisonReview.sha256 = e.reviewSha256;
  f.contract.earlierBenchmarkPins = { mappingSha256: e.mappingSha256, freezeManifestSha256: e.freezeManifestSha256 };
  Object.assign(f.contract.model, e.model); Object.assign(f.contract.baseline, e.baseline);
  // Real pinned forecast/pair identities, but INVENTED admitted operator, results and losses.
  // The native mapping itself does not admit any official result or assign a real grade.
  const pairs = JSON.parse(readFileSync(new URL("../public/data/election-night/references.json", import.meta.url))).entries[0].reference.cohort;
  pairs.forEach((p, i) => {
    for (const target of [f.contract.winnerCohort[i], f.contract.intervalCohort[i]]) {
      target.raceId = p.race_id; target.firstCandidateId = p.first_candidate_id; target.secondCandidateId = p.second_candidate_id; target.denominator = p.denominator;
      if (target.outcomeIds) target.outcomeIds = [p.first_candidate_id, p.second_candidate_id];
    }
    for (const [row, target] of [[f.winnerRows[i],f.contract.winnerCohort[i]], [f.intervalRows[i],f.contract.intervalCohort[i]]]) {
      row.raceId = p.race_id;
      for (const [loss, forecast] of [[row.model,f.contract.model],[row.baseline,f.contract.baseline]]) {
        loss.target = structuredClone(target); loss.forecastSha256 = forecast.resultSha256;
        loss.result.recordId = `INVENTED-result-${p.race_id}`;
        if (target.outcomeIds) loss.observedValue = p.first_candidate_id;
      }
    }
  });
  f.modelForecast = structuredClone(f.contract.model); f.baselineForecast = structuredClone(f.contract.baseline);
  return f;
}
test("only the explicit V2 earlier Senate benchmark is allowed and never described as matched-horizon or causal polling skill", () => {
  const f = earlierFixture(), result = gradeElectionNightRubric(f);
  assert.equal(result.overall.grade, "C"); assert.equal(result.comparison.matchedHorizon, false);
  assert.match(result.comparison.explanation, /different information and lead times/); assert.match(result.comparison.explanation, /does not isolate/);
  const changes = [
    f => f.contract.comparisonMode = "matched_horizon", f => f.contract.id = "INVENTED-other-earlier-comparison",
    f => f.contract.earlierBenchmarkPins.mappingSha256 = hash("9"), f => f.contract.earlierBenchmarkPins.freezeManifestSha256 = hash("9"),
    f => f.contract.comparisonReview.sha256 = hash("9"), f => f.contract.baseline.cutoffAt = f.contract.model.cutoffAt,
    f => f.contract.baseline.nativeForecastSha256 = hash("9"),
    f => { f.contract.baseline.resultSha256 = hash("9"); f.baselineForecast.resultSha256 = hash("9"); },
    f => { f.contract.model.issuedAt = "2026-10-26T15:06:00Z"; f.modelForecast.issuedAt = f.contract.model.issuedAt; },
    f => { f.contract.model.lawId = "INVENTED-other-law"; f.modelForecast.lawId = f.contract.model.lawId; },
    f => { f.contract.winnerCohort.reverse(); f.contract.intervalCohort.reverse(); },
    f => { f.contract.winnerCohort[0].firstCandidateId = "INVENTED-other-person"; }, f => f.contract.winnerCohort.pop(),
  ];
  for (const change of changes) { const x = earlierFixture(); change(x); assert.equal(gradeElectionNightRubric(x).overall.grade, null, String(change)); }
  const generic = fixture(); generic.contract.baseline.cutoffAt = "2026-10-20T15:21:23Z"; generic.baselineForecast = structuredClone(generic.contract.baseline);
  assert.equal(gradeElectionNightRubric(generic).overall.grade, null);
});

test("grading never mutates frozen inputs and output remains serialization safe", () => {
  const f = fixture({ chamber: "house", count: 3 }), before = JSON.stringify(f); freeze(f);
  const result = gradeElectionNightRubric(f); assert.equal(result.overall.grade, "C");
  result.winner.rows[0].modelLoss = 1; result.comparison.label = "changed output only";
  assert.equal(JSON.stringify(f), before); assert.equal(JSON.parse(JSON.stringify(result)).interval.grade, "C");
  const tiny = fixture({ winnerBaseline: Number.MIN_VALUE }); const unrepresentable = gradeElectionNightRubric(tiny);
  assert.equal(unrepresentable.winner.skill, null); assert.equal(unrepresentable.overall.grade, null);
  assert.doesNotThrow(() => JSON.parse(JSON.stringify(unrepresentable)));
});

test("malformed JSON candidates fail closed without exceptions or numeric letters", () => {
  const changes = [
    f => f.contract.winnerCohort[0].outcomeIds = undefined,
    f => f.contract.winnerCohort[0] = null,
    f => f.contract.intervalCohort[0] = null,
    f => f.contract.intervalExclusions = [null],
    f => f.winnerRows[0] = null,
    f => f.intervalRows[0] = null,
    f => f.winnerRows[0].model = null,
    f => f.contract.winnerCohort[0].outcomeIds = ["D", null],
  ];
  for (const change of changes) { const f = fixture(); change(f); let r; assert.doesNotThrow(() => r = gradeElectionNightRubric(f)); assert.equal(r.overall.grade, null); }
});
