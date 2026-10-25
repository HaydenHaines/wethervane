import test from "node:test";
import assert from "node:assert/strict";
import { candidatePairComparisonReadiness as readiness } from "../lib/candidate-pair-comparison-view.ts";
import { partialPairComparisonFixture, missingObservationFixture, incompatiblePairFixture } from "./fixtures/candidate-pair-comparison.mjs";

const first = view => view.rows[0].arms.count;

test("fixed expected coverage retains missing pairs and undefined/unavailable model slots", () => {
  const result=readiness(partialPairComparisonFixture());
  assert.equal(result.valid,true);assert.equal(result.productionActivated,false);
  assert.deepEqual(result.coverage,{expectedPairs:3,suppliedPairs:2,fullyScoredPairs:1,scoredByArm:{count:1,F:1,FP:1},missingPairIds:["pair-Ridge"]});
});

test("exact ordered endpoints, context, source, contest and date bindings cannot be mixed", () => {
  for(const mutate of [
    v=>first(v).binding.first.modelCandidateId="someone-else",
    v=>first(v).binding.second.sourceCandidateId="different-publisher-id",
    v=>first(v).binding.first.identitySha256="9".repeat(64),
    v=>first(v).binding.first.personId=v.requiredPairs[0].second.personId,
    v=>first(v).binding.comparisonContextSha256="9".repeat(64),
    v=>first(v).binding.pairSpecificationSha256="9".repeat(64),
    v=>first(v).binding.artifactSha256=v.binding.sources.F.artifactSha256,
    v=>first(v).binding.sourceId=v.binding.sources.F.id,
    v=>first(v).binding.eventId="other-event",
    v=>first(v).binding.contestId="other-contest",
    v=>first(v).binding.raceId="other-race",
    v=>first(v).binding.electionDate="2024-11-06",
  ]) {const v=partialPairComparisonFixture();mutate(v);assert.equal(readiness(v).valid,false);}
  assert.equal(readiness(incompatiblePairFixture()).display,false);
});

test("named pair denominator and candidate-mark estimand cannot become all-candidate or reversed shares", () => {
  for(const mutate of [
    v=>first(v).binding.denominator.kind="all_candidate_votes",
    v=>first(v).binding.denominator.firstPersonId=v.requiredPairs[0].second.personId,
    v=>first(v).binding.denominator.secondPersonId="third-person",
    v=>first(v).binding.estimand="final_round",
    v=>v.requiredPairs[0].second.sourceCandidateId=v.requiredPairs[0].first.sourceCandidateId,
    v=>v.requiredPairs[0].second.modelCandidateId=v.requiredPairs[0].first.modelCandidateId,
  ]) {const v=partialPairComparisonFixture();mutate(v);assert.equal(readiness(v).display,false);}
});

test("missing observations leave defined predictive summaries available but never fabricate scores", () => {
  const view=missingObservationFixture(),result=readiness(view);
  assert.equal(result.valid,true);assert.equal(result.coverage.fullyScoredPairs,0);
  assert.equal(first(view).distribution.summary.mean,.6);
  const invalid=missingObservationFixture();first(invalid).evaluation={status:"scored",score:{crps:0,weightedIntervalScore:0,absoluteMeanError:0}};
  assert.equal(readiness(invalid).valid,false);
});

test("positive undefined mass is never discarded, converted to unknown, or given a replacement summary", () => {
  for(const mutate of [
    v=>first(v).distribution.undefinedMass=1e-100,
    v=>v.rows[1].arms.count.distribution.undefinedMass=0,
    v=>v.rows[1].arms.count.distribution.summary={mean:.5,median:.5,intervals:[{level:.8,lower:0,upper:1}]},
    v=>v.rows[1].arms.F.distribution.undefinedMass=0,
    v=>v.rows[1].arms.F.distribution.summary=first(v).distribution.summary,
    v=>v.rows[1].arms.count.evaluation={status:"scored",score:{crps:0,weightedIntervalScore:0,absoluteMeanError:0}},
  ]) {const v=partialPairComparisonFixture();mutate(v);assert.equal(readiness(v).valid,false);}
  const allUndefined=partialPairComparisonFixture();allUndefined.rows[1].arms.count.distribution.undefinedMass=1;
  assert.equal(readiness(allUndefined).valid,true);
});

test("an explicitly defined zero share and zero score remain different from unavailable support", () => {
  const v=partialPairComparisonFixture();v.rows[0].observedFirstPairShare=0;
  for(const arm of Object.values(v.rows[0].arms)) {
    arm.distribution.summary={mean:0,median:0,intervals:[{level:.8,lower:0,upper:0}]};
    arm.evaluation.score={crps:0,weightedIntervalScore:0,absoluteMeanError:0};
  }
  assert.equal(readiness(v).valid,true);assert.equal(readiness(v).coverage.fullyScoredPairs,1);
});

test("a supplied absolute mean error must agree with the displayed mean and observed pair share", () => {
  const v=partialPairComparisonFixture();first(v).evaluation.score.absoluteMeanError=0;
  assert.equal(readiness(v).valid,false);
});

test("unsupported qualification claims or absent precision disclosures cannot activate the fixture", () => {
  for(const mutate of [
    v=>v.provenance.kind="actual_posterior",
    v=>v.qualification.experimental=false,
    v=>v.qualification.posterior="qualified",
    v=>v.qualification.adoption="qualified",
    v=>v.qualification.informationSets="matched",
    v=>v.qualification.explanation="",
    v=>first(v).precision=null,
    v=>first(v).precision.status="precise",
  ]) {const v=partialPairComparisonFixture();mutate(v);const result=readiness(v);assert.equal(result.display,false);assert.equal(result.productionActivated,false);}
});

test("malformed scalar values, predictive intervals and score values fail closed", () => {
  for(const mutate of [
    v=>first(v).distribution.summary.mean=NaN,
    v=>first(v).distribution.summary.median=1.1,
    v=>first(v).distribution.summary.intervals=[],
    v=>first(v).distribution.summary.intervals[0].level=0,
    v=>first(v).distribution.summary.intervals[0].lower=.9,
    v=>first(v).distribution.summary.intervals.push({...first(v).distribution.summary.intervals[0]}),
    v=>first(v).evaluation.score.crps=-1,
    v=>first(v).evaluation.score.weightedIntervalScore=Infinity,
    v=>v.rows[0].observedFirstPairShare=undefined,
  ]) {const v=partialPairComparisonFixture();mutate(v);assert.equal(readiness(v).valid,false);}
});

test("duplicate/extra pairs, missing arm slots and cross-race row borrowing cannot change coverage", () => {
  for(const mutate of [
    v=>v.requiredPairs.push(structuredClone(v.requiredPairs[0])),
    v=>v.rows.push(structuredClone(v.rows[0])),
    v=>v.rows[0].pairId="not-in-fixed-roster",
    v=>delete v.rows[0].arms.FP,
    v=>v.rows[0].arms.count=null,
    v=>v.rows[1].arms.FP=structuredClone(v.rows[0].arms.FP),
  ]) {const v=partialPairComparisonFixture();mutate(v);assert.equal(readiness(v).valid,false);}
  const v=partialPairComparisonFixture();v.rows=[];const result=readiness(v);
  assert.equal(result.valid,true);assert.equal(result.coverage.expectedPairs,3);assert.equal(result.coverage.suppliedPairs,0);
});

test("H14 cutoff and election identity need a consistent cycle and explicit UTC time", () => {
  for(const mutate of [
    v=>v.binding.cycle=2022,
    v=>v.binding.informationCutoff="2024-10-22T00:00:00",
    v=>v.binding.informationCutoff="2024-10-21T00:00:00Z",
    v=>v.binding.horizonDays=7,
    v=>v.requiredPairs[0].electionDate="2024-11-31",
  ]) {const v=partialPairComparisonFixture();mutate(v);assert.equal(readiness(v).valid,false);}
});

test("arbitrary malformed values return unavailable rather than throwing", () => {
  for(const value of [null,undefined,[],{},false,"actual-owner-output",{kind:"candidate_pair_comparison_preparation"}])
    assert.doesNotThrow(()=>assert.equal(readiness(value).display,false));
});
