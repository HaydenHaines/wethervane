import test from "node:test";
import assert from "node:assert/strict";
import { parseHouseGroupResult, parseHouseGroupView } from "../lib/house-group-result.ts";
import { groupResultFixture, refreshGroupScores, groupPolicyBinding } from "./fixtures/house-group-result.mjs";
const parse = input => parseHouseGroupResult(input,groupPolicyBinding);
test("all canonical historical rosters and separately supplied probabilities round-trip as path-free views",()=>{
 for(const year of [2018,2020,2022,2024]){
  const source=groupResultFixture(year),view=parse(source);assert.ok(view);assert.equal(view.races.length,435);assert.deepEqual(parseHouseGroupView(view),view);
  assert.equal(view.races[0].probabilities.D,.6);assert.equal(view.races[0].conditionalProbabilityMeans.D,.61);
  assert.equal(view.publicChamberControl,false);assert.equal(view.publicTipping,false);assert.equal(view.metrics,undefined);
  assert.equal(view.provenance.sourcePolicySha256,groupPolicyBinding.sourcePolicySha256);
  assert.doesNotMatch(JSON.stringify(view),/private\/|research\/|majority_probabilities|seat_scores/);
 }
});
test("reject incomplete, duplicated, wrong-cycle and falsely padded at-large districts",()=>{
 for(const change of [x=>x.races.pop(),x=>x.races[1]=structuredClone(x.races[0]),x=>x.races[0].cycle=2022,x=>x.races[0].race_id='2024:AK-00:house:regular',x=>{const r=x.races.find(r=>r.district_key==='AK-00');r.district_key='AK-01';r.race_id='2024:AK-01:house:regular';}]){
  const source=groupResultFixture();change(source);assert.equal(parse(source),null);
 }
});
test("reject malformed distributions and contradictory duplicated group probabilities",()=>{
 for(const change of [r=>r.group_win_probabilities.D=NaN,r=>r.group_win_probabilities.D=.7,r=>r.conditional_probability_means.O=-.1,r=>r.groups[0].win_probability=.61,r=>r.group_win_probabilities.X=0,r=>r.groups.pop(),r=>r.probability_method='structural_conditional_on_source_support']){
  const source=groupResultFixture();change(source.races[0]);assert.equal(parse(source),null);
 }
});
test("O source labels cannot resolve people and malformed member support is rejected",()=>{
 for(const change of [r=>r.groups[2].members[0].kind='named_candidate',r=>r.groups[2].named_singleton_resolution_eligible=true,r=>r.groups[2].individual_candidate_probability=.1,r=>r.groups[0].members=[],r=>r.groups[1].members[0].id=r.groups[0].members[0].id,r=>r.groups[0].named_singleton_resolution_eligible=false,r=>r.groups[0].individual_candidate_probability=.3]){
  const source=groupResultFixture();change(source.races[0]);assert.equal(parse(source),null);
 }
 const source=groupResultFixture();source.races[1].groups[0].members[0].id=source.races[0].groups[0].members[0].id;assert.ok(parse(source),'member identity is scoped to each race');
});
test("explicit same-party structural support keeps multi-member candidate probability unavailable",()=>{
 const source=groupResultFixture(),r=source.races.find(r=>!r.vote_share_reference_available);
 r.groups=[r.groups[0]];r.groups[0].members.push({id:'another-person',name:'Another invented nominee',kind:'named_candidate'});r.groups[0].named_singleton_resolution_eligible=false;r.groups[0].individual_candidate_probability=null;r.groups[0].win_probability=1;r.group_win_probabilities={D:1,R:0,O:0};r.conditional_probability_means={D:1,R:0,O:0};r.source_conditional_structural=true;r.probability_method='structural_conditional_on_source_support';refreshGroupScores(source);
 const view=parse(source);assert.ok(view);assert.equal(view.races.find(v=>v.id===r.race_id).groups[0].individualCandidateProbability,null);
 r.source_conditional_structural=false;assert.equal(parse(source),null);
});
test("NC09 prediction survives with uncertified truth unscored; winner names never establish binding",()=>{
 const source=groupResultFixture(2018),view=parse(source),race=view.races.find(r=>r.districtKey==='NC-09');assert.equal(race.observed.status,'unavailable');assert.equal(race.probabilities.R,.3);assert.equal(view.raceScores[0].n,434);
 const raw=source.races.find(r=>r.district_key==='NC-09');raw.group_score_eligible=true;assert.equal(parse(source),null);
 const other=groupResultFixture();other.races[0].observed_result.winner_names=['Not a matching member name'];assert.ok(parse(other));other.races[0].observed_result.member_id=other.races[0].groups[1].members[0].id;assert.equal(parse(other),null);other.races[0].observed_result.member_id=null;other.races[0].observed_result.group_id='X';assert.equal(parse(other),null);
});
test("score summaries respect probability bounds and exact explicit masks",()=>{
 for(const change of [x=>x.metrics.race_scores.all_scored.categorical_brier=.1,x=>x.metrics.race_scores.legacy_subset.n++,x=>x.metrics.race_scores.newly_included.binary_D_brier=NaN,x=>x.metrics.race_scores.competitive.log_score_floor=0,x=>x.metrics.race_scores.competitive.log_score_with_declared_floor=Infinity,x=>x.metrics.race_scores.newly_included_competitive.n=100,x=>x.metrics.diagnostics.group_win_passed=false,x=>x.metrics.diagnostics.max_group_win_mcse=.01]){
  const source=groupResultFixture();change(source);assert.equal(parse(source),null);
 }
});
test("wrong provenance, unsupported configuration and public control claims fail closed",()=>{
 for(const change of [x=>x.schema='HouseResult',x=>x.configuration='arbitrary',x=>x.id+='changed',x=>x.public_chamber_control_qualified=true,x=>x.no_within_O_candidate_allocation=false,x=>x.tipping=true,x=>x.source_panel.sha256='bad']){
  const source=groupResultFixture();change(source);assert.equal(parse(source),null);
 }
 assert.equal(parseHouseGroupResult(groupResultFixture(),{sourcePolicySha256:'bad'}),null);assert.equal(parseHouseGroupResult(groupResultFixture()),null);
});
test("browser API views repeat full validation and strip unknown claims",()=>{
 const good=parse(groupResultFixture());good.arbitraryLocalPath='/private/raw';assert.equal(parseHouseGroupView(good).arbitraryLocalPath,undefined);
 for(const change of [v=>v.races[0].probabilities.D=.9,v=>v.races[0].groups[0].individualCandidateProbability=.8,v=>v.publicChamberControl=true,v=>v.publicTipping=true,v=>v.raceScores[0].n=1,v=>v.raceDiagnostics.maxGroupWinRhat=2,v=>v.provenance.predictionSha256='bad',v=>v.legacyRunId='wrong']){
  const view=structuredClone(good);change(view);assert.equal(parseHouseGroupView(view),null);
 }
 const snapshot=parseHouseGroupView(good);good.races[0].groups[0].members[0].name='Changed';assert.notEqual(snapshot.races[0].groups[0].members[0].name,'Changed');
});

test("empirical endpoint is not structural certainty and zero true outcomes retain declared log floor",()=>{
 const source=groupResultFixture(),r=source.races[0];r.group_win_probabilities={D:0,R:.9,O:.1};r.groups.forEach(g=>{g.win_probability=r.group_win_probabilities[g.id];g.individual_candidate_probability=g.named_singleton_resolution_eligible?g.win_probability:null;});refreshGroupScores(source);
 const view=parse(source);assert.ok(view);assert.equal(view.races[0].structural,false);assert.equal(view.races[0].probabilityMethod,'modeled_joint_draw_frequency');assert.equal(view.raceScores[0].zeroEmpiricalTrueOutcomes,1);assert.ok(Number.isFinite(view.raceScores[0].logScoreWithDeclaredFloor));
});
