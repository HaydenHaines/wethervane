import assert from 'node:assert/strict';
import test from 'node:test';
import {mkdtempSync,rmSync,readFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {ElectionNightArtifactStore,loadElectionNightEvaluationLock,loadElectionNightResultLedger} from '../lib/election-night-artifacts.ts';
import {prepareSenateResultRevision} from '../lib/election-night-senate-admission.ts';
import {senateResultSelector} from '../lib/election-night-senate-targets.ts';
import {SENATE_PAIR_RESULT_POLICY} from '../lib/election-night-senate-results.ts';
import {evaluateElectionNightForecast} from '../lib/election-night-evaluation.ts';

// The operator policy and candidate identifiers are real frozen fixtures. ALL
// forecast probabilities, result reports, observations, review/admission decisions
// and trust roots below are INVENTED. No actual election performance is measured.
const unique = pins => [...new Map(pins.map(p=>[p.sha256,p])).values()];
const h = c=>c.repeat(64);
function setup(t, {mutateCohort} = {}) {
 const root=mkdtempSync(join(tmpdir(),'senate-admission-'));t.after(()=>rmSync(root,{recursive:true,force:true}));
 const store=new ElectionNightArtifactStore(root);
 const resultPolicy=store.put(readFileSync(new URL('./fixtures/election-night/senate-policy-v1.json',import.meta.url)));
 const mappingPin=store.put(readFileSync(new URL('./fixtures/election-night/senate-mapping-v1.json',import.meta.url)));
 assert.equal(resultPolicy.sha256,SENATE_PAIR_RESULT_POLICY.policySha256);assert.equal(mappingPin.sha256,SENATE_PAIR_RESULT_POLICY.mappingSha256);
 const rows=store.readJson(mappingPin).races.map(r=>({raceId:r.race_id,state:r.state,seatId:r.seat_id,eventKind:r.event_kind,
  generalDate:r.scheduled_general_event_date,runoffDate:r.scheduled_runoff_event_date,firstCandidateId:r.first_candidate_id,
  secondCandidateId:r.second_candidate_id,operatorFamily:r.operator_family}));
 const reviews=Object.fromEntries(['comparison','cohort','rubric','resultPolicy','projection'].map(role=>[role,store.putJson({schema:'INVENTED-review',role})]));
 const cohortValue={policySha256:resultPolicy.sha256,mappingSha256:mappingPin.sha256,senateObservationSelectors:rows.map(row=>({policyRow:row,selector:senateResultSelector(row),semanticRole:'predeclared_observation_selection_rule'}))};
 mutateCohort?.(cohortValue);
 const cohort=store.putJson(cohortValue),rubric=store.putJson({schema:'INVENTED-rubric'}),denominator='Share of votes for the two named principal contenders';
 const source=side=>store.put(Buffer.from(`INVENTED ${side} source`));
 const sources={model:source('model'),baseline:source('baseline')};
 const bindings=Object.fromEntries(Object.entries(sources).map(([side,p])=>[side,{chamber:'senate',runId:`INVENTED-${side}`,resultSha256:p.sha256,
  nativeForecastSha256:h(side==='model'?'a':'b'),targetId:'INVENTED-target',lawId:'INVENTED-law',issuedAt:'2026-11-01T13:00:00Z',cutoffAt:'2026-11-01T12:00:00Z',horizonCalendarDays:2}]));
 const races=rows.map(row=>({id:row.raceId,state:row.state,seatId:row.seatId,electionLabel:'INVENTED observation experiment',ruleFamily:null,notes:[],sourceUrls:[],
  outcomes:[row.firstCandidateId,row.secondCandidateId].map((id,i)=>({id,name:`INVENTED source contender ${i}`,party:i?'R':'D',kind:'candidate',winProbability:i?.4:.6,members:[{id,name:`INVENTED contender ${i}`,party:i?'R':'D',sourceUrls:[]}]})),
  pair:{firstCandidateId:row.firstCandidateId,secondCandidateId:row.secondCandidateId,firstCandidateName:'INVENTED first',secondCandidateName:'INVENTED second',denominator,
   units:'percentage_points',orientation:'first_minus_second',meanMarginPp:2,medianMarginPp:2,range90:{lower:-10,upper:10},firstExceedsHalf:.6,winnerRelationship:'same_event',qualifications:[]}}));
 const forecast=side=>({schema:'ElectionNightForecastV1',editionStatus:'dated_snapshot',chamber:'senate',run:{id:bindings[side].runId,artifactSha256:sources[side].sha256,
  issuedAt:bindings[side].issuedAt,cutoffAt:bindings[side].cutoffAt,targetId:bindings[side].targetId,lawId:bindings[side].lawId},sourceHashes:{forecastSha256:bindings[side].nativeForecastSha256},races});
 const targets=rows.map(row=>({...senateResultSelector(row),raceId:row.raceId,firstCandidateId:row.firstCandidateId,secondCandidateId:row.secondCandidateId,
  denominator,estimand:'senate_principal_pair_lead',outcomeIds:[row.firstCandidateId,row.secondCandidateId]}));
 const contract={schema:'ElectionNightRubricComparisonV1',id:'INVENTED-comparison',sha256:h('c'),model:bindings.model,baseline:bindings.baseline,
  cohortDefinition:{id:'INVENTED-cohort',sha256:cohort.sha256},resultPolicy:{id:'reviewed-rule-fixture',sha256:resultPolicy.sha256},
  rubricDecisionSha256:rubric.sha256,comparisonReview:{id:'INVENTED-comparison-review',sha256:reviews.comparison.sha256},winnerCohort:targets};
 const protocol=side=>({schema:'ElectionNightEvaluationProtocolV1',id:`INVENTED-${side}-protocol`,forecast:{chamber:'senate',runId:bindings[side].runId,
  artifactSha256:sources[side].sha256,issuedAt:bindings[side].issuedAt,cutoffAt:bindings[side].cutoffAt},reviewedDefinition:{sha256:reviews.projection.sha256,url:'https://example.invalid/INVENTED-review'},
  races:rows.map(row=>({raceId:row.raceId,winner:{...senateResultSelector(row),estimand:'conditional_principal_pair_winner',outcomeIds:[row.firstCandidateId,row.secondCandidateId],minimumStatus:'certified'},
   margin:{...senateResultSelector(row),estimand:'named_pair_margin',firstCandidateId:row.firstCandidateId,secondCandidateId:row.secondCandidateId,denominator,units:'percentage_points',orientation:'first_minus_second',pointRule:'mean',minimumStatus:'certified'}}))});
 const artifacts={modelSource:sources.model,baselineSource:sources.baseline,modelForecast:store.putJson(forecast('model')),baselineForecast:store.putJson(forecast('baseline')),
  modelProtocol:store.putJson(protocol('model')),baselineProtocol:store.putJson(protocol('baseline')),comparisonContract:store.putJson(contract),cohort,rubric,resultPolicy};
 function authority(subject,scope,lockSha256,reviewPins,extras){
  const trustedAdmission=store.putJson({schema:'ElectionNightArtifactAdmissionV1',id:'INVENTED-admission',scope,subject,evaluationLockSha256:lockSha256,reviews:unique(reviewPins)});
  return {trustedAdmission,trustedPublication:store.putJson({schema:'ElectionNightArtifactPublicationV1',id:'INVENTED-publication',artifacts:unique([subject,trustedAdmission,...reviewPins,...extras])})};
 }
 const manifest={schema:'ElectionNightEvaluationLockV1',id:'INVENTED-lock',frozenAt:'2026-11-01T14:00:00Z',forecastBindings:bindings,artifacts,reviews,
  projections:Object.fromEntries(['model','baseline'].map(side=>[side,{sourceSha256:sources[side].sha256,projectionSha256:artifacts[`${side}Forecast`].sha256,reviewSha256:reviews.projection.sha256}]))};
 const lockPin=store.putJson(manifest),lock=loadElectionNightEvaluationLock(store,{lock:lockPin,...authority(lockPin,'evaluation_lock',lockPin.sha256,Object.values(reviews),Object.values(artifacts))});
 function ledger(revisions=[]){
  const value={schema:'ElectionNightResultLedgerV1',id:'INVENTED-ledger',evaluationLockSha256:lock.lockSha256,chosenHeadSha256:revisions.at(-1)?.sha256??null,revisions};
  const ledgerPin=store.putJson(value),extras=revisions.flatMap(pin=>{const r=store.readJson(pin);return [pin,r.record,r.source,...r.evidence]});
  return loadElectionNightResultLedger(store,{evaluationLock:lock,ledger:ledgerPin,...authority(ledgerPin,'result_ledger',lock.lockSha256,[reviews.resultPolicy],extras)});
 }
 function request({state='AL',first=600,second=400,change,changeAdmission,changeReview,prior=ledger(),revision='1',extraWinner=false}={}){
  const row=rows.find(r=>r.state===state),source=store.put(Buffer.from(`INVENTED report,${state},first=${first},second=${second},revision=${revision}`));
  const observation={official:true,raceId:row.raceId,seatId:row.seatId,eventKind:row.eventKind,replacementOrIdentityChange:false,relevantRuleChange:false,
   stage:'general',eventDate:row.generalDate,countsByNativeCandidateId:{[row.firstCandidateId]:first,[row.secondCandidateId]:second},certified:true,contestFinal:true,pendingRecount:false,pendingResultDispute:false};
  if(row.operatorFamily==='rcv_locked_final_two'){observation.officialObservedFinalRound=true;observation.continuingCandidateIds=[row.firstCandidateId,row.secondCandidateId];}
  if(state==='GA')observation.runoffRequired=false;
  if(extraWinner)observation.officialOfficeWinnerId='INVENTED:outside-pair';
  const extraction={schema:'SenateSourceExtractionV1',id:`INVENTED-extraction-${revision}`,preparedAt:'2026-11-05T02:00:00Z',evaluationLockSha256:lock.lockSha256,
   source:{artifact:source,url:'https://example.invalid/INVENTED-report',authority:'INVENTED authority',publishedAt:'2026-11-04T00:00:00Z',retrievedAt:'2026-11-05T01:00:00Z'},supportingArtifacts:[],
   sourceContest:{id:'INVENTED-contest',locator:'INVENTED table1',roundId:'INVENTED-round',roundLabel:'INVENTED certified actual round',stage:observation.stage,eventDate:observation.eventDate,
    officialObservedFinalRound:observation.officialObservedFinalRound??null,continuingCandidateIds:observation.continuingCandidateIds??null,runoffRequired:observation.runoffRequired??null,runoffCandidateIds:null},
   candidateCrosswalk:[row.firstCandidateId,row.secondCandidateId,...(extraWinner?['INVENTED:outside-pair']:[])].map((id,i)=>({candidateKey:id,sourceCandidateId:`INVENTED-${i}`,sourceName:`INVENTED row ${i}`,locator:`row${i}`})),
   fieldLocations:['contest','round','finality','firstCandidateCount','secondCandidateCount',...(extraWinner?['officialOfficeWinner']:[])].map(field=>({field,locator:`INVENTED ${field}`})),observation};
  change?.(extraction,row);const extractionPin=store.putJson(extraction);
  const obsReviews=Object.fromEntries(['source','candidateIdentity','eventRound'].map(role=>{const v={schema:'SenateObservationReviewV1',id:`INVENTED-${role}`,role,decision:'admit_extraction',reviewedAt:'2026-11-05T03:00:00Z',reviewer:'INVENTED independent reviewer',evaluationLockSha256:lock.lockSha256,
   extractionSha256:extractionPin.sha256,sourceSha256:source.sha256,supportingArtifactSha256:[],policySha256:resultPolicy.sha256,mappingSha256:mappingPin.sha256};changeReview?.(v);return [role,store.putJson(v)]}));
  const a={schema:'SenateObservationAdmissionV1',id:'INVENTED-observation-admission',evaluationLockSha256:lock.lockSha256,extraction:extractionPin,source,mapping:mappingPin,supportingArtifacts:[],policySha256:resultPolicy.sha256,mappingSha256:mappingPin.sha256,reviews:obsReviews};changeAdmission?.(a);
  const trustedAdmission=store.putJson(a),trustedPublication=store.putJson({schema:'ElectionNightArtifactPublicationV1',id:'INVENTED-observation-publication',artifacts:unique([extractionPin,source,mappingPin,trustedAdmission,...Object.values(obsReviews)])});
  return {extraction:extractionPin,trustedAdmission,trustedPublication,priorLedger:prior,recordRevision:revision,revisionId:`INVENTED-revision-${state}-${revision}`};
 }
 return {store,lock,rows,request,ledger,manifest,authority,cohortValue,mappingPin};
}

test('admitted invented source produces a revision, not a public head, and both evaluators use its exact margin',t=>{
 const f=setup(t),request=f.request({extraWinner:true}),p=prepareSenateResultRevision(f.store,f.lock,request);
 assert.equal(p.publicHeadChanged,false);assert.equal(p.resultLedgerAdmissionIssued,false);assert.equal(request.priorLedger.records.length,0);
 const record=f.store.readJson(p.record),projection=f.store.readJson(p.operatorProjection);
 assert.equal(record.margin.marginPp,20);assert.equal(record.margin.status,'certified');assert.equal(projection.officeWinnerOutsideLockedPair,true);
 assert.equal(projection.legalWinnerGradeSupported,false);assert.notEqual(record.winner.candidateId,'INVENTED:outside-pair');
 const ledger=f.ledger([p.revision]);
 for(const side of ['model','baseline']){const result=evaluateElectionNightForecast(f.lock[`${side}Forecast`],f.lock[`${side}Protocol`],ledger.records);assert.equal(result.rows.find(r=>r.raceId===record.raceId).margin.score.observedMarginPp,20);}
});

test('a correction retains the previous report, record and global head lineage',t=>{
 const f=setup(t),one=prepareSenateResultRevision(f.store,f.lock,f.request()),prior=f.ledger([one.revision]);
 const two=prepareSenateResultRevision(f.store,f.lock,f.request({first:400,second:600,prior,revision:'2'}));
 const revision=f.store.readJson(two.revision);assert.equal(revision.predecessorSha256,one.revision.sha256);assert.equal(revision.supersedesRecordSha256,one.record.sha256);
 const latest=f.ledger([one.revision,two.revision]);assert.equal(latest.records[0].margin.marginPp,-20);assert.equal(latest.history.length,2);assert.equal(f.store.readJson(one.record).margin.marginPp,20);
 assert.throws(()=>prepareSenateResultRevision(f.store,f.lock,f.request({prior:latest,revision:'1'})),/already exists/);
});

test('ties preserve zero margin without a binary win score, and unresolved finality cannot score',t=>{
 const f=setup(t);for(const [name,opts] of [['tie',{first:500,second:500}],['recount',{change:e=>{e.observation.pendingRecount=true;}}]]){
  const p=prepareSenateResultRevision(f.store,f.lock,f.request(opts)),r=f.store.readJson(p.record);
  const score=evaluateElectionNightForecast(f.lock.modelForecast,f.lock.modelProtocol,f.ledger([p.revision]).records).rows.find(x=>x.raceId===r.raceId);
  if(name==='tie'){assert.equal(r.margin.marginPp,0);assert.equal(r.margin.status,'certified');assert.equal(score.winner.score,null);assert.ok(score.margin.score);}
  else{assert.equal(r.margin.status,'provisional');assert.equal(score.margin.score,null);assert.equal(score.winner.score,null);}
 }
});

test('altered source/extraction/admission/review bytes cannot reuse an old decision',t=>{
 const f=setup(t),req=f.request();const changed=f.store.putJson({...f.store.readJson(req.extraction),id:'INVENTED changed'});
 assert.throws(()=>prepareSenateResultRevision(f.store,f.lock,{...req,extraction:changed}),/absent|does not bind/);
 assert.throws(()=>prepareSenateResultRevision(f.store,f.lock,f.request({changeReview:r=>{r.extractionSha256=h('f');}})),/review does not admit/);
 assert.throws(()=>prepareSenateResultRevision(f.store,f.lock,f.request({changeAdmission:a=>{a.mapping={...a.mapping,sha256:h('e')};}})),/mapping bytes/);
 assert.throws(()=>prepareSenateResultRevision(f.store,structuredClone(f.lock),req),/authenticated|loaded|loader/i);
});

test('report round, candidate identifiers and winner diagnostic must have consistent evidence',t=>{
 const f=setup(t);
 for(const change of [e=>{e.sourceContest.eventDate='2026-11-02';},e=>{e.sourceContest.stage='runoff';},e=>{e.candidateCrosswalk[1].sourceCandidateId=e.candidateCrosswalk[0].sourceCandidateId;},e=>{e.observation.officialOfficeWinnerId='INVENTED:unmapped';},e=>{e.fieldLocations=e.fieldLocations.filter(l=>l.field!=='firstCandidateCount');}])
  assert.throws(()=>prepareSenateResultRevision(f.store,f.lock,f.request({change})),/contradicts|share one|diagnostic|source location/);
});

test('conditional state families cannot be relabeled plurality and observed RCV/GA rules remain active',t=>{
 const f=setup(t);
 for(const state of ['AK','ME','GA'])assert.throws(()=>senateResultSelector({...f.rows.find(r=>r.state===state),operatorFamily:'plurality_named_pair'}),/reviewed/);
 const rcv=prepareSenateResultRevision(f.store,f.lock,f.request({state:'ME'}));assert.equal(f.store.readJson(rcv.record).margin.status,'certified');
 const pending=prepareSenateResultRevision(f.store,f.lock,f.request({state:'GA',change:e=>{e.observation.runoffRequired=true;e.sourceContest.runoffRequired=true;}}));
 assert.equal(f.store.readJson(pending.record).margin.status,'unresolved');assert.equal(f.store.readJson(pending.operatorProjection).status,'pending_runoff');
 const wrong=prepareSenateResultRevision(f.store,f.lock,f.request({state:'ME',change:e=>{e.observation.officialObservedFinalRound=false;e.sourceContest.officialObservedFinalRound=false;}}));
 assert.equal(f.store.readJson(wrong.record).margin.status,'unresolved');
});

test('an independently pinned but internally altered cohort cannot impersonate the raw owner mapping',t=>{
 const f=setup(t,{mutateCohort:c=>{c.senateObservationSelectors.find(e=>e.policyRow.state==='AL').policyRow.eventKind='special';}});
 assert.throws(()=>prepareSenateResultRevision(f.store,f.lock,f.request()),/authenticated owner mapping/);
});

test('a certified report cannot be dated before the declared runoff',t=>{
 const f=setup(t),change=e=>{e.observation.stage='runoff';e.observation.eventDate='2026-12-01';e.observation.runoffCandidateIds=[...e.candidateCrosswalk.map(c=>c.candidateKey)];
  e.sourceContest.stage='runoff';e.sourceContest.eventDate='2026-12-01';e.sourceContest.runoffCandidateIds=e.observation.runoffCandidateIds;};
 assert.throws(()=>prepareSenateResultRevision(f.store,f.lock,f.request({state:'GA',change})),/cannot precede/);
 const req=f.request({state:'GA',change:e=>{change(e);e.source.publishedAt='2026-12-02T00:00:00Z';e.source.retrievedAt='2026-12-02T01:00:00Z';e.preparedAt='2026-12-02T02:00:00Z';},changeReview:r=>{r.reviewedAt='2026-12-02T03:00:00Z';}});
 const prepared=prepareSenateResultRevision(f.store,f.lock,req);assert.equal(f.store.readJson(prepared.record).margin.status,'certified');
});
