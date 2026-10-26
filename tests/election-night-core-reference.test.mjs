import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { toSenateElectionNightForecast } from '../lib/election-night-forecast.ts';
import { matchElectionNightCoreReference, supportsElectionNightCoreReference } from '../lib/election-night-core-reference.ts';
const root = new URL('../',import.meta.url), read = p => JSON.parse(readFileSync(new URL(p,root)));
const registry = read('public/data/explorer/registry.json'), catalog = read('public/data/election-night/core-references.json');
const publication = read('data/election-night-core-reference-publication.json');
const forecast = id => { const e=registry.runs.find(e=>e.id===id);return toSenateElectionNightForecast(read('public'+e.href),e); };
const near = (a,b) => assert.ok(Math.abs(a-b)<1e-10, `${a} vs ${b}`);
function freeze(v) { if(v&&typeof v==='object'){Object.values(v).forEach(freeze);Object.freeze(v);}return v; }

test('publication authenticates separate actual F/FP fixed32 bytes with original source pins',()=>{
  assert.equal(publication.href,'/data/election-night/core-references.json');
  assert.equal(publication.sha256,createHash('sha256').update(readFileSync(new URL('public'+publication.href,root))).digest('hex'));
  assert.deepEqual(publication.editions.map(e=>e.runId),['current-fundamentals','current-polling']);
  assert.equal(catalog.entries.length,2);
  for(const e of catalog.entries){ assert.equal(e.sourcePins.reviewSha256,'b82bc61967c3af40274178d79f65ba088e3f5313d32233ec4fbc0c9ac3648c8d');
    assert.equal(e.sourcePins.failedV1EvidenceSha256,'d73ed9b406ba9c1184bca648eaf830a9f8c96ab7f6809e5165d52591554d2550'); }
});

test('all16 actual saved choices only match exact F/FP editions; original35 display rows survive',()=>{
  let matched=0;
  for(const e of registry.runs.filter(e=>e.year===2026)){
    const f=forecast(e.id), before=JSON.stringify(f), r=matchElectionNightCoreReference(f,catalog), eligible=['current-fundamentals','current-polling'].includes(e.id);
    assert.equal(supportsElectionNightCoreReference(f),eligible,e.id); assert.equal(r!==null,eligible,e.id);assert.equal(JSON.stringify(f),before);
    assert.equal(f.races.length,35);if(!r)continue;matched++;
    const native=catalog.entries.find(e=>e.runId===f.run.id).reference;
    assert.equal(r.perRace.length,32);assert.equal(r.cohortLabel,'Primary grade: 32 plurality candidate pairs');
    assert.deepEqual(r.perRace.map(r=>r.raceId),f.races.filter(r=>!r.id.match(/^2026:(AK|GA|ME):/)).map(r=>r.id));
    assert.equal(r.drawIdentity.rowCount,e.id==='current-fundamentals'?100000:40000);
    assert.deepEqual(r.drawIdentity.shape,e.id==='current-fundamentals'?[100000]:[4,10000]);
    assert.equal(r.drawIdentity.weightPerRow,1/r.drawIdentity.rowCount);assert.equal(r.drawIdentity.crossBankPairing,false);assert.equal(r.drawIdentity.resampling,false);
    near(r.summary.meanAbsoluteMarginError.mean,native.per_race.reduce((sum,r)=>sum+r.expected_absolute_margin_error_pp,0)/32);
    near(r.summary.modalWinnerMissCount.mean,r.perRace.reduce((sum,r)=>sum+1-Math.max(r.firstPairWinProbability,1-r.firstPairWinProbability),0));
    for(const [key,nativeKey] of [['meanAbsoluteMarginError','mean_absolute_margin_error_pp'],['modalWinnerMissCount','modal_pair_winner_miss_count'],['interval90CoverageCount','interval_90_coverage_count']]){
      assert.deepEqual(r.summary[key].range90,{lower:native.distributions[nativeKey].quantiles.p05,upper:native.distributions[nativeKey].quantiles.p95});
    }
    assert.equal(r.evaluationLock,false);assert.equal(r.observedResultsConsumed,false);assert.match(r.interpretation,/different dates and independent banks/);
  }
  assert.equal(matched,2);
});

test('stale editions, changed IDs/pairs/denominators and native/projection drift never borrow a reference',()=>{
  const changes=[f=>f.chamber='house',f=>f.run.year=2022,f=>f.editionStatus='historical',f=>f.run.artifactSha256='9'.repeat(64),
    f=>f.run.targetId='wrong',f=>f.run.lawId='wrong',f=>f.run.issuedAt='2026-10-26T16:00:00Z',f=>f.run.cutoffAt='2026-10-20T15:21:23Z',
    f=>f.sourceHashes.forecastSha256='9'.repeat(64),f=>f.sourceHashes.jointDrawsSha256='9'.repeat(64),f=>f.sourceHashes.receiptSha256='9'.repeat(64),
    f=>f.races.reverse(),f=>f.races[1].pair.firstCandidateId='another-person',f=>f.races[1].pair.denominator='all votes',
    f=>f.races[1].pair.meanMarginPp+=.01,f=>f.races[1].pair.range90.lower+=.01,f=>f.races[1].outcomes.reverse(),
    f=>f.races[1].modalPick.outcomeId=f.races[1].outcomes[0].id];
  for(const change of changes){const f=forecast('current-polling');change(f);assert.equal(matchElectionNightCoreReference(f,catalog),null,String(change));}
});

test('catalog tampering, wrong cohort/weights/review, invalid count quantiles and paired-bank claims fail closed',()=>{
  const changes=[c=>c.entries.push(structuredClone(c.entries[1])),c=>c.entries[1].resultSha256='9'.repeat(64),
    c=>c.entries[1].sourcePins.referenceSha256=c.entries[0].sourcePins.referenceSha256,c=>c.entries[1].sourcePins.reviewSha256='9'.repeat(64),
    c=>c.entries[1].sourcePins.policySha256='9'.repeat(64),c=>c.entries[1].reference.cohort.pop(),c=>c.entries[1].reference.per_race.reverse(),
    c=>c.entries[1].reference.cohort[0].first_candidate.id='wrong-person',c=>c.entries[1].reference.cohort[0].denominator='all votes',
    c=>c.entries[1].reference.common_target_cohort_sha256='9'.repeat(64),c=>c.entries[1].reference.draw_identity.weight_per_row=1/100000,
    c=>c.entries[1].reference.draw_identity.cross_bank_pairing=true,c=>c.entries[1].reference.draw_identity.resampling=true,
    c=>c.entries[1].reference.draw_identity.original_race_axis_indices.reverse(),c=>c.entries[1].reference.draw_identity.shape=[40000],
    c=>c.entries[1].reference.race_axis_binding='embedded FP race ids',c=>c.entries[1].reference.per_race[0].modal_pair_winner_id='wrong-person',
    c=>c.entries[1].reference.distributions.modal_pair_winner_miss_count.quantiles.p05=1.5,
    c=>c.entries[1].reference.distributions.interval_90_coverage_count.quantiles.p95=33,
    c=>c.entries[1].reference.distributions.mean_absolute_margin_error_pp.mean+=1,c=>c.entries[1].reference.qualifications=[],
    c=>c.entries[1].reference.excluded_from_this_reference.pop()];
  for(const change of changes){const c=structuredClone(catalog);change(c);assert.equal(matchElectionNightCoreReference(forecast('current-polling'),c),null,String(change));}
  for(const bad of [null,{},[],{schemaVersion:1,entries:[null]}])assert.equal(matchElectionNightCoreReference(forecast('current-polling'),bad),null);
});

test('frozen inputs remain unchanged and output metrics/arrays share no mutable references',()=>{
  const f=freeze(forecast('current-fundamentals')), c=freeze(structuredClone(catalog)), before=JSON.stringify({f,c});
  const r=matchElectionNightCoreReference(f,c);assert.ok(r);r.perRace[0].range90.lower=0;r.sourcePins.reviewSha256='changed';r.excludedRaceIds.pop();r.drawIdentity.shape[0]=0;
  assert.equal(JSON.stringify({f,c}),before);
});
