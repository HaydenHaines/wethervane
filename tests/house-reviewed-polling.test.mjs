import test from "node:test";
import assert from "node:assert/strict";
import {createHash} from "node:crypto";
import {readFile,writeFile,mkdir,mkdtemp,rm,symlink,access} from "node:fs/promises";
import {join,resolve} from "node:path";
import {tmpdir} from "node:os";
import {houseReviewedCatalog,parseHouseReviewedCatalog,parseHouseReviewedPublication,canonicalGroupJson,
  reviewedDigest} from "../lib/house-reviewed-catalog.ts";
import {parseHousePollingGroupResult,parseHousePollingGroupView,HOUSE_POLLING_POLICY_SHA,
  HOUSE_POLLING_INFORMATION_MODE} from "../lib/house-polling-result.ts";
import {parseHousePollingShares} from "../lib/house-polling-shares.ts";
import {parseHouseGroupResult} from "../lib/house-group-result.ts";
import {parseHouseResult} from "../lib/house-result.ts";
import {loadHouseReviewedPublication,loadHousePollingSharePublication} from "../lib/house-reviewed-publication.ts";
import {loadHouseGroupPublication} from "../lib/house-group-publication.ts";
import {createHouseGroupSavedView as createOldSaved} from "../lib/house-group-saved-views.ts";
import {createHouseGroupSavedView,verifyHouseGroupSavedView,parseHouseGroupSavedViews,
  serializeHouseGroupSavedViews,houseGroupSavedViewAvailability} from "../lib/house-reviewed-saved-views.ts";
import {verifyHouseGroupComparison,sharedHouseGroupScoreStrata,houseGroupScoreMask} from "../lib/house-reviewed-comparison.ts";
import {houseGroupModelSlices} from "../lib/house-reviewed-model-slices.ts";
import {reviewedGroupSelection,groupSelectionUrl,groupSelectionForResult} from "../lib/house-reviewed-selection.ts";
import {loadHouseReviewedGroup,loadHousePollingShares} from "../lib/house-reviewed-client.ts";
import {GET as groupGET} from "../app/api/chambers/house/groups/route.ts";
import {GET as shareGET} from "../app/api/chambers/house/groups/shares/route.ts";
import {importHousePolling} from "../scripts/import_house_polling.mjs";

const root=resolve("."),groups=join(root,"data/house-groups"),geometry=join(root,"data/house"),polling=join(root,"data/house-polling");
const read=async path=>JSON.parse(await readFile(path,"utf8"));
const digest=bytes=>createHash("sha256").update(bytes).digest("hex");
const catalog=houseReviewedCatalog();
const native=await read("tests/fixtures/house-polling-native.json");
const inspect=value=>parseHousePollingGroupResult(value,{sourcePolicySha256:HOUSE_POLLING_POLICY_SHA});
const load=id=>loadHouseReviewedPublication(groups,geometry,id);

test("frozen old catalog/default and old native validators remain distinct",async()=>{
  assert.equal(catalog.runs.length,32);
  assert.equal(catalog.initialRunId,"house_history_refreshed_v1-2024-h14-full435-v1");
  assert.equal((await load()).result.id,catalog.initialRunId);
  assert.equal((await load()).entry.polling,false);
  for(const cycle of [2018,2020,2022,2024]){
    assert.equal(catalog.runs.filter(r=>r.cycle===cycle&&r.polling).length,4);
    assert.equal(catalog.runs.filter(r=>r.cycle===cycle&&!r.polling).length,4);
  }
  assert.equal(parseHouseGroupResult(native,{sourcePolicySha256:HOUSE_POLLING_POLICY_SHA}),null);
  const parsed=inspect(native);assert.ok(parsed);assert.equal(parsed.polling,true);
  const shares=await loadHousePollingSharePublication(polling,parsed.conditionalShareRunId);
  assert.equal(parseHouseResult(shares.result),null,"new information mode never enters the old share parser");
  assert.ok(parseHousePollingShares(shares.result));
});

test("new native parser rejects altered polling, capability, probability, score and authority claims",()=>{
  assert.ok(inspect(native));
  for(const change of [
    x=>x.schema="HouseFullRosterReviewedResultV1",x=>x.polling=false,x=>x.current2026=true,
    x=>x.public_chamber_control_qualified=true,x=>x.public_tipping_qualified=true,
    x=>x.target_id="house_aggregate_group_winner_completion_v1",x=>x.cycle=2026,
    x=>x.source_policy_pin.sha256="0".repeat(64),x=>x.audit.sha256="0".repeat(64),
    x=>x.source_admission.sha256="0".repeat(64),x=>x.polling_coverage.polls++,
    x=>x.polling_coverage.horizon_days=7,x=>x.races[0].poll_count=-1,
    x=>x.races[0].poll_count=.5,x=>x.races[0].polling_method="direct_D_R_poll_likelihood",
    x=>x.races[0].group_win_probabilities.D=.9,x=>x.races[0].groups[1].individual_candidate_probability=.1,
    x=>x.races[0].winner_method="fundamentals_pair_with_categorical_outside_mass",
    x=>x.metrics.race_scores[0].categorical_brier+=.01,x=>x.metrics.race_scores[0].n--,
    x=>x.metrics.diagnostics.group_win_passed=false,x=>x.metrics.diagnostics.seat_mean_precision_passed=true,
  ]){const value=structuredClone(native);change(value);assert.equal(inspect(value),null);}
});

test("all32 API views roundtrip, preserve exact F DTOs, maps, counts and new share bindings",async()=>{
  for(const entry of catalog.runs){
    const publication=await load(entry.id);
    assert.ok(await parseHouseReviewedPublication(structuredClone(publication)));
    assert.equal(await reviewedDigest(publication.result),entry.viewSha256);
    assert.equal(publication.geography.geojson.features.length,435);
    assert.equal(publication.result.races.length,435);
    if(!entry.polling){
      const original=await loadHouseGroupPublication(groups,geometry,entry.id);
      assert.deepEqual(publication.result,original.result,"old DTO fields and values are unchanged");
      assert.equal(entry.viewSha256,original.entry.viewSha256);
      assert.equal(entry.resultSha256,original.entry.resultSha256);
    }else{
      const r=publication.result;
      assert.deepEqual(parseHousePollingGroupView(r),r);
      assert.equal(r.informationMode,HOUSE_POLLING_INFORMATION_MODE);
      assert.equal(r.races.filter(x=>x.pollCount>0).length,r.pollingCoverage.polledDistricts);
      assert.equal(r.races.reduce((n,x)=>n+x.pollCount,0),r.pollingCoverage.polls);
      assert.equal(r.raceScores[0].logScoreFloor,1e-6,"exact polling score floor differs from legacy F");
      assert.equal(r.publicChamberControl,false);assert.equal(r.publicTipping,false);
      const share=await loadHousePollingSharePublication(polling,entry.conditionalShareRunId);
      assert.equal(share.resultSha256,entry.conditionalShareSha256);
      assert.equal(await reviewedDigest(share.result),entry.conditionalShareContentSha256);
      assert.deepEqual(share.result.races.map(r=>r.id).sort(),r.races.filter(r=>r.voteShareReferenceAvailable).map(r=>r.id).sort());
    }
  }
});

test("every reviewed F/FP pair verifies explicit support; cross-model pairs fail closed",async()=>{
  for(const pair of catalog.matchedPairs){
    const [a,b]=await Promise.all([load(pair.baselineRunId),load(pair.pollingRunId)]);
    assert.equal((await verifyHouseGroupComparison(a,b)).status,"comparable");
    assert.equal((await verifyHouseGroupComparison(b,a)).status,"comparable");
    const masks=sharedHouseGroupScoreStrata(a,b);
    assert.deepEqual(masks,["all_scored","legacy_subset","newly_included"]);
    for(const stratum of masks)assert.deepEqual(houseGroupScoreMask(a,stratum),houseGroupScoreMask(b,stratum));
    assert.equal(houseGroupScoreMask(a,"competitive"),null,"competitive labels never establish matched race masks");
  }
  const [f,otherFP]=await Promise.all([load(catalog.initialRunId),load("house_relative_lean_v1-2024-h14-full435-polls-v1")]);
  assert.equal((await verifyHouseGroupComparison(f,otherFP)).status,"incomparable");
  const fp=await load(f.entry.id.replace("-full435-v1","-full435-polls-v1"));
  assert.equal((await verifyHouseGroupComparison(fp,otherFP)).status,"comparable");
  const changed=structuredClone(fp);changed.result.races[0].groups[0].members[0].name="Invented alias";
  assert.equal((await verifyHouseGroupComparison(f,changed)).status,"incomparable");
});

test("old saved configurations remain restorable and new FP+matched comparison binds separately",async()=>{
  const original=await loadHouseGroupPublication(groups,geometry);
  const old=await createOldSaved("Original saved view",original,{raceId:original.result.races[0].id,mapView:"forecast"});
  const merged=await load(original.result.id);
  assert.equal(houseGroupSavedViewAvailability(old,catalog).status,"catalog_compatible");
  assert.equal((await verifyHouseGroupSavedView(old,merged)).status,"verified");
  const fp=await load(original.result.id.replace("-full435-v1","-full435-polls-v1"));
  const saved=await createHouseGroupSavedView("Polling saved view",fp,{raceId:fp.result.races[0].id,mapView:"actual",comparison:merged});
  const restored=parseHouseGroupSavedViews(serializeHouseGroupSavedViews([old,saved]));
  assert.deepEqual(restored,[old,saved]);
  assert.equal((await verifyHouseGroupSavedView(saved,fp,merged)).status,"verified");
  assert.equal((await verifyHouseGroupSavedView(old,fp)).status,"unavailable");
});

test("year/model selection and optional measurement checkbox preserve exact polling state",async()=>{
  for(const entry of catalog.runs){
    const publication=await load(entry.id),slices=houseGroupModelSlices(publication);
    const fundamentals=slices.find(s=>s.id==="fundamentals").controls[0];
    assert.equal(fundamentals.options.length,4);
    for(const option of fundamentals.options){const r=catalog.runs.find(r=>r.id===option.id);assert.equal(r.cycle,entry.cycle);assert.equal(r.polling,entry.polling);}
    const measurement=slices.find(s=>s.id==="measurement").controls[0];
    assert.equal(measurement.id,"house-group-polling");assert.equal(measurement.required,false);
    assert.equal(measurement.toggle.label,"Include historical polls");
    for(const [key,state] of [["offValue",false],["onValue",true]]){
      const run=catalog.runs.find(r=>r.id===measurement.toggle[key]);
      assert.equal(run.polling,state);assert.equal(run.fundamentalsId,entry.fundamentalsId);assert.equal(run.cycle,entry.cycle);
    }
    for(const cycle of [2018,2020,2022,2024]){
      const selected=reviewedGroupSelection(catalog,cycle,entry.fundamentalsId,entry.polling);
      assert.equal(catalog.runs.find(r=>r.id===selected).polling,entry.polling);
    }
  }
  assert.equal(reviewedGroupSelection(catalog,2026,"house_history_refreshed_v1",true),null);
  const fp=await load("house_history_refreshed_v1-2024-h14-full435-polls-v1");
  const f=await load(catalog.initialRunId),url=groupSelectionUrl(fp.result,fp.result.races[0].id,"actual",f.result.id);
  assert.equal(new URL("http://test"+url).searchParams.get("compare"),f.result.id);
  assert.equal(new URL("http://test"+groupSelectionForResult(fp.result,f.result,new URL("http://test"+url).search)).searchParams.get("race"),fp.result.races[0].id);
});

test("actual group and conditional-share API/browser roundtrips verify exact source/content bindings",async()=>{
  const originalFetch=globalThis.fetch;
  globalThis.fetch=async url=>{
    const request=new Request("http://local"+url);
    return new URL(request.url).pathname.endsWith("/shares")?shareGET(request):groupGET(request);
  };
  try{
    const controller=new AbortController(),entry=catalog.runs.find(r=>r.polling&&r.cycle===2024);
    const group=await loadHouseReviewedGroup(entry.id,controller.signal);
    assert.equal(group.entry.id,entry.id);
    const shares=await loadHousePollingShares(entry,controller.signal);
    assert.equal(shares.run.id,entry.conditionalShareRunId);
    assert.equal(shares.run.informationMode,HOUSE_POLLING_INFORMATION_MODE);
    globalThis.fetch=async url=>{
      const response=await shareGET(new Request("http://local"+url));const value=await response.json();
      value.result.races[0].outcomes[0].share.mean+=.001;
      return Response.json(value);
    };
    await assert.rejects(loadHousePollingShares(entry,controller.signal),/differ/);
    await assert.rejects(loadHousePollingShares({...entry,conditionalShareSha256:"0".repeat(64)},controller.signal),/exact/);
  }finally{globalThis.fetch=originalFetch;}
  for(const query of ["run=../../bad","run=x&run=y","unknown=value","run=house_history_refreshed_v1-2026-h14-full435-polls-v1"])
    assert.equal((await groupGET(new Request("http://local/api/chambers/house/groups?"+query))).status,400);
  for(const query of ["","run=x","run=x&run=y","run=x&other=y"])
    assert.equal((await shareGET(new Request("http://local/api/chambers/house/groups/shares?"+query))).status,400);
});

test("conditional shares reject inconsistent metadata, actual counts, proxy mode and quantile complements",async()=>{
  const entry=catalog.runs.find(r=>r.polling),publication=await loadHousePollingSharePublication(polling,entry.conditionalShareRunId);
  for(const mutate of [x=>x.run.informationMode="strict_asof",x=>x.currentElectionForecast=true,
    x=>x.sourcePolicy.sha256="0".repeat(64),x=>x.sourceAdmission.sha256="0".repeat(64),
    x=>x.pollingCoverage.polls++,x=>x.races[0].pollCount=-1,x=>x.jointDraws.raceIds.reverse(),
    x=>x.races[0].outcomes[1].share.quantiles.p975-=.001,x=>x.metadata.full_chamber=true,
    x=>x.chamberProof={status:"qualified"}]){
    const value=structuredClone(publication.result);mutate(value);assert.equal(parseHousePollingShares(value),null);
  }
});

test("browser rejects changed catalog, new DTO summaries, names, coverage and maps",async()=>{
  const entry=catalog.runs.find(r=>r.polling),original=await load(entry.id);
  for(const mutate of [x=>x.result.raceScores[0].categoricalBrier+=.01,x=>x.result.pollingCoverage.polls++,
    x=>x.entry.sourcePolicySha256="0".repeat(64),x=>x.result.races[0].groups[0].members[0].name="Unreviewed name",
    x=>x.catalog.initialRunId=x.entry.id,x=>x.result.races[0].pollCount++,
    x=>x.geography.geojson.features[0].properties.injected=true]){
    const changed=structuredClone(original);mutate(changed);assert.equal(await parseHouseReviewedPublication(changed),null);
  }
  const changed=structuredClone(catalog);changed.runs[16].conditionalShareRunId=changed.runs[0].conditionalShareRunId;
  assert.equal(parseHouseReviewedCatalog(changed),null);
});

test("source import rejects malformed pinned catalog and symlink output parent before writing",async()=>{
  const directory=await mkdtemp(join(tmpdir(),"house-polling-negative-"));
  try{
    const source=join(directory,"source"),output=join(directory,"output");
    await mkdir(join(source,"research/house-polling-benchmark-20261024/frontend-results-v1"),{recursive:true});
    await writeFile(join(source,"research/house-polling-benchmark-20261024/frontend-results-v1/catalog.json"),"{}");
    await assert.rejects(importHousePolling({sourceRoot:source,groupDirectory:groups,legacyDirectory:geometry,outputDirectory:output}),/fingerprint/);
    await assert.rejects(access(output));
  }finally{await rm(directory,{recursive:true,force:true});}
});

test("optional owner-checkout import is reproducible and preserves every original F artifact",async t=>{
  const owner="/home/hayden/projects/ballotmancy-house-benchmark";
  try{await access(join(owner,"research/house-polling-benchmark-20261024/frontend-results-v1/catalog.json"));}
  catch{t.skip("optional real House scientific checkout is absent");return;}
  const temporary=await mkdtemp(join(tmpdir(),"house-polling-import-"));
  const oldManifest=await readFile(join(groups,"publication.json")),oldCatalog=await read(join(groups,JSON.parse(oldManifest).catalog.file));
  const protectedFiles=[join(groups,"publication.json"),...oldCatalog.runs.map(r=>join(groups,r.view.file))];
  const before=await Promise.all(protectedFiles.map(async path=>digest(await readFile(path))));
  try{
    const output=join(temporary,"out");
    const result=await importHousePolling({sourceRoot:owner,groupDirectory:groups,legacyDirectory:geometry,outputDirectory:output});
    assert.equal(result.runs,32);assert.equal(result.matchedPairs,16);assert.ok(result.verifiedSourcePins>=327);
    assert.deepEqual(await read(join(output,"reviewed-bindings.json")),catalog);
    for(const pair of catalog.matchedPairs)assert.equal((await loadHouseReviewedPublication(groups,geometry,pair.pollingRunId,output)).entry.polling,true);
    assert.deepEqual(await Promise.all(protectedFiles.map(async path=>digest(await readFile(path)))),before);
    const aliasParent=join(temporary,"alias");await symlink(owner,aliasParent);
    await assert.rejects(importHousePolling({sourceRoot:owner,groupDirectory:groups,legacyDirectory:geometry,outputDirectory:join(aliasParent,"forbidden-output")}),/parent|outside|symlink/);
    await assert.rejects(access(join(owner,"forbidden-output")));
  }finally{await rm(temporary,{recursive:true,force:true});}
});
