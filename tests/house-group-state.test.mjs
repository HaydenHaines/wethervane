import test from "node:test";
import assert from "node:assert/strict";
import { resolve } from "node:path";
import { loadHouseGroupPublication } from "../lib/house-group-publication.ts";
import { groupSelectionUrl,groupSelectionForResult } from "../lib/house-group-selection.ts";
import { verifyHouseGroupComparison } from "../lib/house-group-comparison.ts";
import { HOUSE_GROUP_SAVED_VIEWS_STORAGE_KEY,createHouseGroupSavedView,parseHouseGroupSavedViews,serializeHouseGroupSavedViews,verifyHouseGroupSavedView,houseGroupSavedViewAvailability } from "../lib/house-group-saved-views.ts";
import { HOUSE_SAVED_VIEWS_STORAGE_KEY } from "../lib/house-saved-views.ts";
const directory=resolve('data/house-groups'),geometry=resolve('data/house');
const load=(model,year=2024)=>loadHouseGroupPublication(directory,geometry,`${model}-${year}-h14-full435-v1`);
const primary=await load('house_history_refreshed_v1'),comparison=await load('house_relative_context_v1'),earlier=await load('house_history_refreshed_v1',2018);
const raceId=primary.result.races[0].id;
const save=()=>createHouseGroupSavedView('My group view',primary,{raceId,mapView:'actual',comparison});
test('selection URLs preserve exact same-year districts, view and allowed second group model',()=>{
 const url=groupSelectionUrl(primary.result,raceId,'actual',comparison.result.id),params=new URL('https://example.test'+url).searchParams;
 assert.equal(params.get('chamber'),'house');assert.equal(params.get('run'),primary.result.id);assert.equal(params.get('race'),raceId);assert.equal(params.get('view'),'actual');assert.equal(params.get('compare'),comparison.result.id);
 const changed=new URL('https://example.test'+groupSelectionForResult(comparison.result,primary.result,params.toString())).searchParams;
 assert.equal(changed.get('race'),raceId);assert.equal(changed.get('compare'),null,'self comparison is cleared');
 const cross=new URL('https://example.test'+groupSelectionForResult(earlier.result,primary.result,params.toString())).searchParams;assert.equal(cross.get('race'),null);assert.equal(cross.get('compare'),null);
 for(const id of [earlier.result.id,primary.result.id,'house_history_refreshed_v1-2024-h14','../bad'])assert.equal(new URL('https://example.test'+groupSelectionUrl(primary.result,raceId,'forecast',id)).searchParams.get('compare'),null);
 assert.equal(new URL('https://example.test'+groupSelectionUrl(primary.result,'missing')).searchParams.get('race'),null);
 const changedPanel=structuredClone(primary.result);changedPanel.provenance.sourcePanelSha256='0'.repeat(64);assert.equal(new URL('https://example.test'+groupSelectionForResult(changedPanel,primary.result,params.toString())).searchParams.get('race'),null);
});
test('real same-year results compare exact support without paired uncertainty',async()=>{
 assert.deepEqual(await verifyHouseGroupComparison(primary,comparison),{status:'comparable'});
 assert.equal((await verifyHouseGroupComparison(primary,primary)).status,'incomparable');assert.equal((await verifyHouseGroupComparison(primary,earlier)).status,'incomparable');
 for(const mutate of [p=>p.result.targetId='wrong',p=>p.result.races[0].groups.pop(),p=>p.result.races[0].groupScoreEligible=false,p=>p.result.races[0].observed.groupId='O',p=>p.result.provenance.sourcePanelSha256='0'.repeat(64),p=>p.entry.resultSha256='0'.repeat(64)]){const changed=structuredClone(comparison);mutate(changed);assert.equal((await verifyHouseGroupComparison(primary,changed)).status,'incomparable');}
});
test('all forty-eight offered model pairs verify against the complete published collection', async () => {
 const publications = await Promise.all(primary.catalog.runs.map(entry => loadHouseGroupPublication(directory, geometry, entry.id)));
 let verified = 0;
 for (const selected of publications) {
  for (const reference of publications.filter(item => item.result.cycle === selected.result.cycle && item.result.id !== selected.result.id)) {
   const check = await verifyHouseGroupComparison(selected, reference);
   assert.equal(check.status, 'comparable', `${selected.result.id} / ${reference.result.id}: ${check.reason || ''}`);
   verified++;
  }
 }
 assert.equal(verified, 48);
});
test('separate group storage round-trips both exact bindings and restores after full verification',async()=>{
 assert.notEqual(HOUSE_GROUP_SAVED_VIEWS_STORAGE_KEY,HOUSE_SAVED_VIEWS_STORAGE_KEY);
 const view=await save(),serialized=serializeHouseGroupSavedViews([view]),restored=parseHouseGroupSavedViews(serialized)[0];
 assert.deepEqual(restored,view);assert.equal(view.runId,primary.result.id);assert.equal(view.resultSha256,primary.entry.resultSha256);assert.equal(view.viewSha256,primary.entry.viewSha256);assert.equal(view.comparison.runId,comparison.result.id);assert.equal(view.comparison.qualificationSha256,comparison.catalog.qualificationSha256);assert.equal(view.geographyContentSha256,primary.entry.geographyContentSha256);
 assert.equal(houseGroupSavedViewAvailability(view,primary.catalog).status,'catalog_compatible');assert.equal((await verifyHouseGroupSavedView(view,primary,comparison)).status,'verified');
 assert.equal((await verifyHouseGroupSavedView(view,primary)).status,'unavailable');assert.equal((await verifyHouseGroupSavedView(view,comparison,primary)).status,'unavailable');
});
test('stale immutable bindings remain visible but never silently rebind',async()=>{
 const original=await save();
 for(const field of ['resultSha256','viewSha256','sourcePanelSha256','predictionSha256','auditSha256','sourcePolicySha256','sourceCatalogSha256','qualificationSha256','geographySha256','geographyContentSha256','legacyResultSha256']){
  const view=structuredClone(original);view[field]='0'.repeat(64);assert.equal(parseHouseGroupSavedViews(JSON.stringify({schemaVersion:1,views:[view]})).length,1,'valid stale entry retained');assert.equal((await verifyHouseGroupSavedView(view,primary,comparison)).status,'unavailable',field);
 }
 const changed=structuredClone(original);changed.comparison.resultSha256='0'.repeat(64);assert.equal((await verifyHouseGroupSavedView(changed,primary,comparison)).status,'unavailable');assert.equal(houseGroupSavedViewAvailability(changed,primary.catalog).status,'unavailable');
 const unavailable=structuredClone(original);unavailable.runId='a-removed-result';assert.equal(parseHouseGroupSavedViews(JSON.stringify({schemaVersion:1,views:[unavailable]})).length,1);assert.equal(houseGroupSavedViewAvailability(unavailable,primary.catalog).status,'unavailable');
});
test('invalid districts, cross-year pairs and malformed storage are refused',async()=>{
 await assert.rejects(createHouseGroupSavedView('Wrong district',primary,{raceId:'2024:ZZ-01:house:regular',mapView:'forecast'}));
 await assert.rejects(createHouseGroupSavedView('Wrong pair',primary,{raceId,mapView:'forecast',comparison:earlier}));
 const view=await save(),invalid=structuredClone(view);invalid.comparison.cycle=2018;
 assert.deepEqual(parseHouseGroupSavedViews(JSON.stringify({schemaVersion:1,views:[invalid]})),[]);
 assert.deepEqual(parseHouseGroupSavedViews('{bad json'),[]);assert.deepEqual(parseHouseGroupSavedViews(JSON.stringify({schemaVersion:2,views:[view]})),[]);
 assert.equal(parseHouseGroupSavedViews(JSON.stringify({schemaVersion:1,views:[view,view]})).length,1);
 await assert.rejects(createHouseGroupSavedView(' ',primary,{raceId,mapView:'forecast'}));
});
test('NC09 actual-view save preserves an unavailable observed outcome and readable display labels are not identity',async()=>{
 const nc=earlier.result.races.find(r=>r.districtKey==='NC-09'),view=await createHouseGroupSavedView('NC09',earlier,{raceId:nc.id,mapView:'actual'});
 assert.equal(nc.observed.status,'unavailable');view.runLabel='My display label';assert.equal((await verifyHouseGroupSavedView(view,earlier)).status,'verified');
 assert.equal(view.mapView,'actual');assert.equal(view.raceId,'2018:NC-09:house:regular');
});
