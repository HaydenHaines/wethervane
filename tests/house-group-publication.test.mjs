import test, { after } from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync, mkdtempSync, cpSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { loadHouseGroupPublication, HouseGroupSelectionError } from "../lib/house-group-publication.ts";
import { parseHouseGroupCatalog, parseHouseGroupPublication, canonicalGroupJson, HOUSE_GROUP_QUALIFICATION_SHA, HOUSE_GROUP_POLICY_SHA, HOUSE_GROUP_SOURCE_CATALOG_SHA, HOUSE_GROUP_PUBLIC_CATALOG_SHA } from "../lib/house-group-catalog.ts";
import { parseHouseGroupView } from "../lib/house-group-result.ts";
import { groupRaceForMap } from "../lib/house-group-presentation.ts";
const directory = resolve("data/house-groups"), geometryDirectory = resolve("data/house");
const read = path => JSON.parse(readFileSync(path,"utf8"));
const manifest = read(join(directory,"publication.json"));
const rawCatalog = read(join(directory,manifest.catalog.file));
const digest = bytes => createHash("sha256").update(bytes).digest("hex");
const temporary = mkdtempSync(join(tmpdir(),"house-group-publication-"));
after(()=>rmSync(temporary,{recursive:true,force:true}));
function copyPublication(name) { const target=join(temporary,name);cpSync(directory,target,{recursive:true});return target; }
function rewriteCatalog(target,change) {
 const publication=read(join(target,"publication.json")),catalog=read(join(target,publication.catalog.file));change(catalog);
 const bytes=Buffer.from(canonicalGroupJson(catalog)),sha256=digest(bytes),file=`catalog-${sha256}.json`;
 writeFileSync(join(target,file),bytes);writeFileSync(join(target,"publication.json"),JSON.stringify({...publication,catalog:{file,sha256}}));
}
test("committed catalog binds all16 unique models and years to qualification and policy",()=>{
 assert.equal(rawCatalog.runs.length,16);assert.equal(new Set(rawCatalog.runs.map(r=>r.id)).size,16);
 assert.equal(rawCatalog.qualificationSha256,HOUSE_GROUP_QUALIFICATION_SHA);assert.equal(rawCatalog.sourcePolicySha256,HOUSE_GROUP_POLICY_SHA);assert.equal(rawCatalog.sourceCatalogSha256,HOUSE_GROUP_SOURCE_CATALOG_SHA);
 assert.equal(digest(readFileSync(join(directory,manifest.catalog.file))),manifest.catalog.sha256);
 assert.equal(rawCatalog.initialRunId,"house_history_refreshed_v1-2024-h14-full435-v1");
 for(const year of [2018,2020,2022,2024])assert.equal(rawCatalog.runs.filter(r=>r.cycle===year).length,4);
 assert.ok(parseHouseGroupCatalog(rawCatalog));
 assert.equal(digest(canonicalGroupJson(parseHouseGroupCatalog(rawCatalog))),HOUSE_GROUP_PUBLIC_CATALOG_SHA);
});
test("all16 committed views validate raw result identity, exact legacy and geometry bindings",async()=>{
 const legacyManifest=read(join(geometryDirectory,"publication.json")),legacyCatalog=read(join(geometryDirectory,legacyManifest.catalog.file));
 for(const entry of rawCatalog.runs){
  const publication=await loadHouseGroupPublication(directory,geometryDirectory,entry.id);
  const old=legacyCatalog.runs.find(r=>r.id===entry.legacyRunId);
  assert.equal(publication.result.id,entry.id);assert.equal(publication.result.configuration,`${entry.fundamentalsId}-loading-1`);
  assert.equal(publication.result.provenance.sourcePolicySha256,HOUSE_GROUP_POLICY_SHA);
  assert.equal(entry.legacyResultSha256,old.resultSha256);assert.equal(entry.geographySha256,old.geography.sha256);assert.equal(entry.mapVintage,old.geography.metadata.mapVintage);
  assert.equal(digest(readFileSync(join(directory,entry.view.file))),entry.viewSha256);
  assert.equal(digest(readFileSync(join(geometryDirectory,entry.geography.file))),entry.geographySha256);
  assert.equal(digest(canonicalGroupJson(publication.geography.geojson)),entry.geographyContentSha256);
  assert.notEqual(entry.geographyContentSha256,entry.geographySha256,"content hash is separate from original source-byte serialization");
  assert.equal(digest(canonicalGroupJson(publication.result)),entry.viewSha256);
  assert.equal(publication.result.races.length,435);assert.equal(publication.geography.geojson.features.length,435);
  assert.deepEqual(parseHouseGroupView(publication.result),publication.result);
  assert.ok(await parseHouseGroupPublication(JSON.parse(JSON.stringify(publication))));
  assert.equal(publication.result.raceScores.find(s=>s.stratum==='all_scored').n,entry.cycle===2018?434:435);
  assert.equal(publication.result.publicChamberControl,false);assert.equal(publication.result.publicTipping,false);
 }
});
test("unknown run selection fails without falling back to the default",async()=>{
 for(const id of ['../../publication.json','house_history_refreshed_v1-2026-h14-full435-v1','arbitrary'])await assert.rejects(loadHouseGroupPublication(directory,geometryDirectory,id),HouseGroupSelectionError);
});
test("server rejects changed content and mismatched artifact fingerprints",async()=>{
 const target=copyPublication('tampered-view'),entry=rawCatalog.runs[0];
 const path=join(target,entry.view.file),view=read(path);view.races[0].groups[0].members[0].name='Changed valid name';writeFileSync(path,canonicalGroupJson(view));
 await assert.rejects(loadHouseGroupPublication(target,geometryDirectory,entry.id),/fingerprint/);
 const catalogTarget=copyPublication('tampered-catalog'),p=read(join(catalogTarget,'publication.json'));p.catalog.sha256='0'.repeat(64);writeFileSync(join(catalogTarget,'publication.json'),JSON.stringify(p));await assert.rejects(loadHouseGroupPublication(catalogTarget,geometryDirectory),/fingerprint/);
});
test("server rejects view references, path traversal and declared geometry bindings",async()=>{
 for(const [name,change] of [ ['view-reference',c=>c.runs[0].view.sha256='0'.repeat(64)],['path-traversal',c=>c.runs[0].view.file='../publication.json'],['map-binding',c=>c.runs[0].geographySha256='0'.repeat(64)] ]){
  const target=copyPublication(name);rewriteCatalog(target,change);await assert.rejects(loadHouseGroupPublication(target,geometryDirectory,rawCatalog.runs[0].id));
 }
});
test("browser rejects altered values, support and scope despite valid API structure",async()=>{
 const publication=await loadHouseGroupPublication(directory,geometryDirectory);
 for(const mutate of [p=>p.result.races[0].groups[0].members[0].name='Changed valid name',p=>p.result.races.pop(),p=>p.result.races[0].probabilities.D=.9,p=>p.result.publicChamberControl=true,p=>p.entry.viewSha256='0'.repeat(64),p=>p.catalog.sourcePolicySha256='0'.repeat(64),p=>p.catalog.qualificationSha256='0'.repeat(64)]){
  const value=structuredClone(publication);mutate(value);assert.equal(await parseHouseGroupPublication(value),null);
 }
});
test("browser rejects changed geometry bytes rather than trusting the claimed source hash",async()=>{
 const publication=structuredClone(await loadHouseGroupPublication(directory,geometryDirectory));
 const geometry=publication.geography.geojson.features[0].geometry;
 const polygon=geometry.type==='Polygon'?geometry.coordinates:geometry.coordinates[0];
 for(const ring of polygon)for(const point of ring)point[0]+=.00001;
 assert.ok((await parseHouseGroupPublication(publication)) === null, "mutated geometry must be rejected");
});
test("catalog rejects changed raw-result, legacy-result and default identity bindings",()=>{
 for(const change of [c=>c.runs[0].resultSha256='0'.repeat(64),c=>c.runs[0].legacyResultSha256='0'.repeat(64),c=>c.initialRunId=c.runs[0].id,c=>c.runs[0].geographySha256='0'.repeat(64),c=>c.runs[0].geographyContentSha256='0'.repeat(64),c=>c.sourceCatalogSha256='0'.repeat(64)]){
  const catalog=structuredClone(rawCatalog);change(catalog);assert.ok(parseHouseGroupCatalog(catalog) === null, "changed scientific/default catalog binding must be rejected");
 }
});
test("map projection adapter preserves groups and never allocates pooled probability to a person",async()=>{
 const publication=await loadHouseGroupPublication(directory,geometryDirectory,'house_history_refreshed_v1-2024-h14-full435-v1');
 const race=publication.result.races.find(r=>r.groups.some(g=>g.id==='D'&&g.members.length>1)),mapped=groupRaceForMap(race);
 for(const group of race.groups){const outcome=mapped.outcomes.find(o=>o.id===group.id);assert.equal(outcome.winProbability,group.winProbability);assert.equal(outcome.party,group.id);if(group.individualCandidateProbability===null)assert.equal(outcome.kind,'group');}
 const ncPublication=await loadHouseGroupPublication(directory,geometryDirectory,'house_history_refreshed_v1-2018-h14-full435-v1');const nc=groupRaceForMap(ncPublication.result.races.find(r=>r.districtKey==='NC-09'));assert.equal(nc.observed,undefined);assert.ok(nc.outcomes.length>0);
});

test("geometry canonical identity tolerates object-key serialization changes but rejects replacement content",async()=>{
 const publication=structuredClone(await loadHouseGroupPublication(directory,geometryDirectory));
 const collection=publication.geography.geojson;
 publication.geography.geojson={features:collection.features,type:collection.type};
 assert.ok(await parseHouseGroupPublication(publication));
 publication.geography.geojson.features[0].properties.injectedClaim='Not in the admitted source';
 assert.ok((await parseHouseGroupPublication(publication))===null);
});
