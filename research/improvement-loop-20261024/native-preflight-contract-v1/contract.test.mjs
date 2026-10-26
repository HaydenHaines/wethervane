import test from 'node:test';
import assert from 'node:assert/strict';
import {matchPreflight} from './contract.mjs';
const h = 'a'.repeat(64), pin = {id:'invented-owner',sha256:h};
function fixture() {
  const components = {base:'invented',polling:'invented',uncertainty:'invented',participation:'invented'};
  const catalog = {id:'invented-catalog',sourceCatalog:{sha256:h},slots:Object.keys(components).map(id => ({id,options:[{id:'invented',years:[2020,2022],availability:'requires_computation'}]}))};
  const binding = {year:2020,dateId:'invented-2020-cutoff',components,componentInstances:Object.fromEntries(Object.keys(components).map(k => [k,{providerId:'invented-provider',version:'invented-v1',instanceId:`invented-${k}`} ])),targetId:'invented-target',law:{id:'invented-law',sha256:h},sourceManifest:{id:'invented-sources',sha256:h},outputContract:{id:'invented-output',sha256:h}};
  const metadata = {schema:'ComputationPreflightMetadataV1',catalog:{id:catalog.id,sourceSha256:h},executionAvailable:false,entries:[{id:'invented-entry',ownerHandoff:pin,bindings:[binding,{...structuredClone(binding),year:2022,dateId:'invented-2022-cutoff'}],computation:{kind:'joint_refit',reason:'Invented joint refit required.',requiredSteps:['Review invented example.']},outputs:['invented-summary'],evidence:{status:'experimental',implementation:'Invented',numerical:'Unqualified',predictive:'No evidence',scope:'Invented only'}}]};
  return {metadata,catalog,request:{catalogId:catalog.id,year:2020,dateId:binding.dateId,components:structuredClone(components)},pin:structuredClone(pin)};
}
const run = f => matchPreflight(f.metadata,f.catalog,f.request,f.pin);
test('exact four-slot matching, ordering, explicit refit and immutable output', () => {
  const f = fixture(); f.request.components = Object.fromEntries(Object.entries(f.request.components).reverse());
  const r = run(f); assert.equal(r.state,'match'); assert.equal(r.lifecycle,'requires_computation'); assert.equal(r.executionAvailable,false); assert.equal(r.consistencyOnly,true); assert.equal(r.configurationId,null); assert.equal(r.resultId,null); assert.equal(r.run,null); assert.equal(r.computation.kind,'joint_refit');
  r.binding.components.base = 'changed'; assert.equal(f.metadata.entries[0].bindings[0].components.base,'invented');
});
test('explicit pairs support later dates absent any saved registry; no Cartesian expansion', () => {
  const f=fixture(); f.request.year=2022; f.request.dateId='invented-2022-cutoff'; assert.equal(run(f).state,'match'); f.request.dateId='invented-2020-cutoff'; assert.equal(run(f).state,'none');
});
test('no metadata is distinct from malformed metadata', () => {
  const f=fixture(); f.metadata=undefined; assert.equal(run(f).state,'none'); f.metadata=null; assert.equal(run(f).state,'invalid');
});
const invalidCases = {
  'schema':f=>f.metadata.schema='v2',
  'execution flag':f=>f.metadata.executionAvailable=true,
  'catalog id':f=>f.metadata.catalog.id='other',
  'catalog hash':f=>f.metadata.catalog.sourceSha256='b'.repeat(64),
  'malformed catalog hash':f=>f.catalog.sourceCatalog.sha256='bad',
  'owner id':f=>f.metadata.entries[0].ownerHandoff={...pin,id:'other'},
  'owner hash':f=>f.pin.sha256='b'.repeat(64),
  'missing trusted owner context':f=>f.pin=undefined,
  'malformed handoff':f=>f.metadata.entries[0].ownerHandoff={id:'invented-owner',sha256:'bad'},
  'missing request slot':f=>delete f.request.components.participation,
  'extra request slot':f=>f.request.components.extra='invented',
  'different request selection':f=>f.request.components.base='other',
  'unsupported request year':f=>f.request.year=2024,
  'preset':f=>f.request.presetId='invented-preset',
  'empty date':f=>f.request.dateId='',
  'missing binding slot':f=>delete f.metadata.entries[0].bindings[0].components.participation,
  'extra binding slot':f=>f.metadata.entries[0].bindings[0].components.extra='invented',
  'missing instance slot':f=>delete f.metadata.entries[0].bindings[0].componentInstances.participation,
  'extra instance slot':f=>f.metadata.entries[0].bindings[0].componentInstances.extra={},
  'missing instance identity':f=>delete f.metadata.entries[0].bindings[0].componentInstances.base.instanceId,
  'missing version':f=>delete f.metadata.entries[0].bindings[0].componentInstances.base.version,
  'missing provider':f=>delete f.metadata.entries[0].bindings[0].componentInstances.base.providerId,
  'unsupported binding year':f=>f.metadata.entries[0].bindings[0].year=2024,
  'held option':f=>f.catalog.slots[0].options[0].availability='unavailable',
  'unknown option':f=>f.metadata.entries[0].bindings[0].components.base='other',
  'duplicate slots':f=>f.catalog.slots.push(structuredClone(f.catalog.slots[0])),
  'duplicate options':f=>f.catalog.slots[0].options.push(structuredClone(f.catalog.slots[0].options[0])),
  'duplicate entry IDs':f=>f.metadata.entries.push(structuredClone(f.metadata.entries[0])),
  'no refit':f=>f.metadata.entries[0].computation.kind='reweight',
  'missing steps':f=>f.metadata.entries[0].computation.requiredSteps=[],
  'missing outputs':f=>f.metadata.entries[0].outputs=[],
  'missing evidence':f=>delete f.metadata.entries[0].evidence.scope,
  'empty bindings':f=>f.metadata.entries[0].bindings=[],
  'empty entries':f=>f.metadata.entries=[],
  'target missing':f=>delete f.metadata.entries[0].bindings[0].targetId,
};
for (const field of ['law','sourceManifest','outputContract']) invalidCases[`malformed ${field}`]=f=>f.metadata.entries[0].bindings[0][field].sha256='bad';
for (const [name,change] of Object.entries(invalidCases)) test(`reject ${name}`,()=>{const f=fixture(); change(f); assert.equal(run(f).state,'invalid');});
test('duplicate exact rows across or within entries are ambiguous',()=>{
  const f=fixture(); const e=structuredClone(f.metadata.entries[0]); e.id='invented-second'; f.metadata.entries.push(e); assert.equal(run(f).state,'ambiguous');
  f.metadata.entries.pop(); f.metadata.entries[0].bindings.push(structuredClone(f.metadata.entries[0].bindings[0])); assert.equal(run(f).state,'ambiguous');
});
test('unrelated duplicate rows also fail closed',()=>{const f=fixture(); f.metadata.entries[0].bindings.push(structuredClone(f.metadata.entries[0].bindings[1])); assert.equal(run(f).state,'ambiguous');});
test('malformed unrelated row cannot be ignored',()=>{const f=fixture(); delete f.metadata.entries[0].bindings[1].law; assert.equal(run(f).state,'invalid');});
test('ordinary object malformed inputs return invalid rather than throw',()=>{
  for(const value of [null,[],{},0,'text']) {
    const f=fixture(); f.request=value; assert.equal(run(f).state,'invalid');
    const g=fixture(); g.catalog=value; assert.equal(run(g).state,'invalid');
  }
});
test('known alternate selection without exact binding returns none',()=>{
  const f=fixture(); f.catalog.slots[0].options.push({id:'invented-alternate',years:[2020],availability:'requires_computation'}); f.request.components.base='invented-alternate'; assert.equal(run(f).state,'none');
});
test('metadata matcher cannot mutate catalog, request, metadata or owner context',()=>{
  const f=fixture(), before=structuredClone(f); run(f); assert.deepEqual(f,before);
});
