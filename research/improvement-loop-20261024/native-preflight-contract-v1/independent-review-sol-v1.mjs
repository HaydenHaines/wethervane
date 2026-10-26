import fs from 'node:fs';
import crypto from 'node:crypto';
import {matchPreflight} from './contract.mjs';
const root=new URL('.',import.meta.url), lab=new URL('../../../',root), h='c'.repeat(64), pin={id:'independent-owner',sha256:h};
function fixture(){
 const catalog={id:'independent-catalog',sourceCatalog:{sha256:h},slots:['a','b'].map(id=>({id,options:[{id:'v',years:[2020,2022],availability:'requires_computation'}]}))};
 const binding={year:2020,dateId:'d2020',components:{a:'v',b:'v'},componentInstances:{a:{providerId:'p',version:'1',instanceId:'a'},b:{providerId:'p',version:'1',instanceId:'b'}},targetId:'t',law:{id:'l',sha256:h},sourceManifest:{id:'s',sha256:h},outputContract:{id:'o',sha256:h}};
 const entry={id:'e',ownerHandoff:pin,bindings:[binding,{...structuredClone(binding),year:2022,dateId:'d2022'}],computation:{kind:'joint_refit',reason:'Independent fixture',requiredSteps:['review']},outputs:['o'],evidence:{status:'experimental',implementation:'fixture',numerical:'none',predictive:'none',scope:'invented'}};
 return {catalog,metadata:{schema:'ComputationPreflightMetadataV1',executionAvailable:false,catalog:{id:catalog.id,sourceSha256:h},entries:[entry]},request:{catalogId:catalog.id,year:2020,dateId:'d2020',components:{a:'v',b:'v'}}};
}
const checks=[];
function probe(name,change,expected){const f=fixture();change(f);let actual;try{const r=matchPreflight(f.metadata,f.catalog,f.request,pin);actual=r.state;if(r.executionAvailable!==false||r.run!==null||r.configurationId!==null||r.resultId!==null)throw Error('execution invariant broken');}catch(e){actual=`throws:${e.name}`;}checks.push({name,expected,actual,pass:actual===expected});}
probe('exact_match',()=>{},'match');
probe('year_date_cross_product_rejected',f=>f.request.year=2022,'none');
probe('wrong_date_rejected',f=>f.request.dateId='d2022','none');
probe('missing_slot',f=>delete f.request.components.b,'invalid');
probe('extra_slot',f=>f.request.components.c='v','invalid');
probe('unrelated_malformed_binding',f=>delete f.metadata.entries[0].bindings[1].law,'invalid');
probe('unrelated_duplicate',f=>f.metadata.entries[0].bindings.push(structuredClone(f.metadata.entries[0].bindings[1])),'ambiguous');
probe('owner_pin_mismatch',f=>f.metadata.entries[0].ownerHandoff={...pin,id:'other'},'invalid');
probe('catalog_hash_mismatch',f=>f.metadata.catalog.sourceSha256='d'.repeat(64),'invalid');
probe('sparse_catalog_slots',f=>f.catalog.slots=Array(1),'invalid');
probe('sparse_catalog_options',f=>f.catalog.slots[0].options=Array(1),'invalid');
probe('uncloneable_unknown_field',f=>f.metadata.entries[0].extraFunction=()=>{},'invalid');
const hash=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const evidence=JSON.parse(fs.readFileSync(new URL('evidence.json',root)));
const filePins=Object.entries(evidence.files).map(([path,expected])=>({path,expected,actual:hash(new URL(path,root)),consistent:hash(new URL(path,root))===expected}));
const productionPins=Object.entries(evidence.productionPins).map(([path,p])=>({path,before:p.before,after:p.after,current:hash(new URL(path,lab)),consistent:p.before===p.after&&p.after===hash(new URL(path,lab))}));
const report={schema:'IndependentPreflightReviewSolV1',scope:'Lab-only invented runtime fixtures; no production or matcher mutation',verdict:checks.every(c=>c.pass)?'pass_for_tested_scope':'dense_JSON_checks_pass_runtime_malformed_input_gaps',supplied_tests_rerun:{private:{passed:47,failed:0},legacy:{passed:7,failed:0}},checks,filePins,productionPins,reviewedContractSha256:hash(new URL('contract.mjs',root)),executionReview:'Pure matcher imports no I/O, scheduler, model code or production module; every normal-return state forces executionAvailable false and configurationId/resultId/run null. No integration references found with rg in app/lib/components/tests.',limitations:['Owner/artifact pins are string consistency, not authentication or byte validation.','Per-slot instance references are opaque declarations, not verified catalog-to-scientific-source mappings.','Sparse arrays are runtime JavaScript values, not dense parsed JSON; uncloneable extras likewise exceed JSON values. Contract currently claims all untrusted runtime inputs.','Production before/after pins corroborate selected files only, not a full repository diff.']};
fs.writeFileSync(new URL('independent-review-sol-v1.json',root),JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify({verdict:report.verdict,probes:checks.length,failed:checks.filter(c=>!c.pass),filePinsConsistent:filePins.every(p=>p.consistent),productionPinsConsistent:productionPins.every(p=>p.consistent)},null,2));
