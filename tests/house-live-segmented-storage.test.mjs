import assert from 'node:assert/strict';
import test from 'node:test';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { appendFile, copyFile, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { checkHouseLiveCandidate } from '../scripts/check_house_live_candidate.mjs';
import { resolveHouseLivePredictionPlan } from '../scripts/house_live_evidence.mjs';
import { makeHouseLiveResult } from './fixtures/house-live-fixtures.mjs';
const digest=b=>createHash('sha256').update(b).digest('hex');
async function put(root,path,value){const b=Buffer.from(JSON.stringify(value,null,2)+'\n');await mkdir(dirname(join(root,path)),{recursive:true});await writeFile(join(root,path),b);return {path,sha256:digest(b)};}
async function fixture(t){
 const owner=await mkdtemp(join(tmpdir(),'house-v2-storage-'));t.after(()=>rm(owner,{recursive:true,force:true}));const native=join(owner,'native');await mkdir(native);
 const result=makeHouseLiveResult();result.qualification.publication_status='not_reviewed';
 const generated=spawnSync('python3',['-B',fileURLToPath(new URL('./site_house_live_conditional_storage_test.py',import.meta.url)),'--fixture'],{input:JSON.stringify({root:owner,ids:result.races.filter(r=>r.conditional_shares!==null).map(r=>r.race_id),configuration:{id:result.id,...result.configuration}}),encoding:'utf8'});
 assert.equal(generated.status,0,generated.stderr);const storage=JSON.parse(generated.stdout);
 const rows=[Array(435).fill(0),Array(435).fill(0),Array(435).fill(1),Array(435).fill(1)];rows[1][1]=rows[3][1]=2;rows[2][2]=rows[3][2]=0;
 const archivePath='private/candidate-joint.npz';await mkdir(join(native,'private'));
 const written=spawnSync('python3',[fileURLToPath(new URL('./fixtures/house-live-joint-archive.py',import.meta.url))],{input:JSON.stringify({destination:join(native,archivePath),raceIds:result.joint_draws.ordered_race_ids,rows,chains:2,sourceRunLengths:[2,2]}),encoding:'utf8'});assert.equal(written.status,0,written.stderr);
 const archiveSha=digest(await readFile(join(native,archivePath)));Object.assign(result.joint_draws,{path:archivePath,sha256:archiveSha});result.chamber.joint_draws_sha256=archiveSha;
 const receipt=JSON.parse(await readFile(join(owner,storage.receipt.path),'utf8'));const joint=receipt.outputs.find(p=>p.path.endsWith('/joint-draws.npz'));await copyFile(join(native,archivePath),join(owner,joint.path));joint.sha256=archiveSha;
 storage.receipt=await put(owner,storage.receipt.path,receipt);
 async function local(pin,path){await mkdir(dirname(join(native,path)),{recursive:true});await copyFile(join(owner,pin.path),join(native,path));return {path,sha256:pin.sha256};}
 const planPin=await local(storage.plan,'local/plan.json'),receiptPin=await local(storage.receipt,'local/receipt.json');
 const reviewValue={schema:'HouseLivePredictionBindingReviewV2',status:'exact_receipt_plan_and_outputs_bound',configuration:receipt.configuration,prediction_receipt:storage.receipt,input_pins:[storage.manifest,storage.plan,storage.receipt]};
 const reviewPin=await put(native,'local/binding-review.json',reviewValue);result.qualification.evidence=[planPin,receiptPin,reviewPin];const resultPin=await put(native,'candidate.json',result);
 const documents=[{pin:planPin,value:JSON.parse(await readFile(join(native,planPin.path),'utf8'))},{pin:receiptPin,value:receipt},{pin:reviewPin,value:reviewValue}];
 return {owner,native,result,storage,resultPin,documents,options:{sourceRoot:native,result:resultPin,seatQuantileConvention:'inverse_cdf',conditionalStorageRoot:owner}};
}

test('explicit V2 offline candidate path authenticates segmented children and preserves raw winner NPZ/public qualification',async t=>{
 const f=await fixture(t),before=JSON.stringify(f.result.qualification);const checked=await checkHouseLiveCandidate(f.options);
 assert.equal(checked.jointArchive.rowsChecked,4);assert.equal(checked.jointArchive.archiveFormat,'house_live_npz_combined_chain_major_v1');
 assert.equal(checked.jointArchive.conditionalStorageManifestSha256,f.storage.manifest.sha256);assert.equal(checked.publicationAdmission,false);
 assert.equal(JSON.stringify(checked.qualification),before);assert.equal(checked.qualification.publication_status,'not_reviewed');
});

test('V2 requires an explicit root, exact storage review and matching versioned receipt',async t=>{
 const f=await fixture(t);await assert.rejects(checkHouseLiveCandidate({...f.options,conditionalStorageRoot:undefined}),/explicit offline conditional storage root/);
 const changes=[docs=>docs.pop(),docs=>docs.push(structuredClone(docs[2])),docs=>docs[1].value.schema='HouseLivePredictionReceiptV1',docs=>docs[0].value.conditional_storage.weights='equal_source_weight',docs=>docs[2].value.prediction_receipt.sha256='9'.repeat(64)];
 for(const change of changes){const docs=structuredClone(f.documents);change(docs);assert.throws(()=>resolveHouseLivePredictionPlan(f.result,docs));}
});

test('manifest or child corruption cannot pass the actual V2 checker and cannot confer publication admission',async t=>{
 const f=await fixture(t);const manifest=JSON.parse(await readFile(join(f.owner,f.storage.manifest.path),'utf8'));
 await appendFile(join(f.owner,manifest.sources[0].path),'changed');await assert.rejects(checkHouseLiveCandidate(f.options),/Source byte hash changed|decoder failed/);
 const g=await fixture(t);await appendFile(join(g.owner,g.storage.manifest.path),'changed');await assert.rejects(checkHouseLiveCandidate(g.options),/Source byte hash changed|decoder failed/);
});

test('the existing CLI accepts an explicit V2 storage root without staging or publication',async t=>{
 const f=await fixture(t);const invoked=spawnSync(process.execPath,['--experimental-strip-types',fileURLToPath(new URL('../scripts/check_house_live_candidate.mjs',import.meta.url)),
  '--source-root',f.native,'--result',f.resultPin.path,'--result-sha256',f.resultPin.sha256,'--seat-quantile-convention','inverse_cdf','--conditional-storage-root',f.owner],{encoding:'utf8'});
 assert.equal(invoked.status,0,invoked.stderr);const checked=JSON.parse(invoked.stdout);assert.equal(checked.jointArchive.conditionalStorageManifestSha256,f.storage.manifest.sha256);assert.equal(checked.publicationAdmission,false);
});
