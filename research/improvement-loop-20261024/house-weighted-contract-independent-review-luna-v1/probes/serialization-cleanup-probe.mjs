import assert from 'node:assert/strict';
import {writeFile,readFile,mkdtemp,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {pathToFileURL} from 'node:url';
const source=process.env.HOUSE_SOURCE;
if(!source)throw Error('Set HOUSE_SOURCE to a clean archive of commit 9a4dc1a.');
const {checkWeightedResearchStage}=await import(pathToFileURL(join(source,'lib/house-weighted-research-stage.ts')));
const dir=await mkdtemp(join(tmpdir(),'house-weighted-cleanup-probe-'));
let pid;
try{
 const pidfile=join(dir,'fake-decoder.pid'),fake=join(dir,'fake-decoder.sh');
 await writeFile(fake,`#!/bin/sh\necho $$ > '${pidfile}'\nexec /bin/sleep 60\n`,{mode:0o700});
 const identity={office:'house',electionYear:2026,forecastCutoff:'2026-10-26T12:00:00Z',mapVintage:'m',scenarioId:'s'};identity.self=identity;
 const request=checkWeightedResearchStage(dir,{joint_draws:{count:1,chain_count:1,ordered_race_ids:[],winner_encoding:{D:0,R:1,O:2}}},{receipt:{path:'stage.json',sha256:'0'.repeat(64)},configuration:{id:'x',base:'y',polling:'standard',uncertainty:'standard'},candidateEdition:'test',forecastIdentity:identity,pythonExecutable:fake});
 await assert.rejects(request,/circular/i);
 for(let i=0;i<100;i++){try{pid=Number((await readFile(pidfile,'utf8')).trim());break}catch{await new Promise(r=>setTimeout(r,10));}}
 assert.ok(Number.isInteger(pid)&&pid>1,'controlled child started');
 let alive=true;try{process.kill(pid,0)}catch{alive=false}
 assert.equal(alive,true,'child remains alive after serialization rejection');
 console.log(JSON.stringify({status:'REPRODUCED',serialization_error:'circular forecastIdentity',child_alive_after_rejection:true}));
}finally{
 if(pid){try{process.kill(pid,'SIGKILL')}catch{}}
 if(pid){for(let i=0;i<100;i++){try{process.kill(pid,0)}catch{pid=undefined;break}await new Promise(r=>setTimeout(r,10));}}
 await rm(dir,{recursive:true,force:true});
}
if(pid)throw Error('Controlled child cleanup did not confirm process disappearance.');
