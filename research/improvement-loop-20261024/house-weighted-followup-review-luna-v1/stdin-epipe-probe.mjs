import assert from 'node:assert/strict';
import {mkdtemp,writeFile,readFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {pathToFileURL} from 'node:url';
const source=process.env.HOUSE_SOURCE;
const {checkWeightedResearchStage}=await import(pathToFileURL(join(source,'lib/house-weighted-research-stage.ts')));
const dir=await mkdtemp(join(tmpdir(),'weighted-stdin-probe-'));
let pid;
try{
 const marker=join(dir,'pid'),fake=join(dir,'close-stdin');
 await writeFile(fake,`#!/usr/bin/env node\nrequire('node:fs').writeFileSync(${JSON.stringify(marker)},String(process.pid));\nprocess.stdin.once('data',()=>{process.stdin.destroy();process.exit(0)});\n`,{mode:0o700});
 const ids=Array.from({length:435},(_,i)=>`2026:R${i}`);
 const result={races:ids.map(race_id=>({race_id,outcomes:[{outcome_id:'D',p_win:1}]})),joint_draws:{count:1,chain_count:1,ordered_race_ids:ids,winner_encoding:{D:0,R:1,O:2}},chamber:null};
 let err;try{await checkWeightedResearchStage(dir,result,{receipt:{path:'stage.json',sha256:'0'.repeat(64)},configuration:{id:'x',base:'y',polling:'standard',uncertainty:'standard'},candidateEdition:'x',forecastIdentity:{office:'house',electionYear:2026,forecastCutoff:'2026-10-26T12:00:00Z',mapVintage:'m'.repeat(2_000_000),scenarioId:'s'},pythonExecutable:fake,timeoutMs:4000})}catch(e){err=String(e)}
 assert.match(String(err),/decoder failed|stream|pipe/i);
 pid=Number(await readFile(marker,'utf8'));
 assert.throws(()=>process.kill(pid,0),e=>e.code==='ESRCH');
 console.log(JSON.stringify({status:'PASS',failure:'early child stdin close rejected',child_reaped:true,error:String(err).slice(0,200)}));
}finally{
 if(pid){try{process.kill(pid,'SIGKILL')}catch{}}
 await rm(dir,{recursive:true,force:true});
}
