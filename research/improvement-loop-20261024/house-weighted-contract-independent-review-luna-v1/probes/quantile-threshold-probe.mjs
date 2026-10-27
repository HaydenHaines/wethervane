import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
import {join} from 'node:path';
const source=process.env.HOUSE_SOURCE;
if(!source)throw Error('Set HOUSE_SOURCE to a clean archive of commit 9a4dc1a.');
const {checkWeightedJointRows}=await import(pathToFileURL(join(source,'lib/house-weighted-research-check.ts')));
const qs=[['p05',.05],['p10',.1],['p50',.5],['p90',.9],['p95',.95]],ids=Array.from({length:435},(_,i)=>`2026:R${i}`);
async function run(logs){
 const w=logs.map(Math.exp),total=w.reduce((a,b)=>a+b,0),rows=[0,1].map((j)=>({jointDrawId:j,predictiveChain:j,predictiveDraw:0,winnerCodes:Array(435).fill(j),logFactor:logs[j]}));
 const parties=['D','R','O'],result={races:ids.map(race_id=>({race_id,outcomes:[{outcome_id:'D',p_win:w[0]/total},{outcome_id:'R',p_win:w[1]/total}]})),joint_draws:{count:2,chain_count:2,ordered_race_ids:ids,winner_encoding:{D:0,R:1,O:2}},chamber:{seats:{},majority:{D:w[0]/total,R:w[1]/total,unresolved:0}}};
 for(const [k,p] of parties.entries()){
  const mass=Array(436).fill(0);if(k===0){mass[435]=w[0];mass[0]=w[1]}else if(k===1){mass[0]=w[0];mass[435]=w[1]}else mass[0]=total;
  const pmf=mass.map(x=>x/total),quantiles={};for(const [name,q] of qs){let cum=0;for(let s=0;s<436;s++){cum+=mass[s];if(cum >= (q-1e-12)*total){quantiles[name]=s;break}}}
  result.chamber.seats[p]={expected:pmf.reduce((s,x,i)=>s+x*i,0),pmf,quantiles};
 }
 return checkWeightedJointRows(result,rows);
}
const tie=await run([0,0]);assert.equal(tie.quantileAmbiguities.filter(x=>x.quantile===.5).length,2);
const inside=await run([0,Math.log(1+2e-12)]);assert.equal(inside.quantileAmbiguities.filter(x=>x.quantile===.5).length,2);
const outside=await run([0,Math.log(1+8e-12)]);assert.equal(outside.quantileAmbiguities.filter(x=>x.quantile===.5).length,0);
console.log(JSON.stringify({status:'PASS',exactTie:tie.quantileAmbiguities,justInsideTolerance:inside.quantileAmbiguities,justOutsideTolerance:outside.quantileAmbiguities}));
