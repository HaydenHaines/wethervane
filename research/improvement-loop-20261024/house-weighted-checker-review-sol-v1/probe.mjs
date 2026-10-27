import assert from 'node:assert/strict';
import {checkWeightedJointRows as check} from './pinned/house-weighted-research-check.ts';
function fixture(logs=[0,0]) {
 const max=Math.max(...logs),w=logs.map(x=>Math.exp(x-max)),total=w.reduce((a,b)=>a+b),p=w[0]/total;
 const seats=(a,b)=>{let pmf=Array(436).fill(0);pmf[a]+=p;pmf[b]+=1-p;let quantiles={};for(const [key,q]of Object.entries({p05:.05,p10:.1,p50:.5,p90:.9,p95:.95})){let c=0;quantiles[key]=pmf.findIndex(x=>(c+=x)>=(q-1e-12));}return {pmf,expected:a*p+b*(1-p),quantiles};};
 let result={races:Array.from({length:435},(_,i)=>({race_id:'r'+i,outcomes:[{outcome_id:'D',p_win:p},{outcome_id:'R',p_win:1-p}]})),joint_draws:{count:2,chain_count:1,ordered_race_ids:Array.from({length:435},(_,i)=>'r'+i),winner_encoding:{D:0,R:1,O:2}},chamber:{seats:{D:seats(435,0),R:seats(0,435),O:seats(0,0)},majority:{D:p,R:1-p,unresolved:0}}};
 let rows=logs.map((logFactor,i)=>({jointDrawId:i,predictiveChain:0,predictiveDraw:i,winnerCodes:Array(435).fill(i),logFactor}));return {result,rows};
}
let observations=[];
for(const logs of [[0,0],[10000,10000],[-10000,-10000],[0,-10000],[-10000,0],[Math.log(.05),Math.log(.95)]]){let f=fixture(logs),r=await check(f.result,f.rows);assert.ok(Number.isFinite(r.logWeightSum));assert.ok(r.kishEss>=1&&r.kishEss<=2);if(logs[0]===logs[1]){assert.equal(r.kishEss,2);assert.equal(r.maxNormalizedWeight,.5);}if(Math.abs(logs[0]-logs[1])===10000){assert.equal(r.kishEss,1);assert.equal(r.maxNormalizedWeight,1);}observations.push({logs,...r});}
let f=fixture();for(const mutate of [x=>x.rows.reverse(),x=>x.rows.pop(),x=>x.rows[1].winnerCodes.pop(),x=>x.result.races[1].race_id=x.result.races[0].race_id,x=>x.rows[0].logFactor=Infinity]){let g=structuredClone(f);mutate(g);await assert.rejects(check(g.result,g.rows));}
let boundary=await check(f.result,f.rows);assert.ok(boundary.quantileAmbiguities.some(x=>x.quantile===.5&&x.lower===0&&x.upper===435));
let g=structuredClone(f);g.result.configuration={id:'wrong'};g.result.forecast_cutoff='2099-01-01';g.result.candidate_edition='wrong';await check(g.result,g.rows);
let h=structuredClone(f);h.result.chamber.seats.D.quantiles.p50+=5e-7;await check(h.result,h.rows);
console.log(JSON.stringify({passed:14,observations,metadata_unchecked_by_arithmetic_layer:true,noninteger_quantile_within_tolerance_accepted:true},null,2));
