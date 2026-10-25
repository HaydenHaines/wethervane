import assert from 'node:assert/strict';
import {evaluationScoreboardReadiness as ready,compareSuppliedMetric as compare} from '/home/hayden/projects/ballotmancy-count-comparison-prep-20261025/lib/evaluation-scoreboard-view.ts';
import {completeFixture} from '/home/hayden/projects/ballotmancy-count-comparison-prep-20261025/tests/fixtures/evaluation-scoreboard.mjs';
let checks=0;
for(const metric of ['pair_crps','pair_wis','leading_brier','leading_log','vector_energy']){
const d=completeFixture(),c=d.cycles[0],r=c.models.poll,p=r.pairs[c.pairRoster[0].id];p.observedShare=.54;
assert.equal(ready(d).valid,true);assert.equal(compare(d,c,'count','poll',metric).scopeMatch,'unmatched');assert.equal(compare(d,c,'count','poll',metric).delta,null);checks++;
}
for(const metric of ['report_crps','report_wis','report_log']){
const d=completeFixture(),c=d.cycles[0],r=c.models.poll;r.reports[c.reportRoster[0].id].observedCount=1;
assert.equal(ready(d).valid,true);assert.equal(compare(d,c,'count','poll',metric).scopeMatch,'unmatched');assert.equal(compare(d,c,'count','poll',metric).delta,null);checks++;
}
const d=completeFixture(),c=d.cycles[0];c.models.poll.metrics.pair_crps.value=.04;
const result=compare(d,c,'count','poll','pair_crps');assert.equal(result.informationMatch,'different');assert.equal(result.scopeMatch,'matched');assert.equal(result.delta,.04);checks++;
console.log(JSON.stringify({checks,result:'pass',scope:'Synthetic exact-observation matching and intentional different-input comparison only.'},null,2));
