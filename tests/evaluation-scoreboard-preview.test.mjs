import test, { after } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync, rmSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import ts from "typescript";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { evaluationFixture, completeFixture, scoredPair, scoredReport, refreshMetrics } from "./fixtures/evaluation-scoreboard.mjs";

const root=join(dirname(fileURLToPath(import.meta.url)),"..");
const directory=mkdtempSync(join(root,"tests",".evaluation-preview-"));
after(()=>rmSync(directory,{recursive:true,force:true}));
writeFileSync(join(directory,"package.json"),'{"type":"module"}\n');
for(const relative of ["components/EvaluationScoreboardPreview.tsx","lib/evaluation-scoreboard-view.ts"]){
  const output=ts.transpileModule(readFileSync(join(root,relative),"utf8"),{fileName:relative,compilerOptions:{
    target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ESNext,jsx:ts.JsxEmit.ReactJSX,rewriteRelativeImportExtensions:true,
  }}).outputText;
  const destination=join(directory,relative.replace(/\.tsx?$/,".js"));mkdirSync(dirname(destination),{recursive:true});writeFileSync(destination,output);
}
const {default:Preview}=await import(pathToFileURL(join(directory,"components/EvaluationScoreboardPreview.js")).href);
const {evaluationScoreboardReadiness,coverageFor,compareSuppliedMetric}=await import(pathToFileURL(join(directory,"lib/evaluation-scoreboard-view.js")).href);
const render=(view=evaluationFixture(),selectedCycleId="2032",selectedModelId="count",compareModelId="poll")=>
  renderToStaticMarkup(React.createElement(Preview,{view,selectedCycleId,selectedModelId,compareModelId}));

test("invented provenance, separate statuses, named-pair and report meanings render",()=>{
  const html=render();
  assert.match(html,/Illustrative data only/);assert.match(html,/Declared hashes do not authenticate real data/);
  assert.match(html,/Evidence<\/dt><dd>experimental/);assert.match(html,/Public adoption<\/dt><dd>not qualified/);
  assert.match(html,/Different inputs/);assert.match(html,/Scope match:<\/strong> matched/);
  assert.match(html,/strictly/);assert.match(html,/not legal winner or chamber probability/);
  assert.match(html,/No cross-definition or cross-cycle mean/);
  assert.match(html,/2 historical cycles supplied/);
  assert.match(html,/0 report counts/);assert.match(html,/Inactive event/);
  assert.match(html,/Source variance unknown/);
  assert.match(html,/fresh illustrative release effect/);
  assert.doesNotMatch(html,/<dd>qualified for scope|97 independent|203 historical|four historical cycles/i);
});

test("partial values remain labelled partial and expected rosters retain missing rows",()=>{
  const d=evaluationFixture(),c=d.cycles[0];
  assert.deepEqual(coverageFor(c,"count"),{pairs:{expected:2,supplied:2,scored:1},reports:{expected:2,supplied:2,observed:1,scored:1}});
  const html=render(d);
  assert.match(html,/Supplied partial score/);assert.match(html,/1 of 2 expected named pairs scored/);
  assert.match(html,/no full-roster mean is claimed/);assert.match(html,/1 of 2 expected recorded results covered \(partial: 1 evaluable\)/);
  assert.match(html,/Recorded outcome missing/);assert.match(html,/inactive; count and mean are unavailable/);
  assert.doesNotMatch(html,/Complete supplied score/);
});

test("same scored scope permits descriptive delta despite different inputs",()=>{
  const d=completeFixture(),c=d.cycles[0];c.models.count.metrics.pair_crps.value=.06;
  const cmp=compareSuppliedMetric(d,c,"count","poll","pair_crps");
  assert.equal(cmp.scopeMatch,"matched");assert.equal(cmp.informationMatch,"different");assert.ok(Math.abs(cmp.delta+.02)<1e-12);
  const html=render(d);assert.match(html,/Descriptive difference:/);assert.match(html,/2 percentage points/);
  assert.match(html,/does not isolate a component effect/);
});

test("target, ordered roster, cutoff, outcome, units, report profile and scored IDs guard comparison",()=>{
  for(const edit of [
    m=>m.binding.target="other target",m=>m.binding.orderedRosterIds.reverse(),m=>m.binding.cutoff="2032-10-20T00:00:00Z",
    m=>m.binding.outcomeView="other observations",m=>m.binding.unit="report_count",m=>m.binding.metricDefinition="other definition",
    m=>m.binding.weighting="other weights",m=>m.scoredIds.pop(),
  ]){const d=completeFixture(),c=d.cycles[0];edit(c.models.poll.metrics.pair_crps);
    assert.equal(compareSuppliedMetric(d,c,"count","poll","pair_crps").delta,null);
  }
  const d=completeFixture(),c=d.cycles[0];c.models.poll.metrics.report_crps.binding.reportProfile="different profile";
  assert.equal(compareSuppliedMetric(d,c,"count","poll","report_crps").scopeMatch,"unmatched");
  assert.equal(evaluationScoreboardReadiness(d).valid,false); // report profile must match the actual roster
  c.models.poll.metrics.report_crps.binding.unit="share_fraction";
  assert.equal(evaluationScoreboardReadiness(d).valid,false);
});

test("matching outcome-view declarations cannot hide different observed values",()=>{
  const d=completeFixture(),c=d.cycles[0];
  c.models.poll.pairs[c.pairRoster[0].id].observedShare=.55;
  assert.equal(evaluationScoreboardReadiness(d).valid,true);
  assert.equal(compareSuppliedMetric(d,c,"count","poll","pair_crps").scopeMatch,"unmatched");
  assert.equal(compareSuppliedMetric(d,c,"count","poll","vector_energy").delta,null);
  c.models.poll.pairs[c.pairRoster[0].id].observedShare=.53;
  c.models.poll.reports[c.reportRoster[0].id].observedCount=1;
  assert.equal(evaluationScoreboardReadiness(d).valid,true);
  assert.equal(compareSuppliedMetric(d,c,"count","poll","report_crps").scopeMatch,"unmatched");
});

test("arbitrary model and cycle counts are taken from the supplied roster",()=>{
  const d=evaluationFixture();d.models.push({id:"third",name:"Third invented stack",ingredients:["Different input"],informationKey:"third"});
  for(const c of d.cycles)c.models.third=structuredClone(c.models.count);
  d.cycles.push(structuredClone(d.cycles[1]));d.cycles[2].id="2040";d.cycles[2].label="2040 invented cycle";
  d.cycles[2].cutoff="2040-10-19T00:00:00Z";d.cycles[2].outcomeView="invented-2040-outcomes";
  for(const r of Object.values(d.cycles[2].models))for(const m of Object.values(r.metrics)){
    m.binding.cutoff=d.cycles[2].cutoff;m.binding.outcomeView=d.cycles[2].outcomeView;
  }
  assert.equal(evaluationScoreboardReadiness(d).valid,true);
  const html=render(d,"2040","third");assert.match(html,/3 historical cycles supplied/);assert.match(html,/Third invented stack/);
});

test("missing outcome, positive tiny undefined mass, withheld vector and active zero remain distinct",()=>{
  const d=evaluationFixture(),c=d.cycles[0],p=c.pairRoster[0].id;
  c.models.count.pairs[p]={state:"undefined_mass",undefinedMass:1e-100,observedShare:null,crps:null,wis:null,
    intervals:[],leading:null,reason:"One positive-weight row has a zero denominator."};
  c.models.count.energy.positiveWeightMissingIds=[p];c.models.count.energy.undefinedMass=1e-100;
  refreshMetrics(d);const html=render(d);
  assert.match(html,/less than 0.1% of predicted outcomes/);assert.doesNotMatch(html,/Share undefined in 0%/);
  assert.match(html,/positive-weight missing coordinates/);assert.match(html,/Active event with a recorded total of 0 report counts/);
  assert.match(html,/Recorded outcome missing/);
  c.models.count.energy.state="scored";c.models.count.energy.value=.08;
  assert.equal(evaluationScoreboardReadiness(d).valid,false);
});

test("infinite loss is distinct from unavailable and true zero; precision remains separate",()=>{
  const d=completeFixture(),r=d.cycles[0].models.count;
  for(const p of Object.values(r.pairs)){p.leading.logLoss=null;p.leading.logLossInfinite=true;}
  r.metrics.leading_log.value=null;r.metrics.leading_log.infinite=true;
  r.metrics.pair_crps.value=0;r.metrics.pair_crps.precision={status:"limited",note:"Invented Monte Carlo check did not meet its supplied rule.",rule:"Invented rule A"};
  assert.equal(evaluationScoreboardReadiness(d).valid,true);
  const html=render(d);assert.match(html,/∞ \(infinite loss\)/);assert.match(html,/limited/);
  assert.match(html,/white-space:nowrap;font:inherit;color:inherit">0 <small[^>]*>percentage points<\/small><\/span>/);
  assert.match(html,/Invented rule A/);assert.match(html,/2 of 2 expected named pairs scored/);
});

test("each displayed score keeps its number and unit in one unbreakable group",()=>{
  const d=completeFixture(),r=d.cycles[0].models.count;
  r.metrics.pair_crps.value=1e-100;
  for(const p of Object.values(r.pairs)){p.leading.logLoss=null;p.leading.logLossInfinite=true;}
  r.metrics.leading_log.value=null;r.metrics.leading_log.infinite=true;
  const html=render(d);
  const cards=[...html.matchAll(/<article class="count-quantity" aria-label="[^"]+"[\s\S]*?<\/article>/g)];
  assert.equal(cards.length,8);
  for(const card of cards){
    if(!card[0].includes('Supplied score')&&!card[0].includes('Complete supplied score'))continue;
    assert.match(card[0],/<span style="white-space:nowrap;font:inherit;color:inherit">[^<]+ <small[^>]*>(?:percentage points|report counts|dimensionless)<\/small><\/span>/);
  }
  assert.match(html,/1\.00e-98 <small[^>]*>percentage points<\/small>/);
  assert.match(html,/∞ <small[^>]*>dimensionless<\/small>/);
});

test("incompatible units, positive undefined mass, missing roster and incomplete vector reject a complete claim",()=>{
  const edits=[
    d=>d.cycles[0].models.count.metrics.pair_crps.binding.unit="report_count",
    d=>{d.cycles[0].models.count.pairs[d.cycles[0].pairRoster[0].id].undefinedMass=.01;},
    d=>{delete d.cycles[0].models.count.pairs[d.cycles[0].pairRoster[0].id];},
    d=>{d.cycles[0].models.count.energy.positiveWeightMissingIds=[d.cycles[0].pairRoster[0].id];},
  ];
  for(const edit of edits){const d=completeFixture();edit(d);assert.equal(evaluationScoreboardReadiness(d).valid,false);}
});

test("energy and metric bindings retain the exact ordered pair vector",()=>{
  for(const edit of [
    d=>d.cycles[0].models.count.energy.orderedPairIds.reverse(),
    d=>d.cycles[0].models.count.metrics.pair_crps.binding.orderedRosterIds.reverse(),
    d=>{const p=d.cycles[0].pairRoster[0].id;d.cycles[0].models.count.pairs[p].state="outcome_missing";
      d.cycles[0].models.count.pairs[p].observedShare=null;d.cycles[0].models.count.pairs[p].crps=null;d.cycles[0].models.count.pairs[p].wis=null;},
  ]){const d=completeFixture();edit(d);assert.equal(evaluationScoreboardReadiness(d).valid,false);}
});

test("contradictory infinite flags, unknown statuses and unsafe positive-mass summaries are rejected",()=>{
  const edits=[
    d=>{d.cycles[0].models.count.pairs[d.cycles[0].pairRoster[0].id].leading.logLossInfinite=true;},
    d=>{d.cycles[0].models.count.reports[d.cycles[0].reportRoster[0].id].logLossInfinite=true;},
    d=>{d.cycles[0].models.count.metrics.pair_crps.value=null;d.cycles[0].models.count.metrics.pair_crps.infinite=true;},
    d=>{d.cycles[0].models.count.metrics.pair_crps.precision.status="magic";},
    d=>{d.status.evidence="qualified_for_scope";},
    d=>{const p=d.cycles[0].models.count.pairs[d.cycles[0].pairRoster[0].id];p.state="undefined_mass";p.undefinedMass=.01;p.crps=null;p.wis=null;},
    d=>{const q=d.cycles[0].models.count.reports[d.cycles[0].reportRoster[0].id];q.state="undefined_mass";q.undefinedMass=.01;q.crps=null;q.wis=null;},
  ];
  for(const edit of edits){const d=completeFixture();edit(d);assert.equal(evaluationScoreboardReadiness(d).valid,false);}
});

test("a forecast summary remains readable when its observation is missing",()=>{
  const d=evaluationFixture(),c=d.cycles[0],id=c.reportRoster[1].id;
  c.models.count.reports[id]={state:"outcome_missing",eventPresent:true,undefinedMass:0,observedCount:null,
    predictedMean:12,crps:null,wis:null,intervals:[{level:.8,lower:5,upper:20,width:15,covered:null}],
    logLoss:null,logLossInfinite:false,reason:"Recorded report unavailable."};
  assert.equal(evaluationScoreboardReadiness(d).valid,true);
  const html=render(d);assert.match(html,/Predicted mean 12 report counts; recorded total unavailable/);
  assert.match(html,/width 15 report counts; recorded total not assessed/);
  assert.match(html,/width 10 report counts to 15 report counts across supplied ranges/);
  assert.match(html,/Forecast chance that the first candidate strictly leads this named pair: 48%/);
});

test("missing outcomes cannot claim interval coverage or impossible leading probability",()=>{
  for(const edit of [
    d=>{const c=d.cycles[0],p=c.models.count.pairs[c.pairRoster[1].id];p.leading.probability=2;},
    d=>{const c=d.cycles[0],p=c.models.count.pairs[c.pairRoster[1].id];p.intervals[0].covered=true;},
    d=>{const c=d.cycles[0],r=c.models.count.reports[c.reportRoster[1].id];Object.assign(r,{state:"outcome_missing",eventPresent:true,undefinedMass:0,observedCount:null,
      predictedMean:12,crps:null,wis:null,intervals:[{level:.8,lower:5,upper:20,width:15,covered:true}],logLoss:null,logLossInfinite:false});},
    d=>{const c=d.cycles[0],r=c.models.count.reports[c.reportRoster[1].id];Object.assign(r,{state:"outcome_missing",eventPresent:true,undefinedMass:0,observedCount:null,
      predictedMean:-1,crps:null,wis:null,intervals:[],logLoss:null,logLossInfinite:false});},
  ]){const d=evaluationFixture();edit(d);assert.equal(evaluationScoreboardReadiness(d).valid,false);}
});

test("malformed status, metric and vector records fail closed without throwing",()=>{
  for(const edit of [
    d=>{d.status.run="magic";},d=>{d.cycles[0].models.count.energy.state="magic";},
    d=>{d.cycles[0].models.count.metrics.pair_crps.state="magic";},
    d=>{d.cycles[0].models.count.metrics.pair_crps.scoredIds=null;},
    d=>{d.cycles[0].models.count.pairs[d.cycles[0].pairRoster[0].id].intervals=null;},
    d=>{d.cycles[0].models.count.energy.positiveWeightMissingIds=null;},
  ]){const d=evaluationFixture();edit(d);assert.doesNotThrow(()=>evaluationScoreboardReadiness(d));
    assert.equal(evaluationScoreboardReadiness(d).valid,false);}
});

test("strict-leading, interval coverage and whole-vector values agree with supplied observations",()=>{
  const edits=[
    d=>{const c=d.cycles[0],p=c.models.count.pairs[c.pairRoster[0].id];p.observedShare=.5;},
    d=>{const c=d.cycles[0],p=c.models.count.pairs[c.pairRoster[0].id];p.intervals[0].covered=false;},
    d=>{const c=d.cycles[0],r=c.models.count.reports[c.reportRoster[0].id];r.intervals[0].covered=false;},
    d=>{d.cycles[0].models.count.metrics.vector_energy.value=.08;},
  ];
  for(const edit of edits){const d=completeFixture();edit(d);assert.equal(evaluationScoreboardReadiness(d).valid,false);}
});

test("standalone static HTML export for visual review",{skip:!process.env.BALLOTMANCY_EVALUATION_PREVIEW_OUTPUT},()=>{
  const css=readFileSync(join(root,"app/globals.css"),"utf8");
  const extremes=completeFixture(),r=extremes.cycles[0].models.count;
  r.metrics.pair_crps.value=1e-100;
  for(const p of Object.values(r.pairs)){p.leading.logLoss=null;p.leading.logLossInfinite=true;}
  r.metrics.leading_log.value=null;r.metrics.leading_log.infinite=true;
  extremes.label="Tiny finite and infinite loss · illustrative";
  const html=[render(evaluationFixture()),render(completeFixture()),render(extremes)].join('<hr style="margin:60px 0">');
  writeFileSync(process.env.BALLOTMANCY_EVALUATION_PREVIEW_OUTPUT,`<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Invented evaluation scoreboard</title><style>${css}</style><body><main style="padding:16px;min-width:0">${html}</main></body></html>`);
});
