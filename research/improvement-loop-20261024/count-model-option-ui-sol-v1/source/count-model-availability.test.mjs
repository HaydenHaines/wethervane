import test, { after } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, writeFileSync, mkdirSync, rmSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import ts from 'typescript';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
const root=join(dirname(fileURLToPath(import.meta.url)),'..');
const temporary=mkdtempSync(join(root,'tests','.count-availability-'));
after(()=>rmSync(temporary,{recursive:true,force:true}));
writeFileSync(join(temporary,'package.json'),'{"type":"module"}');
for(const name of ['components/StackControls.tsx','components/ModelSliceControls.tsx','components/CountModelAvailability.tsx','lib/model-slices.ts','lib/forecast-copy.ts','lib/explorer-state.ts']){
  const output=ts.transpileModule(readFileSync(join(root,name),'utf8'),{fileName:name,compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ESNext,jsx:ts.JsxEmit.ReactJSX}}).outputText.replace(/from "(\.{1,2}\/[^".]+)"/g,'from "$1.js"');
  const destination=join(temporary,name.replace(/\.tsx?$/,'.js'));mkdirSync(dirname(destination),{recursive:true});writeFileSync(destination,output);
}
const {default:Controls}=await import(pathToFileURL(join(temporary,'components/StackControls.js')).href);
const registry=JSON.parse(readFileSync(join(root,'public/data/explorer/registry.json'),'utf8'));
const catalog=JSON.parse(readFileSync(join(root,'public/data/explorer/capabilities.json'),'utf8'));
const bundleFor=(id)=>JSON.parse(readFileSync(join(root,registry.runs.find(r=>r.id===id).href.replace(/^\//,'public/')),'utf8'));
function props(id='current-polling'){
  const bundle=bundleFor(id), configuration=catalog.saved.find(c=>c.runId===id);
  return {registry,catalog,bundle,busy:false,compare:false,notice:null,request:{catalogId:catalog.id,year:bundle.run.year,dateId:bundle.run.dateId,components:{...configuration.components}},displayedConfiguration:configuration,resolution:null,savedViews:[],baselineRunId:null,...Object.fromEntries(['onStudy','onYear','onDate','onComponents','onPreset','onCompare','onShare','onReset','onSave','onRestore','onDelete','onBaseline'].map(k=>[k,()=>{}]))};
}
const render=(p)=>renderToStaticMarkup(React.createElement(Controls,p));
const countCard=(html)=>html.split('<section class="count-model-availability"')[1]?.split('</section>')[0];

test('real saved forecasts retain the collapsed builder and unavailable count option',()=>{
  for(const id of ['current-polling','history-2018-FP-1','history-2022-FP-1']){
    const html=render(props(id)),card=countCard(html);
    assert.match(html,/<details class="stack-builder collapsible-stack"/);
    assert.doesNotMatch(html,/<details class="stack-builder collapsible-stack"[^>]*\sopen/);
    assert.match(card,/<button type="button" disabled=""[^>]*>Count model · Not available<\/button>/);
    assert.match(card,/No count-model result is connected/);
    assert.match(card,/No count-model diagnostics connected/);
    assert.match(card,/No supported race or chamber coverage connected/);
    assert.doesNotMatch(card,/<select|type="radio"|aria-pressed|Illustrative data|invented_fixture/);
    assert.match(html,/Preserved presets/);
  }
});

test('unsupported or pending aggregate choices never relabel the displayed result or enable count readiness',()=>{
  for(const state of ['incompatible','requires_computation','ready']){
    const p=props();p.request.components.base='unsupported-request';p.resolution={state,reason:'Test-only requested aggregate status',configurationId:'different',resultId:null,configuration:null,run:null,evidence:null,requiredSteps:[],outputs:[],executionAvailable:false,request:p.request};
    const html=render(p),card=countCard(html);
    assert.match(html,/Still displaying/);
    assert.match(card,/displayed forecast remains <strong>Standard polling<\/strong>/);
    assert.match(card,/Count model · Not available/);
    assert.doesNotMatch(card,/Count model · Ready|count forecast loaded|aria-pressed="true"/);
    assert.match(card,/No new calculation has been requested or scheduled/);
  }
});

test('count target is explained separately from component weights and default promotion',()=>{
  const card=countCard(render(props()));
  assert.match(card,/participation and votes for named candidates/);
  assert.match(card,/cannot be substituted for the electoral-history setting or added as a weight/);
  assert.match(card,/optional result without becoming the default/);
  assert.match(card,/separately from evidence about predictive performance/);
  assert.equal(registry.defaults['2026'],'current-polling');
  assert.equal(catalog.defaultPresetId,'best_guess_polling_20261026_v1');
  assert.equal(catalog.slots.some(s=>s.options.some(o=>/count/i.test(o.id)&&o.availability==='saved')),false);
});
