import test, { after } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync, rmSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import ts from "typescript";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { partialPairComparisonFixture, missingObservationFixture, incompatiblePairFixture } from "./fixtures/candidate-pair-comparison.mjs";

const root=join(dirname(fileURLToPath(import.meta.url)),"..");
const directory=mkdtempSync(join(root,"tests",".candidate-pair-preview-"));
after(()=>rmSync(directory,{recursive:true,force:true}));
writeFileSync(join(directory,"package.json"),'{"type":"module"}\n');
for(const relative of ["components/CandidatePairComparisonPreview.tsx","lib/candidate-pair-comparison-view.ts"]) {
  const js=ts.transpileModule(readFileSync(join(root,relative),"utf8"),{fileName:relative,compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ESNext,jsx:ts.JsxEmit.ReactJSX,rewriteRelativeImportExtensions:true}}).outputText;
  const output=join(directory,relative.replace(/\.tsx?$/,".js"));mkdirSync(dirname(output),{recursive:true});writeFileSync(output,js);
}
const {default:Preview}=await import(pathToFileURL(join(directory,"components/CandidatePairComparisonPreview.js")).href);
const render=view=>renderToStaticMarkup(React.createElement(Preview,{view}));

test("pair denominator, differing information and qualification stay visible beside supplied numbers",()=>{
  const html=render(partialPairComparisonFixture());
  assert.match(html,/Illustrative data only · Private comparison study/);
  assert.match(html,/not an owner-approved format for real results/);
  assert.match(html,/Votes included in the share: Alex Harbor Example plus Blair Harbor Example/);
  assert.match(html,/neither all-candidate share nor a probability of winning/);
  assert.match(html,/60% <span>average pair share \(mean\)/);
  assert.match(html,/Experimental/);assert.match(html,/Model checks/);assert.match(html,/does not approve a forecast or change the public default/);
  assert.match(html,/Different across these models/);assert.match(html,/cannot tell us what polling alone changed/);
  assert.match(html,/lead this pair and still trail a third person/);
  assert.doesNotMatch(html,/Alex Harbor Example wins|type="checkbox"|<select/);
});

test("score units, meanings and accessible heading levels are explained in plain language",()=>{
  const html=render(partialPairComparisonFixture());
  assert.match(html, /mean is the average of possible pair shares/);
  assert.match(html, /median is the middle predicted share/);
  assert.match(html, /compares all the possible shares and the probabilities assigned to them/);
  assert.match(html, /rewards narrower prediction ranges and penalizes ranges that miss/);
  assert.match(html, /All three errors are shown in percentage-point units/);
  assert.match(html, /0 is perfect and lower is better/);
  assert.match(html, /<dt>Full-distribution error \(CRPS\)<\/dt><dd[^>]*>8 points/);
  assert.match(html, /<h4>Experimental vote-count model<\/h4>/);
  assert.match(html, /<dd style="white-space:nowrap">12 points<\/dd>/);
  assert.match(html, /<h5>Errors against the invented result<\/h5>/);
  assert.doesNotMatch(html, /Posterior qualification|Adoption status|Information sets|renormalized|Monte Carlo error bars/);
  assert.doesNotMatch(html.split('<details class="count-checks">')[0], /invented-Harbor-race|invented-Harbor-event|model-Harbor-first/);
});

test("coverage retains missing and undefined pairs instead of advertising a complete mean loss",()=>{
  const html=render(partialPairComparisonFixture());
  assert.match(html,/1 of 3 expected pairs have scores for all three models/);
  assert.match(html,/Results are supplied for 2 of 3 pairs/);
  assert.match(html,/No result supplied for this expected pair/);
  assert.match(html,/Alex Ridge Example/);
  assert.match(html,/share cannot be calculated in 20% of the predicted outcomes/);
  assert.match(html,/remaining outcomes do not get extra weight/);
  assert.match(html,/We do not know how often the share could be calculated/);
  assert.match(html,/No overall average error is shown using only the pairs that happened to have results/);
  assert.match(html,/Scores unavailable/);assert.match(html,/No perfect score is substituted/);
});

test("tiny positive and nearly complete undefined shares never round to exact zero or one",()=>{
  for(const [mass,label] of [[1e-100,"less than 0.1%"],[.0001,"less than 0.1%"],[1-1e-10,"more than 99.9%"],[1,"100%"]]) {
    const view=partialPairComparisonFixture();view.rows[1].arms.count.distribution.undefinedMass=mass;
    const html=render(view);
    assert.ok(html.includes(`share cannot be calculated in ${label} of the predicted outcomes`));
    assert.doesNotMatch(html,/share cannot be calculated in 0% of/);
    if(mass<1)assert.doesNotMatch(html,/share cannot be calculated in 100% of/);
  }
});

test("missing truth preserves predictive summaries without a fabricated score",()=>{
  const html=render(missingObservationFixture());
  assert.match(html,/Observed pair share unavailable/);assert.match(html,/60% <span>average pair share \(mean\)/);
  assert.match(html,/0 of 3 expected pairs have scores/);assert.doesNotMatch(html,/<dt>Full-distribution error \(CRPS\)<\/dt>/);
});

test("simulation precision is not confused with a predictive interval",()=>{
  const html=render(partialPairComparisonFixture());
  assert.match(html,/80% predictive range: 30%–75%/);
  assert.match(html,/Calculation precision unavailable/);
  assert.match(html,/not that calculation error/);
  assert.doesNotMatch(html,/significant improvement|qualified posterior|±/i);
});

test("incompatible identities or denominators suppress numerical display",()=>{
  const denominator=partialPairComparisonFixture();denominator.rows[0].arms.count.binding.denominator.kind="all_candidate_votes";
  for(const value of [incompatiblePairFixture(),denominator,null]) {
    const html=render(value);assert.match(html,/Comparison unavailable/);
    assert.doesNotMatch(html,/count-value|60%|<dt>CRPS<\/dt>/);
  }
});

test("preview and fixture stay out of all existing production pages, APIs and validators",()=>{
  const files=path=>readdirSync(path,{withFileTypes:true}).flatMap(entry=>entry.isDirectory()?files(join(path,entry.name)):[join(path,entry.name)]);
  for(const file of [...files(join(root,"app")),...files(join(root,"components")),...files(join(root,"lib"))].filter(file=>/\.[cm]?[jt]sx?$/.test(file)&&!file.endsWith("CandidatePairComparisonPreview.tsx")&&!file.endsWith("candidate-pair-comparison-view.ts")))
    assert.doesNotMatch(readFileSync(file,"utf8"),/(?:from\s*|import\s*\(|require\s*\()["'][^"']*(?:CandidatePairComparisonPreview|candidate-pair-comparison-view|fixtures\/candidate-pair-comparison)/,file);
});

test("export optional standalone private examples for visual review",{skip:!process.env.BALLOTMANCY_PAIR_PREVIEW_OUTPUT},()=>{
  const css=readFileSync(join(root,"app/globals.css"),"utf8");
  const html=[partialPairComparisonFixture,missingObservationFixture,incompatiblePairFixture].map(f=>render(f())).join('<hr style="margin:60px 0">');
  writeFileSync(process.env.BALLOTMANCY_PAIR_PREVIEW_OUTPUT,`<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Invented candidate-pair comparison</title><style>${css}</style><body><main style="padding:20px">${html}</main></body></html>`);
});
