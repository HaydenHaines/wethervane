import test, { after } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync, rmSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import ts from "typescript";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { modeledWinnerFixture, samePartyWinnerFixture, residualWinnerFixture, sourceReportedNameWinnerFixture, unadmittedWinnerFixture, rankedChoiceWinnerFixture, uncertifiedWinnerFixture, eligibleSingletonWinnerFixture, eligibleUnavailableWinnerFixture, eligibleStructuralWinnerFixture } from "./fixtures/house-winner.mjs";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const directory = mkdtempSync(join(root, "tests", ".house-winner-preview-"));
after(() => rmSync(directory, { recursive: true, force: true }));
writeFileSync(join(directory, "package.json"), '{"type":"module"}\n');
for (const relative of ["components/HouseWinnerPreview.tsx", "lib/house-winner-view.ts"]) {
  const output = ts.transpileModule(readFileSync(join(root, relative), "utf8"), { fileName: relative, compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext, jsx: ts.JsxEmit.ReactJSX, rewriteRelativeImportExtensions: true } }).outputText;
  const destination = join(directory, relative.replace(/\.tsx?$/, ".js"));
  mkdirSync(dirname(destination), { recursive: true }); writeFileSync(destination, output);
}
const { default: Preview } = await import(pathToFileURL(join(directory, "components/HouseWinnerPreview.js")).href);
const render = fixture => renderToStaticMarkup(React.createElement(Preview, { fixture }));
const group = (html, id) => html.split(`data-group="${id}"`)[1]?.split("</section>")[0] || "";
const observed = html => html.split('data-quantity="observed"')[1]?.split("</section>")[0] || "";

test("group chances render once per group with explicit whole-group and rule scope", () => {
  const fixture = modeledWinnerFixture();
  const html = render(fixture);
  assert.match(html, /Illustrative data only/);
  assert.match(html, /Chance each party or group wins/);
  assert.match(html, /Each percentage applies to a whole group/);
  assert.match(html, /does not simulate ranked-choice transfers or a later runoff/);
  assert.equal((html.match(/class="count-value"/g) || []).length, fixture.groups.length);
  for (const entry of fixture.groups) {
    const card = group(html, entry.id);
    assert.match(card, /Modeled chance to win/);
    assert.match(card, /Individual candidate chances are not supplied/);
    assert.doesNotMatch(card.split("<ul>")[1]?.split("</ul>")[0] || "", /%/);
  }
  if (process.env.BALLOTMANCY_WINNER_PREVIEW_OUTPUT) {
    const css = readFileSync(join(root, "app/globals.css"), "utf8");
    const scenarios = [modeledWinnerFixture(), samePartyWinnerFixture(), residualWinnerFixture(), sourceReportedNameWinnerFixture(), unadmittedWinnerFixture(), rankedChoiceWinnerFixture(), uncertifiedWinnerFixture(), eligibleSingletonWinnerFixture(), eligibleUnavailableWinnerFixture(), eligibleStructuralWinnerFixture()].map(render).join("\n");
    writeFileSync(process.env.BALLOTMANCY_WINNER_PREVIEW_OUTPUT, `<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Invented House group winner preview</title><style>${css}</style><body><main style="padding:20px">${scenarios}</main></body></html>`);
  }
});

test("structural group certainty does not split between same-party members", () => {
  const fixture = samePartyWinnerFixture();
  const html = render(fixture);
  assert.match(html, /All listed candidates are in this group/);
  assert.match(html, /100% <span>among the listed candidates/);
  assert.match(html, /not a guarantee about the election/);
  assert.equal((html.match(/class="count-value"/g) || []).length, 1);
  assert.doesNotMatch(html, /Modeled chance to win/);
  assert.ok(fixture.groups[0].members.length > 1);
});

test("a singleton pooled source report stays a category, with no candidate probability", () => {
  const fixture = residualWinnerFixture();
  const residual = fixture.groups.find(entry => entry.members.some(member => member.kind === "pooled_or_residual"));
  assert.ok(residual);
  const html = render(fixture), card = group(html, residual.id);
  assert.match(card, /Pooled report:/);
  assert.match(card, /It is not one candidate/);
  assert.match(card, /Other candidates &amp; write-ins/);
  assert.doesNotMatch(card.split("<ul>")[1]?.split("</ul>")[0] || "", /%/);
});

test("source-reported names are explicitly not admitted individual candidates", () => {
  const fixture = sourceReportedNameWinnerFixture();
  fixture.groups[2].namedSingletonResolutionEligible = false;
  const card = group(render(fixture), "O");
  assert.match(card, /Source-reported name:<\/strong> Source-listed entry 7/);
  assert.match(card, /does not by itself identify a verified candidate/);
  assert.match(card, /Individual candidate chances are not supplied/);
  assert.doesNotMatch(card, /Chance applies to that person|Source-listed entry 7<\/h3>/);
  const invalid = sourceReportedNameWinnerFixture();
  invalid.groups[2].namedSingletonResolutionEligible = true;
  assert.throws(() => render(invalid), TypeError);
});

test("missing prediction does not turn sole roster support into certainty or hide recorded truth", () => {
  const fixture = unadmittedWinnerFixture();
  const html = render(fixture);
  assert.match(html, /Winner forecast unavailable/);
  assert.match(html, /Chance unavailable/);
  assert.doesNotMatch(html, /class="count-value"|All listed candidates are in this group/);
  if (fixture.truth.status === "observed") assert.match(observed(html), /Recorded winner’s group/);
});

test("recorded final outcomes and absent certified truth remain separate from forecasts", () => {
  const ranked = render(rankedChoiceWinnerFixture());
  assert.match(ranked, /Election rule: <strong>Ranked choice/);
  assert.match(observed(ranked), /Recorded winner’s group/);
  assert.match(ranked, /First-ballot party vote share is a separate measure/);
  const uncertified = render(uncertifiedWinnerFixture());
  assert.match(observed(uncertified), /Recorded winner unavailable. Not scored/);
  assert.doesNotMatch(observed(uncertified), /Recorded winner’s group|Reported winner:/);
  assert.match(uncertified, /class="count-value"/);
});

test("modeled endpoints and rounding near endpoints never claim structural certainty", () => {
  const fixture = modeledWinnerFixture();
  fixture.prediction.probabilities.forEach((row, index) => row.probability = index === 0 ? 1 : 0);
  let html = render(fixture);
  assert.match(html, /model assigns 100.0%/);
  assert.match(html, /not a guarantee/);
  assert.doesNotMatch(html, /All listed candidates are in this group/);
  fixture.prediction.probabilities.forEach((row, index) => row.probability = index === 0 ? .9999 : index === 1 ? .0001 : 0);
  html = render(fixture);
  assert.match(html, /&gt;99.9%/);
  assert.match(html, /&lt;0.1%/);
  assert.doesNotMatch(html, /class="count-value">100.0%/);
});

test("explicitly eligible named candidate displays the whole-group chance without duplicating it", () => {
  const fixture = eligibleSingletonWinnerFixture();
  const card = group(render(fixture), "D");
  assert.match(card, /<h3>Dana Example<\/h3>/);
  assert.match(card, /60.0%/);
  assert.match(card, /displayed chance applies to that person/);
  assert.equal((card.match(/class="count-value"/g) || []).length, 1);
  assert.doesNotMatch(card, /Individual candidate chances are not supplied/);
  const structural = group(render(eligibleStructuralWinnerFixture()), "D");
  assert.match(structural, /<h3>Dana Example<\/h3>/);
  assert.match(structural, /100% <span>among the listed candidates/);
  assert.match(structural, /not a guarantee about the election/);
});

test("eligibility alone never changes a pending or malformed prediction into a candidate chance", () => {
  const unavailable = eligibleUnavailableWinnerFixture();
  const invalid = eligibleSingletonWinnerFixture();
  invalid.prediction.probabilities[0].probability = NaN;
  for (const fixture of [unavailable, invalid]) {
    const card = group(render(fixture), "D");
    assert.match(card, /<h3>Democratic group<\/h3>/);
    assert.match(card, /a valid forecast is not available yet/);
    assert.doesNotMatch(card, /class="count-value"|<h3>Dana Example<\/h3>|displayed chance applies/);
  }
});

test("private group preview remains absent from production callers and fixture imports", () => {
  function scan(directory) { return readdirSync(directory, { withFileTypes: true }).flatMap(entry => entry.isDirectory() ? scan(join(directory, entry.name)) : /\.[cm]?[jt]sx?$/.test(entry.name) ? [join(directory, entry.name)] : []); }
  for (const base of ["app", "components", "lib", "scripts"]) for (const file of scan(join(root, base))) {
    if (["/HouseWinnerPreview.tsx", "/house-winner-view.ts", "/HouseGroupRibbonPreview.tsx", "/house-group-ribbon.ts"].some(name => file.endsWith(name))) continue;
    assert.doesNotMatch(readFileSync(file, "utf8"), /(?:from\s*|import\s*\(|require\s*\()["'][^"']*(?:house-winner|HouseWinnerPreview)/, file);
  }
});
