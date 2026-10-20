import test, { after } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync, rmSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import ts from "typescript";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { partialTexasFixture, failedParentFixture, unsupportedRuleFixture, unresolvedRaceTieFixture, absentContestFixture, chamberTieScenarioFixture } from "./fixtures/count-view.mjs";

// Compile only the private presentation component for a real React server render.
// No test route, public fixture endpoint or scientific module is introduced.
const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const directory = mkdtempSync(join(root, "tests", ".count-preview-"));
after(() => rmSync(directory, { recursive: true, force: true }));
writeFileSync(join(directory, "package.json"), '{"type":"module"}\n');
for (const relative of ["components/CountForecastPreview.tsx", "lib/count-view.ts"]) {
  const source = readFileSync(join(root, relative), "utf8");
  const output = ts.transpileModule(source, { fileName: relative, compilerOptions: {
    target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext, jsx: ts.JsxEmit.ReactJSX, rewriteRelativeImportExtensions: true,
  } }).outputText;
  const destination = join(directory, relative.replace(/\.tsx?$/, ".js"));
  mkdirSync(dirname(destination), { recursive: true }); writeFileSync(destination, output);
}
const { default: Preview } = await import(pathToFileURL(join(directory, "components/CountForecastPreview.js")).href);
const render = (view) => renderToStaticMarkup(React.createElement(Preview, { view }));
const quantity = (html, id) => html.split(`data-quantity="${id}"`)[1]?.split("</div>")[0] || "";

test("fixture render distinguishes reports, predictions, units, ranges and exposure ratios above one", () => {
  const html = render(partialTexasFixture());
  assert.match(html, /Illustrative data only/);
  assert.match(html, /No full-Senate control estimate/);
  assert.match(quantity(html, "participants"), /100<span> people/);
  assert.match(quantity(html, "F1a-report"), /102<span> people/);
  assert.match(quantity(html, "F1a-report"), /Reported observation/);
  assert.match(quantity(html, "participants"), /80% predicted range/);
  assert.match(quantity(html, "turnout"), /115%/);
  assert.match(quantity(html, "turnout"), /Citizen-adult exposure/);
  assert.match(quantity(html, "D-share"), /All countable candidate votes/);
  assert.doesNotMatch(html, /type="checkbox"|<select/);
  if (process.env.BALLOTMANCY_PREVIEW_OUTPUT) {
    const css = readFileSync(join(root, "app/globals.css"), "utf8");
    writeFileSync(process.env.BALLOTMANCY_PREVIEW_OUTPUT, `<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Invented count preview</title><style>${css}</style><body><main style="padding:20px">${html}</main></body></html>`);
  }
});

test("a failed population state withholds all predicted numbers while retaining the separately dated report", () => {
  const html = render(failedParentFixture());
  assert.match(html, /some population states failed or are missing/);
  assert.match(quantity(html, "participants"), /Not available/);
  assert.doesNotMatch(quantity(html, "participants"), /count-value/);
  assert.match(quantity(html, "F1a-report"), /102<span> people/);
  assert.doesNotMatch(html, /Alice Example wins/);
});

test("an administrative report held for its own source review cannot appear as an admitted observation", () => {
  const view = partialTexasFixture();
  const report = view.quantities.find(q => q.id === "F1a-report");
  report.sourceAdmission = "held";
  report.sourceQualification = "Invented report awaiting source review.";
  const html = render(view);
  assert.match(quantity(html, "F1a-report"), /awaiting source review/);
  assert.doesNotMatch(quantity(html, "F1a-report"), /count-value|102<span>/);
  assert.match(quantity(html, "participants"), /100<span> people/);
});

test("the displayed parent status uses the same numerical tolerance as readiness", () => {
  const view = partialTexasFixture();
  view.audit.parents[0].representedMass += 1e-12;
  const html = render(view);
  assert.doesNotMatch(html, /some population states failed or are missing/);
  assert.match(quantity(html, "participants"), /100<span> people/);
});

test("unsupported rules and absent contests do not fabricate a winner or an undervote", () => {
  const unsupported = render(unsupportedRuleFixture()), absent = render(absentContestFixture());
  assert.match(unsupported, /Winner probabilities are unavailable/);
  assert.doesNotMatch(unsupported, /Alice Example wins/);
  assert.match(absent, /No contest applies to this event/);
  assert.match(absent, /different from leaving an applicable contest blank/);
  assert.doesNotMatch(absent, /Candidate votes|Alice Example wins/);
  assert.match(quantity(absent, "participants"), /100<span> people/);
});

test("unresolved race probability stays distinct from a chamber organizing tie", () => {
  const race = render(unresolvedRaceTieFixture()), chamber = render(chamberTieScenarioFixture());
  assert.match(race, /Tied highest candidate vote count · Unresolved<\/dt><dd>100%/);
  assert.match(race, /Invented Alice Example wins<\/dt><dd>0%/);
  assert.match(chamber, /Chamber organizing scenario/);
  assert.match(chamber, /Republican control<\/dt><dd>100%/);
  assert.match(chamber, /distinct from tied candidate vote totals/);
  assert.match(chamber, /Illustrative data only/);
});

test("unknown values remain unavailable and missing support cannot become zero", () => {
  const view = partialTexasFixture(), old = view.quantities.find(q => q.id === "participants");
  view.quantities = view.quantities.map(q => q.id === old.id ? { id:q.id,label:q.label,unit:q.unit,quantityType:q.quantityType,denominator:q.denominator,kind:"unavailable",reason:"Participant summary not supplied." } : q);
  const html = render(view);
  assert.match(quantity(html, "participants"), /Participant summary not supplied/);
  assert.doesNotMatch(quantity(html, "participants"), /count-value/);
});

test("the prototype and fixtures are absent from every production page and API import", () => {
  function files(path) { return readdirSync(path, { withFileTypes:true }).flatMap(entry => entry.isDirectory() ? files(join(path,entry.name)) : [join(path,entry.name)]); }
  for (const file of [...files(join(root,"app")), ...files(join(root,"components")), ...files(join(root,"lib"))].filter(file => /\.[cm]?[jt]sx?$/.test(file) && !file.endsWith("CountForecastPreview.tsx") && !file.endsWith("count-view.ts")))
    assert.doesNotMatch(readFileSync(file,"utf8"), /(?:from\s*|import\s*\(|require\s*\()["'][^"']*(?:CountForecastPreview|count-view|fixtures\/count)/, file);
});
