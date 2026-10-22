import test, { after } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync, rmSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import ts from "typescript";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { runoffFixture, inactiveRunoffFixture, rankedFixture, unknownLeaderFixture, missingParentFixture } from "./fixtures/election-outcome.mjs";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const directory = mkdtempSync(join(root, "tests", ".election-outcome-preview-"));
after(() => rmSync(directory, { recursive: true, force: true }));
writeFileSync(join(directory, "package.json"), '{"type":"module"}\n');
for (const relative of ["components/ElectionOutcomePreview.tsx", "lib/election-outcome-view.ts"]) {
  const output = ts.transpileModule(readFileSync(join(root, relative), "utf8"), { fileName: relative, compilerOptions: {
    target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext, jsx: ts.JsxEmit.ReactJSX, rewriteRelativeImportExtensions: true,
  } }).outputText;
  const destination = join(directory, relative.replace(/\.tsx?$/, ".js"));
  mkdirSync(dirname(destination), { recursive: true }); writeFileSync(destination, output);
}
const { default: Preview } = await import(pathToFileURL(join(directory, "components/ElectionOutcomePreview.js")).href);
const render = view => renderToStaticMarkup(React.createElement(Preview, { view }));

test("final winners, advancing and conditional runoff turnout have distinct labels", () => {
  const html = render(runoffFixture());
  assert.match(html, /Illustrative data only/);
  assert.match(html, /Final election outcome/);
  assert.match(html, /Alex Example wins overall<\/dt><dd>50%/);
  assert.match(html, /Alex Example wins the general election<\/dt><dd>30%/);
  assert.match(html, /Alex Example wins the runoff<\/dt><dd>50%/);
  assert.match(html, /Alex Example advances<\/dt><dd>40%/);
  assert.match(html, /need not add to 100%/);
  assert.match(html, /Chance a runoff occurs: 40%/);
  assert.match(html, /Runoff participation, if this runoff occurs/);
  assert.match(html, /750.5/);
  assert.match(html, /Tie over who advances to the runoff · Unresolved/);
});

test("an inactive runoff is not presented as zero turnout or another winner", () => {
  const html = render(inactiveRunoffFixture());
  const runoff = html.split('aria-label="Runoff on ')[1].split("</article>")[0];
  assert.match(runoff, /Not held in this example/);
  assert.doesNotMatch(runoff, /count-value|wins<|0 people/);
});

test("unidentified-person leadership is an unresolved outcome, not an Other candidate", () => {
  const html = render(unknownLeaderFixture());
  assert.match(html, /Unidentified person leads · Identity or eligibility unresolved/);
  assert.match(html, /No valid candidate votes · Unresolved/);
  assert.doesNotMatch(html, /Other wins|Unidentified person wins/);
});

test("ranked examples cannot be mistaken for an averaged prediction", () => {
  const html = render(rankedFixture());
  assert.match(html, /One illustrative simulation, not the predicted sequence/);
  assert.match(html, /Round 1/);
  assert.match(html, /Tie over who is eliminated · Unresolved/);
  assert.match(html, /does not transfer ballots or break ties/);
  assert.match(html, /Voters rank candidates/);
});

test("failed-parent or unsupported-rule examples suppress all predictive values", () => {
  const held = runoffFixture(); held.evidence.sourceAndRuleAdmitted = false;
  for (const view of [missingParentFixture(), held]) {
    const html = render(view);
    assert.match(html, /Outcome estimates unavailable/);
    assert.doesNotMatch(html, /count-value|Alex Example wins|750.5|40%/);
  }
});

test("outcome preparation stays out of public pages and existing count previews", () => {
  function files(path) { return readdirSync(path, { withFileTypes: true }).flatMap(entry => entry.isDirectory() ? files(join(path, entry.name)) : [join(path, entry.name)]); }
  for (const file of [...files(join(root, "app")), ...files(join(root, "components")), ...files(join(root, "lib"))].filter(file => /\.[cm]?[jt]sx?$/.test(file) && !file.endsWith("ElectionOutcomePreview.tsx") && !file.endsWith("election-outcome-view.ts")))
    assert.doesNotMatch(readFileSync(file, "utf8"), /(?:from\s*|import\s*\(|require\s*\()["'][^"']*(?:ElectionOutcomePreview|election-outcome-view|fixtures\/election-outcome)/, file);
});

test("export standalone examples for optional visual review", { skip: !process.env.BALLOTMANCY_OUTCOME_PREVIEW_OUTPUT }, () => {
  const css = readFileSync(join(root, "app/globals.css"), "utf8");
  const html = [runoffFixture, inactiveRunoffFixture, rankedFixture, unknownLeaderFixture, missingParentFixture].map(f => render(f())).join('<hr style="margin:60px 0">');
  writeFileSync(process.env.BALLOTMANCY_OUTCOME_PREVIEW_OUTPUT, `<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Invented election outcome examples</title><style>${css}</style><body><main style="padding:20px">${html}</main></body></html>`);
});
