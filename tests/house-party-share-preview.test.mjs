import test, { after } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync, rmSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import ts from "typescript";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { partyShareFixture, samePartyFixture, structuralZeroFixture, zeroDenominatorFixture, invalidIntervalFixture } from "./fixtures/house-party-share.mjs";
const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const directory = mkdtempSync(join(root, "tests", ".house-party-preview-"));
after(() => rmSync(directory, { recursive: true, force: true }));
writeFileSync(join(directory, "package.json"), '{"type":"module"}\n');
for (const relative of ["components/HousePartySharePreview.tsx", "lib/house-party-share-view.ts"]) {
  const output = ts.transpileModule(readFileSync(join(root, relative), "utf8"), { fileName: relative, compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext, jsx: ts.JsxEmit.ReactJSX, rewriteRelativeImportExtensions: true } }).outputText;
  const destination = join(directory, relative.replace(/\.tsx?$/, ".js")); mkdirSync(dirname(destination), { recursive: true }); writeFileSync(destination, output);
}
const { default: Preview } = await import(pathToFileURL(join(directory, "components/HousePartySharePreview.js")).href);
const render = fixture => renderToStaticMarkup(React.createElement(Preview, { fixture }));
const quantity = (html, id) => html.split(`data-quantity="${id}"`)[1]?.split("</section>")[0] || "";
test("SSR distinguishes party mean and observed ratio from unavailable candidate winners", () => {
  const html = render(partyShareFixture());
  for (const text of [/Illustrative data only/, /CA-01/, /First ballot/, /share fraction/, /invented fixture/, /D \/ \(D \+ R\)/, /Third-party votes are excluded/]) assert.match(html, text);
  assert.match(quantity(html, "estimate"), /Model estimate[\s\S]*65.0%/);
  assert.match(quantity(html, "estimate"), /Middle 80% of modeled shares/);
  assert.match(quantity(html, "estimate"), /Middle 90% of modeled shares/);
  assert.match(quantity(html, "observed"), /65 Democratic votes out of 100 Democratic and Republican votes/);
  assert.match(quantity(html, "winner"), /Not available/);
  assert.match(html, /Ranked-choice and runoff final winners are separate outputs/);
  if (process.env.BALLOTMANCY_PARTY_SHARE_PREVIEW_OUTPUT) {
    const css = readFileSync(join(root, "app/globals.css"), "utf8");
    const scenarios = [partyShareFixture(), samePartyFixture(), structuralZeroFixture(), zeroDenominatorFixture(), invalidIntervalFixture()].map(render).join("\n");
    writeFileSync(process.env.BALLOTMANCY_PARTY_SHARE_PREVIEW_OUTPUT, `<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Invented House party share preview</title><style>${css}</style><body><main style="padding:20px">${scenarios}</main></body></html>`);
  }
  assert.doesNotMatch(html, /Chance of winning|Safe|Likely|Tossup|65.0% chance|favorite/i);
});
test("SSR structural endpoints carry absence reasons and never claim simulated wins", () => {
  for (const [fixture, endpoint, absent] of [[samePartyFixture(), "100.0%", "Republican"], [structuralZeroFixture(), "0.0%", "Democratic"]]) {
    const html = render(fixture), estimate = quantity(html, "estimate");
    assert.ok(estimate.includes(`Fixed by party absence`)); assert.ok(estimate.includes(endpoint)); assert.ok(estimate.includes(`No ${absent} nominee is present`));
    assert.match(estimate, /not a simulation result/);
    assert.doesNotMatch(estimate, /Model estimate|Middle [0-9]+%|Won every time|No wins/);
  }
});
test("SSR unavailable denominator and incomplete interval never display fabricated estimates", () => {
  const zero = render(zeroDenominatorFixture());
  assert.match(quantity(zero, "observed"), /undefined/); assert.doesNotMatch(quantity(zero, "observed"), /0.0%|NaN|Infinity/);
  const invalid = render(invalidIntervalFixture());
  assert.match(quantity(invalid, "estimate"), /Not available/); assert.doesNotMatch(quantity(invalid, "estimate"), /65.0%|undefined|NaN/);
  assert.match(quantity(invalid, "observed"), /65.0%/);
});
test("private preview has no production callers or fixture imports", () => {
  function scan(directory) { return readdirSync(directory, { withFileTypes: true }).flatMap(entry => entry.isDirectory() ? scan(join(directory, entry.name)) : /\.[cm]?[jt]sx?$/.test(entry.name) ? [join(directory, entry.name)] : []); }
  for (const base of ["app", "components", "lib", "scripts"]) for (const file of scan(join(root, base))) {
    if (file.endsWith("/HousePartySharePreview.tsx") || file.endsWith("/house-party-share-view.ts")) continue;
    assert.doesNotMatch(readFileSync(file, "utf8"), /(?:from\s*|import\s*\(|require\s*\()["'][^"']*(?:house-party-share|HousePartySharePreview)/, file);
  }
});
