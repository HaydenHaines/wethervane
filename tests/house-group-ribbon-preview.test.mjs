import test, { after } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync, rmSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";
import ts from "typescript";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { groupRibbonFixtures } from "./fixtures/house-group-ribbon.mjs";
const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const directory = mkdtempSync(join(root, "tests", ".house-group-ribbon-"));
after(() => rmSync(directory, { recursive: true, force: true }));
writeFileSync(join(directory, "package.json"), '{"type":"module"}\n');
for (const relative of ["components/HouseGroupRibbonPreview.tsx", "lib/house-group-ribbon.ts", "lib/house-winner-view.ts", "lib/serpentine-layout.ts", "lib/presentation.ts"]) {
  const output = ts.transpileModule(readFileSync(join(root, relative), "utf8"), { fileName: relative, compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext, jsx: ts.JsxEmit.ReactJSX, rewriteRelativeImportExtensions: true } }).outputText;
  const destination = join(directory, relative.replace(/\.tsx?$/, ".js"));
  mkdirSync(dirname(destination), { recursive: true }); writeFileSync(destination, output);
}
const { default: Preview } = await import(pathToFileURL(join(directory, "components/HouseGroupRibbonPreview.js")).href);
const render = fixtures => renderToStaticMarkup(React.createElement(Preview, { fixtures }));
test("ribbon accounts for every district without making missing probabilities zero", () => {
  for (const size of [0, 1, 16, 435]) {
    const html = render(groupRibbonFixtures(size));
    assert.equal((html.match(/data-district=/g) || []).length, size);
    assert.match(html, /Illustrative data only/);
    assert.doesNotMatch(html, /majority-position|218th|controlProbability/);
    if (size === 0) assert.match(html, /No districts supplied/);
  }
  const html = render(groupRibbonFixtures());
  const missing = html.split('data-segment="unavailable"')[1].split("</section>")[0];
  assert.match(missing, /forecast unavailable/); assert.doesNotMatch(missing, /0.0%|balance/);
});
test("Other leaders and ties have distinct sections and preserve all three chances", () => {
  const html = render(groupRibbonFixtures());
  assert.match(html, /aria-label="Other group leads"/);
  assert.match(html, /aria-label="Top ties involving Other"/);
  assert.match(html, /Democratic 10.0%, Republican 10.0%, Other 80.0%/);
  assert.match(html, /top chances tied: Democratic group and Republican group and Other group/);
  assert.match(html, /not a vote margin and does not estimate House control or a tipping point/);
});
test("private ribbon has no production callers", () => {
  function scan(directory) { return readdirSync(directory, { withFileTypes: true }).flatMap(entry => entry.isDirectory() ? scan(join(directory, entry.name)) : /\.[cm]?[jt]sx?$/.test(entry.name) ? [join(directory, entry.name)] : []); }
  for (const base of ["app", "components", "lib", "scripts"]) for (const file of scan(join(root, base))) {
    if (file.endsWith("/HouseGroupRibbonPreview.tsx") || file.endsWith("/house-group-ribbon.ts")) continue;
    assert.doesNotMatch(readFileSync(file, "utf8"), /(?:from\s*|import\s*\(|require\s*\()["'][^"']*(?:house-group-ribbon|HouseGroupRibbonPreview)/, file);
  }
});

// Optional standalone browser artifact for responsive, selection and keyboard checks.
// The private fixture never needs a production route or a running Next test server.
test("generate the optional standalone browser preview", { skip: !process.env.BALLOTMANCY_GROUP_RIBBON_PREVIEW_OUTPUT }, async () => {
  mkdirSync(join(directory, "fixtures"));
  for (const filename of ["house-winner.mjs", "house-group-ribbon.mjs"]) writeFileSync(join(directory, "fixtures", filename), readFileSync(join(root, "tests/fixtures", filename)));
  const entry = join(directory, "entry.js");
  writeFileSync(entry, `import React from 'react';import {createRoot} from 'react-dom/client';import Preview from './components/HouseGroupRibbonPreview.js';import {groupRibbonFixtures} from './fixtures/house-group-ribbon.mjs';const root=createRoot(document.getElementById('root'));globalThis.showFixtureCount=count=>root.render(React.createElement(Preview,{fixtures:groupRibbonFixtures(count)}));globalThis.showFixtureCount(435);`);
  const require = createRequire(import.meta.url);
  const { webpack } = require("next/dist/compiled/webpack/webpack");
  await new Promise((resolve, reject) => webpack({ mode: "development", target: "web", devtool: false, entry, output: { path: join(directory, "bundle"), filename: "preview.js" } }, (error, stats) => error || stats.hasErrors() ? reject(error || Error(stats.toString({ all: false, errors: true }))) : resolve()));
  const css = readFileSync(join(root, "app/globals.css"), "utf8");
  const script = readFileSync(join(directory, "bundle/preview.js"), "utf8").replaceAll("</script", "<\\/script");
  writeFileSync(process.env.BALLOTMANCY_GROUP_RIBBON_PREVIEW_OUTPUT, `<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Invented House ribbon preview</title><style>${css}</style><body><main id="root" style="padding:20px"></main><script>${script}</script></body></html>`);
});
