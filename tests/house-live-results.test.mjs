import test, { after } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync, rmSync, existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";
import ts from "typescript";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { makeHouseLiveResult, makeHouseLiveCatalog } from "./fixtures/house-live-fixtures.mjs";
import { parseHouseLiveResult } from "../lib/house-live-result.ts";
import { parseHouseLiveCatalog } from "../lib/house-live-catalog.ts";
import { houseLiveView, houseLiveCatalogView } from "../lib/house-live-view.ts";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const directory = mkdtempSync(join(root, "tests", ".house-live-render-"));
after(() => rmSync(directory, { recursive: true, force: true }));
writeFileSync(join(directory, "package.json"), '{"type":"module"}\n');

// Compile the real component dependency tree; never create a fixture route or publication.
const compiled = new Set();
function compile(relative) {
  if (compiled.has(relative)) return;
  compiled.add(relative);
  const source = join(root, relative);
  let output = ts.transpileModule(readFileSync(source, "utf8"), { fileName: source,
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext, jsx: ts.JsxEmit.ReactJSX } }).outputText;
  output = output.replace(/(from\s*|import\s*)(["'])(\.[^"']+)\2/g, (match, prefix, quote, specifier) => {
    const target = resolve(dirname(source), specifier);
    const filename = [target, `${target}.ts`, `${target}.tsx`].find(path => existsSync(path));
    assert.ok(filename, `Cannot resolve ${specifier} from ${source}`);
    compile(filename.slice(root.length + 1));
    return `${prefix}${quote}${specifier.replace(/\.tsx?$/, "")}.js${quote}`;
  });
  const destination = join(directory, relative.replace(/\.tsx?$/, ".js"));
  mkdirSync(dirname(destination), { recursive: true });
  writeFileSync(destination, output);
}
compile("components/HouseLiveResults.tsx");
const { default: Results } = await import(pathToFileURL(join(directory, "components/HouseLiveResults.js")).href);
const nativeCatalog = parseHouseLiveCatalog(makeHouseLiveCatalog());
assert.ok(nativeCatalog);
const catalog = houseLiveCatalogView(nativeCatalog);
const views = nativeCatalog.entries.map(entry => {
  const raw = makeHouseLiveResult(entry.configuration);
  // This flag exercises the display projection only. All bytes and geometry are
  // explicitly invented, and no importer/server admission is invoked here.
  raw.qualification.publication_status = "reviewed";
  const result = parseHouseLiveResult(raw);
  assert.ok(result);
  return houseLiveView(result, entry);
});
const view = views.find(item => item.run.id === catalog.defaultId);
const noop = () => {};
const props = { view, catalog, geography: null, builderExpanded: false,
  onBuilderExpandedChange: noop, onSelect: noop, onRetry: noop, onHistorical: noop };
const render = (overrides = {}) => renderToStaticMarkup(React.createElement(Results, { ...props, ...overrides }));

test("all sixteen saved configurations use the same complete result-driven renderer", () => {
  for (const current of views) {
    const html = render({ view: current });
    assert.equal((html.match(/data-district=/g) || []).length, 435);
    assert.match(html, /10 of 435/);
    assert.match(html, /Neither party reaches 218/);
    assert.match(html, /Other seats remain unassigned/);
    assert.doesNotMatch(html, /<details[^>]*house-configuration[^>]* open/);
    assert.doesNotMatch(html, /Actual winner:|How this model did|Load saved result/);
    assert.ok(html.indexOf('aria-label="House map"') < html.indexOf('id="house-live-control-title"'));
  }
});

test("Other leaders stay out of the D/R path; a missing chamber never creates control numbers", () => {
  const html = render();
  const other = html.split('data-segment="other"')[1].split("</section>")[0];
  assert.match(other, /data-district="AL-02"/);
  const main = html.split('data-segment="main"')[1].split("</section>")[0];
  assert.doesNotMatch(main, /data-district="AL-02"/);
  const missing = render({ view: { ...view, chamber: null, chamberPrecision: null } });
  assert.match(missing, /House control distribution is unavailable/);
  assert.doesNotMatch(missing, /id="house-live-control-title"/);
});

function inventedGeography() {
  return { metadata: { cycle: 2026, mapVintage: view.mapVintage, admission: "verified",
    sourceSha256: view.geographySha256, sourceName: "Invented grid for UI tests; not district boundaries" },
  geojson: { type: "FeatureCollection", features: view.races.map((race, index) => {
    const x = -120 + (index % 25), y = 26 + Math.floor(index / 25);
    return { type: "Feature", properties: { districtKey: race.district.districtKey },
      geometry: { type: "Polygon", coordinates: [[[x, y], [x + .5, y], [x + .5, y + .5], [x, y + .5], [x, y]]] } };
  }) } };
}

let browserPreview;
async function buildBrowserPreview() {
  if (browserPreview) return browserPreview;
  const entry = join(directory, "entry.js");
  writeFileSync(entry, `import React,{useState} from 'react';import {createRoot} from 'react-dom/client';import Results from './components/HouseLiveResults.js';
const views=${JSON.stringify(views)};const catalog=${JSON.stringify(catalog)};const geography=${JSON.stringify(inventedGeography())};
globalThis.selectionCalls=[];globalThis.failSelection=false;
function Harness(){const [view,setView]=useState(views.find(v=>v.run.id===catalog.defaultId));const [expanded,setExpanded]=useState(false);const [busy,setBusy]=useState(false);const [error,setError]=useState('');
function select(id){globalThis.selectionCalls.push(id);setBusy(true);setError('');setTimeout(()=>{if(globalThis.failSelection){setError('Invented failed load.');}else{setView(views.find(v=>v.run.id===id));}setBusy(false);},50);}
return React.createElement('div',{className:'shell','data-run-id':view.run.id},React.createElement('p',{},'INVENTED UI TEST DATA. Shapes are a grid, not legal district boundaries.'),React.createElement(Results,{view,catalog,geography,builderExpanded:expanded,onBuilderExpandedChange:setExpanded,busy,error,onSelect:select,onRetry:()=>select(globalThis.selectionCalls.at(-1)),onHistorical:()=>{globalThis.historyClicked=true;}}));}
createRoot(document.getElementById('root')).render(React.createElement(Harness));`);
  const require = createRequire(import.meta.url);
  const { webpack } = require("next/dist/compiled/webpack/webpack");
  await new Promise((resolve, reject) => webpack({ mode: "development", target: "web", devtool: false, entry,
    output: { path: join(directory, "bundle"), filename: "preview.js" } }, (error, stats) =>
    error || stats.hasErrors() ? reject(error || Error(stats.toString({ all: false, errors: true }))) : resolve()));
  const css = readFileSync(join(root, "app/globals.css"), "utf8");
  const script = readFileSync(join(directory, "bundle/preview.js"), "utf8").replaceAll("</script", "<\\/script");
  browserPreview = `<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Invented live House UI test</title><style>${css}</style><body><div id="root"></div><script>${script}</script></body></html>`;
  if (process.env.BALLOTMANCY_LIVE_HOUSE_PREVIEW_OUTPUT) writeFileSync(process.env.BALLOTMANCY_LIVE_HOUSE_PREVIEW_OUTPUT, browserPreview);
  return browserPreview;
}

test("browser: instant selection, retained failures, drafts, ribbon, sidebar, and mobile", {
  skip: !process.env.BALLOTMANCY_LIVE_HOUSE_BROWSER, timeout: 120000,
}, async () => {
  const { chromium, expect } = await import("@playwright/test");
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, reducedMotion: "reduce" });
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  try {
    await page.setContent(await buildBrowserPreview());
    const builder = page.locator("details.house-configuration");
    await expect(builder).not.toHaveAttribute("open", "");
    await expect(page.locator(".house-ribbon-cell")).toHaveCount(435);
    await expect(page.locator("#house-live-race-list > button")).toHaveCount(10);
    await builder.locator("summary").first().click();
    const base = page.getByLabel("Electoral-history model", { exact: true });
    const uncertainty = page.getByLabel("Uncertainty scenario", { exact: true });
    await base.selectOption("recent_calendar_history");
    const alternative = catalog.entries.find(e => e.configuration.base === "recent_calendar_history" && e.configuration.polling === "standard");
    await expect(page.locator("[data-run-id]")).toHaveAttribute("data-run-id", alternative.id);
    assert.equal(await page.evaluate(() => globalThis.selectionCalls.length), 1);
    await uncertainty.selectOption("cycle-low");
    await expect(builder.getByRole("status")).toContainText("needs a new");
    await expect(base).toHaveValue("recent_calendar_history");
    await expect(page.locator("[data-run-id]")).toHaveAttribute("data-run-id", alternative.id);
    assert.equal(await page.evaluate(() => globalThis.selectionCalls.length), 1);
    await builder.getByRole("button", { name: "Reset to published default" }).click();
    await expect(page.locator("[data-run-id]")).toHaveAttribute("data-run-id", catalog.defaultId);
    await expect(uncertainty).toHaveValue("standard");

    // A failed selection preserves displayed data and can be retried without another change.
    await page.evaluate(() => { globalThis.failSelection = true; });
    await base.selectOption("recent_calendar_history");
    await expect(page.getByRole("alert")).toContainText("Unable to change House models");
    await expect(page.locator("[data-run-id]")).toHaveAttribute("data-run-id", catalog.defaultId);
    await page.evaluate(() => { globalThis.failSelection = false; });
    await page.getByRole("button", { name: "Try again", exact: true }).click();
    await expect(page.locator("[data-run-id]")).toHaveAttribute("data-run-id", alternative.id);
    await builder.getByRole("button", { name: "Reset to published default" }).click();
    await expect(page.locator("[data-run-id]")).toHaveAttribute("data-run-id", catalog.defaultId);

    // Reset also clears a failed draft while the default result is still displayed.
    await page.evaluate(() => { globalThis.failSelection = true; });
    await base.selectOption("recent_calendar_history");
    await expect(page.getByRole("alert")).toBeVisible();
    await page.evaluate(() => { globalThis.failSelection = false; });
    await builder.getByRole("button", { name: "Reset to published default" }).click();
    await expect(base).toHaveValue("fundamentals");
    await expect(page.locator("[data-run-id]")).toHaveAttribute("data-run-id", catalog.defaultId);
    await expect(page.getByRole("alert")).toHaveCount(0);
    await builder.locator("summary").first().click();

    await page.locator('[aria-controls="house-live-race-list"]').click();
    await expect(page.locator("#house-live-race-list > button")).toHaveCount(435);
    await page.locator('[aria-controls="house-live-race-list"]').click();
    await expect(page.locator("#house-live-race-list > button")).toHaveCount(10);
    const other = page.locator('[data-segment="other"] [data-district="AL-02"]');
    await other.click();
    await expect(page.locator("#house-live-detail-title")).toHaveText("AL-02");
    await expect(page.locator("#house-live-race-detail")).toContainText("Independent");
    const first = page.locator('[data-segment="main"] button').first();
    await first.focus();
    await page.keyboard.press("ArrowRight");
    await page.keyboard.press("Enter");
    const focused = await page.locator(":focus").getAttribute("data-district");
    await expect(page.locator("#house-live-detail-title")).toHaveText(focused);
    await page.locator('.house-ribbon-cell[data-district="AL-01"]').click();
    await expect(page.locator("#house-live-detail-title")).toHaveText("AL-01");
    await page.setViewportSize({ width: 1440, height: 600 });
    const detail = page.locator("#house-live-race-detail");
    await detail.scrollIntoViewIfNeeded();
    const scrollState = await detail.evaluate(node => {
      const pageTop = window.scrollY;
      node.scrollTop = 100;
      return { overflow: getComputedStyle(node).overflowY, scrollTop: node.scrollTop,
        pageTop, pageAfter: window.scrollY, height: node.clientHeight, content: node.scrollHeight };
    });
    assert.equal(scrollState.overflow, "auto");
    assert.ok(scrollState.content > scrollState.height && scrollState.scrollTop > 0);
    assert.equal(scrollState.pageAfter, scrollState.pageTop);
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.locator('section[aria-label="House map"]').scrollIntoViewIfNeeded();
    if (process.env.BALLOTMANCY_LIVE_HOUSE_SCREENSHOT_DIR) {
      mkdirSync(process.env.BALLOTMANCY_LIVE_HOUSE_SCREENSHOT_DIR, { recursive: true });
      await page.screenshot({ path: join(process.env.BALLOTMANCY_LIVE_HOUSE_SCREENSHOT_DIR, "desktop.png") });
    }
    await page.setViewportSize({ width: 390, height: 844 });
    await page.locator('.house-ribbon-cell[data-district="AL-02"]').click();
    await expect(page.locator("#house-live-detail-title")).toBeFocused();
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
    if (process.env.BALLOTMANCY_LIVE_HOUSE_SCREENSHOT_DIR)
      await page.screenshot({ path: join(process.env.BALLOTMANCY_LIVE_HOUSE_SCREENSHOT_DIR, "mobile.png") });
    assert.deepEqual(errors, []);
  } finally { await browser.close(); }
});
