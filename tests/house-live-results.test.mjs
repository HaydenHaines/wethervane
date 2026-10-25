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
import { makeHouseLivePublication } from "./fixtures/house-live-publication.mjs";

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

test("browser workspace: verified requests, comparisons, failure recovery, navigation and mobile tables", {
  skip: !process.env.BALLOTMANCY_LIVE_HOUSE_BROWSER, timeout: 120000,
}, async () => {
  compile("components/HouseLiveWorkspace.tsx");
  const entry = join(directory, "workspace-entry.js");
  writeFileSync(entry, `import React from 'react';import {createRoot} from 'react-dom/client';import Workspace from './components/HouseLiveWorkspace.js';
createRoot(document.getElementById('root')).render(React.createElement('div',{className:'shell'},React.createElement('p',{},'INVENTED UI TEST DATA. No current election results.'),React.createElement(Workspace,{onHistorical:()=>{globalThis.historicalRequested=true;},onReturn:()=>{globalThis.senateRequested=true;}})));`);
  const require = createRequire(import.meta.url), { webpack } = require("next/dist/compiled/webpack/webpack");
  await new Promise((resolve, reject) => webpack({ mode: "development", target: "web", devtool: false, entry,
    output: { path: join(directory, "workspace-bundle"), filename: "preview.js" } }, (error, stats) =>
    error || stats.hasErrors() ? reject(error || Error(stats.toString({ all: false, errors: true }))) : resolve()));
  const script = readFileSync(join(directory, "workspace-bundle/preview.js"), "utf8").replaceAll("</script", "<\\/script");
  const html = `<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Invented House workspace</title><style>${readFileSync(join(root, "app/globals.css"), "utf8")}</style><body><div id="root"></div><script>${script}</script></body></html>`;
  const publications = new Map(nativeCatalog.entries.map(entry => {
    const value = makeHouseLivePublication(entry.configuration);
    return [value.view.run.id, value];
  }));
  const primary = publications.get(catalog.defaultId);
  const baseline = primary.catalog.entries.find(entry => entry.configuration.base === "fundamentals" && entry.configuration.polling === "none");
  const alternative = primary.catalog.entries.find(entry => entry.configuration.base === "recent_calendar_history" && entry.configuration.polling === "standard");
  const { chromium, expect } = await import("@playwright/test");
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, reducedMotion: "reduce" });
  const pageErrors = [], requests = [];
  let failedId = null, delayedId = null, releaseDelayed = null;
  page.on("pageerror", error => pageErrors.push(error.message));
  await page.route("http://house-ui.test/**", async route => {
    const url = new URL(route.request().url());
    if (url.pathname === "/") return route.fulfill({ contentType: "text/html", body: html });
    if (url.pathname === "/api/chambers/house/live") {
      const id = url.searchParams.get("run") ?? primary.catalog.defaultId;
      requests.push(id);
      if (id === failedId) return route.fulfill({ status: 503, json: { error: "Invented failed response" } });
      if (id === delayedId) await new Promise(resolve => { releaseDelayed = resolve; });
      return route.fulfill({ status: publications.has(id) ? 200 : 400, json: publications.get(id) ?? { error: "Unknown model" } }).catch(() => {});
    }
    return route.fulfill({ status: 404, body: "No historical fallback in this fixture" });
  });
  try {
    await page.goto("http://house-ui.test/?chamber=house&edition=current&race=2026%3AAL-04%3Ahouse%3Aregular");
    await expect(page.locator("#house-live-detail-title")).toHaveText("AL-04");
    const builder = page.locator("details.house-configuration");
    await expect(builder).not.toHaveAttribute("open", "");
    const compareSelect = page.getByLabel("Compare with", { exact: true });
    const comparePanel = page.getByRole("region", { name: "Current House model comparison", exact: true });
    await compareSelect.selectOption(baseline.id);
    await expect(comparePanel.getByRole("table")).toHaveCount(3);
    await expect(comparePanel).toContainText("not accuracy scores");
    await expect(comparePanel.getByRole("table").last()).toContainText("AL-04");
    assert.equal(new URL(page.url()).searchParams.get("compare"), baseline.id);

    failedId = alternative.id;
    await compareSelect.selectOption(alternative.id);
    await expect(comparePanel.getByRole("alert")).toContainText("Comparison unavailable");
    await expect(comparePanel).toContainText("last verified comparison");
    await expect(comparePanel.getByRole("table")).toHaveCount(3);
    failedId = null;
    await comparePanel.getByRole("button", { name: "Try comparison again" }).click();
    await expect(comparePanel.getByRole("alert")).toHaveCount(0);
    await expect(comparePanel.getByRole("heading", { level: 2 })).toContainText(alternative.label);
    if (process.env.BALLOTMANCY_LIVE_HOUSE_SCREENSHOT_DIR) {
      mkdirSync(process.env.BALLOTMANCY_LIVE_HOUSE_SCREENSHOT_DIR, { recursive: true });
      await comparePanel.scrollIntoViewIfNeeded();
      await page.screenshot({ path: join(process.env.BALLOTMANCY_LIVE_HOUSE_SCREENSHOT_DIR, "comparison-desktop.png") });
    }
    await page.setViewportSize({ width: 390, height: 844 });
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
    if (process.env.BALLOTMANCY_LIVE_HOUSE_SCREENSHOT_DIR)
      await page.screenshot({ path: join(process.env.BALLOTMANCY_LIVE_HOUSE_SCREENSHOT_DIR, "comparison-mobile.png") });
    await page.setViewportSize({ width: 1440, height: 900 });

    await builder.locator("summary").first().click();
    const base = page.getByLabel("Electoral-history model", { exact: true });
    failedId = alternative.id;
    await base.selectOption("recent_calendar_history");
    await expect(page.getByRole("alert")).toContainText("Unable to change House models");
    await expect(page.locator(".hero > div > p")).toContainText(primary.view.run.label);
    assert.equal(new URL(page.url()).searchParams.get("run"), primary.view.run.id);
    failedId = null;
    await page.getByRole("button", { name: "Try again", exact: true }).click();
    await expect(page.locator(".hero > div > p")).toContainText(alternative.label);
    await expect(compareSelect).toHaveValue("");
    await expect(page.locator("#house-live-detail-title")).toHaveText("AL-04");
    assert.equal(new URL(page.url()).searchParams.get("run"), alternative.id);
    await page.goBack();
    await expect(page.locator(".hero > div > p")).toContainText(primary.view.run.label);
    await expect(compareSelect).toHaveValue(alternative.id);

    // Returning to the previous history entry aborts a slow new primary selection.
    await expect(page.getByRole("status").filter({ hasText: "Loading comparison…" })).toHaveCount(0);
    delayedId = alternative.id;
    await base.selectOption("recent_calendar_history");
    await expect(page.getByRole("status").filter({ hasText: "Loading the selected House result" })).toBeVisible();
    await page.goBack();
    await expect(compareSelect).toHaveValue(baseline.id);
    delayedId = null;
    releaseDelayed?.();
    await expect(page.locator(".hero > div > p")).toContainText(primary.view.run.label);
    await expect(page.getByRole("alert")).toHaveCount(0);

    await page.goto("http://house-ui.test/?chamber=house&edition=current&compare=house-current-polling");
    await expect(page.getByRole("heading", { name: "Current House forecast unavailable" })).toBeVisible();
    await expect(page.getByRole("alert")).toContainText("different model");
    await page.getByRole("button", { name: "Load current default" }).click();
    await expect(page.locator(".hero > div > p")).toContainText(primary.view.run.label);
    await expect(compareSelect).toHaveValue("");
    assert.ok(requests.every(id => publications.has(id)));
    assert.equal(await page.evaluate(() => localStorage.length), 0);
    assert.deepEqual(pageErrors, []);
  } finally { releaseDelayed?.(); await browser.close(); }
});
