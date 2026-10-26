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
import { withInventedPolling } from "./fixtures/house-live-polling-publication.mjs";
import { makeHousePublication, fixtureSha256 } from "./fixtures/house.mjs";
import { parseHousePublicationWithCatalog } from "../lib/house-catalog.ts";

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
  if (relative.endsWith(".json")) {
    const destination = join(directory, relative);
    mkdirSync(dirname(destination), { recursive: true });
    writeFileSync(destination, readFileSync(source));
    return;
  }
  let output = ts.transpileModule(readFileSync(source, "utf8"), { fileName: source,
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext, jsx: ts.JsxEmit.ReactJSX } }).outputText;
  output = output.replace(/(from\s*|import\s*)(["'])(\.[^"']+)\2/g, (match, prefix, quote, specifier) => {
    const target = resolve(dirname(source), specifier);
    const filename = [target, `${target}.ts`, `${target}.tsx`].find(path => existsSync(path));
    assert.ok(filename, `Cannot resolve ${specifier} from ${source}`);
    compile(filename.slice(root.length + 1));
    return `${prefix}${quote}${filename.endsWith(".json") ? specifier : `${specifier.replace(/\.tsx?$/, "")}.js`}${quote}`;
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

test("source capture notes appear only for nonempty limitations and keep each source link", () => {
  assert.doesNotMatch(render(), /Source capture notes/);
  const noted = structuredClone(view);
  noted.sources[0].capture_kind = "rendered_primary_source";
  noted.sources[0].limitations = ["Invented earlier capture; current refresh failed."];
  noted.sources.push({ ...noted.sources[0], id: "invented-second-source", label: "Invented rendered source",
    url: "https://example.invalid/rendered", limitations: ["Invented rendered text, not original PDF bytes."] });
  const html = render({ view: noted });
  assert.match(html, /Source capture notes \(2\)/);
  assert.match(html, /saved file fingerprint identifies that text/);
  assert.match(html, /Invented earlier capture; current refresh failed/);
  assert.match(html, /Invented rendered text, not original PDF bytes/);
  assert.match(html, /href="https:\/\/example.invalid\/rendered"/);
});

test("NY-21 notice is visible for the pinned release, detailed only in NY-21 focus, and absent for replacement bytes", () => {
  const pinned = structuredClone(view);
  pinned.run.id = "house-current-polling";
  pinned.run.resultSha256 = "c84248d56ff0fd0e291538729680cf8cdaa51bad9fb456c3d359fa022b76451d";
  const ny21 = render({ view: pinned, selectedRaceId: "2026:NY-21:house:regular" });
  assert.match(ny21, /NY-21 is under review\. The October 25 forecast omits declared write-in candidate Christopher Schmidt and excludes two polls that name him\. These published estimates have not yet been updated for that finding\./);
  assert.match(ny21, /View NY-21/);
  assert.ok(ny21.indexOf('aria-label="NY-21 forecast input notice"') < ny21.indexOf('aria-label="House group-win probability ribbon"'));
  assert.match(ny21, /NY-21 forecast inputs under review/);
  assert.match(ny21, /aria-label="NY-21 forecast input detail"/);
  assert.match(ny21, /aria-label="NY-21 input review"/);
  assert.match(ny21, /The displayed probabilities remain the October 25 estimates/);
  assert.match(ny21, /New York candidate list|NCPR: write-in campaign|Siena poll/);
  assert.doesNotMatch(ny21, /source-coverage erratum|original dated artifact|survey-menu evidence/);
  assert.match(ny21, /northcountrypublicradio\.org\/news\/story\/54097/);
  assert.match(ny21, /sri\.siena\.edu\/2026\/10\/09/);
  assert.match(ny21, /elections\.ny\.gov\/certification-november-3-2026/);
  const other = render({ view: pinned, selectedRaceId: "2026:AL-01:house:regular" });
  assert.match(other, /aria-label="NY-21 forecast input notice"/);
  assert.doesNotMatch(other, /aria-label="NY-21 forecast input detail"/);
  const replacement = structuredClone(pinned);
  replacement.run.resultSha256 = "0".repeat(64);
  assert.doesNotMatch(render({ view: replacement, selectedRaceId: "2026:NY-21:house:regular" }), /NY-21 is under review|aria-label="NY-21 input review"/);
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

test("mixed Other groups retain member affiliations without adopting one member's party label", () => {
  const grouped = structuredClone(view);
  const race = grouped.races.find(item => item.district.districtKey === "AL-02");
  const outcome = race.outcomes.find(item => item.party === "O");
  const independent = race.candidates.find(item => item.id === outcome.members[0].id);
  const libertarian = { ...independent, id: "invented-libertarian", name: "Invented Libertarian", partyLabel: "Libertarian" };
  race.candidates.push(libertarian);
  outcome.kind = "group";
  outcome.name = "Other candidates";
  outcome.members.push({ ...outcome.members[0], id: libertarian.id, name: libertarian.name });

  const html = render({ view: grouped, selectedRaceId: race.id });
  assert.match(html, /class="party-label">Other \(O\) · Group outcome<\/span>/);
  assert.doesNotMatch(html, /class="party-label">Independent · Other \(O\) · Group outcome/);
  assert.match(html, /Invented AL-02-I1 · Independent/);
  assert.match(html, /Invented Libertarian · Libertarian/);
  assert.match(html, /Combined probability for Invented AL-02-I1; Invented Libertarian/);
  assert.match(html, /class="candidate-prob"[^>]*>50\.0%<\/strong>/);

  const singleton = render({ selectedRaceId: race.id });
  assert.match(singleton, /class="party-label">Independent · Other \(O\)<\/span>/);
});

test("polling display separates included questions, zero local inputs, disabled polling, and unpublished details", () => {
  const included = withInventedPolling(makeHouseLivePublication());
  const html = render({ view: included.view, polling: included.polling, selectedRaceId: "2026:AL-01:house:regular" });
  assert.match(html, /3 selected poll-question results/);
  assert.match(html, /2 of 435 districts/);
  assert.match(html, /Sample size not reported/);
  assert.match(html, /See all 12 records/);
  assert.match(html, /not a count of independent surveys/);
  assert.doesNotMatch(html, /167|1,461|1,628|admitted observations|covariance|logit/);
  const zero = render({ view: included.view, polling: included.polling, selectedRaceId: "2026:AL-03:house:regular" });
  assert.match(zero, /No eligible local poll question was selected/);
  assert.match(zero, /does not mean no poll was conducted/);
  const off = withInventedPolling(makeHouseLivePublication({ base: "fundamentals", polling: "none", uncertainty: "standard" }));
  const disabled = render({ view: off.view, polling: off.polling });
  assert.match(disabled, /Polling is switched off/);
  assert.doesNotMatch(disabled, /aria-label="Included poll questions/);
  assert.match(render(), /selection details have not been published/);
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
  await page.addInitScript(() => {
    Object.defineProperty(navigator, "clipboard", { configurable: true, value: {
      writeText: async value => {
        if (globalThis.denyClipboard) throw Error("Invented clipboard denial");
        globalThis.copiedViewUrl = value;
      },
    } });
  });
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
    await builder.locator("summary").first().click();
    await page.getByRole("button", { name: "Share view" }).click();
    const retainedComparisonUrl = await page.evaluate(() => globalThis.copiedViewUrl);
    assert.equal(new URL(retainedComparisonUrl).searchParams.get("compare"), baseline.id);
    await builder.locator("summary").first().click();
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
    // A district chosen after failure must survive retrying that model request.
    await page.locator('.house-ribbon-cell[data-district="AL-03"]').click();
    await expect(page.locator("#house-live-detail-title")).toHaveText("AL-03");
    failedId = null;
    await page.getByRole("button", { name: "Try again", exact: true }).click();
    await expect(page.locator(".hero > div > p")).toContainText(alternative.label);
    await expect(compareSelect).toHaveValue("");
    await expect(page.locator("#house-live-detail-title")).toHaveText("AL-03");
    assert.equal(new URL(page.url()).searchParams.get("race"), "2026:AL-03:house:regular");
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
    await expect(page.locator("#house-live-detail-title")).toHaveText("AL-04");
    assert.equal(new URL(page.url()).searchParams.get("race"), "2026:AL-04:house:regular");

    // District navigation while a model is loading must not refetch or revert.
    delayedId = alternative.id;
    releaseDelayed = null;
    await base.selectOption("recent_calendar_history");
    await expect(page.getByRole("status").filter({ hasText: "Loading the selected House result" })).toBeVisible();
    await expect.poll(() => typeof releaseDelayed).toBe("function");
    const pendingRequests = requests.filter(id => id === alternative.id).length;
    await page.locator('.house-ribbon-cell[data-district="AL-02"]').click();
    await expect(page.locator("#house-live-detail-title")).toHaveText("AL-02");
    assert.equal(new URL(page.url()).searchParams.get("race"), "2026:AL-02:house:regular");
    assert.equal(requests.filter(id => id === alternative.id).length, pendingRequests);
    delayedId = null;
    releaseDelayed();
    await expect(page.locator(".hero > div > p")).toContainText(alternative.label);
    await expect(page.locator("#house-live-detail-title")).toHaveText("AL-02");
    assert.equal(new URL(page.url()).searchParams.get("race"), "2026:AL-02:house:regular");
    assert.equal(requests.filter(id => id === alternative.id).length, pendingRequests);

    await page.goto("http://house-ui.test/?chamber=house&edition=current&compare=house-current-polling");
    await expect(page.getByRole("heading", { name: "Current House forecast unavailable" })).toBeVisible();
    await expect(page.getByRole("alert")).toContainText("different model");
    await page.getByRole("button", { name: "Load current default" }).click();
    await expect(page.locator(".hero > div > p")).toContainText(primary.view.run.label);
    await expect(compareSelect).toHaveValue("");
    // Share the displayed non-default result and restore its complete selection.
    await builder.locator("summary").first().click();
    await base.selectOption("recent_calendar_history");
    await expect(page.locator(".hero > div > p")).toContainText(alternative.label);
    await compareSelect.selectOption(baseline.id);
    await expect(comparePanel.getByRole("table")).toHaveCount(3);
    await page.locator('.house-ribbon-cell[data-district="AL-03"]').click();
    await page.getByRole("button", { name: "Share view" }).click();
    await expect(page.getByRole("status").filter({ hasText: "Link copied." })).toBeVisible();
    const sharedUrl = await page.evaluate(() => globalThis.copiedViewUrl);
    assert.equal(sharedUrl, page.url());
    const shared = new URL(sharedUrl);
    assert.equal(shared.searchParams.get("chamber"), "house");
    assert.equal(shared.searchParams.get("edition"), "current");
    assert.equal(shared.searchParams.get("run"), alternative.id);
    assert.equal(shared.searchParams.get("race"), "2026:AL-03:house:regular");
    assert.equal(shared.searchParams.get("compare"), baseline.id);
    await page.evaluate(() => { globalThis.denyClipboard = true; });
    await page.getByRole("button", { name: "Share view" }).click();
    await expect(page.getByRole("status").filter({ hasText: "Copy this link to share the displayed forecast" })).toBeVisible();
    await expect(page.getByLabel("Link to displayed House forecast", { exact: true })).toHaveValue(sharedUrl);
    await page.goto(sharedUrl);
    await expect(page.locator(".hero > div > p")).toContainText(alternative.label);
    await expect(page.locator("#house-live-detail-title")).toHaveText("AL-03");
    await expect(compareSelect).toHaveValue(baseline.id);
    await expect(comparePanel.getByRole("table")).toHaveCount(3);
    for (const direction of ["back", "forward"]) {
      await page.goto("http://house-ui.test/?chamber=house&edition=current&race=2026%3AAL-04%3Ahouse%3Aregular");
      await expect(page.locator(".hero > div > p")).toContainText(primary.view.run.label);
      await builder.locator("summary").first().click();
      await base.selectOption("recent_calendar_history");
      await expect(page.locator(".hero > div > p")).toContainText(alternative.label);
      await page.locator('.house-ribbon-cell[data-district="AL-03"]').click();
      if (direction === "forward") {
        await page.goBack();
        await expect(page.locator(".hero > div > p")).toContainText(primary.view.run.label);
      }
      failedId = direction === "back" ? primary.view.run.id : alternative.id;
      const retained = direction === "back" ? alternative : primary.view.run;
      const retainedDistrict = direction === "back" ? "AL-03" : "AL-04";
      if (direction === "back") await page.goBack();
      else await page.goForward();
      await expect(page.getByRole("alert")).toContainText("Unable to change House models");
      await expect(page.locator(".hero > div > p")).toContainText(retained.label);
      await expect(page.locator("#house-live-detail-title")).toHaveText(retainedDistrict);
      assert.equal(new URL(page.url()).searchParams.get("run"), failedId);
      await page.getByRole("button", { name: "Share view" }).click();
      const retainedUrl = await page.evaluate(() => globalThis.copiedViewUrl);
      assert.equal(new URL(retainedUrl).searchParams.get("run"), retained.id);
      assert.equal(new URL(retainedUrl).searchParams.get("race"), `2026:${retainedDistrict}:house:regular`);
      await page.evaluate(() => { globalThis.denyClipboard = true; });
      await page.getByRole("button", { name: "Share view" }).click();
      await expect(page.getByLabel("Link to displayed House forecast", { exact: true })).toHaveValue(retainedUrl);
      await page.setViewportSize({ width: 390, height: 844 });
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
      await page.setViewportSize({ width: 1440, height: 900 });
      failedId = null;
      await page.goto(retainedUrl);
      await expect(page.locator(".hero > div > p")).toContainText(retained.label);
      await expect(page.locator("#house-live-detail-title")).toHaveText(retainedDistrict);
    }
    assert.ok(requests.every(id => publications.has(id)));
    assert.equal(await page.evaluate(() => localStorage.length), 0);
    assert.deepEqual(pageErrors, []);
  } finally { releaseDelayed?.(); await browser.close(); }
});

let routerPreview;
async function buildRouterPreview() {
  if (routerPreview) return routerPreview;
  compile("components/HouseWorkspace.tsx");
  const entry = join(directory, "router-entry.js");
  writeFileSync(entry, `import React from 'react';import {createRoot} from 'react-dom/client';import HouseWorkspace from './components/HouseWorkspace.js';
createRoot(document.getElementById('root')).render(React.createElement('div',{className:'shell'},React.createElement('p',{},'INVENTED ROUTER FIXTURE. No published election data.'),React.createElement(HouseWorkspace,{onReturn:()=>{globalThis.senateRequested=true;}})));`);
  const require = createRequire(import.meta.url), { webpack } = require("next/dist/compiled/webpack/webpack");
  await new Promise((resolve, reject) => webpack({ mode: "development", target: "web", devtool: false, entry,
    output: { path: join(directory, "router-bundle"), filename: "preview.js" } }, (error, stats) =>
    error || stats.hasErrors() ? reject(error || Error(stats.toString({ all: false, errors: true }))) : resolve()));
  const script = readFileSync(join(directory, "router-bundle/preview.js"), "utf8").replaceAll("</script", "<\\/script");
  routerPreview = `<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Invented House edition router</title><style>${readFileSync(join(root, "app/globals.css"), "utf8")}</style><body><div id="root"></div><script>${script}</script></body></html>`;
  return routerPreview;
}

test("browser House router: publication availability, edition links, failures, history, and mobile navigation", {
  skip: !process.env.BALLOTMANCY_LIVE_HOUSE_BROWSER, timeout: 120000,
}, async () => {
  const html = await buildRouterPreview();
  const current = makeHouseLivePublication();
  current.publicationReview = { limitations: ["Invented publication review note for browser verification."] };
  const historical = makeHousePublication({ eligibleCount: 16, year: 2022 });
  const historicalId = "house_history_refreshed_v1-2022-h14";
  historical.result.run.id = historicalId;
  historical.resultSha256 = fixtureSha256(historical.result);
  historical.catalog = { schemaVersion: 1, defaultRunId: historicalId,
    selectionNote: "Invented historical test catalog only.",
    runs: [{ ...historical.result.run, resultSha256: historical.resultSha256,
      raceCount: historical.result.races.length, available: true, unavailableReason: null }] };
  const earlier = makeHousePublication({ eligibleCount: 16, year: 2020 });
  const earlierId = "house_history_refreshed_v1-2020-h14";
  earlier.result.run.id = earlierId;
  earlier.result.run.configurationId = historical.result.run.configurationId;
  earlier.result.metadata.configurationId = historical.result.run.configurationId;
  earlier.resultSha256 = fixtureSha256(earlier.result);
  historical.catalog.runs.push({ ...earlier.result.run, resultSha256: earlier.resultSha256,
    raceCount: earlier.result.races.length, available: true, unavailableReason: null });
  earlier.catalog = historical.catalog;
  assert.ok(parseHousePublicationWithCatalog(historical));
  assert.ok(parseHousePublicationWithCatalog(earlier));
  const { chromium, expect } = await import("@playwright/test");
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, reducedMotion: "reduce" });
  const requests = { availability: 0, live: 0, historical: 0, group: 0 };
  const pageErrors = [];
  let availabilityStatus = 200, available = false, liveStatus = 200;
  page.on("pageerror", error => pageErrors.push(error.message));
  await page.route("http://house-router.test/**", route => {
    const url = new URL(route.request().url());
    if (url.pathname === "/") return route.fulfill({ contentType: "text/html", body: html });
    if (url.pathname === "/api/chambers/house/live/availability") {
      requests.availability++;
      return route.fulfill({ status: availabilityStatus, json: availabilityStatus === 200
        ? { schema: "HouseLiveAvailabilityV1", available } : { error: "Invented unavailable status" } });
    }
    if (url.pathname === "/api/chambers/house/live") {
      requests.live++;
      return route.fulfill({ status: liveStatus, json: liveStatus === 200 ? current : { error: "Invented current outage" } });
    }
    if (url.pathname === "/api/chambers/house") {
      requests.historical++;
      const id = url.searchParams.get("run");
      if (id === earlierId) return route.fulfill({ json: earlier });
      return route.fulfill({ status: id === null || id === historicalId ? 200 : 400,
        json: id === null || id === historicalId ? historical : { error: "Unknown invented historical run" } });
    }
    if (url.pathname.startsWith("/api/chambers/house/groups")) requests.group++;
    return route.fulfill({ status: 404, body: "No fixture endpoint" });
  });
  const go = search => page.goto(`http://house-router.test/${search}`);
  const currentHero = page.locator(".hero .eyebrow").filter({ hasText: "2026 United States House" });
  const historicalHero = page.locator(".hero .eyebrow").filter({ hasText: "2022 United States House" });
  try {
    // Bare House links may choose history only after an explicit unpublished response.
    await go("?chamber=house");
    await expect(historicalHero).toBeVisible();
    assert.equal(requests.live, 0);
    assert.ok(requests.historical > 0);
    await expect(page.getByRole("button", { name: "Current · 2026" })).toBeDisabled();
    await page.setViewportSize({ width: 390, height: 844 });
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
    await expect(page.getByRole("group", { name: "House election edition" })).toBeVisible();

    available = true;
    await page.setViewportSize({ width: 1440, height: 900 });
    await go("?chamber=house");
    await expect(currentHero).toBeVisible();
    await expect(page.getByRole("region", { name: "Publication review notes" }))
      .toContainText(current.publicationReview.limitations[0]);
    const historicalBeforeSwitch = requests.historical;
    await page.getByRole("button", { name: "Historical elections" }).click();
    await expect(historicalHero).toBeVisible();
    assert.ok(requests.historical > historicalBeforeSwitch);
    await page.goBack();
    await expect(currentHero).toBeVisible();
    await expect(page.getByRole("button", { name: "Current · 2026" })).toHaveAttribute("aria-pressed", "true");
    await page.setViewportSize({ width: 390, height: 844 });
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
    if (process.env.BALLOTMANCY_LIVE_HOUSE_SCREENSHOT_DIR) {
      mkdirSync(process.env.BALLOTMANCY_LIVE_HOUSE_SCREENSHOT_DIR, { recursive: true });
      await page.screenshot({ path: join(process.env.BALLOTMANCY_LIVE_HOUSE_SCREENSHOT_DIR, "edition-navigation-mobile.png") });
    }

    // A failing availability check is an error, never an unpublished signal.
    await page.setViewportSize({ width: 1440, height: 900 });
    availabilityStatus = 503;
    const historicalBeforeError = requests.historical;
    await go("?chamber=house");
    await expect(page.getByRole("heading", { name: "House forecast availability could not be checked" })).toBeVisible();
    assert.equal(requests.historical, historicalBeforeError);

    // Explicit current intent stays current even if both endpoints fail.
    liveStatus = 503;
    await go("?chamber=house&edition=current");
    await expect(page.getByRole("heading", { name: "Current House forecast unavailable" })).toBeVisible();
    assert.equal(requests.historical, historicalBeforeError);
    liveStatus = 200;
    availabilityStatus = 200;

    const liveBeforeHistorical = requests.live;
    await go("?chamber=house&edition=historical&year=2022");
    await expect(historicalHero).toBeVisible();
    assert.equal(requests.live, liveBeforeHistorical);
    await go(`?chamber=house&run=${historicalId}`);
    await expect(historicalHero).toBeVisible();
    assert.equal(requests.live, liveBeforeHistorical);

    // A year-only link must load that year, even when the catalog defaults to another.
    await go("?chamber=house&edition=historical&year=2020&view=actual");
    await expect(page.locator(".hero .eyebrow").filter({ hasText: "2020 United States House" })).toBeVisible();
    assert.equal(new URL(page.url()).searchParams.get("run"), earlierId);
    assert.equal(new URL(page.url()).searchParams.get("view"), "actual");
    await go("?chamber=house&edition=historical&year=2018");
    await expect(page.getByRole("heading", { name: "House view unavailable" })).toBeVisible();
    await expect(page.getByRole("alert")).toContainText("2018");
    await expect(historicalHero).toHaveCount(0);
    await page.getByRole("button", { name: "Load published default" }).click();
    await expect(historicalHero).toBeVisible();
    assert.equal(new URL(page.url()).searchParams.get("year"), null);

    const bothBeforeInvalid = { live: requests.live, historical: requests.historical };
    await go("?chamber=house&edition=current&year=2024");
    await expect(page.getByRole("heading", { name: "This House link could not be opened" })).toBeVisible();
    assert.equal(requests.live, bothBeforeInvalid.live);
    assert.equal(requests.historical, bothBeforeInvalid.historical);
    assert.equal(requests.group, 0);
    assert.equal(await page.evaluate(() => localStorage.length), 0);
    assert.deepEqual(pageErrors, []);
  } finally { await browser.close(); }
});

test("browser polling: source details follow district/model changes without mobile focus jumps", {
  skip: !process.env.BALLOTMANCY_LIVE_HOUSE_BROWSER, timeout: 120000,
}, async () => {
  const html = await buildRouterPreview();
  const publications = new Map(nativeCatalog.entries.map(entry => {
    const value = withInventedPolling(makeHouseLivePublication(entry.configuration));
    return [value.view.run.id, value];
  }));
  const primary = publications.get(catalog.defaultId);
  const { chromium, expect } = await import("@playwright/test");
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, reducedMotion: "reduce" });
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.route("http://house-polling.test/**", route => {
    const url = new URL(route.request().url());
    if (url.pathname === "/") return route.fulfill({ contentType: "text/html", body: html });
    if (url.pathname === "/api/chambers/house/live/availability")
      return route.fulfill({ json: { schema: "HouseLiveAvailabilityV1", available: true } });
    if (url.pathname === "/api/chambers/house/live")
      return route.fulfill({ json: publications.get(url.searchParams.get("run") ?? primary.view.run.id) });
    return route.fulfill({ status: 404 });
  });
  try {
    await page.goto("http://house-polling.test/?chamber=house&edition=current&race=2026%3AAL-01%3Ahouse%3Aregular");
    const panel = page.getByRole("region", { name: "Which polls are included?", exact: true });
    await expect(panel).toContainText("3 selected poll-question results");
    await expect(panel.locator("details[open]")).toHaveCount(0);
    await panel.getByText("How polls are chosen", { exact: true }).click();
    await expect(panel).toContainText("The original publication time is not verified");
    await panel.getByText("Poll details for AL-01", { exact: true }).click();
    const included = panel.getByRole("list", { name: "Included poll questions for AL-01" });
    await expect(included.getByRole("listitem")).toHaveCount(2);
    await expect(included).toContainText("Sample size not reported");
    await expect(included.getByRole("link", { name: "Poll source" }).first())
      .toHaveAttribute("href", "https://example.invalid/invented-poll-0");
    await panel.getByText("12 source question records left out for AL-01", { exact: true }).click();
    const excluded = panel.getByRole("list", { name: "Excluded source questions for AL-01" });
    await expect(excluded.getByRole("listitem")).toHaveCount(10);
    await panel.getByRole("button", { name: "See all 12 records" }).click();
    await expect(excluded.getByRole("listitem")).toHaveCount(12);
    await panel.getByRole("heading", { name: "Which polls are included?" }).scrollIntoViewIfNeeded();
    if (process.env.BALLOTMANCY_LIVE_HOUSE_SCREENSHOT_DIR) {
      mkdirSync(process.env.BALLOTMANCY_LIVE_HOUSE_SCREENSHOT_DIR, { recursive: true });
      await page.screenshot({ path: join(process.env.BALLOTMANCY_LIVE_HOUSE_SCREENSHOT_DIR, "polling-desktop.png") });
    }
    await page.setViewportSize({ width: 390, height: 844 });
    const district = panel.getByLabel("District to inspect");
    await district.focus();
    await district.selectOption("2026:AL-03:house:regular");
    await expect(district).toBeFocused();
    await expect(panel).toContainText("No eligible local poll question was selected for AL-03");
    await expect(page.locator("#house-live-detail-title")).toHaveText("AL-03");
    await expect(page.locator("#house-live-detail-title")).not.toBeFocused();
    await district.selectOption("2026:AL-01:house:regular");
    await panel.getByText("12 source question records left out for AL-01", { exact: true }).click();
    await expect(excluded.getByRole("listitem")).toHaveCount(10);
    await panel.getByText("Why 14 source question records were left out", { exact: true }).click();
    await expect(panel.getByRole("table")).toContainText("A different election or contest");
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
    if (process.env.BALLOTMANCY_LIVE_HOUSE_SCREENSHOT_DIR) {
      await district.scrollIntoViewIfNeeded();
      await page.screenshot({ path: join(process.env.BALLOTMANCY_LIVE_HOUSE_SCREENSHOT_DIR, "polling-mobile.png") });
    }
    await page.setViewportSize({ width: 1440, height: 900 });
    const builder = page.locator("details.house-configuration");
    await builder.locator("summary").first().click();
    await page.getByRole("checkbox", { name: "Include current polling evidence" }).uncheck();
    await expect(panel).toContainText("Polling is switched off");
    await expect(panel.getByRole("list", { name: /^Included poll questions/ })).toHaveCount(0);
    await expect(panel).toContainText("They are not used by this model");
    await page.getByRole("checkbox", { name: "Include current polling evidence" }).check();
    await expect(panel).toContainText("3 selected poll-question results");
    assert.equal(await page.evaluate(() => localStorage.length), 0);
    assert.deepEqual(errors, []);
  } finally { await browser.close(); }
});
