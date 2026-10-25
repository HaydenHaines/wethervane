import test, { after } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync, rmSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import ts from "typescript";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { loadCountCatalogView } from "../lib/count-catalog-server.ts";
import { countCatalogPresentation } from "../lib/count-catalog-view.ts";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const directory = mkdtempSync(join(root, "tests", ".count-catalog-preview-"));
after(() => rmSync(directory, { recursive: true, force: true }));
writeFileSync(join(directory, "package.json"), '{"type":"module"}\n');
const relative = "components/CountCatalogPreview.tsx";
const compiled = ts.transpileModule(readFileSync(join(root, relative), "utf8"), { fileName: relative, compilerOptions: {
  target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext, jsx: ts.JsxEmit.ReactJSX, rewriteRelativeImportExtensions: true,
} }).outputText;
mkdirSync(join(directory, "components"));
writeFileSync(join(directory, "components/CountCatalogPreview.js"), compiled);
const { default: Preview } = await import(pathToFileURL(join(directory, "components/CountCatalogPreview.js")).href);
const catalog = await loadCountCatalogView(join(root, "data/count-catalog"));
const view = countCatalogPresentation(catalog);
const render = (presentation = view) => renderToStaticMarkup(React.createElement(Preview, { view: presentation }));
const standalone = () => `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Private Texas count research catalog</title><style>${readFileSync(join(root, "app/globals.css"), "utf8")}</style></head><body><main style="padding:20px">${render()}</main></body></html>`;

test("actual metadata render keeps scope, result, evidence and site state explicit without forecast charts", () => {
  const html = render();
  assert.match(html, /Private research setup/);
  assert.match(html, /Prepared for the 2026 election/);
  assert.match(html, /2020 and 2024/);
  assert.match(html, /Earlier elections specified for this setup/);
  assert.match(html, /does not support changing them independently/);
  assert.match(html, /datetime="2026-11-03"/i);
  assert.match(html, /datetime="2026-10-20T22:13:33Z"/i);
  assert.match(html, /No completed forecast supplied/);
  assert.match(html, /Forecast accuracy not assessed here/);
  assert.match(html, /cannot schedule a calculation/);
  assert.match(html, /does not report current job activity/);
  assert.equal((html.match(/data-count-role=/g) || []).length, 8);
  assert.equal((html.match(/data-catalog-status=/g) || []).length, 4);
  assert.doesNotMatch(html, /<(?:input|select|button|svg|canvas)\b|data-quantity=|count-value|progressbar|saved forecast loads/i);
});

test("technical metadata is optional, escaped and does not expose native source paths", () => {
  const html = render();
  assert.equal((html.match(/<details\b/g) || []).length, 9);
  assert.doesNotMatch(html, /<details[^>]*\bopen(?:=|\s|>)/);
  assert.match(html, /candidate_TX_ballot_role_profile_choice_v1/);
  assert.match(html, /requires_computation/);
  assert.match(html, /No adjustable settings in this recorded version/);
  assert.doesNotMatch(html, /\/home\/|file:\/\/|execute\(|entrypoint|prepared-v4/);
  const input = structuredClone(catalog);
  input.roles[0].providers[0].limitations = ["<img onerror=alert(1)>"];
  const escaped = render(countCatalogPresentation(input));
  assert.match(escaped, /&lt;img onerror=alert\(1\)&gt;/);
  assert.doesNotMatch(escaped, /<img\b/);
});

test("headings and details keep labels unique across multiple preview instances", () => {
  const html = renderToStaticMarkup(React.createElement("div", null, React.createElement(Preview, { view }), React.createElement(Preview, { view })));
  const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map(match => match[1]);
  assert.equal(new Set(ids).size, ids.length);
  for (const label of [...html.matchAll(/aria-labelledby="([^"]+)"/g)].map(match => match[1])) assert.ok(ids.includes(label));
});

test("invalid metadata has an honest unavailable state with no invented components or results", () => {
  const html = render(null);
  assert.match(html, /Count research catalog unavailable/);
  assert.doesNotMatch(html, /data-count-role=|<svg|<canvas|No completed forecast supplied/);
});

test("the private catalog has no production imports or execution, storage and fetch actions", () => {
  function files(path) { return readdirSync(path, { withFileTypes: true }).flatMap(entry => entry.isDirectory() ? files(join(path, entry.name)) : [join(path, entry.name)]); }
  for (const file of files(join(root, "app")).filter(path => /\.[cm]?[jt]sx?$/.test(path)))
    assert.doesNotMatch(readFileSync(file, "utf8"), /(?:from\s*|import\s*\(|require\s*\()["'][^"']*(?:CountCatalogPreview|count-catalog)/, file);
  for (const file of ["components/CountCatalogPreview.tsx", "lib/count-catalog-view.ts"])
    assert.doesNotMatch(readFileSync(join(root, file), "utf8"), /localStorage|sessionStorage|fetch\(|onClick|onChange|ModelSliceControls/);
});

test("write optional standalone actual-metadata HTML without a public route", { skip: !process.env.BALLOTMANCY_COUNT_CATALOG_PREVIEW_OUTPUT }, () => {
  writeFileSync(process.env.BALLOTMANCY_COUNT_CATALOG_PREVIEW_OUTPUT, standalone());
});

test("browser: collapsed details open with the keyboard and fit mobile/desktop", { skip: !process.env.BALLOTMANCY_COUNT_CATALOG_BROWSER, timeout: 60000 }, async () => {
  const { chromium, expect } = await import("@playwright/test");
  const browser = await chromium.launch({ headless: true });
  try {
    for (const width of [375, 1440]) {
      const page = await browser.newPage({ viewport: { width, height: 900 }, reducedMotion: "reduce" });
      await page.setContent(standalone());
      await expect(page.getByRole("heading", { name: "Texas count-model research" })).toBeVisible();
      await expect(page.locator("[data-count-role]")).toHaveCount(8);
      await expect(page.locator("input,select,button,svg,canvas")).toHaveCount(0);
      const details = page.locator('[data-count-role="population"] details');
      await expect(details).not.toHaveAttribute("open", "");
      const summary = details.locator("summary");
      await summary.focus();
      await page.keyboard.press("Enter");
      await expect(details).toHaveAttribute("open", "");
      await expect(details.getByText("protected_full_parent_exposure_v1", { exact: true })).toBeVisible();
      const identity = page.locator("details.catalog-details");
      await identity.locator("summary").focus();
      await page.keyboard.press("Space");
      await expect(identity).toHaveAttribute("open", "");
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true);
      if (process.env.BALLOTMANCY_COUNT_CATALOG_SCREENSHOT_PREFIX)
        await page.screenshot({ path: `${process.env.BALLOTMANCY_COUNT_CATALOG_SCREENSHOT_PREFIX}-${width}.png`, fullPage: true });
      await page.close();
    }
  } finally {
    await browser.close();
  }
});
