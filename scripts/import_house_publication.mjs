#!/usr/bin/env -S node --experimental-strip-types
/** Admit completed reviewed outputs; no models, fits, or source writes. */
import { createHash } from "node:crypto";
import { readFile, realpath, mkdir, mkdtemp, rm, writeFile, rename, lstat } from "node:fs/promises";
import { dirname, isAbsolute, join, relative, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { parseHouseRun, parseHouseCatalog, houseRunMatches } from "../lib/house-catalog.ts";
import { parseHouseResult } from "../lib/house-result.ts";
import { loadHousePublication } from "../lib/house-publication.ts";

const require = (condition, message) => { if (!condition) throw Error(message); };
const digest = bytes => createHash("sha256").update(bytes).digest("hex");
const bytesOf = value => Buffer.from(JSON.stringify(value, null, 2) + "\n");
const record = value => !!value && typeof value === "object" && !Array.isArray(value);
const inside = (root, path) => { const rel = relative(root, path); return rel === "" || (!rel.startsWith("..") && !isAbsolute(rel)); };

/** Exported for isolated synthetic tests. Output requires an explicit directory. */
export async function importHousePublication({ sourceRoot, catalogPath, reviewReceiptPath, defaultRunId, outputDirectory }) {
  require([sourceRoot, catalogPath, reviewReceiptPath, defaultRunId, outputDirectory].every(value => typeof value === "string" && value.length), "Explicit source, catalog, review receipt, default run, and output directory are required.");
  const root = await realpath(sourceRoot);
  async function sourceFile(path) {
    require(typeof path === "string" && path.length > 0, "Invalid source path.");
    const resolved = await realpath(resolve(root, path));
    require(inside(root, resolved), "Source artifact escapes source root.");
    return { path: resolved, bytes: await readFile(resolved) };
  }
  async function pinned(pin) {
    require(record(pin) && typeof pin.path === "string" && !isAbsolute(pin.path) && !pin.path.split(/[\\/]/).includes("..") &&
      typeof pin.sha256 === "string" && /^[a-f0-9]{64}$/.test(pin.sha256), "Invalid source fingerprint reference.");
    const file = await sourceFile(pin.path);
    require(digest(file.bytes) === pin.sha256, "Source artifact fingerprint mismatch.");
    return file;
  }
  const catalogFile = await sourceFile(catalogPath), receiptFile = await sourceFile(reviewReceiptPath);
  const sourceCatalog = JSON.parse(catalogFile.bytes), receipt = JSON.parse(receiptFile.bytes);
  require(sourceCatalog.schema === "house-historical-capability-catalog-v1" && Array.isArray(sourceCatalog.configurations) &&
    sourceCatalog.configurations.length > 0 && sourceCatalog.configurations.length <= 128 && sourceCatalog.default_selection === null,
    "Invalid reviewed House source catalog.");
  require(receipt.schema === "house-actual-output-review-v1" && Array.isArray(receipt.outputs) && Array.isArray(receipt.cells), "Invalid House review receipt.");
  const reviewedOutputs = new Map();
  for (const pin of receipt.outputs) {
    const file = await pinned(pin);
    require(!reviewedOutputs.has(file.path), "Duplicate reviewed output.");
    reviewedOutputs.set(file.path, pin.sha256);
  }
  require(reviewedOutputs.get(catalogFile.path) === digest(catalogFile.bytes), "Catalog is not pinned by the review receipt.");
  for (const key of ["review_source", "panel", "scores"]) await pinned(receipt[key]);
  const audits = new Map();
  for (const cell of receipt.cells) {
    require(record(cell) && Number.isInteger(cell.cycle) && typeof cell.configuration === "string" &&
      cell.structural_and_score_audit_passed === true && ["passed", "diagnostic_failure"].includes(cell.numerical_status), "Incomplete House cell review.");
    const key = JSON.stringify([cell.cycle, cell.configuration]);
    require(!audits.has(key), "Duplicate reviewed comparison cell.");
    const fit = JSON.parse((await pinned(cell.fit_receipt)).bytes);
    require(fit.status === cell.numerical_status, "Fit status conflicts with its review.");
    const plan = JSON.parse((await pinned(fit.plan)).bytes);
    require(plan.target_cycle === cell.cycle && plan.house_configuration_id === cell.configuration, "Fit plan identity conflicts with its review.");
    audits.set(key, { ...cell, planSha256: fit.plan.sha256 });
  }
  require(audits.size === sourceCatalog.configurations.length, "Review cells do not match catalog support.");
  const runs = [], files = new Map(), cells = new Set();
  for (const entry of sourceCatalog.configurations) {
    const run = parseHouseRun(entry);
    require(run && run.historical && typeof entry.publication_eligible === "boolean" &&
      Number.isInteger(entry.race_count) && entry.race_count > 0 && entry.race_count <= 435, "Invalid source run metadata.");
    const key = JSON.stringify([run.year, run.configurationId]);
    const cell = audits.get(key);
    require(cell && !cells.has(key) && cell.races === entry.race_count && run.configurationSha256 === cell.planSha256, "Catalog cell or configuration fingerprint does not match review.");
    cells.add(key);
    const resultFile = await pinned(entry.result);
    require(reviewedOutputs.get(resultFile.path) === entry.result.sha256, "Result is not pinned by review outputs.");
    const rawResult = JSON.parse(resultFile.bytes), resultRun = parseHouseRun(rawResult.run);
    require(resultRun && houseRunMatches(run, resultRun) && Array.isArray(rawResult.races) && rawResult.races.length === entry.race_count &&
      entry.publication_eligible === (rawResult.reviewStatus === "reviewed") &&
      entry.publication_eligible === (cell.numerical_status === "passed"), "Catalog metadata or eligibility conflicts with result/review.");
    if (entry.publication_eligible) {
      const result = parseHouseResult(rawResult);
      require(result && !result.metadata.full_chamber && result.metadata.scope === "eligible_race_subset", "Reviewed historical result is invalid or exceeds source catalog scope.");
      const filename = `result-${entry.result.sha256}.json`;
      files.set(filename, resultFile.bytes);
      runs.push({ ...run, resultSha256: entry.result.sha256, raceCount: entry.race_count, available: true, unavailableReason: null,
        result: { file: filename, sha256: entry.result.sha256 } });
    } else {
      require(rawResult.reviewStatus === "diagnostic_failure", "Unrecognized unqualified result status.");
      runs.push({ ...run, resultSha256: null, raceCount: entry.race_count, available: false,
        unavailableReason: "Numerical diagnostics did not pass review; this result is not available for publication." });
    }
  }
  const catalog = { schemaVersion: 1, defaultRunId, selectionNote: sourceCatalog.selection_note, runs, capabilities: sourceCatalog.capabilities };
  require(parseHouseCatalog(catalog), "Explicit default must identify an available reviewed result, and catalog capabilities must be valid.");
  require(catalog.capabilities.historical_races === true && Object.entries(catalog.capabilities).every(([key, value]) => key === "historical_races" || value === false), "Historical source catalog claims unsupported capabilities.");
  const catalogBytes = bytesOf(catalog), catalogSha256 = digest(catalogBytes), catalogFilename = `catalog-${catalogSha256}.json`;
  files.set(catalogFilename, catalogBytes);
  const manifest = { schemaVersion: 1, status: "available", catalog: { file: catalogFilename, sha256: catalogSha256 } };
  const output = resolve(outputDirectory);
  require(!inside(root, output), "Output must be outside read-only source root.");
  const outputParent = await realpath(dirname(output));
  require(outputParent === dirname(output) && !inside(root, output), "Output must be outside read-only source root and must not traverse symlinks.");
  try { require(!(await lstat(output)).isSymbolicLink(), "Output directory must not be a symlink."); }
  catch (error) { if (error.code !== "ENOENT") throw error; }
  const staging = await mkdtemp(join(outputParent, ".house-publication-"));
  try {
    for (const [filename, bytes] of files) await writeFile(join(staging, filename), bytes, { flag: "wx" });
    await writeFile(join(staging, "publication.json"), bytesOf(manifest), { flag: "wx" });
    await loadHousePublication(staging);
    await mkdir(output, { recursive: true });
    for (const [filename, bytes] of files) {
      try { await writeFile(join(output, filename), bytes, { flag: "wx" }); }
      catch (error) {
        if (error.code !== "EEXIST") throw error;
        const existing = await realpath(join(output, filename));
        require(dirname(existing) === output && digest(await readFile(existing)) === digest(bytes), "Existing publication artifact conflicts with import.");
      }
    }
    await rename(join(staging, "publication.json"), join(output, "publication.json"));
    return { defaultRunId, availableRuns: runs.filter(run => run.available).length, unavailableRuns: runs.filter(run => !run.available).length, catalogSha256 };
  } finally { await rm(staging, { recursive: true, force: true }); }
}

async function main() {
  const options = {}, names = { "--source-root": "sourceRoot", "--catalog": "catalogPath", "--review-receipt": "reviewReceiptPath", "--default-run": "defaultRunId", "--output-dir": "outputDirectory" };
  const args = process.argv.slice(2);
  for (let index = 0; index < args.length; index += 2) {
    const name = names[args[index]];
    require(name && args[index + 1] && !options[name], "Expected --source-root, --catalog, --review-receipt, --default-run and --output-dir exactly once.");
    options[name] = args[index + 1];
  }
  console.log(JSON.stringify(await importHousePublication(options)));
}
if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch(error => { console.error(error.message); process.exitCode = 1; });
}
