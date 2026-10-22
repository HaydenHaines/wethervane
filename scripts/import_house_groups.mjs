#!/usr/bin/env -S node --experimental-strip-types
/** Consume the sealed experimental group packet. No fitting or source writes. */
import { createHash } from "node:crypto";
import { readFile, realpath, mkdir, mkdtemp, writeFile, rename, rm } from "node:fs/promises";
import { dirname, join, resolve, relative, isAbsolute } from "node:path";
import { pathToFileURL } from "node:url";
import { parseHouseGroupResult } from "../lib/house-group-result.ts";
import { canonicalGroupJson, HOUSE_GROUP_QUALIFICATION_SHA, HOUSE_GROUP_POLICY_SHA, HOUSE_GROUP_SOURCE_CATALOG_SHA, HOUSE_GROUP_PUBLIC_CATALOG_SHA, parseHouseGroupCatalog } from "../lib/house-group-catalog.ts";
import { houseFundamentalsLabel } from "../lib/house-group-presentation.ts";
import { loadHouseGroupPublication } from "../lib/house-group-publication.ts";
import { loadHousePublication } from "../lib/house-publication.ts";

const BASE = "research/house-full-roster-20261022";
const hash = bytes => createHash("sha256").update(bytes).digest("hex");
const encoded = value => Buffer.from(canonicalGroupJson(value));
const require = (condition, message) => { if (!condition) throw Error(message); };
export async function importHouseGroups({ sourceRoot, legacyDirectory, outputDirectory }) {
  const root = await realpath(sourceRoot), legacy = await realpath(legacyDirectory), output = resolve(outputDirectory);
  const outputRelative = relative(root, output);
  require(outputRelative === ".." || outputRelative.startsWith("../"), "Output must be outside the scientific checkout.");
  async function pinned(reference) {
    require(reference && typeof reference.path === "string" && !isAbsolute(reference.path) && !reference.path.split(/[\\/]/).includes("..") && /^[a-f0-9]{64}$/.test(reference.sha256), "Invalid source pin.");
    const path = await realpath(join(root, reference.path));
    require(path.startsWith(root + "/"), "Source pin leaves checkout.");
    const bytes = await readFile(path);
    require(hash(bytes) === reference.sha256, `Source fingerprint changed: ${reference.path}`);
    return bytes;
  }
  const qualification = JSON.parse(await pinned({ path: `${BASE}/qualification-v1.json`, sha256: HOUSE_GROUP_QUALIFICATION_SHA }));
  require(qualification.status === "qualified_for_experimental_historical_group_display" && qualification.external_review.claim === false &&
    qualification.input_review.group_win_cells_passed === 64 && qualification.input_review.seat_mean_precision_cells_failed === 64,
    "Group qualification scope changed.");
  for (const pin of [...qualification.artifacts, ...qualification.cycle_audits, ...qualification.unchanged_predecessor_artifacts]) await pinned(pin);
  const catalogSource = JSON.parse(await pinned({ path: `${BASE}/frontend-results-v1/catalog.json`, sha256: HOUSE_GROUP_SOURCE_CATALOG_SHA }));
  const policy = JSON.parse(await pinned(catalogSource.policy));
  require(catalogSource.schema === "HouseFullRosterResearchCatalogV1" && catalogSource.default_selection === null && catalogSource.entries.length === 16 &&
    catalogSource.policy.sha256 === HOUSE_GROUP_POLICY_SHA && policy.capabilities.public_chamber_control === false && policy.capabilities.tipping === false && policy.capabilities.individual_O_winners === false,
    "Source group catalog or policy scope changed.");
  const legacyPublication = await loadHousePublication(legacy);
  require(legacyPublication.status === "available" && legacyPublication.catalog?.runs.length === 16, "Reviewed legacy collection is required.");
  const legacyManifest = JSON.parse(await readFile(join(legacy, "publication.json"))), legacyCatalogBytes = await readFile(join(legacy, legacyManifest.catalog.file));
  require(hash(legacyCatalogBytes) === legacyManifest.catalog.sha256, "Legacy catalog fingerprint changed.");
  const legacyCatalog = JSON.parse(legacyCatalogBytes), files = new Map(), runs = [];
  for (const entry of catalogSource.entries) {
    const sourceBytes = await pinned(entry.result), raw = JSON.parse(sourceBytes);
    const result = parseHouseGroupResult(raw, { sourcePolicySha256: HOUSE_GROUP_POLICY_SHA });
    require(result && result.id === entry.id && result.cycle === entry.cycle && result.configuration === `${entry.fundamentals_id}-loading-1` && entry.coverage === 435,
      "Reviewed group result does not match source catalog.");
    const old = legacyCatalog.runs.find(run => run.id === result.legacyRunId);
    require(old?.available && old.geography && old.year === result.cycle && old.configurationId === entry.fundamentals_id,
      "Exact legacy share result and matching geography are required.");
    const oldSourceBytes = await pinned({ path: raw.legacy_share_packet, sha256: old.resultSha256 });
    const oldResult = JSON.parse(oldSourceBytes);
    const originalIds = oldResult.races.map(race => race.id).sort();
    require(JSON.stringify(originalIds) === JSON.stringify(result.races.filter(race => race.voteShareReferenceAvailable).map(race => race.id).sort()), "Legacy share coverage differs.");
    const geometryBytes = await readFile(join(legacy, old.geography.file));
    require(hash(geometryBytes) === old.geography.sha256, "Legacy geometry source bytes changed.");
    const geographyContentSha256 = hash(encoded(JSON.parse(geometryBytes)));
    const bytes = encoded(result), viewSha256 = hash(bytes), file = `view-${viewSha256}.json`;
    files.set(file, bytes);
    runs.push({ id: result.id, cycle: result.cycle, fundamentalsId: entry.fundamentals_id,
      label: houseFundamentalsLabel(entry.fundamentals_id), resultSha256: entry.result.sha256, viewSha256,
      legacyRunId: result.legacyRunId, legacyResultSha256: old.resultSha256,
      mapVintage: old.geography.metadata.mapVintage, geographySha256: old.geography.sha256, geographyContentSha256,
      view: { file, sha256: viewSha256 }, geography: old.geography });
  }
  const catalog = { schemaVersion: 1, qualificationSha256: HOUSE_GROUP_QUALIFICATION_SHA, sourcePolicySha256: HOUSE_GROUP_POLICY_SHA,
    sourceCatalogSha256: HOUSE_GROUP_SOURCE_CATALOG_SHA, initialRunId: `${legacyCatalog.defaultRunId}-full435-v1`, runs };
  const publicCatalog = parseHouseGroupCatalog(catalog);
  require(publicCatalog && hash(encoded(publicCatalog)) === HOUSE_GROUP_PUBLIC_CATALOG_SHA, "Derived public catalog differs from reviewed identity bindings.");
  files.set(`bindings-${HOUSE_GROUP_PUBLIC_CATALOG_SHA}.json`, encoded(publicCatalog));
  const catalogBytes = encoded(catalog), catalogSha256 = hash(catalogBytes), catalogFile = `catalog-${catalogSha256}.json`;
  files.set(catalogFile, catalogBytes);
  const manifest = { schemaVersion: 1, status: "available", catalog: { file: catalogFile, sha256: catalogSha256 } };
  const parent = await realpath(dirname(output));
  require(parent === dirname(output), "Output parent must not traverse symlinks.");
  const stage = await mkdtemp(join(parent, ".house-groups-"));
  try {
    for (const [name, bytes] of files) await writeFile(join(stage, name), bytes, { flag: "wx" });
    await writeFile(join(stage, "publication.json"), encoded(manifest), { flag: "wx" });
    for (const run of runs) await loadHouseGroupPublication(stage, legacy, run.id);
    await mkdir(output, { recursive: true });
    require(await realpath(output) === output, "Output must not be a symlink.");
    for (const [name, bytes] of files) {
      try { await writeFile(join(output, name), bytes, { flag: "wx" }); }
      catch (error) { if (error.code !== "EEXIST") throw error; require(hash(await readFile(join(output, name))) === hash(bytes), "Conflicting group artifact."); }
    }
    await rename(join(stage, "publication.json"), join(output, "publication.json"));
    return { runs: runs.length, catalogSha256, sourceQualificationSha256: HOUSE_GROUP_QUALIFICATION_SHA };
  } finally { await rm(stage, { recursive: true, force: true }); }
}
if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const args = process.argv.slice(2);
  if (args.length !== 6 || args[0] !== "--source-root" || args[2] !== "--legacy-dir" || args[4] !== "--output-dir") throw Error("Expected --source-root, --legacy-dir and --output-dir.");
  console.log(JSON.stringify(await importHouseGroups({ sourceRoot: args[1], legacyDirectory: args[3], outputDirectory: args[5] })));
}
