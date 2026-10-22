#!/usr/bin/env -S node --experimental-strip-types
/** Attach independently reviewed boundaries without rewriting scientific results. */
import { createHash } from "node:crypto";
import { readFile, realpath, mkdtemp, writeFile, rename, rm } from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { loadHousePublication } from "../lib/house-publication.ts";
import { joinDistrictGeography } from "../lib/district-geography.ts";

const digest = bytes => createHash("sha256").update(bytes).digest("hex");
const encode = value => Buffer.from(JSON.stringify(value, null, 2) + "\n");
const require = (condition, message) => { if (!condition) throw Error(message); };
const filename = value => typeof value === "string" && /^[a-zA-Z0-9][a-zA-Z0-9_-]*\.(?:geojson|json)$/.test(value);
const hash = value => typeof value === "string" && /^[a-f0-9]{64}$/.test(value);
async function local(root, name) {
  require(filename(name), "Invalid local artifact filename.");
  const path = await realpath(join(root, name));
  require(dirname(path) === root, "Artifact must stay within its directory.");
  return readFile(path);
}

export async function attachHouseGeography({ publicationDirectory, geographyManifest }) {
  const directory = await realpath(publicationDirectory);
  const inputPath = await realpath(geographyManifest), inputRoot = dirname(inputPath);
  const input = JSON.parse(await local(inputRoot, basename(inputPath)));
  require(input.schemaVersion === 1 && Array.isArray(input.editions) && input.editions.length > 0, "No reviewed geography editions supplied.");
  // Validate the original publication before preparing any new references.
  await loadHousePublication(directory);
  const manifest = JSON.parse(await local(directory, "publication.json"));
  require(manifest.catalog && !manifest.result && !manifest.geography, "A catalog publication is required.");
  const originalCatalogBytes = await local(directory, manifest.catalog.file);
  require(digest(originalCatalogBytes) === manifest.catalog.sha256, "Original catalog fingerprint mismatch.");
  const catalog = JSON.parse(originalCatalogBytes);
  const files = new Map([[manifest.catalog.file, originalCatalogBytes]]);
  for (const run of catalog.runs) {
    if (!run.available) continue;
    for (const reference of [run.result, run.geography].filter(Boolean)) {
      const bytes = await local(directory, reference.file);
      require(digest(bytes) === reference.sha256, "Original artifact fingerprint mismatch.");
      files.set(reference.file, bytes);
    }
  }
  const seen = new Set(), attached = [];
  for (const edition of input.editions) {
    require(Number.isInteger(edition.cycle) && edition.cycle > 0 && !seen.has(edition.cycle) &&
      edition.mapVintage === `house_election_map_${edition.cycle}` && hash(edition.sha256) &&
      typeof edition.sourceName === "string" && edition.sourceName.trim().length > 0 &&
      typeof edition.sourceUrl === "string" && new URL(edition.sourceUrl).protocol === "https:" &&
      typeof edition.sourceNote === "string" && edition.sourceNote.trim().length > 0,
    "Invalid or duplicate geography edition or source attribution.");
    seen.add(edition.cycle);
    const bytes = await local(inputRoot, edition.file);
    require(digest(bytes) === edition.sha256, "Geography fingerprint mismatch.");
    const geojson = JSON.parse(bytes);
    require(geojson.type === "FeatureCollection" && geojson.features?.length === 435, "A complete 435-district map is required.");
    const metadata = { cycle: edition.cycle, mapVintage: edition.mapVintage, admission: "verified",
      sourceSha256: edition.sha256, sourceName: edition.sourceName, sourceUrl: edition.sourceUrl, sourceNote: edition.sourceNote };
    const targetRuns = catalog.runs.filter(run => run.available && run.year === edition.cycle);
    require(targetRuns.length > 0, "Geography edition has no published results.");
    const file = `geometry-${edition.sha256}.json`;
    for (const run of targetRuns) {
      const result = JSON.parse(files.get(run.result.file));
      require(result.metadata.mapVintage === edition.mapVintage, "Geometry does not match the result's map edition.");
      const keys = geojson.features.map(feature => feature.properties?.districtKey).sort();
      require(JSON.stringify(keys) === JSON.stringify(result.coverage.map(row => row.districtKey).sort()),
        "Geometry districts do not match the complete election coverage ledger.");
      const joined = joinDistrictGeography(result.races.map(race => ({ race, districtKey: race.district.districtKey })),
        { cycle: edition.cycle, mapVintage: edition.mapVintage }, { metadata, geojson });
      require(["ready", "partial"].includes(joined.status) && joined.matches.length === result.races.length && !joined.missingDistrictKeys.length,
        "Geometry does not match every published race.");
      run.geography = { file, sha256: edition.sha256, metadata };
      attached.push({ runId: run.id, cycle: edition.cycle, modeled: joined.matches.length, gray: joined.unmatchedFeatures.length });
    }
    files.set(file, bytes);
  }
  catalog.capabilities.district_geometry = true;
  const catalogBytes = encode(catalog), catalogSha256 = digest(catalogBytes), catalogFile = `catalog-${catalogSha256}.json`;
  files.set(catalogFile, catalogBytes);
  const nextManifest = { ...manifest, catalog: { file: catalogFile, sha256: catalogSha256 } };
  const staging = await mkdtemp(join(dirname(directory), ".house-geography-"));
  try {
    for (const [name, bytes] of files) await writeFile(join(staging, name), bytes, { flag: "wx" });
    await writeFile(join(staging, "publication.json"), encode(nextManifest), { flag: "wx" });
    await loadHousePublication(staging);
    for (const [name, bytes] of files) {
      try { await writeFile(join(directory, name), bytes, { flag: "wx" }); }
      catch (error) {
        if (error.code !== "EEXIST") throw error;
        require(digest(await local(directory, name)) === digest(bytes), "Existing artifact conflicts with geography attachment.");
      }
    }
    await rename(join(staging, "publication.json"), join(directory, "publication.json"));
    return { catalogSha256, attached };
  } finally { await rm(staging, { recursive: true, force: true }); }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const args = process.argv.slice(2);
  if (args.length !== 4 || args[0] !== "--publication-dir" || args[2] !== "--geography-manifest")
    throw Error("Usage: attach_house_geography.mjs --publication-dir DIRECTORY --geography-manifest FILE");
  console.log(JSON.stringify(await attachHouseGeography({ publicationDirectory: args[1], geographyManifest: args[3] }), null, 2));
}
