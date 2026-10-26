#!/usr/bin/env node
/** Build the runtime scorecard reference index from published, byte-pinned catalogs. */
import { createHash } from "node:crypto";
import { readFileSync, mkdirSync, writeFileSync, existsSync } from "node:fs";
import { dirname, join, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

export const REPO = fileURLToPath(new URL("../", import.meta.url));
export const FULL = "data/election-night-reference-publication.json";
export const PRIMARY = "data/election-night-core-reference-publication.json";
export const HOUSE = "data/election-night-house-reference-publication.json";
export const OUTPUT = "public/data/election-night/publication-index.json";
export const HREFS = { full: "/data/election-night/references.json", primary: "/data/election-night/core-references.json",
  house: "/data/election-night/house-references.json" };

function requireValue(value, message) {
  if (!value) throw Error(`Election-night publication index: ${message}`);
}
function readJson(path) {
  const value = JSON.parse(readFileSync(path, "utf8"));
  requireValue(value && typeof value === "object" && !Array.isArray(value), `Expected JSON object: ${path}`);
  return value;
}
const hash = value => typeof value === "string" && /^[a-f0-9]{64}$/.test(value);
const digest = bytes => createHash("sha256").update(bytes).digest("hex");
const canonical = value => Array.isArray(value) ? value.map(canonical) : value && typeof value === "object"
  ? Object.fromEntries(Object.entries(value).sort(([a], [b]) => a.localeCompare(b)).map(([key, item]) => [key, canonical(item)])) : value;
const sameObject = (a, b) => JSON.stringify(canonical(a)) === JSON.stringify(canonical(b));
const HOUSE_PIN_KEYS = ["collectionSha256", "collectionReceiptSha256", "nativeCatalogSha256", "readyPacketSha256",
  "referenceSha256", "sidecarReceiptSha256", "reviewSha256", "auditorSha256", "sourceReceiptSha256",
  "jointDrawsSha256", "sourcePlanSha256", "conditionalSharesSha256", "jointMetricArchiveSha256",
  "scientificReviewSha256", "nativeNumericalReviewSha256", "qualificationSha256", "selectionSha256",
  "builderSha256", "operatorReviewSha256"];

export function publication(repo, role, manifestName) {
  const value = readJson(join(repo, manifestName));
  requireValue(value.schemaVersion === 1 && value.href === HREFS[role] && hash(value.sha256) &&
    Array.isArray(value.editions), `Invalid ${role} publication manifest`);
  const catalogBytes = readFileSync(join(repo, "public", value.href.slice(1)));
  requireValue(digest(catalogBytes) === value.sha256, `Changed ${role} catalog bytes`);
  const catalog = JSON.parse(catalogBytes.toString("utf8"));
  requireValue(catalog && typeof catalog === "object" && catalog.schemaVersion === 1 &&
    Array.isArray(catalog.entries), `Invalid ${role} catalog`);
  const registry = readJson(join(repo, "public/data/explorer/registry.json"));
  const runs = new Map((registry.runs ?? []).filter(row => row && typeof row === "object" && "id" in row)
    .map(row => [row.id, row]));
  const identities = [], descriptors = [];
  for (const entry of catalog.entries) {
    requireValue(entry && typeof entry === "object" && !Array.isArray(entry), `Invalid ${role} catalog entry`);
    identities.push({ chamber: entry.chamber, runId: entry.runId, resultSha256: entry.resultSha256 });
    const run = runs.get(entry.runId);
    requireValue(entry.chamber === "senate" && typeof entry.targetId === "string" &&
      typeof entry.lawId === "string" && typeof entry.forecastIssuedAt === "string" &&
      typeof entry.forecastCutoffAt === "string", `Invalid ${role} edition binding`);
    // An old reference may remain published after its run ID points to new bytes.
    if (run && run.sha256 === entry.resultSha256) {
      requireValue(run.targetId === entry.targetId && run.lawId === entry.lawId &&
        run.issuedAt === entry.forecastIssuedAt && run.cutoffAt === entry.forecastCutoffAt,
      `${role} current reference edition metadata changed`);
      const path = join(repo, "public", run.href.slice(1));
      const bytes = readFileSync(path), bundle = JSON.parse(bytes.toString("utf8"));
      requireValue(digest(bytes) === run.sha256 &&
        bundle.provenance?.sources?.forecastSha256 === entry.sourcePins?.forecastSha256,
      `${role} current native forecast provenance changed`);
    }
    const ref = entry.reference ?? {};
    const draw = role === "full" ? ref.drawIdentity : ref.draw_identity;
    requireValue(draw && Array.isArray(draw.shape) && entry.sourcePins &&
      typeof entry.sourcePins === "object" && !Array.isArray(entry.sourcePins) &&
      Object.values(entry.sourcePins).every(hash), `Invalid ${role} reviewed source pins`);
    let rowCount = 1;
    for (const dimension of draw.shape) {
      requireValue(Number.isSafeInteger(dimension) && dimension > 0, `Invalid ${role} draw shape`);
      rowCount *= dimension;
    }
    requireValue(Number.isSafeInteger(rowCount) && rowCount > 0, `Invalid ${role} draw row count`);
    const descriptor = {
      targetId: entry.targetId, lawId: entry.lawId,
      forecastIssuedAt: entry.forecastIssuedAt, forecastCutoffAt: entry.forecastCutoffAt,
      referenceIssuedAt: role === "full" ? entry.referenceIssuedAt : ref.issued_utc,
      cohortSha256: role === "full" ? ref.cohortSha256 : ref.common_target_cohort_sha256,
      sourcePins: entry.sourcePins, shape: draw.shape, rowCount,
    };
    if (run && run.sha256 === entry.resultSha256) {
      requireValue(run.components && typeof run.components.base === "string" &&
        typeof run.components.polling === "string" && typeof run.components.uncertainty === "string" &&
        typeof run.strictAsOf === "boolean", `Invalid ${role} current model configuration`);
      descriptor.configuration = { base: run.components.base, polling: run.components.polling,
        uncertainty: run.components.uncertainty };
      descriptor.strictAsOf = run.strictAsOf;
    }
    if (role === "full") descriptor.structuralCutoffAt = entry.structuralCutoffAt;
    else {
      const oldFCutoff = catalog.entries.find(row => row.reference?.arm === "F")?.forecastCutoffAt;
      const currentRun = run && run.sha256 === entry.resultSha256;
      const structural = ref.arm === "F" ? entry.forecastCutoffAt : currentRun
        ? run.inputCutoffs?.find(row => row.label === "Structural inputs")?.at : oldFCutoff;
      requireValue(typeof structural === "string" && structural, `Missing ${role} structural cutoff`);
      Object.assign(descriptor, {
        arm: ref.arm, cohortId: ref.cohort_id,
        excludedRaceIds: ref.excluded_from_this_reference,
        raceAxisBinding: ref.race_axis_binding, structuralCutoffAt: structural,
      });
    }
    descriptors.push(descriptor);
  }
  requireValue(JSON.stringify(value.editions) === JSON.stringify(identities) &&
    new Set(identities.map(item => `${item.chamber}:${item.runId}:${item.resultSha256}`)).size === identities.length,
  `${role} publication editions differ from the catalog`);
  return { ...value, editions: identities.map((identity, index) => ({ ...identity, qualification: descriptors[index] })) };
}

export function build(repo = REPO) {
  const index = { schemaVersion: 1, full: publication(repo, "full", FULL),
    primary: publication(repo, "primary", PRIMARY) };
  const houseManifest = existsSync(join(repo, HOUSE)), houseCatalog = existsSync(join(repo, "public", HREFS.house.slice(1)));
  requireValue(houseManifest === houseCatalog, "House publication manifest/catalog must be installed together");
  if (houseManifest) index.house = housePublication(repo);
  return index;
}

export function housePublication(repo) {
  const value = readJson(join(repo, HOUSE));
  requireValue(value.schemaVersion === 1 && value.href === HREFS.house && hash(value.sha256) &&
    Array.isArray(value.editions), "Invalid House publication manifest");
  const bytes = readFileSync(join(repo, "public", HREFS.house.slice(1)));
  requireValue(digest(bytes) === value.sha256, "Changed House catalog bytes");
  const catalog = JSON.parse(bytes.toString("utf8"));
  requireValue(catalog.schema === "HouseExpectedErrorPublicCatalogV1" && catalog.schemaVersion === 1 &&
    catalog.chamber === "house" && catalog.evaluationLock === false &&
    catalog.observedResultsConsumed === false && Array.isArray(catalog.entries), "Invalid House catalog");
  const current = readJson(join(repo, "data/house-live/catalog.json"));
  const runs = new Map(current.entries.map(entry => [entry.id, entry]));
  const editions = catalog.entries.map(entry => {
    requireValue(entry.chamber === "house" && typeof entry.runId === "string" && hash(entry.resultSha256) &&
      entry.configuration && typeof entry.configuration === "object" &&
      [entry.configuration.base, entry.configuration.polling, entry.configuration.uncertainty,
        entry.forecastIssuedAt, entry.forecastCutoffAt, entry.referenceIssuedAt].every(v => typeof v === "string" && v) &&
      Number.isSafeInteger(entry.rows) && entry.rows > 0 && Number.isSafeInteger(entry.chains) && entry.chains > 0 &&
      entry.rows % entry.chains === 0 && entry.winnerRaceCount === 435 && entry.pairRaceCount === 412 &&
      entry.pairInapplicableRaceCount === 23 && entry.sourcePins &&
      HOUSE_PIN_KEYS.every(key => hash(entry.sourcePins[key])) &&
      Object.values(entry.sourcePins).every(hash) &&
      Number.isFinite(Date.parse(entry.forecastIssuedAt)) && Number.isFinite(Date.parse(entry.forecastCutoffAt)) &&
      Number.isFinite(Date.parse(entry.referenceIssuedAt)) &&
      Date.parse(entry.forecastCutoffAt) <= Date.parse(entry.forecastIssuedAt) &&
      Date.parse(entry.forecastIssuedAt) <= Date.parse(entry.referenceIssuedAt), "Invalid House reference entry");
    const run = runs.get(entry.runId);
    if (run && run.result.sha256 === entry.resultSha256) {
      requireValue(sameObject(run.configuration, entry.configuration) &&
        /^results\/[A-Za-z0-9_-]+\.json$/.test(run.result.path), "House current reference configuration changed");
      const root = resolve(repo, "data/house-live"), source = resolve(root, run.result.path);
      requireValue(source.startsWith(`${root}${sep}`), "House current result path escaped its root");
      const nativeBytes = readFileSync(source), native = JSON.parse(nativeBytes.toString("utf8"));
      requireValue(digest(nativeBytes) === run.result.sha256 && native.id === entry.runId &&
        native.issued_at === entry.forecastIssuedAt && native.forecast_cutoff === entry.forecastCutoffAt &&
        native.joint_draws.sha256 === entry.sourcePins.jointDrawsSha256 &&
        native.joint_draws.count === entry.rows && native.joint_draws.chain_count === entry.chains,
      "House current native result/reference identity changed");
    }
    return { chamber: "house", runId: entry.runId, resultSha256: entry.resultSha256,
      qualification: { configuration: entry.configuration, forecastIssuedAt: entry.forecastIssuedAt,
        forecastCutoffAt: entry.forecastCutoffAt, referenceIssuedAt: entry.referenceIssuedAt,
        rows: entry.rows, chains: entry.chains, winnerRaceCount: entry.winnerRaceCount,
        pairRaceCount: entry.pairRaceCount, pairInapplicableRaceCount: entry.pairInapplicableRaceCount,
        sourcePins: entry.sourcePins } };
  });
  requireValue(new Set(editions.map(e => `${e.runId}:${e.resultSha256}`)).size === editions.length &&
    sameObject(value.editions, editions),
  "House publication editions differ from the catalog");
  return { ...value, editions };
}
export function exportIndex(repo = REPO, output = join(repo, OUTPUT)) {
  const value = build(repo);
  mkdirSync(dirname(output), { recursive: true });
  writeFileSync(output, `${JSON.stringify(value, null, 2)}\n`, "utf8");
  return value;
}
if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) exportIndex();
