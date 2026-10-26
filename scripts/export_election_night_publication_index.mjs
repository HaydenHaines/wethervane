#!/usr/bin/env node
/** Build the runtime scorecard reference index from published, byte-pinned catalogs. */
import { createHash } from "node:crypto";
import { readFileSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

export const REPO = fileURLToPath(new URL("../", import.meta.url));
export const FULL = "data/election-night-reference-publication.json";
export const PRIMARY = "data/election-night-core-reference-publication.json";
export const OUTPUT = "public/data/election-night/publication-index.json";
export const HREFS = { full: "/data/election-night/references.json", primary: "/data/election-night/core-references.json" };

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
  return { schemaVersion: 1, full: publication(repo, "full", FULL),
    primary: publication(repo, "primary", PRIMARY) };
}
export function exportIndex(repo = REPO, output = join(repo, OUTPUT)) {
  const value = build(repo);
  mkdirSync(dirname(output), { recursive: true });
  writeFileSync(output, `${JSON.stringify(value, null, 2)}\n`, "utf8");
  return value;
}
if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) exportIndex();
