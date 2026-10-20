import { readFile, realpath } from "node:fs/promises";
import { dirname, join } from "node:path";
import { createHash } from "node:crypto";
import {
  parseHouseResult,
  houseReadinessInput,
} from "./house-result.ts";
import { houseRunMatches, parseHouseCatalog, validHouseRunId, type HouseCatalog, type HousePublicationWithCatalog } from "./house-catalog.ts";
import { validateChamberReadiness } from "./chamber.ts";
import {
  joinDistrictGeography,
  type DistrictGeography,
} from "./district-geography.ts";

async function readLocalFile(directory: string, file: string): Promise<Buffer> {
  const root = await realpath(directory),
    path = await realpath(join(root, file));
  if (dirname(path) !== root)
    throw Error("House artifact must stay within the publication directory.");
  return readFile(path);
}

async function readArtifact(
  directory: string,
  entry: unknown,
): Promise<unknown> {
  const artifact = entry as { file?: string; sha256?: string };
  if (
    !artifact ||
    typeof artifact.file !== "string" ||
    !/^[a-zA-Z0-9][a-zA-Z0-9_-]*\.json$/.test(artifact.file) ||
    typeof artifact.sha256 !== "string" ||
    !/^[a-f0-9]{64}$/.test(artifact.sha256)
  )
    throw Error("Invalid House artifact reference.");
  const bytes = await readLocalFile(directory, artifact.file);
  if (createHash("sha256").update(bytes).digest("hex") !== artifact.sha256)
    throw Error("House artifact fingerprint mismatch.");
  return JSON.parse(bytes.toString("utf8"));
}

export class HouseRunSelectionError extends Error {}

async function loadResult(directory: string, manifest: Record<string, unknown>): Promise<Extract<HousePublicationWithCatalog, { status: "available" }>> {
  const result = parseHouseResult(
    await readArtifact(directory, manifest.result),
  );
  if (!result) throw Error("House result has not passed publication checks.");
  let geography: DistrictGeography | null = null;
  if (manifest.geography) {
    const geographyReference = manifest.geography as { metadata: DistrictGeography["metadata"]; sha256: string };
    // The digest identifies the raw FeatureCollection bytes, avoiding a self-referential hash.
    geography = {
      metadata: {
        ...geographyReference.metadata,
        sourceSha256: geographyReference.sha256,
      },
      geojson: (await readArtifact(
        directory,
        manifest.geography,
      )) as DistrictGeography["geojson"],
    };
    const joined = joinDistrictGeography(
      result.races.map((race) => ({
        race,
        districtKey: race.district.districtKey,
      })),
      { cycle: result.run.year, mapVintage: result.metadata.mapVintage },
      geography,
    );
    if (
      ["invalid", "unavailable"].includes(joined.status) ||
      joined.missingDistrictKeys.length
    )
      throw Error("House geography has not passed publication checks.");
  }
  const readiness = validateChamberReadiness(
    houseReadinessInput(result, geography),
  );
  if (
    !readiness.canShowRaces ||
    (result.metadata.full_chamber && !readiness.canShowControl)
  )
    throw Error("House forecast coverage has not passed publication checks.");
  return {
    schemaVersion: 1,
    status: "available",
    result,
    resultSha256: (manifest.result as { sha256: string }).sha256,
    geography,
  };
}

/** Only an explicit manifest admits reviewed, hash-pinned publication artifacts. */
export async function loadHousePublication(directory: string, runId?: string | null): Promise<HousePublicationWithCatalog> {
  if (runId !== undefined && runId !== null && !validHouseRunId(runId)) throw new HouseRunSelectionError("Invalid House result selection.");
  const manifest = JSON.parse((await readLocalFile(directory, "publication.json")).toString("utf8"));
  if (manifest?.schemaVersion !== 1) throw Error("Invalid House publication manifest.");
  if (manifest.status === "preparing") {
    if (runId != null) throw new HouseRunSelectionError("This House result is not available.");
    return { schemaVersion: 1, status: "preparing" };
  }
  if (manifest.status !== "available") throw Error("Invalid House publication status.");
  if (!manifest.catalog) {
    const publication = await loadResult(directory, manifest);
    if (runId != null && runId !== publication.result.run.id) throw new HouseRunSelectionError("Unknown House result selection.");
    return publication;
  }
  if (manifest.result || manifest.geography) throw Error("Ambiguous House publication manifest.");
  const rawCatalog = await readArtifact(directory, manifest.catalog) as { runs?: Record<string, unknown>[] };
  const catalog: HouseCatalog | null = parseHouseCatalog(rawCatalog);
  if (!catalog) throw Error("Invalid House publication catalog.");
  const selectedId = runId ?? catalog.defaultRunId;
  const selected = catalog.runs.find(run => run.id === selectedId);
  if (!selected?.available) throw new HouseRunSelectionError("This House result is unknown or unavailable.");
  let publication: Extract<HousePublicationWithCatalog, { status: "available" }> | undefined;
  for (const entry of catalog.runs) {
    const reference = rawCatalog.runs!.find(run => run.id === entry.id)!;
    if (!entry.available) {
      if (reference.result !== undefined || reference.geography !== undefined) throw Error("Unavailable House results must not have publication references.");
      continue;
    }
    const loaded = await loadResult(directory, reference);
    if (!houseRunMatches(entryRun(entry), loaded.result.run) || loaded.resultSha256 !== entry.resultSha256 || loaded.result.races.length !== entry.raceCount)
      throw Error("House catalog metadata does not match its result.");
    if (entry.id === selectedId) publication = loaded;
  }
  return { ...publication!, catalog };
}

function entryRun(entry: HouseCatalog["runs"][number]) {
  const { resultSha256: _digest, raceCount: _count, available: _available, unavailableReason: _reason, ...run } = entry;
  return run;
}
