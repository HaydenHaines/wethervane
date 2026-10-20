import { readFile, realpath } from "node:fs/promises";
import { dirname, join } from "node:path";
import { createHash } from "node:crypto";
import {
  parseHouseResult,
  houseReadinessInput,
  type HousePublication,
} from "./house-result.ts";
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

/** Only a committed manifest can admit reviewed, hash-pinned publication artifacts. */
export async function loadHousePublication(
  directory: string,
): Promise<HousePublication> {
  const manifest = JSON.parse(
    (await readLocalFile(directory, "publication.json")).toString("utf8"),
  );
  if (manifest?.schemaVersion !== 1)
    throw Error("Invalid House publication manifest.");
  if (manifest.status === "preparing")
    return { schemaVersion: 1, status: "preparing" };
  if (manifest.status !== "available")
    throw Error("Invalid House publication status.");
  const result = parseHouseResult(
    await readArtifact(directory, manifest.result),
  );
  if (!result) throw Error("House result has not passed publication checks.");
  let geography: DistrictGeography | null = null;
  if (manifest.geography) {
    // The digest identifies the raw FeatureCollection bytes, avoiding a self-referential hash.
    geography = {
      metadata: {
        ...manifest.geography.metadata,
        sourceSha256: manifest.geography.sha256,
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
    resultSha256: manifest.result.sha256,
    geography,
  };
}
