import { readFile, realpath } from "node:fs/promises";
import { dirname, join } from "node:path";
import { createHash } from "node:crypto";
import { parseHouseGroupCatalog, parseHouseGroupPublication, groupRunId, type HouseGroupPublication } from "./house-group-catalog.ts";

export class HouseGroupSelectionError extends Error {}
const digest = (bytes: Buffer) => createHash("sha256").update(bytes).digest("hex");
async function local(directory: string, name: string): Promise<Buffer> {
  if (!/^[a-zA-Z0-9][a-zA-Z0-9_-]*\.json$/.test(name)) throw Error("Invalid group artifact name.");
  const root = await realpath(directory), file = await realpath(join(root, name));
  if (dirname(file) !== root) throw Error("Group artifact leaves its publication directory.");
  return readFile(file);
}
async function artifact(directory: string, reference: { file: string; sha256: string }) {
  if (!reference || typeof reference.file !== "string" || !/^[a-f0-9]{64}$/.test(reference.sha256)) throw Error("Invalid group artifact reference.");
  const bytes = await local(directory, reference.file);
  if (digest(bytes) !== reference.sha256) throw Error("Group artifact fingerprint mismatch.");
  return JSON.parse(bytes.toString("utf8"));
}
export async function loadHouseGroupPublication(directory: string, geometryDirectory: string, runId?: string | null): Promise<HouseGroupPublication> {
  if (runId != null && !groupRunId(runId)) throw new HouseGroupSelectionError("Invalid group result selection.");
  const manifest = JSON.parse((await local(directory, "publication.json")).toString("utf8"));
  if (manifest.schemaVersion !== 1 || manifest.status !== "available") throw Error("Group publication is not available.");
  const raw = await artifact(directory, manifest.catalog), catalog = parseHouseGroupCatalog(raw);
  if (!catalog) throw Error("Invalid group catalog.");
  const entry = catalog.runs.find(run => run.id === (runId ?? catalog.initialRunId));
  if (!entry) throw new HouseGroupSelectionError("This group result is not published.");
  const bound = raw.runs.find((run: { id: string }) => run.id === entry.id);
  if (bound.view?.sha256 !== entry.viewSha256 || bound.geography?.sha256 !== entry.geographySha256) throw Error("Group publication bindings differ.");
  const result = await artifact(directory, bound.view);
  const geography = { metadata: { ...bound.geography.metadata, sourceSha256: bound.geography.sha256 }, geojson: await artifact(geometryDirectory, bound.geography) };
  const parsed = await parseHouseGroupPublication({ schemaVersion: 1, status: "available", family: "house_groups", catalog, entry, result, geography });
  if (!parsed) throw Error("Group result or geography failed verification.");
  return parsed;
}
