import { readFile, realpath } from "node:fs/promises";
import { dirname, join } from "node:path";
import { createHash } from "node:crypto";
import { HouseGroupSelectionError } from "./house-group-publication.ts";
import { parseHouseReviewedCatalog, parseHouseReviewedPublication, reviewedGroupRunId,
  type HouseReviewedPublication } from "./house-reviewed-catalog.ts";
import { parseHousePollingShares } from "./house-polling-shares.ts";
export { HouseGroupSelectionError };
const digest = (bytes:Buffer)=>createHash("sha256").update(bytes).digest("hex");
async function artifact(directory:string, reference:{file:string;sha256:string}) {
  if (!reference || !/^[a-zA-Z0-9][a-zA-Z0-9_-]*\.json$/.test(reference.file) || !/^[a-f0-9]{64}$/.test(reference.sha256)) throw Error("Invalid reviewed artifact reference.");
  const root = await realpath(directory), path = await realpath(join(root,reference.file));
  if (dirname(path)!==root) throw Error("Reviewed artifact leaves its collection.");
  const bytes = await readFile(path);
  if (digest(bytes)!==reference.sha256) throw Error("Reviewed artifact fingerprint mismatch.");
  return JSON.parse(bytes.toString("utf8"));
}
async function collection(directory:string) {
  const root = await realpath(directory), path = await realpath(join(root,"publication.json"));
  if (dirname(path)!==root) throw Error("Reviewed manifest leaves its collection.");
  const manifest = JSON.parse(await readFile(path,"utf8"));
  if (manifest.schemaVersion!==2 || manifest.status!=="available") throw Error("Reviewed collection unavailable.");
  const raw = await artifact(directory,manifest.catalog), catalog = parseHouseReviewedCatalog(raw.bindings);
  if (!catalog) throw Error("Invalid reviewed collection bindings.");
  return {raw,catalog};
}
/** Existing three-argument server call shape, with a separate sibling polling collection. */
export async function loadHouseReviewedPublication(groupDirectory:string, geometryDirectory:string, runId?:string|null,
  pollingDirectory=join(dirname(groupDirectory),"house-polling")):Promise<HouseReviewedPublication> {
  if (runId!=null && !reviewedGroupRunId(runId)) throw new HouseGroupSelectionError("Invalid reviewed group selection.");
  const {raw,catalog} = await collection(pollingDirectory);
  const entry = catalog.runs.find(r=>r.id===(runId??catalog.initialRunId));
  if (!entry) throw new HouseGroupSelectionError("This reviewed group result is not published.");
  const bound = raw.artifacts.find((r:{id:string})=>r.id===entry.id);
  if (bound?.view?.sha256!==entry.viewSha256 || bound?.geography?.sha256!==entry.geographySha256) throw Error("Reviewed artifact bindings differ.");
  const result = await artifact(entry.polling?pollingDirectory:groupDirectory,bound.view);
  const geography = {metadata:{...bound.geography.metadata,sourceSha256:bound.geography.sha256},
    geojson:await artifact(geometryDirectory,bound.geography)};
  const publication = await parseHouseReviewedPublication({schemaVersion:1,status:"available",family:"house_groups",catalog,entry,result,geography});
  if (!publication) throw Error("Reviewed result or geography failed verification.");
  return publication;
}
export async function loadHousePollingSharePublication(pollingDirectory:string,runId:string) {
  const {raw,catalog} = await collection(pollingDirectory);
  const entry = catalog.runs.find(r=>r.polling && r.conditionalShareRunId===runId);
  if (!entry) throw new HouseGroupSelectionError("This conditional polling-share result is not published.");
  const bound = raw.artifacts.find((r:{id:string})=>r.id===entry.id);
  if (bound?.shares?.sha256!==entry.conditionalShareSha256) throw Error("Conditional share binding differs.");
  const result = parseHousePollingShares(await artifact(pollingDirectory,bound.shares));
  if (!result || result.run.id!==runId) throw Error("Invalid conditional polling-share result.");
  return {schemaVersion:1,status:"available",family:"house_polling_shares",entry,result,resultSha256:entry.conditionalShareSha256};
}
