import { createHash } from "node:crypto";
import { makeHouseLiveCatalog, makeHouseLiveResult } from "./house-live-fixtures.mjs";
import { parseHouseLiveCatalog } from "../../lib/house-live-catalog.ts";
import { parseHouseLiveResult } from "../../lib/house-live-result.ts";
import { houseLiveCatalogView, houseLiveView } from "../../lib/house-live-view.ts";

/** Invented browser response only; never admitted by a server importer or published. */
export function makeHouseLivePublication(configuration) {
  const native = makeHouseLiveCatalog();
  for (const entry of native.entries)
    entry.result.sha256 = createHash("sha256").update(`invented revision ${entry.id}`).digest("hex");
  const catalog = parseHouseLiveCatalog(native);
  const raw = makeHouseLiveResult(configuration);
  raw.qualification.publication_status = "reviewed"; // Exercises projection, not scientific review.
  const result = parseHouseLiveResult(raw);
  if (!catalog || !result) throw Error("Invalid invented native fixture.");
  const entry = catalog.entries.find(entry => entry.id === result.id);
  return { schema: "HouseLivePublicationV1", status: "available", catalog: houseLiveCatalogView(catalog),
    view: houseLiveView(result, entry), geography: null };
}
