import { readFile } from "node:fs/promises";
import { join } from "node:path";
import type { CompositionCatalog } from "../../../../lib/composition-types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/** Serve public selection metadata through Node; SiteGround blocks the static filename. */
export async function GET(): Promise<Response> {
  try {
    const catalog = JSON.parse(await readFile(
      join(process.cwd(), "public/data/explorer/capabilities.json"), "utf8",
    )) as CompositionCatalog;
    if (catalog.schemaVersion !== 1 || !catalog.saved?.length || !catalog.slots?.length)
      throw Error("Invalid composition catalog.");
    return Response.json(catalog, {headers: {"Cache-Control": "no-cache"}});
  } catch (error) {
    console.error("Composition catalog is unavailable:", error);
    return Response.json({error: "The component catalog is temporarily unavailable."}, {status: 503});
  }
}
