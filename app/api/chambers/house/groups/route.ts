import { join } from "node:path";
import { HouseGroupSelectionError, loadHouseGroupPublication } from "../../../../../lib/house-group-publication.ts";
export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export async function GET(request: Request): Promise<Response> {
  try {
    const params = new URL(request.url).searchParams;
    if ([...params.keys()].some(key => key !== "run") || params.getAll("run").length > 1) throw new HouseGroupSelectionError("Invalid result query.");
    return Response.json(await loadHouseGroupPublication(join(process.cwd(), "data/house-groups"), join(process.cwd(), "data/house"), params.get("run")),
      { headers: { "Cache-Control": "no-cache" } });
  } catch (error) {
    if (error instanceof HouseGroupSelectionError) return Response.json({ error: error.message }, { status: 400 });
    console.error("House group result unavailable:", error);
    return Response.json({ error: "The all-district House result is temporarily unavailable." }, { status: 503 });
  }
}
