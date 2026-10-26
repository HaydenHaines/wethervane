import { join } from "node:path";
import { HouseLiveSelectionError, loadHouseLiveSitePublication } from "../../../../../lib/house-live-server.ts";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(request: Request): Promise<Response> {
  try {
    const params = new URL(request.url).searchParams;
    if ([...params.keys()].some(key => key !== "run") || params.getAll("run").length > 1)
      throw new HouseLiveSelectionError("Invalid current House model query.");
    return Response.json(await loadHouseLiveSitePublication(join(process.cwd(), "data/house-live"), params.get("run")),
      { headers: { "Cache-Control": "no-store" } });
  } catch (error) {
    if (error instanceof HouseLiveSelectionError)
      return Response.json({ error: error.message }, { status: 400, headers: { "Cache-Control": "no-store" } });
    console.error("Current House publication unavailable:", error);
    return Response.json({ error: "The current House forecast is temporarily unavailable. Please try again." },
      { status: 503, headers: { "Cache-Control": "no-store" } });
  }
}
