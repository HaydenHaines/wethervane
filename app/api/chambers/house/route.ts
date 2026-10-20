import { join } from "node:path";
import { HouseRunSelectionError, loadHousePublication } from "../../../../lib/house-publication.ts";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(request: Request): Promise<Response> {
  try {
    const params = new URL(request.url).searchParams;
    if ([...params.keys()].some(key => key !== "run") || params.getAll("run").length > 1)
      throw new HouseRunSelectionError("Invalid House result query.");
    return Response.json(
      await loadHousePublication(join(process.cwd(), "data/house"), params.get("run")),
      {
        headers: { "Cache-Control": "no-cache" },
      },
    );
  } catch (error) {
    if (error instanceof HouseRunSelectionError) return Response.json({ error: error.message }, { status: 400, headers: { "Cache-Control": "no-cache" } });
    console.error("House publication unavailable:", error);
    return Response.json(
      { error: "The House view is temporarily unavailable. Please try again." },
      { status: 503 },
    );
  }
}
