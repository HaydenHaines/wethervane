import { join } from "node:path";
import { loadHousePublication } from "../../../../lib/house-publication";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(): Promise<Response> {
  try {
    return Response.json(
      await loadHousePublication(join(process.cwd(), "data/house")),
      {
        headers: { "Cache-Control": "no-cache" },
      },
    );
  } catch (error) {
    console.error("House publication unavailable:", error);
    return Response.json(
      { error: "The House view is temporarily unavailable. Please try again." },
      { status: 503 },
    );
  }
}
