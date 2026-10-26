import { join } from "node:path";
import { readHouseLiveAvailability } from "../../../../../../lib/house-live-server.ts";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(request: Request): Promise<Response> {
  if (new URL(request.url).searchParams.size)
    return Response.json({ error: "This availability request does not accept model selections." },
      { status: 400, headers: { "Cache-Control": "no-store" } });
  try {
    const { available } = await readHouseLiveAvailability(join(process.cwd(), "data/house-live"));
    return Response.json({ schema: "HouseLiveAvailabilityV1", available }, { headers: { "Cache-Control": "no-store" } });
  } catch (error) {
    console.error("Current House availability could not be verified:", error);
    // A broken existing release is not the same thing as an unpublished current edition.
    return Response.json({ error: "Current House publication status could not be verified. Please try again." },
      { status: 503, headers: { "Cache-Control": "no-store" } });
  }
}
