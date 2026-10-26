import { loadConfiguredElectionNightEvaluation, type EvaluationChamber } from "../../../../lib/election-night-evaluation-server.ts";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
const headers = { "Cache-Control": "no-store, max-age=0" };

export async function GET(request: Request): Promise<Response> {
  const query = new URL(request.url).searchParams;
  if ([...query.keys()].some(key => key !== "chamber") || query.getAll("chamber").length !== 1 ||
    !["senate", "house"].includes(query.get("chamber") ?? "")) {
    return Response.json({ error: "Choose Senate or House." }, { status: 400, headers });
  }
  const view = await loadConfiguredElectionNightEvaluation(query.get("chamber") as EvaluationChamber);
  return Response.json(view, { status: view.status === "unavailable" ? 503 : 200, headers });
}
