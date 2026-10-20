import { join } from "node:path";
import { ComponentStudyInputError, loadComponentStudies, resolveComponentStudy } from "../../../lib/component-study-server.ts";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
const headers = { "Cache-Control": "no-cache" };
const failure = (error: unknown) => {
  if (error instanceof ComponentStudyInputError) return Response.json({ error: error.message }, { status: 400, headers });
  console.error("Component studies unavailable:", error);
  return Response.json({ error: "The component studies are temporarily unavailable. Please try again." }, { status: 503, headers });
};

export async function GET(request: Request): Promise<Response> {
  try {
    const query = new URL(request.url).searchParams;
    if ([...query.keys()].some(key => key !== "result") || query.getAll("result").length > 1) throw new ComponentStudyInputError("Invalid component-study query.");
    const { catalog, results, resultBytes } = await loadComponentStudies(join(process.cwd(), "data"));
    if (query.has("result")) {
      const result = results.find(result => result.id === query.get("result"));
      if (!result) throw new ComponentStudyInputError("Unknown component-study result.");
      return new Response(new Uint8Array(resultBytes[result.id]), { headers: { ...headers, "Content-Type": "application/json; charset=utf-8" } });
    }
    return Response.json(catalog, { headers });
  } catch (error) { return failure(error); }
}

export async function POST(request: Request): Promise<Response> {
  try {
    if (new URL(request.url).search) throw new ComponentStudyInputError("Component-study selection does not accept query parameters.");
    if (Number(request.headers.get("content-length")) > 32000) throw new ComponentStudyInputError("Component-study request is too large.");
    const body = await request.text();
    if (Buffer.byteLength(body, "utf8") > 32000) throw new ComponentStudyInputError("Component-study request is too large.");
    let value: unknown;
    try { value = JSON.parse(body); } catch { throw new ComponentStudyInputError("Supply a valid JSON component-study request."); }
    const { catalog } = await loadComponentStudies(join(process.cwd(), "data"));
    return Response.json(resolveComponentStudy(catalog, value), { headers });
  } catch (error) { return failure(error); }
}
