import { join } from "node:path";
import { HouseGroupSelectionError, loadHousePollingSharePublication } from "../../../../../../lib/house-reviewed-publication.ts";
export const runtime="nodejs";
export const dynamic="force-dynamic";
export async function GET(request:Request):Promise<Response> {
  try {
    const params=new URL(request.url).searchParams,run=params.get("run");
    if(!run||[...params.keys()].some(k=>k!=="run")||params.getAll("run").length!==1)
      throw new HouseGroupSelectionError("Select exactly one reviewed conditional polling-share run.");
    return Response.json(await loadHousePollingSharePublication(join(process.cwd(),"data/house-polling"),run),
      {headers:{"Cache-Control":"no-cache"}});
  } catch(error) {
    if(error instanceof HouseGroupSelectionError)return Response.json({error:error.message},{status:400});
    console.error("Reviewed conditional polling shares unavailable:",error);
    return Response.json({error:"The conditional polling-share result is temporarily unavailable."},{status:503});
  }
}
